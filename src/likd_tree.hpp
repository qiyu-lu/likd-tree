/*
Copyright 2026 Liu Yang
Copyright 2026 qiyu-lu
Distributed under MIT license. See LICENSE for more information.
*/

// likd-tree: A Lightweight Incremental KD-Tree for dynamic point insertion
// with automatic background rebalancing. Header-only C++17 library.
//
// Points live in leaf buckets of up to Options::LEAF_SIZE points; inner nodes
// only split space. A full leaf splits in two when a point is added to it.
//
// Thread safety: queries may run concurrently with each other and with
// writers. Writers (build / addPoints / deletePoints / deleteBoxes) are
// serialized internally, so calling them from several threads is safe but
// gains no parallelism.
//
// While a background rebuild runs, writes are queued and applied in order once
// it finishes, so queries may briefly not see the latest writes. A write call
// is applied in chunks of WRITE_CHUNK_SIZE operations: once a chunk unbalances
// a subtree, the rebuild starts and the rest of the call is queued behind it.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <execution>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <vector>

constexpr double KDTREE_ALPHA = 0.75;
// Rebuild a subtree once more than this fraction of its points is deleted
constexpr double KDTREE_DELETE_ALPHA = 0.5;
// Queued writes are replayed in batches of this many, each under one lock
constexpr int INSERTION_BATCH_SIZE = 100;
// A write call checks for unbalanced subtrees after every this many operations
constexpr size_t WRITE_CHUNK_SIZE = 2000;
// Subtrees larger than this are built with parallel tasks (TBB only)
constexpr size_t MIN_PARALLEL_BUILD_SIZE = 20000;
#ifdef LIKD_TREE_USE_TBB
#include <tbb/parallel_invoke.h>
#define TREE_PAR std::execution::par
#else
#define TREE_PAR std::execution::seq
#endif

// Default PointTraits for point types with x, y, z members
template <typename PointType>
struct PointTraits {
  static constexpr int DIM = 3;
  static inline float coord(const PointType& pt, int axis) {
    return axis == 0 ? pt.x : (axis == 1 ? pt.y : pt.z);
  }
  static inline float sqrDist(const PointType& a, const PointType& b) {
    float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
  }
};

template <typename PointType>
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

// Compile-time options. To change one, derive from this struct and override
// it, e.g. struct MyOptions : DefaultOptions { static constexpr int LEAF_SIZE
// = 64; }, then use KDTree<PointType, PointTraits<PointType>, MyOptions>.
struct DefaultOptions {
  // Points stored in each leaf, 2 to 64
  static constexpr int LEAF_SIZE = 32;
};

// PointType must be default-constructible and copy-assignable.
template <typename PointType, typename Traits = PointTraits<PointType>,
          typename Options = DefaultOptions>
class KDTree {
  static constexpr int LeafSize = Options::LEAF_SIZE;
  static_assert(LeafSize >= 2 && LeafSize <= 64, "LEAF_SIZE must be 2 to 64");
  // Smaller subtrees are never rebuilt: a few leaves are not worth it
  static constexpr int MIN_REBUILD_SIZE = 4 * LeafSize;
  // Points per leaf when building: ceil(0.8 * LeafSize), leaving room to
  // insert before a leaf splits
  static constexpr int BUILD_LEAF_POINTS = (4 * LeafSize + 4) / 5;

 public:
  struct AABB {
    std::array<float, Traits::DIM> min, max;
    AABB();
    AABB(const std::array<float, Traits::DIM>& min_corner,
         const std::array<float, Traits::DIM>& max_corner);
    void expand(const PointType& pt);
    void expand(const AABB& box);
    float sqrDist(const PointType& pt) const;
    // Closed intervals: points on the boundary are inside.
    bool contains(const PointType& pt) const;
    bool contains(const AABB& box) const;
    bool intersects(const AABB& box) const;
  };

  KDTree();
  ~KDTree();

  void build(const PointVector<PointType>& pts);
  void addPoints(const PointVector<PointType>& pts, bool wait_for_rebuild = false);
  // Deletes one stored copy of each point whose coordinates match exactly.
  void deletePoints(const PointVector<PointType>& pts,
                    bool wait_for_rebuild = false);
  // Deletes every point inside the boxes (boundary included).
  void deleteBoxes(const std::vector<AABB>& boxes, bool wait_for_rebuild = false);
  void deleteBox(const AABB& box, bool wait_for_rebuild = false);
  // Returns a copy of the nearest point (empty if the tree is empty): a
  // background rebuild may free the node right after the lock is released.
  std::pair<std::optional<PointType>, float> nearestNeighbors(
      const PointType& query) const;
  void nearestNeighbors(const PointVector<PointType>& queries,
                        PointVector<PointType>& results,
                        std::vector<float>& distances) const;
  void radiusSearch(const PointType& query, float radius,
                    PointVector<PointType>& results,
                    std::vector<float>& distances) const;
  // k nearest neighbors sorted by distance. Fewer than k are returned if the
  // tree is smaller or max_dist excludes the rest.
  void knnSearch(const PointType& query, int k, PointVector<PointType>& results,
                 std::vector<float>& distances,
                 float max_dist = INFINITY) const;
  // Batch version; queries run in parallel with LIKD_TREE_USE_TBB.
  void knnSearch(const PointVector<PointType>& queries, int k,
                 std::vector<PointVector<PointType>>& results,
                 std::vector<std::vector<float>>& distances,
                 float max_dist = INFINITY) const;
  // All points inside the box (boundary included), in no particular order.
  void boxSearch(const AABB& box, PointVector<PointType>& results) const;
  // Number of points that are not deleted. Writes still queued by a running
  // rebuild are not reflected yet.
  int size() const;
  // Points held in memory, including deleted points not yet reclaimed by a
  // rebuild.
  int nodeCount() const;
  // Bytes held by the tree's nodes, deleted points included, excluding
  // allocator overhead.
  size_t memoryUsage() const;
  // Blocks until the background rebuild, including the writes queued while
  // it ran, has finished.
  void waitForRebuild() const;

#ifdef LIKD_TREE_STATS
  // Background rebuild statistics, collected only when LIKD_TREE_STATS is
  // defined before including this header.
  struct RebuildStats {
    size_t rounds = 0;          // rebuild rounds run by the worker
    double max_round_ms = 0;    // longest round: collect, build, swap, free
    // Most writes waiting at once, sampled whenever writes are queued. Writes
    // the worker set aside to rebuild before replaying them count as waiting.
    size_t max_queued_ops = 0;
    // Longest time from queueing a write until the worker has replayed it and
    // every write it took from the queue together with it, rebuilds in
    // between included: how long a write may stay invisible to queries.
    double max_queued_ms = 0;
  };
  RebuildStats rebuildStats() const;
#endif

#ifdef LIKD_TREE_TESTING
  // Checks the links, counts, bounds and flags of every node against values
  // recomputed from scratch. Call it while no rebuild runs, e.g. after
  // waitForRebuild().
  bool validate() const;
#endif

 private:
  using Mask = std::conditional_t<(LeafSize <= 32), uint32_t, uint64_t>;

  struct Node {
    Node* parent = nullptr;
    AABB aabb;  // bounds of the non-deleted points in this subtree
    int size = 0;   // points stored in this subtree, deleted ones included
    int valid = 0;  // non-deleted points in this subtree
    bool is_leaf;
    bool is_left_child = false;  // true if this is parent's left child
    bool need_rebuild = false;   // inner nodes only
    // Lazy tag, inner nodes only: the whole subtree is deleted. Descendants
    // are only updated when a writer descends here (pushDown); queries never
    // get past a node with valid == 0, so they never see stale descendants.
    bool tree_deleted = false;
    explicit Node(bool leaf) : is_leaf(leaf) {}
  };

  struct Inner : Node {
    Node* left = nullptr;  // either child may be empty
    Node* right = nullptr;
    float split;  // points with coord < split are inserted on the left
    int axis;
    Inner(float s, int ax) : Node(false), split(s), axis(ax) {}
  };

  struct Leaf : Node {
    Mask deleted = 0;  // bit i: pts[i] is deleted
    PointType pts[LeafSize];  // the first `size` slots are in use
    Leaf() : Node(true) {}
  };

  // std::shared_mutex on glibc prefers readers: back-to-back queries from
  // other threads could starve the writers and the rebuild swap forever.
  // New readers therefore wait while a writer is queued.
  class SharedMutex {
   public:
    void lock() {
      waiting_writers_.fetch_add(1);
      mutex_.lock();
      waiting_writers_.fetch_sub(1);
    }
    void unlock() { mutex_.unlock(); }
    void lock_shared() {
      while (waiting_writers_.load() > 0)
        std::this_thread::yield();
      mutex_.lock_shared();
    }
    void unlock_shared() { mutex_.unlock_shared(); }

   private:
    std::shared_mutex mutex_;
    std::atomic<int> waiting_writers_{0};
  };

  // A write queued while a rebuild runs
  enum class OpType { kInsert, kDeletePoint, kDeleteBox };
  struct Op {
    OpType type;
    PointType point;
    AABB box;
  };
  using OpLog = std::vector<Op, Eigen::aligned_allocator<Op>>;

  static Leaf* asLeaf(Node* n) { return static_cast<Leaf*>(n); }
  static const Leaf* asLeaf(const Node* n) { return static_cast<const Leaf*>(n); }
  static Inner* asInner(Node* n) { return static_cast<Inner*>(n); }
  static const Inner* asInner(const Node* n) {
    return static_cast<const Inner*>(n);
  }
  static Mask bit(int i) { return Mask(1) << i; }
  // The lowest n bits; n may equal the width of Mask
  static Mask lowBits(int n) {
    return n >= static_cast<int>(sizeof(Mask) * 8) ? ~Mask(0) : bit(n) - 1;
  }
  // Calls fn(i) for each slot in use whose point is not deleted
  template <typename Fn>
  static void forEachValid(const Leaf* leaf, Fn&& fn);
  static int longestAxis(const AABB& box);

  template <typename ApplyFn, typename OpFn>
  void write(size_t count, ApplyFn&& apply, OpFn&& to_op, bool wait_for_rebuild);
  void applyOp(const Op& op, std::vector<Node*>* candidates);
  Leaf* newLeaf();
  Inner* newInner(float split, int axis);
  Node* insertInternal(Node* node, const PointType& pt,
                       std::vector<Node*>* candidates);
  static void appendPoint(Leaf* leaf, const PointType& pt);
  Node* makeRoom(Leaf* leaf);
  bool deletePointInternal(Node* node, const PointType& pt,
                           std::vector<Node*>* candidates);
  int deleteBoxInternal(Node* node, const AABB& box,
                        std::vector<Node*>* candidates);
  static void killSubtree(Node* node);
  static void pushDown(Inner* node);
  static bool samePoint(const PointType& a, const PointType& b);
  static void updateLeaf(Leaf* leaf);
  static void updateInner(Inner* node);
  bool needRebuild(const Inner* node) const;
  void markIfUnbalanced(Inner* node, std::vector<Node*>* candidates);
  void collect(const Node* node, PointVector<PointType>& pts) const;
  void destroy(Node* node);
  Node* buildRecursive(PointVector<PointType>& pts, size_t l, size_t r);
  void nearestNeighborInternal(const Node* node, const PointType& query,
                               const PointType*& best_pt,
                               float& best_dist2) const;
  void radiusSearchInternal(const Node* node, const PointType& query,
                            float radius2, PointVector<PointType>& results,
                            std::vector<float>& distances2) const;
  // best: (squared distance, point), ascending, at most k entries
  void knnSearchInternal(const Node* node, const PointType& query, size_t k,
                         float max_dist2,
                         std::vector<std::pair<float, const PointType*>>& best) const;
  void knnSearchLocked(const PointType& query, int k, float max_dist,
                       PointVector<PointType>& results,
                       std::vector<float>& distances) const;
  void boxSearchInternal(const Node* node, const AABB& box,
                         PointVector<PointType>& results) const;
  bool checkAncestorNeedsRebuild(Node* node) const;
  std::vector<Node*> topmostCandidates(const std::vector<Node*>& candidates) const;
  void rebuildSubtrees(const std::vector<Node*>& nodes_to_rebuild);
  void workerLoop();

  Node* root_ = nullptr;
  // Nodes allocated, for memoryUsage(). Parallel builds allocate concurrently.
  std::atomic<size_t> leaf_count_{0};
  std::atomic<size_t> inner_count_{0};
  mutable SharedMutex tree_mutex_;
  // Serializes writers.
  std::mutex write_mutex_;
  // Guards rebuilding_ transitions, pending_ops_, rebuild_job_ and stop_.
  mutable std::mutex pending_mutex_;
  mutable std::condition_variable idle_cv_;
  std::condition_variable job_cv_;
  std::atomic<bool> rebuilding_{false};
  OpLog pending_ops_;
  std::vector<Node*> rebuild_job_;
  bool stop_ = false;
  std::thread worker_;
#ifdef LIKD_TREE_STATS
  // Guarded by pending_mutex_
  RebuildStats stats_;
  // When the oldest write in pending_ops_ was queued
  std::chrono::steady_clock::time_point queued_since_;
  // Writes the worker took from pending_ops_ and set aside while it rebuilds
  // a subtree, to replay right after
  size_t held_ops_ = 0;
#endif
};

// AABB: axis-aligned bounding boxes
template <typename PointType, typename Traits, typename Options>
KDTree<PointType, Traits, Options>::AABB::AABB() {
  for (int i = 0; i < Traits::DIM; ++i) {
    min[i] = std::numeric_limits<float>::max();
    max[i] = std::numeric_limits<float>::lowest();
  }
}

template <typename PointType, typename Traits, typename Options>
KDTree<PointType, Traits, Options>::AABB::AABB(
    const std::array<float, Traits::DIM>& min_corner,
    const std::array<float, Traits::DIM>& max_corner)
    : min(min_corner), max(max_corner) {}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::AABB::expand(const PointType& pt) {
  for (int i = 0; i < Traits::DIM; ++i) {
    min[i] = std::min(min[i], Traits::coord(pt, i));
    max[i] = std::max(max[i], Traits::coord(pt, i));
  }
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::AABB::expand(const AABB& box) {
  for (int i = 0; i < Traits::DIM; ++i) {
    min[i] = std::min(min[i], box.min[i]);
    max[i] = std::max(max[i], box.max[i]);
  }
}

template <typename PointType, typename Traits, typename Options>
float KDTree<PointType, Traits, Options>::AABB::sqrDist(const PointType& pt) const {
  float d2 = 0;
  for (int i = 0; i < Traits::DIM; ++i) {
    float v = Traits::coord(pt, i);
    if (v < min[i])
      d2 += (min[i] - v) * (min[i] - v);
    else if (v > max[i])
      d2 += (v - max[i]) * (v - max[i]);
  }
  return d2;
}

template <typename PointType, typename Traits, typename Options>
bool KDTree<PointType, Traits, Options>::AABB::contains(const PointType& pt) const {
  for (int i = 0; i < Traits::DIM; ++i) {
    float v = Traits::coord(pt, i);
    if (v < min[i] || v > max[i])
      return false;
  }
  return true;
}

template <typename PointType, typename Traits, typename Options>
bool KDTree<PointType, Traits, Options>::AABB::contains(const AABB& box) const {
  for (int i = 0; i < Traits::DIM; ++i) {
    if (box.min[i] < min[i] || box.max[i] > max[i])
      return false;
  }
  return true;
}

template <typename PointType, typename Traits, typename Options>
bool KDTree<PointType, Traits, Options>::AABB::intersects(const AABB& box) const {
  for (int i = 0; i < Traits::DIM; ++i) {
    if (box.max[i] < min[i] || box.min[i] > max[i])
      return false;
  }
  return true;
}

template <typename PointType, typename Traits, typename Options>
KDTree<PointType, Traits, Options>::KDTree() : root_(nullptr) {}

template <typename PointType, typename Traits, typename Options>
KDTree<PointType, Traits, Options>::~KDTree() {
  {
    std::unique_lock<std::mutex> lock(pending_mutex_);
    idle_cv_.wait(lock, [this] { return !rebuilding_.load(); });
    stop_ = true;
  }
  job_cv_.notify_all();
  if (worker_.joinable())
    worker_.join();
  destroy(root_);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::build(const PointVector<PointType>& pts) {
  std::lock_guard<std::mutex> write_lock(write_mutex_);
  // The worker may still hold pointers into the current tree
  waitForRebuild();
  PointVector<PointType> tmp = pts;
  Node* new_root = buildRecursive(tmp, 0, tmp.size());
  Node* old_root;
  {
    std::unique_lock<SharedMutex> lock(tree_mutex_);
    old_root = root_;
    root_ = new_root;
  }
  destroy(old_root);
}

// Performs the count operations of one write call in order: apply(i,
// candidates) applies operation i to the tree, to_op(i) returns it for the
// queue. While a rebuild runs, all of them are queued. Otherwise they are
// applied in chunks of WRITE_CHUNK_SIZE, and once a chunk unbalances a
// subtree, the subtree goes to the worker and the rest of the call is queued
// behind it. A long run of writes into one place, such as points sorted along
// an axis, is thus rebalanced as it goes instead of growing into a chain that
// every later write has to walk.
template <typename PointType, typename Traits, typename Options>
template <typename ApplyFn, typename OpFn>
void KDTree<PointType, Traits, Options>::write(size_t count, ApplyFn&& apply,
                                      OpFn&& to_op, bool wait_for_rebuild) {
  std::lock_guard<std::mutex> write_lock(write_mutex_);
  size_t done = 0;
  // Queues the operations not applied yet. Caller holds pending_mutex_.
  auto queue_rest = [&] {
#ifdef LIKD_TREE_STATS
    if (pending_ops_.empty())
      queued_since_ = std::chrono::steady_clock::now();
#endif
    for (; done < count; ++done)
      pending_ops_.push_back(to_op(done));
#ifdef LIKD_TREE_STATS
    stats_.max_queued_ops =
        std::max(stats_.max_queued_ops, pending_ops_.size() + held_ops_);
#endif
  };
  {
    // rebuilding_ is checked under pending_mutex_, the same lock the worker
    // holds when it leaves the rebuilding state, so queued writes can never
    // be left behind in pending_ops_.
    std::lock_guard<std::mutex> lock(pending_mutex_);
    if (rebuilding_.load())
      queue_rest();
  }

  while (done < count) {
    std::vector<Node*> nodes_to_rebuild;
    // Write + filtering phase - protected by exclusive lock
    {
      std::unique_lock<SharedMutex> lock(tree_mutex_);
      std::vector<Node*> candidates;
      size_t end = std::min(count, done + WRITE_CHUNK_SIZE);
      for (; done < end; ++done)
        apply(done, &candidates);
      nodes_to_rebuild = topmostCandidates(candidates);
    }
    if (nodes_to_rebuild.empty())
      continue;

    // Hand the unbalanced subtrees to the worker and queue the rest of the
    // call behind them. Writers are serialized and only writers set
    // rebuilding_, so it is still false here and pending_ops_ is empty.
    {
      std::lock_guard<std::mutex> lock(pending_mutex_);
      rebuilding_ = true;
      rebuild_job_ = std::move(nodes_to_rebuild);
      if (done < count)
        queue_rest();
      if (!worker_.joinable())
        worker_ = std::thread(&KDTree::workerLoop, this);
    }
    job_cv_.notify_one();
  }
  // Optionally wait for rebuild to complete before returning
  if (wait_for_rebuild) {
    waitForRebuild();
  }
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::addPoints(const PointVector<PointType>& pts,
                                          bool wait_for_rebuild) {
  write(
      pts.size(),
      [&](size_t i, std::vector<Node*>* candidates) {
        root_ = insertInternal(root_, pts[i], candidates);
      },
      [&](size_t i) { return Op{OpType::kInsert, pts[i], AABB()}; },
      wait_for_rebuild);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::deletePoints(const PointVector<PointType>& pts,
                                             bool wait_for_rebuild) {
  write(
      pts.size(),
      [&](size_t i, std::vector<Node*>* candidates) {
        deletePointInternal(root_, pts[i], candidates);
      },
      [&](size_t i) { return Op{OpType::kDeletePoint, pts[i], AABB()}; },
      wait_for_rebuild);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::deleteBoxes(const std::vector<AABB>& boxes,
                                            bool wait_for_rebuild) {
  write(
      boxes.size(),
      [&](size_t i, std::vector<Node*>* candidates) {
        deleteBoxInternal(root_, boxes[i], candidates);
      },
      [&](size_t i) { return Op{OpType::kDeleteBox, PointType(), boxes[i]}; },
      wait_for_rebuild);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::deleteBox(const AABB& box,
                                          bool wait_for_rebuild) {
  deleteBoxes(std::vector<AABB>{box}, wait_for_rebuild);
}

template <typename PointType, typename Traits, typename Options>
std::pair<std::optional<PointType>, float>
KDTree<PointType, Traits, Options>::nearestNeighbors(const PointType& query) const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);

  if (root_ == nullptr || root_->valid == 0) {
    return {std::nullopt, INFINITY};
  }

  const PointType* best_pt = nullptr;
  float best_dist2 = INFINITY;
  nearestNeighborInternal(root_, query, best_pt, best_dist2);
  return {*best_pt, std::sqrt(best_dist2)};
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::nearestNeighbors(
    const PointVector<PointType>& queries, PointVector<PointType>& results,
    std::vector<float>& distances) const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);

  results.resize(queries.size());
  distances.assign(queries.size(), INFINITY);

  // If tree is empty, all queries return INFINITY distance
  if (root_ == nullptr || root_->valid == 0) {
    return;
  }

  std::vector<size_t> indices(queries.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::for_each(TREE_PAR, indices.begin(), indices.end(), [&](size_t i) {
    const PointType* best_pt = nullptr;
    float best_dist2 = INFINITY;
    nearestNeighborInternal(root_, queries[i], best_pt, best_dist2);
    distances[i] = std::sqrt(best_dist2);
    results[i] = *best_pt;
  });
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::radiusSearch(
    const PointType& query, float radius, PointVector<PointType>& results,
    std::vector<float>& distances) const {
  results.clear();
  distances.clear();

  if (radius < 0.0f) {
    return;
  }

  std::shared_lock<SharedMutex> lock(tree_mutex_);
  if (root_ == nullptr) {
    return;
  }

  const float radius2 = radius * radius;
  radiusSearchInternal(root_, query, radius2, results, distances);

  std::vector<size_t> indices(results.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
    if (distances[lhs] != distances[rhs]) {
      return distances[lhs] < distances[rhs];
    }
    for (int axis = 0; axis < Traits::DIM; ++axis) {
      float lhs_coord = Traits::coord(results[lhs], axis);
      float rhs_coord = Traits::coord(results[rhs], axis);
      if (lhs_coord != rhs_coord) {
        return lhs_coord < rhs_coord;
      }
    }
    return false;
  });

  PointVector<PointType> sorted_results;
  sorted_results.reserve(results.size());
  std::vector<float> sorted_distances;
  sorted_distances.reserve(distances.size());

  for (size_t index : indices) {
    sorted_results.push_back(results[index]);
    sorted_distances.push_back(std::sqrt(distances[index]));
  }

  results.swap(sorted_results);
  distances.swap(sorted_distances);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::knnSearch(const PointType& query, int k,
                                          PointVector<PointType>& results,
                                          std::vector<float>& distances,
                                          float max_dist) const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  knnSearchLocked(query, k, max_dist, results, distances);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::knnSearch(
    const PointVector<PointType>& queries, int k,
    std::vector<PointVector<PointType>>& results,
    std::vector<std::vector<float>>& distances, float max_dist) const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  results.resize(queries.size());
  distances.resize(queries.size());

  std::vector<size_t> indices(queries.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(TREE_PAR, indices.begin(), indices.end(), [&](size_t i) {
    knnSearchLocked(queries[i], k, max_dist, results[i], distances[i]);
  });
}

// Caller holds tree_mutex_ (shared)
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::knnSearchLocked(
    const PointType& query, int k, float max_dist,
    PointVector<PointType>& results, std::vector<float>& distances) const {
  results.clear();
  distances.clear();
  if (k <= 0 || root_ == nullptr || root_->valid == 0 || max_dist < 0.0f) {
    return;
  }
  const float max_dist2 = std::isinf(max_dist) ? INFINITY : max_dist * max_dist;
  std::vector<std::pair<float, const PointType*>> best;
  best.reserve(k + 1);
  if (root_->aabb.sqrDist(query) <= max_dist2) {
    knnSearchInternal(root_, query, k, max_dist2, best);
  }
  // Copy while the lock is held: a background rebuild may free the nodes
  results.reserve(best.size());
  distances.reserve(best.size());
  for (const auto& [dist2, pt] : best) {
    results.push_back(*pt);
    distances.push_back(std::sqrt(dist2));
  }
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::boxSearch(const AABB& box,
                                          PointVector<PointType>& results) const {
  results.clear();
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  boxSearchInternal(root_, box, results);
}

template <typename PointType, typename Traits, typename Options>
int KDTree<PointType, Traits, Options>::size() const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  return root_ ? root_->valid : 0;
}

template <typename PointType, typename Traits, typename Options>
int KDTree<PointType, Traits, Options>::nodeCount() const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  return root_ ? root_->size : 0;
}

template <typename PointType, typename Traits, typename Options>
size_t KDTree<PointType, Traits, Options>::memoryUsage() const {
  return leaf_count_.load(std::memory_order_relaxed) * sizeof(Leaf) +
         inner_count_.load(std::memory_order_relaxed) * sizeof(Inner);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::waitForRebuild() const {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  idle_cv_.wait(lock, [this] { return !rebuilding_.load(); });
}

#ifdef LIKD_TREE_STATS
template <typename PointType, typename Traits, typename Options>
typename KDTree<PointType, Traits, Options>::RebuildStats
KDTree<PointType, Traits, Options>::rebuildStats() const {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  return stats_;
}
#endif

#ifdef LIKD_TREE_TESTING
template <typename PointType, typename Traits, typename Options>
bool KDTree<PointType, Traits, Options>::validate() const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  auto same_box = [](const AABB& a, const AABB& b) {
    return a.min == b.min && a.max == b.max;
  };
  bool ok = !root_ || (root_->parent == nullptr && !root_->is_left_child);
  size_t leaves = 0, inners = 0;
  // (node, below a tree_deleted tag: its counts and bounds are stale)
  std::vector<std::pair<const Node*, bool>> stack;
  if (root_)
    stack.push_back({root_, false});
  while (!stack.empty()) {
    auto [n, stale] = stack.back();
    stack.pop_back();
    // Every candidate is rebuilt, or freed with an ancestor, before the
    // worker goes idle
    ok = ok && !n->need_rebuild;
    if (n->is_leaf) {
      ++leaves;
      const Leaf* leaf = asLeaf(n);
      ok = ok && !leaf->tree_deleted && leaf->size >= 0 &&
           leaf->size <= LeafSize && (leaf->deleted & ~lowBits(leaf->size)) == 0;
      if (!stale) {
        int valid = 0;
        AABB box;
        forEachValid(leaf, [&](int i) {
          ++valid;
          box.expand(leaf->pts[i]);
        });
        ok = ok && leaf->valid == valid && same_box(leaf->aabb, box);
      }
      continue;
    }
    ++inners;
    const Inner* inner = asInner(n);
    int size = 0, valid = 0;
    AABB box;
    for (const Node* child : {inner->left, inner->right}) {
      if (!child)
        continue;
      ok = ok && child->parent == inner &&
           child->is_left_child == (child == inner->left);
      size += child->size;
      if (child->valid > 0) {
        valid += child->valid;
        box.expand(child->aabb);
      }
      stack.push_back({child, stale || inner->tree_deleted});
    }
    ok = ok && inner->size == size;
    if (inner->tree_deleted)
      ok = ok && inner->valid == 0;
    else if (!stale)
      ok = ok && inner->valid == valid && same_box(inner->aabb, box);
  }
  return ok && leaves == leaf_count_.load() && inners == inner_count_.load();
}
#endif

template <typename PointType, typename Traits, typename Options>
template <typename Fn>
void KDTree<PointType, Traits, Options>::forEachValid(const Leaf* leaf, Fn&& fn) {
  if (leaf->deleted == 0) {
    for (int i = 0; i < leaf->size; ++i)
      fn(i);
  } else {
    for (int i = 0; i < leaf->size; ++i) {
      if (!(leaf->deleted & bit(i)))
        fn(i);
    }
  }
}

template <typename PointType, typename Traits, typename Options>
int KDTree<PointType, Traits, Options>::longestAxis(const AABB& box) {
  int axis = 0;
  for (int a = 1; a < Traits::DIM; ++a) {
    if (box.max[a] - box.min[a] > box.max[axis] - box.min[axis])
      axis = a;
  }
  return axis;
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::applyOp(const Op& op,
                                        std::vector<Node*>* candidates) {
  switch (op.type) {
    case OpType::kInsert:
      root_ = insertInternal(root_, op.point, candidates);
      break;
    case OpType::kDeletePoint:
      deletePointInternal(root_, op.point, candidates);
      break;
    case OpType::kDeleteBox:
      deleteBoxInternal(root_, op.box, candidates);
      break;
  }
}

template <typename PointType, typename Traits, typename Options>
typename KDTree<PointType, Traits, Options>::Leaf*
KDTree<PointType, Traits, Options>::newLeaf() {
  leaf_count_.fetch_add(1, std::memory_order_relaxed);
  return new Leaf();
}

template <typename PointType, typename Traits, typename Options>
typename KDTree<PointType, Traits, Options>::Inner*
KDTree<PointType, Traits, Options>::newInner(float split, int axis) {
  inner_count_.fetch_add(1, std::memory_order_relaxed);
  return new Inner(split, axis);
}

// Inserts pt below node and returns the subtree's root, which is new if the
// subtree was empty or its root was a full leaf that split.
template <typename PointType, typename Traits, typename Options>
typename KDTree<PointType, Traits, Options>::Node*
KDTree<PointType, Traits, Options>::insertInternal(Node* node, const PointType& pt,
                                          std::vector<Node*>* candidates) {
  if (!node) {
    Leaf* leaf = newLeaf();
    appendPoint(leaf, pt);
    return leaf;
  }
  if (node->is_leaf) {
    Leaf* leaf = asLeaf(node);
    if (leaf->size == LeafSize)
      node = makeRoom(leaf);
    if (node->is_leaf) {
      appendPoint(asLeaf(node), pt);
      return node;
    }
  }
  Inner* inner = asInner(node);
  pushDown(inner);
  bool left = Traits::coord(pt, inner->axis) < inner->split;
  Node*& child = left ? inner->left : inner->right;
  child = insertInternal(child, pt, candidates);
  child->parent = inner;
  child->is_left_child = left;
  updateInner(inner);
  markIfUnbalanced(inner, candidates);
  return inner;
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::appendPoint(Leaf* leaf, const PointType& pt) {
  leaf->pts[leaf->size++] = pt;
  ++leaf->valid;
  leaf->aabb.expand(pt);
}

// Makes room in a full leaf and returns what takes its place, which the
// caller links in. A leaf with deleted slots is compacted and returned
// itself. Otherwise it splits at the median of its points along their longest
// extent: it keeps the lower half, a new leaf takes the upper half, and a new
// inner node holds both.
template <typename PointType, typename Traits, typename Options>
typename KDTree<PointType, Traits, Options>::Node*
KDTree<PointType, Traits, Options>::makeRoom(Leaf* leaf) {
  PointType kept[LeafSize];
  int n = 0;
  forEachValid(leaf, [&](int i) { kept[n++] = leaf->pts[i]; });
  if (n < leaf->size) {
    std::copy(kept, kept + n, leaf->pts);
    leaf->size = n;
    leaf->deleted = 0;
    updateLeaf(leaf);
    return leaf;
  }
  // The leaf's bounds cover exactly the points kept
  int axis = longestAxis(leaf->aabb);
  int m = n / 2;
  std::nth_element(kept, kept + m, kept + n,
                   [axis](const PointType& a, const PointType& b) {
                     return Traits::coord(a, axis) < Traits::coord(b, axis);
                   });
  Inner* inner = newInner(Traits::coord(kept[m], axis), axis);
  Leaf* upper = newLeaf();
  std::copy(kept, kept + m, leaf->pts);
  leaf->size = m;
  leaf->deleted = 0;
  updateLeaf(leaf);
  std::copy(kept + m, kept + n, upper->pts);
  upper->size = n - m;
  updateLeaf(upper);
  inner->parent = leaf->parent;
  inner->is_left_child = leaf->is_left_child;
  inner->left = leaf;
  inner->right = upper;
  leaf->parent = inner;
  leaf->is_left_child = true;
  upper->parent = inner;
  upper->is_left_child = false;
  updateInner(inner);
  return inner;
}

// Descends by bounding box rather than by the split comparison: nth_element
// leaves points equal to the median on both sides, so "equal goes right"
// would miss them.
template <typename PointType, typename Traits, typename Options>
bool KDTree<PointType, Traits, Options>::deletePointInternal(
    Node* node, const PointType& pt, std::vector<Node*>* candidates) {
  if (!node || node->valid == 0 || !node->aabb.contains(pt))
    return false;
  if (node->is_leaf) {
    Leaf* leaf = asLeaf(node);
    for (int i = 0; i < leaf->size; ++i) {
      if (!(leaf->deleted & bit(i)) && samePoint(leaf->pts[i], pt)) {
        leaf->deleted |= bit(i);
        updateLeaf(leaf);
        return true;
      }
    }
    return false;
  }
  Inner* inner = asInner(node);
  if (!deletePointInternal(inner->left, pt, candidates) &&
      !deletePointInternal(inner->right, pt, candidates))
    return false;
  updateInner(inner);
  markIfUnbalanced(inner, candidates);
  return true;
}

// Returns the number of points deleted
template <typename PointType, typename Traits, typename Options>
int KDTree<PointType, Traits, Options>::deleteBoxInternal(
    Node* node, const AABB& box, std::vector<Node*>* candidates) {
  if (!node || node->valid == 0 || !box.intersects(node->aabb))
    return 0;
  if (box.contains(node->aabb)) {
    // Whole subtree inside: tag it in O(1). A dead subtree that is large
    // enough is rebuilt to nothing, which frees its nodes.
    int removed = node->valid;
    killSubtree(node);
    if (!node->is_leaf)
      markIfUnbalanced(asInner(node), candidates);
    return removed;
  }
  int removed = 0;
  if (node->is_leaf) {
    Leaf* leaf = asLeaf(node);
    for (int i = 0; i < leaf->size; ++i) {
      if (!(leaf->deleted & bit(i)) && box.contains(leaf->pts[i])) {
        leaf->deleted |= bit(i);
        ++removed;
      }
    }
    if (removed > 0)
      updateLeaf(leaf);
    return removed;
  }
  Inner* inner = asInner(node);
  removed += deleteBoxInternal(inner->left, box, candidates);
  removed += deleteBoxInternal(inner->right, box, candidates);
  if (removed > 0) {
    updateInner(inner);
    markIfUnbalanced(inner, candidates);
  }
  return removed;
}

// Deletes every point below node in O(1): a leaf marks all its slots, an
// inner node gets the lazy tag.
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::killSubtree(Node* node) {
  if (node->is_leaf)
    asLeaf(node)->deleted = lowBits(node->size);
  else
    node->tree_deleted = true;
  node->valid = 0;
  node->aabb = AABB();
}

// Called by writers before descending into a node
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::pushDown(Inner* node) {
  if (!node->tree_deleted)
    return;
  if (node->left)
    killSubtree(node->left);
  if (node->right)
    killSubtree(node->right);
  node->tree_deleted = false;
}

template <typename PointType, typename Traits, typename Options>
bool KDTree<PointType, Traits, Options>::samePoint(const PointType& a,
                                          const PointType& b) {
  for (int i = 0; i < Traits::DIM; ++i) {
    if (Traits::coord(a, i) != Traits::coord(b, i))
      return false;
  }
  return true;
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::updateLeaf(Leaf* leaf) {
  leaf->valid = 0;
  leaf->aabb = AABB();
  forEachValid(leaf, [&](int i) {
    ++leaf->valid;
    leaf->aabb.expand(leaf->pts[i]);
  });
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::updateInner(Inner* node) {
  node->size = 0;
  node->valid = 0;
  node->aabb = AABB();
  for (Node* child : {node->left, node->right}) {
    if (!child)
      continue;
    node->size += child->size;
    if (!node->tree_deleted && child->valid > 0) {
      node->valid += child->valid;
      node->aabb.expand(child->aabb);
    }
  }
}

template <typename PointType, typename Traits, typename Options>
bool KDTree<PointType, Traits, Options>::needRebuild(const Inner* node) const {
  if (node->size < MIN_REBUILD_SIZE)
    return false;
  int lsz = node->left ? node->left->size : 0;
  int rsz = node->right ? node->right->size : 0;
  int maxsz = std::max(lsz, rsz);
  int deleted = node->size - node->valid;
  return maxsz > KDTREE_ALPHA * node->size ||
         deleted > KDTREE_DELETE_ALPHA * node->size;
}

// Only set need_rebuild to true, never clear it
// Only delete a node marked by need_rebuild in rebuilding thread
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::markIfUnbalanced(
    Inner* node, std::vector<Node*>* candidates) {
  if (candidates && !node->need_rebuild && needRebuild(node)) {
    node->need_rebuild = true;
    candidates->push_back(node);
  }
}

// Appends the non-deleted points. Iterative: an unbalanced chain must not
// overflow the stack.
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::collect(const Node* node,
                                        PointVector<PointType>& pts) const {
  if (!node)
    return;
  std::vector<const Node*> stack{node};
  while (!stack.empty()) {
    const Node* n = stack.back();
    stack.pop_back();
    if (n->valid == 0)
      continue;
    if (n->is_leaf) {
      const Leaf* leaf = asLeaf(n);
      forEachValid(leaf, [&](int i) { pts.push_back(leaf->pts[i]); });
    } else {
      const Inner* inner = asInner(n);
      if (inner->left)
        stack.push_back(inner->left);
      if (inner->right)
        stack.push_back(inner->right);
    }
  }
}

// Iterative, like collect. Node has no virtual destructor, so each node is
// deleted as its real type.
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::destroy(Node* node) {
  std::vector<Node*> stack;
  if (node)
    stack.push_back(node);
  while (!stack.empty()) {
    Node* n = stack.back();
    stack.pop_back();
    if (n->is_leaf) {
      delete asLeaf(n);
      leaf_count_.fetch_sub(1, std::memory_order_relaxed);
    } else {
      Inner* inner = asInner(n);
      if (inner->left)
        stack.push_back(inner->left);
      if (inner->right)
        stack.push_back(inner->right);
      delete inner;
      inner_count_.fetch_sub(1, std::memory_order_relaxed);
    }
  }
}

template <typename PointType, typename Traits, typename Options>
typename KDTree<PointType, Traits, Options>::Node*
KDTree<PointType, Traits, Options>::buildRecursive(PointVector<PointType>& pts, size_t l,
                                          size_t r) {
  if (l >= r)
    return nullptr;
  const size_t n = r - l;
  if (n <= static_cast<size_t>(LeafSize)) {
    Leaf* leaf = newLeaf();
    std::copy(pts.begin() + l, pts.begin() + r, leaf->pts);
    leaf->size = static_cast<int>(n);
    updateLeaf(leaf);
    return leaf;
  }
  // Aim for leaves of BUILD_LEAF_POINTS points: split into `leaves` of them
  // and give the left side leaves / 2 of those. Unlike a median split, the
  // fill rate then does not swing between 50% and 100% with the point count.
  // An odd leaf count makes the sides uneven, at worst 1:2 for three leaves,
  // which only happens below MIN_REBUILD_SIZE; from five leaves on the
  // larger side holds less than 3/5 + 1/n. A new subtree is never
  // unbalanced.
  const size_t target = BUILD_LEAF_POINTS;
  const size_t leaves = std::max<size_t>(2, (n + target - 1) / target);
  const size_t m = l + n * (leaves / 2) / leaves;
  // Split along the longest extent rather than cycling the axes: LiDAR maps
  // are flat, and z splits near the top of the tree prune poorly.
  AABB box;
  for (size_t i = l; i < r; ++i) {
    box.expand(pts[i]);
  }
  int axis = longestAxis(box);
  std::nth_element(pts.begin() + l, pts.begin() + m, pts.begin() + r,
                   [&](const PointType& a, const PointType& b) {
                     return Traits::coord(a, axis) < Traits::coord(b, axis);
                   });
  Inner* node = newInner(Traits::coord(pts[m], axis), axis);
  // Both sides hold at least one point: n > LeafSize >= 2
#ifdef LIKD_TREE_USE_TBB
  if (r - l > MIN_PARALLEL_BUILD_SIZE) {
    tbb::parallel_invoke([&] { node->left = buildRecursive(pts, l, m); },
                         [&] { node->right = buildRecursive(pts, m, r); });
  } else
#endif
  {
    node->left = buildRecursive(pts, l, m);
    node->right = buildRecursive(pts, m, r);
  }
  node->left->parent = node;
  node->left->is_left_child = true;
  node->right->parent = node;
  node->right->is_left_child = false;
  updateInner(node);
  return node;
}

// Called only on nodes with valid > 0
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::nearestNeighborInternal(
    const Node* node, const PointType& query, const PointType*& best_pt,
    float& best_dist2) const {
  if (node->is_leaf) {
    const Leaf* leaf = asLeaf(node);
    forEachValid(leaf, [&](int i) {
      float d2 = Traits::sqrDist(leaf->pts[i], query);
      if (d2 < best_dist2) {
        best_dist2 = d2;
        best_pt = &leaf->pts[i];
      }
    });
    return;
  }
  const Inner* inner = asInner(node);
  bool left_first = Traits::coord(query, inner->axis) < inner->split;
  const Node* near = left_first ? inner->left : inner->right;
  const Node* far = left_first ? inner->right : inner->left;
  if (near && near->valid > 0 && near->aabb.sqrDist(query) < best_dist2)
    nearestNeighborInternal(near, query, best_pt, best_dist2);
  if (far && far->valid > 0 && far->aabb.sqrDist(query) < best_dist2)
    nearestNeighborInternal(far, query, best_pt, best_dist2);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::radiusSearchInternal(
    const Node* node, const PointType& query, float radius2,
    PointVector<PointType>& results, std::vector<float>& distances2) const {
  if (!node || node->valid == 0 || node->aabb.sqrDist(query) > radius2)
    return;
  if (node->is_leaf) {
    const Leaf* leaf = asLeaf(node);
    forEachValid(leaf, [&](int i) {
      float d2 = Traits::sqrDist(leaf->pts[i], query);
      if (d2 <= radius2) {
        results.push_back(leaf->pts[i]);
        distances2.push_back(d2);
      }
    });
    return;
  }
  const Inner* inner = asInner(node);
  radiusSearchInternal(inner->left, query, radius2, results, distances2);
  radiusSearchInternal(inner->right, query, radius2, results, distances2);
}

// Called only on nodes with valid > 0
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::knnSearchInternal(
    const Node* node, const PointType& query, size_t k, float max_dist2,
    std::vector<std::pair<float, const PointType*>>& best) const {
  if (node->is_leaf) {
    const Leaf* leaf = asLeaf(node);
    forEachValid(leaf, [&](int i) {
      float d2 = Traits::sqrDist(leaf->pts[i], query);
      if (d2 <= max_dist2 && (best.size() < k || d2 < best.back().first)) {
        if (best.size() == k)
          best.pop_back();
        auto pos = std::upper_bound(
            best.begin(), best.end(), d2,
            [](float v, const std::pair<float, const PointType*>& e) {
              return v < e.first;
            });
        best.insert(pos, {d2, &leaf->pts[i]});
      }
    });
    return;
  }
  const Inner* inner = asInner(node);
  bool left_first = Traits::coord(query, inner->axis) < inner->split;
  const Node* near = left_first ? inner->left : inner->right;
  const Node* far = left_first ? inner->right : inner->left;
  // Search radius: the k-th best so far, or max_dist until k are found
  if (near && near->valid > 0 &&
      near->aabb.sqrDist(query) <=
          (best.size() < k ? max_dist2 : best.back().first))
    knnSearchInternal(near, query, k, max_dist2, best);
  if (far && far->valid > 0 &&
      far->aabb.sqrDist(query) <=
          (best.size() < k ? max_dist2 : best.back().first))
    knnSearchInternal(far, query, k, max_dist2, best);
}

template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::boxSearchInternal(
    const Node* node, const AABB& box, PointVector<PointType>& results) const {
  if (!node || node->valid == 0 || !box.intersects(node->aabb))
    return;
  if (box.contains(node->aabb)) {
    // Whole subtree inside: no per-point test needed
    collect(node, results);
    return;
  }
  if (node->is_leaf) {
    const Leaf* leaf = asLeaf(node);
    forEachValid(leaf, [&](int i) {
      if (box.contains(leaf->pts[i]))
        results.push_back(leaf->pts[i]);
    });
    return;
  }
  const Inner* inner = asInner(node);
  boxSearchInternal(inner->left, box, results);
  boxSearchInternal(inner->right, box, results);
}

template <typename PointType, typename Traits, typename Options>
bool KDTree<PointType, Traits, Options>::checkAncestorNeedsRebuild(Node* node) const {
  Node* ancestor = node->parent;
  bool needs_rebuild = false;
  while (ancestor) {
    if (ancestor->need_rebuild) {
      needs_rebuild = true;
      break;
    }
    ancestor = ancestor->parent;
  }
  return needs_rebuild;
}

// Filter: remove nodes whose ancestors also need rebuild
template <typename PointType, typename Traits, typename Options>
std::vector<typename KDTree<PointType, Traits, Options>::Node*>
KDTree<PointType, Traits, Options>::topmostCandidates(
    const std::vector<Node*>& candidates) const {
  std::vector<Node*> topmost;
  for (Node* candidate : candidates) {
    if (!checkAncestorNeedsRebuild(candidate)) {
      topmost.push_back(candidate);
    }
  }
  return topmost;
}

// Runs on the worker thread while rebuilding_ is set, i.e. while every writer
// is diverted to pending_ops_: the old subtrees can be read without the tree
// lock.
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::rebuildSubtrees(
    const std::vector<Node*>& nodes_to_rebuild) {
  // Pre-allocate new_nodes for thread-safe parallel access
  std::vector<Node*> new_nodes(nodes_to_rebuild.size());

  // Create index vector for parallel iteration
  std::vector<size_t> indices(nodes_to_rebuild.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::for_each(TREE_PAR, indices.begin(), indices.end(), [&](size_t i) {
    // Deleted points are dropped; a fully deleted subtree becomes nullptr
    PointVector<PointType> pts;
    pts.reserve(nodes_to_rebuild[i]->valid);
    collect(nodes_to_rebuild[i], pts);
    new_nodes[i] = buildRecursive(pts, 0, pts.size());
  });

  // Critical section - swap pointers (brief exclusive lock)
  {
    std::unique_lock<SharedMutex> lock(tree_mutex_);

    for (size_t i = 0; i < nodes_to_rebuild.size(); ++i) {
      Node* old_node = nodes_to_rebuild[i];
      Node* new_node = new_nodes[i];
      Node* parent = old_node->parent;
      if (new_node) {
        new_node->parent = parent;
        new_node->is_left_child = old_node->is_left_child;
      }

      // Update parent's pointer to new subtree
      if (parent) {
        if (old_node->is_left_child) {
          asInner(parent)->left = new_node;
        } else {
          asInner(parent)->right = new_node;
        }
      } else {
        // This was the root
        root_ = new_node;
      }
      // Deleted points are gone: shrink the ancestors' counts
      for (Node* ancestor = parent; ancestor; ancestor = ancestor->parent) {
        updateInner(asInner(ancestor));
      }
    }
  }
  // Readers can no longer reach the old subtrees: free them outside the lock
  for (Node* old_node : nodes_to_rebuild) {
    destroy(old_node);
  }
}

// Background rebuild thread: started on the first rebuild, joined in ~KDTree
template <typename PointType, typename Traits, typename Options>
void KDTree<PointType, Traits, Options>::workerLoop() {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  while (true) {
    job_cv_.wait(lock, [this] { return stop_ || !rebuild_job_.empty(); });
    if (rebuild_job_.empty()) {
      return;  // stop requested
    }
    std::vector<Node*> job = std::move(rebuild_job_);
    rebuild_job_.clear();
    // Writes taken from pending_ops_. Those from `replayed` on are not
    // applied yet; they come before everything queued since.
    OpLog ops;
    size_t replayed = 0;
#ifdef LIKD_TREE_STATS
    std::chrono::steady_clock::time_point queued_since;
#endif
    while (true) {
      lock.unlock();
#ifdef LIKD_TREE_STATS
      auto round_start = std::chrono::steady_clock::now();
#endif
      if (!job.empty()) {
        rebuildSubtrees(job);
      }
#ifdef LIKD_TREE_STATS
      std::chrono::duration<double, std::milli> round =
          std::chrono::steady_clock::now() - round_start;
#endif
      lock.lock();
#ifdef LIKD_TREE_STATS
      if (!job.empty()) {
        ++stats_.rounds;
        stats_.max_round_ms = std::max(stats_.max_round_ms, round.count());
      }
#endif
      if (replayed == ops.size()) {
        // Leave the rebuilding state only after seeing an empty queue while
        // holding pending_mutex_: writers check rebuilding_ under this lock.
        if (pending_ops_.empty()) {
          break;
        }
        ops = std::move(pending_ops_);
        pending_ops_.clear();
        replayed = 0;
#ifdef LIKD_TREE_STATS
        queued_since = queued_since_;
#endif
      }
#ifdef LIKD_TREE_STATS
      held_ops_ = 0;
#endif
      lock.unlock();

      // Replay in order, in small batches to allow queries to interleave.
      // Stop after a batch that unbalances a subtree: the next round rebuilds
      // it before the rest is replayed, so that a long run of writes into one
      // place cannot grow a chain that every later write has to walk.
      std::vector<Node*> candidates;
      job.clear();
      while (replayed < ops.size() && candidates.empty()) {
        std::unique_lock<SharedMutex> tree_lock(tree_mutex_);
        size_t end = std::min(replayed + INSERTION_BATCH_SIZE, ops.size());
        for (; replayed < end; ++replayed) {
          applyOp(ops[replayed], &candidates);
        }
        if (!candidates.empty()) {
          job = topmostCandidates(candidates);
        }
      }
      lock.lock();
#ifdef LIKD_TREE_STATS
      held_ops_ = ops.size() - replayed;
      if (replayed == ops.size()) {
        std::chrono::duration<double, std::milli> queued =
            std::chrono::steady_clock::now() - queued_since;
        stats_.max_queued_ms = std::max(stats_.max_queued_ms, queued.count());
      }
#endif
    }
    rebuilding_ = false;
    idle_cv_.notify_all();
  }
}
