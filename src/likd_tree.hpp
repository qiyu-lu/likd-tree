/*
Copyright 2026 Liu Yang
Copyright 2026 qiyu-lu
Distributed under MIT license. See LICENSE for more information.
*/

// likd-tree: A Lightweight Incremental KD-Tree for dynamic point insertion
// with automatic background rebalancing. Header-only C++17 library.
//
// Thread safety: queries may run concurrently with each other and with
// writers. Writers (build / addPoints / deletePoints / deleteBoxes) are
// serialized internally, so calling them from several threads is safe but
// gains no parallelism.
//
// While a background rebuild runs, writes are queued and applied in order once
// it finishes, so queries may briefly not see the latest writes.

#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <execution>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <vector>

constexpr double KDTREE_ALPHA = 0.75;
// Rebuild a subtree once more than this fraction of its nodes is deleted
constexpr double KDTREE_DELETE_ALPHA = 0.5;
constexpr int INSERTION_BATCH_SIZE = 100;
constexpr int MIN_SUB_NUM = 8;
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

template <typename PointType, typename Traits = PointTraits<PointType>>
class KDTree {
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

  struct Node {
    Node* left = nullptr;
    Node* right = nullptr;
    Node* parent = nullptr;
    AABB aabb;  // bounds of the non-deleted points in this subtree
    PointType point;
    int axis;
    int subtree_size = 1;  // nodes in this subtree, deleted ones included
    int valid_size = 1;    // non-deleted points in this subtree
    bool is_left_child = false;  // true if this is parent's left child
    bool need_rebuild = false;
    bool deleted = false;  // this node's point is deleted
    // Lazy tag: the whole subtree is deleted. Descendants are only updated
    // when a writer descends here (pushDown); queries never get past a node
    // with valid_size == 0, so they never see stale descendants.
    bool tree_deleted = false;
    Node(const PointType& pt, int ax);
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
  // Nodes held in memory, including deleted points not yet reclaimed by a
  // rebuild.
  int nodeCount() const;
  // Blocks until the background rebuild, including the writes queued while
  // it ran, has finished.
  void waitForRebuild() const;

 private:
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

  template <typename ApplyFn, typename EnqueueFn>
  void write(ApplyFn&& apply_now, EnqueueFn&& enqueue, bool wait_for_rebuild);
  void applyOp(const Op& op, std::vector<Node*>* candidates);
  Node* insertInternal(Node* node, const PointType& pt, int depth,
                       std::vector<Node*>* candidates);
  bool deletePointInternal(Node* node, const PointType& pt,
                           std::vector<Node*>* candidates);
  int deleteBoxInternal(Node* node, const AABB& box,
                        std::vector<Node*>* candidates);
  static void killSubtree(Node* node);
  static void pushDown(Node* node);
  static bool samePoint(const PointType& a, const PointType& b);
  void update(Node* node);
  bool needRebuild(Node* node) const;
  void markIfUnbalanced(Node* node, std::vector<Node*>* candidates);
  void collect(Node* node, PointVector<PointType>& pts) const;
  static void destroy(Node* node);
  Node* buildRecursive(PointVector<PointType>& pts, size_t l, size_t r);
  void nearestNeighborInternal(Node* node, const PointType& query,
                               const PointType*& best_pt,
                               float& best_dist2) const;
  void radiusSearchInternal(Node* node, const PointType& query, float radius2,
                            PointVector<PointType>& results,
                            std::vector<float>& distances2) const;
  // best: (squared distance, point), ascending, at most k entries
  void knnSearchInternal(Node* node, const PointType& query, size_t k,
                         float max_dist2,
                         std::vector<std::pair<float, const PointType*>>& best) const;
  void knnSearchLocked(const PointType& query, int k, float max_dist,
                       PointVector<PointType>& results,
                       std::vector<float>& distances) const;
  void boxSearchInternal(Node* node, const AABB& box,
                         PointVector<PointType>& results) const;
  bool checkAncestorNeedsRebuild(Node* node) const;
  std::vector<Node*> topmostCandidates(const std::vector<Node*>& candidates) const;
  void rebuildSubtrees(const std::vector<Node*>& nodes_to_rebuild);
  void workerLoop();

  Node* root_ = nullptr;
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
};

// AABB: axis-aligned bounding boxes
template <typename PointType, typename Traits>
KDTree<PointType, Traits>::AABB::AABB() {
  for (int i = 0; i < Traits::DIM; ++i) {
    min[i] = std::numeric_limits<float>::max();
    max[i] = std::numeric_limits<float>::lowest();
  }
}

template <typename PointType, typename Traits>
KDTree<PointType, Traits>::AABB::AABB(
    const std::array<float, Traits::DIM>& min_corner,
    const std::array<float, Traits::DIM>& max_corner)
    : min(min_corner), max(max_corner) {}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::AABB::expand(const PointType& pt) {
  for (int i = 0; i < Traits::DIM; ++i) {
    min[i] = std::min(min[i], Traits::coord(pt, i));
    max[i] = std::max(max[i], Traits::coord(pt, i));
  }
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::AABB::expand(const AABB& box) {
  for (int i = 0; i < Traits::DIM; ++i) {
    min[i] = std::min(min[i], box.min[i]);
    max[i] = std::max(max[i], box.max[i]);
  }
}

template <typename PointType, typename Traits>
float KDTree<PointType, Traits>::AABB::sqrDist(const PointType& pt) const {
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

template <typename PointType, typename Traits>
bool KDTree<PointType, Traits>::AABB::contains(const PointType& pt) const {
  for (int i = 0; i < Traits::DIM; ++i) {
    float v = Traits::coord(pt, i);
    if (v < min[i] || v > max[i])
      return false;
  }
  return true;
}

template <typename PointType, typename Traits>
bool KDTree<PointType, Traits>::AABB::contains(const AABB& box) const {
  for (int i = 0; i < Traits::DIM; ++i) {
    if (box.min[i] < min[i] || box.max[i] > max[i])
      return false;
  }
  return true;
}

template <typename PointType, typename Traits>
bool KDTree<PointType, Traits>::AABB::intersects(const AABB& box) const {
  for (int i = 0; i < Traits::DIM; ++i) {
    if (box.max[i] < min[i] || box.min[i] > max[i])
      return false;
  }
  return true;
}

template <typename PointType, typename Traits>
KDTree<PointType, Traits>::Node::Node(const PointType& pt, int ax)
    : point(pt), axis(ax) {
  aabb.expand(pt);
}

template <typename PointType, typename Traits>
KDTree<PointType, Traits>::KDTree() : root_(nullptr) {}

template <typename PointType, typename Traits>
KDTree<PointType, Traits>::~KDTree() {
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

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::build(const PointVector<PointType>& pts) {
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

// Runs apply_now(candidates) on the tree, or enqueue(pending_ops_) while a
// rebuild is running, then hands any unbalanced subtrees to the worker.
template <typename PointType, typename Traits>
template <typename ApplyFn, typename EnqueueFn>
void KDTree<PointType, Traits>::write(ApplyFn&& apply_now, EnqueueFn&& enqueue,
                                      bool wait_for_rebuild) {
  std::lock_guard<std::mutex> write_lock(write_mutex_);
  bool buffered = false;
  {
    // rebuilding_ is checked under pending_mutex_, the same lock the worker
    // holds when it leaves the rebuilding state, so queued writes can never
    // be left behind in pending_ops_.
    std::lock_guard<std::mutex> lock(pending_mutex_);
    if (rebuilding_.load()) {
      enqueue(pending_ops_);
      buffered = true;
    }
  }

  if (!buffered) {
    std::vector<Node*> nodes_to_rebuild;
    // Write + filtering phase - protected by exclusive lock
    {
      std::unique_lock<SharedMutex> lock(tree_mutex_);
      std::vector<Node*> candidates;
      apply_now(&candidates);
      nodes_to_rebuild = topmostCandidates(candidates);
    }

    // Hand the unbalanced subtrees to the worker. Writers are serialized and
    // only writers set rebuilding_, so it is still false here.
    if (!nodes_to_rebuild.empty()) {
      {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        rebuilding_ = true;
        rebuild_job_ = std::move(nodes_to_rebuild);
        if (!worker_.joinable())
          worker_ = std::thread(&KDTree::workerLoop, this);
      }
      job_cv_.notify_one();
    }
  }
  // Optionally wait for rebuild to complete before returning
  if (wait_for_rebuild) {
    waitForRebuild();
  }
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::addPoints(const PointVector<PointType>& pts,
                                          bool wait_for_rebuild) {
  write(
      [&](std::vector<Node*>* candidates) {
        for (const auto& p : pts) {
          root_ = insertInternal(root_, p, 0, candidates);
        }
      },
      [&](OpLog& ops) {
        for (const auto& p : pts) {
          ops.push_back({OpType::kInsert, p, AABB()});
        }
      },
      wait_for_rebuild);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::deletePoints(const PointVector<PointType>& pts,
                                             bool wait_for_rebuild) {
  write(
      [&](std::vector<Node*>* candidates) {
        for (const auto& p : pts) {
          deletePointInternal(root_, p, candidates);
        }
      },
      [&](OpLog& ops) {
        for (const auto& p : pts) {
          ops.push_back({OpType::kDeletePoint, p, AABB()});
        }
      },
      wait_for_rebuild);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::deleteBoxes(const std::vector<AABB>& boxes,
                                            bool wait_for_rebuild) {
  write(
      [&](std::vector<Node*>* candidates) {
        for (const auto& box : boxes) {
          deleteBoxInternal(root_, box, candidates);
        }
      },
      [&](OpLog& ops) {
        for (const auto& box : boxes) {
          ops.push_back({OpType::kDeleteBox, PointType(), box});
        }
      },
      wait_for_rebuild);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::deleteBox(const AABB& box,
                                          bool wait_for_rebuild) {
  deleteBoxes(std::vector<AABB>{box}, wait_for_rebuild);
}

template <typename PointType, typename Traits>
std::pair<std::optional<PointType>, float>
KDTree<PointType, Traits>::nearestNeighbors(const PointType& query) const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);

  if (root_ == nullptr || root_->valid_size == 0) {
    return {std::nullopt, INFINITY};
  }

  const PointType* best_pt = nullptr;
  float best_dist2 = INFINITY;
  nearestNeighborInternal(root_, query, best_pt, best_dist2);
  return {*best_pt, std::sqrt(best_dist2)};
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::nearestNeighbors(
    const PointVector<PointType>& queries, PointVector<PointType>& results,
    std::vector<float>& distances) const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);

  results.resize(queries.size());
  distances.assign(queries.size(), INFINITY);

  // If tree is empty, all queries return INFINITY distance
  if (root_ == nullptr || root_->valid_size == 0) {
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

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::radiusSearch(
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

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::knnSearch(const PointType& query, int k,
                                          PointVector<PointType>& results,
                                          std::vector<float>& distances,
                                          float max_dist) const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  knnSearchLocked(query, k, max_dist, results, distances);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::knnSearch(
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
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::knnSearchLocked(
    const PointType& query, int k, float max_dist,
    PointVector<PointType>& results, std::vector<float>& distances) const {
  results.clear();
  distances.clear();
  if (k <= 0 || root_ == nullptr || root_->valid_size == 0 || max_dist < 0.0f) {
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

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::boxSearch(const AABB& box,
                                          PointVector<PointType>& results) const {
  results.clear();
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  boxSearchInternal(root_, box, results);
}

template <typename PointType, typename Traits>
int KDTree<PointType, Traits>::size() const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  return root_ ? root_->valid_size : 0;
}

template <typename PointType, typename Traits>
int KDTree<PointType, Traits>::nodeCount() const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);
  return root_ ? root_->subtree_size : 0;
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::waitForRebuild() const {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  idle_cv_.wait(lock, [this] { return !rebuilding_.load(); });
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::applyOp(const Op& op,
                                        std::vector<Node*>* candidates) {
  switch (op.type) {
    case OpType::kInsert:
      root_ = insertInternal(root_, op.point, 0, candidates);
      break;
    case OpType::kDeletePoint:
      deletePointInternal(root_, op.point, candidates);
      break;
    case OpType::kDeleteBox:
      deleteBoxInternal(root_, op.box, candidates);
      break;
  }
}

template <typename PointType, typename Traits>
typename KDTree<PointType, Traits>::Node*
KDTree<PointType, Traits>::insertInternal(Node* node, const PointType& pt,
                                          int depth,
                                          std::vector<Node*>* candidates) {
  if (!node)
    return new Node(pt, depth % Traits::DIM);
  pushDown(node);
  int ax = node->axis;
  float v = Traits::coord(pt, ax);
  float nv = Traits::coord(node->point, ax);
  if (v < nv) {
    node->left = insertInternal(node->left, pt, depth + 1, candidates);
    node->left->parent = node;
    node->left->is_left_child = true;
  } else {
    node->right = insertInternal(node->right, pt, depth + 1, candidates);
    node->right->parent = node;
    node->right->is_left_child = false;
  }
  update(node);
  markIfUnbalanced(node, candidates);
  return node;
}

// Descends by bounding box rather than by the split comparison: nth_element
// leaves points equal to the median on both sides, so "equal goes right"
// would miss them.
template <typename PointType, typename Traits>
bool KDTree<PointType, Traits>::deletePointInternal(
    Node* node, const PointType& pt, std::vector<Node*>* candidates) {
  if (!node || node->valid_size == 0 || !node->aabb.contains(pt))
    return false;
  bool found = false;
  if (!node->deleted && samePoint(node->point, pt)) {
    node->deleted = true;
    found = true;
  } else {
    found = deletePointInternal(node->left, pt, candidates) ||
            deletePointInternal(node->right, pt, candidates);
  }
  if (found) {
    update(node);
    markIfUnbalanced(node, candidates);
  }
  return found;
}

// Returns the number of points deleted
template <typename PointType, typename Traits>
int KDTree<PointType, Traits>::deleteBoxInternal(
    Node* node, const AABB& box, std::vector<Node*>* candidates) {
  if (!node || node->valid_size == 0 || !box.intersects(node->aabb))
    return 0;
  if (box.contains(node->aabb)) {
    // Whole subtree inside: tag it in O(1). A dead subtree that is large
    // enough is rebuilt to nothing, which frees its nodes.
    int removed = node->valid_size;
    killSubtree(node);
    markIfUnbalanced(node, candidates);
    return removed;
  }
  int removed = 0;
  if (!node->deleted && box.contains(node->point)) {
    node->deleted = true;
    ++removed;
  }
  removed += deleteBoxInternal(node->left, box, candidates);
  removed += deleteBoxInternal(node->right, box, candidates);
  if (removed > 0) {
    update(node);
    markIfUnbalanced(node, candidates);
  }
  return removed;
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::killSubtree(Node* node) {
  node->tree_deleted = true;
  node->valid_size = 0;
  node->aabb = AABB();
}

// Called by writers before descending into a node
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::pushDown(Node* node) {
  if (!node->tree_deleted)
    return;
  node->deleted = true;
  if (node->left)
    killSubtree(node->left);
  if (node->right)
    killSubtree(node->right);
  node->tree_deleted = false;
}

template <typename PointType, typename Traits>
bool KDTree<PointType, Traits>::samePoint(const PointType& a,
                                          const PointType& b) {
  for (int i = 0; i < Traits::DIM; ++i) {
    if (Traits::coord(a, i) != Traits::coord(b, i))
      return false;
  }
  return true;
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::update(Node* node) {
  node->subtree_size = 1;
  if (node->left) {
    node->subtree_size += node->left->subtree_size;
  }
  if (node->right) {
    node->subtree_size += node->right->subtree_size;
  }
  node->aabb = AABB();
  if (node->tree_deleted) {
    node->valid_size = 0;
    return;
  }
  node->valid_size = node->deleted ? 0 : 1;
  if (!node->deleted) {
    node->aabb.expand(node->point);
  }
  for (Node* child : {node->left, node->right}) {
    if (child && child->valid_size > 0) {
      node->valid_size += child->valid_size;
      node->aabb.expand(child->aabb);
    }
  }
}

template <typename PointType, typename Traits>
bool KDTree<PointType, Traits>::needRebuild(Node* node) const {
  if (node->subtree_size < MIN_SUB_NUM)
    return false;
  int lsz = node->left ? node->left->subtree_size : 0;
  int rsz = node->right ? node->right->subtree_size : 0;
  int maxsz = std::max(lsz, rsz);
  int deleted = node->subtree_size - node->valid_size;
  return maxsz > KDTREE_ALPHA * node->subtree_size ||
         deleted > KDTREE_DELETE_ALPHA * node->subtree_size;
}

// Only set need_rebuild to true, never clear it
// Only delete a node marked by need_rebuild in rebuilding thread
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::markIfUnbalanced(
    Node* node, std::vector<Node*>* candidates) {
  if (candidates && !node->need_rebuild && needRebuild(node)) {
    node->need_rebuild = true;
    candidates->push_back(node);
  }
}

// Appends the non-deleted points. Iterative: an unbalanced chain must not
// overflow the stack.
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::collect(Node* node,
                                        PointVector<PointType>& pts) const {
  if (!node)
    return;
  std::vector<Node*> stack{node};
  while (!stack.empty()) {
    Node* n = stack.back();
    stack.pop_back();
    if (n->valid_size == 0)
      continue;
    if (!n->deleted)
      pts.push_back(n->point);
    if (n->left)
      stack.push_back(n->left);
    if (n->right)
      stack.push_back(n->right);
  }
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::destroy(Node* node) {
  std::vector<Node*> stack;
  if (node)
    stack.push_back(node);
  while (!stack.empty()) {
    Node* n = stack.back();
    stack.pop_back();
    if (n->left)
      stack.push_back(n->left);
    if (n->right)
      stack.push_back(n->right);
    delete n;
  }
}

template <typename PointType, typename Traits>
typename KDTree<PointType, Traits>::Node*
KDTree<PointType, Traits>::buildRecursive(PointVector<PointType>& pts, size_t l,
                                          size_t r) {
  if (l >= r)
    return nullptr;
  // Split along the longest extent rather than cycling the axes: LiDAR maps
  // are flat, and z splits near the top of the tree prune poorly.
  AABB box;
  for (size_t i = l; i < r; ++i) {
    box.expand(pts[i]);
  }
  int axis = 0;
  for (int a = 1; a < Traits::DIM; ++a) {
    if (box.max[a] - box.min[a] > box.max[axis] - box.min[axis])
      axis = a;
  }
  size_t m = l + (r - l) / 2;
  std::nth_element(pts.begin() + l, pts.begin() + m, pts.begin() + r,
                   [&](const PointType& a, const PointType& b) {
                     return Traits::coord(a, axis) < Traits::coord(b, axis);
                   });
  Node* node = new Node(pts[m], axis);
#ifdef LIKD_TREE_USE_TBB
  if (r - l > MIN_PARALLEL_BUILD_SIZE) {
    tbb::parallel_invoke([&] { node->left = buildRecursive(pts, l, m); },
                         [&] { node->right = buildRecursive(pts, m + 1, r); });
  } else
#endif
  {
    node->left = buildRecursive(pts, l, m);
    node->right = buildRecursive(pts, m + 1, r);
  }
  if (node->left) {
    node->left->parent = node;
    node->left->is_left_child = true;
  }
  if (node->right) {
    node->right->parent = node;
    node->right->is_left_child = false;
  }
  update(node);
  return node;
}

// Called only on nodes with valid_size > 0
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::nearestNeighborInternal(
    Node* node, const PointType& query, const PointType*& best_pt,
    float& best_dist2) const {
  if (!node->deleted) {
    float d2 = Traits::sqrDist(node->point, query);
    if (d2 < best_dist2) {
      best_dist2 = d2;
      best_pt = &node->point;
    }
  }
  int ax = node->axis;
  float qv = Traits::coord(query, ax);
  float nv = Traits::coord(node->point, ax);
  Node* near = qv < nv ? node->left : node->right;
  Node* far = qv < nv ? node->right : node->left;
  if (near && near->valid_size > 0 && near->aabb.sqrDist(query) < best_dist2)
    nearestNeighborInternal(near, query, best_pt, best_dist2);
  if (far && far->valid_size > 0 && far->aabb.sqrDist(query) < best_dist2)
    nearestNeighborInternal(far, query, best_pt, best_dist2);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::radiusSearchInternal(
    Node* node, const PointType& query, float radius2,
    PointVector<PointType>& results, std::vector<float>& distances2) const {
  if (!node || node->valid_size == 0 || node->aabb.sqrDist(query) > radius2)
    return;

  if (!node->deleted) {
    float d2 = Traits::sqrDist(node->point, query);
    if (d2 <= radius2) {
      results.push_back(node->point);
      distances2.push_back(d2);
    }
  }

  radiusSearchInternal(node->left, query, radius2, results, distances2);
  radiusSearchInternal(node->right, query, radius2, results, distances2);
}

// Called only on nodes with valid_size > 0
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::knnSearchInternal(
    Node* node, const PointType& query, size_t k, float max_dist2,
    std::vector<std::pair<float, const PointType*>>& best) const {
  if (!node->deleted) {
    float d2 = Traits::sqrDist(node->point, query);
    if (d2 <= max_dist2 && (best.size() < k || d2 < best.back().first)) {
      if (best.size() == k)
        best.pop_back();
      auto pos = std::upper_bound(
          best.begin(), best.end(), d2,
          [](float v, const std::pair<float, const PointType*>& e) {
            return v < e.first;
          });
      best.insert(pos, {d2, &node->point});
    }
  }
  int ax = node->axis;
  float qv = Traits::coord(query, ax);
  float nv = Traits::coord(node->point, ax);
  Node* near = qv < nv ? node->left : node->right;
  Node* far = qv < nv ? node->right : node->left;
  // Search radius: the k-th best so far, or max_dist until k are found
  if (near && near->valid_size > 0 &&
      near->aabb.sqrDist(query) <=
          (best.size() < k ? max_dist2 : best.back().first))
    knnSearchInternal(near, query, k, max_dist2, best);
  if (far && far->valid_size > 0 &&
      far->aabb.sqrDist(query) <=
          (best.size() < k ? max_dist2 : best.back().first))
    knnSearchInternal(far, query, k, max_dist2, best);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::boxSearchInternal(
    Node* node, const AABB& box, PointVector<PointType>& results) const {
  if (!node || node->valid_size == 0 || !box.intersects(node->aabb))
    return;
  if (box.contains(node->aabb)) {
    // Whole subtree inside: no per-point test needed
    collect(node, results);
    return;
  }
  if (!node->deleted && box.contains(node->point)) {
    results.push_back(node->point);
  }
  boxSearchInternal(node->left, box, results);
  boxSearchInternal(node->right, box, results);
}

template <typename PointType, typename Traits>
bool KDTree<PointType, Traits>::checkAncestorNeedsRebuild(Node* node) const {
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
template <typename PointType, typename Traits>
std::vector<typename KDTree<PointType, Traits>::Node*>
KDTree<PointType, Traits>::topmostCandidates(
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
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::rebuildSubtrees(
    const std::vector<Node*>& nodes_to_rebuild) {
  // Pre-allocate new_nodes for thread-safe parallel access
  std::vector<Node*> new_nodes(nodes_to_rebuild.size());

  // Create index vector for parallel iteration
  std::vector<size_t> indices(nodes_to_rebuild.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::for_each(TREE_PAR, indices.begin(), indices.end(), [&](size_t i) {
    // Deleted points are dropped; a fully deleted subtree becomes nullptr
    PointVector<PointType> pts;
    pts.reserve(nodes_to_rebuild[i]->valid_size);
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
          parent->left = new_node;
        } else {
          parent->right = new_node;
        }
      } else {
        // This was the root
        root_ = new_node;
      }
      // Deleted nodes are gone: shrink the ancestors' node counts
      for (Node* ancestor = parent; ancestor; ancestor = ancestor->parent) {
        update(ancestor);
      }
    }
  }
  // Readers can no longer reach the old subtrees: free them outside the lock
  for (Node* old_node : nodes_to_rebuild) {
    destroy(old_node);
  }
}

// Background rebuild thread: started on the first rebuild, joined in ~KDTree
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::workerLoop() {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  while (true) {
    job_cv_.wait(lock, [this] { return stop_ || !rebuild_job_.empty(); });
    if (rebuild_job_.empty()) {
      return;  // stop requested
    }
    std::vector<Node*> job = std::move(rebuild_job_);
    rebuild_job_.clear();
    while (true) {
      lock.unlock();
      if (!job.empty()) {
        rebuildSubtrees(job);
      }
      lock.lock();
      // Leave the rebuilding state only after seeing an empty queue while
      // holding pending_mutex_: writers check rebuilding_ under this lock.
      if (pending_ops_.empty()) {
        break;
      }
      OpLog ops = std::move(pending_ops_);
      pending_ops_.clear();
      lock.unlock();

      // Replay in order, in small batches to allow queries to interleave.
      // Subtrees these writes unbalance are rebuilt in the next round.
      std::vector<Node*> candidates;
      for (size_t i = 0; i < ops.size(); i += INSERTION_BATCH_SIZE) {
        std::unique_lock<SharedMutex> tree_lock(tree_mutex_);
        size_t end = std::min(i + INSERTION_BATCH_SIZE, ops.size());
        for (size_t j = i; j < end; ++j) {
          applyOp(ops[j], &candidates);
        }
        if (end == ops.size()) {
          job = topmostCandidates(candidates);
        }
      }
      lock.lock();
    }
    rebuilding_ = false;
    idle_cv_.notify_all();
  }
}
