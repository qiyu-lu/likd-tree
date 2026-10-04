/*
Copyright 2026 Liu Yang
Distributed under MIT license. See LICENSE for more information.
*/

// likd-tree: A Lightweight Incremental KD-Tree for dynamic point insertion
// with automatic background rebalancing. Header-only C++17 library.
//
// Thread safety: queries may run concurrently with each other and with
// writers. Writers (build / addPoints) are serialized internally, so calling
// them from several threads is safe but gains no parallelism.

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
constexpr int INSERTION_BATCH_SIZE = 100;
constexpr int MIN_SUB_NUM = 8;
#ifdef LIKD_TREE_USE_TBB
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
    AABB aabb;
    PointType point;
    int axis;
    int subtree_size = 1;
    bool is_left_child = false;  // true if this is parent's left child
    bool need_rebuild = false;
    Node(const PointType& pt, int ax);
  };

  KDTree();
  ~KDTree();

  void build(const PointVector<PointType>& pts);
  void addPoints(const PointVector<PointType>& pts, bool wait_for_rebuild = false);
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
  // Points still buffered by a running rebuild are not counted.
  int size() const;
  // Blocks until the background rebuild, including the insertion of points
  // buffered while it ran, has finished.
  void waitForRebuild() const;

 private:
  // std::shared_mutex on glibc prefers readers: back-to-back queries from
  // other threads could starve addPoints() and the rebuild swap forever.
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

  Node* insertInternal(Node* node, const PointType& pt, int depth,
                       std::vector<Node*>* candidates);
  void update(Node* node);
  bool needRebuild(Node* node) const;
  void collect(Node* node, PointVector<PointType>& pts) const;
  static void destroy(Node* node);
  Node* buildRecursive(PointVector<PointType>& pts, size_t l, size_t r,
                       int axis);
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
  // Serializes writers (build / addPoints).
  std::mutex write_mutex_;
  // Guards rebuilding_ transitions, pending_points_, rebuild_job_ and stop_.
  mutable std::mutex pending_mutex_;
  mutable std::condition_variable idle_cv_;
  std::condition_variable job_cv_;
  std::atomic<bool> rebuilding_{false};
  PointVector<PointType> pending_points_;
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
  Node* new_root = tmp.empty() ? nullptr : buildRecursive(tmp, 0, tmp.size(), 0);
  Node* old_root;
  {
    std::unique_lock<SharedMutex> lock(tree_mutex_);
    old_root = root_;
    root_ = new_root;
  }
  destroy(old_root);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::addPoints(const PointVector<PointType>& pts,
                                          bool wait_for_rebuild) {
  std::lock_guard<std::mutex> write_lock(write_mutex_);
  bool buffered = false;
  {
    // rebuilding_ is checked under pending_mutex_, the same lock the worker
    // holds when it leaves the rebuilding state, so buffered points can never
    // be left behind in pending_points_.
    std::lock_guard<std::mutex> lock(pending_mutex_);
    if (rebuilding_.load()) {
      pending_points_.insert(pending_points_.end(), pts.begin(), pts.end());
      buffered = true;
    }
  }

  if (!buffered) {
    std::vector<Node*> nodes_to_rebuild;
    // Insertion + filtering phase - protected by exclusive lock
    {
      std::unique_lock<SharedMutex> lock(tree_mutex_);
      std::vector<Node*> candidates;
      for (const auto& p : pts) {
        root_ = insertInternal(root_, p, 0, &candidates);
      }
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
std::pair<std::optional<PointType>, float>
KDTree<PointType, Traits>::nearestNeighbors(const PointType& query) const {
  std::shared_lock<SharedMutex> lock(tree_mutex_);

  if (root_ == nullptr) {
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
  if (root_ == nullptr) {
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
  if (k <= 0 || root_ == nullptr || max_dist < 0.0f) {
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
  return root_ ? root_->subtree_size : 0;
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::waitForRebuild() const {
  std::unique_lock<std::mutex> lock(pending_mutex_);
  idle_cv_.wait(lock, [this] { return !rebuilding_.load(); });
}

template <typename PointType, typename Traits>
typename KDTree<PointType, Traits>::Node*
KDTree<PointType, Traits>::insertInternal(Node* node, const PointType& pt,
                                          int depth,
                                          std::vector<Node*>* candidates) {
  if (!node)
    return new Node(pt, depth % Traits::DIM);
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
  // Only set need_rebuild to true, never clear it
  // Only delete a node marked by need_rebuild in rebuilding thread
  if (candidates && !node->need_rebuild && needRebuild(node)) {
    node->need_rebuild = true;
    candidates->push_back(node);
  }
  return node;
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::update(Node* node) {
  node->subtree_size = 1;
  node->aabb = AABB();
  node->aabb.expand(node->point);
  if (node->left) {
    node->subtree_size += node->left->subtree_size;
    node->aabb.expand(node->left->aabb);
  }
  if (node->right) {
    node->subtree_size += node->right->subtree_size;
    node->aabb.expand(node->right->aabb);
  }
}

template <typename PointType, typename Traits>
bool KDTree<PointType, Traits>::needRebuild(Node* node) const {
  int lsz = node->left ? node->left->subtree_size : 0;
  int rsz = node->right ? node->right->subtree_size : 0;
  int maxsz = std::max(lsz, rsz);
  return maxsz > KDTREE_ALPHA * node->subtree_size &&
         node->subtree_size >= MIN_SUB_NUM;
}

// Iterative: an unbalanced chain must not overflow the stack.
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::collect(Node* node,
                                        PointVector<PointType>& pts) const {
  if (!node)
    return;
  std::vector<Node*> stack{node};
  while (!stack.empty()) {
    Node* n = stack.back();
    stack.pop_back();
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
                                          size_t r, int axis) {
  if (l >= r)
    return nullptr;
  size_t m = l + (r - l) / 2;
  std::nth_element(pts.begin() + l, pts.begin() + m, pts.begin() + r,
                   [&](const PointType& a, const PointType& b) {
                     return Traits::coord(a, axis) < Traits::coord(b, axis);
                   });
  Node* node = new Node(pts[m], axis);
  node->left = buildRecursive(pts, l, m, (axis + 1) % Traits::DIM);
  if (node->left) {
    node->left->parent = node;
    node->left->is_left_child = true;
  }
  node->right = buildRecursive(pts, m + 1, r, (axis + 1) % Traits::DIM);
  if (node->right) {
    node->right->parent = node;
    node->right->is_left_child = false;
  }
  update(node);
  return node;
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::nearestNeighborInternal(
    Node* node, const PointType& query, const PointType*& best_pt,
    float& best_dist2) const {
  if (!node)
    return;
  float d2 = Traits::sqrDist(node->point, query);
  if (d2 < best_dist2) {
    best_dist2 = d2;
    best_pt = &node->point;
  }
  int ax = node->axis;
  float qv = Traits::coord(query, ax);
  float nv = Traits::coord(node->point, ax);
  Node* near = qv < nv ? node->left : node->right;
  Node* far = qv < nv ? node->right : node->left;
  if (near)
    nearestNeighborInternal(near, query, best_pt, best_dist2);
  if (far && far->aabb.sqrDist(query) < best_dist2)
    nearestNeighborInternal(far, query, best_pt, best_dist2);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::radiusSearchInternal(
    Node* node, const PointType& query, float radius2,
    PointVector<PointType>& results, std::vector<float>& distances2) const {
  if (!node || node->aabb.sqrDist(query) > radius2)
    return;

  float d2 = Traits::sqrDist(node->point, query);
  if (d2 <= radius2) {
    results.push_back(node->point);
    distances2.push_back(d2);
  }

  radiusSearchInternal(node->left, query, radius2, results, distances2);
  radiusSearchInternal(node->right, query, radius2, results, distances2);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::knnSearchInternal(
    Node* node, const PointType& query, size_t k, float max_dist2,
    std::vector<std::pair<float, const PointType*>>& best) const {
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
  int ax = node->axis;
  float qv = Traits::coord(query, ax);
  float nv = Traits::coord(node->point, ax);
  Node* near = qv < nv ? node->left : node->right;
  Node* far = qv < nv ? node->right : node->left;
  // Search radius: the k-th best so far, or max_dist until k are found
  if (near && near->aabb.sqrDist(query) <=
                  (best.size() < k ? max_dist2 : best.back().first))
    knnSearchInternal(near, query, k, max_dist2, best);
  if (far && far->aabb.sqrDist(query) <=
                 (best.size() < k ? max_dist2 : best.back().first))
    knnSearchInternal(far, query, k, max_dist2, best);
}

template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::boxSearchInternal(
    Node* node, const AABB& box, PointVector<PointType>& results) const {
  if (!node || !box.intersects(node->aabb))
    return;
  if (box.contains(node->aabb)) {
    // Whole subtree inside: no per-point test needed
    collect(node, results);
    return;
  }
  if (box.contains(node->point)) {
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
// is diverted to pending_points_: the old subtrees can be read without the
// tree lock.
template <typename PointType, typename Traits>
void KDTree<PointType, Traits>::rebuildSubtrees(
    const std::vector<Node*>& nodes_to_rebuild) {
  // Pre-allocate new_nodes for thread-safe parallel access
  std::vector<Node*> new_nodes(nodes_to_rebuild.size());

  // Create index vector for parallel iteration
  std::vector<size_t> indices(nodes_to_rebuild.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::for_each(TREE_PAR, indices.begin(), indices.end(), [&](size_t i) {
    PointVector<PointType> pts;
    pts.reserve(nodes_to_rebuild[i]->subtree_size);
    collect(nodes_to_rebuild[i], pts);

    new_nodes[i] =
        buildRecursive(pts, 0, pts.size(), nodes_to_rebuild[i]->axis);

    new_nodes[i]->parent = nodes_to_rebuild[i]->parent;
    new_nodes[i]->is_left_child = nodes_to_rebuild[i]->is_left_child;
  });

  // Critical section - swap pointers (brief exclusive lock)
  {
    std::unique_lock<SharedMutex> lock(tree_mutex_);

    for (size_t i = 0; i < nodes_to_rebuild.size(); ++i) {
      Node* new_node = new_nodes[i];

      // Update parent's pointer to new subtree
      if (new_node->parent) {
        if (new_node->is_left_child) {
          new_node->parent->left = new_node;
        } else {
          new_node->parent->right = new_node;
        }
      } else {
        // This was the root
        root_ = new_node;
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
      // Leave the rebuilding state only after seeing an empty buffer while
      // holding pending_mutex_: writers check rebuilding_ under this lock.
      if (pending_points_.empty()) {
        break;
      }
      PointVector<PointType> points_to_insert = std::move(pending_points_);
      pending_points_.clear();
      lock.unlock();

      // Insert in small batches to allow queries to interleave. Imbalance
      // caused by these points is detected too and rebuilt in the next round.
      std::vector<Node*> candidates;
      for (size_t i = 0; i < points_to_insert.size(); i += INSERTION_BATCH_SIZE) {
        std::unique_lock<SharedMutex> tree_lock(tree_mutex_);
        size_t end = std::min(i + INSERTION_BATCH_SIZE, points_to_insert.size());
        for (size_t j = i; j < end; ++j) {
          root_ = insertInternal(root_, points_to_insert[j], 0, &candidates);
        }
        if (end == points_to_insert.size()) {
          job = topmostCandidates(candidates);
        }
      }
      lock.lock();
    }
    rebuilding_ = false;
    idle_cv_.notify_all();
  }
}
