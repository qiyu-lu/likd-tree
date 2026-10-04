// unit_tests.cpp - correctness and thread-safety tests for likd-tree.
// Every query is checked against brute force. Returns non-zero on failure.
// The brute-force tests run for leaf sizes 2, 4, 32 (the default) and 64:
// small leaves split and rebuild often. The concurrency tests run for leaf
// sizes 2 and 32.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

// Compiles KDTree::validate()
#define LIKD_TREE_TESTING
#include "../src/likd_tree.hpp"

// Sanitizers slow the tests down several times: skip the timing checks
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define LIKD_TEST_SANITIZED
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define LIKD_TEST_SANITIZED
#endif
#endif

namespace {

struct Pt {
  float x, y, z;
};
using Points = PointVector<Pt>;

template <int N>
struct LeafOptions : DefaultOptions {
  static constexpr int LEAF_SIZE = N;
};
template <int N>
using LeafTree = KDTree<Pt, PointTraits<Pt>, LeafOptions<N>>;

int g_failures = 0;

#define CHECK(cond)                                              \
  do {                                                           \
    if (!(cond)) {                                               \
      ++g_failures;                                              \
      std::printf("  FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                            \
  } while (0)

float dist(const Pt& a, const Pt& b) {
  return std::sqrt(PointTraits<Pt>::sqrDist(a, b));
}

// Gaussian blob around a random center: skewed batches trigger rebuilds.
Points blob(std::mt19937& rng, size_t n, float spread = 3.0f) {
  std::uniform_real_distribution<float> center(-50.0f, 50.0f);
  std::normal_distribution<float> noise(0.0f, spread);
  float cx = center(rng), cy = center(rng), cz = center(rng);
  Points pts(n);
  for (auto& p : pts) {
    p = {cx + noise(rng), cy + noise(rng), cz + noise(rng)};
  }
  return pts;
}

// Gaussian blob around a given center
Points cluster(std::mt19937& rng, size_t n, Pt c, float spread) {
  std::normal_distribution<float> noise(0.0f, spread);
  Points pts(n);
  for (auto& p : pts) {
    p = {c.x + noise(rng), c.y + noise(rng), c.z + noise(rng)};
  }
  return pts;
}

float bruteNearest(const Points& pts, const Pt& q) {
  float best = INFINITY;
  for (const auto& p : pts) best = std::min(best, dist(p, q));
  return best;
}

template <typename Tree>
bool matchesBruteForce(const Tree& tree, const Points& all, std::mt19937& rng,
                       int num_queries) {
  std::uniform_real_distribution<float> u(-60.0f, 60.0f);
  bool ok = true;
  for (int i = 0; i < num_queries; ++i) {
    Pt q{u(rng), u(rng), u(rng)};
    auto [nearest, d] = tree.nearestNeighbors(q);
    float expected = bruteNearest(all, q);
    ok = ok && nearest && std::fabs(d - expected) < 1e-4f &&
         std::fabs(dist(*nearest, q) - expected) < 1e-4f;

    Points res;
    std::vector<float> dists;
    tree.radiusSearch(q, 6.0f, res, dists);
    size_t count = 0;
    for (const auto& p : all) count += dist(p, q) <= 6.0f;
    ok = ok && res.size() == count;
  }
  return ok;
}

// Sorted distances of the k nearest points within max_dist
std::vector<float> bruteKnn(const Points& pts, const Pt& q, size_t k,
                            float max_dist = INFINITY) {
  std::vector<float> d;
  for (const auto& p : pts) {
    float v = dist(p, q);
    if (v <= max_dist) d.push_back(v);
  }
  k = std::min(d.size(), k);
  std::partial_sort(d.begin(), d.begin() + k, d.end());
  d.resize(k);
  return d;
}

bool sameDistances(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::fabs(a[i] - b[i]) > 1e-4f) return false;
  return true;
}

// Removes one copy of p from the model, if there is one
void eraseOne(Points& model, const Pt& p) {
  auto it = std::find_if(model.begin(), model.end(), [&](const Pt& m) {
    return m.x == p.x && m.y == p.y && m.z == p.z;
  });
  if (it != model.end()) {
    *it = model.back();
    model.pop_back();
  }
}

template <typename Tree>
void testIncrementalQueries() {
  std::printf("[incremental queries]\n");
  std::mt19937 rng(1);
  Tree tree;
  Points all = blob(rng, 2000);
  tree.build(all);
  for (int i = 0; i < 60; ++i) {
    Points b = blob(rng, 500);
    tree.addPoints(b, true);
    all.insert(all.end(), b.begin(), b.end());
  }
  CHECK(tree.size() == static_cast<int>(all.size()));
  CHECK(matchesBruteForce(tree, all, rng, 300));

  Tree empty;
  CHECK(!empty.nearestNeighbors(Pt{0, 0, 0}).first);
  CHECK(empty.size() == 0);
}

template <typename Tree>
void testKnnSearch() {
  std::printf("[knn search]\n");
  std::mt19937 rng(7);
  Tree tree;
  Points all = blob(rng, 3000, 10.0f);
  tree.build(all);
  for (int i = 0; i < 40; ++i) {
    Points b = blob(rng, 500);
    tree.addPoints(b, true);
    all.insert(all.end(), b.begin(), b.end());
  }
  std::uniform_real_distribution<float> u(-60.0f, 60.0f);
  Points queries;
  for (int i = 0; i < 300; ++i) {
    Pt q{u(rng), u(rng), u(rng)};
    queries.push_back(q);
    for (int k : {1, 5, 17}) {
      for (float max_dist : {INFINITY, 4.0f}) {
        Points res;
        std::vector<float> d;
        tree.knnSearch(q, k, res, d, max_dist);
        bool ok = sameDistances(d, bruteKnn(all, q, k, max_dist)) &&
                  res.size() == d.size();
        for (size_t j = 0; ok && j < res.size(); ++j)
          ok = std::fabs(dist(res[j], q) - d[j]) < 1e-4f;
        CHECK(ok);
      }
    }
  }
  // Batch API agrees with the single-query API
  std::vector<Points> batch_res;
  std::vector<std::vector<float>> batch_d;
  tree.knnSearch(queries, 5, batch_res, batch_d);
  CHECK(batch_res.size() == queries.size() && batch_d.size() == queries.size());
  for (size_t i = 0; i < queries.size(); ++i)
    CHECK(sameDistances(batch_d[i], bruteKnn(all, queries[i], 5)));

  Points res;
  std::vector<float> d;
  tree.knnSearch(queries[0], 0, res, d);
  CHECK(res.empty() && d.empty());
  Tree empty;
  empty.knnSearch(queries[0], 5, res, d);
  CHECK(res.empty() && d.empty());
}

template <typename Tree>
void testBoxSearch() {
  std::printf("[box search]\n");
  std::mt19937 rng(8);
  Tree tree;
  Points all = blob(rng, 3000, 10.0f);
  tree.build(all);
  for (int i = 0; i < 40; ++i) {
    Points b = blob(rng, 500);
    tree.addPoints(b, true);
    all.insert(all.end(), b.begin(), b.end());
  }
  std::uniform_real_distribution<float> u(-60.0f, 60.0f);
  std::uniform_real_distribution<float> half(0.5f, 30.0f);
  for (int i = 0; i < 300; ++i) {
    float cx = u(rng), cy = u(rng), cz = u(rng), h = half(rng);
    typename Tree::AABB box({cx - h, cy - h, cz - h}, {cx + h, cy + h, cz + h});
    Points res;
    tree.boxSearch(box, res);
    size_t expected = std::count_if(all.begin(), all.end(),
                                    [&](const Pt& p) { return box.contains(p); });
    bool ok = res.size() == expected;
    for (const auto& p : res) ok = ok && box.contains(p);
    CHECK(ok);
  }
  // Box containing everything, and an empty one
  Points res;
  tree.boxSearch(typename Tree::AABB({-1e9f, -1e9f, -1e9f}, {1e9f, 1e9f, 1e9f}),
                 res);
  CHECK(res.size() == all.size());
  tree.boxSearch(typename Tree::AABB({1e8f, 1e8f, 1e8f}, {1e9f, 1e9f, 1e9f}), res);
  CHECK(res.empty());
}

// Coordinates snapped to a 0.5 grid: many points tie on the split axes.
Points gridBlob(std::mt19937& rng, size_t n) {
  Points pts = blob(rng, n, 4.0f);
  for (auto& p : pts) {
    p = {std::round(p.x * 2) / 2, std::round(p.y * 2) / 2,
         std::round(p.z * 2) / 2};
  }
  return pts;
}

template <typename Tree>
bool sameAsModel(const Tree& tree, const Points& model, std::mt19937& rng) {
  if (tree.size() != static_cast<int>(model.size())) return false;
  std::uniform_real_distribution<float> u(-60.0f, 60.0f);
  for (int i = 0; i < 30; ++i) {
    Pt q{u(rng), u(rng), u(rng)};
    int k = 1 + i % 8;
    Points res;
    std::vector<float> d;
    tree.knnSearch(q, k, res, d);
    if (!sameDistances(d, bruteKnn(model, q, k))) return false;
    auto [nearest, nd] = tree.nearestNeighbors(q);
    if (model.empty() ? nearest.has_value()
                      : std::fabs(nd - bruteNearest(model, q)) > 1e-4f)
      return false;
    tree.radiusSearch(q, 8.0f, res, d);
    size_t in_radius = 0;
    for (const auto& p : model) in_radius += dist(p, q) <= 8.0f;
    if (res.size() != in_radius) return false;
    typename Tree::AABB box({q.x - 6, q.y - 6, q.z - 6}, {q.x + 6, q.y + 6, q.z + 6});
    tree.boxSearch(box, res);
    size_t in_box = std::count_if(model.begin(), model.end(),
                                  [&](const Pt& p) { return box.contains(p); });
    if (res.size() != in_box) return false;
  }
  return true;
}

// Random inserts, point deletes and box deletes against a brute-force model.
// Without wait_for_rebuild, many writes land in the queue of a running rebuild.
template <typename Tree>
void testDeleteDifferential() {
  std::printf("[delete: differential]\n");
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> u(-50.0f, 50.0f);
  Tree tree;
  Points model = gridBlob(rng, 3000);
  tree.build(model);
  for (int round = 0; round < 600; ++round) {
    int kind = rng() % 10;
    if (kind < 5) {
      Points b = gridBlob(rng, 200 + rng() % 800);
      tree.addPoints(b);
      model.insert(model.end(), b.begin(), b.end());
    } else if (kind < 8) {
      Points del;
      for (int i = 0; i < 300 && !model.empty(); ++i)
        del.push_back(model[rng() % model.size()]);
      del.push_back({1000.0f, 1000.0f, 1000.0f});  // absent: no-op
      tree.deletePoints(del);
      for (const auto& p : del) eraseOne(model, p);
    } else {
      float cx = u(rng), cy = u(rng), h = 2.0f + rng() % 15;
      typename Tree::AABB box({cx - h, cy - h, -100.0f}, {cx + h, cy + h, 100.0f});
      tree.deleteBox(box);
      model.erase(std::remove_if(model.begin(), model.end(),
                                 [&](const Pt& p) { return box.contains(p); }),
                  model.end());
    }
    if (round % 6 == 5) {
      tree.waitForRebuild();
      CHECK(tree.validate());
      CHECK(sameAsModel(tree, model, rng));
    }
  }
  tree.waitForRebuild();
  CHECK(tree.validate());
  CHECK(sameAsModel(tree, model, rng));
  // Rebuilds reclaim deleted points
  CHECK(tree.nodeCount() <= 2 * tree.size() + 64);
}

// Every point deleted once by value, with many ties on split values.
template <typename Tree>
void testDeleteTies() {
  std::printf("[delete: tied coordinates]\n");
  std::mt19937 rng(10);
  Points pts = gridBlob(rng, 20000);
  Tree tree;
  tree.build(pts);
  tree.deletePoints(pts, true);
  CHECK(tree.validate());
  CHECK(tree.size() == 0);
  CHECK(!tree.nearestNeighbors(Pt{0, 0, 0}).first);
}

template <typename Tree>
void testDeleteAll() {
  std::printf("[delete: everything, then reuse]\n");
  std::mt19937 rng(11);
  Tree tree;
  tree.build(blob(rng, 5000));
  tree.deleteBox(typename Tree::AABB({-1e9f, -1e9f, -1e9f}, {1e9f, 1e9f, 1e9f}),
                 true);
  CHECK(tree.validate());
  CHECK(tree.size() == 0);
  CHECK(tree.nodeCount() == 0);
  CHECK(tree.memoryUsage() == 0);
  CHECK(!tree.nearestNeighbors(Pt{0, 0, 0}).first);
  Points res, queries(3, Pt{0, 0, 0});
  std::vector<float> d;
  tree.knnSearch(Pt{0, 0, 0}, 5, res, d);
  CHECK(res.empty());
  tree.nearestNeighbors(queries, res, d);
  CHECK(d.size() == 3 && std::isinf(d[0]));
  Points again = blob(rng, 1000);
  tree.addPoints(again, true);
  CHECK(tree.size() == 1000);
  CHECK(matchesBruteForce(tree, again, rng, 50));
}

// Deleting most points and letting the rebuild reclaim them shrinks the memory.
template <typename Tree>
void testMemoryUsage() {
  std::printf("[memory usage]\n");
  std::mt19937 rng(12);
  Tree tree;
  CHECK(tree.memoryUsage() == 0);
  Points pts = blob(rng, 20000, 10.0f);
  tree.build(pts);
  size_t full = tree.memoryUsage();
  CHECK(full >= pts.size() * sizeof(Pt));
  std::vector<float> xs;
  for (const auto& p : pts) xs.push_back(p.x);
  std::nth_element(xs.begin(), xs.begin() + xs.size() * 9 / 10, xs.end());
  float x90 = xs[xs.size() * 9 / 10];
  tree.deleteBox(typename Tree::AABB({-1e9f, -1e9f, -1e9f}, {x90, 1e9f, 1e9f}),
                 true);
  CHECK(tree.validate());
  CHECK(tree.size() < static_cast<int>(pts.size()) / 5);
  CHECK(tree.memoryUsage() <= full / 4);
}

// More identical points than a leaf holds: no split value separates them.
template <typename Tree>
void testDuplicates() {
  std::printf("[duplicates]\n");
  std::mt19937 rng(14);
  const Pt same{1.5f, -2.0f, 3.0f};
  Points model = blob(rng, 500);
  model.insert(model.end(), 100, same);
  Tree tree;
  tree.build(model);
  // One point per write, so that full leaves of copies keep splitting
  for (int i = 0; i < 150; ++i) {
    tree.addPoints(Points{same});
    model.push_back(same);
  }
  tree.addPoints(Points(150, same), true);
  model.insert(model.end(), 150, same);
  CHECK(tree.validate());
  CHECK(sameAsModel(tree, model, rng));
  Points res;
  std::vector<float> d;
  tree.knnSearch(same, 64, res, d);
  CHECK(d.size() == 64 && d.back() == 0.0f);
  tree.radiusSearch(same, 0.0f, res, d);
  CHECK(res.size() == 400);
  // Delete the copies one at a time
  for (int i = 0; i < 400; ++i) {
    tree.deletePoints(Points{same});
    eraseOne(model, same);
    if (i % 100 == 99) {
      tree.waitForRebuild();
      CHECK(tree.validate());
      CHECK(sameAsModel(tree, model, rng));
      tree.radiusSearch(same, 0.0f, res, d);
      CHECK(res.size() == static_cast<size_t>(399 - i));
    }
  }
  tree.deletePoints(Points{same}, true);  // none left: no-op
  CHECK(tree.size() == static_cast<int>(model.size()));
}

// Deleting every point of a region, one by one, empties whole leaves; the
// region then fills up again.
template <typename Tree>
void testEmptiedLeaves() {
  std::printf("[emptied leaves]\n");
  std::mt19937 rng(15);
  Points model = blob(rng, 4000, 10.0f);
  Tree tree;
  tree.build(model);
  std::vector<float> xs;
  for (const auto& p : model) xs.push_back(p.x);
  std::nth_element(xs.begin(), xs.begin() + xs.size() * 3 / 10, xs.end());
  float x30 = xs[xs.size() * 3 / 10];
  Points gone;
  for (const auto& p : model)
    if (p.x < x30) gone.push_back(p);
  tree.deletePoints(gone, true);
  for (const auto& p : gone) eraseOne(model, p);
  CHECK(tree.validate());
  CHECK(sameAsModel(tree, model, rng));
  Points back(gone.begin(), gone.begin() + gone.size() / 2);
  tree.addPoints(back, true);
  model.insert(model.end(), back.begin(), back.end());
  CHECK(tree.validate());
  CHECK(sameAsModel(tree, model, rng));
}

// Box deletion tags whole subtrees, and a rebuild can leave an inner node with
// an empty child. Later inserts land in both kinds of places.
template <typename Tree>
void testInsertAfterDeletingSubtrees() {
  std::printf("[insert after deleting subtrees]\n");
  std::mt19937 rng(16);
  const Pt a{-30, 0, 0}, b{30, 0, 0};
  typename Tree::AABB around_b({20, -15, -15}, {40, 15, 15});
  typename Tree::AABB half_of_a({-45, -15, -15}, {-30, 15, 15});
  Points model = cluster(rng, 3000, a, 2.0f);
  Points bs = cluster(rng, 1000, b, 2.0f);
  model.insert(model.end(), bs.begin(), bs.end());
  Tree tree;
  tree.build(model);
  auto erase_box = [&](const typename Tree::AABB& box) {
    model.erase(std::remove_if(model.begin(), model.end(),
                               [&](const Pt& p) { return box.contains(p); }),
                model.end());
  };
  for (int round = 0; round < 3; ++round) {
    // Kill b and half of a, then refill both without waiting: some inserts
    // pass through tagged subtrees
    tree.deleteBoxes({around_b, half_of_a});
    erase_box(around_b);
    erase_box(half_of_a);
    for (int i = 0; i < 5; ++i) {
      Points more = cluster(rng, 100, i % 2 ? b : Pt{-35, 0, 0}, 2.0f);
      tree.addPoints(more);
      model.insert(model.end(), more.begin(), more.end());
    }
    tree.waitForRebuild();
    CHECK(tree.validate());
    CHECK(sameAsModel(tree, model, rng));
    // Kill b again and let the rebuild drop it, then insert where it was
    tree.deleteBox(around_b, true);
    erase_box(around_b);
    Points more = cluster(rng, 300, b, 2.0f);
    tree.addPoints(more, true);
    model.insert(model.end(), more.begin(), more.end());
    CHECK(tree.validate());
    CHECK(sameAsModel(tree, model, rng));
  }
}

// A full leaf with a deleted slot takes a new point by compacting, without a
// split. N is the leaf size.
template <typename Tree, int N>
void testReuseDeletedSlot() {
  std::printf("[deleted slots reused]\n");
  std::mt19937 rng(18);
  Points pts = blob(rng, N);
  Tree tree;
  tree.build(pts);  // a single full leaf
  size_t one_leaf = tree.memoryUsage();
  tree.deletePoints(Points{pts[0]}, true);
  tree.addPoints(Points{Pt{pts[0].x + 0.5f, pts[0].y, pts[0].z}}, true);
  CHECK(tree.memoryUsage() == one_leaf);
  CHECK(tree.nodeCount() == N);
  CHECK(tree.size() == N);
  CHECK(tree.validate());
}

// One long write is applied in chunks: its first chunk unbalances the tree,
// and the rest of it is queued behind the rebuild, ahead of the calls that
// follow. The worker then stops replaying whenever a batch unbalances a
// subtree. Through all of that, deletes must still come after the inserts
// they refer to, and inserts after the deletes before them.
template <typename Tree>
void testWriteOrderAcrossChunks() {
  std::printf("[write order across chunks]\n");
  std::mt19937 rng(20);
  const size_t n = 100000;
  Points pts(n);
  for (size_t i = 0; i < n; ++i) pts[i] = {i * 0.01f, 0, 0};
  auto range = [&](size_t a, size_t b) {
    return Points(pts.begin() + a, pts.begin() + b);
  };
  typename Tree::AABB box({700, -1, -1}, {800, 1, 1});
  Tree tree;
  tree.addPoints(pts);
  // Let the worker take the queued rest of that call and start replaying it,
  // so that the calls below are queued while it still holds part of it back.
  // In a different order, a delete would miss its point or an insert would
  // be deleted.
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  tree.deletePoints(range(0, n / 2));
  tree.addPoints(range(0, n / 4));
  tree.deletePoints(range(0, n / 8));
  tree.deleteBox(box);
  tree.addPoints(range(75000, 76000));
  Points model = range(n / 8, n / 4);
  for (size_t i = n / 2; i < n; ++i)
    if (!box.contains(pts[i])) model.push_back(pts[i]);
  Points refill = range(75000, 76000);
  model.insert(model.end(), refill.begin(), refill.end());
  auto matches = [&] {
    if (tree.size() != static_cast<int>(model.size())) return false;
    std::uniform_real_distribution<float> u(-10.0f, n * 0.01f + 10.0f);
    for (int i = 0; i < 50; ++i) {
      Pt q{u(rng), 0.2f, -0.1f};
      Points res;
      std::vector<float> d;
      tree.knnSearch(q, 8, res, d);
      if (!sameDistances(d, bruteKnn(model, q, 8))) return false;
    }
    return true;
  };
  tree.waitForRebuild();
  CHECK(tree.validate());
  CHECK(matches());
  // From an idle tree: the first chunk of this delete leaves its subtrees
  // mostly deleted, so the rest of it and the insert after it are queued
  tree.deletePoints(range(n / 2, n / 2 + 20000));
  tree.addPoints(range(n / 2, n / 2 + 5000));
  model.erase(std::remove_if(model.begin(), model.end(),
                             [&](const Pt& p) {
                               return p.x >= pts[n / 2].x &&
                                      p.x <= pts[n / 2 + 19999].x;
                             }),
              model.end());
  Points back = range(n / 2, n / 2 + 5000);
  model.insert(model.end(), back.begin(), back.end());
  tree.waitForRebuild();
  CHECK(tree.validate());
  CHECK(matches());
}

// Random writes, each followed by waiting for the rebuild and a check of the
// whole tree's structure.
template <typename Tree>
void testValidatedRandomWrites() {
  std::printf("[validated random writes]\n");
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> u(-50.0f, 50.0f);
  Tree tree;
  Points model = gridBlob(rng, 2000);
  tree.build(model);
  CHECK(tree.validate());
  for (int round = 0; round < 300; ++round) {
    int kind = rng() % 10;
    if (kind < 5) {
      Points b = gridBlob(rng, 50 + rng() % 400);
      tree.addPoints(b);
      model.insert(model.end(), b.begin(), b.end());
    } else if (kind < 8) {
      Points del;
      for (int i = 0; i < 100 && !model.empty(); ++i)
        del.push_back(model[rng() % model.size()]);
      tree.deletePoints(del);
      for (const auto& p : del) eraseOne(model, p);
    } else {
      float cx = u(rng), cy = u(rng), h = 2.0f + rng() % 15;
      typename Tree::AABB box({cx - h, cy - h, -100.0f}, {cx + h, cy + h, 100.0f});
      tree.deleteBox(box);
      model.erase(std::remove_if(model.begin(), model.end(),
                                 [&](const Pt& p) { return box.contains(p); }),
                  model.end());
    }
    tree.waitForRebuild();
    CHECK(tree.validate());
    if (round % 20 == 19) CHECK(sameAsModel(tree, model, rng));
  }
}

// Batches that arrive while the worker drains the pending buffer used to be
// stranded there once the rebuild flag flipped back.
template <typename Tree>
void testNoPointsLostDuringRebuild() {
  std::printf("[no points lost during rebuild]\n");
  std::mt19937 rng(2);
  for (int trial = 0; trial < 20; ++trial) {
    Tree tree;
    tree.build(blob(rng, 1000));
    size_t expected = 1000;
    for (int i = 0; i < 300; ++i) {
      Points b = blob(rng, 2000, 0.5f);
      tree.addPoints(b);
      expected += b.size();
    }
    tree.waitForRebuild();
    CHECK(tree.validate());
    CHECK(tree.size() == static_cast<int>(expected));
  }
}

#ifdef LIKD_TREE_STATS
// Skewed batches without waiting: the statistics see the rebuild rounds.
template <typename Tree>
void testRebuildStats() {
  std::printf("[rebuild statistics]\n");
  std::mt19937 rng(13);
  Tree tree;
  tree.build(blob(rng, 1000));
  for (int i = 0; i < 100; ++i) tree.addPoints(blob(rng, 2000, 0.5f));
  tree.waitForRebuild();
  typename Tree::RebuildStats s = tree.rebuildStats();
  CHECK(s.rounds > 0);
  CHECK(s.max_round_ms > 0);
  // Whether writes got queued depends on timing
  CHECK((s.max_queued_ops == 0) == (s.max_queued_ms == 0));
}
#endif

// Points sorted along x, or all identical, written in one call or in
// 2000-point frames without waiting. Each of them lands in the same leaf as
// the one before, so without rebalancing between chunks of a write and
// between batches of a replay, the tree grew into a chain and the time grew
// with the square of the count: about a minute for 400k points.
template <typename Tree>
void testOrderedWrites() {
  std::printf("[ordered writes]\n");
  std::mt19937 rng(19);
  const size_t n = 400000, frame = 2000;
  for (bool identical : {false, true}) {
    Points pts(n);
    for (size_t i = 0; i < n; ++i)
      pts[i] = identical ? Pt{1.5f, -2.0f, 3.0f} : Pt{i * 0.01f, 0, 0};
    for (bool frames : {false, true}) {
      Tree tree;
      auto start = std::chrono::steady_clock::now();
      if (frames) {
        for (size_t s = 0; s < n; s += frame)
          tree.addPoints(Points(pts.begin() + s, pts.begin() + std::min(n, s + frame)));
      } else {
        tree.addPoints(pts);
      }
      tree.waitForRebuild();
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - start)
                      .count();
      std::printf("  %s points, %s: %.0f ms\n", identical ? "identical" : "sorted",
                  frames ? "2000-point frames" : "one call", ms);
#ifndef LIKD_TEST_SANITIZED
      // Well under a second; the margin covers a busy machine
      CHECK(ms < 10000);
#endif
      CHECK(tree.validate());
      CHECK(tree.size() == static_cast<int>(n));
      std::uniform_real_distribution<float> u(-0.05f * n * 0.01f, 1.05f * n * 0.01f);
      for (int i = 0; i < 10; ++i) {
        Pt q = identical ? Pt{1.5f + i * 0.1f, -2.0f, 3.0f} : Pt{u(rng), 0.3f, -0.2f};
        Points res;
        std::vector<float> d;
        tree.knnSearch(q, 8, res, d);
        CHECK(sameDistances(d, bruteKnn(pts, q, 8)));
      }
    }
  }
}

template <typename Tree>
void testWaitForRebuild() {
  std::printf("[wait_for_rebuild]\n");
  std::mt19937 rng(3);
  Tree tree;
  tree.build(blob(rng, 50000));
  size_t expected = 50000;
  for (int i = 0; i < 30; ++i) {
    Points b = blob(rng, 5000, 0.5f);
    tree.addPoints(b, /*wait_for_rebuild=*/true);
    expected += b.size();
    CHECK(tree.size() == static_cast<int>(expected));
  }
}

// The returned point is a copy and stays valid after its node is rebuilt.
template <typename Tree>
void testNearestIsCopy() {
  std::printf("[single query returns a copy]\n");
  std::mt19937 rng(4);
  Tree tree;
  tree.build(blob(rng, 100));
  auto [nearest, d] = tree.nearestNeighbors(Pt{0, 0, 0});
  CHECK(nearest.has_value());
  Pt before = *nearest;
  tree.addPoints(blob(rng, 20000, 0.2f), true);  // rebuilds the root
  CHECK(nearest->x == before.x && nearest->y == before.y);
}

template <typename Tree>
void testConcurrentWriters() {
  std::printf("[concurrent writers]\n");
  std::mt19937 rng(5);
  Tree tree;
  Points all = blob(rng, 1000);
  tree.build(all);
  const int kThreads = 4, kBatches = 100;
  std::vector<std::vector<Points>> batches(kThreads);
  for (auto& per_thread : batches) {
    for (int i = 0; i < kBatches; ++i) {
      per_thread.push_back(blob(rng, 300, 1.0f));
      all.insert(all.end(), per_thread.back().begin(), per_thread.back().end());
    }
  }
  std::vector<std::thread> writers;
  for (auto& per_thread : batches) {
    writers.emplace_back([&tree, &per_thread] {
      for (const auto& b : per_thread) tree.addPoints(b);
    });
  }
  for (auto& w : writers) w.join();
  tree.waitForRebuild();
  CHECK(tree.validate());
  CHECK(tree.size() == static_cast<int>(all.size()));
  CHECK(matchesBruteForce(tree, all, rng, 200));
}

// Readers run while a writer inserts and occasionally calls build().
template <typename Tree>
void testReadersDuringWritesAndBuild() {
  std::printf("[readers during writes and build]\n");
  std::mt19937 rng(6);
  Tree tree;
  tree.build(blob(rng, 5000));
  std::atomic<bool> done{false};
  std::atomic<long> answered{0};
  std::vector<std::thread> readers;
  for (int r = 0; r < 3; ++r) {
    readers.emplace_back([&, r] {
      std::mt19937 local(100 + r);
      std::uniform_real_distribution<float> u(-60.0f, 60.0f);
      Points queries(64), res;
      std::vector<float> dists;
      while (!done) {
        for (auto& q : queries) q = {u(local), u(local), u(local)};
        tree.nearestNeighbors(queries, res, dists);
        answered += tree.nearestNeighbors(queries[0]).first.has_value();
        answered += tree.size() > 0;
      }
    });
  }
  Points last;
  std::uniform_real_distribution<float> u(-50.0f, 50.0f);
  for (int i = 0; i < 200; ++i) {
    if (i % 50 == 49) {
      last = blob(rng, 3000);
      tree.build(last);
    } else if (i % 7 == 6) {
      float cx = u(rng), cy = u(rng);
      typename Tree::AABB box({cx - 10, cy - 10, -100}, {cx + 10, cy + 10, 100});
      tree.deleteBox(box);
      last.erase(std::remove_if(last.begin(), last.end(),
                                [&](const Pt& p) { return box.contains(p); }),
                 last.end());
    } else {
      Points b = blob(rng, 1000, 1.0f);
      tree.addPoints(b);
      last.insert(last.end(), b.begin(), b.end());
    }
  }
  done = true;
  for (auto& r : readers) r.join();
  tree.waitForRebuild();
  CHECK(tree.validate());
  CHECK(answered > 0);
  CHECK(tree.size() == static_cast<int>(last.size()));
  CHECK(matchesBruteForce(tree, last, rng, 100));
}

// N is the tree's leaf size
template <typename Tree, int N>
void bruteForceTests(const char* label) {
  std::printf("== brute force, %s\n", label);
  testIncrementalQueries<Tree>();
  testKnnSearch<Tree>();
  testBoxSearch<Tree>();
  testDeleteDifferential<Tree>();
  testDeleteTies<Tree>();
  testDeleteAll<Tree>();
  testMemoryUsage<Tree>();
  testDuplicates<Tree>();
  testEmptiedLeaves<Tree>();
  testInsertAfterDeletingSubtrees<Tree>();
  testValidatedRandomWrites<Tree>();
  testReuseDeletedSlot<Tree, N>();
  testWriteOrderAcrossChunks<Tree>();
}

template <typename Tree>
void concurrencyTests(const char* label) {
  std::printf("== concurrency, %s\n", label);
  testNoPointsLostDuringRebuild<Tree>();
#ifdef LIKD_TREE_STATS
  testRebuildStats<Tree>();
#endif
  testWaitForRebuild<Tree>();
  testNearestIsCopy<Tree>();
  testConcurrentWriters<Tree>();
  testReadersDuringWritesAndBuild<Tree>();
  testOrderedWrites<Tree>();
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  bruteForceTests<LeafTree<2>, 2>("leaf size 2");
  bruteForceTests<LeafTree<4>, 4>("leaf size 4");
  bruteForceTests<KDTree<Pt>, DefaultOptions::LEAF_SIZE>("default leaf size (32)");
  bruteForceTests<LeafTree<64>, 64>("leaf size 64");
  concurrencyTests<LeafTree<2>>("leaf size 2");
  concurrencyTests<KDTree<Pt>>("default leaf size (32)");
  if (g_failures) {
    std::printf("%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
