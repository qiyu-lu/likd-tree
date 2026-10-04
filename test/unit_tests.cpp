// unit_tests.cpp - correctness and thread-safety tests for likd-tree.
// Every query is checked against brute force. Returns non-zero on failure.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

#include "../src/likd_tree.hpp"

namespace {

struct Pt {
  float x, y, z;
};
using Tree = KDTree<Pt>;
using Points = PointVector<Pt>;

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

float bruteNearest(const Points& pts, const Pt& q) {
  float best = INFINITY;
  for (const auto& p : pts) best = std::min(best, dist(p, q));
  return best;
}

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
  std::sort(d.begin(), d.end());
  d.resize(std::min(d.size(), k));
  return d;
}

bool sameDistances(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::fabs(a[i] - b[i]) > 1e-4f) return false;
  return true;
}

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
    Tree::AABB box({cx - h, cy - h, cz - h}, {cx + h, cy + h, cz + h});
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
  tree.boxSearch(Tree::AABB({-1e9f, -1e9f, -1e9f}, {1e9f, 1e9f, 1e9f}), res);
  CHECK(res.size() == all.size());
  tree.boxSearch(Tree::AABB({1e8f, 1e8f, 1e8f}, {1e9f, 1e9f, 1e9f}), res);
  CHECK(res.empty());
}

// Batches that arrive while the worker drains the pending buffer used to be
// stranded there once the rebuild flag flipped back.
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
    CHECK(tree.size() == static_cast<int>(expected));
  }
}

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
  CHECK(tree.size() == static_cast<int>(all.size()));
  CHECK(matchesBruteForce(tree, all, rng, 200));
}

// Readers run while a writer inserts and occasionally calls build().
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
  for (int i = 0; i < 200; ++i) {
    if (i % 50 == 49) {
      last = blob(rng, 3000);
      tree.build(last);
    } else {
      Points b = blob(rng, 1000, 1.0f);
      tree.addPoints(b);
      last.insert(last.end(), b.begin(), b.end());
    }
  }
  done = true;
  for (auto& r : readers) r.join();
  tree.waitForRebuild();
  CHECK(answered > 0);
  CHECK(tree.size() == static_cast<int>(last.size()));
  CHECK(matchesBruteForce(tree, last, rng, 100));
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  testIncrementalQueries();
  testKnnSearch();
  testBoxSearch();
  testNoPointsLostDuringRebuild();
  testWaitForRebuild();
  testNearestIsCopy();
  testConcurrentWriters();
  testReadersDuringWritesAndBuild();
  if (g_failures) {
    std::printf("%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all tests passed\n");
  return 0;
}
