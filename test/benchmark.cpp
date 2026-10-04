// benchmark.cpp - likd-tree vs ikd-tree
//
// Both trees answer the same queries with the same threading: sequential
// loops, or TBB parallel loops for both. Usage:
//   ./benchmark            100K uniform random points, 1000-point frames
//   ./benchmark map.pcd    stream a real map in file order, 2000-point frames,
//                          plus a local-map test with box deletion

// Enable TBB parallel execution (define before including likd_tree.hpp)
#define LIKD_TREE_USE_TBB

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../src/likd_tree.hpp"
#include "ikd_Tree.h"

using PointType = pcl::PointXYZ;
using LikdTree = KDTree<PointType>;
using IkdTree = KD_TREE<PointType>;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int K = 5;  // FAST-LIO matches each point against 5 neighbors

double elapsedMs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

std::vector<float> likdNN(const LikdTree& tree, const PointVector<PointType>& qs) {
  std::vector<float> d(qs.size());
  for (size_t i = 0; i < qs.size(); ++i) d[i] = tree.nearestNeighbors(qs[i]).second;
  return d;
}

// Distance to the k-th neighbor (or the farthest found) for each query
std::vector<float> likdKnn(const LikdTree& tree, const PointVector<PointType>& qs,
                           int k, bool parallel) {
  std::vector<float> kth(qs.size(), INFINITY);
  auto one = [&](size_t i) {
    PointVector<PointType> res;
    std::vector<float> d;
    tree.knnSearch(qs[i], k, res, d);
    if (!d.empty()) kth[i] = d.back();
  };
  if (parallel)
    tbb::parallel_for(size_t(0), qs.size(), one);
  else
    for (size_t i = 0; i < qs.size(); ++i) one(i);
  return kth;
}

std::vector<float> ikdKnn(IkdTree& tree, const PointVector<PointType>& qs, int k,
                          bool parallel) {
  std::vector<float> kth(qs.size(), INFINITY);
  auto one = [&](size_t i) {
    PointVector<PointType> res;
    std::vector<float> d;
    tree.Nearest_Search(qs[i], k, res, d);
    if (!d.empty()) kth[i] = std::sqrt(d.back());  // ikd returns squared
  };
  if (parallel)
    tbb::parallel_for(size_t(0), qs.size(), one);
  else
    for (size_t i = 0; i < qs.size(); ++i) one(i);
  return kth;
}

struct Totals {
  double build = 0, insert = 0, insert_max = 0, nn = 0, knn = 0, knn_par = 0,
         del = 0;
};

void printRow(const char* name, double likd, double ikd) {
  printf("  %-28s likd-tree %9.2f ms | ikd-tree %9.2f ms | %5.2fx\n", name, likd,
         ikd, ikd / likd);
}

// Feed `pts` frame by frame: query the frame against the map, then insert it.
// With local_map_half > 0, every 10 frames delete everything outside a cube
// around the frame centroid (like FAST-LIO's map segmentation).
void runStream(const PointVector<PointType>& pts, size_t frame,
               float local_map_half) {
  LikdTree likd;
  IkdTree* ikd = new IkdTree();  // ~48 MB operation queue: keep off the stack
  Totals L, I;
  long stale = 0, total = 0;

  PointVector<PointType> first(pts.begin(), pts.begin() + frame);
  auto t0 = Clock::now();
  likd.build(first);
  auto t1 = Clock::now();
  ikd->Build(first);
  auto t2 = Clock::now();
  L.build = elapsedMs(t0, t1);
  I.build = elapsedMs(t1, t2);

  size_t frames = 0;
  for (size_t s = frame; s < pts.size(); s += frame, ++frames) {
    PointVector<PointType> batch(pts.begin() + s,
                                 pts.begin() + std::min(s + frame, pts.size()));
    // Query phase (scan matching against the current map)
    auto q0 = Clock::now();
    likdNN(likd, batch);
    auto q1 = Clock::now();
    ikdKnn(*ikd, batch, 1, false);
    auto q2 = Clock::now();
    std::vector<float> lk = likdKnn(likd, batch, K, false);
    auto q3 = Clock::now();
    std::vector<float> ik = ikdKnn(*ikd, batch, K, false);
    auto q4 = Clock::now();
    likdKnn(likd, batch, K, true);
    auto q5 = Clock::now();
    ikdKnn(*ikd, batch, K, true);
    auto q6 = Clock::now();
    L.nn += elapsedMs(q0, q1);
    I.nn += elapsedMs(q1, q2);
    L.knn += elapsedMs(q2, q3);
    I.knn += elapsedMs(q3, q4);
    L.knn_par += elapsedMs(q4, q5);
    I.knn_par += elapsedMs(q5, q6);
    // ikd-tree applies every write immediately: a different k-th distance
    // means likd-tree answered from a map without its queued writes
    for (size_t i = 0; i < batch.size(); ++i, ++total)
      stale += std::fabs(lk[i] - ik[i]) > 1e-3f;

    // Insert phase
    auto a0 = Clock::now();
    likd.addPoints(batch);
    auto a1 = Clock::now();
    ikd->Add_Points(batch, false);
    auto a2 = Clock::now();
    L.insert += elapsedMs(a0, a1);
    I.insert += elapsedMs(a1, a2);
    L.insert_max = std::max(L.insert_max, elapsedMs(a0, a1));
    I.insert_max = std::max(I.insert_max, elapsedMs(a1, a2));

    if (local_map_half > 0 && frames % 10 == 9) {
      float c[3] = {0, 0, 0};
      for (const auto& p : batch) {
        c[0] += p.x / batch.size();
        c[1] += p.y / batch.size();
        c[2] += p.z / batch.size();
      }
      std::vector<LikdTree::AABB> boxes;
      std::vector<BoxPointType> ikd_boxes;
      for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
          LikdTree::AABB box({-1e6f, -1e6f, -1e6f}, {1e6f, 1e6f, 1e6f});
          if (side == 0)
            box.max[axis] = c[axis] - local_map_half;
          else
            box.min[axis] = c[axis] + local_map_half;
          boxes.push_back(box);
          BoxPointType b;
          for (int i = 0; i < 3; ++i) {
            b.vertex_min[i] = box.min[i];
            b.vertex_max[i] = box.max[i];
          }
          ikd_boxes.push_back(b);
        }
      }
      auto d0 = Clock::now();
      likd.deleteBoxes(boxes);
      auto d1 = Clock::now();
      ikd->Delete_Point_Boxes(ikd_boxes);
      auto d2 = Clock::now();
      L.del += elapsedMs(d0, d1);
      I.del += elapsedMs(d1, d2);
    }
  }
  auto w0 = Clock::now();
  likd.waitForRebuild();
  double wait_ms = elapsedMs(w0, Clock::now());

  printf("  frames: %zu x %zu points\n", frames, frame);
  printRow("Build (first frame)", L.build, I.build);
  printRow("Insert total", L.insert, I.insert);
  printRow("Insert worst frame", L.insert_max, I.insert_max);
  printRow("1-NN query total (seq)", L.nn, I.nn);
  printRow("5-NN query total (seq)", L.knn, I.knn);
  printRow("5-NN query total (TBB)", L.knn_par, I.knn_par);
  if (local_map_half > 0) {
    printRow("Box delete total", L.del, I.del);
    printf("  %-28s likd-tree %9d    | ikd-tree %9d\n", "Points kept", likd.size(),
           ikd->validnum());
    printf("  %-28s likd-tree %9d    | ikd-tree %9d\n", "Nodes held",
           likd.nodeCount(), ikd->size());
  }
  printf("  likd-tree final wait for rebuild: %.2f ms\n", wait_ms);
  printf("  5-NN answers from a map with queued writes (likd-tree): %.3f%%\n",
         100.0 * stale / total);
  delete ikd;
}

}  // namespace

int main(int argc, char** argv) {
  PointVector<PointType> pts;
  size_t frame;
  if (argc > 1) {
    pcl::PointCloud<PointType> cloud;
    if (pcl::io::loadPCDFile<PointType>(argv[1], cloud) < 0) {
      std::cerr << "Failed to read " << argv[1] << std::endl;
      return 1;
    }
    for (const auto& p : cloud.points)
      if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
        pts.push_back(p);
    frame = 2000;
    std::cout << "Map " << argv[1] << ": " << pts.size() << " points\n";
  } else {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    pts.resize(100000);
    for (auto& p : pts) {
      p.x = dist(rng);
      p.y = dist(rng);
      p.z = dist(rng);
    }
    frame = 1000;
    std::cout << "100K uniform random points in [-100, 100]^3\n";
  }

  // ============================================================
  // Part 1: Batch Build Test (Build all points at once)
  // ============================================================
  std::cout << "\n=== Part 1: Batch build + 1000 queries ===" << std::endl;
  {
    LikdTree likd;
    IkdTree* ikd = new IkdTree();
    auto t0 = Clock::now();
    likd.build(pts);
    auto t1 = Clock::now();
    ikd->Build(pts);
    auto t2 = Clock::now();
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    PointVector<PointType> queries(1000);
    for (size_t i = 0; i < queries.size(); ++i) {
      queries[i] = pts[(i * 7919) % pts.size()];
      queries[i].x += noise(rng);
      queries[i].y += noise(rng);
      queries[i].z += noise(rng);
    }
    auto q0 = Clock::now();
    likdNN(likd, queries);
    auto q1 = Clock::now();
    ikdKnn(*ikd, queries, 1, false);
    auto q2 = Clock::now();
    likdKnn(likd, queries, K, false);
    auto q3 = Clock::now();
    ikdKnn(*ikd, queries, K, false);
    auto q4 = Clock::now();
    printRow("Build", elapsedMs(t0, t1), elapsedMs(t1, t2));
    printRow("1-NN x1000 (seq)", elapsedMs(q0, q1), elapsedMs(q1, q2));
    printRow("5-NN x1000 (seq)", elapsedMs(q2, q3), elapsedMs(q3, q4));
    delete ikd;
  }

  // ============================================================
  // Part 2: Incremental Insertion Test
  // ============================================================
  std::cout << "\n=== Part 2: Incremental insertion (query, then insert) ==="
            << std::endl;
  runStream(pts, frame, 0.0f);

  if (argc > 1) {
    std::cout << "\n=== Part 3: Local map, 100 m cube, box delete every 10 "
                 "frames ===" << std::endl;
    runStream(pts, frame, 50.0f);
  }
  return 0;
}
