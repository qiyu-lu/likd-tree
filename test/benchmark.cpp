// benchmark.cpp - likd-tree vs ikd-tree
//
// Both trees answer the same queries with the same threading: sequential
// loops, or TBB parallel loops for both. Usage:
//   ./benchmark            100K uniform random points, 1000-point frames
//   ./benchmark map.pcd    stream a real map in file order, 2000-point frames,
//                          plus a local-map test with box deletion
//
// Memory is the heap growth (glibc mallinfo, all arenas) while one tree is
// built or fed alone, divided by the points it holds. RSS growth is printed
// next to it because the heap figure leaves out fragmentation.

// Enable TBB parallel execution and rebuild statistics (define before
// including likd_tree.hpp)
#define LIKD_TREE_USE_TBB
#define LIKD_TREE_STATS

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <tbb/parallel_for.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../src/likd_tree.hpp"
#include "ikd_Tree.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

using PointType = pcl::PointXYZ;
using LikdTree = KDTree<PointType>;
using IkdTree = KD_TREE<PointType>;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int K = 5;  // FAST-LIO matches each point against 5 neighbors

double elapsedMs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Heap bytes in use across all malloc arenas, or -1 off glibc. The mallinfo()
// fields are int and overflow past 2 GB (4 GB when read as unsigned); glibc
// 2.31 has no mallinfo2(), and this benchmark stays below ~300 MB.
long heapBytes() {
#ifdef __GLIBC__
  struct mallinfo m = mallinfo();
  return static_cast<long>(static_cast<unsigned>(m.uordblks)) +
         static_cast<long>(static_cast<unsigned>(m.hblkhd));
#else
  return -1;
#endif
}

// Resident set size in bytes, or -1 without /proc
long rssBytes() {
  std::ifstream statm("/proc/self/statm");
  long size = 0, resident = 0;
  if (!(statm >> size >> resident)) return -1;
  return resident * sysconf(_SC_PAGESIZE);
}

// Hands freed heap pages back to the OS, so that RSS growth measured from
// here on comes from new allocations rather than reused pages.
void releaseFreeMemory() {
#ifdef __GLIBC__
  malloc_trim(0);
#endif
}

struct MemorySample {
  long heap, rss;
  static MemorySample now() { return {heapBytes(), rssBytes()}; }
};

// Bytes per point grown between two samples, NAN where unavailable
double perPoint(long before, long after, long points) {
  if (before < 0 || after < 0 || points <= 0) return NAN;
  return double(after - before) / points;
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

void printRow(const char* name, double likd, double ikd,
              const char* unit = "ms") {
  printf("  %-28s likd-tree %9.2f %-4s | ikd-tree %9.2f %-4s | %5.2fx\n", name,
         likd, unit, ikd, unit, ikd / likd);
}

void printLikd(const char* name, double likd, const char* unit) {
  printf("  %-28s likd-tree %9.2f %s\n", name, likd, unit);
}

void printLikdCount(const char* name, size_t likd, const char* unit) {
  printf("  %-28s likd-tree %9zu %s\n", name, likd, unit);
}

// Total number of points found by `search` over all queries
template <typename Search>
size_t countResults(const PointVector<PointType>& qs, Search search) {
  size_t found = 0;
  PointVector<PointType> res;
  for (const auto& q : qs) {
    res.clear();
    search(q, res);
    found += res.size();
  }
  return found;
}

// Boxes that delete everything outside a cube of half size `half` around the
// batch centroid (like FAST-LIO's map segmentation)
std::vector<LikdTree::AABB> localMapBoxes(const PointVector<PointType>& batch,
                                          float half) {
  float c[3] = {0, 0, 0};
  for (const auto& p : batch) {
    c[0] += p.x / batch.size();
    c[1] += p.y / batch.size();
    c[2] += p.z / batch.size();
  }
  std::vector<LikdTree::AABB> boxes;
  for (int axis = 0; axis < 3; ++axis) {
    for (int side = 0; side < 2; ++side) {
      LikdTree::AABB box({-1e6f, -1e6f, -1e6f}, {1e6f, 1e6f, 1e6f});
      if (side == 0)
        box.max[axis] = c[axis] - half;
      else
        box.min[axis] = c[axis] + half;
      boxes.push_back(box);
    }
  }
  return boxes;
}

BoxPointType toIkdBox(const LikdTree::AABB& box) {
  BoxPointType b;
  for (int i = 0; i < 3; ++i) {
    b.vertex_min[i] = box.min[i];
    b.vertex_max[i] = box.max[i];
  }
  return b;
}

std::vector<BoxPointType> toIkdBoxes(const std::vector<LikdTree::AABB>& boxes) {
  std::vector<BoxPointType> out;
  for (const auto& box : boxes) out.push_back(toIkdBox(box));
  return out;
}

// Feed `pts` frame by frame: query the frame against the map, then insert it.
// With local_map_half > 0, every 10 frames delete everything outside a cube
// around the frame centroid.
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
      std::vector<LikdTree::AABB> boxes = localMapBoxes(batch, local_map_half);
      std::vector<BoxPointType> ikd_boxes = toIkdBoxes(boxes);
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
  LikdTree::RebuildStats rs = likd.rebuildStats();
  printLikdCount("Rebuild rounds", rs.rounds, "");
  printLikd("Longest rebuild round", rs.max_round_ms, "ms");
  printLikdCount("Most writes queued", rs.max_queued_ops, "ops");
  printLikd("Longest queued-write wait", rs.max_queued_ms, "ms");
  delete ikd;
}

// The same frames and box deletions as runStream, fed into one tree alone and
// without queries.
template <typename Build, typename Add, typename DeleteBoxes>
void feed(const PointVector<PointType>& pts, size_t frame, float local_map_half,
          Build build, Add add, DeleteBoxes delete_boxes) {
  build(PointVector<PointType>(pts.begin(), pts.begin() + frame));
  size_t frames = 0;
  for (size_t s = frame; s < pts.size(); s += frame, ++frames) {
    PointVector<PointType> batch(pts.begin() + s,
                                 pts.begin() + std::min(s + frame, pts.size()));
    add(batch);
    if (local_map_half > 0 && frames % 10 == 9)
      delete_boxes(localMapBoxes(batch, local_map_half));
  }
}

// runStream interleaves the two trees, so their heap growth can't be told
// apart there: feed each tree alone and divide by the points it still holds.
void streamMemory(const PointVector<PointType>& pts, size_t frame,
                  float local_map_half) {
  double likd_heap, likd_rss, likd_usage, ikd_heap, ikd_rss;
  {
    releaseFreeMemory();
    MemorySample before = MemorySample::now();
    LikdTree likd;
    feed(pts, frame, local_map_half,
         [&](const PointVector<PointType>& f) { likd.build(f); },
         [&](const PointVector<PointType>& b) { likd.addPoints(b); },
         [&](const std::vector<LikdTree::AABB>& boxes) { likd.deleteBoxes(boxes); });
    likd.waitForRebuild();
    MemorySample after = MemorySample::now();
    int valid = likd.size();
    likd_heap = perPoint(before.heap, after.heap, valid);
    likd_rss = perPoint(before.rss, after.rss, valid);
    likd_usage = double(likd.memoryUsage()) / valid;
  }
  {
    IkdTree* ikd = new IkdTree();  // operation queue allocated here, not counted
    releaseFreeMemory();
    MemorySample before = MemorySample::now();
    feed(pts, frame, local_map_half,
         [&](const PointVector<PointType>& f) { ikd->Build(f); },
         [&](PointVector<PointType>& b) { ikd->Add_Points(b, false); },
         [&](const std::vector<LikdTree::AABB>& boxes) {
           std::vector<BoxPointType> b = toIkdBoxes(boxes);
           ikd->Delete_Point_Boxes(b);
         });
    // ikd-Tree has no call that waits for its rebuild thread
    std::this_thread::sleep_for(std::chrono::seconds(1));
    MemorySample after = MemorySample::now();
    int valid = ikd->validnum();
    ikd_heap = perPoint(before.heap, after.heap, valid);
    ikd_rss = perPoint(before.rss, after.rss, valid);
    delete ikd;
  }
  printRow("Heap per valid point", likd_heap, ikd_heap, "B/pt");
  printRow("RSS growth per valid point", likd_rss, ikd_rss, "B/pt");
  printLikd("memoryUsage() per valid pt", likd_usage, "B/pt");
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
  // Start TBB's worker threads before any memory is measured
  tbb::parallel_for(0, 1 << 20, [](int) {});

  // ============================================================
  // Part 1: Batch Build Test (Build all points at once)
  // ============================================================
  std::cout << "\n=== Part 1: Batch build + 200k queries ===" << std::endl;
  {
    const long n = static_cast<long>(pts.size());
    // Radius and box half size giving tens to a hundred points per query
    const float radius = argc > 1 ? 1.0f : 10.0f;
    const float box_half = radius;
    LikdTree likd;
    releaseFreeMemory();
    MemorySample m0 = MemorySample::now();
    auto t0 = Clock::now();
    likd.build(pts);
    auto t1 = Clock::now();
    MemorySample m1 = MemorySample::now();
    IkdTree* ikd = new IkdTree();  // operation queue allocated here, not counted
    releaseFreeMemory();
    MemorySample m2 = MemorySample::now();
    auto t2 = Clock::now();
    ikd->Build(pts);
    auto t3 = Clock::now();
    MemorySample m3 = MemorySample::now();
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 0.05f);
    PointVector<PointType> queries(200000);
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

    // Radius and box search on the first 20k queries
    PointVector<PointType> range_queries(queries.begin(), queries.begin() + 20000);
    std::vector<float> dists;
    auto r0 = Clock::now();
    size_t likd_in_radius = countResults(
        range_queries, [&](const PointType& q, PointVector<PointType>& res) {
          likd.radiusSearch(q, radius, res, dists);
        });
    auto r1 = Clock::now();
    size_t ikd_in_radius = countResults(
        range_queries, [&](const PointType& q, PointVector<PointType>& res) {
          ikd->Radius_Search(q, radius, res);
        });
    auto r2 = Clock::now();
    auto boxAround = [&](const PointType& q) {
      return LikdTree::AABB({q.x - box_half, q.y - box_half, q.z - box_half},
                            {q.x + box_half, q.y + box_half, q.z + box_half});
    };
    size_t likd_in_box = countResults(
        range_queries, [&](const PointType& q, PointVector<PointType>& res) {
          likd.boxSearch(boxAround(q), res);
        });
    auto r3 = Clock::now();
    size_t ikd_in_box = countResults(
        range_queries, [&](const PointType& q, PointVector<PointType>& res) {
          ikd->Box_Search(toIkdBox(boxAround(q)), res);
        });
    auto r4 = Clock::now();

    printRow("Build", elapsedMs(t0, t1), elapsedMs(t2, t3));
    printRow("1-NN x200k (seq)", elapsedMs(q0, q1), elapsedMs(q1, q2));
    printRow("5-NN x200k (seq)", elapsedMs(q2, q3), elapsedMs(q3, q4));
    printRow("Radius search x20k (seq)", elapsedMs(r0, r1), elapsedMs(r1, r2));
    printRow("Box search x20k (seq)", elapsedMs(r2, r3), elapsedMs(r3, r4));
    printf("  radius %.1f m: %.1f points per query, box half size %.1f m: %.1f\n",
           radius, double(likd_in_radius) / range_queries.size(), box_half,
           double(likd_in_box) / range_queries.size());
    if (likd_in_radius != ikd_in_radius)
      printf("  WARNING: radius search found %zu points, ikd-tree %zu\n",
             likd_in_radius, ikd_in_radius);
    // ikd-tree's boxes are half-open, [min, max): only likd-tree finds points
    // lying exactly on a max face
    if (likd_in_box < ikd_in_box)
      printf("  WARNING: box search found %zu points, ikd-tree %zu\n",
             likd_in_box, ikd_in_box);
    printRow("Heap per point", perPoint(m0.heap, m1.heap, n),
             perPoint(m2.heap, m3.heap, n), "B/pt");
    printRow("RSS growth per point", perPoint(m0.rss, m1.rss, n),
             perPoint(m2.rss, m3.rss, n), "B/pt");
    printLikd("memoryUsage() per point", double(likd.memoryUsage()) / n, "B/pt");
    delete ikd;
  }

  // ============================================================
  // Part 2: Incremental Insertion Test
  // ============================================================
  std::cout << "\n=== Part 2: Incremental insertion (query, then insert) ==="
            << std::endl;
  runStream(pts, frame, 0.0f);
  streamMemory(pts, frame, 0.0f);

  if (argc > 1) {
    std::cout << "\n=== Part 3: Local map, 100 m cube, box delete every 10 "
                 "frames ===" << std::endl;
    runStream(pts, frame, 50.0f);
    streamMemory(pts, frame, 50.0f);
  }
  return 0;
}
