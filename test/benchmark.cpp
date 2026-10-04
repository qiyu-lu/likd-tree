// benchmark.cpp - likd-tree vs ikd-tree
//
// Both trees answer the same queries with the same threading: sequential
// loops, or TBB parallel loops for both. Usage:
//   ./benchmark            100K uniform random points, 1000-point frames
//   ./benchmark map.pcd    stream a real map in file order, 2000-point frames,
//                          plus a local-map test with box deletion
// Options for a map: --max-points N keeps only its first N points, and
// --frame N sets the points per frame.
//
// Memory is the heap growth (glibc mallinfo, all arenas) while one tree is
// built or fed alone, divided by the points it holds. RSS growth is printed
// next to it because the heap figure leaves out fragmentation.
//
// Compile-time switches for sweeps:
//   -DLIKD_BENCH_LEAF_SIZE=N   likd-tree leaf size (default: the library's)
//   -DLIKD_BENCH_XYZINORMAL    pcl::PointXYZINormal (48 bytes) instead of
//                              pcl::PointXYZ
//   -DLIKD_BENCH_STAMPS        likd-tree keeps stamps (TRACK_STAMPS); frames
//                              are stamped with their number. Adds Part 4
//                              (Part 2 with touch) and, for a map, Part 5
//                              (expiry).

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
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "../src/likd_tree.hpp"
#include "ikd_Tree.h"

#ifdef __GLIBC__
#include <malloc.h>
#endif

// Optional reference: static nanoflann trees over the same points
#ifdef LIKD_BENCH_NANOFLANN
#include <nanoflann.hpp>
#if NANOFLANN_VERSION < 0x150
#error "The nanoflann reference needs nanoflann 1.5.0 or newer"
#endif
#endif

#ifdef LIKD_BENCH_XYZINORMAL
using PointType = pcl::PointXYZINormal;
#else
using PointType = pcl::PointXYZ;
#endif
#if defined(LIKD_BENCH_LEAF_SIZE) || defined(LIKD_BENCH_STAMPS)
struct BenchOptions : DefaultOptions {
#ifdef LIKD_BENCH_LEAF_SIZE
  static constexpr int LEAF_SIZE = LIKD_BENCH_LEAF_SIZE;
#endif
#ifdef LIKD_BENCH_STAMPS
  static constexpr bool TRACK_STAMPS = true;
#endif
};
using LikdTree = KDTree<PointType, PointTraits<PointType>, BenchOptions>;
#else
using LikdTree = KDTree<PointType>;
#endif
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

// With stamps, the first frame (or the whole map) is stamped 0 and frame f
// of a stream is stamped f
void likdBuild(LikdTree& tree, const PointVector<PointType>& pts) {
#ifdef LIKD_BENCH_STAMPS
  tree.build(pts, LikdTree::Stamp{0});
#else
  tree.build(pts);
#endif
}

void likdAdd(LikdTree& tree, const PointVector<PointType>& pts,
             [[maybe_unused]] uint32_t frame) {
#ifdef LIKD_BENCH_STAMPS
  tree.addPoints(pts, LikdTree::Stamp{frame});
#else
  tree.addPoints(pts);
#endif
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

#ifdef LIKD_BENCH_STAMPS
// likdKnn, sequential, also appending every neighbor found to `found`
std::vector<float> likdKnnFound(const LikdTree& tree, const PointVector<PointType>& qs,
                                int k, PointVector<PointType>& found) {
  std::vector<float> kth(qs.size(), INFINITY);
  PointVector<PointType> res;
  std::vector<float> d;
  for (size_t i = 0; i < qs.size(); ++i) {
    tree.knnSearch(qs[i], k, res, d);
    if (!d.empty()) kth[i] = d.back();
    found.insert(found.end(), res.begin(), res.end());
  }
  return kth;
}

double sumOf(const std::vector<double>& v) {
  return std::accumulate(v.begin(), v.end(), 0.0);
}

double median(std::vector<double> v) {
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// Distinct points in pts (reorders it)
size_t countDistinct(PointVector<PointType>& pts) {
  std::sort(pts.begin(), pts.end(), [](const PointType& a, const PointType& b) {
    return std::tie(a.x, a.y, a.z) < std::tie(b.x, b.y, b.z);
  });
  return std::unique(pts.begin(), pts.end(),
                     [](const PointType& a, const PointType& b) {
                       return a.x == b.x && a.y == b.y && a.z == b.z;
                     }) -
         pts.begin();
}
#endif

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
  std::vector<double> frame_inserts;  // insert time of each frame
};

// Mean of the slowest 1% of frame times (at least one frame): less noisy
// than the single slowest frame
double slowestPercentMean(std::vector<double> times) {
  if (times.empty()) return NAN;
  size_t n = std::max<size_t>(1, (times.size() + 99) / 100);
  std::partial_sort(times.begin(), times.begin() + n, times.end(),
                    std::greater<double>());
  return std::accumulate(times.begin(), times.begin() + n, 0.0) / n;
}

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
// around the frame centroid. With touch (stamps only), likd-tree also
// touches, after each frame's queries, every neighbor its sequential 5-NN
// queries returned, and only the touch rows and rebuild statistics are
// printed.
void runStream(const PointVector<PointType>& pts, size_t frame,
               float local_map_half, [[maybe_unused]] bool touch = false) {
  LikdTree likd;
  IkdTree* ikd = new IkdTree();  // ~48 MB operation queue: keep off the stack
  Totals L, I;
  long stale = 0, total = 0;
  [[maybe_unused]] std::vector<double> touch_ms;
  [[maybe_unused]] size_t neighbors = 0, distinct = 0;

  PointVector<PointType> first(pts.begin(), pts.begin() + frame);
  auto t0 = Clock::now();
  likdBuild(likd, first);
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
#ifdef LIKD_BENCH_STAMPS
    PointVector<PointType> found;
    std::vector<float> lk = touch ? likdKnnFound(likd, batch, K, found)
                                  : likdKnn(likd, batch, K, false);
#else
    std::vector<float> lk = likdKnn(likd, batch, K, false);
#endif
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
#ifdef LIKD_BENCH_STAMPS
    if (touch) {
      auto t0 = Clock::now();
      likd.touchPoints(found, LikdTree::Stamp{static_cast<uint32_t>(frames + 1)});
      touch_ms.push_back(elapsedMs(t0, Clock::now()));
      neighbors += found.size();
      distinct += countDistinct(found);
    }
#endif

    // Insert phase
    auto a0 = Clock::now();
    likdAdd(likd, batch, frames + 1);
    auto a1 = Clock::now();
    ikd->Add_Points(batch, false);
    auto a2 = Clock::now();
    L.insert += elapsedMs(a0, a1);
    I.insert += elapsedMs(a1, a2);
    L.insert_max = std::max(L.insert_max, elapsedMs(a0, a1));
    I.insert_max = std::max(I.insert_max, elapsedMs(a1, a2));
    L.frame_inserts.push_back(elapsedMs(a0, a1));
    I.frame_inserts.push_back(elapsedMs(a1, a2));

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
  LikdTree::RebuildStats rs = likd.rebuildStats();

#ifdef LIKD_BENCH_STAMPS
  if (touch) {
    std::vector<double> ratio;
    for (size_t i = 0; i < touch_ms.size(); ++i)
      ratio.push_back(touch_ms[i] / L.frame_inserts[i]);
    printf("  frames: %zu x %zu points, %.0f neighbors per frame, %.1f%% distinct\n",
           frames, frame, double(neighbors) / frames, 100.0 * distinct / neighbors);
    printLikd("Touch total", sumOf(touch_ms), "ms");
    printLikd("Touch slowest 1% (mean)", slowestPercentMean(touch_ms), "ms");
    printLikd("Touch worst frame", *std::max_element(touch_ms.begin(), touch_ms.end()),
              "ms");
    printLikd("Insert total", L.insert, "ms");
    printLikd("Touch / insert, total", sumOf(touch_ms) / L.insert, "x");
    printLikd("Touch / insert, median frame", median(ratio), "x");
    printLikdCount("Rebuild rounds", rs.rounds, "");
    printLikdCount("Most writes queued", rs.max_queued_ops, "ops");
    printLikd("Longest queued-write wait", rs.max_queued_ms, "ms");
    delete ikd;
    return;
  }
#endif

  printf("  frames: %zu x %zu points\n", frames, frame);
  printRow("Build (first frame)", L.build, I.build);
  printRow("Insert total", L.insert, I.insert);
  printRow("Insert worst frame", L.insert_max, I.insert_max);
  printRow("Insert slowest 1% (mean)", slowestPercentMean(L.frame_inserts),
           slowestPercentMean(I.frame_inserts));
  printRow("1-NN query total (seq)", L.nn, I.nn);
  printRow("5-NN query total (seq)", L.knn, I.knn);
  printRow("5-NN query total (TBB)", L.knn_par, I.knn_par);
  if (local_map_half > 0) {
    printRow("Box delete total", L.del, I.del);
    printf("  %-28s likd-tree %9d    | ikd-tree %9d\n", "Points kept", likd.size(),
           ikd->validnum());
    printf("  %-28s likd-tree %9d    | ikd-tree %9d\n",
           "Points stored, incl. deleted", likd.nodeCount(), ikd->size());
  }
  printf("  likd-tree final wait for rebuild: %.2f ms\n", wait_ms);
  printf("  5-NN answers from a map with queued writes (likd-tree): %.3f%%\n",
         100.0 * stale / total);
  printLikdCount("Rebuild rounds", rs.rounds, "");
  printLikd("Longest rebuild round", rs.max_round_ms, "ms");
  printLikdCount("Most writes queued", rs.max_queued_ops, "ops");
  printLikd("Longest queued-write wait", rs.max_queued_ms, "ms");
  delete ikd;
}

// The same frames and box deletions as runStream, fed into one tree alone and
// without queries. add gets each frame and its number; end_frame runs after
// each frame.
template <typename Build, typename Add, typename DeleteBoxes, typename EndFrame>
void feed(const PointVector<PointType>& pts, size_t frame, float local_map_half,
          Build build, Add add, DeleteBoxes delete_boxes, EndFrame end_frame) {
  build(PointVector<PointType>(pts.begin(), pts.begin() + frame));
  size_t frames = 0;
  for (size_t s = frame; s < pts.size(); s += frame, ++frames) {
    PointVector<PointType> batch(pts.begin() + s,
                                 pts.begin() + std::min(s + frame, pts.size()));
    add(batch, frames + 1);
    if (local_map_half > 0 && frames % 10 == 9)
      delete_boxes(localMapBoxes(batch, local_map_half));
    end_frame();
  }
}

// runStream interleaves the two trees, so their heap growth can't be told
// apart there: feed each tree alone and divide by the points it still holds.
// likd-tree waits for its rebuild after every frame. Frames fed back to back
// would otherwise pile up behind a running rebuild, and the RSS would measure
// that queue instead of the tree.
void streamMemory(const PointVector<PointType>& pts, size_t frame,
                  float local_map_half) {
  double likd_heap, likd_rss, likd_usage, ikd_heap, ikd_rss;
  {
    releaseFreeMemory();
    MemorySample before = MemorySample::now();
    LikdTree likd;
    feed(pts, frame, local_map_half,
         [&](const PointVector<PointType>& f) { likdBuild(likd, f); },
         [&](const PointVector<PointType>& b, uint32_t frame) {
           likdAdd(likd, b, frame);
         },
         [&](const std::vector<LikdTree::AABB>& boxes) { likd.deleteBoxes(boxes); },
         [&] { likd.waitForRebuild(); });
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
         [&](PointVector<PointType>& b, uint32_t) { ikd->Add_Points(b, false); },
         [&](const std::vector<LikdTree::AABB>& boxes) {
           std::vector<BoxPointType> b = toIkdBoxes(boxes);
           ikd->Delete_Point_Boxes(b);
         },
         [] {});
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

#ifdef LIKD_BENCH_STAMPS
// Frames kept by Part 5's time-to-live: about as many points as Part 3's
// 100 m cube keeps on the sparse map
constexpr uint32_t kTtlFrames = 50;

// Part 3's frames, likd-tree alone, with a time-to-live instead of a local
// map box: every 10 frames, expire what was stamped more than kTtlFrames
// frames ago. 5-NN queries run without and with a min_stamp of half that age.
// An expiry is one write: queued behind a running rebuild it would cost the
// caller nothing, so each waits for the rebuild first (untimed) and the time
// is that of the expiry itself.
void ttlStream(const PointVector<PointType>& pts, size_t frame) {
  LikdTree likd;
  likdBuild(likd, PointVector<PointType>(pts.begin(), pts.begin() + frame));
  LikdTree::SearchOptions recent;
  double expire = 0, knn = 0, knn_recent = 0;
  long differ = 0, total_answers = 0;
  uint32_t f = 0;
  for (size_t s = frame; s < pts.size(); s += frame) {
    ++f;
    PointVector<PointType> batch(pts.begin() + s,
                                 pts.begin() + std::min(s + frame, pts.size()));
    recent.min_stamp = LikdTree::Stamp{f > kTtlFrames / 2 ? f - kTtlFrames / 2 : 0};
    std::vector<float> all(batch.size()), only_recent(batch.size());
    PointVector<PointType> res;
    std::vector<float> d;
    auto q0 = Clock::now();
    for (size_t i = 0; i < batch.size(); ++i) {
      likd.knnSearch(batch[i], K, res, d);
      all[i] = d.empty() ? INFINITY : d.back();
    }
    auto q1 = Clock::now();
    for (size_t i = 0; i < batch.size(); ++i) {
      likd.knnSearch(batch[i], K, res, d, recent);
      only_recent[i] = d.empty() ? INFINITY : d.back();
    }
    auto q2 = Clock::now();
    knn += elapsedMs(q0, q1);
    knn_recent += elapsedMs(q1, q2);
    for (size_t i = 0; i < batch.size(); ++i, ++total_answers)
      differ += !(std::fabs(all[i] - only_recent[i]) <= 1e-3f);
    likdAdd(likd, batch, f);
    if (f % 10 == 0 && f > kTtlFrames) {
      likd.waitForRebuild();
      auto e0 = Clock::now();
      likd.expireBefore(LikdTree::Stamp{f - kTtlFrames});
      expire += elapsedMs(e0, Clock::now());
    }
  }
  likd.waitForRebuild();
  long removed = static_cast<long>(pts.size()) - likd.size();
  printf("  frames: %u x %zu points, time to live %u frames, expired every 10\n",
         f, frame, kTtlFrames);
  printLikd("Expire total", expire, "ms");
  printLikd("Expire per point removed", 1e6 * expire / std::max(1L, removed), "ns");
  printLikdCount("Points kept", likd.size(), "");
  printLikdCount("Points removed", removed, "");
  printLikd("5-NN query total (seq)", knn, "ms");
  printLikd("5-NN, recent half (seq)", knn_recent, "ms");
  printLikd("5-NN answers that differ", 100.0 * differ / total_answers, "%");
}
#endif

#ifdef LIKD_BENCH_NANOFLANN
// What leaf buckets and pooled nodes give a static tree, which supports no
// insertion or deletion. nanoflann indexes points the caller keeps.
struct NanoflannCloud {
  const PointVector<PointType>& pts;
  size_t kdtree_get_point_count() const { return pts.size(); }
  float kdtree_get_pt(size_t i, size_t dim) const {
    const PointType& p = pts[i];
    return dim == 0 ? p.x : (dim == 1 ? p.y : p.z);
  }
  template <class BBox>
  bool kdtree_get_bbox(BBox&) const {
    return false;
  }
};
using NanoflannTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, NanoflannCloud, float, uint32_t>,
    NanoflannCloud, 3, uint32_t>;

void nanoflannReference(const PointVector<PointType>& pts,
                        const PointVector<PointType>& queries,
                        const PointVector<PointType>& range_queries,
                        float radius) {
  NanoflannCloud cloud{pts};
  for (size_t leaf : {10, 32}) {
    releaseFreeMemory();
    MemorySample m0 = MemorySample::now();
    auto t0 = Clock::now();
    NanoflannTree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(leaf));
    auto t1 = Clock::now();
    MemorySample m1 = MemorySample::now();
    auto knn = [&](size_t k) {
      std::vector<uint32_t> idx(k);
      std::vector<float> d2(k);
      for (const auto& q : queries) {
        const float qv[3] = {q.x, q.y, q.z};
        nanoflann::KNNResultSet<float, uint32_t> found(k);
        found.init(idx.data(), d2.data());
        tree.findNeighbors(found, qv);
      }
    };
    auto q0 = Clock::now();
    knn(1);
    auto q1 = Clock::now();
    knn(K);
    auto q2 = Clock::now();
    std::vector<nanoflann::ResultItem<uint32_t, float>> matches;
    for (const auto& q : range_queries) {
      const float qv[3] = {q.x, q.y, q.z};
      tree.radiusSearch(qv, radius * radius, matches);
    }
    auto q3 = Clock::now();
    printf("  nanoflann static, leaf %2zu: build %.2f ms, 1-NN x200k %.2f ms, "
           "5-NN x200k %.2f ms, radius x20k %.2f ms, heap %.2f B/pt\n",
           leaf, elapsedMs(t0, t1), elapsedMs(q0, q1), elapsedMs(q1, q2),
           elapsedMs(q2, q3), perPoint(m0.heap, m1.heap, pts.size()));
  }
  printf("  (nanoflann heap excludes the %zu B/pt of points it indexes)\n",
         sizeof(PointType));
}
#endif

}  // namespace

int main(int argc, char** argv) {
  const char* map = nullptr;
  size_t max_points = 0, frame = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--max-points" || arg == "--frame") && i + 1 < argc) {
      (arg == "--frame" ? frame : max_points) = std::stoul(argv[++i]);
    } else if (arg[0] != '-' && !map) {
      map = argv[i];
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [map.pcd [--max-points N] [--frame N]]" << std::endl;
      return 1;
    }
  }
  PointVector<PointType> pts;
  if (map) {
    pcl::PointCloud<PointType> cloud;
    if (pcl::io::loadPCDFile<PointType>(map, cloud) < 0) {
      std::cerr << "Failed to read " << map << std::endl;
      return 1;
    }
    for (const auto& p : cloud.points) {
      if (max_points && pts.size() == max_points) break;
      if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
        pts.push_back(p);
    }
    if (!frame) frame = 2000;
    std::cout << "Map " << map << ": " << pts.size() << " points, " << frame
              << "-point frames\n";
  } else {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    pts.resize(100000);
    for (auto& p : pts) {
      p.x = dist(rng);
      p.y = dist(rng);
      p.z = dist(rng);
    }
    if (!frame) frame = 1000;
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
    const float radius = map ? 1.0f : 10.0f;
    const float box_half = radius;
    LikdTree likd;
    releaseFreeMemory();
    MemorySample m0 = MemorySample::now();
    auto t0 = Clock::now();
    likdBuild(likd, pts);
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
    // ikd-tree does not sort its radius results: compare it with both. Runs
    // last so that the rows above are timed as before.
    LikdTree::SearchOptions unsorted;
    unsorted.sorted = false;
    size_t likd_in_radius_unsorted = countResults(
        range_queries, [&](const PointType& q, PointVector<PointType>& res) {
          likd.radiusSearch(q, radius, res, dists, unsorted);
        });
    auto r5 = Clock::now();

    printRow("Build", elapsedMs(t0, t1), elapsedMs(t2, t3));
    printRow("1-NN x200k (seq)", elapsedMs(q0, q1), elapsedMs(q1, q2));
    printRow("5-NN x200k (seq)", elapsedMs(q2, q3), elapsedMs(q3, q4));
    printRow("Radius search x20k (seq)", elapsedMs(r0, r1), elapsedMs(r1, r2));
    printRow("Radius unsorted x20k (seq)", elapsedMs(r4, r5), elapsedMs(r1, r2));
    printRow("Box search x20k (seq)", elapsedMs(r2, r3), elapsedMs(r3, r4));
    printf("  radius %.1f m: %.1f points per query, box half size %.1f m: %.1f\n",
           radius, double(likd_in_radius) / range_queries.size(), box_half,
           double(likd_in_box) / range_queries.size());
    if (likd_in_radius != ikd_in_radius)
      printf("  WARNING: radius search found %zu points, ikd-tree %zu\n",
             likd_in_radius, ikd_in_radius);
    if (likd_in_radius_unsorted != likd_in_radius)
      printf("  WARNING: unsorted radius search found %zu points, sorted %zu\n",
             likd_in_radius_unsorted, likd_in_radius);
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
#ifdef LIKD_BENCH_NANOFLANN
    nanoflannReference(pts, queries, range_queries, radius);
#endif
    delete ikd;
  }

  // ============================================================
  // Part 2: Incremental Insertion Test
  // ============================================================
  std::cout << "\n=== Part 2: Incremental insertion (query, then insert) ==="
            << std::endl;
  runStream(pts, frame, 0.0f);
  streamMemory(pts, frame, 0.0f);

  if (map) {
    std::cout << "\n=== Part 3: Local map, 100 m cube, box delete every 10 "
                 "frames ===" << std::endl;
    runStream(pts, frame, 50.0f);
    streamMemory(pts, frame, 50.0f);
  }
#ifdef LIKD_BENCH_STAMPS
  std::cout << "\n=== Part 4: Part 2 again, likd-tree touching each frame's "
               "5-NN neighbors ===" << std::endl;
  runStream(pts, frame, 0.0f, /*touch=*/true);
  if (map) {
    std::cout << "\n=== Part 5: Part 3's frames with a time to live instead "
                 "of a box (likd-tree only) ===" << std::endl;
    ttlStream(pts, frame);
  }
#endif
  return 0;
}
