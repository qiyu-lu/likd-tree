# likd-tree

**A lightweight incremental KD-tree for robotics**

[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`likd-tree` is a header-only C++17 KD-tree for maps that change: points are
inserted and deleted while the tree rebalances itself on a background thread.
It is inspired by [ikd-Tree](https://github.com/hku-mars/ikd-Tree) and written
from scratch.

> This repository continues [scomup/likd-tree](https://github.com/scomup/likd-tree)
> by Liu Yang. See [License & Acknowledgements](#-license--acknowledgements).

## 🚀 Key Features

- **🔄 Incremental**: Insert points, delete by point or by box, with automatic background rebalancing
- **🔍 Queries**: Nearest neighbor, k-nearest neighbors, radius and box search
- **🪶 Lightweight**: Header-only library (~2100 lines of C++17) - no build required
- **📦 Compact**: About 25 bytes per `pcl::PointXYZ` point, a sixth of ikd-tree's: points are stored in leaf buckets
- **⚡ Fast**: On a 1.3M-point LiDAR map, 3.8x faster insertion, 5.6x faster 5-NN search and 27x faster box deletion than ikd-tree
- **🧠 Intelligent**: Delayed and batched rebuilding of multiple non-overlapping unbalanced subtrees *(paper-worthy?)*
- **⏱️ Time-aware (optional)**: Per-point stamps to refresh points seen again, expire old ones, and query only recent ones, in the spirit of Redis' LRU and TTL
- **🔒 Thread-safe**: Queries run concurrently with each other and with writes
- **🔧 Flexible**: Any point type through the `PointTraits` template

## 📊 Performance

![likd-tree compared with ikd-tree: 10.1x faster batch build, 3.8x faster insertion, 19.1x faster slowest frames, 5.5-6.8x faster nearest-neighbor queries, 27.5x faster box deletion, 7.3x less memory](imgs/benchmark.png)

Full tables, conditions and how to reproduce them: [docs/benchmark.md](docs/benchmark.md).

## 🎯 Quick Start

Copy [`src/likd_tree.hpp`](src/likd_tree.hpp) into your project and include
it. It needs C++17 and Eigen, nothing else: no build step, and no need to
clone the `thirdparty` submodule, which only the benchmark uses.

```cpp
#include <pcl/point_types.h>
#include "likd_tree.hpp"

using PointType = pcl::PointXYZ;
KDTree<PointType> tree;

PointVector<PointType> points = {...};
tree.build(points);
tree.addPoints(new_points);

// Nearest neighbor: a copy of the point (std::optional) and its distance
auto [nearest, dist] = tree.nearestNeighbors(query);

// k nearest neighbors and radius search, nearest first
PointVector<PointType> results;
std::vector<float> distances;
tree.knnSearch(query, 5, results, distances);
tree.radiusSearch(query, 2.0f, results, distances);

// Options are set by name. Skipping the sort makes a radius search that
// returns many points several times faster.
KDTree<PointType>::SearchOptions options;
options.sorted = false;   // radiusSearch: any order
options.max_dist = 1.0f;  // knnSearch: only neighbors within 1 m
tree.radiusSearch(query, 2.0f, results, distances, options);

// Box search and deletion (boundary included)
KDTree<PointType>::AABB box({-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f});
tree.boxSearch(box, results);
tree.deleteBox(box);
tree.deletePoints(points_to_remove);  // exact coordinates

int n = tree.size();
size_t bytes = tree.memoryUsage();
```

Batch versions of `nearestNeighbors` and `knnSearch` take a vector of queries.

### Parallel queries with TBB

Define `LIKD_TREE_USE_TBB` before including the header and link against TBB.
Batch queries and large builds then run in parallel.

### Thread safety

- Queries can run from any number of threads, concurrently with writes.
- Writes (`build`, `addPoints`, `deletePoints`, `deleteBox(es)`) are
  serialized internally.
- While a rebuild runs in the background, writes are queued and applied in
  order right after, so a query may briefly not see them. A write call larger
  than 2000 points may also return before all of it is visible.
- Call `waitForRebuild()`, or pass `wait_for_rebuild = true`, when writes
  must be visible before you continue. Doing so from time to time also keeps
  the queue short when writes arrive back to back with no pause.

### Custom point types

Any type with `x`, `y`, `z` members works as is. For anything else,
specialize `PointTraits`:

```cpp
struct MyPoint {
  float coords[3];
};

template <>
struct PointTraits<MyPoint> {
  static constexpr int DIM = 3;
  static inline float coord(const MyPoint& pt, int axis) {
    return pt.coords[axis];
  }
  static inline float sqrDist(const MyPoint& a, const MyPoint& b) {
    float dx = a.coords[0] - b.coords[0];
    float dy = a.coords[1] - b.coords[1];
    float dz = a.coords[2] - b.coords[2];
    return dx * dx + dy * dy + dz * dz;
  }
};

KDTree<MyPoint> tree;
```

The point type must be default-constructible and copy-assignable. More
examples are in [test/demo.cpp](test/demo.cpp).

### Stamps (optional)

With `TRACK_STAMPS`, every point keeps a stamp: a frame number or a time, in
any unit that never decreases or wraps.

```cpp
struct StampedOptions : DefaultOptions {
  static constexpr bool TRACK_STAMPS = true;
};
using StampedTree = KDTree<PointType, PointTraits<PointType>, StampedOptions>;
using Stamp = StampedTree::Stamp;

StampedTree map;
map.addPoints(scan, Stamp{frame});         // insert, stamped with the frame
map.touchPoints(neighbors, Stamp{frame});  // seen or used again: refresh
map.expireBefore(Stamp{frame - 100});      // drop what was not seen in 100 frames

StampedTree::SearchOptions recent;
recent.min_stamp = Stamp{frame - 10};      // only points seen in the last 10
map.knnSearch(query, 5, results, distances, recent);
```

- Points written without a stamp never expire, and `min_stamp` never hides
  them.
- `touchPoints` matches exact coordinates and never lowers a stamp.
- Stamps cost 5–7 bytes per point. Without `TRACK_STAMPS` the tree has the
  same layout and speed as before, and the calls above do not compile.

### Leaf size

Points are stored in leaf buckets of up to 32 points. The size is a
compile-time option:

```cpp
struct MyOptions : DefaultOptions {
  static constexpr int LEAF_SIZE = 64;  // 2 to 64
};
KDTree<PointType, PointTraits<PointType>, MyOptions> tree;
```

On the benchmark maps, 64 saves another 10% of memory but makes 5-NN queries
up to 14% slower.

## 🧩 How it works

Points live in leaf buckets and inner nodes only split space. Every node
keeps the bounding box of its points: searches prune with it, and box
deletion uses it to drop whole subtrees in O(1). When a subtree becomes
unbalanced, or more than half of its points are deleted, a background thread
rebuilds it while queries keep running.

[docs/design.md](docs/design.md) describes the layout, each operation, the
rebuild protocol and the rules that make it thread-safe.

Incremental insertion without and with rebalancing (recorded with the
original one-point-per-node tree):

![no-rebuild](imgs/no-rebuild.gif)
![rebuild](imgs/rebuild.gif)

## 🛠️ Demo, Tests & Benchmark

The library needs no build. CMake is only for what is in `test/`:

```bash
git clone https://github.com/qiyu-lu/likd-tree.git
cd likd-tree
cmake -B build && cmake --build build
./build/demo                              # usage examples
(cd build && ctest --output-on-failure)   # unit tests
```

The tests compare every query against brute force, including random mixes of
insertion and deletion, and exercise concurrent readers and writers.

With PCL visualization installed, CMake also builds two demos that run a
query on a PCD map, check it against brute force and show it in a viewer
(`--no-vis` skips the viewer):

```bash
./build/nearest_search_pcd_demo <map.pcd> <qx> <qy> <qz>
./build/radius_search_pcd_demo <map.pcd> <qx> <qy> <qz> <radius>
```

The benchmark against ikd-tree is built with `-DBUILD_BENCHMARK=ON`, which
fetches ikd-Tree. See [docs/benchmark.md](docs/benchmark.md).

## 📋 TODO

- [ ] Python bindings

## 📄 License & Acknowledgements

MIT, see [LICENSE](LICENSE). The original likd-tree is © 2026 Liu Yang
([scomup/likd-tree](https://github.com/scomup/likd-tree)); changes made in this
repository are © 2026 qiyu-lu.

The design is inspired by [ikd-Tree](https://github.com/hku-mars/ikd-Tree)
(GPL-2.0). The library contains no ikd-Tree code; only the benchmark builds
against it, pulled in as a git submodule.
