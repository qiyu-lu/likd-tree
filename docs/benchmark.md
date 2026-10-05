# Benchmark

likd-tree against [ikd-Tree](https://github.com/hku-mars/ikd-Tree), and how to
reproduce the numbers. ikd-Tree is only needed here: it is not part of the
library.

![likd-tree compared with ikd-tree: 10.1x faster batch build, 3.8x faster insertion, 19.1x faster slowest frames, 5.5-6.8x faster nearest-neighbor queries, 27.5x faster box deletion, 7.3x less memory](../imgs/benchmark.png)

Both trees answer the same queries with the same threading (both sequential,
or both TBB-parallel). Each frame is queried against the map before it is
inserted, as in LiDAR odometry. Medians of 10 runs on an AMD Ryzen 7 7700,
GCC 9.4, -O3. Memory is the heap growth per point (glibc `mallinfo()`),
allocator overhead included.

## Real LiDAR maps
Both maps are streamed in file order, 2000-point frames. A frame of
`Global_map_sprase.pcd` (1.29M points) spans about 31 m, like a real scan.
`globalMap.pcd` (1.74M points) is stored in an order that makes each frame
span most of the map, so its streaming numbers resemble random insertion.

`Global_map_sprase.pcd`:

| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| Batch build, all points | 36 ms | 365 ms | **10.1x** |
| Insert, total | 353 ms | 1352 ms | **3.8x** |
| Insert, slowest 1% of frames | 1.1 ms | 21.4 ms | **19.1x** |
| 1-NN queries, sequential | 360 ms | 2446 ms | **6.8x** |
| 5-NN queries, sequential | 771 ms | 4291 ms | **5.6x** |
| 5-NN queries, TBB | 153 ms | 834 ms | **5.5x** |
| Memory after batch build | 25.3 B/point | 160 B/point | |
| Memory after streaming | 25.5 B/point | 186 B/point | |

`globalMap.pcd`:

| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| Batch build, all points | 54 ms | 579 ms | **10.6x** |
| Insert, total | 707 ms | 2621 ms | **3.7x** |
| Insert, slowest 1% of frames | 2.2 ms | 42.9 ms | **19.5x** |
| 1-NN queries, sequential | 1210 ms | 2751 ms | **2.3x** |
| 5-NN queries, sequential | 1934 ms | 5191 ms | **2.7x** |
| 5-NN queries, TBB | 299 ms | 712 ms | **2.4x** |
| Memory after batch build | 25.3 B/point | 160 B/point | |
| Memory after streaming | 28.1 B/point | 252 B/point | |

Local map kept to a 100 m cube around the sensor, box deletion every 10
frames, `Global_map_sprase.pcd`:

| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| Box deletion, total | 1.04 ms | 28.6 ms | **27.5x** |
| Points stored, deleted ones included (105,921 kept) | 105,968 | 107,705 | |
| Memory per point kept | 26.2 B | 2272 B | |

## 100K uniform random points
1000-point frames:

| Metric | likd-tree | ikd-tree | Speedup |
|--------|-----------|----------|---------|
| Insert, total | 24.3 ms | 75.8 ms | **3.1x** |
| 5-NN queries, sequential | 80.2 ms | 191.0 ms | **2.4x** |
| 5-NN queries, TBB | 15.3 ms | 27.4 ms | **1.8x** |
| Memory after streaming | 29.5 B/point | 161 B/point | |

Earlier versions of this README compared TBB-parallel likd-tree queries with
sequential ikd-tree queries; the query numbers above are like-for-like.

The chart above is drawn by `python3 tools/plot_benchmark.py` from the
numbers in the first table.

## Reproducing the results

```bash
cmake -B build -DBUILD_BENCHMARK=ON
cmake --build build
./build/benchmark                                    # random points
./build/benchmark ./test/pcd/Global_map_sprase.pcd   # real map + local-map test
./build/benchmark ./test/pcd/globalMap.pcd
```

CMake gets ikd-Tree through the git submodule, or downloads it. Without
network access, compile against a local ikd-Tree checkout instead:

```bash
g++ -std=c++17 -O3 test/benchmark.cpp <ikd-Tree>/ikd-Tree/ikd_Tree.cpp \
  -I<ikd-Tree>/ikd-Tree -Isrc -I/usr/include/eigen3 -I/usr/include/pcl-1.10 \
  -lpcl_io -lpcl_common -lboost_system -ltbb -lpthread -o build/benchmark
```

Besides timings, the benchmark reports memory per point: heap growth from
glibc's `mallinfo()`, RSS growth, and `memoryUsage()`. It also defines
`LIKD_TREE_STATS`, so it reports the longest background rebuild and how long a
queued write waited before queries could see it. To compare against static
nanoflann trees (1.5.0 or newer) as well, pass
`-DNANOFLANN_INCLUDE_DIR=<dir>` to CMake, or add
`-DLIKD_BENCH_NANOFLANN -I<dir>` to the command above. Adding
`-DLIKD_BENCH_STAMPS` makes likd-tree keep stamps: Parts 1–3 then show what
they cost, Part 4 replays Part 2 touching each frame's 5-NN neighbors, and on
a map Part 5 replaces Part 3's box with a time to live of 50 frames.

## Comparing two versions

Single runs on the same machine can differ by 10%
or more, so a few runs of each version can't show a 5% change.
Keep an executable built from the baseline, run it and the new one
alternately at least five times each, and compare the medians:

```bash
cp build/benchmark build/benchmark_base    # built from the baseline commit
# ...rebuild build/benchmark from the new version...
python3 test/compare_benchmarks.py build/benchmark_base build/benchmark \
    --map test/pcd/globalMap.pcd --runs 5
```

The script swaps which executable runs first every round. A process's first
large allocations, such as Part 1's build, depend on the process that ran
just before it: with the baseline always first, a build whose code had not
changed measured 5–11% slower, and 2–10% faster with the order reversed.

Passing the same executable twice shows the noise floor. The ikd-tree code is
identical in both executables, so its new/base ratio shows how much the
machine drifted during the comparison.
