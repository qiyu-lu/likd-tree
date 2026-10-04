# likd-tree 改进计划

- 状态：v2，已审阅。Phase 0、Phase 1 可以开工；Phase 2 推迟；Phase 3 开工前需再审
- 日期：2026-10-04
- 分支：`improvements`（基于 `main` @ `6443087`）
- 分工：执行方为 Claude（编写 v1 的会话），审阅方为另一个 Claude 会话

v2 由审阅方直接修改本文档得到，没有单独的 review 文件；v1 见提交 `58e6335`，审阅结论和改动理由见第 10 节“决定记录”。执行中如果达不到验收标准，或者需要偏离本计划，执行方先在“决定记录”里写明原因并回报用户，不自行放宽标准。

## 1. 目标、非目标与思路来源

**目标**（按优先级）：

1. **降低内存**。当前每存一个 `pcl::PointXYZ` 点要占 96 B，而点本身只有 16 B。目标是不超过 35 B/点（一次性构建约 26 B/点，流式插入约 31 B/点，见 5.1）。
2. **不牺牲速度**。构建、插入、kNN、box 删除都不能回退，具体阈值见各阶段的验收标准。
3. **时间感知的地图管理**。借鉴 Redis 的 TTL 和近似 LRU，给点加时间戳，支持 `touch` 和 `expire`，为以后在 LIO 中对比地图淘汰策略打基础。
4. （可选）重建期间写入立即可见。

**非目标**（本轮不做）：

- SIMD 距离计算
- 无锁读 / RCU
- Python 绑定更新
- FAST-LIO2 集成与整机实验（见第 8 节）
- 改变现有查询接口的语义

**与 Redis 的对照**。决定借鉴什么之前先做了这张对照：

| Redis 机制 | likd-tree 现状 | 本计划 |
|---|---|---|
| 惰性删除（访问时检查） | 已有：`deleted` 标记和整树 `tree_deleted` 标记，查询在 `valid_size == 0` 处剪枝 | 保留 |
| 主动删除（周期性清理） | 已有：子树删除比例超过 50% 时后台重建回收 | 保留 |
| AOF 重写缓冲（后台重写期间的写入先缓存，完成后追加） | 已有，是同一模式：重建期间写入进入 `pending_ops_`，结束后回放 | Phase 2 可选改进 |
| TTL、近似 LRU（对象头 24 bit 时钟加采样淘汰） | 无 | Phase 3 |
| IO 多路复用 | 不适用：kd 树没有 IO，瓶颈在 CPU 和访存。Redis 单线程的本质是“单写者”，likd-tree 的写者本来就是串行的 | 不做 |
| 通用对象模型（dictEntry、robj 等） | 不适用，而且正是 Redis 费内存的原因 | 不做 |

**与 nanoflann 的对照**：借鉴它静态树的布局（叶子桶、更少的分配次数），不借鉴它的动态方案。数据见 2.2。

## 2. 现状与基线

### 2.1 内存去向

`KDTree<pcl::PointXYZ>::Node` 是 80 B：

| 字段 | 字节 |
|---|---|
| left / right / parent 指针 | 24 |
| AABB（6 个 float） | 24 |
| point（PointXYZ，16 字节对齐） | 16 |
| axis、subtree_size、valid_size | 12 |
| 4 个 bool | 4 |

每个节点单独 `new`，glibc 每次分配还要多占约 16 B（块头加对齐），实测为 96.0 B/点。

### 2.2 与 ikd-Tree、nanoflann 的对比

测试条件：

- 数据：`test/pcd/globalMap.pcd`，1,742,788 点
- 线程：单线程，未定义 `LIKD_TREE_USE_TBB`
- 查询：20 万次 5-NN，查询点为随机地图点加 σ=5 cm 的高斯噪声
- 内存：用 glibc `mallinfo()` 统计建树前后的堆增量
- 结果：三者的 kNN 结果逐一核对，完全一致

| | 构建 | 20 万次 5-NN | 每点内存 |
|---|---|---|---|
| likd-tree | 325 ms | 371 ms | 96.0 B |
| ikd-Tree | 513 ms | 547 ms | 160.0 B（不含构造时分配的约 48 MB 操作队列） |
| nanoflann 1.5.0 静态，叶子 10 点 | 222 ms | 313 ms | 15.5 B 索引 + 16 B 点数据 |
| nanoflann 1.5.0 静态，叶子 32 点 | 213 ms | 320 ms | 7.7 B 索引 + 16 B 点数据 |

流式场景：每帧 2000 点，按文件顺序，先做 5-NN 查询再插入。

| | 查询总计 | 插入总计 | 最慢一帧插入 | 每点内存 |
|---|---|---|---|---|
| likd-tree | 3700 ms | 818 ms | 6.9 ms | 96.1 B |
| nanoflann 动态版（`maximumPointCount = 2^21`） | 5583 ms | 703 ms | 101.4 ms | 21.3 B 索引 + 16 B 点数据 |

由此得到三点结论：

- **静态 nanoflann 的查询只快约 18%**。likd-tree 逐节点 AABB 剪枝已经有效，差距主要在内存（3~4 倍）和构建速度。
- **nanoflann 动态版不适合做增量方案**。`KDTreeSingleIndexDynamicAdaptor` 用的是 Bentley–Saxe 对数方法：
  - 查询要遍历全部子树，比 likd-tree 慢 1.5 倍；
  - 合并在调用线程里同步进行，最慢一帧要 101 ms；
  - `removePoint` 只是把 `treeIndex_[idx]` 置为 -1，被删的点永不回收，点数据也必须由调用方一直保留。
- **方向**：借鉴 nanoflann 的静态布局，保留 likd-tree 的增量机制。增量机制包括：替罪羊式局部重建、后台线程、惰性删除标记，以及逐节点 AABB（box 删除比 ikd-Tree 快 20 多倍就靠它）。

以上数字来自执行方会话里的临时程序。Phase 0 会把内存统计和 nanoflann 对比加进仓库的 benchmark，之后这些数字就可以从仓库复现。

### 2.3 benchmark 基线

条件：`main` @ `6443087`，本机，`LIKD_TREE_USE_TBB`，`-O3`，GCC 9.4。

`./benchmark test/pcd/globalMap.pcd` 的结果：

| 项 | likd-tree | ikd-tree |
|---|---|---|
| Part 1：构建 | 99 ms | 572 ms |
| Part 1：1-NN ×1000 / 5-NN ×1000 | 1.46 / 2.03 ms | 1.82 / 2.90 ms |
| Part 2：插入总计 / 最慢一帧 | 1072 ms / 6.0 ms | 2531 ms / 64.7 ms |
| Part 2：1-NN 顺序 / 5-NN 顺序 / 5-NN TBB | 1959 / 3511 / 483 ms | 2586 / 4898 / 658 ms |
| Part 2：查询读到“排队写入尚未生效”的地图的比例 | 0.014% | — |
| Part 3：插入总计 / 最慢一帧 | 668 ms / 1.8 ms | 2388 ms / 32.9 ms |
| Part 3：1-NN 顺序 / 5-NN 顺序 / 5-NN TBB | 1217 / 2481 / 368 ms | 1863 / 3951 / 674 ms |
| Part 3：box 删除总计 | 10.8 ms | 313 ms |
| Part 3：保留点数 / 内存中节点数 | 78,334 / 78,338 | 78,334 / 250,341 |
| Part 3：查询读到排队写入前地图的比例 | 0.000% | — |

`./benchmark`（10 万个均匀随机点）的结果：

| 项 | likd-tree |
|---|---|
| 构建 | 8.9 ms |
| 插入总计 / 最慢一帧 | 34.5 ms / 0.45 ms |
| 5-NN 顺序 / TBB | 162 ms / 24 ms |

同一台机器上，不同次运行的波动可达约 15%（README 里的插入总计是 1236 ms，今天测得 1072 ms）。3 次取中位数分辨不出 5% 的差异，因此验收一律按下面的**交替对比测法**：

- 保留一份从基线提交编译的 benchmark 可执行文件（`build/benchmark_base`）；
- 基线与新版本交替运行，各不少于 5 次；
- 比较两者的中位数。

## 3. 路线总览

| 阶段 | 内容 | 主要收益 | 风险 | 优先级 |
|---|---|---|---|---|
| 0 | 库增加 `memoryUsage()`；benchmark 增加内存统计、重建延迟指标和 nanoflann 参照（可选依赖）；约定交替对比测法 | 可复现的度量 | 低 | 必做 |
| 1 | 叶子桶 + 紧凑节点 | 内存约降 3 倍，构建和插入加快 | 中：重写核心数据结构 | 高 |
| 2 | 重建期间写入立即可见 | 读一致性 | 高：要改并发协议 | 低，实测陈旧比例只有 0.014%，建议推迟 |
| 3 | 时间戳、`touchPoints`、`expireBefore`（默认关闭） | 时间感知的地图管理 | 中 | 中，开工前再审 |
| 4 | FAST-LIO2 适配与整机实验 | 系统级结论 | — | 本轮不做 |

## 4. Phase 0：测量基础设施

1. **库内的 `memoryUsage()`**。返回树结构占用的字节数，不含 malloc 开销。
   - Phase 0 按现有结构返回 `nodeCount() * sizeof(Node)`；
   - Phase 1 改为 `叶子数 * sizeof(Leaf) + 内部节点数 * sizeof(Inner)`，为此要维护两个计数器。
   - 有了它，Phase 1 改变 `nodeCount()` 的语义后，内存仍然可以直接观测。
2. **benchmark 的内存统计**。
   - Part 1：两棵树先后构建，分别取 glibc `mallinfo()` 的堆增量（`uordblks + hblkhd`，覆盖所有 arena），换算成字节/点。ikd-tree 的操作队列在构造时分配，所以从构造之后开始计量。
   - Part 2/3：`runStream` 里两棵树在同一个循环里交替插入，堆增量是两者之和，分不开。因此增加**单独的内存测量轮**：只跑一棵树的流式插入，取堆增量，再换另一棵树。
   - 同时打印 RSS 增量（`/proc/self/statm`）：`uordblks` 不含碎片，而重建会反复申请和释放节点。
   - Part 3 结束时按“字节/有效点”报告（分母是 `size()`）。
   - 非 glibc 平台跳过 `mallinfo()` 一项。`mallinfo()` 的字段是 `int`，超过 2 GB 会溢出；本机 glibc 2.31 没有 `mallinfo2()`，现在的数据量（最大约 280 MB）没有问题，代码里加注释说明。
3. **重建延迟指标**。benchmark 打印单次重建的最长耗时，以及重建期间 `pending_ops_` 的最大长度。它们回答“一次写入最久要等多久才对查询可见”，是 Phase 2 是否要做的依据。需要的计数放在库里，用宏 `LIKD_TREE_STATS` 控制，默认不编译。
4. **补齐计时项**。
   - Part 1 的查询从 1000 次改为 20 万次（现在总共只有 1.5–2 ms，测不准）。
   - 增加 `radiusSearch` 和 `boxSearch` 各一行计时，Phase 1 会改这两条路径。
5. **nanoflann 参照（可选）**。增加 CMake 选项 `-DNANOFLANN_INCLUDE_DIR=<dir>`；提供时，benchmark 多输出一行 nanoflann 静态树的构建、查询和内存，作为“静态布局上限”的参照。
   - nanoflann 是 BSD 许可证，只在 benchmark 中使用，不进入库本身。
6. **文档化手动构建方式**。`thirdparty/ikd-Tree` 子模块为空时，直接用 `../ikd-Tree` 编译（命令见第 10 节）；同时写明 2.3 的交替对比测法。

提交：

- `Add memoryUsage()`
- `Report memory per point in the benchmark`
- `Report rebuild latency and time radius and box search`
- （可选）`Add a nanoflann static reference to the benchmark`

Phase 0 完成后，用新的 benchmark 重测基线并记入“结果记录”，Phase 1 的验收以这份基线为准。

## 5. Phase 1：叶子桶 + 紧凑节点

### 5.1 数据结构

点只存放在叶子里，内部节点只负责划分。

```cpp
struct DefaultOptions {
  static constexpr int LEAF_SIZE = 32;
  // Phase 3 在这里增加 TRACK_STAMPS，不再增加模板参数
};

template <typename PointType, typename Traits = PointTraits<PointType>,
          typename Options = DefaultOptions>
class KDTree {
  static constexpr int LeafSize = Options::LEAF_SIZE;
  static_assert(LeafSize >= 2 && LeafSize <= 64);
  using Mask = std::conditional_t<(LeafSize <= 32), uint32_t, uint64_t>;

  struct Node {                 // 公共头
    Node* parent = nullptr;
    AABB aabb;                  // 子树中未删除点的包围盒
    int size = 0;               // 子树中存储的点数，含已删除未回收的
    int valid = 0;              // 子树中未删除的点数
    bool is_leaf;
    bool is_left_child = false;
    bool need_rebuild = false;  // 仅内部节点使用
    bool tree_deleted = false;  // 惰性整树删除标记，仅内部节点使用
  };
  struct Inner : Node {
    Node* left = nullptr;       // 子节点可以为空（例如整棵子树被重建成空）
    Node* right = nullptr;
    float split;                // 插入路由：coord < split 走左边
    int axis;
  };
  struct Leaf : Node {
    Mask deleted = 0;           // 第 i 位为 1 表示 pts[i] 已删除
    PointType pts[LeafSize];    // 已使用的槽位数就是 Node::size
  };
};
```

- **节点大小**：内部节点 72 B；`PointXYZ`、`LeafSize = 32` 时，叶子 576 B。
- **内存估算**。审阅时用 globalMap 模拟了叶子桶的构建与分裂（只做插入和分裂，不含重建与删除；字节数含每次分配 16 B 的 malloc 开销）：

  | LeafSize | 一次性构建 | 流式插入（每帧 2000 点，文件顺序） |
  |---|---|---|
  | 8 | 填充 83.1%，44.5 B/点 | 填充 70.6%，52.4 B/点 |
  | 16 | 83.1%，31.9 B/点 | 69.8%，38.0 B/点 |
  | 32 | 83.1%，25.6 B/点 | 69.3%，30.7 B/点 |
  | 64 | 83.1%，22.4 B/点 | 69.0%，27.0 B/点 |

  - 流式场景的填充率约 69%，是对半分裂的稳态值。重建会把子树重新打包到 (½, 1]，平均约 72%，改变不大。
  - **最坏情况**：重建后填充率下限 50%，再叠加删除比例上限 50%，约 75 B/有效点，仍低于现在的最好情况 96 B。所以不需要为稀疏叶子另加合并或按容量触发的重建。
- **`LeafSize` 通过 `Options` 配置**，默认 32：
  - 单元测试用 2、4、32、64 四个实例；
  - benchmark 扫描 8/16/32/64。16 及以下在流式场景过不了 35 B/点，所以**默认值只在 32 和 64 之间选**，8 和 16 只作为速度参照。
- **对 `PointType` 的新要求**：可默认构造、可拷贝赋值。仓库里现有的点类型（PCL、`Eigen::Vector3f`、demo 里的自定义结构体）都满足。

### 5.2 算法

**构建**：沿包围盒最长边取中位数递归划分，沿用现有的 `nth_element` 做法；区间满足 `r - l <= LeafSize` 时生成叶子。这样构建出的叶子填充率落在 (½, 1] 之间。TBB 并行构建方式不变。

- 中位数位置 `m = l + (r - l) / 2`，`split = coord(pts[m], axis)`，左子树取 `[l, m)`，右子树取 `[m, r)`。现在的代码把 `pts[m]` 放在节点自身、右子树取 `[m+1, r)`，这里要改。

**插入**：按 `split` 下降到叶子（`coord < split` 走左边）；遇到带 `tree_deleted` 标记的节点，先 `pushDown` 再下降。

- **路由到空位置**（`root_` 为空，或 `Inner` 的该侧子节点为空）：由写线程新建一个叶子挂上去。只新增节点，不违反不变量 1。

到达叶子后分三种情况：

- **叶子未满**：直接追加，扩展 AABB，O(1)。
- **叶子已满但有删除槽**：先就地压缩，复杂度 O(LeafSize)，然后追加。
- **叶子已满且没有删除槽**：分裂。
  - 沿叶内点的最长边取中位数（`m = size / 2`），`split = coord(pts[m], axis)`，左半是 `[0, m)`，右半是 `[m, size)`，与路由规则一致。
  - 原叶子保留左半；新分配一个叶子放右半；再新分配一个 `Inner` 节点，替换原叶子在父节点（或 `root_`）中的位置。
  - 分裂只新增节点，不释放节点。
- **回溯**：沿路径对祖先执行 `update()`，并对内部节点执行 `markIfUnbalanced()`。失衡判据与现在相同：`max(child.size) > α·size`，或 `size - valid > δ·size`，其中 α = 0.75、δ = 0.5。
- **全是重复点时**：中位数分裂退化为按位置对半分。查询和删除只依赖 AABB，不依赖 `split`，所以正确性不受影响；由此产生的失衡交给重建判据处理。

**按点删除**：与现在一样，按 AABB 包含关系下降（中位数两侧都可能有恰好等于 `split` 的点）。在叶子里找到第一个坐标相等且未删除的槽，置删除位，再重算叶子的 AABB，复杂度 O(LeafSize)。语义不变：每个请求只删除一份拷贝。

**按 box 删除**：

- 内部节点的 AABB 完全落在 box 内：打惰性整树标记，O(1)，与现在相同。
- 叶子：逐点检查并置位。
- 回溯时更新祖先。
- `pushDown` 推到叶子时，把该叶子的所有槽都置为删除。

**重建**：判据、后台线程、写入排队与回放的协议**全部不变**。区别只在两处：`collect` 跳过删除槽，`buildRecursive` 生成叶子。

**查询**（NN / kNN / radius / box）：

- 内部节点：照旧用 AABB 剪枝，先访问近的子树再访问远的子树。
- 叶子：顺序扫描；`deleted == 0` 时走不做位检测的快路径。

**最小重建规模**：`MIN_SUB_NUM`（现在是 8 个节点）改为按点数计算的 `MIN_REBUILD_SIZE`，初值为 `4 * LeafSize`，避免只有两三个叶子的小子树来回重建。

**实现时容易出错的两处**：

- **按真实类型释放节点**。`Node` 没有虚析构，`destroy` 必须按 `is_leaf` 转成 `Leaf*` 或 `Inner*` 再 `delete`；通过基类指针直接 `delete` 是未定义行为（大小和对齐都不同）。
- **掩码移位**。`LeafSize > 32` 时必须写 `Mask(1) << i`。“全部置删除”不能写 `(Mask(1) << size) - 1`：`size` 等于位宽时是未定义行为，要单独处理满叶的情况。

### 5.3 必须保持的不变量

1. **节点的释放**：除 `build()` 以外，节点只由重建线程在交换之后释放；写线程只新增节点（叶子分裂时），从不释放。
2. **结构修改的加锁**：追加、压缩、分裂、置删除位，都在 `tree_mutex_` 独占锁下进行。查询持有共享锁，看不到中间状态。
3. **重建期间的写入**：照旧进入 `pending_ops_`，所以 worker 无锁读取旧子树仍然安全。
4. **叶子不会成为重建候选**：失衡判据只对内部节点定义，因此叶子分裂不会让候选列表里出现悬空指针。
5. **字段语义不变**：`valid`、`size`、`aabb` 与现在的含义一致。查询在 `valid == 0` 处剪枝，不会读到惰性标记下尚未更新的数据。

### 5.4 考虑过但未采用的方案

- **只压缩单点节点**（32 位下标代替指针、标志位用位域）：约 64 B/点，只能省三分之一。
- **nanoflann 式“下标数组 + 外部点存储”**：需要稳定的点 ID，以及额外的压缩和重排逻辑，增量插入删除更复杂，收益与叶子桶相近。
- **删除时立即把点从叶子里移除**：会产生空叶和稀疏叶，需要在写路径上合并或摘除节点，从而在写路径上释放节点，破坏不变量 1。

### 5.5 接口变化

- **新增模板参数 `Options`**，有默认值。现有的 `KDTree<P>` 和 `KDTree<P, Traits>` 写法源码兼容。以后的编译期选项（Phase 3 的时间戳开关）都加在 `Options` 里，不再增加模板参数。
- **`Node` 从 public 改为 private**。仓库内（Python 绑定、demo、测试）没有用到它；外部代码如果用到会编译失败，需要在 README 里说明。
- **`nodeCount()` 的语义**改为“内存中存储的点数，包括已删除但尚未回收的点”。分桶之后这个数与内存脱钩（只有 1 个点的叶子也占 576 B），所以 README 和 benchmark 里表示内存的地方改用 `memoryUsage()`，按字节/有效点报告。
- **`python/src/likd_tree.hpp` 是头文件的一份旧拷贝**，本轮不更新，在 README 里注明它落后于 `src/`。

### 5.6 测试

- 现有单元测试全部通过，包括顺序版和 TBB 版两个可执行文件。
- 新增以下测试：
  - `LeafSize` 取 2、4、32、64 四个实例，跑同一组暴力对比测试。64 是必需的：没有它，`uint64_t` 掩码这条路径完全没有覆盖；
  - 超过 `LeafSize` 个完全相同的点：插入、kNN、逐个按点删除；
  - 叶子被删空；
  - box 删除整个叶子或整棵子树后，再插入（包括插入到空子节点的位置）；
  - 大规模删除后，确认内存被回收（`memoryUsage()` 回落）。
- **结构自检函数 `validate()`**，只在定义 `LIKD_TREE_TESTING` 时编译。遍历整棵树检查：
  - `parent` / `is_left_child` 与实际位置一致；
  - `size`、`valid`、`aabb` 等于由子节点（或叶内未删除的槽）重新算出的值；带 `tree_deleted` 标记的子树只检查标记节点自身的 `valid == 0`；
  - 叶子 `size <= LeafSize`，掩码中 `size` 以上的位为 0；
  - 叶子数和内部节点数与 `memoryUsage()` 用的计数器一致。

  `LeafSize` 为 2 和 4 的随机测试里，每次写操作并等重建结束后调用一次。
- 并发测试在 clang++-10 的 TSan 下没有告警。
- 全部单元测试在 ASan + UBSan（`-fsanitize=address,undefined`）下没有告警，覆盖基类指针转换和移位。

### 5.7 验收标准

一律按 2.3 的交替对比测法，与 Phase 0 完成后重测的基线比较。内存取 mallinfo 堆增量。

- **内存**（globalMap，默认 `LeafSize`）：
  - Part 1 建完树后：不超过 28 B/点；
  - Part 2 流式插入结束后：不超过 35 B/点，预期约 31 B；
  - Part 3 结束时：报告字节/有效点，不超过 75 B（5.1 的最坏情况上界）。
- **速度**：构建、Part 2/3 的插入总计、1-NN 和 5-NN（顺序与 TBB）、`radiusSearch`、`boxSearch`，回退都不超过 5%。
- **最慢一帧插入**：不超过基线 + 1 ms。
- **box 删除总计**：不超过基线的 1.2 倍。叶子内删除后要重算 AABB，所以允许小幅变慢。
- **大点类型**：扫描 `LeafSize` 时顺带测 `pcl::PointXYZINormal`（48 B，`LeafSize = 32` 时一个叶子约 1.6 KB），确认 5-NN 相对基线不回退。

达不到标准时，执行方先分析原因并回报用户，不自行放宽标准。

### 5.8 可选子步骤

是否做，视 5.7 的测量结果决定：

- **小子树就地重建**：规模小于 1024 点的候选，由写线程在独占锁内直接重建，不交给后台线程，以减少线程交接和写入排队。如果采用：
  - 不变量 1 要改成“节点只在独占锁内或在交换之后释放”；
  - 每次写调用设总预算（初值为累计 4096 点），超出的候选仍交给后台线程，否则一帧里触发几十次就会超过“最慢一帧 + 1 ms”；
  - 必须在 `topmostCandidates` 过滤**之后**执行：过滤要读每个候选节点的 `parent`，先重建会读到已释放的内存。
- **按子节点 AABB 距离决定访问顺序**：替代现在按 `split` 比较决定先访问哪个子树。
- **内存池**：分桶后分配次数大约只有现在的 1/12，malloc 开销预计只剩约 1.3 B/点。只有 profile 显示分配是瓶颈时才做。

### 5.9 提交拆分

1. `Add an Options template parameter and parameterized tests`：先让测试可以按 `LeafSize` 参数化。
2. `Store points in leaf buckets`：核心改动，连同 `validate()` 和 `memoryUsage()` 的新实现。满叶分裂时丢掉已删除的槽，不把它们拷进新叶子。
3. `Reuse deleted slots before splitting a full leaf`：只增加“有删除槽时压缩而不分裂”这一个判断。
4. `Pick the default leaf size from a sweep`：附带测量数据。
5. README：补充数据结构说明，更新性能表，增加内存一列。

## 6. Phase 2：重建期间写入立即可见（建议推迟）

**现状**：重建期间所有写入都进入 `pending_ops_`，回放之前查询看不到。实测影响很小：globalMap 流式测试中只有 0.014% 的 5-NN 结果受影响，局部地图测试中为 0%。Phase 1 会让重建更快，如果再加上小子树就地重建，这个比例预计还会下降。

**候选方案（如果决定要做）**：快照 + 直接写入 + 日志回放。

1. **快照**：worker 在共享锁下收集候选子树中的有效点，并记下每棵子树日志的起始位置。
2. **构建期间**：写线程直接修改活树（包括正在重建的旧子树），所以查询立即可见。
   - 触及正在重建子树的操作，额外追加到该子树的日志里。
   - 旧子树内部不再标记重建候选。
3. **回放与交换**：worker 在锁外分批把日志回放到尚未发布的新子树上；追到只剩少量尾部操作时，再取独占锁回放剩余部分并完成交换。这一步类似 Redis 的 AOF 重写缓冲和主从复制追赶。

**代价**：

- 收集快照期间，写线程会被阻塞。而 `SharedMutex` 偏向写者：一旦有写者在排队，新来的读者也要等。根级重建时，收集最坏可能需要 10–20 ms。
- 协议复杂度明显上升，需要处理日志起点、候选过滤、多棵子树同时重建等细节。

**被否决的方案**：查询时叠加扫描 `pending_ops_`。这样四类查询都要合并结果，而且有重复点时，按点删除“只删一份”的语义无法精确实现。

**结论**：推迟。Phase 1 完成后重新测量陈旧比例，连同 Phase 0 新增的两个指标（单次重建最长耗时、`pending_ops_` 最大长度）一起报告，再决定是否要做。陈旧比例只说明有多少查询读到旧地图；对 LIO 更要紧的是一次写入最久要等多久才可见，根级重建时这个值可能达到一帧的量级。

## 7. Phase 3：时间戳、touch 与 expire（默认关闭）

### 7.1 语义

| 接口 | 含义 | 对应的 Redis 机制 |
|---|---|---|
| `addPoints(pts, stamp)` | 插入并记录时间戳（帧号或秒） | SET + EXPIRE |
| `touchPoints(pts, stamp)` | 刷新时间戳：点被再次观测到，或在配准中被用到 | 访问时更新 LRU 时钟 |
| `expireBefore(T)` | 删除时间戳早于 T 的点 | 主动过期 |
| 查询参数 `min_stamp` | 跳过时间戳过旧的点，但不删除它们 | 惰性过期 |

- **刷新由写线程批量调用**，查询本身仍然只读，并发模型不变。
- **为什么必须有 touch**：在 FAST-LIO2 式建图里，体素中已经有点时，新点通常不会再插入；墙面这类静态结构的插入时间会一直停在第一次观测。如果只按插入时间做 TTL，会把静态结构误删掉。所以过期必须基于“最后观测或使用的时间”，这本质上就是 LRU。
- **不做 `touchBox`**（v1 里有，审阅后去掉）。它需要惰性刷新标记：两种惰性标记的下推顺序、查询沿路径携带未下推的值、重建线程收集点时也要带着标记，是这一阶段最复杂的部分。而它的语义并不可靠：box 不等于视场，被遮挡的点也会被刷新。
- **留待以后**：LFU（计数加衰减）、按内存上限淘汰（类似 Redis maxmemory 的采样淘汰），本阶段不做。

### 7.2 实现

- **开关 `Options::TRACK_STAMPS`**，默认 `false`。关闭时，`sizeof` 和性能与 Phase 1 完全一致，用条件基类加空基类优化实现。
- **时间戳存在库内，类型为 `uint32_t`**。`uint16_t` 按 10 Hz 帧号不到两小时就回绕，回绕后 `[t_min, t_max]` 的比较全部失效。PCL 点类型的第 4 个 float 是填充，可以零成本放时间戳，但取出的点会带着非 1.0 的 `data[3]`，不适合做默认行为，留作以后的可选优化。
- **开启时的额外存储**：
  - 叶子增加 `uint32_t stamps[LeafSize]`，每点多 4 B；
  - 每个节点增加子树的 `[t_min, t_max]`，每节点多 8 B，摊到每点约 0.7 B。
- **`Op` 日志**增加时间戳字段，以及 touch 和 expire 两种操作；重建期间照常排队，按顺序回放。
- **叶内按点删除或 touch 后**，`[t_min, t_max]` 与 AABB 一起重算，复杂度 O(LeafSize)。
- **`expireBefore(T)`**：相当于在时间维上做 box 删除。
  - `t_max < T`：整棵子树打惰性删除标记，O(1)；
  - `t_min >= T`：整棵子树跳过；
  - 其他情况：继续下降。
  - 物理回收沿用 Phase 1 的删除比例判据。
- **带 `min_stamp` 的查询**：在 `t_max < min_stamp` 的子树处剪枝，叶内逐点比较时间戳。没有惰性刷新标记，不需要沿路径携带额外状态。
- **`touchPoints`**：按坐标精确查找，与按点删除走同一条下降路径。
  - 开销是这一阶段的主要风险：按“配准用到的近邻都刷新”的用法，每帧要刷新约 5 × 帧点数个点，每个点一次按 AABB 的下降，都在独占锁内。
  - 如果达不到 7.3 的上限，再考虑叶子级时间戳（每个叶子一个时间戳，粒度相当于 iVox 的体素级 LRU）。
- **剪枝效果**：取决于时间与空间的相关性。机器人在移动时效果好；原地静止时会退化成逐叶扫描时间戳数组。最坏情况下全图约 170 万次比较，量级在 1 ms，可以接受。

### 7.3 测试与验收

- **正确性**：建立带时间戳的暴力参考模型；随机混合执行 add / touch / expire / delete 后，对比所有查询结果和 `size()`。`validate()` 增加对 `[t_min, t_max]` 的检查。
- **关闭开关时**：benchmark 结果与 Phase 1 的差异在噪声范围内（±5%）。
- **开启开关时**：
  - 内存增量不超过 5 B/点；
  - 在流式地图上，`expireBefore` 与同等删除量的 box 删除耗时处于同一量级；
  - `touchPoints`：benchmark 每帧刷新该帧 5-NN 查询返回的全部近邻，每帧耗时不超过同帧插入耗时的 3 倍。

Phase 3 开工前，执行方按 Phase 1 的实际结构把本节补全（接口签名、`Op` 的布局、提交拆分），再审一次。

## 8. Phase 4：系统级验证（本轮不做，需要用户决定）

- **适配**：编写 FAST-LIO2 适配层，用 likd-tree 替换 ikd-Tree。
- **对比的淘汰策略**：
  - 立方体裁剪（FAST-LIO2 原生）
  - 体素级 LRU（Faster-LIO 的 iVox）
  - 树内 LRU
  - 树内 TTL
- **指标**：ATE/RPE、地图中动态物体的残影点数、内存上界、每帧耗时。
- **对比基线**：除 ikd-Tree 外，还应包括 iVox、i-Octree（ICRA 2024）和 LIO-HKDT。
- **需要用户决定**：使用哪些数据集（应包含有动态物体的序列），以及评测方式。

## 9. 审阅中确定的问题

v1 在这里列了 7 个请审阅方判断的问题，结论如下，细节已并入各节。

| 问题 | 结论 |
|---|---|
| `LeafSize` 默认 32，做成编译期参数 | 同意。通过 `Options` 配置；内存指标把默认值限制在 32 和 64 之间（5.1） |
| 删除采用置删除位、延迟回收，满叶先压缩再分裂 | 同意。最坏内存有界，不需要在写路径上合并叶子（5.1） |
| `MIN_REBUILD_SIZE = 4 * LeafSize`，就地重建阈值 1024 | 同意。就地重建另加每次写调用的总预算（5.8） |
| Phase 2 推迟 | 同意。Phase 1 完成后连同重建延迟指标重新评估（第 6 节） |
| 时间戳存在库内还是由 `Traits` 提供，`uint32_t` 还是 `uint16_t` | 库内，`uint32_t`（7.2） |
| `Node` 改 private，`nodeCount()` 语义调整 | 同意。配上 `memoryUsage()`（5.5） |
| 验收阈值 | 35 B/点保留，“目标 30”改为分场景的阈值；5% 的速度阈值配合交替对比测法（2.3、5.7） |

## 10. 执行流程与记录

- **分支与提交**：在 `improvements` 分支上工作，按 5.9 的粒度提交，不推送到远端。Phase 0 和 Phase 1 已审阅通过；Phase 3 开工前要再审。
- **单元测试**：每次提交前运行：
  ```bash
  cmake -B build && cmake --build build && (cd build && ctest --output-on-failure)
  ```
- **TSan**：涉及并发的改动额外运行。gcc 9 缺少 `libtsan_preinit.o`，只能用 clang++-10：
  ```bash
  clang++-10 -std=c++17 -O1 -g -fsanitize=thread test/unit_tests.cpp \
    -Isrc -I/usr/include/eigen3 -lpthread -o build/unit_tests_tsan
  ./build/unit_tests_tsan
  ```
- **benchmark**：`thirdparty/ikd-Tree` 子模块为空，不要让 CMake 自动拉取，直接用 `../ikd-Tree` 手动编译：
  ```bash
  g++ -std=c++17 -O3 test/benchmark.cpp ../ikd-Tree/ikd-Tree/ikd_Tree.cpp \
    -I../ikd-Tree/ikd-Tree -Isrc -I/usr/include/eigen3 -I/usr/include/pcl-1.10 \
    -lpcl_io -lpcl_common -lboost_system -ltbb -lpthread -o build/benchmark
  ./build/benchmark && ./build/benchmark test/pcd/globalMap.pcd
  ```
- **记录**：每个阶段结束后，把测量结果追加到下面的“结果记录”。

### 决定记录

**2026-10-04，v1 审阅（审阅方）**

审阅方对照 `src/likd_tree.hpp` 和 `test/benchmark.cpp` 核对了 v1。5.3 的不变量与现有并发协议一致，没有发现会导致数据竞争或悬空指针的设计错误。结论：Phase 0、Phase 1 通过，Phase 2 推迟，Phase 3 收窄范围后再审。v2 相对 v1 的改动：

| 改动 | 理由 |
|---|---|
| 内存目标从“约 28 B/点”改为分场景的阈值；默认 `LeafSize` 只在 32 和 64 之间选（5.1、5.7） | 模拟显示流式插入的填充率约 69%，不是 v1 假设的 75%；`LeafSize = 16` 流式为 38 B/点 |
| 增加 `memoryUsage()` 和单独的内存测量轮（第 4 节） | Part 2/3 里两棵树交替插入，`mallinfo()` 的堆增量分不开；`nodeCount()` 分桶后不再反映内存 |
| “3 次取中位数”改为基线与新版本交替运行各 5 次以上；Part 1 查询加到 20 万次（2.3） | 同机波动约 15%，3 次中位数分辨不出 5% 的回退 |
| 增加重建延迟指标（第 4 节、第 6 节） | 陈旧比例不能说明一次写入最久要等多久才可见 |
| 模板参数从 `int LeafSize` 改为 `Options` 类型（5.1、5.5） | Phase 3 还要加开关，避免公开接口变成一长串位置参数 |
| 去掉 `Leaf::count`（5.1） | 与 `Node::size` 重复 |
| 写明空子节点上的插入、分裂与构建的区间划分、按真实类型释放节点、掩码移位（5.2） | v1 没有写，实现时都会遇到 |
| 测试增加 `LeafSize = 64`、`validate()`、ASan + UBSan（5.6） | `uint64_t` 掩码路径原先没有覆盖；整体重写核心结构需要结构级的自检 |
| 就地重建加总预算，并规定在候选过滤之后执行（5.8） | 控制最慢一帧；避免读到已释放的节点 |
| 满叶分裂时丢掉已删除的槽（5.9） | 避免提交 2 和 3 之间出现把已删除点拷进新叶子的中间状态 |
| Phase 3 去掉 `touchBox` 和惰性刷新标记；`touchPoints` 增加耗时上限；时间戳定为库内 `uint32_t`（第 7 节） | 惰性刷新标记最复杂而语义不可靠；`touchPoints` 的开销是主要风险 |

填充率模拟是审阅会话里的临时程序（只做构建、插入和分裂），没有进仓库；Phase 1 完成后以 benchmark 的实测内存为准。

### 结果记录

（执行时追加）
