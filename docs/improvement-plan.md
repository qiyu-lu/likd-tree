# likd-tree 改进计划

- 状态：v2，已审阅。Phase 0、Phase 1 已完成并验收；Phase 2 已关闭；5.10 的修复已完成，验收中有一项需要用户决定（见决定记录）；第 7 节已按 Phase 1 的实际结构补全；两者待审，审阅通过后再开工 Phase 3
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
| 1 | 叶子桶 + 紧凑节点 | 内存约降 3 倍，构建和插入加快 | 中：重写核心数据结构 | 高，已完成 |
| 1.1 | 有序写入时的再平衡（5.10） | 有序写入的耗时不再随点数平方增长 | 低：不改并发协议和不变量 | 审阅时发现，在 Phase 3 之前做 |
| 2 | 重建期间写入立即可见 | 读一致性 | 高：要改并发协议 | 已关闭（第 6 节） |
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
     - 测量轮中，likd-tree 每帧之后调用 `waitForRebuild()`。测量轮帧与帧之间没有查询，不等待的话，写入会在运行中的重建后面积压（Phase 0 实测最多约 130 万个），测到的是队列，而不是树本身（见第 10 节决定记录）。
   - 同时打印 RSS 增量（`/proc/self/statm`）：`uordblks` 不含碎片，而重建会反复申请和释放节点。RSS 只报告，不设阈值。
   - Part 3 结束时按“字节/有效点”报告（分母是 `size()`）。
   - 非 glibc 平台跳过 `mallinfo()` 一项。`mallinfo()` 的字段是 `int`，超过 2 GB 会溢出；本机 glibc 2.31 没有 `mallinfo2()`，现在的数据量（最大约 280 MB）没有问题，代码里加注释说明。
3. **重建延迟指标**。benchmark 打印单次重建的最长耗时，以及重建期间 `pending_ops_` 的最大长度。它们回答“一次写入最久要等多久才对查询可见”，是 Phase 2 是否要做的依据。需要的计数放在库里，用宏 `LIKD_TREE_STATS` 控制，默认不编译。
4. **补齐计时项**。
   - Part 1 的查询从 1000 次改为 20 万次（现在总共只有 1.5–2 ms，测不准）。
   - 增加 `radiusSearch` 和 `boxSearch` 各一行计时，Phase 1 会改这两条路径。
   - Part 2/3 增加“最慢 1% 帧的平均插入耗时”，取帧数的 1%，向上取整，至少 1 帧。单帧最大值的 A/A 噪声超过 1 ms（见第 10 节决定记录）。
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
  - 上表的“一次性构建”用的是中位数划分，填充率随点数在 50%–100% 之间摆动，第二张地图 sparse 上只有 62%（34.5 B/点）。构建规则已改为按目标填充率划分（5.2），模拟中两张地图都是 81%（26.2 B/点）。
  - 空间连续的流式插入（sparse），在不重建时填充率只有 55%（38.4 B/点）。重建会重新打包，实际应当更好，但模拟不含重建。见第 10 节决定记录。
  - **最坏情况**：重建后填充率下限 50%，再叠加删除比例上限 50%，约 75 B/有效点，仍低于现在的最好情况 96 B。所以不需要为稀疏叶子另加合并或按容量触发的重建。
- **`LeafSize` 通过 `Options` 配置**，默认 32：
  - 单元测试用 2、4、32、64 四个实例；
  - benchmark 扫描 8/16/32/64。16 及以下在流式场景过不了 35 B/点，所以**默认值只在 32 和 64 之间选**，8 和 16 只作为速度参照。
- **对 `PointType` 的新要求**：可默认构造、可拷贝赋值。仓库里现有的点类型（PCL、`Eigen::Vector3f`、demo 里的自定义结构体）都满足。

### 5.2 算法

**构建**：按目标填充率递归划分。v2 原定的中位数划分已改掉，见第 10 节决定记录。

- 区间点数 `n = r - l <= LeafSize` 时生成叶子。
- 否则取目标叶子点数 `T = ceil(0.8 * LeafSize)`、叶子数 `L = max(2, ceil(n / T))`，左子树取 `n * (L / 2) / L` 个点（`L / 2` 为整数除法），其余归右子树，即 `m = l + n * (L / 2) / L`。
- 沿包围盒最长边用 `nth_element` 把第 m 个点放到位，`split = coord(pts[m], axis)`，左子树取 `[l, m)`，右子树取 `[m, r)`。现在的代码把 `pts[m]` 放在节点自身、右子树取 `[m+1, r)`，这里要改。
- 叶子填充率约为 80%，模拟中两张地图都是 81%。
- 新建的子树不会立刻满足失衡判据：
  - L 为偶数时，两侧点数相差不超过 1。
  - L 为奇数（`L = 2k + 1`）时，大的一侧占比不超过 `(k + 1) / (2k + 1) + 1 / n`。
  - 最差的 L = 3 时约为 2/3 + 1/n，只出现在 `n <= 3T` 时。而 `3T <= 2.4 * LeafSize + 3 < 4 * LeafSize = MIN_REBUILD_SIZE`，这样的子树不参与失衡判断。
  - L ≥ 5 时占比不超过 3/5 + 1/n < α = 0.75。
- TBB 并行构建方式不变。

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

验收用两张地图：`test/pcd/globalMap.pcd`（174 万点）和 `test/pcd/Global_map_sprase.pcd`（129 万点，下称 sparse，文件名拼写如此）。

- **内存**（`LeafSize = 32`）：
  - Part 1 建完树后：两张地图都不超过 28 B/点；
  - Part 2 流式插入结束后：globalMap 不超过 35 B/点，sparse 不超过 40 B/点。sparse 实测超过 35 B/点时，报告叶子填充率的分布，不自行改分裂策略；
  - Part 3 结束时：报告字节/有效点，不超过 75 B（5.1 的最坏情况上界）。
- **速度**：两张地图上，以下各项回退都不超过 5%：
  - 构建。随机点的构建只有约 9 ms，A/A 噪声 5.2%，只报告，不参与验收；
  - Part 2/3 的插入总计；
  - 1-NN 和 5-NN（顺序与 TBB）；
  - `radiusSearch`、`boxSearch`。
- **插入的尾部耗时**：两张地图都适用。它取代原来的“最慢一帧不超过基线 + 1 ms”，那一条作废。
  - 最慢 1% 帧的平均插入耗时：
    - Part 2 回退不超过 10%；
    - Part 3 回退不超过 10% 或 0.3 ms，取较宽者；
    - 随机点只报告，不参与验收（99 帧的 1% 就是单帧最大值）。
  - 单帧最大值照常报告，任何一次运行都不超过基线各次运行中的最大值 + 2 ms。
- **RSS**：只报告，不设阈值。Part 3 的 RSS 也只报告，不再追查。
- **Part 3 以 sparse 为准**：globalMap 按文件顺序每帧 2000 点，一帧的水平跨度中位数是 403 m，几乎覆盖全图，它的 Part 3 不代表真实的局部地图场景。
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
6. 验收完成之后：benchmark 增加 `--max-points` 和 `--frame` 两个选项，用 `Global_map_dense.pcd`（4719 万点）的前 500 万点、每帧 2 万点跑一次，只报告结果，不参与验收。整张 dense 地图不能用：两棵树同时建会超过本机可用内存，`mallinfo()` 的计数超过 2 GB 也会溢出。

### 5.10 Phase 1 之后的修复：有序写入时的再平衡

审阅方在 Phase 1 验收时发现的性能缺陷，在 Phase 3 之前作为单独的一步修复。修法由审阅方给定。

**问题**：按坐标有序的点连续写入时，树退化成长链，耗时随点数平方增长。

- **原因**：每个新点都落进同一个（最右边的）叶子。叶子满了就对半分裂，链上多出一个内部节点，下一个点要走过整条链。两次重建之间没有再平衡：
  - 一次写调用内部不会重建：候选要等整个调用结束才交给 worker；
  - worker 回放 `pending_ops_` 时也不会：回放完整个队列才重建这期间产生的候选，而回放期间新到的写入又排在后面，下一轮回放同样长。
- **审阅方实测**（`struct {float x, y, z}`，`x = i * 0.01`，其余为 0，默认 `Options`）：
  - 空树上一次 `addPoints`：10 万点 3.6 s，40 万点 64 s；
  - 每帧 2000 点连续 `addPoints`、不等待：20 万点 8.1 s，40 万点 53 s；只有 3 轮重建，最多积压 36.4 万个写入；
  - Phase 1 之前的代码同样有这个问题（一次调用 10 万点 57 s），不是 Phase 1 引入的。
- 执行方在本机的复现见“结果记录”。

**修法**（不改并发协议，不改 5.3 的不变量）：

1. **写调用分块**：`write()` 把一次调用的操作切成每块 `WRITE_CHUNK_SIZE = 2000` 个。每块在独占锁内做完后，如果出现了重建候选，就把候选交给 worker，并把这次调用剩下的操作按原顺序放进 `pending_ops_`。
2. **回放中断**：worker 每回放完一个 `INSERTION_BATCH_SIZE` 批次（100 个操作），如果已经出现候选，就停止回放，先做这一轮重建，再接着回放。没回放的操作排在这期间新排队的写入之前。
3. **语义不变**：写入顺序保持不变，`wait_for_rebuild` 和 `waitForRebuild()` 的语义不变。

实现上的两点说明：

- **没回放的操作不拷回 `pending_ops_`**。它们留在 worker 取出的那份日志里（下标 `replayed` 之后），下一轮先接着回放它们，回放完才从 `pending_ops_` 取新的写入，逻辑上就是排在队列最前面。拷回 `pending_ops_` 的头部，每一轮都要搬动整个积压（40 万个操作约 25 MB）。worker 只在日志回放完、且 `pending_ops_` 为空时退出重建状态。
- **一次写调用返回时可能只有一部分可见**：超过 2000 个操作的调用，在某一块之后出现候选时，剩下的操作在队列里。这是第 1 点的直接结果，README 的线程安全说明已补充。

**5.3 的不变量**：五条都不变。

- **不变量 1**：分块只改变写线程何时把候选交出去。写线程仍然只追加、压缩、分裂、置删除位，不释放节点。
- **不变量 2**：每块在独占锁内完成；块与块之间释放锁，查询可以穿插进来。
- **不变量 3**：写线程交出候选时，在同一个 `pending_mutex_` 临界区内置 `rebuilding_`、交出候选、把剩下的操作入队；此后直到 worker 退出重建状态，写线程只入队。worker 重建（无锁读取旧子树）期间自己也不回放。
- **不变量 4、5**：不涉及。
- **候选指针的有效性**：每段回放结束时，候选在独占锁内过滤；此后到重建开始之前，树上没有任何写入（写线程被分流到队列，worker 自己直接去重建）。

**统计口径**（`LIKD_TREE_STATS`）：

- **最多排队写入**：也计入 worker 为了先重建而暂时留在手里的写入（新增计数 `held_ops_`），否则回放被中断之后，积压会被低估。
- **排队写入最长等待**：从 worker 一次取出的那批写入中最早的一个入队，到这批全部回放完为止，中间穿插的重建也计入。修复前回放不会中断，这批写入是一次回放完的，口径相同。

**验收标准**（审阅方给定）：

- **回归测试**：上面两种场景各 40 万个有序点，以及 40 万个完全相同的点，结果正确（`size()`、`validate()`、抽查 kNN）。耗时各不超过 2 s；测试里的断言放宽到 10 s，避免机器繁忙时误报，实测值写进结果记录。
- **benchmark**：按 5.7 的测法在 globalMap 和 sparse 上对比修复前后。各项回退不超过 5%，最慢 1% 帧按 5.7 的原标准，内存不变；同时报告重建轮数、最多排队写入和排队写入最长等待的变化。
- **检查**：单元测试、TSan、ASan + UBSan 全部通过。Phase 1 压缩那次提交（`bfc6204`）没有重跑 TBB 版的 ASan + UBSan，这次一起补上。
- 某项达不到，或者修法需要改动不变量时，停下来报告。

### 5.11 Phase 3 之前：查询选项与不排序的 radiusSearch

用户在 5.10 验收后安排的单独一步，设计由用户给定。

**动机**：`radiusSearch` 把结果按距离排序后返回，每次返回的点多时，排序占了大头。执行方的临时程序在 globalMap 上测得（2 万次查询，单线程）：

| 每次返回的点数 | 排序（现在） | 不排序 | ikd-tree（不排序） |
|---|---|---|---|
| 82（r = 1 m） | 108 ms | 55 ms | 210 ms |
| 2625（r = 5 m） | 2679 ms | 438 ms | 1297 ms |

PCL（`setSortedResults`）和 nanoflann（`SearchParameters::sorted`）都提供同样的开关。

**设计**：

```cpp
  // Options for the queries that take them. Each field applies to some
  // queries only and is ignored by the others.
  struct SearchOptions {
    float max_dist = INFINITY;  // knnSearch: skip points farther than this
    bool sorted = true;         // radiusSearch: sort the results by distance
  };
```

- **嵌套在 `KDTree` 里**：Phase 3 的 `min_stamp` 只在 `TRACK_STAMPS` 为真时存在（7.4），这取决于 `Options`。
- **四类查询的六个公开函数各加一个接受 `const SearchOptions&` 的重载**，现有签名保留，内部转调，已有代码不用改。
  - `nearestNeighbors` 和 `knnSearch` 各有单个和批量两个版本，批量版本也加：按 7.9 第 6 条的决定，Phase 3 的 `min_stamp` 要通过 `SearchOptions` 传给批量查询。
  - 新重载的 `SearchOptions` 参数没有默认值，否则 `knnSearch(q, k, res, d)` 会与现有签名二义。
- **`sorted = false`**：`radiusSearch` 跳过排序，按遍历顺序返回，距离照常给出。
- **字段按名字设置**（`opts.sorted = false;`）。Phase 3 会给 `SearchOptions` 加一个条件基类，按位置的聚合初始化（`SearchOptions{2.0f, false}`）到时会失效，README 里只给按名字设置的写法。

**测试**：不排序的结果与排序的结果是同一个集合（按坐标排序后逐个比较，距离与点对应）；其余查询带 `SearchOptions` 的结果与原签名相同。

**benchmark**：Part 1 在 radius 一行之后增加一行不排序的 radius 查询，同样 2 万次，与 ikd-tree 的同一次计时对比；两者找到的点数必须相同。

**验收**：按 5.7 的测法（交替各 10 次，随机点、globalMap、sparse），现有各项回退不超过 5%。

**提交**：

1. `Add SearchOptions and an unsorted radius search`：库、测试、README。
2. `Benchmark the unsorted radius search`。

## 6. Phase 2：重建期间写入立即可见（已关闭）

**已关闭，不做**（2026-10-04，用户转达的审阅决定）。Phase 1 之后，排队写入的最长等待从约 50–70 ms 降到约 6 ms（globalMap 6.5 ms，sparse 6.2 ms，约为 10 Hz 下一帧的 6%），读到旧地图的比例 globalMap 为 0.00%，sparse 为 0.03%。下面的快照加日志回放方案会明显增加并发协议的复杂度，已经不值得做。原来的分析保留在下面供以后参考；5.10 的修复对这几项指标的影响见“结果记录”。

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

**v2 的结论（已由本节开头的决定取代）**：推迟。Phase 1 完成后重新测量陈旧比例，连同 Phase 0 新增的两个指标（单次重建最长耗时、`pending_ops_` 最大长度）一起报告，再决定是否要做。陈旧比例只说明有多少查询读到旧地图；对 LIO 更要紧的是一次写入最久要等多久才可见，根级重建时这个值可能达到一帧的量级。

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
- **开启时的额外存储**（按 Phase 1 实测的填充率重新估算，见 7.5）：
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
  - 按 7.9 第 2 条的决定：先对输入去重；不做共同下降，也不做句柄。叶子的 `[t_min, t_max]` 没有变化时，不更新祖先。
- **剪枝效果**：取决于时间与空间的相关性。机器人在移动时效果好；原地静止时会退化成逐叶扫描时间戳数组。最坏情况下全图约 170 万次比较，量级在 1 ms，可以接受。

### 7.3 测试与验收

- **正确性**：建立带时间戳的暴力参考模型；随机混合执行 add / touch / expire / delete 后，对比所有查询结果和 `size()`。`validate()` 增加对 `[t_min, t_max]` 的检查。
- **关闭开关时**：benchmark 结果与 Phase 1 的差异在噪声范围内（±5%）。
- **开启开关时**：
  - 内存增量不超过 6.5 B/点（原为 5 B/点，按 7.9 第 1 条的决定修改）；
  - 在流式地图上，`expireBefore` 与同等删除量的 box 删除耗时处于同一量级；
  - `touchPoints`：Part 2 每帧刷新该帧 5-NN 查询返回的全部近邻，每帧耗时的最慢 1% 平均不超过 5 ms，globalMap 和 sparse 都要满足；与同帧插入耗时之比只报告，不作标准（原为“不超过同帧插入耗时的 3 倍”，按 7.9 第 2 条的决定修改）。

Phase 3 开工前，执行方按 Phase 1 的实际结构把本节补全（接口签名、`Op` 的布局、提交拆分），再审一次。补全的内容见 7.4–7.9（执行方，2026-10-04），其中 7.9 列出需要审阅方决定的问题。

### 7.4 接口签名

按 7.9 的决定同步（2026-10-04）。

开关和时间戳类型：

```cpp
struct DefaultOptions {
  static constexpr int LEAF_SIZE = 32;
  // Keep a stamp per point. Off: layout and speed as without the option.
  static constexpr bool TRACK_STAMPS = false;
};

template <typename PointType, typename Traits, typename Options>
class KDTree {
 public:
  // A frame number or a time, in any unit that never decreases or wraps.
  // The largest value, kNever, means the point never expires: writes
  // without a stamp use it.
  struct Stamp {
    static constexpr uint32_t kNever = UINT32_MAX;
    uint32_t value = 0;
  };
  ...
};
```

- **`Stamp` 是独立类型，不直接用 `uint32_t`**（第 3 条决定）。`addPoints(pts, wait_for_rebuild)` 的第二个参数是 `bool`。如果时间戳是整数，`addPoints(pts, frame)` 在 `frame` 为 `int` 时与 `bool` 重载二义，`frame` 为 `bool` 时会静默选中不带时间戳的重载。独立类型让两者在编译期分开。
- **嵌套在 `KDTree` 里**，与 `AABB` 一致，不往全局命名空间里加 `Stamp` 这样的通用名字。
- **最大值 `Stamp::kNever` 表示永不过期**（第 4 条决定）：`expireBefore` 只删除时间戳小于阈值的点，带 `min_stamp` 的查询只跳过时间戳小于 `min_stamp` 的点，所以带这个值的点永远不会过期，也总能被查到。

只在 `TRACK_STAMPS` 为真时可用的写接口。关闭开关时调用它们是编译错误：函数体里有 `static_assert`，类模板的成员只在被调用时才实例化。

```cpp
  void build(const PointVector<PointType>& pts, Stamp stamp);
  void addPoints(const PointVector<PointType>& pts, Stamp stamp,
                 bool wait_for_rebuild = false);
  // Raises the stamp of every stored copy of each point (exact coordinates)
  // to `stamp`; copies with a newer stamp keep it. Duplicate points in
  // `pts` are dropped first.
  void touchPoints(const PointVector<PointType>& pts, Stamp stamp,
                   bool wait_for_rebuild = false);
  // Deletes every point whose stamp is older than `stamp`.
  void expireBefore(Stamp stamp, bool wait_for_rebuild = false);
```

带 `min_stamp` 的查询（第 6 条决定）：不加单独的重载。`min_stamp` 是 `SearchOptions`（5.11）的字段，只在开启开关时存在，关闭时使用它是编译错误；通过各查询接受 `SearchOptions` 的重载传入。时间戳早于 `min_stamp` 的点视为不存在：

```cpp
  struct NoStampFilter {};
  struct StampFilter {
    Stamp min_stamp;  // skip points stamped earlier; the default skips none
  };
  struct SearchOptions
      : std::conditional_t<TRACK_STAMPS, StampFilter, NoStampFilter> {
    float max_dist = INFINITY;  // knnSearch
    bool sorted = true;         // radiusSearch
  };
```

已有接口在开启开关时的语义：

- **不带时间戳的 `build(pts)` 和 `addPoints(pts, wait)`**：点的时间戳取 `Stamp::kNever`，永不过期，对应 Redis 中没有设置 TTL 的键（第 4 条决定）。这样已有的全部单元测试可以原样在开启开关的实例上运行。
- **`deletePoints`**：开启时删除时间戳最旧的那份拷贝；关闭时照旧删除先找到的一份（第 5 条决定）。有重复点时，删掉哪一份会影响以后的过期，规定为最旧的一份，暴力模型才能精确对照。
- **`touchPoints`** 先对输入按坐标去重，再逐点查找（第 2 条决定）。
- **不带 `min_stamp` 的查询**行为不变。`size()` 仍是未删除的点数，与时间戳无关。

内部的查询函数和 `collect` 增加 `min_stamp` 参数。关闭开关时它恒为 0，相关判断用 `if constexpr` 去掉；开启开关但 `min_stamp` 为 0 时，查询走不检查时间戳的那份实例。

### 7.5 数据布局

节点与叶子。`TRACK_STAMPS` 为假时，空基类经空基类优化不占空间，布局与 Phase 1 逐字节相同：

```cpp
struct NoStampBounds {};
struct StampBounds {  // stamps of the non-deleted points below; [max, 0] if none
  uint32_t t_min = UINT32_MAX;
  uint32_t t_max = 0;
};
struct Node : std::conditional_t<TRACK_STAMPS, StampBounds, NoStampBounds> {
  ...  // the Phase 1 fields, unchanged
};
struct Inner : Node { ... };                                     // unchanged
struct Leaf : Node { Mask deleted; PointType pts[LeafSize]; };  // unchanged
struct StampedLeaf : Leaf {  // the leaf allocated with TRACK_STAMPS
  uint32_t stamps[LeafSize];  // stamps[i] belongs to pts[i]
};
```

- **时间戳数组放在 `pts` 之后**（派生类 `StampedLeaf`），叶子头、`deleted` 和 `pts` 的偏移与 Phase 1 相同，不带 `min_stamp` 的查询不会多读一条 cache line。若放进基类，它会插在 `deleted` 和 `pts` 之间。
- **开启时按 `StampedLeaf` 分配和释放**。`destroy` 按真实类型 `delete`，与 5.2 的要求一致。
- **多个空基类用不同的类型**，否则同类型的两个空子对象不能共用地址，空基类优化会失效。

`pcl::PointXYZ`、`LeafSize = 32` 时的大小：

| | Phase 1 | 开启 `TRACK_STAMPS` |
|---|---|---|
| `Node` | 48 B | 56 B |
| `Inner` | 72 B（malloc 块 80 B） | 80 B（块 96 B） |
| `Leaf` | 576 B（块 592 B） | 704 B（块 720 B；`Node` 增加的 8 B 落在 `pts` 前原有的对齐空隙里） |

按 Phase 1 实测的结构估算堆内存增量。`memoryUsage()` 约为 648 B × 叶子数（每个叶子 576 B，加上数量相近的内部节点 72 B），由每点的 `memoryUsage()` 反推每叶点数 p；增量约为 (128 + 16) / p：

| 场景（取自 `phase1-final`） | 每点 `memoryUsage()` | 每叶点数 p | 堆增量估算 |
|---|---|---|---|
| Part 1 建树，两张地图 | 24.31 B | 26.7 | 5.4 B/点 |
| Part 2 流式，globalMap | 27.06 B | 23.9 | 6.0 B/点 |
| Part 2 流式，sparse | 24.57 B | 26.4 | 5.5 B/点 |
| Part 3，globalMap / sparse | 25.07 / 24.92 B | 25.8 / 26.0 | 5.6 / 5.5 B/点 |

7.2 的估算是“每点 4 B 加每点约 0.7 B”，没有计入叶子填充率：时间戳按槽位分配，摊到每个点是 4 B ÷ 填充率。由此 7.3 的“不超过 5 B/点”按现在的布局达不到，见 7.9 第 1 条。

`Op` 的布局：

```cpp
enum class OpType : uint8_t { kInsert, kDeletePoint, kDeleteBox, kTouch, kExpire };
struct NoOpStamp {};
struct OpStamp {
  uint32_t stamp;  // insert, touch: the stamp; expire: the cutoff
};
struct Op : std::conditional_t<TRACK_STAMPS, OpStamp, NoOpStamp> {
  OpType type;
  PointType point;  // insert, delete point, touch
  AABB box;         // delete box
};
```

- 关闭开关时 `Op` 的大小不变：`pcl::PointXYZ` 64 B，三个 float 的点 40 B。
- 开启时 `pcl::PointXYZ` 仍是 64 B（时间戳和类型落在点之前的对齐空隙里），三个 float 的点变为 44 B。
- `point` 和 `box` 不合并成 union：队列不是内存的主要部分。

重建与分裂时携带时间戳：

- 开启时，`collect` 和 `buildRecursive` 操作 `struct Stamped { PointType pt; uint32_t stamp; }` 数组（`pcl::PointXYZ` 时每项 32 B），`nth_element` 连同时间戳一起移动。关闭时仍操作 `PointVector<PointType>`，代码路径不变。
- `makeRoom` 的压缩和分裂同理。

### 7.6 惰性标记与时间戳的交互

Phase 1 的惰性标记只有一种，即整树删除的 `tree_deleted`；v2 去掉了 `touchBox`，刷新没有惰性形式。规则如下：

1. **`[t_min, t_max]` 与 `aabb` 同语义**：只覆盖未删除的点，`valid == 0` 时为空区间 `[max, 0]`。`updateLeaf`、`updateInner`、`killSubtree` 在更新 `aabb` 的同一处更新它。
2. **惰性删除标记之下的时间戳是陈旧的，但不会被读到**：
   - 查询、`touchPoints`、`deletePoints`、`expireBefore` 和 `collect` 都在 `valid == 0` 处停下，到不了标记之下；
   - 插入在下降前照旧 `pushDown`，下推时 `killSubtree` 把子节点的 `[t_min, t_max]` 连同 `aabb` 一起置空；
   - `validate()` 对带标记的子树只检查标记节点自身（`valid == 0`，包围盒和时间戳范围都为空），与 5.6 的规则一致。
3. **`expireBefore` 复用同一个标记，不新增标记类型**：`t_max < T` 的子树直接 `killSubtree`（内部节点打标记，叶子把所有槽置删除位），O(1)；之后照常 `markIfUnbalanced`。删除比例超过一半的子树由重建回收，与 box 删除完全相同。
4. **touch 只会提高时间戳**：
   - 叶内改完后重算叶子的 `[t_min, t_max]`；被刷新的点若恰好是最小值，`t_min` 会上升。
   - 祖先沿递归路径更新 `[t_min, t_max]`，O(深度)。touch 不改变 `aabb`、`size` 和 `valid`，只需要合并子节点的时间戳范围。
   - 叶子的 `[t_min, t_max]` 没有变化时，不更新祖先（7.9 第 2 条允许的优化）；同理，某个祖先的范围没有变化时，更上层的祖先也不必更新。
5. **重建收集时带着时间戳**：worker 无锁读取旧子树的时间戳，依据与 5.3 的不变量 3 相同（重建期间的写入都进队列）。
6. **队列的顺序决定语义**：touch 和 expire 在重建期间照常排队，按调用顺序回放；5.10 的分块与回放中断对它们同样适用。例如，先排队的 touch 能让点躲过随后排队的 expire，顺序反过来就不能。这类顺序由 7.3 的差分测试覆盖，测试中写操作不等待重建，大量操作会经过队列。
7. **失衡判据不变**：时间戳不影响平衡，expire 只通过删除比例影响重建。

### 7.7 提交拆分

1. **`Add a TRACK_STAMPS option`**：
   - 内容：`Options::TRACK_STAMPS`、`Stamp`、条件基类、`StampedLeaf`、带时间戳的 `build` 和 `addPoints`、`Op` 的时间戳字段；分裂、压缩、重建都携带时间戳；`validate()` 检查 `[t_min, t_max]`。
   - 测试：已有的暴力对比测试在 `TRACK_STAMPS = true` 的实例上再跑一遍（不带时间戳的写入永不过期），确认行为不变。
2. **`Filter queries by min_stamp`**：
   - 内容：`SearchOptions` 增加 `min_stamp`（条件基类，只在开启开关时存在）；在 `t_max < min_stamp` 的子树处剪枝，叶内逐点比较；开启开关时 `deletePoints` 删除最旧的一份。
   - 测试：带时间戳的暴力模型，随机插入、按点删除、box 删除后，对比带 `min_stamp` 的四类查询。
   - 先做只读的查询，后面两个提交的测试就能通过查询观察时间戳。
3. **`Add touchPoints and expireBefore`**：
   - 内容：两种新的 `Op`、写路径、回放。
   - 测试：完整的暴力参考模型。插入、touch、expire、按点删除、box 删除随机混合，不等待重建；对比查询结果和 `size()`，每次写操作并等待重建后调用 `validate()`。
4. **`Benchmark stamps, touch and expire`**：见 7.8。
5. **README**。

### 7.8 benchmark 方案

- **开关**：编译选项 `-DLIKD_BENCH_STAMPS`，likd-tree 改用开启 `TRACK_STAMPS` 的实例，做法与 `LIKD_BENCH_LEAF_SIZE` 相同。时间戳取帧号，Part 1 的建树用 0。
- **关闭开关时**（7.3 第 2 项）：默认编译的 benchmark 与 5.10 修复后的版本交替各 10 次，两张地图，5.7 的各项回退不超过 5%。
- **开启开关本身的代价**：同一份代码，开关打开与关闭交替对比，不使用 touch、expire 和 `min_stamp`。
  - Part 1/2/3 的堆内存增量，对照 7.3 第 3 项（标准见 7.9 第 1 条）；
  - 构建、插入、四类查询的耗时，只报告。
- **touch**：Part 2 中每帧 5-NN 查询之后，调用 `touchPoints(本帧查询返回的全部近邻, 帧号)`。
  - 报告每帧 touch 耗时（总计、最慢 1% 帧）、与同帧插入耗时之比，以及近邻去重后剩下的比例。
  - 标准：每帧耗时的最慢 1% 平均不超过 5 ms，globalMap 和 sparse 都要满足；与同帧插入耗时之比只报告（7.9 第 2 条的决定）。
- **expire**：新增 Part 4，帧与 Part 3 相同，但不做 box 删除，改为每 10 帧 `expireBefore(帧号 - W)`。
  - W 取使保留点数与 Part 3 相近的值，在 sparse 上先跑一次确定。
  - 报告 expire 总耗时和平均每删除一个点的耗时，与 Part 3 的 box 删除对比（7.3：处于同一量级）。
  - 以 sparse 为准，理由与 5.7 相同。
- **`min_stamp` 查询**：Part 4 中另跑一组 `min_stamp = 帧号 - W / 2` 的 5-NN，报告耗时，以及结果与不带 `min_stamp` 时不同的比例。
- ikd-tree 没有对应功能，这几项只报告 likd-tree。

### 7.9 需要审阅方决定的问题

以下各条已由用户在 2026-10-04 决定，见决定记录；正文已同步。保留原来的分析，供以后参考。

1. **内存标准**：7.3 的“不超过 5 B/点”按每槽一个 `uint32_t` 的方案达不到。7.5 的估算是 5.4–6.0 B/点：时间戳按槽位分配，摊到每个点是 4 B ÷ 填充率（Phase 1 实测每叶 24–27 点，填充率 75–83%）；内部节点的 malloc 块还要从 80 B 变为 96 B。可选做法：
   - **(a) 标准改为 6.5 B/点**：按每槽 4 B、填充率不低于 70%，再加内部节点的增量估算。语义精确，实现最简单。执行方建议采用。
   - **(b) 叶内存 16 位的相对时间戳，每叶另存一个 32 位基准**，约 2.4–2.7 B/点。
     - 约束：同一叶子里未删除点的时间戳跨度不能超过 65535 个单位。按 10 Hz 帧号约 1.8 小时，以毫秒为单位只有 65 秒。
     - 超出时只能把最旧的时间戳抬高，等于推迟它们的过期。
     - 插入、分裂、重建都要处理基准的调整。
   - **(c) 只存叶子级时间戳**（7.2 的后备方案），约 0.6 B/点。粒度是整个叶子：touch 一个点，整个叶子一起续期。
   - 重排成员（把 `split`、`axis` 移进 `Node` 的对齐空隙）可以让 `Inner` 保持 72 B，增量降到 4.8–5.4 B/点。但这仍超过 5 B/点，而且会改变关闭开关时的布局，不建议。
   - **决定**：采用 (a)，标准改为不超过 6.5 B/点。
2. **touch 的耗时标准**：执行方用一个临时程序（没有进仓库）估算了 touch 的开销。
   - 做法：按 7.4 的语义查找每个点的全部拷贝，刷新所在的叶子和沿途的祖先，只是不写时间戳。
   - 流程：按 Part 2 的方式，每帧先做 5-NN 查询，再 touch 本帧查询返回的全部近邻，最后插入本帧。touch 前先等待重建结束，所以插入总走直接写入的路径。
   - 结果（与 5.10 修复后的库一起编译，单次运行）：

     | 地图 | 每帧近邻数 | 去重后剩下 | touch 全部 / 插入 | 先去重再 touch / 插入 |
     |---|---|---|---|---|
     | globalMap | 9993 | 87.6% | 3.47 倍（每帧之比的中位数 3.56） | 3.53 倍 |
     | sparse | 9988 | 3.8% | 3.34 倍（3.36） | 0.93 倍 |

   - 逐点下降的 touch 约为插入的 3.3–3.5 倍，略超过 3 倍的标准。
   - 去重的效果取决于近邻的重复程度。sparse 的一帧集中在约 31 m 内，近邻高度重复；globalMap 的一帧散布全图，几乎不重复。两张图都不是真实的扫描序列，真实的 LIO 扫描每帧都覆盖已建好的区域，重复程度介于两者之间。
   - 执行方的计划：`touchPoints` 从一开始就先对输入去重（排序加 unique）。如果仍超过 3 倍，按代价从小到大再考虑：
     - 对排序后的整批点做一次共同下降，落在同一棵子树里的点一起走，减少重复的路径和 cache miss；
     - 由查询返回叶子和槽位的句柄，touch 按句柄直接更新，不再下降。句柄要在两次加锁之间保持有效，会改变并发协议，需要单独审阅。
   - 请审阅方决定 3 倍的标准是否保留。保留的话，大概率需要上面的共同下降。
   - **决定**：标准改为绝对值，每帧耗时的最慢 1% 平均不超过 5 ms（两张地图）；先去重，不做共同下降和句柄；与插入之比只报告；叶子的 `[t_min, t_max]` 不变时可以不更新祖先。
3. **`Stamp` 用独立类型还是直接用 `uint32_t`**：执行方建议独立类型，理由见 7.4。**决定**：独立类型。
4. **开启开关时不带时间戳的写入**：执行方建议视为永不过期（7.4），这样已有测试可以原样复用。更严格的做法是开启时禁止不带时间戳的写入（编译错误），但已有测试就不能直接复用了。**决定**：永不过期，文档写明 `Stamp` 的最大值有这个含义。
5. **有重复点时 `deletePoints` 删最旧的一份**（7.4）：开启开关时要看完所有包含该点的子树，比现在多一点开销，换来确定的语义。**决定**：开启时删最旧的一份，关闭时行为不变。
6. **查询参数的形式**：7.4 给每个查询加一个带 `Stamp` 的重载，共六个。
   - 如果以后还要给 `radiusSearch` 加“不排序”选项，参数会继续增加。
   - 另一种做法是像 nanoflann 的 `SearchParameters` 那样，引入一个查询选项结构体（`max_dist`、`min_stamp`、`sorted`），一次把重载定下来。
   - 建议与“不排序”选项一起决定。
   - **决定**：不加六个带 `Stamp` 的重载；`min_stamp` 作为 `SearchOptions`（5.11）的字段，只在开启开关时存在。

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

- **分支与提交**：在 `improvements` 分支上工作，按 5.9 的粒度提交，不推送到远端。Phase 0 和 Phase 1 已审阅通过，Phase 2 已关闭；5.10 的修复和第 7 节的补全待审；Phase 3 开工前要再审。
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
- **ASan + UBSan**：顺序版和 TBB 版各一次。TBB 2020 自身在退出时留下 3 块共 6168 B 的分配，TBB 版需要抑制它：
  ```bash
  g++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
    -fno-omit-frame-pointer test/unit_tests.cpp -Isrc -I/usr/include/eigen3 \
    -lpthread -o build/unit_tests_asan
  ./build/unit_tests_asan
  # TBB 版：编译时加 -DLIKD_TREE_USE_TBB -ltbb，运行时
  echo "leak:libtbb.so" > build/lsan_tbb.supp
  LSAN_OPTIONS=suppressions=build/lsan_tbb.supp ./build/unit_tests_asan_tbb
  ```
- **benchmark**：`thirdparty/ikd-Tree` 子模块为空，不要让 CMake 自动拉取，直接用 `../ikd-Tree` 手动编译：
  ```bash
  g++ -std=c++17 -O3 test/benchmark.cpp ../ikd-Tree/ikd-Tree/ikd_Tree.cpp \
    -I../ikd-Tree/ikd-Tree -Isrc -I/usr/include/eigen3 -I/usr/include/pcl-1.10 \
    -lpcl_io -lpcl_common -lboost_system -ltbb -lpthread -o build/benchmark
  ./build/benchmark && ./build/benchmark test/pcd/globalMap.pcd
  ./build/benchmark test/pcd/Global_map_sprase.pcd
  ```
- **交替对比**：
  ```bash
  python3 test/compare_benchmarks.py build/benchmark_base build/benchmark \
    --map test/pcd/globalMap.pcd --map test/pcd/Global_map_sprase.pcd --runs 6
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

**2026-10-04，Phase 0 执行中的补充（执行方，未改变方案）**

| 补充 | 理由 |
|---|---|
| 重建统计除计划中的两项外，多记一项“排队写入的最长等待”（`max_queued_ms`，从写入入队到回放它的那一批结束） | 计划中的两项只能间接推断一次写入最久要等多久才可见，这一项直接测量 |
| 新增 CMake 目标 `unit_tests_stats`（开启 `LIKD_TREE_STATS`）和测试 `testRebuildStats` | 统计代码默认不编译，没有这个目标它就不会被 ctest 编译和运行 |
| 新增 `test/compare_benchmarks.py`，README 说明用法 | 实现 2.3 的交替对比测法：交替运行、取中位数，并给出 ikd-tree 的 new/base 作为机器漂移的对照 |
| Part 1 的 radius 和 box 查询各用前 2 万个查询点；半径和 box 半边长在地图上取 1 m，随机点取 10 m | 计划没有规定。这样每次查询平均命中 50–122 个点，总耗时 50–140 ms，便于计时 |
| RSS 取基准前先调用 `malloc_trim(0)` | 把空闲页还给系统，使 RSS 增长只来自新的分配 |
| ikd-tree 的内存测量轮在取样前等待 1 s | ikd-tree 没有等待其重建线程结束的接口 |
| box 查询结果数只在 likd-tree 比 ikd-tree 少时告警 | ikd-tree 的 box 是半开区间 [min, max)，落在 max 面上的点只有 likd-tree 会返回 |

**2026-10-04，Phase 0 完成后需要用户决定的问题（执行方）**

详细数据见下面“结果记录”。以下几点涉及验收标准或测量方式，执行方没有自行处理：

1. **“最慢一帧插入不超过基线 + 1 ms”在当前测法下分辨不出。** A/A 测试中，同一个可执行文件在地图 Part 2 上两组的中位数分别是 6.04 ms 和 7.12 ms，相差 1.08 ms，已经超过阈值。可选做法：
   - 只对这一项增加运行次数；
   - 改用对单帧尖峰不那么敏感的统计量，例如每次运行最慢 1% 帧的平均；
   - 保留标准，但结论改为“差异不超过 A/A 噪声”。
2. **随机点 Part 1 的构建只有约 9 ms，A/A 差 5.2%，超过 5% 阈值本身。** 地图上的构建 A/A 差 1.0%，可以分辨。建议构建以地图为准，随机点只看量级。
3. **内存测量轮的 RSS 反映的是写入队列的峰值，而不是碎片**（见观察 3）。验收用的堆增量是在 `waitForRebuild()` 之后取的，不受影响。是否把测量轮改成每帧之后 `waitForRebuild()`，或在帧间加入与计时轮相同的查询？这会改变 4.2 规定的测量方式，所以没有自行改。
4. **写入队列没有上界**（观察 3、4）。这是现有设计的特性，与 Phase 2 有关，建议作为 Phase 2 评估的依据，本轮不处理。

**2026-10-04，Phase 0 验收（用户转达的审阅结论）**

Phase 0 验收通过。上面四个问题的决定如下，已同步修改第 4 节第 2、4 项和 5.7：

| 决定 | 理由 |
|---|---|
| benchmark 增加“最慢 1% 帧的平均插入耗时”。5.7 改为：它的回退不超过 10%；单帧最大值照常报告，任何一次运行都不超过基线各次运行中的最大值 + 2 ms。原来的“不超过基线 + 1 ms”作废 | 同一个可执行文件在 A/A 中，最慢一帧的两组中位数相差 1.08 ms，原标准落在噪声之内 |
| 构建速度以 globalMap 为准，随机点的构建只报告，不参与验收 | 随机点构建约 9 ms，A/A 噪声 5.2%，大于 5% 的阈值 |
| 内存测量轮改为每帧之后调用 `waitForRebuild()`；RSS 只报告，不设阈值 | 不等待时，测到的是写入队列的积压，而不是树本身 |
| 写入队列无上限，本轮不处理，留给 Phase 2 评估；README 的线程安全说明补充这一已知限制 | 属于 Phase 2 的范围 |

接下来的步骤：

- 改完后重新编译基线，再跑一轮 A/A，确认两件事：最慢 1% 统计量的噪声明显小于 10%，以及新测法下测量轮的 RSS。
- 如果噪声接近或超过 10%，停下来报告；否则直接开始 Phase 1。
- Phase 1 的报告除 5.7 各项外，还要单独列出 Part 2 的重建轮数、最多排队写入、排队写入最长等待，与基线对比，作为决定 Phase 2 是否要做的依据。

**2026-10-04，“最慢 1% 帧”的噪声未全部明显小于 10%，Phase 1 暂未开工（执行方）**

按上一条的约定停下来报告，数据见“结果记录”中 2026-10-04 的第二轮 A/A。

- globalMap Part 2：A/A 噪声 4.0%，明显小于 10%。
- globalMap Part 3：A/A 噪声 11.0%，超过 10%。这一项本身只有约 1.5 ms，11% 相当于 0.16 ms。ikd-tree 同一项的 A/A 差也有 10.3%，说明主要是机器噪声。
- 随机点 Part 2：A/A 噪声 10.2%。这里只有 99 帧，1% 向上取整就是 1 帧，这一项和单帧最大值是同一个数。

需要用户决定的可选做法，执行方没有自行选择：

1. 10% 的标准只用于 globalMap Part 2；Part 3 和随机点只报告，或者改用“10% 或 0.3 ms，取较宽者”。
2. 对这一项增加运行次数，例如各 10 次，以降低中位数的噪声。
3. 帧数少时扩大尾部窗口，例如“最慢 1% 且至少 10 帧”。

**2026-10-04，尾部耗时标准、第二张验收地图、构建规则与内存阈值（用户转达的审阅结论）**

已同步修改 5.1、5.2、5.7、5.9 和本节的命令。

| 决定 | 理由 |
|---|---|
| 最慢 1% 帧采用上一条的做法 1：地图 Part 2 回退不超过 10%；地图 Part 3 回退不超过 10% 或 0.3 ms，取较宽者；随机点只报告。Part 3 的 RSS 只报告，不再追查 | 第二轮 A/A 中，Part 2 的噪声 4.0%，Part 3 是 11.0%（约 0.16 ms），随机点 10.2%（1 帧即最大值） |
| 增加第二张验收地图 `test/pcd/Global_map_sprase.pcd`（129 万点，文件名拼写如此）。5.7 的速度和尾部耗时标准对两张地图都适用，Part 3 以 sparse 为准。用当前基线可执行文件在 sparse 上补一轮 A/A | globalMap 按文件顺序每 2000 点一帧，一帧的水平跨度中位数是 403 m，几乎覆盖全图，流式测试相当于全图随机插入；sparse 的一帧约 31 m，接近真实扫描 |
| 构建由中位数划分改为按目标填充率划分（5.2）；叶子分裂仍然对半分 | 中位数划分的填充率随点数在 50%–100% 之间摆动：模拟中 globalMap 是 83%，sparse 只有 62%（34.5 B/点）。新规则在两张地图上都是 81%（26.2 B/点） |
| 内存阈值（`LeafSize = 32`）：建完树后两张地图都不超过 28 B/点；流式插入后 globalMap 不超过 35 B/点，sparse 不超过 40 B/点。sparse 实测超过 35 B/点时，报告叶子填充率的分布，不自行改分裂策略 | 模拟显示，空间连续的插入在不重建时填充率只有 55%（38.4 B/点）。重建会重新打包，实际应当更好，但模拟不含重建，给不出准确值 |
| `Global_map_dense.pcd`（4719 万点）不参与验收。Phase 1 验收完成后，给 benchmark 加 `--max-points` 和 `--frame`，用前 500 万点、每帧 2 万点跑一次，只报告（5.9 第 6 项） | 两棵树同时建会超过本机可用内存，`mallinfo()` 的计数超过 2 GB 也会溢出 |

接下来：先在 sparse 上补一轮 A/A，写进结果记录，然后直接开始 Phase 1。完成后的报告要求不变。

**2026-10-04，Phase 1 验收，Phase 2 关闭（用户转达的审阅结论）**

- **Phase 1 验收通过**。审阅方重跑了单元测试，并通读了 `likd_tree.hpp` 的改动，没有发现正确性问题。
- **Phase 2 关闭，不做**。写入的最长等待已降到约 6 ms，见第 6 节开头。
- **Phase 3 之前先修一个性能缺陷**：审阅时发现，有序写入会让树退化成长链（5.10）。作为单独的一步，修法由审阅方给定，不改并发协议和 5.3 的不变量；达不到验收标准或需要改不变量时，停下来报告。
- **修复验收后补全第 7 节**：按 7.3 末尾的要求，依据 Phase 1 的实际结构，补全接口签名、`Op` 的布局、惰性标记与时间戳的交互、提交拆分和 benchmark 方案。只改计划文档，不写 Phase 3 的代码，然后停下来等审阅。
- 另外请执行方对 `radiusSearch` 增加“不排序”选项给出意见，不实现。

**2026-10-04，5.10 的验收中需要用户决定的一项（执行方）**

- **globalMap Part 2 的最慢 1% 帧**：按 5.7 第一次测得 1.133，超过 1.10。
- **补测**：同样条件下按“基线、新版、基线”交替 10 轮，新版与两组基线之比为 0.972 和 1.043，两组基线之间的 A/A 差 6.8%。
- **合并**：50 次运行合并后，中位数之比为 1.058，差异不显著（Mann–Whitney p = 0.24）。
- 这一项在 globalMap 上测的是没有改动的代码路径。执行方判断为噪声，但按约定不自行判为通过，请用户决定。
- 其余各项全部达标，详见结果记录。

**2026-10-04，5.10 验收、不排序的 radiusSearch、Phase 3 的设计（用户转达的决定）**

- **5.10**：globalMap Part 2 的最慢 1% 帧按噪声处理，判定通过，5.10 验收完成。占用一个核的 `chain` 进程是审阅方留下的，已经停掉。在干净的条件下把这一项重测一次（基线与新版交替各 10 次），结果补进结果记录；比值仍超过 1.10 时停下来报告。
- **不排序的 radiusSearch**：作为 Phase 3 之前的单独一步，见 5.11。
- **Phase 3 设计审阅通过，按第 7 节开工**。7.9 的决定如下，正文已同步：

| 7.9 的问题 | 决定 |
|---|---|
| 1. 内存标准 | 改为不超过 6.5 B/点，采用方案 (a)，即每槽一个 `uint32_t` |
| 2. touch 的耗时标准 | 改为绝对值：Part 2 每帧刷新该帧 5-NN 返回的全部近邻，每帧耗时的最慢 1% 平均不超过 5 ms，globalMap 和 sparse 都要满足。`touchPoints` 先对输入去重；不做共同下降，不做句柄。与同帧插入耗时之比只报告，不作标准。允许的小优化：叶子的 `[t_min, t_max]` 没有变化时，不必更新祖先 |
| 3. `Stamp` 的类型 | 独立类型 |
| 4. 开启开关时不带时间戳的写入 | 视为永不过期；文档写明 `Stamp` 的最大值有这个含义 |
| 5. 有重复点时的 `deletePoints` | 开启开关时删最旧的一份；关闭时行为不变 |
| 6. 查询参数的形式 | 不加六个带 `Stamp` 的查询重载。`min_stamp` 作为 `SearchOptions` 的字段，只在 `TRACK_STAMPS` 为真时存在，关闭开关时使用它是编译错误 |

- **流程与之前相同**：按 7.7 的粒度提交，每次提交前跑单元测试、TSan、ASan + UBSan。关闭开关时，布局和性能必须与现在一致（7.3 第 2 项，按 5.7 的测法）。全部做完后按 7.3 逐项验收，实测数字写进结果记录，再报告。达不到标准或需要偏离设计时，停下来报告，不自行放宽。

### 结果记录

**2026-10-04，Phase 0 完成，Phase 1 的基线（执行方）**

提交：

- `5369b72` Add memoryUsage()
- `489f409` Report memory per point in the benchmark
- `d6822f9` Report rebuild latency and time radius and box search
- `a8c9232` Add a nanoflann static reference to the benchmark
- `b20f3ef` Document benchmark builds and interleaved comparisons

检查：

- 每次提交前 ctest 全部通过：`unit_tests`、新增的 `unit_tests_stats`、`unit_tests_tbb`。
- TSan（clang++-10，开启 `LIKD_TREE_STATS`）：全部通过，0 条告警。
- `-Wall -Wextra`：库和单元测试在默认、`LIKD_TREE_STATS`、`LIKD_TREE_STATS + LIKD_TREE_USE_TBB` 三种组合下都没有警告。
- ASan + UBSan 本阶段没有跑（5.6 只对 Phase 1 有要求）。

基线可执行文件与测法：

- 基线是 `build/benchmark_base`，从 `a8c9232` 编译（`b20f3ef` 只改了文档），编译命令同第 10 节，没有启用 nanoflann。`build/` 不入库。
- 测法是 A/A 交替：把同一个可执行文件当作 base 和 new，交替运行各 6 次，随机点和 globalMap 各测一组。
- 下表给出全部 12 次运行的中位数 [最小, 最大]。“A/A”是两组中位数之比，代表这套测法本身的噪声。
- 用的是 `test/compare_benchmarks.py`；每次运行的原始输出在 `build/bench_runs/phase0-aa/`。
- 机器负载约 1.5（一个 Chrome 进程持续占用约一个核），没有做隔离。

globalMap（1,742,788 点）：

| 项 | likd-tree 中位数 [最小, 最大] | A/A | ikd-tree 中位数 |
|---|---|---|---|
| Part 1：构建 | 105.4 ms [103.8, 142.8] | 0.990 | 578.4 ms |
| Part 1：1-NN ×20万 | 193.5 ms [188.4, 258.7] | 1.000 | 264.0 ms |
| Part 1：5-NN ×20万 | 388.8 ms [380.1, 451.4] | 1.011 | 574.9 ms |
| Part 1：radius ×2万（r = 1 m，平均 82 点） | 136.8 ms [134.5, 138.0] | 0.995 | 173.9 ms |
| Part 1：box ×2万（半边长 1 m，平均 122 点） | 95.4 ms [94.5, 102.1] | 0.990 | 123.3 ms |
| Part 1：堆 / RSS / `memoryUsage()` | 96.03 / 110.87 / 80.00 B/点 | 1.000 / 0.999 / 1.000 | 堆 160.00 B/点 |
| Part 2：插入总计 | 1095.8 ms [1071.6, 1154.2] | 1.004 | 2560.6 ms |
| Part 2：最慢一帧插入 | 6.44 ms [5.73, 8.90] | 1.178 | 62.8 ms |
| Part 2：1-NN 顺序 / 5-NN 顺序 / 5-NN TBB | 1990.0 / 3532.8 / 472.4 ms | 0.999 / 0.994 / 0.992 | 2579.8 / 4882.5 / 664.8 ms |
| Part 2：读到排队写入前地图的比例 | 0.01% [0.01, 0.03] | — | — |
| Part 2：重建轮数 / 最长一轮 | 870 / 73.1 ms [68.5, 81.7] | 1.000 / 0.988 | — |
| Part 2：最多排队写入 / 排队写入最长等待 | 4000 / 54.1 ms [46.0, 61.6] | 1.000 / 0.981 | — |
| Part 2：每有效点的堆 / RSS / `memoryUsage()` | 96.05 / 209.1 [168.8, 239.0] / 80.00 B | 1.000 / 1.194 / 1.000 | 堆 252.0 B |
| Part 3：插入总计 | 653.8 ms [644.9, 681.7] | 1.006 | 2338.9 ms |
| Part 3：最慢一帧插入 | 1.94 ms [1.63, 2.51] | 0.904 | 51.5 ms |
| Part 3：1-NN 顺序 / 5-NN 顺序 / 5-NN TBB | 1205.7 / 2421.6 / 355.2 ms | 1.001 / 1.004 / 1.012 | 1820.7 / 3901.2 / 676.3 ms |
| Part 3：box 删除总计 | 10.52 ms [9.33, 12.37] | 1.028 | 269.4 ms |
| Part 3：保留点数 / 内存中节点数 | 78,334 / 78,338 | — | 78,334 / 约 22.5 万 |
| Part 3：重建轮数 / 最长一轮 | 958 / 21.0 ms [19.7, 23.9] | 1.000 / 0.953 | — |
| Part 3：最多排队写入 / 排队写入最长等待 | 6 [6, 2000] / 3.06 ms [1.69, 3.80] | 双峰 / 1.594 | — |
| Part 3：每有效点的堆 / RSS / `memoryUsage()` | 96.16 / 2955 [1955, 3965] / 80.00 B | 1.000 / 1.041 / 1.000 | 堆 2092 B |

10 万个均匀随机点：

| 项 | likd-tree 中位数 [最小, 最大] | A/A | ikd-tree 中位数 |
|---|---|---|---|
| Part 1：构建 | 9.12 ms [7.97, 10.32] | 1.052 | 30.0 ms |
| Part 1：1-NN ×20万 / 5-NN ×20万 | 69.9 / 239.1 ms | 0.996 / 0.986 | 112.2 / 333.7 ms |
| Part 1：radius ×2万（r = 10 m，平均 50 点） | 76.8 ms [76.0, 98.4] | 0.974 | 112.6 ms |
| Part 1：box ×2万（半边长 10 m，平均 94 点） | 50.2 ms [49.6, 51.1] | 0.992 | 62.0 ms |
| Part 1：堆 / RSS / `memoryUsage()` | 96.47 / 87.39 / 80.00 B/点 | 1.000 | 堆 160.00 B/点 |
| Part 2：插入总计 / 最慢一帧插入 | 34.30 ms / 0.46 ms [0.40, 0.90] | 1.001 / 0.927 | 83.0 / 1.61 ms |
| Part 2：1-NN 顺序 / 5-NN 顺序 / 5-NN TBB | 74.0 / 156.1 / 23.7 ms | 1.005 / 1.002 / 0.987 | 90.1 / 199.2 / 27.0 ms |
| Part 2：重建轮数 / 最长一轮 / 最多排队写入 | 99 / 0.51 ms [0.25, 4.78] / 0 | — | — |
| Part 2：每有效点的堆 / RSS / `memoryUsage()` | 96.63 / 244.5 / 80.00 B | 1.000 / 1.002 / 1.000 | 堆 160.7 B |

nanoflann 1.5.0 参照（globalMap，单次运行，编译时加 `-DLIKD_BENCH_NANOFLANN`；nanoflann 单线程构建，likd-tree 用 TBB 并行构建）：

| | 构建 | 1-NN ×20万 | 5-NN ×20万 | radius ×2万 | 堆 |
|---|---|---|---|---|---|
| nanoflann 静态，叶子 10 点 | 235.3 ms | 155.2 ms | 285.5 ms | 136.4 ms | 15.47 B/点 + 16 B/点的点数据 |
| nanoflann 静态，叶子 32 点 | 202.4 ms | 149.4 ms | 290.7 ms | 127.1 ms | 7.69 B/点 + 16 B/点的点数据 |
| 同一次运行中的 likd-tree | 169.4 ms | 207.4 ms | 385.4 ms | 136.2 ms | 96.02 B/点 |

观察：

1. **A/A 噪声。**
   - 地图上，构建、插入总计、各类查询、box 删除，两组中位数之差都在 ±2.8% 以内，多数在 ±1.3% 以内；随机点上除构建外在 ±2.6% 以内。对这些项，5% 的阈值可以分辨。
   - 分辨不出的项：随机点的构建、各个“最慢一帧”、重建延迟几项、RSS。见“决定记录”第 1、2 条。
2. **nanoflann 与 likd-tree 的查询差距随查询集变化。** 在这个查询集上，静态 nanoflann 的 kNN 比 likd-tree 快 1.3–1.4 倍，radius 查询相近（127–136 ms 对 136 ms）。2.2 节的临时程序用的是另一组查询点，测得只快约 18%。
3. **内存测量轮的 RSS 主要来自积压的写入队列。**
   - 测量轮里帧与帧之间没有查询，写入连续到来。临时插桩（没有进仓库）显示：Part 2 的测量轮总共只发生 6 轮重建，`pending_ops_` 最多积压 99.2 万个操作，单个写入最长等待 1703 ms；Part 3 是 130.6 万个操作、1485 ms。
   - 此时堆的峰值比起点高 165–239 MB（四次插桩运行，其中一次关闭了 TBB），而树里只有约 26–39 万个节点。
   - 用 `MALLOC_ARENA_MAX=1` 重跑，Part 3 的 RSS 仍是 2455 B/有效点，排除了“多 arena 造成”的解释。
4. **写入队列没有上界。** 写入比重建快时，队列长度和写入可见前的等待时间会随之增长。按计时轮的节奏（帧间有查询，接近 LIO 的用法），最多排队 4000 个写入，最长等待约 54 ms。
5. **ikd-tree 在 Part 3 结束时，堆增量约 2090 B/有效点。** 它保留了约 22.5 万个节点（节点本身约 460 B/有效点），其余部分来自哪里没有分析，仅作参照。

**2026-10-04，测量调整后的第二轮 A/A（执行方）**

- 提交：`4304b6d` Adjust the measurements after the Phase 0 review。benchmark 增加“最慢 1% 帧的平均插入耗时”；内存测量轮中 likd-tree 每帧之后 `waitForRebuild()`；README 注明写入队列没有上限。
- 检查：提交前 ctest 三个目标全部通过。这次库没有改动，没有跑 TSan。
- 基线：`build/benchmark_base` 已从 `4304b6d` 重新编译，此后作为 Phase 1 的基线。
- 测法：与第一轮相同，A/A 各 6 次，随机点和 globalMap 各一组；原始输出在 `build/bench_runs/phase0-aa2/`。

最慢 1% 帧的平均插入耗时（A/A 两组各 6 次）：

| 数据与部分 | 帧数 → 取几帧 | A 组中位数 [范围] | B 组中位数 [范围] | A/A | ikd-tree 的 A/A |
|---|---|---|---|---|---|
| globalMap Part 2 | 871 → 9 | 4.11 ms [3.54, 4.52] | 4.28 ms [3.78, 4.82] | 1.040 | 1.062 |
| globalMap Part 3 | 871 → 9 | 1.46 ms [1.44, 1.59] | 1.62 ms [1.45, 2.05] | 1.110 | 0.897 |
| 随机点 Part 2 | 99 → 1（即单帧最大值） | 0.54 ms [0.41, 0.64] | 0.48 ms [0.40, 0.62] | 0.898 | 1.077 |

结论：只有 globalMap Part 2 的噪声明显小于 10%，其余两项接近或超过 10%。按约定停下来报告，见“决定记录”。

新测法下内存测量轮的 RSS 增长（12 次运行的中位数 [范围]，只报告）：

| | 每有效点的 RSS 增长 | 同轮的堆增量 | 第一轮（不等待）的 RSS 增长 |
|---|---|---|---|
| globalMap Part 2 | 118.3 B [113.1, 123.2] | 96.08 B | 209.1 B |
| globalMap Part 3 | 681.5 B [638.3, 714.3] | 97.36 B | 2955 B |
| 随机点 Part 2 | 68.9 B [64.8, 81.0] | 97.05 B | 244.5 B |

- 每帧等待重建之后，Part 2 的 RSS 只比堆增量高约 23%，写入队列的积压已经排除。
- Part 3 的 RSS 仍是堆增量的 7 倍。Part 3 结束时只剩 7.8 万个点，而释放的内存并不会全部还给系统，所以 RSS 反映的应该是过程中的峰值。峰值本身没有单独测量。
- 测量轮改为每帧等待后，重建轮数变多，最终结构也随之改变：Part 3 的堆增量从 96.16 变为 97.36 B/有效点，随机点 Part 2 从 96.63 变为 97.05 B/有效点。
- 随机点的 RSS 增长小于堆增量，是因为新分配复用了基准取样时已驻留、但 `malloc_trim` 无法归还的页。

**2026-10-04，sparse 的 A/A（执行方）**

- 基线：用 `build/benchmark_base`（从 `4304b6d` 编译）在 `Global_map_sprase.pcd`（1,290,429 点，645 帧）上 A/A 交替运行，各 6 次。
- 原始输出在 `build/bench_runs/phase0-aa-sparse/`。

| 项 | likd-tree 中位数 [最小, 最大] | A/A | ikd-tree 中位数 |
|---|---|---|---|
| Part 1：构建 | 72.9 ms [68.7, 116.5] | **1.101** | 353.2 ms |
| Part 1：1-NN ×20万 | 173.6 ms [164.9, 197.6] | **1.057** | 238.0 ms |
| Part 1：5-NN ×20万 | 400.4 ms [390.8, 424.6] | 1.027 | 579.9 ms |
| Part 1：radius ×2万（r = 1 m，平均 34 点） | 78.0 ms [77.4, 80.0] | 0.989 | 135.9 ms |
| Part 1：box ×2万（半边长 1 m，平均 54 点） | 64.2 ms [63.0, 65.8] | 1.019 | 88.0 ms |
| Part 1：堆 / RSS / `memoryUsage()` | 96.03 / 110.97 / 80.00 B/点 | 1.000 | 堆 160.00 B/点 |
| Part 2：插入总计 | 428.6 ms [418.0, 436.9] | 1.001 | 1337.5 ms |
| Part 2：最慢 1% 帧 / 单帧最大 | 1.45 ms [1.30, 1.59] / 1.62 ms [1.36, 2.11] | 1.025 / 0.976 | 18.67 / 26.18 ms |
| Part 2：1-NN 顺序 / 5-NN 顺序 / 5-NN TBB | 503.4 / 999.3 / 197.9 ms | 1.008 / 1.004 / 0.976 | 2244.3 / 4041.3 / 790.7 ms |
| Part 2：读到排队写入前地图的比例 | 0.63% [0.53, 0.68] | — | — |
| Part 2：重建轮数 / 最长一轮 | 640 / 79.1 ms [76.7, 90.0] | 1.001 / 1.033 | — |
| Part 2：最多排队写入 / 排队写入最长等待 | 8000 [6000, 10000] / 72.4 ms [64.3, 79.6] | 1.000 / 0.998 | — |
| Part 2：每有效点的堆 / RSS / `memoryUsage()` | 96.07 / 113.4 / 80.00 B | 1.000 / 0.993 / 1.000 | 堆 186.4 B |
| Part 3：插入总计 | 396.4 ms [389.2, 402.0] | 1.012 | 1214.2 ms |
| Part 3：最慢 1% 帧 / 单帧最大 | 1.43 ms [1.31, 1.54] / 1.66 ms [1.39, 2.48] | 1.057 / 1.256 | 11.80 / 16.48 ms |
| Part 3：1-NN 顺序 / 5-NN 顺序 / 5-NN TBB | 444.6 / 859.7 / 170.9 ms | 1.002 / 1.011 / 1.006 | 2012.9 / 3577.5 / 726.8 ms |
| Part 3：box 删除总计 | 1.85 ms [1.68, 2.14] | 1.049 | 29.0 ms |
| Part 3：保留点数 / 内存中节点数 | 105,921 / 105,933 | — | 105,921 / 107,705 |
| Part 3：重建轮数 / 最长一轮 / 排队写入最长等待 | 706 / 8.7 ms / 4.3 ms | — | — |
| Part 3：每有效点的堆 / RSS / `memoryUsage()` | 96.55 / 308.8 / 80.01 B | 1.000 / 0.988 / 1.000 | 堆 2270 B |

观察：

1. **尾部耗时的噪声在新标准之内。** Part 2 最慢 1% 帧差 2.5%；Part 3 差 5.7%，只相当于约 0.08 ms，在 0.3 ms 的下限之内。
2. **两项速度指标的 A/A 噪声超过 5%。**
   - Part 1 的构建（10.1%）：12 次中有 4 次出现 88–117 ms 的离群值，中位数约 73 ms。构建用 TBB 并行，对后台负载敏感。
   - Part 1 的 1-NN（5.7%）：6 对运行中，后跑的那一次全都更慢，原因不明。
   - 对这两项，6 次运行分辨不了 5% 的回退。验收时打算各跑 10 次（计划要求“各不少于 5 次”），并同时报告 A/A 噪声；如果差异落在噪声之内，如实写“无法判定”，不判为通过。
3. **sparse 上写入排队更明显，与 Phase 2 有关。** 最多排队 1 万个写入，单个写入最长等待约 72 ms，读到旧地图的比例为 0.63%；globalMap 上分别是 4000 个、约 51 ms、0.01%。
4. **sparse 上 radius 查询结果数与 ikd-tree 差 1 个点（671,794 对 671,795）。** likd-tree 按平方距离 `d2 <= r2` 判断，ikd-tree 用开方后的距离和包围球判断，差异落在边界上，属于浮点舍入；likd-tree 的半径查询在单元测试里与暴力结果一致。

**2026-10-04，Phase 1 完成与验收（执行方）**

提交（按 5.9）：

| # | 提交 | 内容 |
|---|---|---|
| 1 | `f102ab8` | `Options` 模板参数。单元测试按叶子大小 2/4/32/64 参数化，新增三类测试：重复点、叶子删空、删除子树后再插入 |
| 2 | `10fe65d` | 叶子桶核心改动，连同 `validate()` 和 `memoryUsage()` 的新实现。benchmark 的 “Nodes held” 一行改名为 “Points stored, incl. deleted” |
| 3 | `bfc6204` | 满叶有删除槽时压缩而不分裂，新增测试 `testReuseDeletedSlot` |
| 4 | `d2664c6` | benchmark 增加叶子大小和点类型两个开关；默认叶子大小保持 32 |
| 5 | `ee4ba03` | README：数据结构、性能表、内存 |
| 6 | `a1ab1ff` | 验收之后：benchmark 增加 `--max-points` 和 `--frame` |

检查：

- **ctest**：`unit_tests`、`unit_tests_stats`、`unit_tests_tbb` 三个目标在每次提交前都全部通过。
- **TSan**（clang++-10，开启 `LIKD_TREE_STATS`）：提交 2、3 各跑一次，都是 0 条告警。
- **ASan + UBSan**（g++ 9，`-fno-sanitize-recover=all`，开启泄漏检测）：
  - 提交 2 的非 TBB 版：0 条错误。
  - 提交 2 的 TBB 版：只报出 TBB 2020 自身的 3 块分配，共 6168 B。用一个只调用一次 `tbb::parallel_invoke` 的最小程序得到完全相同的报告。加 `leak:libtbb.so` 抑制后为 0 条错误。
  - 提交 3 的非 TBB 版：0 条错误。提交 3 的 TBB 版**没有重跑**。
- **`validate()`**：以下几处都调用，全部通过。
  - 四种叶子大小的随机写测试中，每次写操作并等待重建结束后调用一次。计划只要求叶子大小 2 和 4。
  - 其余随机测试的每个等待点。
  - 并发测试结束时。
  - 另外，`validate()` 比计划多检查一项：重建结束后树中不应残留 `need_rebuild` 标记。
- **提交 3 的新测试**：去掉压缩判断后，它在叶子大小 4/32/64 下失败，说明它确实测到了这一行为。
- **`-Wall -Wextra`**：没有警告。

实现与计划文字的出入（方案本身未改）：

- 满叶处理函数叫 `makeRoom`，不叫计划里的 `splitLeaf`，因为提交 3 之后它也负责压缩。
- 提交 2 中，满叶的有效点少于 2 个时只压缩、不分裂，以免产生空叶子。
- benchmark 的 “Nodes held” 改名：`nodeCount()` 改变语义后原名不再准确。这一行不在验收项里；对比脚本按行名匹配，基线里没有这一行。

叶子大小扫描（详细数据见 `d2664c6` 的提交说明）：

- **64 对 32**，交替各 6 次。以下是 64 相对 32 的比值：
  - 内存 0.90，插入 0.88–0.92；
  - sparse 上顺序 5-NN 为 1.06（Part 1）、1.13（Part 2）、1.12（Part 3），TBB 5-NN 为 1.10；
  - box 查询 1.09–1.22。
  - LIO 以查询为主，所以保持 32。
- **8 和 16**，各 3 次，只作速度参照：都比 32 慢，建完树的内存分别是 39 和 31 B/点。

**验收（5.7）**：

- 基线是 `build/benchmark_base`（`4304b6d`），新版是从 `d2664c6` 编译的 benchmark。
- 交替运行各 10 次，原始输出在 `build/bench_runs/phase1-final/`。
- 比值是新版与基线中位数之比；“A/A”是基线自身的 A/A 噪声，取自 `phase0-aa2` 和 `phase0-aa-sparse`。
- **42 项全部通过，没有未达标的项。**

| 项 | 标准 | globalMap：基线 → 新版（比值） | sparse：基线 → 新版（比值） |
|---|---|---|---|
| Part 1 堆内存 | ≤ 28 B/点 | 96.03 → **25.26** | 96.03 → **25.27** |
| Part 2 堆内存（每有效点） | globalMap ≤ 35，sparse ≤ 40 | 96.08 → **28.10** | 96.08 → **25.52** |
| Part 3 堆内存（每有效点） | ≤ 75 | 97.36 → **27.64** | 96.55 → **26.16** |
| 构建 | ≤ 1.05 | 108.1 → 51.0 ms（0.472；A/A 1.072） | 85.0 → 33.0 ms（0.388；A/A 1.101） |
| Part 1 1-NN ×20万 | ≤ 1.05 | 188.0 → 110.4 ms（0.588） | 171.9 → 94.2 ms（0.548；A/A 1.057） |
| Part 1 5-NN ×20万 | ≤ 1.05 | 381.9 → 208.6 ms（0.546） | 410.1 → 202.6 ms（0.494） |
| Part 1 radius ×2万 | ≤ 1.05 | 135.1 → 87.5 ms（0.648） | 78.2 → 45.0 ms（0.576） |
| Part 1 box ×2万 | ≤ 1.05 | 94.9 → 47.6 ms（0.501） | 65.2 → 34.7 ms（0.531） |
| Part 2 插入总计 | ≤ 1.05 | 1057 → 683 ms（0.646） | 426 → 352 ms（0.827） |
| Part 2 1-NN / 5-NN 顺序 / 5-NN TBB | ≤ 1.05 | 0.597 / 0.534 / 0.612 | 0.726 / 0.776 / 0.795 |
| Part 3 插入总计 | ≤ 1.05 | 641 → 463 ms（0.723） | 393 → 291 ms（0.741） |
| Part 3 1-NN / 5-NN 顺序 / 5-NN TBB | ≤ 1.05 | 0.582 / 0.580 / 0.655 | 0.706 / 0.799 / 0.811 |
| Part 2 最慢 1% 帧 | ≤ 1.10 | 4.28 → 2.12 ms（0.495） | 1.42 → 1.12 ms（0.792） |
| Part 3 最慢 1% 帧 | ≤ 1.10 或 + 0.3 ms，取较宽者 | 1.45 → 1.11 ms（0.766） | 1.42 → 0.95 ms（0.669） |
| 单帧最大，每次运行 | ≤ 基线各次最大值 + 2 ms | Part 2：6.45 ≤ 10.10；Part 3：2.11 ≤ 6.46 | Part 2：1.27 ≤ 3.92；Part 3：1.40 ≤ 4.22 |
| Part 3 box 删除总计 | ≤ 1.2 | 10.19 → 5.12 ms（0.503） | 1.81 → 0.96 ms（0.532） |

- sparse 的构建和 Part 1 1-NN 的 A/A 噪声分别是 10.1% 和 5.7%，但新版只有基线的 0.39 倍和 0.55 倍，差距远大于噪声，可以判定。
- globalMap 的 Part 3 不代表真实的局部地图场景，Part 3 以 sparse 为准。两张地图的 Part 3 都达标。
- **大点类型**（`pcl::PointXYZINormal`，48 B，叶子大小 32，交替各 6 次）：
  - 5-NN 是基线耗时的 0.58–0.80 倍（随机点、两张地图、Part 1/2/3），没有回退；
  - 内存从 128 B/点降到约 65 B/点。

**Phase 2 的依据**：Part 2 的重建统计（基线 → 新版，各 10 次的中位数 [范围]）：

| 数据 | 重建轮数 | 最长一轮 | 最多排队写入 | 排队写入最长等待 | 读到旧地图的比例 |
|---|---|---|---|---|---|
| globalMap | 870 → 688 | 69.9 → 25.1 ms | 4000 → 2000 [0, 2000] | 53.0 → **6.5 ms** [0, 10.1] | 0.01% → 0.00% |
| sparse | 640 → 645 | 77.3 → 23.4 ms | 8000 → 2000 [2000, 2000] | 67.2 → **6.2 ms** [3.2, 10.4] | 0.64% → 0.03% |
| 随机点 | 99 → 2 | 0.30 → 0.08 ms | 0 → 0 | 0 → 0 | 0 → 0 |
| dense 前 500 万点（单次） | 249 → 249 | 371 → 108 ms | 20000 → 0 | 12.4 → 0 ms | 0 → 0 |

叶子桶让重建快了约 3 倍。一次写入在被查询看到之前最多等待的时间，从约 50–70 ms 降到约 6 ms，大约是 10 Hz 下一帧的 6%。

**RSS 增长**（只报告，每有效点，10 次中位数，基线 → 新版）：

| 数据 | Part 1 | Part 2 | Part 3 |
|---|---|---|---|
| globalMap | 110.9 → 40.6 B | 116.0 → 31.5 B | 678 → 221 B |
| sparse | 110.1 → 39.3 B | 114.2 → 28.5 B | 344 → 98 B |
| 随机点 | 87.4 → 21.8 B | 72.9 → 27.0 B | — |

**随机点**（只报告，10 次中位数，基线 → 新版）：

- 构建：9.02 → 5.18 ms；
- 堆内存：96.47 → 25.70 B/点；Part 2 每有效点 97.05 → 29.45 B；
- 最慢 1% 帧（即单帧最大值）：0.49 → 0.30 ms。

**dense 地图**（`Global_map_dense.pcd` 前 500 万点、每帧 2 万点，基线与新版各一次，只报告；原始输出在 `build/dense_base.txt` 和 `build/dense_new.txt`）：

| 项 | 基线 | 新版 | ikd-tree（新版那次运行） |
|---|---|---|---|
| 构建 | 301 ms | 122 ms | 1428 ms |
| Part 1 1-NN / 5-NN ×20万 | 270 / 476 ms | 178 / 291 ms | 372 / 694 ms |
| Part 1 radius ×2万（平均每次 2086 点） | 2416 ms | 2058 ms | 940 ms |
| Part 1 box ×2万（平均每次 3123 点） | 1130 ms | 393 ms | 929 ms |
| Part 1 堆内存 | 96.01 B/点 | 25.24 B/点 | 160.00 B/点 |
| Part 2 插入总计 / 最慢 1% 帧 | 2045 / 14.2 ms | 1663 / 10.4 ms | 5766 / 100.6 ms |
| Part 2 5-NN 顺序 / TBB | 4243 / 782 ms | 3410 / 653 ms | 19243 / 2458 ms |
| Part 3 box 删除总计 | 4.35 ms | 1.44 ms | 50.24 ms |
| Part 3 每有效点堆内存 | 96.30 B | 25.48 B | 3732 B |

dense 上 likd-tree 的 `radiusSearch` 比 ikd-tree 慢一倍以上，基线也是如此（2416 对 1011 ms）。likd-tree 会把结果按距离排序后返回，ikd-tree 不排序；每次查询的结果点数多时，排序占了大头。这一项不在验收范围内，新版也比基线快。sparse 和 dense 上的 radius 查询结果数都与 ikd-tree 差 1 个点，原因是浮点边界，见 sparse A/A 的观察 4。

**2026-10-04，有序写入的修复（5.10，执行方）**

提交：`a2cb083` Rebalance long writes and replays as they go（库、单元测试、README）。

**有序写入**。执行方的临时程序，40 万点，`struct {float x, y, z}`，默认 `Options`，未定义 `LIKD_TREE_USE_TBB`，各运行一次。耗时从第一次 `addPoints` 算到 `waitForRebuild()` 返回：

| 场景 | 修复前 | 修复后 | 重建轮数（前 → 后） | 最多排队写入（前 → 后） |
|---|---|---|---|---|
| 有序点，一次调用 | 64.5 s | 0.26 s | 1 → 3981 | 0 → 39.8 万 |
| 有序点，每帧 2000 点、不等待 | 62.7 s | 0.31 s | 3 → 3981 | 39.2 万 → 39.0 万 |
| 相同的点，一次调用 | 65.3 s | 0.22 s | 1 → 3981 | 0 → 39.8 万 |
| 相同的点，每帧 2000 点、不等待 | 36.7 s | 0.22 s | 4 → 3981 | 28.4 万 → 39.2 万 |

- **修复前**的数字与审阅方一致：一次调用 10 万点本机 3.6 s。有序帧 20 万点本机 13.9 s（审阅方 8.1 s）：帧在被 worker 取走之前积压多少，取决于时序。
- **修复后**约每回放 100 个操作重建一轮：有序写入的每一批都会让最右边的子树失衡。
- **一次调用的场景里“最多排队写入”变大，不是退化**。修复前整个调用在独占锁内做完，不经过队列，但查询要等 64 s；修复后第一块之后剩下的 39.8 万个操作进入队列，调用本身 14 ms 返回，最后一个写入 0.26 s 后可见。
- **其他配置**（修复后）：开启 TBB 为 0.20–0.24 s；叶子大小 2、4、64 分别为 0.73–0.77 s、0.50–0.56 s、0.20–0.22 s。
- **回归测试** `testOrderedWrites` 的实测值（ctest 中的 `unit_tests`）：叶子大小 32 为 211–250 ms，叶子大小 2 为 731–780 ms，都低于 2 s。把这个测试放到修复前的头文件上，4 项全部超过 10 s 的断言（36–76 s）。

**检查**：

- **ctest**：`unit_tests`、`unit_tests_stats`、`unit_tests_tbb` 全部通过。`-Wall -Wextra` 在默认、`LIKD_TREE_STATS`、`LIKD_TREE_STATS + LIKD_TREE_USE_TBB` 三种组合下都没有警告。
- **TSan**（clang++-10，开启 `LIKD_TREE_STATS`）：全部通过，0 条告警。
- **ASan + UBSan**（g++ 9，`-fno-sanitize-recover=all`，开启泄漏检测）：顺序版 0 条错误。TBB 版只报出 TBB 2020 自身的 3 块共 6168 B，抑制后 0 条错误，与 Phase 1 相同。
- **补上的 TBB 版 ASan + UBSan**：在修复前的 `93c1d4d` 上运行，它的库和单元测试与 `bfc6204` 完全相同。结果与上一条相同，0 条错误。
- **写入顺序的变异检验**：把 worker 手里没回放的写入放到新排队的写入之后（错误的顺序），`testWriteOrderAcrossChunks` 在四种叶子大小下各跑 3 次，12 次全部失败；正确的实现全部通过。
  - 这个测试在第一次调用之后等待 10 ms，再发出后续的写入，让 worker 先取走第一次调用剩下的部分。
  - 不等待时，所有写入往往在 worker 取走之前就排进了同一份日志，错误的顺序也暴露不出来。最初的版本在叶子大小 2 上就没有测出来。

**benchmark**（5.7 的测法）：

- 基线是 `build/benchmark_phase1`，从 `93c1d4d` 编译；新版从 `a2cb083` 编译。
- 交替运行各 10 次，覆盖随机点、globalMap 和 sparse；原始输出在 `build/bench_runs/fix-ordered/`。
- **测量条件与 Phase 1 不同**：另一个 Claude 会话的临时程序 `chain`（在该会话的 scratchpad 里）从 17:52 起一直占满一个核，它遍历 800 万个节点的链，访存量很大。本轮的绝对耗时因此比 Phase 1 验收时高，例如 globalMap Part 2 的插入总计从 683 ms 变为约 910 ms。基线和新版同样受影响。执行方没有停止这个进程。

5.7 各项（新版 / 基线，10 次中位数之比）：

| 项 | 标准 | globalMap | sparse |
|---|---|---|---|
| 构建 | ≤ 1.05 | 0.981 | 1.007 |
| Part 1 1-NN / 5-NN ×20万 | ≤ 1.05 | 1.005 / 0.993 | 0.969 / 0.956 |
| Part 1 radius / box ×2万 | ≤ 1.05 | 1.014 / 0.995 | 0.995 / 1.012 |
| Part 2 插入总计 | ≤ 1.05 | 0.996 | 1.020 |
| Part 2 1-NN / 5-NN 顺序 / 5-NN TBB | ≤ 1.05 | 0.999 / 1.008 / 1.020 | 1.020 / 1.020 / 1.024 |
| Part 3 插入总计 | ≤ 1.05 | 0.998 | 1.010 |
| Part 3 1-NN / 5-NN 顺序 / 5-NN TBB | ≤ 1.05 | 1.001 / 0.999 / 0.987 | 1.007 / 1.009 / 0.985 |
| Part 2 最慢 1% 帧 | ≤ 1.10 | **2.58 → 2.93 ms（1.133），超过标准** | 1.12 → 1.15 ms（1.022） |
| Part 3 最慢 1% 帧 | ≤ 1.10 或 + 0.3 ms，取较宽者 | 1.15 → 1.14 ms（0.991） | 0.93 → 0.94 ms（1.011） |
| 单帧最大，每次运行 | ≤ 基线各次最大值 + 2 ms | Part 2：6.00 ≤ 8.90；Part 3：1.91 ≤ 3.80 | Part 2：1.29 ≤ 3.26；Part 3：1.10 ≤ 3.23 |
| Part 3 box 删除总计 | ≤ 1.2 | 1.000 | 1.008 |
| 堆内存，Part 1 / 2 / 3（B/点） | 不变 | 25.26 / 28.10 / 27.65，基线 25.26 / 28.10 / 27.64 | 25.26 / 25.52 / 26.16，基线 25.26 / 25.52 / 26.15 |

- **内存不变**：`memoryUsage()` 每点逐位相同，堆内存的差异不超过 0.01 B/点。
- **随机点**（只报告）：各项之比在 0.956–1.019 之间；重建轮数相同（2 轮），没有写入排队。
- 只有 globalMap Part 2 的最慢 1% 帧超过标准，见下面的“未达标项”。

**重建统计**（Part 2，10 次中位数 [范围]，基线 → 新版）：

| 地图 | 重建轮数 | 最长一轮 | 最多排队写入 | 排队写入最长等待 | 读到旧地图的比例 |
|---|---|---|---|---|---|
| globalMap | 688 → 688 | 24.1 → 21.2 ms | 1000 [0, 2000] → 0 [0, 0] | 0.82 [0, 7.04] → 0 [0, 0] ms | 0.00% → 0.00% |
| sparse | 645 → 665 [658, 665] | 25.0 → 25.1 ms | 2000 → 2000 | 8.79 [2.62, 12.93] → 9.03 [5.14, 12.61] ms | 0.10% → 0.11% |

- **globalMap 上新旧版本做的是同一套树操作**：重建轮数完全相同；新版 10 次运行里没有一帧排队，每一帧都走与修复前相同的直接写入路径（2000 点的帧正好是一块）。
- **sparse 上回放中断起了作用**：重建轮数多 20 轮左右（3%），最多排队写入和最长等待没有变化。
- **Part 3**：两张地图的重建轮数都相同（947 轮、705 轮），最多排队 6 个写入，最长等待 0.4–1.9 ms，基线和新版相近。
- **与 Phase 2 的结论一致**：本轮条件下，排队写入的最长等待基线是 0.8 ms（globalMap）和 8.8 ms（sparse），新版相同。Phase 1 验收时是约 6 ms，差别来自机器负载。修复没有改变关闭 Phase 2 的依据。

**未达标项：globalMap Part 2 的最慢 1% 帧**（第一次测量 1.133，标准 ≤ 1.10）

- **这一项在 globalMap 上测的是没有改动的代码路径**。
  - 新版 10 次运行中没有一帧排队；2000 点的帧正好是一块，`write()` 的直接写入路径只是把原来按元素的循环换成了按下标的循环。
  - 重建轮数（688 轮）与基线完全相同。
  - 修复只在写入排队、或一次写入超过 2000 个操作时才改变行为。
- **补测**：在同样的机器条件下（另一个会话的进程仍在运行），按“基线、新版、基线”的顺序交替运行 10 轮，原始输出在 `build/bench_runs/fix-ordered-aab/`。

  | | 基线（第 1 组） | 新版 | 基线（第 2 组） |
  |---|---|---|---|
  | 最慢 1% 帧，10 次中位数 | 2.71 ms | 2.64 ms | 2.53 ms |

  - 两组基线之间的 A/A 差 6.8%（0.932）；新版落在两组基线之间，与它们之比为 0.972 和 1.043。
  - 三组基线（第一次测量的一组加补测的两组）的中位数分别是 2.585、2.715、2.530 ms，彼此相差约 7%。Phase 0 在负载约 1.5 时测得这一项的 A/A 噪声是 4.0%，本轮机器更忙，噪声更大。
  - 两次测量合并（基线 30 次，新版 20 次）：中位数 2.59 → 2.74 ms（1.058），均值之比 1.050；Mann–Whitney 检验 p = 0.24，差异不显著。
  - 补测中其他各项也在标准之内：Part 2/3 的插入总计之比 1.000–1.011，各类查询 0.992–1.018。
- **执行方的判断**：第一次的 1.133 是噪声，补测中没有重现。但第一次测量是按 5.7 的测法做的，按“不自行放宽标准”的约定，这一项记为“第一次测量未达标，补测达标，差异不显著”，由用户决定是否接受。

**其他观察**：

- **小批量的密集写入多了一些重建**。单元测试 `testNoPointsLostDuringRebuild`（每批 2000 个 σ = 0.5 m 的密集点，不等待）变慢：叶子大小 2 从 10.6 s 到 13.4 s，叶子大小 32 从 4.9 s 到 5.8 s（+18% 到 +26%）。
  - 原因：每批点集中在一处，回放每 100 个操作就被打断一次，去重建逐渐变大的同一片子树；修复前是整批回放完再重建一次。
  - 地图 benchmark 上没有出现：sparse 的重建轮数只多 3%，globalMap 没有变化。
  - 如果要处理，一个折中是回放时每 2000 个操作才检查一次候选，与写调用的分块一致；代价是有序写入时，每块之内仍会长出约 2000 / 16 = 125 层的链。执行方没有改，按审阅方给定的修法执行。
- **重复点同样会长链，修复对它也有效**：同一组单元测试中，`testDeleteDifferential`（网格化坐标，大量重复点）在叶子大小 2 上从 4.8 s 降到 1.0 s。

**2026-10-04，5.10 的补测：干净条件下的 globalMap Part 2（执行方）**

- 条件：`chain` 进程停掉之后，负载约 0.4。基线 `build/benchmark_phase1`（`93c1d4d`），新版从 `a2cb083` 重新编译（与验收时用的可执行文件逐字节相同），globalMap 交替各 10 次；原始输出在 `build/bench_runs/fix-ordered-clean/`。
- **最慢 1% 帧：2.26 → 2.12 ms（0.938），在 1.10 之内。**

| 项（Part 2） | 基线中位数 [最小, 最大] | 新版 | 新版 / 基线 |
|---|---|---|---|
| 插入总计 | 715.3 ms [682.0, 728.8] | 701.7 ms [690.1, 779.8] | 0.981 |
| 最慢 1% 帧 | 2.26 ms [1.88, 3.81] | 2.12 ms [1.92, 3.86] | 0.938 |
| 单帧最大 | 3.46 ms [2.51, 7.17] | 3.12 ms [2.79, 5.56] | 0.904 |
| 1-NN 顺序 / 5-NN 顺序 / 5-NN TBB | 1189.8 / 1905.0 / 290.7 ms | 1182.6 / 1879.9 / 294.9 ms | 0.994 / 0.987 / 1.014 |
| 重建轮数 | 688 | 688 | 1.000 |
| 最多排队写入 | 2000 [0, 2000] | 2000 [0, 2000] | 1.000 |
| 排队写入最长等待 | 6.21 ms [0, 9.74] | 5.01 ms [0, 9.75] | 0.807 |

- 绝对耗时回到了 Phase 1 验收时的水平（插入总计约 700 ms，Phase 1 为 683 ms）。上一轮比这一轮高约 30%，来自 `chain` 进程。
- 排队写入的最长等待约 5–6 ms，与关闭 Phase 2 时的依据一致。
