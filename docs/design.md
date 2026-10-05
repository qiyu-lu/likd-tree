# likd-tree design

How the tree in `src/likd_tree.hpp` is laid out, how each operation works, and
which rules keep readers, writers and the background rebuild thread safe from
each other. For how to call it, see the [README](../README.md).

## Layout

Points live in the leaves. Inner nodes only divide space.

```
                      Inner  split x = 3.2
                     /                    \
          Inner  split y = -1.0          Leaf  [p p p p p . . .]
          /                  \
  Leaf [p p x p p p . .]    Leaf [p p p p p p p p]
           ^ deleted slot             full: the next insert splits it
```

| Node | Holds | Size with `pcl::PointXYZ`, leaf size 32 |
|---|---|---|
| Every node | parent pointer, bounding box, `size`, `valid`, four flags | 48 B |
| `Inner` | the above, two child pointers, split axis and value | 72 B |
| `Leaf` | the above, a deletion bit mask, 32 point slots | 560 B |

- **`size`** counts the points stored below a node, deleted ones included.
  **`valid`** counts the ones that are not deleted.
- **The bounding box** covers exactly the non-deleted points below the node.
  Searches prune with it, and deletions use it to find or discard whole
  subtrees.
- **A leaf** uses its first `size` slots. Bit *i* of the mask marks slot *i*
  as deleted. A leaf with no deleted slots is scanned without testing bits.
- **An inner node** may have an empty child, left behind when a rebuild found
  nothing to keep on that side.

One allocation per node, and about 26 points per leaf, gives roughly 25 bytes
per point. A tree with one point per node needs 96.

The leaf size is `Options::LEAF_SIZE` (2 to 64, default 32). Larger leaves
save memory and speed up insertion; smaller ones speed up queries.

## Building

`build()` and every rebuild use the same recursion on a range of *n* points:

1. If *n* ≤ leaf size, make a leaf.
2. Otherwise aim for leaves of 26 points (80% of the leaf size): the range
   needs `L = ceil(n / 26)` leaves, at least 2.
3. Split along the longest side of the range's bounding box, giving the left
   side `floor(L / 2)` leaves' worth of points (`nth_element`).

A plain median split would make the fill rate swing between 50% and 100%
depending on *n*. This rule gives 81% for any point count, and leaves room to
insert before a leaf has to split. Subtrees above 20,000 points are built with
parallel tasks when TBB is enabled.

## Writing

### Insertion

Descend by the split values (`coord < split` goes left) to a leaf, then:

- **Room left**: append the point and grow the bounding box.
- **Full, with deleted slots**: compact the leaf in place, then append.
- **Full, nothing deleted**: split. The leaf keeps the lower half of its
  points along their longest extent, a new leaf takes the upper half, and a
  new inner node takes the leaf's place.
- **Empty child or empty tree**: make a new leaf there.

On the way back up, each ancestor recomputes its counts and bounding box from
its children.

Points equal to a split value can end up on either side, so only insertion
trusts the split values. Everything else navigates by bounding box.

### Deletion

- **By point**: descend into every subtree whose box contains the point, set
  the bit of the first matching slot, and recompute the leaf's box.
- **By box**: a subtree whose box lies entirely inside the deletion box is
  deleted in O(1) (see the lazy tag below). Leaves that only overlap it are
  scanned.

Deleted points stay in memory until a rebuild drops them, or until their leaf
is compacted to make room.

### The lazy tag

Deleting a whole subtree sets `tree_deleted` on its root, zeroes `valid` and
empties the bounding box. Nothing below is touched.

- Everything that reads the tree stops at a node with `valid == 0`, so stale
  descendants are never seen.
- A writer that must descend through a tagged node (only insertion does)
  first pushes the tag one level down.

### Rebalancing

After a write changes a node, the node becomes a rebuild candidate when it
holds at least 128 points (4 × leaf size) and either

- one child holds more than 75% of its points, or
- more than half of its points are deleted.

Only the topmost candidates are rebuilt: a candidate below another candidate
is covered by its ancestor's rebuild. The worker thread collects the
non-deleted points of each subtree, builds a replacement as above, and swaps
it in. A subtree with nothing left becomes an empty child.

## Concurrency

Three kinds of thread touch the tree: any number of readers, one writer at a
time, and one worker that rebuilds.

| Lock | Protects |
|---|---|
| `tree_mutex_` (shared / exclusive) | The tree. Queries hold it shared; changes hold it exclusive. |
| `write_mutex_` | Serializes the write calls. |
| `pending_mutex_` | The rebuilding state, the queue of pending writes, the worker's job. |

`tree_mutex_` makes new readers wait while a writer is waiting. Without that,
back-to-back queries from other threads could starve writers indefinitely
with glibc's reader-preferring `shared_mutex`.

### A rebuild, step by step

1. A write finds candidates. Under `pending_mutex_` it sets `rebuilding_` and
   hands them to the worker.
2. **While `rebuilding_` is set, every write is queued** instead of applied.
   The tree is therefore frozen, and the worker reads the old subtrees and
   builds the new ones without holding `tree_mutex_`. Queries keep running.
3. The worker takes `tree_mutex_` exclusively only to swap the pointers and
   update the ancestors' counts, then frees the old subtrees outside the lock.
4. It replays the queued writes in order, 100 at a time, releasing the lock
   between batches so queries can interleave.
5. If a replayed batch produces new candidates, the worker stops replaying,
   rebuilds them, and then continues. It leaves the rebuilding state only
   when it sees an empty queue while holding `pending_mutex_`.

### Rules the code relies on

1. **Only the worker frees nodes**, and only after swapping them out under
   the exclusive lock. (`build()` also frees the old tree, after waiting for
   the worker.) Writers add nodes but never free them.
2. **Every structural change happens under the exclusive lock**: append,
   compact, split, set a deletion bit. A query never sees a half-made change.
3. **Writes issued during a rebuild go to the queue**, which is what lets the
   worker read old subtrees without a lock.
4. **Leaves are never rebuild candidates.** Only inner nodes are, and a leaf
   split creates nodes without freeing any, so a candidate list never holds a
   dangling pointer.
5. **Queries copy their results before releasing the lock.** A rebuild may
   free the nodes right after.

### Long writes

A write call is applied in chunks of 2000 operations. As soon as a chunk
produces a candidate, the rebuild starts and the rest of the call is queued
behind it.

Without this, a long run of points sorted along an axis grows the tree into
a chain between rebuilds, and every later insertion walks the whole chain:
400,000 sorted points took over a minute. With it they take about 0.3 s. The
same rule in step 5 above keeps a long replay from doing the same.

The consequence for callers: a write may return before all of it is visible
to queries. `waitForRebuild()`, or `wait_for_rebuild = true`, waits for it.

## Searching

All four searches descend from the root, skip subtrees with `valid == 0`, and
prune by bounding box.

- **Nearest and k-nearest**: visit the child on the query's side of the split
  first, then the other one if its box is closer than the current worst
  result. Leaves are scanned linearly.
- **Radius**: visit every subtree whose box is within the radius. Results are
  sorted by distance unless `SearchOptions::sorted` is false; for queries that
  return thousands of points the sort is most of the cost.
- **Box**: a subtree whose box lies inside the query box is collected without
  testing its points.

## Stamps (optional)

With `Options::TRACK_STAMPS`, each point carries a 32-bit stamp, and each
node the range `[t_min, t_max]` of the stamps of the non-deleted points below
it. The range is maintained wherever the bounding box is.

- **Layout**: a leaf gains a `uint32_t stamps[LEAF_SIZE]` array placed after
  the points, so the fields a query reads keep their offsets. This costs 5 to
  7 bytes per point. With the option off, no field is added anywhere.
- **`expireBefore(T)`** is a box deletion along time. A subtree with
  `t_max < T` gets the lazy tag in O(1), one with `t_min >= T` is skipped, and
  anything else is descended.
- **`touchPoints`** finds each point like a deletion does and raises its
  stamp. It skips subtrees with `t_min >=` the new stamp, and stops updating
  ancestors once a node's range no longer changes.
- **`min_stamp` queries** skip subtrees with `t_max < min_stamp` and compare
  stamps point by point inside leaves.
- **`deletePoints`** removes the copy with the oldest stamp when a point is
  stored more than once, so which copy survives is well defined.
- A point written without a stamp gets the largest value and never expires.

There is no lazy form of touch. A box-wide touch was considered and dropped:
it needs a second lazy tag that queries would have to carry down the tree,
and "everything in this box was seen" is not true of occluded points.

## What was tried and not adopted

- **Sorted pages indexed along x, y and z**, as in a database B+ tree. Order
  along one axis does not keep 3D neighbors together, three indexes triple
  the pointers, and merging half-empty pages on deletion would free nodes on
  the write path, which rule 1 forbids.
- **nanoflann's dynamic index** (logarithmic merging). Queries visit every
  sub-tree, merges run on the calling thread (a 101 ms worst frame on the
  benchmark map), and removed points are never reclaimed.
- **Removing points from leaves immediately on deletion.** It leaves sparse
  and empty leaves that have to be merged on the write path. With deferred
  reclamation the worst case is bounded anyway: a leaf at least half full
  after a rebuild, at most half of the points deleted before the next one.
- **Making writes visible during a rebuild.** After leaf buckets, a queued
  write waits about 6 ms on the benchmark maps, which did not justify a more
  complex protocol.
- **Leaves without stored stamp ranges.** It saves about 0.6 bytes per point
  but makes touch 6–12% and expiry 15–35% slower.
