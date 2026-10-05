#!/usr/bin/env python3
"""Draws the benchmark summary chart in the README.

The numbers are the docs/benchmark.md table for Global_map_sprase.pcd: medians of 10
benchmark runs. Update ROWS when that table changes.

  python3 tools/plot_benchmark.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, likd-tree, ikd-tree, unit, ikd-tree / likd-tree as the table gives
# it, computed there from unrounded values)
ROWS = [
    ("Batch build", 36, 365, "ms", 10.1),
    ("Insert, total", 353, 1352, "ms", 3.8),
    ("Insert, slowest 1% of frames", 1.1, 21.4, "ms", 19.1),
    ("1-NN queries", 360, 2446, "ms", 6.8),
    ("5-NN queries", 771, 4291, "ms", 5.6),
    ("5-NN queries, TBB", 153, 834, "ms", 5.5),
    ("Box deletion (local map)", 1.04, 28.6, "ms", 27.5),
    ("Memory after streaming", 25.5, 186, "B/point", 7.3),
]

COLORS = dict(surface="#fcfcfb", text="#0b0b0b", secondary="#52514e",
              grid="#e3e2dd", bar="#2a78d6")


def draw(out):
    c = COLORS
    ratios = [row[4] for row in ROWS]
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=144)
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    ys = range(len(ROWS))
    ax.barh(ys, ratios, height=0.5, color=c["bar"], zorder=3)
    for y, (label, likd, ikd, unit, ratio) in zip(ys, ROWS):
        ax.annotate(f"{ratio:.1f}×", (ratio, y), xytext=(8, 0),
                    textcoords="offset points", va="center", color=c["text"],
                    fontsize=11, fontweight="bold")
        ax.annotate(f"{likd:g} vs {ikd:g} {unit}", (ratio, y), xytext=(58, 0),
                    textcoords="offset points", va="center",
                    color=c["secondary"], fontsize=9.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([row[0] for row in ROWS], color=c["text"], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 36)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels(["1×", "5×", "10×", "15×", "20×", "25×", "30×"],
                       color=c["secondary"], fontsize=10)
    ax.xaxis.grid(True, color=c["grid"], linewidth=0.8, zorder=0)
    ax.axvline(1, color=c["secondary"], linewidth=1, zorder=2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.03, 0.94, "How many times faster and smaller than ikd-tree",
             color=c["text"], fontsize=14, fontweight="bold")
    fig.text(0.03, 0.885,
             "ikd-tree ÷ likd-tree, time or memory (likd-tree vs ikd-tree beside each "
             "bar). 1.29M-point LiDAR map, 2000-point frames.",
             color=c["secondary"], fontsize=10)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.83, bottom=0.08)
    fig.savefig(out, facecolor=c["surface"])
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    draw("imgs/benchmark.png")
