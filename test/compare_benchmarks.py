#!/usr/bin/env python3
"""Interleaved A/B comparison of two benchmark executables.

Runs `base` and `new` alternately, `--runs` times each, on the random points
and optionally on a map, then prints the median of every likd-tree row with
its min and max, the new/base ratio, and the same ratio for ikd-tree. ikd-tree
is the same code in both executables, so its ratio shows how much the machine
drifted between the two sets of runs.

Odd rounds run `base` first and even rounds `new` first. A process's first
large allocations, such as Part 1's build, depend on the process that ran
just before it: with a fixed order, the same build measured up to 11% slower
or 10% faster depending only on which executable went first.

    python3 test/compare_benchmarks.py build/benchmark_base build/benchmark \
        --map test/pcd/globalMap.pcd --map test/pcd/Global_map_sprase.pcd

Passing the same executable twice (an A/A run) measures the noise floor.
"""

import argparse
import os
import re
import statistics
import subprocess
import sys

ROW = re.compile(
    r"^  (?P<name>\S.*?)\s+likd-tree\s+(?P<likd>-?[\d.]+|-?nan|-?inf)\s*(?P<unit>[^|\s]*)"
    r"(?:\s*\|\s*ikd-tree\s+(?P<ikd>-?[\d.]+|-?nan|-?inf))?")
STALE = re.compile(r"queued writes \(likd-tree\): (?P<pct>[\d.]+)%")
PART = re.compile(r"^=== (Part \d+)")


def parse(output):
    """Returns {(part, name): (likd, ikd or None, unit)} for one run."""
    rows, part = {}, ""
    for line in output.splitlines():
        m = PART.match(line)
        if m:
            part = m.group(1)
            continue
        m = STALE.search(line)
        if m:
            rows[(part, "Answers from stale map")] = (float(m.group("pct")), None, "%")
            continue
        m = ROW.match(line)
        if m:
            ikd = float(m.group("ikd")) if m.group("ikd") else None
            rows[(part, m.group("name"))] = (float(m.group("likd")), ikd, m.group("unit"))
    return rows


def run(exe, args, save_to):
    out = subprocess.run([exe] + args, check=True, capture_output=True,
                         text=True).stdout
    if save_to:
        with open(save_to, "w") as f:
            f.write(out)
    return parse(out)


def fmt(values):
    return f"{statistics.median(values):.2f} [{min(values):.2f}, {max(values):.2f}]"


def ratio(new, base):
    b = statistics.median(base)
    return f"{statistics.median(new) / b:.3f}" if b else "-"


def compare(base_exe, new_exe, args, runs, label, save_dir):
    results = {"base": [], "new": []}
    for i in range(runs):
        order = [("base", base_exe), ("new", new_exe)]
        if i % 2 == 1:  # even rounds, counting from 1: new first
            order.reverse()
        for kind, exe in order:
            save_to = (os.path.join(save_dir, f"{label}-{kind}-{i}.txt")
                       if save_dir else None)
            print(f"  {label}: {kind} run {i + 1}/{runs}", file=sys.stderr)
            results[kind].append(run(exe, args, save_to))
    print(f"\n### {label}: median [min, max] of {runs} interleaved runs each\n")
    print("| Part | Metric | Base | New | New/base | ikd-tree new/base |")
    print("|---|---|---|---|---|---|")
    for key in results["base"][0]:
        base = [r[key] for r in results["base"] if key in r]
        new = [r[key] for r in results["new"] if key in r]
        if len(base) != runs or len(new) != runs:
            continue
        unit = f" ({base[0][2]})" if base[0][2] else ""
        likd_base, likd_new = [v[0] for v in base], [v[0] for v in new]
        ikd = "-"
        if base[0][1] is not None:
            ikd = ratio([v[1] for v in new], [v[1] for v in base])
        print(f"| {key[0]} | {key[1]}{unit} | {fmt(likd_base)} | "
              f"{fmt(likd_new)} | {ratio(likd_new, likd_base)} | {ikd} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--map", action="append", default=[],
                        help="PCD map to run as well; repeat for several")
    parser.add_argument("--skip-random", action="store_true",
                        help="skip the 100K random points")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--save-dir", help="keep every run's output here")
    args = parser.parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    if not args.skip_random:
        compare(args.base, args.new, [], args.runs, "random", args.save_dir)
    for path in args.map:
        label = os.path.splitext(os.path.basename(path))[0]
        compare(args.base, args.new, [path], args.runs, label, args.save_dir)


if __name__ == "__main__":
    main()
