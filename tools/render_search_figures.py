#!/usr/bin/env python3
"""Renders the search figures in the README from a PCD map.

Runs the two PCD demos with --no-vis, draws the points they report on a
perspective view of the map, and stops if the points it is about to highlight
are not the ones the demos found.

  cmake -B build && cmake --build build
  python3 tools/render_search_figures.py test/pcd/globalMap.pcd

Needs numpy, scipy, matplotlib and a PCD file with binary x, y, z fields.
"""
import argparse
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from scipy.spatial import ConvexHull

SURFACE = np.array([0x1A, 0x1A, 0x19]) / 255.0
CLOUD = np.array([0xEC, 0xEB, 0xE4]) / 255.0
TEXT = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
RESULT = "#3987e5"  # the points a search returns
QUERY = "#d95926"
WIDTH, HEIGHT, FOV_DEG = 1440, 810, 60.0


def load_pcd(path):
    """Finite x, y, z of a binary PCD file as an (n, 3) float64 array."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("#"):
                continue
            key, *values = line.split()
            header[key] = values
            if key == "DATA":
                break
        if values != ["binary"]:
            sys.exit(f"{path}: DATA {values[0]} is not supported, only binary")
        dtype = np.dtype({
            "names": header["FIELDS"],
            "formats": [f"<{t.lower()}{s}" for t, s in
                        zip(header["TYPE"], header["SIZE"])],
        })
        count = int(header["POINTS"][0])
        data = np.frombuffer(f.read(count * dtype.itemsize), dtype, count)
    pts = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    return pts[np.isfinite(pts).all(axis=1)]


class Camera:
    def __init__(self, eye, target):
        self.eye = np.asarray(eye, float)
        forward = np.asarray(target, float) - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        self.rot = np.stack([right, np.cross(right, forward), forward])
        self.focal = 0.5 * WIDTH / np.tan(np.radians(FOV_DEG) / 2)

    def project(self, pts):
        """Pixel x, y and depth; points behind the camera get depth <= 0."""
        cam = (np.atleast_2d(pts) - self.eye) @ self.rot.T
        depth = cam[:, 2]
        safe = np.where(depth > 0.3, depth, np.nan)
        x = WIDTH / 2 + self.focal * cam[:, 0] / safe
        y = HEIGHT / 2 - self.focal * cam[:, 1] / safe
        return x, y, np.where(depth > 0.3, depth, 0.0)


def splat(cam, pts, far=140.0, exposure=1.3):
    """Additive point splatting with distance fade, as an RGB image."""
    x, y, depth = cam.project(pts)
    keep = (depth > 0) & (depth < far) & (x >= 0) & (x < WIDTH - 1) & \
           (y >= 0) & (y < HEIGHT - 1)
    x, y, depth = x[keep], y[keep], depth[keep]
    weight = np.clip(1.2 - depth / far, 0.15, 1.0) * np.clip(8.0 / depth, 0.3, 2.5)
    xi, yi = x.astype(int), y.astype(int)
    fx, fy = x - xi, y - yi
    light = np.zeros((HEIGHT, WIDTH))
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(light, (yi + dy, xi + dx), weight * w)
    tone = 1.0 - np.exp(-exposure * light)
    return SURFACE + tone[..., None] * (CLOUD - SURFACE)


def run_demo(binary, args):
    result = subprocess.run([binary, *map(str, args), "--no-vis"],
                            capture_output=True, text=True)
    if result.returncode != 0 or "Brute-force check: MATCH" not in result.stdout:
        sys.exit(f"{binary} failed:\n{result.stdout}{result.stderr}")
    return result.stdout


def field(output, name):
    match = re.search(rf"^{re.escape(name)}: (.*)$", output, re.M)
    if not match:
        sys.exit(f"'{name}' not found in the demo output")
    return match.group(1)


def new_figure(image):
    fig = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=(0, WIDTH, HEIGHT, 0), interpolation="nearest")
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    return fig, ax


HALO = [patheffects.withStroke(linewidth=3, foreground="#1a1a19")]


def caption(ax, title, detail):
    ax.text(40, 56, title, color=TEXT, fontsize=19, fontweight="bold",
            path_effects=HALO)
    ax.text(40, 88, detail, color=TEXT_SECONDARY, fontsize=12.5,
            path_effects=HALO)


def label(ax, xy, text, offset, align="left"):
    ax.annotate(text, xy, xytext=offset, textcoords="offset points",
                color=TEXT, fontsize=12.5, ha=align, va="center",
                path_effects=HALO)


def marker(ax, x, y, color, size):
    # A ring in the surface color keeps a mark apart from what it overlaps
    ax.scatter(x, y, s=size, color=color, edgecolors="#1a1a19", linewidths=1.5,
               zorder=5)


def view_direction(yaw_deg):
    yaw = np.radians(yaw_deg)
    return np.array([np.cos(yaw), np.sin(yaw), 0.0])


def render_nearest(pts, query, demo_dir, map_path, yaw, out):
    output = run_demo(f"{demo_dir}/nearest_search_pcd_demo", [map_path, *query])
    reported = np.array(re.findall(r"-?\d+\.\d+", field(output, "Nearest")), float)
    dists = np.linalg.norm(pts - query, axis=1)
    nearest = pts[np.argmin(dists)]
    if not np.allclose(nearest, reported, atol=1e-4):
        sys.exit(f"nearest point differs from the demo's: {nearest} vs {reported}")

    ahead = view_direction(yaw)
    cam = Camera(query - 9.0 * ahead + [0, 0, 1.8],
                 query + 6.0 * ahead + [0, 0, 0.2])
    fig, ax = new_figure(splat(cam, pts))
    (qx,), (qy,), _ = cam.project(query)
    (nx,), (ny,), _ = cam.project(nearest)
    ax.plot([qx, nx], [qy, ny], color=TEXT, linewidth=2, zorder=4,
            path_effects=[patheffects.withStroke(linewidth=4, foreground="#1a1a19")])
    marker(ax, qx, qy, QUERY, 170)
    marker(ax, nx, ny, RESULT, 170)
    label(ax, (qx, qy), "query", (14, 0))
    label(ax, (nx, ny), "nearest neighbor", (14, 0))
    label(ax, ((qx + nx) / 2, (qy + ny) / 2), f"{dists.min():.2f} m", (-12, 0),
          align="right")
    caption(ax, "Nearest neighbor search",
            f"{len(pts):,} points  ·  result checked against brute force")
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    print(f"{out}: nearest {nearest.round(3)} at {dists.min():.3f} m")


def render_radius(pts, query, radius, demo_dir, map_path, yaw, out):
    output = run_demo(f"{demo_dir}/radius_search_pcd_demo",
                      [map_path, *query, radius])
    reported = int(field(output, "Radius search points"))
    found = pts[np.linalg.norm(pts - query, axis=1) <= radius]
    if len(found) != reported:
        sys.exit(f"{len(found)} points in the radius, the demo found {reported}")

    ahead = view_direction(yaw)
    cam = Camera(query - 13.0 * ahead + [0, 0, 2.6],
                 query + 6.0 * ahead + [0, 0, 1.2])
    fig, ax = new_figure(splat(cam, pts))

    # The sphere: its outline, and three great circles for depth
    rng = np.random.default_rng(0)
    on_sphere = rng.normal(size=(4000, 3))
    on_sphere = query + radius * on_sphere / np.linalg.norm(on_sphere, axis=1)[:, None]
    sx, sy, _ = cam.project(on_sphere)
    hull = ConvexHull(np.stack([sx, sy], axis=1)).vertices
    ax.fill(sx[hull], sy[hull], facecolor=RESULT, alpha=0.10, zorder=2)
    ax.plot(np.append(sx[hull], sx[hull[0]]), np.append(sy[hull], sy[hull[0]]),
            color=RESULT, linewidth=2, zorder=3)
    angle = np.linspace(0, 2 * np.pi, 200)
    cos, sin, zero = np.cos(angle), np.sin(angle), np.zeros_like(angle)
    for circle in ((cos, sin, zero), (cos, zero, sin), (zero, cos, sin)):
        cx, cy, _ = cam.project(query + radius * np.stack(circle, axis=1))
        ax.plot(cx, cy, color=RESULT, linewidth=1, alpha=0.45, zorder=3)

    fx, fy, depth = cam.project(found)
    order = np.argsort(-depth)
    ax.scatter(fx[order], fy[order], s=40, color=RESULT, edgecolors="#1a1a19",
               linewidths=1.2, zorder=4)
    (qx,), (qy,), _ = cam.project(query)
    marker(ax, qx, qy, QUERY, 170)
    label(ax, (qx, qy), "query", (14, 0))
    label(ax, (qx, sy[hull].max()), f"{len(found)} points within {radius:g} m",
          (0, -18), align="center")
    caption(ax, "Radius search",
            f"{len(pts):,} points  ·  result checked against brute force")
    fig.savefig(out, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    print(f"{out}: {len(found)} points within {radius:g} m")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("map", help="PCD file, e.g. test/pcd/globalMap.pcd")
    parser.add_argument("--query", type=float, nargs=3, default=[0.9, 0.1, 0.0])
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--yaw", type=float, default=45.0,
                        help="viewing direction in the xy plane, degrees")
    parser.add_argument("--demo-dir", default="build")
    parser.add_argument("--out-dir", default="imgs")
    args = parser.parse_args()

    pts = load_pcd(args.map)
    query = np.array(args.query)
    render_nearest(pts, query, args.demo_dir, args.map, args.yaw,
                   f"{args.out_dir}/nearest_search_pcd_result.png")
    render_radius(pts, query, args.radius, args.demo_dir, args.map, args.yaw,
                  f"{args.out_dir}/radius_search_pcd_result.png")


if __name__ == "__main__":
    main()
