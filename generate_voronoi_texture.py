"""
Generate adjustable Voronoi/Tyson polygon textures.

Change the parameters in CONFIG, then run:
    python generate_voronoi_texture.py

Outputs are saved next to this script.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box


CONFIG = {
    # Output canvas size in pixels.
    # 整张图的像素尺寸；整图面积 = canvas_width * canvas_height。
    # 如果你想让输出图更大或更小，改这里。
    "canvas_width": 1280,
    "canvas_height": 640,

    # Average polygon width/height. Smaller values create more polygons.
    # 这里控制普通输出 cells/islands 中“单个泰森多边形”的平均长宽。
    # polygon_width / polygon_height 越大，多边形越大、数量越少；
    # polygon_width / polygon_height 越小，多边形越小、数量越多。
    # 近似单个多边形面积可以理解为 polygon_width * polygon_height。
    "polygon_width": 80,
    "polygon_height": 70,

    # "islands": separated polygons on matrix background, like pores/particles.
    # "cells": full Voronoi cells with colored walls, like a honeycomb network.
    # "mixed_overlay": sharp Voronoi outlines + smooth inset outlines on one white canvas.
    # "both": generate all example styles.
    "mode": "both",

    # In islands mode: target filled polygon area / total image area.
    # In cells mode this is ignored because cells cover the full image.
    # 这里控制 islands 输出中“浅色多边形总面积 / 整张图面积”的目标占比。
    # 例如 0.42 表示浅色多边形大约占整张图 42%，红色基底约占 58%。
    # 数值越大，浅色多边形越胖、间隙越窄；数值越小，浅色多边形越瘦、间隙越宽。
    # 注意：cells 输出是完整泰森网格，默认铺满整张图，所以不使用这个占比参数。
    "target_polygon_area_ratio": 0.3,

    # Line/wall style.
    "wall_thickness": 5,
    "outline_thickness": 3,
    "mixed_sharp_thickness": 2,
    "mixed_smooth_thickness": 2,
    # True 时，mixed_overlay 也使用 polygon_width / polygon_height / position_jitter。
    # 这能保证你修改主参数后，所有输出图都会一起变化。
    # False 时，mixed_overlay 才使用下面的 mixed_polygon_width / mixed_polygon_height。
    "use_main_polygon_size_for_mixed": True,
    # mixed_overlay 是你要的“红色锐利外多边形 + 绿色平滑内缩线”叠加图。
    # 下面两个参数只在 use_main_polygon_size_for_mixed = False 时生效。
    # 它们控制 mixed_overlay 中外层红色泰森多边形的大致长宽。
    # 改小：外层红色多边形更密、更小；改大：外层红色多边形更稀、更大。
    "mixed_polygon_width": 44,
    "mixed_polygon_height": 36,
    "mixed_position_jitter": 0.38,
    # mixed_inner_offset 控制绿色平滑封闭曲线相对红色外多边形向内缩多少像素。
    # 数值越大，绿色内部曲线面积越小，红绿之间的距离越宽；
    # 数值越小，绿色内部曲线面积越大，更贴近红色外多边形。
    # 这个参数是 mixed_overlay 里最接近“绿色内部曲线占比”的控制项。
    "mixed_inner_offset": 7,
    # mixed_round_radius_ratio / mixed_smooth_iterations 控制绿色内部曲线的圆滑程度。
    # 数值越大，绿色线越圆润；太大时可能会让小多边形消失或形状过圆。
    "mixed_round_radius_ratio": 0.30,
    "mixed_smooth_iterations": 4,

    # Randomness. Set a fixed number for repeatable output.
    "random_seed": 20260603,
    "position_jitter": 0.36,
    "shape_jitter": 0.04,

    # Colors are BGR for OpenCV.
    "background_color": (77, 79, 174),      # reddish matrix
    "polygon_color": (154, 191, 213),      # tan polygons
    "wall_color": (66, 66, 160),           # dark red wall
    "outline_color": (48, 49, 92),         # dark outline
    "mixed_background_color": (255, 255, 255),
    "mixed_sharp_color": (105, 105, 245),  # red/pink sharp Voronoi lines
    "mixed_smooth_color": (75, 210, 75),   # green smooth inset lines

    # File names.
    "output_islands": "voronoi_islands1.png",
    "output_cells": "voronoi_cells1.png",
    "output_preview": "voronoi_preview1.png",
    "output_mixed": "voronoi_mixed_overlay1.png",
}


def save_png(path: Path, image: np.ndarray) -> None:
    """Save PNG reliably on Windows paths containing non-ASCII characters."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def voronoi_finite_polygons_2d(vor: Voronoi, radius: float | None = None):
    """Reconstruct infinite Voronoi regions into finite polygons."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        polygon_points = np.asarray([new_vertices[v] for v in new_region])
        centroid = polygon_points.mean(axis=0)
        angles = np.arctan2(
            polygon_points[:, 1] - centroid[1],
            polygon_points[:, 0] - centroid[0],
        )
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def generate_seed_points(width: int, height: int, cell_w: float, cell_h: float, rng: np.random.Generator, jitter: float):
    cols = math.ceil(width / cell_w) + 4
    rows = math.ceil(height / cell_h) + 4
    points = []
    for row in range(rows):
        for col in range(cols):
            x = (col - 2 + 0.5) * cell_w
            y = (row - 2 + 0.5) * cell_h
            x += rng.uniform(-cell_w * jitter, cell_w * jitter)
            y += rng.uniform(-cell_h * jitter, cell_h * jitter)
            points.append((x, y))
    return np.asarray(points, dtype=np.float64)


def make_cells(config: dict) -> list[Polygon]:
    width = int(config["canvas_width"])
    height = int(config["canvas_height"])
    rng = np.random.default_rng(int(config["random_seed"]))
    points = generate_seed_points(
        width,
        height,
        float(config["polygon_width"]),
        float(config["polygon_height"]),
        rng,
        float(config["position_jitter"]),
    )

    regions, vertices = voronoi_finite_polygons_2d(Voronoi(points), radius=max(width, height) * 4)
    frame = box(0, 0, width, height)
    cells = []
    for region in regions:
        poly = Polygon(vertices[region]).intersection(frame)
        if not poly.is_empty and poly.area > 4:
            cells.append(poly)
    return cells


def smooth_closed_coords(coords: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin smoothing for a closed polygon outline."""
    points = np.asarray(coords, dtype=np.float32)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    for _ in range(iterations):
        next_points = np.roll(points, -1, axis=0)
        q = 0.75 * points + 0.25 * next_points
        r = 0.25 * points + 0.75 * next_points
        smoothed = np.empty((len(points) * 2, 2), dtype=np.float32)
        smoothed[0::2] = q
        smoothed[1::2] = r
        points = smoothed
    return points


def polygon_to_points(poly: Polygon, smooth_iterations: int = 0) -> np.ndarray:
    coords = np.asarray(poly.exterior.coords, dtype=np.float32)
    if smooth_iterations > 0:
        coords = smooth_closed_coords(coords, smooth_iterations)
    return np.round(coords).astype(np.int32).reshape((-1, 1, 2))


def round_polygon(poly: Polygon, radius: float) -> Polygon:
    """Round sharp corners while keeping a Voronoi-like particle footprint."""
    rounded = poly.buffer(radius, join_style=1).buffer(-radius, join_style=1)
    if rounded.is_empty:
        return poly
    if rounded.geom_type == "MultiPolygon":
        rounded = max(rounded.geoms, key=lambda item: item.area)
    return rounded


def shrink_cells_to_ratio(cells: list[Polygon], target_ratio: float, image_area: float) -> list[Polygon]:
    if target_ratio >= 0.99:
        return cells

    low, high = 0.0, 80.0
    best = cells
    for _ in range(24):
        mid = (low + high) / 2
        shrunk = []
        total = 0.0
        for cell in cells:
            poly = cell.buffer(-mid, join_style=1)
            if poly.is_empty:
                continue
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
            if poly.area > 12:
                shrunk.append(poly)
                total += poly.area
        ratio = total / image_area
        if ratio > target_ratio:
            low = mid
            best = shrunk
        else:
            high = mid
    return best


def render(config: dict, mode: str) -> np.ndarray:
    width = int(config["canvas_width"])
    height = int(config["canvas_height"])
    image = np.full((height, width, 3), config["background_color"], dtype=np.uint8)

    if mode == "mixed_overlay":
        mixed_config = dict(config)
        if not config.get("use_main_polygon_size_for_mixed", True):
            mixed_config["polygon_width"] = config["mixed_polygon_width"]
            mixed_config["polygon_height"] = config["mixed_polygon_height"]
            mixed_config["position_jitter"] = config["mixed_position_jitter"]
        cells = make_cells(mixed_config)
        image = np.full((height, width, 3), config["mixed_background_color"], dtype=np.uint8)
        inner_offset = float(config["mixed_inner_offset"])
        smooth_radius = min(float(mixed_config["polygon_width"]), float(mixed_config["polygon_height"])) * float(
            config["mixed_round_radius_ratio"]
        )

        for cell in cells:
            cv2.polylines(
                image,
                [polygon_to_points(cell)],
                True,
                config["mixed_sharp_color"],
                int(config["mixed_sharp_thickness"]),
                lineType=cv2.LINE_AA,
            )

        for cell in cells:
            inset = cell.buffer(-inner_offset, join_style=1)
            if inset.is_empty:
                continue
            if inset.geom_type == "MultiPolygon":
                inset = max(inset.geoms, key=lambda item: item.area)
            if inset.area <= 10:
                continue
            inset = round_polygon(inset, radius=smooth_radius)
            inset_points = polygon_to_points(inset, smooth_iterations=int(config["mixed_smooth_iterations"]))
            cv2.polylines(
                image,
                [inset_points],
                True,
                config["mixed_smooth_color"],
                int(config["mixed_smooth_thickness"]),
                lineType=cv2.LINE_AA,
            )
        return image

    cells = make_cells(config)
    if mode == "cells":
        for cell in cells:
            cv2.fillPoly(image, [polygon_to_points(cell)], config["polygon_color"], lineType=cv2.LINE_AA)
        for cell in cells:
            cv2.polylines(
                image,
                [polygon_to_points(cell)],
                True,
                config["wall_color"],
                int(config["wall_thickness"]),
                lineType=cv2.LINE_AA,
            )
        return image

    target_ratio = float(config["target_polygon_area_ratio"])
    cells = shrink_cells_to_ratio(cells, target_ratio, width * height)
    rng = np.random.default_rng(int(config["random_seed"]) + 17)
    for cell in cells:
        cell = round_polygon(cell, radius=min(float(config["polygon_width"]), float(config["polygon_height"])) * 0.18)
        # A tiny random buffer gives less perfectly mechanical edges.
        delta = rng.uniform(-float(config["shape_jitter"]), float(config["shape_jitter"])) * min(
            float(config["polygon_width"]),
            float(config["polygon_height"]),
        )
        poly = cell.buffer(delta, join_style=1)
        if poly.is_empty:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        points = polygon_to_points(poly, smooth_iterations=2)
        cv2.fillPoly(image, [points], config["polygon_color"], lineType=cv2.LINE_AA)
        cv2.polylines(
            image,
            [points],
            True,
            config["outline_color"],
            int(config["outline_thickness"]),
            lineType=cv2.LINE_AA,
        )
    return image


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    mode = str(CONFIG["mode"]).lower()
    if mode not in {"islands", "cells", "mixed_overlay", "both"}:
        raise ValueError('CONFIG["mode"] must be "islands", "cells", "mixed_overlay", or "both"')

    print(
        "CONFIG:",
        f"polygon_width={CONFIG['polygon_width']}",
        f"polygon_height={CONFIG['polygon_height']}",
        f"target_polygon_area_ratio={CONFIG['target_polygon_area_ratio']}",
        f"use_main_polygon_size_for_mixed={CONFIG['use_main_polygon_size_for_mixed']}",
        f"mixed_inner_offset={CONFIG['mixed_inner_offset']}",
    )

    rendered = {}
    modes = ["islands", "cells", "mixed_overlay"] if mode == "both" else [mode]
    filenames = {
        "islands": CONFIG["output_islands"],
        "cells": CONFIG["output_cells"],
        "mixed_overlay": CONFIG["output_mixed"],
    }
    for item in modes:
        image = render(CONFIG, item)
        rendered[item] = image
        filename = filenames[item]
        save_png(out_dir / filename, image)
        print(f"saved: {out_dir / filename}")

    if mode == "both":
        cells = rendered["cells"]
        islands = rendered["islands"]
        width = cells.shape[1]
        preview = cells.copy()
        preview[:, width // 2 :] = islands[:, width // 2 :]
        preview_path = out_dir / str(CONFIG["output_preview"])
        save_png(preview_path, preview)
        print(f"saved: {preview_path}")


if __name__ == "__main__":
    main()
