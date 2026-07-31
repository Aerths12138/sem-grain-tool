"""
Generate closed-curve textures made from non-overlapping rectangles.

This is a companion script to generate_voronoi_texture.py and
generate_ellipse_texture.py. It replaces polygons/ellipses with rectangular
closed shapes.

Run:
    python generate_rectangle_texture.py
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


CONFIG = {
    # Canvas size in pixels.
    "canvas_width": 1280,
    "canvas_height": 640,

    # Nominal rectangle size.
    # Larger values produce fewer, larger rectangles.
    "rect_width": 42,
    "rect_height": 28,

    # Target total filled rectangle area / full image area.
    # Non-overlap placement may stop below this if the requested ratio is too high.
    "target_area_ratio": 0.32,

    # Placement mode:
    # - "non_overlap": rejection sampling with collision checks
    # - "grid": fast grid+jitter placement, may overlap
    "placement_mode": "non_overlap",

    # Minimum gap between rectangles in pixels for non_overlap mode.
    "min_gap": 3.0,

    # Stop trying after this many rejected candidates.
    "max_attempts": 20000,

    # Randomness.
    "random_seed": 20260611,
    "position_jitter": 0.35,
    "size_jitter": 0.25,

    # Rotation control.
    # 0 means axis-aligned rectangles. 180 allows arbitrary random rotation.
    "angle_jitter_degrees": 180,

    # Collision method:
    # - "precise": checks rotated rectangle intersection and gap, denser but slower
    # - "bounding_circle": conservative and faster, less dense
    "collision_method": "precise",

    # Drawing style.
    "outline_thickness": 2,

    # Output mode:
    # - "filled": filled rectangles on matrix background
    # - "outline": outline-only rectangles on white background
    # - "both": generate both
    "mode": "both",

    # Colors are BGR for OpenCV.
    "background_color": (77, 79, 174),
    "rectangle_color": (154, 191, 213),
    "outline_color": (48, 49, 92),
    "outline_background_color": (255, 255, 255),
    "outline_curve_color": (75, 210, 75),

    # File names.
    "output_filled": "rectangle_filled.png",
    "output_outline": "rectangle_outline.png",
}


def save_png(path: Path, image: np.ndarray) -> None:
    """Save PNG reliably on Windows paths containing non-ASCII characters."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def rectangle_points(cx: float, cy: float, width: float, height: float, angle_deg: float) -> np.ndarray:
    hw = width / 2.0
    hh = height / 2.0
    corners = np.asarray(
        [
            (-hw, -hh),
            (hw, -hh),
            (hw, hh),
            (-hw, hh),
        ],
        dtype=np.float32,
    )
    angle = math.radians(angle_deg)
    ca = math.cos(angle)
    sa = math.sin(angle)
    rot = np.asarray([[ca, -sa], [sa, ca]], dtype=np.float32)
    points = corners @ rot.T
    points[:, 0] += cx
    points[:, 1] += cy
    return points


def points_to_int_contour(points: np.ndarray) -> np.ndarray:
    return np.round(points).astype(np.int32).reshape((-1, 1, 2))


def polygon_area(points: np.ndarray) -> float:
    return float(cv2.contourArea(points_to_int_contour(points)))


def point_to_segment_distance(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    ab = seg_b - seg_a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - seg_a))
    t = float(np.dot(point - seg_a, ab) / denom)
    t = max(0.0, min(1.0, t))
    closest = seg_a + t * ab
    return float(np.linalg.norm(point - closest))


def segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def polygons_overlap_or_too_close(points_a: np.ndarray, points_b: np.ndarray, min_gap: float) -> bool:
    contour_a = points_a.reshape((-1, 1, 2)).astype(np.float32)
    contour_b = points_b.reshape((-1, 1, 2)).astype(np.float32)
    if cv2.pointPolygonTest(contour_a, tuple(points_b[0]), False) >= 0:
        return True
    if cv2.pointPolygonTest(contour_b, tuple(points_a[0]), False) >= 0:
        return True

    a_next = np.roll(points_a, -1, axis=0)
    b_next = np.roll(points_b, -1, axis=0)
    min_dist = float("inf")
    for a0, a1 in zip(points_a, a_next):
        for b0, b1 in zip(points_b, b_next):
            if segments_intersect(a0, a1, b0, b1):
                return True
            min_dist = min(
                min_dist,
                point_to_segment_distance(a0, b0, b1),
                point_to_segment_distance(a1, b0, b1),
                point_to_segment_distance(b0, a0, a1),
                point_to_segment_distance(b1, a0, a1),
            )
            if min_dist < min_gap:
                return True
    return False


def make_one_rectangle(rng: np.random.Generator, config: dict, cx: float, cy: float) -> dict:
    scale_value = 1.0 + rng.uniform(-float(config["size_jitter"]), float(config["size_jitter"]))
    width = float(config["rect_width"]) * scale_value
    height = float(config["rect_height"]) * scale_value
    angle = rng.uniform(-float(config["angle_jitter_degrees"]), float(config["angle_jitter_degrees"]))
    points = rectangle_points(cx, cy, width, height, angle)
    return {
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "angle": angle,
        "bound_radius": math.hypot(width, height) / 2.0,
        "points_float": points,
        "points_int": points_to_int_contour(points),
        "area": width * height,
    }


def rectangle_inside_canvas(candidate: dict, width: float, height: float, margin: float) -> bool:
    points = candidate["points_float"]
    return (
        float(points[:, 0].min()) >= margin
        and float(points[:, 0].max()) <= width - margin
        and float(points[:, 1].min()) >= margin
        and float(points[:, 1].max()) <= height - margin
    )


def collides(candidate: dict, placed: list[dict], min_gap: float, method: str) -> bool:
    cx = candidate["cx"]
    cy = candidate["cy"]
    radius = candidate["bound_radius"] + min_gap
    for item in placed:
        center_dist = math.hypot(cx - item["cx"], cy - item["cy"])
        if center_dist > radius + item["bound_radius"] + min_gap:
            continue
        if method == "bounding_circle":
            return True
        if polygons_overlap_or_too_close(candidate["points_float"], item["points_float"], min_gap):
            return True
    return False


def grid_key(x: float, y: float, cell_size: float) -> tuple[int, int]:
    return (int(x // cell_size), int(y // cell_size))


def nearby_items(candidate: dict, grid: dict[tuple[int, int], list[int]], placed: list[dict], cell_size: float) -> list[dict]:
    gx, gy = grid_key(candidate["cx"], candidate["cy"], cell_size)
    search_radius = int(math.ceil((candidate["bound_radius"] * 2.5) / cell_size)) + 1
    result = []
    seen = set()
    for ix in range(gx - search_radius, gx + search_radius + 1):
        for iy in range(gy - search_radius, gy + search_radius + 1):
            for index in grid.get((ix, iy), []):
                if index in seen:
                    continue
                seen.add(index)
                result.append(placed[index])
    return result


def add_to_grid(item: dict, index: int, grid: dict[tuple[int, int], list[int]], cell_size: float) -> None:
    key = grid_key(item["cx"], item["cy"], cell_size)
    grid.setdefault(key, []).append(index)


def make_grid_rectangles(config: dict) -> list[np.ndarray]:
    rng = np.random.default_rng(int(config["random_seed"]))
    width = float(config["canvas_width"])
    height = float(config["canvas_height"])
    cell_w = float(config["rect_width"]) * 1.35
    cell_h = float(config["rect_height"]) * 1.35
    jitter = float(config["position_jitter"])

    cols = math.ceil(width / cell_w)
    rows = math.ceil(height / cell_h)
    rectangles = []
    for row in range(rows):
        for col in range(cols):
            cx = (col + 0.5) * cell_w + rng.uniform(-cell_w * jitter, cell_w * jitter)
            cy = (row + 0.5) * cell_h + rng.uniform(-cell_h * jitter, cell_h * jitter)
            item = make_one_rectangle(rng, config, cx, cy)
            if rectangle_inside_canvas(item, width, height, 0.0):
                rectangles.append(item["points_int"])
    return rectangles


def make_non_overlapping_rectangles(config: dict) -> list[np.ndarray]:
    rng = np.random.default_rng(int(config["random_seed"]) + 17)
    width = float(config["canvas_width"])
    height = float(config["canvas_height"])
    target_total_area = width * height * float(config["target_area_ratio"])
    min_gap = float(config["min_gap"])
    max_attempts = int(config["max_attempts"])
    method = str(config["collision_method"])
    margin = math.hypot(float(config["rect_width"]), float(config["rect_height"])) / 2.0 * 1.35 + min_gap
    grid_cell_size = max(math.hypot(float(config["rect_width"]), float(config["rect_height"])) * 1.5, 16.0)

    placed: list[dict] = []
    grid: dict[tuple[int, int], list[int]] = {}
    current_area = 0.0
    attempts = 0

    while current_area < target_total_area and attempts < max_attempts:
        attempts += 1
        cx = rng.uniform(margin, width - margin)
        cy = rng.uniform(margin, height - margin)
        candidate = make_one_rectangle(rng, config, cx, cy)
        if not rectangle_inside_canvas(candidate, width, height, min_gap):
            continue
        candidates_to_check = nearby_items(candidate, grid, placed, grid_cell_size)
        if collides(candidate, candidates_to_check, min_gap, method):
            continue
        add_to_grid(candidate, len(placed), grid, grid_cell_size)
        placed.append(candidate)
        current_area += candidate["area"]

    actual_ratio = current_area / (width * height)
    print(
        "non_overlap:",
        f"rectangles={len(placed)}",
        f"target_area_ratio={float(config['target_area_ratio']):.3f}",
        f"actual_area_ratio={actual_ratio:.3f}",
        f"attempts={attempts}",
    )
    return [item["points_int"] for item in placed]


def make_rectangles(config: dict) -> list[np.ndarray]:
    if str(config["placement_mode"]).lower() == "grid":
        return make_grid_rectangles(config)
    return make_non_overlapping_rectangles(config)


def render(config: dict, mode: str) -> np.ndarray:
    width = int(config["canvas_width"])
    height = int(config["canvas_height"])
    rectangles = make_rectangles(config)

    if mode == "outline":
        image = np.full((height, width, 3), config["outline_background_color"], dtype=np.uint8)
        for item in rectangles:
            cv2.polylines(
                image,
                [item],
                True,
                config["outline_curve_color"],
                int(config["outline_thickness"]),
                lineType=cv2.LINE_AA,
            )
        return image

    image = np.full((height, width, 3), config["background_color"], dtype=np.uint8)
    for item in rectangles:
        cv2.fillPoly(image, [item], config["rectangle_color"], lineType=cv2.LINE_AA)
        cv2.polylines(
            image,
            [item],
            True,
            config["outline_color"],
            int(config["outline_thickness"]),
            lineType=cv2.LINE_AA,
        )
    return image


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    mode = str(CONFIG["mode"]).lower()
    if mode not in {"filled", "outline", "both"}:
        raise ValueError('CONFIG["mode"] must be "filled", "outline", or "both"')

    print(
        "CONFIG:",
        f"rect_width={CONFIG['rect_width']}",
        f"rect_height={CONFIG['rect_height']}",
        f"target_area_ratio={CONFIG['target_area_ratio']}",
        f"placement_mode={CONFIG['placement_mode']}",
    )

    modes = ["filled", "outline"] if mode == "both" else [mode]
    for item in modes:
        image = render(CONFIG, item)
        filename = CONFIG["output_filled"] if item == "filled" else CONFIG["output_outline"]
        save_png(out_dir / filename, image)
        print(f"saved: {out_dir / filename}")


if __name__ == "__main__":
    main()
