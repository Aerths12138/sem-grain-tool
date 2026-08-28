"""
Generate closed-curve textures made from circles or ellipses.

This is a companion script to generate_voronoi_texture.py. It keeps the same
basic parameter style, but replaces polygon cells with smooth closed curves.

Run:
    python generate_ellipse_texture.py
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

    # Average spacing between ellipse centers.
    # Larger values produce fewer, larger ellipses; smaller values produce more.
    "cell_width": 54,
    "cell_height": 44,

    # Target total filled ellipse area / full image area.
    # Example: 0.42 means ellipses occupy about 42% of the image.
    # For non-overlap placement, very high ratios may be impossible for random
    # ellipses. If placement cannot reach this value, the script prints the
    # actual achieved ratio.
    "target_area_ratio": 0.30,

    # Placement mode:
    # - "grid": old fast grid+jitter placement, may overlap
    # - "non_overlap": rejection sampling with collision checks
    "placement_mode": "non_overlap",

    # Minimum gap between ellipses in pixels for non_overlap mode.
    "min_gap": 3.0,

    # Collision method:
    # - "bounding_circle": fast and conservative; ellipses are guaranteed not to overlap
    # - "precise": slower polygon distance check, can pack denser
    "collision_method": "bounding_circle",

    # Stop trying after this many rejected candidates.
    "max_attempts": 15000,

    # Shape mode:
    # - "ellipse": variable ellipses
    # - "circle": force equal x/y radius
    "shape_mode": "ellipse",

    # Ellipse aspect ratio range, used only when shape_mode = "ellipse".
    # 1.0 is circular. Larger values create longer ellipses.
    "min_aspect_ratio": 1.0,
    "max_aspect_ratio": 1.8,

    # Randomness.
    "random_seed": 20260610,
    "position_jitter": 0.34,
    "size_jitter": 0.22,
    "angle_jitter_degrees": 180,

    # Ellipse drawing quality.
    "curve_points": 64,
    "outline_thickness": 2,

    # Output mode:
    # - "filled": filled ellipses on matrix background
    # - "outline": outline-only ellipses on white background
    # - "both": generate both
    "mode": "both",

    # Colors are BGR for OpenCV.
    "background_color": (77, 79, 174),
    "ellipse_color": (154, 191, 213),
    "outline_color": (48, 49, 92),
    "outline_background_color": (255, 255, 255),
    "outline_curve_color": (75, 210, 75),

    # File names.
    "output_filled": "ellipse_filled.png",
    "output_outline": "ellipse_outline.png",
}


def polygon_area(points: np.ndarray) -> float:
    contour = points.reshape((-1, 1, 2)).astype(np.float32)
    return float(cv2.contourArea(contour))


def save_png(path: Path, image: np.ndarray) -> None:
    """Save PNG reliably on Windows paths containing non-ASCII characters."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def generate_centers(config: dict) -> np.ndarray:
    width = int(config["canvas_width"])
    height = int(config["canvas_height"])
    cell_w = float(config["cell_width"])
    cell_h = float(config["cell_height"])
    jitter = float(config["position_jitter"])
    rng = np.random.default_rng(int(config["random_seed"]))

    cols = math.ceil(width / cell_w)
    rows = math.ceil(height / cell_h)
    centers = []
    for row in range(rows):
        for col in range(cols):
            x = (col + 0.5) * cell_w + rng.uniform(-cell_w * jitter, cell_w * jitter)
            y = (row + 0.5) * cell_h + rng.uniform(-cell_h * jitter, cell_h * jitter)
            if 0 <= x < width and 0 <= y < height:
                centers.append((x, y))
    return np.asarray(centers, dtype=np.float32)


def ellipse_polygon(cx: float, cy: float, rx: float, ry: float, angle_deg: float, point_count: int) -> np.ndarray:
    angle = math.radians(angle_deg)
    ca = math.cos(angle)
    sa = math.sin(angle)
    points = []
    for i in range(point_count):
        t = 2 * math.pi * i / point_count
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        points.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return np.round(np.asarray(points, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))


def ellipse_float_points(
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    angle_deg: float,
    point_count: int,
) -> np.ndarray:
    angle = math.radians(angle_deg)
    ca = math.cos(angle)
    sa = math.sin(angle)
    points = []
    for i in range(point_count):
        t = 2 * math.pi * i / point_count
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        points.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return np.asarray(points, dtype=np.float32)


def make_one_ellipse(
    rng: np.random.Generator,
    config: dict,
    base_radius: float,
    cx: float,
    cy: float,
) -> dict:
    size_jitter = float(config["size_jitter"])
    min_ar = float(config["min_aspect_ratio"])
    max_ar = float(config["max_aspect_ratio"])
    max_angle = float(config["angle_jitter_degrees"])

    scale_value = 1.0 + rng.uniform(-size_jitter, size_jitter)
    if str(config["shape_mode"]).lower() == "circle":
        aspect = 1.0
    else:
        aspect = rng.uniform(min_ar, max_ar)
        if rng.random() < 0.5:
            aspect = 1.0 / aspect

    rx = base_radius * math.sqrt(aspect) * scale_value
    ry = base_radius / math.sqrt(aspect) * scale_value
    angle = rng.uniform(-max_angle, max_angle)
    points_float = ellipse_float_points(cx, cy, rx, ry, angle, int(config["curve_points"]))
    points_int = np.round(points_float).astype(np.int32).reshape((-1, 1, 2))
    return {
        "cx": cx,
        "cy": cy,
        "rx": rx,
        "ry": ry,
        "bound_radius": max(rx, ry),
        "points_float": points_float,
        "points_int": points_int,
        "area": math.pi * rx * ry,
    }


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
    # OpenCV point tests catch containment even when boundaries do not cross.
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


def ellipse_inside_canvas(candidate: dict, width: float, height: float, margin: float) -> bool:
    points = candidate["points_float"]
    return (
        float(points[:, 0].min()) >= margin
        and float(points[:, 0].max()) <= width - margin
        and float(points[:, 1].min()) >= margin
        and float(points[:, 1].max()) <= height - margin
    )


def collides(candidate: dict, placed: list[dict], min_gap: float, method: str = "bounding_circle") -> bool:
    cx = candidate["cx"]
    cy = candidate["cy"]
    radius = candidate["bound_radius"] + min_gap
    for item in placed:
        center_dist = math.hypot(cx - item["cx"], cy - item["cy"])
        if center_dist <= radius + item["bound_radius"] + min_gap:
            if method == "bounding_circle":
                return True
        else:
            continue
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


def make_ellipses(config: dict) -> list[np.ndarray]:
    if str(config["placement_mode"]).lower() == "non_overlap":
        return make_non_overlapping_ellipses(config)

    centers = generate_centers(config)
    rng = np.random.default_rng(int(config["random_seed"]) + 17)
    width = float(config["canvas_width"])
    height = float(config["canvas_height"])
    target_total_area = width * height * float(config["target_area_ratio"])
    count = max(len(centers), 1)

    # Base area per ellipse. We solve radii from area = pi * rx * ry.
    base_area = target_total_area / count
    base_radius = math.sqrt(base_area / math.pi)

    ellipses = []
    size_jitter = float(config["size_jitter"])
    min_ar = float(config["min_aspect_ratio"])
    max_ar = float(config["max_aspect_ratio"])
    max_angle = float(config["angle_jitter_degrees"])

    for cx, cy in centers:
        item = make_one_ellipse(rng, config, base_radius, cx, cy)
        ellipses.append(item["points_int"])
    return ellipses


def make_non_overlapping_ellipses(config: dict) -> list[np.ndarray]:
    rng = np.random.default_rng(int(config["random_seed"]) + 17)
    width = float(config["canvas_width"])
    height = float(config["canvas_height"])
    target_total_area = width * height * float(config["target_area_ratio"])
    cell_area = float(config["cell_width"]) * float(config["cell_height"])
    nominal_count = max(int(width * height / max(cell_area, 1.0)), 1)
    base_area = target_total_area / nominal_count
    base_radius = math.sqrt(base_area / math.pi)
    min_gap = float(config["min_gap"])
    max_attempts = int(config["max_attempts"])

    placed: list[dict] = []
    grid: dict[tuple[int, int], list[int]] = {}
    current_area = 0.0
    attempts = 0
    margin = base_radius * (float(config["max_aspect_ratio"]) ** 0.5) + min_gap
    grid_cell_size = max(base_radius * 3.0, 16.0)

    while current_area < target_total_area and attempts < max_attempts:
        attempts += 1
        cx = rng.uniform(margin, width - margin)
        cy = rng.uniform(margin, height - margin)
        candidate = make_one_ellipse(rng, config, base_radius, cx, cy)
        if not ellipse_inside_canvas(candidate, width, height, min_gap):
            continue
        candidates_to_check = nearby_items(candidate, grid, placed, grid_cell_size)
        if collides(candidate, candidates_to_check, min_gap, str(config["collision_method"])):
            continue
        add_to_grid(candidate, len(placed), grid, grid_cell_size)
        placed.append(candidate)
        current_area += candidate["area"]

    actual_ratio = current_area / (width * height)
    print(
        "non_overlap:",
        f"ellipses={len(placed)}",
        f"target_area_ratio={float(config['target_area_ratio']):.3f}",
        f"actual_area_ratio={actual_ratio:.3f}",
        f"attempts={attempts}",
    )
    return [item["points_int"] for item in placed]


def render(config: dict, mode: str) -> np.ndarray:
    width = int(config["canvas_width"])
    height = int(config["canvas_height"])
    ellipses = make_ellipses(config)

    if mode == "outline":
        image = np.full((height, width, 3), config["outline_background_color"], dtype=np.uint8)
        for item in ellipses:
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
    for item in ellipses:
        cv2.fillPoly(image, [item], config["ellipse_color"], lineType=cv2.LINE_AA)
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
        f"cell_width={CONFIG['cell_width']}",
        f"cell_height={CONFIG['cell_height']}",
        f"target_area_ratio={CONFIG['target_area_ratio']}",
        f"shape_mode={CONFIG['shape_mode']}",
    )

    modes = ["filled", "outline"] if mode == "both" else [mode]
    for item in modes:
        image = render(CONFIG, item)
        filename = CONFIG["output_filled"] if item == "filled" else CONFIG["output_outline"]
        save_png(out_dir / filename, image)
        print(f"saved: {out_dir / filename}")


if __name__ == "__main__":
    main()
