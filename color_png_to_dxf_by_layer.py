"""
Convert a colored Voronoi PNG into a COMSOL-friendly DXF.

This version does not skeletonize colored lines. For COMSOL import, closed
regions are usually more reliable than fragmented center lines, so the default
mode extracts the tan cell regions as closed LWPOLYLINE entities.

Requirements:
    pip install ezdxf opencv-python numpy

Run in the image folder:
    python color_png_to_dxf_by_layer.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import ezdxf
import numpy as np


# -------------------------- adjustable parameters --------------------------
# Input image. For your current task this is the Voronoi cell PNG.
input_image_path = "voronoi_cells1.png"

# Output DXF.
output_dxf_path = "voronoi_cells1_closed_cells.dxf"

# Export mode:
# - "tan_cells": export each tan cell as one closed polyline. Recommended for COMSOL.
# - "red_walls": export outlines of the red wall regions as closed polylines.
export_mode = "tan_cells"

# Coordinate scale. 1.0 means 1 pixel = 1 CAD unit.
scale = 1.0

# Polyline simplification in pixels.
# Smaller = closer to PNG, more vertices. Larger = cleaner/smaller DXF.
contour_epsilon = 1.2

# Remove tiny regions/noise.
min_contour_area = 80.0

# If True, writes debug masks next to the input PNG.
save_debug_masks = True

# Color targets are BGR because OpenCV reads images as BGR.
# These match the generated voronoi_cells*.png colors.
tan_color_bgr = np.array([154, 191, 213], dtype=np.int16)
red_color_bgr = np.array([66, 66, 160], dtype=np.int16)

# Color tolerance for anti-aliased PNG pixels.
# Increase if contours are missing; decrease if red walls leak into tan cells.
tan_color_tolerance = 58
red_color_tolerance = 75

# Optional frame rectangle around the entire image. Useful if COMSOL should see
# the full rectangular domain explicitly.
add_outer_frame = True
frame_layer = "IMAGE_FRAME"
# ---------------------------------------------------------------------------


LAYER_DEFS = {
    "tan_cells": {
        "layer": "CELL_CLOSED_POLYGONS",
        "aci_color": 2,  # yellow
    },
    "red_walls": {
        "layer": "RED_WALL_OUTLINES",
        "aci_color": 1,  # red
    },
    "frame": {
        "layer": frame_layer,
        "aci_color": 7,  # white/black depending on CAD background
    },
}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def color_distance_mask(image: np.ndarray, target_bgr: np.ndarray, tolerance: int) -> np.ndarray:
    diff = image.astype(np.int16) - target_bgr.reshape((1, 1, 3))
    distance = np.sqrt(np.sum(diff * diff, axis=2))
    return (distance <= tolerance).astype(np.uint8) * 255


def build_mask(image: np.ndarray) -> np.ndarray:
    if export_mode == "tan_cells":
        mask = color_distance_mask(image, tan_color_bgr, tan_color_tolerance)
        # Fill small anti-aliased gaps along cell edges.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return mask

    if export_mode == "red_walls":
        mask = color_distance_mask(image, red_color_bgr, red_color_tolerance)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return mask

    raise ValueError('export_mode must be "tan_cells" or "red_walls"')


def contour_to_cad_points(contour: np.ndarray, image_height: int) -> list[tuple[float, float]]:
    approx = cv2.approxPolyDP(contour, epsilon=contour_epsilon, closed=True)
    points = []
    for point in approx:
        x = float(point[0][0]) * scale
        y = float(image_height - point[0][1]) * scale
        points.append((x, y))
    return points


def add_layer_if_missing(doc: ezdxf.EzDxf, name: str, color: int) -> None:
    if name not in doc.layers:
        doc.layers.add(name, color=color)


def add_closed_polyline(msp, layer: str, color: int, points: list[tuple[float, float]]) -> None:
    if len(points) < 3:
        return
    # Remove duplicate final point if OpenCV returned one.
    if points[0] == points[-1]:
        points = points[:-1]
    if len(points) < 3:
        return
    msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer, "color": color})


def main() -> None:
    input_path = Path(input_image_path)
    output_path = Path(output_dxf_path)

    image = read_image(input_path)
    image_h, image_w = image.shape[:2]
    mask = build_mask(image)

    if save_debug_masks:
        save_png(input_path.with_name(f"debug_{export_mode}_mask.png"), mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    doc = ezdxf.new(dxfversion="R2010")
    layer_def = LAYER_DEFS[export_mode]
    add_layer_if_missing(doc, layer_def["layer"], layer_def["aci_color"])
    add_layer_if_missing(doc, LAYER_DEFS["frame"]["layer"], LAYER_DEFS["frame"]["aci_color"])
    msp = doc.modelspace()

    kept = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        points = contour_to_cad_points(contour, image_h)
        add_closed_polyline(msp, layer_def["layer"], layer_def["aci_color"], points)
        kept += 1

    if add_outer_frame:
        frame_points = [
            (0.0, 0.0),
            (image_w * scale, 0.0),
            (image_w * scale, image_h * scale),
            (0.0, image_h * scale),
        ]
        add_closed_polyline(
            msp,
            LAYER_DEFS["frame"]["layer"],
            LAYER_DEFS["frame"]["aci_color"],
            frame_points,
        )

    doc.saveas(output_path)
    print(f"mode: {export_mode}")
    print(f"closed polylines: {kept}")
    print(f"saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
