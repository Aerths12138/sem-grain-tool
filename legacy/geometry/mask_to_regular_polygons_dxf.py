import argparse
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import feature, segmentation


ROOT = Path(__file__).resolve().parent


def imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def save_png_unicode(path: Path, img: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, data = cv2.imencode(suffix, img)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    data.tofile(str(path))


def extract_target_mask(img_bgr: np.ndarray, mode: str) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    colored = (saturation > 35) & (value > 45)
    white = (gray > 180) & (saturation < 45)

    if mode == "color":
        mask = colored
    elif mode == "white":
        mask = white
    else:
        # Bright white on a dark background is a target. On screenshot-like
        # white backgrounds, white would select the page, so use color only.
        bright_ratio = float((gray > 220).mean())
        if bright_ratio > 0.35 and colored.mean() > 0.002:
            mask = colored
        else:
            mask = colored | white

    return mask.astype(np.uint8) * 255


def smooth_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(smoothed, (k, k), 0)
    _, out = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return out


def split_touching_regions(mask: np.ndarray, min_distance: int) -> list[np.ndarray]:
    if min_distance <= 0:
        return [mask]

    binary = mask > 0
    distance = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary,
        exclude_border=False,
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    for idx, (row, col) in enumerate(coords, start=1):
        markers[row, col] = idx
    markers = ndi.label(markers > 0)[0]
    if markers.max() == 0:
        return [mask]

    labels = segmentation.watershed(-distance, markers, mask=binary)
    return [(labels == label_id).astype(np.uint8) * 255 for label_id in range(1, int(labels.max()) + 1)]


def contours_to_polygons(mask: np.ndarray, epsilon_ratio: float, min_area: float, polygon_mode: str, split_distance: int) -> list[np.ndarray]:
    contours = []
    for part in split_touching_regions(mask, split_distance):
        part_contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours.extend(part_contours)

    polygons: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        source = cv2.convexHull(contour) if polygon_mode == "hull" else contour
        perimeter = cv2.arcLength(source, True)
        epsilon = max(1.0, epsilon_ratio * perimeter)
        poly = cv2.approxPolyDP(source, epsilon, True).reshape(-1, 2)
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons


def write_dxf(path: Path, polygons: list[np.ndarray], height: int, scale: float) -> None:
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    doc.layers.new("SEM_POLYGONS", dxfattribs={"color": 7})

    for poly in polygons:
        points = [(float(x) * scale, float(height - y) * scale) for x, y in poly]
        msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "SEM_POLYGONS"})

    doc.saveas(path)


def draw_preview(img_bgr: np.ndarray, mask: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for poly in polygons:
        cv2.polylines(preview, [poly.astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
    return preview


def process_image(args: argparse.Namespace, input_path: Path) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    stem = input_path.stem.replace(" ", "_")
    img = imread_unicode(input_path)
    raw_mask = extract_target_mask(img, args.target)
    mask = smooth_mask(raw_mask, args.smooth)
    polygons = contours_to_polygons(mask, args.epsilon, args.min_area, args.polygon_mode, args.split_distance)

    mask_path = output_dir / f"{stem}_binary_mask.png"
    preview_path = output_dir / f"{stem}_polygon_preview.png"
    dxf_path = output_dir / f"{stem}_polygons.dxf"

    save_png_unicode(mask_path, mask)
    save_png_unicode(preview_path, draw_preview(img, mask, polygons))
    write_dxf(dxf_path, polygons, mask.shape[0], args.scale)

    vertices = sum(len(poly) for poly in polygons)
    print(f"{input_path}")
    print(f"  polygons={len(polygons)}, vertices={vertices}")
    print(f"  mask={mask_path}")
    print(f"  preview={preview_path}")
    print(f"  dxf={dxf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-dir", default=str(ROOT / "dxf_polygon_outputs"))
    parser.add_argument("--target", choices=["auto", "color", "white"], default="auto")
    parser.add_argument("--polygon-mode", choices=["contour", "hull"], default="contour")
    parser.add_argument("--smooth", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.015)
    parser.add_argument("--min-area", type=float, default=80)
    parser.add_argument("--split-distance", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    for item in args.inputs:
        process_image(args, Path(item))


if __name__ == "__main__":
    main()
