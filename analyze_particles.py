import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from skimage import measure


ROOT = Path(__file__).resolve().parent


def equivalent_diameter(area: float) -> float:
    return 2.0 * math.sqrt(area / math.pi)


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def particle_rows(labels: np.ndarray, um_per_px: float | None) -> list[dict[str, float | int]]:
    rows = []
    for region in measure.regionprops(labels):
        area_px = float(region.area)
        diameter_px = equivalent_diameter(area_px)
        center_y, center_x = region.centroid
        min_row, min_col, max_row, max_col = region.bbox

        row: dict[str, float | int] = {
            "particle_id": int(region.label),
            "center_x_px": round(float(center_x), 3),
            "center_y_px": round(float(center_y), 3),
            "area_px": int(region.area),
            "equivalent_diameter_px": round(diameter_px, 3),
            "bbox_min_x_px": int(min_col),
            "bbox_min_y_px": int(min_row),
            "bbox_max_x_px": int(max_col - 1),
            "bbox_max_y_px": int(max_row - 1),
        }
        if um_per_px is not None:
            row["area_um2"] = round(area_px * um_per_px * um_per_px, 6)
            row["equivalent_diameter_um"] = round(diameter_px * um_per_px, 6)
            row["center_x_um"] = round(float(center_x) * um_per_px, 6)
            row["center_y_um"] = round(float(center_y) * um_per_px, 6)
        rows.append(row)
    return rows


def write_particles_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(
    particles: list[dict[str, float | int]],
    bins: int,
    use_um: bool,
) -> tuple[list[dict[str, float | int]], np.ndarray, np.ndarray]:
    key = "equivalent_diameter_um" if use_um else "equivalent_diameter_px"
    values = np.array([float(p[key]) for p in particles], dtype=float)
    if len(values) == 0:
        return [], np.array([]), np.array([])

    hist, edges = np.histogram(values, bins=bins)
    total = int(hist.sum())
    rows = []
    unit = "um" if use_um else "px"
    for i, count in enumerate(hist):
        rows.append({
            f"diameter_min_{unit}": round(float(edges[i]), 6),
            f"diameter_max_{unit}": round(float(edges[i + 1]), 6),
            "count": int(count),
            "percent": round(float(count) / total * 100.0, 3) if total else 0.0,
        })
    return rows, hist, edges


def write_distribution_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_histogram(path: Path, hist: np.ndarray, edges: np.ndarray, unit: str) -> None:
    width, height = 900, 560
    margin_left, margin_right, margin_top, margin_bottom = 80, 35, 45, 90
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if len(hist) == 0:
        Image.fromarray(canvas).save(path)
        return

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_count = max(int(hist.max()), 1)

    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (40, 40, 40), 2)
    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (40, 40, 40), 2)
    cv2.putText(canvas, f"Particle diameter distribution ({unit})", (margin_left, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    bar_gap = 5
    bar_w = max(8, int((plot_w - bar_gap * (len(hist) - 1)) / len(hist)))
    for i, count in enumerate(hist):
        x0 = margin_left + i * (bar_w + bar_gap)
        x1 = x0 + bar_w
        bar_h = int((int(count) / max_count) * plot_h)
        y0 = height - margin_bottom - bar_h
        y1 = height - margin_bottom
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 145, 210), -1)
        cv2.putText(canvas, str(int(count)), (x0, max(margin_top + 18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        label = f"{edges[i]:.1f}-{edges[i + 1]:.1f}"
        cv2.putText(canvas, label, (x0, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)

    Image.fromarray(canvas).save(path)


def save_center_overlay(path: Path, gray: np.ndarray, labels: np.ndarray, particles: list[dict[str, float | int]]) -> None:
    overlay = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    rng = np.random.default_rng(23)
    for particle in particles:
        label_id = int(particle["particle_id"])
        region = labels == label_id
        color = rng.integers(70, 255, size=3)
        overlay[region] = overlay[region] * 0.45 + color * 0.55

    out = np.clip(overlay, 0, 255).astype(np.uint8)
    for particle in particles:
        x = int(round(float(particle["center_x_px"])))
        y = int(round(float(particle["center_y_px"])))
        label_id = int(particle["particle_id"])
        cv2.drawMarker(out, (x, y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        cv2.putText(out, str(label_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(out, str(label_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    Image.fromarray(out).save(path)


def write_summary(path: Path, particles: list[dict[str, float | int]], use_um: bool) -> None:
    diam_key = "equivalent_diameter_um" if use_um else "equivalent_diameter_px"
    area_key = "area_um2" if use_um else "area_px"
    diameters = np.array([float(p[diam_key]) for p in particles], dtype=float)
    areas = np.array([float(p[area_key]) for p in particles], dtype=float)
    summary = {
        "particle_count": int(len(particles)),
        "diameter_unit": "um" if use_um else "px",
        "area_unit": "um2" if use_um else "px",
        "diameter_min": round(float(diameters.min()), 6) if len(diameters) else 0,
        "diameter_median": round(float(np.median(diameters)), 6) if len(diameters) else 0,
        "diameter_mean": round(float(diameters.mean()), 6) if len(diameters) else 0,
        "diameter_max": round(float(diameters.max()), 6) if len(diameters) else 0,
        "area_min": round(float(areas.min()), 6) if len(areas) else 0,
        "area_median": round(float(np.median(areas)), 6) if len(areas) else 0,
        "area_mean": round(float(areas.mean()), 6) if len(areas) else 0,
        "area_max": round(float(areas.max()), 6) if len(areas) else 0,
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="1_i313")
    parser.add_argument("--mask", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--um-per-px", type=float, default=None)
    args = parser.parse_args()

    mask_path = Path(args.mask) if args.mask else ROOT / "cellpose_results_pretrained" / f"{args.sample}_sem_cp_masks.tif"
    image_path = Path(args.image) if args.image else ROOT / "cellpose_dataset" / f"{args.sample}_sem.png"
    out_dir = ROOT / "particle_analysis"
    out_dir.mkdir(exist_ok=True)

    labels = tifffile.imread(mask_path).astype(np.int32)
    gray = load_gray(image_path)
    particles = particle_rows(labels, args.um_per_px)
    use_um = args.um_per_px is not None
    dist_rows, hist, edges = distribution_rows(particles, args.bins, use_um)
    unit = "um" if use_um else "px"

    particle_csv = out_dir / f"{args.sample}_particles.csv"
    distribution_csv = out_dir / f"{args.sample}_diameter_distribution.csv"
    histogram_png = out_dir / f"{args.sample}_diameter_distribution.png"
    center_overlay_png = out_dir / f"{args.sample}_particle_centers.png"
    summary_json = out_dir / f"{args.sample}_summary.json"

    write_particles_csv(particle_csv, particles)
    write_distribution_csv(distribution_csv, dist_rows)
    save_histogram(histogram_png, hist, edges, unit)
    save_center_overlay(center_overlay_png, gray, labels, particles)
    write_summary(summary_json, particles, use_um)

    print(f"particle_count={len(particles)}")
    print(f"saved particles: {particle_csv}")
    print(f"saved distribution: {distribution_csv}")
    print(f"saved histogram: {histogram_png}")
    print(f"saved centers: {center_overlay_png}")
    print(f"saved summary: {summary_json}")


if __name__ == "__main__":
    main()
