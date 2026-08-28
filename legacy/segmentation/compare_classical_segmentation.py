import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation

from analyze_particles import particle_rows
from make_cellpose_overlay import overlay_labels
from sem_grain_app import find_sem_crop, read_sem_image


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "classical_segmentation_results"


def relabel(labels: np.ndarray) -> np.ndarray:
    return measure.label(labels > 0, connectivity=2).astype(np.int32)


def clean_binary(binary: np.ndarray, min_area: int, hole_area: int, close_radius: int) -> np.ndarray:
    cleaned = morphology.remove_small_objects(binary.astype(bool), min_size=min_area)
    cleaned = morphology.remove_small_holes(cleaned, area_threshold=hole_area)
    if close_radius > 0:
        cleaned = morphology.binary_closing(cleaned, morphology.disk(close_radius))
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=hole_area)
    return cleaned


def imagej_otsu_labels(gray: np.ndarray, min_area: int, hole_area: int, close_radius: int, blur_sigma: float) -> np.ndarray:
    work = filters.gaussian(gray, sigma=blur_sigma, preserve_range=True).astype(np.uint8) if blur_sigma > 0 else gray
    threshold = filters.threshold_otsu(work)
    binary = work > threshold
    binary = clean_binary(binary, min_area, hole_area, close_radius)
    return relabel(binary)


def watershed_labels(
    gray: np.ndarray,
    min_area: int,
    hole_area: int,
    close_radius: int,
    blur_sigma: float,
    min_distance: int,
) -> np.ndarray:
    work = filters.gaussian(gray, sigma=blur_sigma, preserve_range=True).astype(np.uint8) if blur_sigma > 0 else gray
    threshold = filters.threshold_otsu(work)
    binary = clean_binary(work > threshold, min_area, hole_area, close_radius)
    if not binary.any():
        return np.zeros_like(gray, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary)
    coords = feature.peak_local_max(
        distance,
        labels=binary,
        min_distance=min_distance,
        exclude_border=False,
    )
    markers = np.zeros_like(gray, dtype=np.int32)
    for marker_id, (row, col) in enumerate(coords, start=1):
        markers[row, col] = marker_id
    if markers.max() == 0:
        markers = measure.label(binary, connectivity=2).astype(np.int32)

    labels = segmentation.watershed(-distance, markers, mask=binary)
    labels = morphology.remove_small_objects(labels, min_size=min_area)
    return relabel(labels)


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | int]], method: str, sample: str) -> dict[str, float | int | str]:
    diameters = np.array([float(row["equivalent_diameter_px"]) for row in rows], dtype=float)
    areas = np.array([float(row["area_px"]) for row in rows], dtype=float)
    return {
        "sample": sample,
        "method": method,
        "particle_count": int(len(rows)),
        "diameter_unit": "px",
        "area_unit": "px",
        "diameter_min": round(float(diameters.min()), 6) if len(diameters) else 0,
        "diameter_median": round(float(np.median(diameters)), 6) if len(diameters) else 0,
        "diameter_mean": round(float(diameters.mean()), 6) if len(diameters) else 0,
        "diameter_max": round(float(diameters.max()), 6) if len(diameters) else 0,
        "area_min": round(float(areas.min()), 6) if len(areas) else 0,
        "area_median": round(float(np.median(areas)), 6) if len(areas) else 0,
        "area_mean": round(float(areas.mean()), 6) if len(areas) else 0,
        "area_max": round(float(areas.max()), 6) if len(areas) else 0,
    }


def save_outlines(path: Path, gray: np.ndarray, labels: np.ndarray) -> None:
    outlines = segmentation.mark_boundaries(
        np.repeat(gray[..., None], 3, axis=2),
        labels,
        color=(1, 0, 0),
        mode="thick",
    )
    Image.fromarray((outlines * 255).astype(np.uint8)).save(path)


def save_method_outputs(sample_dir: Path, sample: str, method: str, gray: np.ndarray, labels: np.ndarray) -> dict:
    tifffile.imwrite(sample_dir / f"{sample}_{method}_mask.tif", labels.astype(np.int32))
    Image.fromarray(overlay_labels(gray, labels)).save(sample_dir / f"{sample}_{method}_overlay.png")
    save_outlines(sample_dir / f"{sample}_{method}_outlines.png", gray, labels)

    rows = particle_rows(labels, None)
    write_csv(sample_dir / f"{sample}_{method}_particles.csv", rows)
    summary = summarize(rows, method, sample)
    (sample_dir / f"{sample}_{method}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def process_image(path: Path, args: argparse.Namespace) -> list[dict]:
    sample = path.stem
    sample_dir = OUT_DIR / sample
    sample_dir.mkdir(parents=True, exist_ok=True)

    gray = read_sem_image(path)
    crop_bottom = find_sem_crop(gray)
    gray = gray[:crop_bottom]
    Image.fromarray(gray).save(sample_dir / f"{sample}_prepared.png")

    threshold = imagej_otsu_labels(
        gray,
        min_area=args.min_area,
        hole_area=args.hole_area,
        close_radius=args.close_radius,
        blur_sigma=args.blur_sigma,
    )
    watershed = watershed_labels(
        gray,
        min_area=args.min_area,
        hole_area=args.hole_area,
        close_radius=args.close_radius,
        blur_sigma=args.blur_sigma,
        min_distance=args.min_distance,
    )

    return [
        save_method_outputs(sample_dir, sample, "imagej_otsu", gray, threshold),
        save_method_outputs(sample_dir, sample, "watershed", gray, watershed),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--hole-area", type=int, default=400)
    parser.add_argument("--close-radius", type=int, default=2)
    parser.add_argument("--blur-sigma", type=float, default=1.0)
    parser.add_argument("--min-distance", type=int, default=18)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    summaries = []
    for image in args.images:
        summaries.extend(process_image(image, args))

    summary_path = OUT_DIR / "classical_segmentation_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in summaries:
        print(
            f"{row['sample']} {row['method']}: "
            f"count={row['particle_count']}, "
            f"diameter_median={row['diameter_median']}, "
            f"area_median={row['area_median']}"
        )
    print(f"saved summary: {summary_path}")


if __name__ == "__main__":
    main()
