import argparse
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from skimage import measure, morphology


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "具体法"
OUT_DIR = ROOT / "cellpose_dataset"


def find_sem_crop(gray: np.ndarray) -> int:
    """Return bottom row index before the black microscope info bar."""
    row_mean = gray.mean(axis=1)
    dark_rows = np.where(row_mean < 20)[0]
    if len(dark_rows) == 0:
        return gray.shape[0]

    # The SEM image has one large black information bar at the bottom.
    bottom_dark = dark_rows[dark_rows > gray.shape[0] * 0.75]
    if len(bottom_dark) == 0:
        return gray.shape[0]
    return int(bottom_dark[0])


def colored_annotation_mask(annotation_rgb: np.ndarray) -> np.ndarray:
    rgb = annotation_rgb.astype(np.int16)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    value = rgb.max(axis=2)

    # Manual overlays are saturated pastel colors; the raw SEM background is gray.
    mask = (saturation > 28) & (value > 55)
    mask = morphology.remove_small_objects(mask, min_size=80)
    mask = morphology.remove_small_holes(mask, area_threshold=80)
    return mask


def save_overlay(gray: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    rng = np.random.default_rng(7)
    overlay = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if not region.any():
            continue
        color = rng.integers(80, 255, size=3)
        overlay[region] = overlay[region] * 0.45 + color * 0.55
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="1_i311")
    parser.add_argument("--raw", default=None)
    parser.add_argument("--annotation", default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    raw_path = Path(args.raw) if args.raw else SRC_DIR / f"{args.sample}_raw_preview.png"
    annotation_path = Path(args.annotation) if args.annotation else SRC_DIR / f"{args.sample}_人工目视标注.png"

    raw = np.array(Image.open(raw_path).convert("L"))
    crop_bottom = find_sem_crop(raw)

    raw_crop = raw[:crop_bottom]

    image_path = OUT_DIR / f"{args.sample}_sem.png"
    mask_path = OUT_DIR / f"{args.sample}_sem_masks.tif"
    overlay_path = OUT_DIR / f"{args.sample}_sem_mask_preview.png"

    Image.fromarray(raw_crop).save(image_path)
    print(f"saved image: {image_path}")
    print(f"crop: {raw.shape[1]}x{crop_bottom}")

    if annotation_path.exists():
        annotation = np.array(Image.open(annotation_path).convert("RGB"))
        annotation_crop = annotation[:crop_bottom]
        binary = colored_annotation_mask(annotation_crop)

        labels = measure.label(binary, connectivity=2).astype(np.uint16)
        tifffile.imwrite(mask_path, labels)
        save_overlay(raw_crop, labels, overlay_path)

        areas = np.bincount(labels.ravel())[1:]
        print(f"saved mask:  {mask_path}")
        print(f"saved preview: {overlay_path}")
        print(f"instances: {labels.max()}")
        print(f"area px: min={areas.min() if len(areas) else 0}, median={np.median(areas) if len(areas) else 0:.1f}, max={areas.max() if len(areas) else 0}")


if __name__ == "__main__":
    main()
