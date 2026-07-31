from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation


ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "cellpose_dataset" / "1_i313_sem.png"
OUT_DIR = ROOT / "cellpose_results_pretrained"
MASK_PATH = OUT_DIR / "1_i313_sem_direct_masks.tif"
OVERLAY_PATH = OUT_DIR / "1_i313_sem_direct_overlay.png"
OUTLINES_PATH = OUT_DIR / "1_i313_sem_direct_outlines.png"


def label_to_overlay(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(313)
    overlay = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if not region.any():
            continue
        color = rng.integers(70, 255, size=3)
        overlay[region] = overlay[region] * 0.42 + color * 0.58
    return np.clip(overlay, 0, 255).astype(np.uint8)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    gray = np.array(Image.open(IMAGE_PATH).convert("L"))

    denoised = cv2.fastNlMeansDenoising(gray, h=5)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    local_bg = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=45, sigmaY=45)
    dark_regions = enhanced < (local_bg + 3)

    dark_regions = morphology.remove_small_objects(dark_regions, min_size=500)
    dark_regions = morphology.binary_closing(dark_regions, morphology.disk(5))
    dark_regions = morphology.binary_opening(dark_regions, morphology.disk(3))
    dark_regions = ndi.binary_fill_holes(dark_regions)
    dark_regions = morphology.remove_small_holes(dark_regions, area_threshold=1800)

    labels = measure.label(dark_regions, connectivity=2)
    cleaned = np.zeros_like(labels, dtype=np.uint16)
    next_id = 1
    for region in measure.regionprops(labels):
        area = region.area
        if area < 900 or area > 140000:
            continue
        minr, minc, maxr, maxc = region.bbox
        if (maxr - minr) < 22 or (maxc - minc) < 22:
            continue
        cleaned[labels == region.label] = next_id
        next_id += 1

    tifffile.imwrite(MASK_PATH, cleaned)
    Image.fromarray(label_to_overlay(gray, cleaned)).save(OVERLAY_PATH)

    outlines = segmentation.mark_boundaries(
        np.repeat(gray[..., None], 3, axis=2),
        cleaned,
        color=(1, 0, 0),
        mode="thick",
    )
    Image.fromarray((outlines * 255).astype(np.uint8)).save(OUTLINES_PATH)

    areas = np.bincount(cleaned.ravel())[1:]
    print(f"instances={int(cleaned.max())}")
    print(f"area_median={np.median(areas) if len(areas) else 0:.1f}")
    print(f"saved mask: {MASK_PATH}")
    print(f"saved overlay: {OVERLAY_PATH}")
    print(f"saved outlines: {OUTLINES_PATH}")


if __name__ == "__main__":
    main()
