import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage import measure, segmentation


ROOT = Path(__file__).resolve().parent

def overlay_labels(gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(11)
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if not region.any():
            continue
        color = rng.integers(70, 255, size=3)
        rgb[region] = rgb[region] * 0.45 + color * 0.55
    return np.clip(rgb, 0, 255).astype(np.uint8)


def stats(labels: np.ndarray) -> tuple[int, float, int, int]:
    areas = np.bincount(labels.ravel())[1:]
    if len(areas) == 0:
        return 0, 0.0, 0, 0
    return int(labels.max()), float(np.median(areas)), int(areas.min()), int(areas.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="1_i311")
    parser.add_argument("--skip-gt", action="store_true")
    args = parser.parse_args()

    image_path = ROOT / "cellpose_dataset" / f"{args.sample}_sem.png"
    pred_mask_path = ROOT / "cellpose_results_pretrained" / f"{args.sample}_sem_cp_masks.tif"
    gt_mask_path = ROOT / "cellpose_dataset" / f"{args.sample}_sem_masks.tif"
    out_path = ROOT / "cellpose_results_pretrained" / f"{args.sample}_sem_cpsam_overlay.png"
    out_outlines_path = ROOT / "cellpose_results_pretrained" / f"{args.sample}_sem_cpsam_outlines.png"

    gray = np.array(Image.open(image_path).convert("L"))
    pred = tifffile.imread(pred_mask_path).astype(np.int32)

    overlay = overlay_labels(gray, pred)
    Image.fromarray(overlay).save(out_path)

    outlines = segmentation.mark_boundaries(
        np.repeat(gray[..., None], 3, axis=2),
        pred,
        color=(1, 0, 0),
        mode="thick",
    )
    Image.fromarray((outlines * 255).astype(np.uint8)).save(out_outlines_path)

    pred_stats = stats(pred)
    print(f"pred instances={pred_stats[0]}, area_median={pred_stats[1]:.1f}, area_min={pred_stats[2]}, area_max={pred_stats[3]}")
    if not args.skip_gt and gt_mask_path.exists():
        gt = tifffile.imread(gt_mask_path).astype(np.int32)
        gt_stats = stats(gt)
        print(f"manual instances={gt_stats[0]}, area_median={gt_stats[1]:.1f}, area_min={gt_stats[2]}, area_max={gt_stats[3]}")
    print(f"saved overlay: {out_path}")
    print(f"saved outlines: {out_outlines_path}")


if __name__ == "__main__":
    main()
