import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from skimage import segmentation

from make_cellpose_overlay import overlay_labels, stats


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
MODEL_CACHE = ROOT / "cellpose_cache" / "models"
CELLPOSE_CACHE = ROOT / "cellpose_cache"
DATASET_DIR = ROOT / "cellpose_dataset"


def run_command(command: list[str], output_dir: Path) -> None:
    env = os.environ.copy()
    env["USERPROFILE"] = str(CELLPOSE_CACHE)
    env["CELLPOSE_LOCAL_MODELS_PATH"] = str(MODEL_CACHE)
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def save_overlay_outputs(image_path: Path, mask_path: Path, output_prefix: Path) -> dict[str, float | int | str]:
    gray = np.array(Image.open(image_path).convert("L"))
    labels = tifffile.imread(mask_path).astype(np.int32)

    overlay = overlay_labels(gray, labels)
    overlay_path = output_prefix.with_name(output_prefix.name + "_overlay.png")
    Image.fromarray(overlay).save(overlay_path)

    outlines = segmentation.mark_boundaries(
        np.repeat(gray[..., None], 3, axis=2),
        labels,
        color=(1, 0, 0),
        mode="thick",
    )
    outlines_path = output_prefix.with_name(output_prefix.name + "_outlines.png")
    Image.fromarray((outlines * 255).astype(np.uint8)).save(outlines_path)

    count, area_median, area_min, area_max = stats(labels)
    return {
        "instances": count,
        "area_median_px": area_median,
        "area_min_px": area_min,
        "area_max_px": area_max,
        "mask": str(mask_path),
        "overlay": str(overlay_path),
        "outlines": str(outlines_path),
    }


def run_one(sample: str, diameter: float, sweep_dir: Path) -> dict[str, float | int | str]:
    image_path = DATASET_DIR / f"{sample}_sem.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing prepared image: {image_path}")

    out_dir = sweep_dir / f"{sample}_diameter_{diameter:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_command([
        str(PYTHON),
        "-m",
        "cellpose",
        "--image_path",
        str(image_path.relative_to(ROOT)),
        "--pretrained_model",
        "cpsam",
        "--diameter",
        str(diameter),
        "--save_tif",
        "--savedir",
        str(out_dir.relative_to(ROOT)),
        "--verbose",
    ], out_dir)

    generated_mask = out_dir / f"{sample}_sem_cp_masks.tif"
    if not generated_mask.exists():
        raise FileNotFoundError(f"Cellpose did not write mask: {generated_mask}")

    prefix = out_dir / f"{sample}_diameter_{diameter:g}"
    result = save_overlay_outputs(image_path, generated_mask, prefix)
    result["sample"] = sample
    result["diameter"] = diameter
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="od_metaldam_04")
    parser.add_argument("--diameters", nargs="+", type=float, default=[20, 30, 45, 60, 90])
    parser.add_argument("--output-dir", default=str(ROOT / "diameter_sweep_results"))
    parser.add_argument("--clean-sample", action="store_true")
    args = parser.parse_args()

    sweep_dir = Path(args.output_dir)
    if args.clean_sample:
        for diameter in args.diameters:
            target = sweep_dir / f"{args.sample}_diameter_{diameter:g}"
            if target.exists():
                shutil.rmtree(target)
    sweep_dir.mkdir(exist_ok=True)

    results = []
    for diameter in args.diameters:
        print(f"Running {args.sample} with diameter={diameter:g}", flush=True)
        results.append(run_one(args.sample, diameter, sweep_dir))

    summary_path = sweep_dir / f"{args.sample}_diameter_sweep_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
