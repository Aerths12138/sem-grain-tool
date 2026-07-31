import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sem_grain_app import find_sem_crop, read_sem_image


ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
MODEL_CACHE = ROOT / "cellpose_cache" / "models"
CELLPOSE_CACHE = ROOT / "cellpose_cache"
DATASET_DIR = ROOT / "cellpose_dataset"
RESULT_DIR = ROOT / "cellpose_results_pretrained"
ANALYSIS_DIR = ROOT / "particle_analysis"
TEST_DIR = ROOT / "od_metaldam_cellpose_test"


def require_fiftyone():
    try:
        import fiftyone as fo  # noqa: F401
        from fiftyone.utils.huggingface import load_from_hub
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: fiftyone\n"
            "Install it in the required environment first:\n"
            r"E:\conda_envs\sem_sam_env\python.exe -m pip install fiftyone"
        ) from exc
    return load_from_hub


def ensure_dirs() -> None:
    for path in (MODEL_CACHE, CELLPOSE_CACHE, DATASET_DIR, RESULT_DIR, ANALYSIS_DIR, TEST_DIR):
        path.mkdir(exist_ok=True)
    (TEST_DIR / "originals").mkdir(exist_ok=True)


def prepare_image_for_cellpose(source_path: Path, sample_name: str) -> Path:
    gray = read_sem_image(source_path)
    crop_bottom = find_sem_crop(gray)
    prepared = gray[:crop_bottom]
    out_path = DATASET_DIR / f"{sample_name}_sem.png"
    Image.fromarray(prepared).save(out_path)
    return out_path


def run_command(command: list[str]) -> None:
    env = os.environ.copy()
    env["USERPROFILE"] = str(CELLPOSE_CACHE)
    env["CELLPOSE_LOCAL_MODELS_PATH"] = str(MODEL_CACHE)
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def process_sample(source_path: Path, sample_name: str, diameter: float) -> dict[str, str]:
    copied_source = TEST_DIR / "originals" / f"{sample_name}{source_path.suffix.lower() or '.png'}"
    if not copied_source.exists():
        shutil.copy2(source_path, copied_source)

    prepared = prepare_image_for_cellpose(source_path, sample_name)
    image_rel = str(prepared.relative_to(ROOT))

    run_command([
        str(PYTHON),
        "-m",
        "cellpose",
        "--image_path",
        image_rel,
        "--pretrained_model",
        "cpsam",
        "--diameter",
        str(diameter),
        "--save_tif",
        "--savedir",
        str(RESULT_DIR.relative_to(ROOT)),
        "--verbose",
    ])
    run_command([str(PYTHON), str(ROOT / "make_cellpose_overlay.py"), "--sample", sample_name, "--skip-gt"])
    run_command([str(PYTHON), str(ROOT / "analyze_particles.py"), "--sample", sample_name, "--bins", "10"])

    return {
        "sample": sample_name,
        "source": str(copied_source),
        "prepared": str(prepared),
        "mask": str(RESULT_DIR / f"{sample_name}_sem_cp_masks.tif"),
        "overlay": str(RESULT_DIR / f"{sample_name}_sem_cpsam_overlay.png"),
        "outlines": str(RESULT_DIR / f"{sample_name}_sem_cpsam_outlines.png"),
        "particles_csv": str(ANALYSIS_DIR / f"{sample_name}_particles.csv"),
        "summary_json": str(ANALYSIS_DIR / f"{sample_name}_summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Voxel51/OD_MetalDAM")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--diameter", type=float, default=90)
    parser.add_argument("--prefix", default="od_metaldam")
    args = parser.parse_args()

    ensure_dirs()
    load_from_hub = require_fiftyone()

    dataset = load_from_hub(args.dataset, max_samples=args.max_samples)
    samples = list(dataset.limit(args.max_samples))
    if len(samples) < args.max_samples:
        raise RuntimeError(f"Expected {args.max_samples} samples, got {len(samples)}")

    outputs = []
    for index, sample in enumerate(samples, start=1):
        source_path = Path(sample.filepath)
        sample_name = f"{args.prefix}_{index:02d}"
        print(f"Processing {sample_name}: {source_path}", flush=True)
        outputs.append(process_sample(source_path, sample_name, args.diameter))

    manifest = TEST_DIR / "manifest.json"
    manifest.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest}")


if __name__ == "__main__":
    main()
