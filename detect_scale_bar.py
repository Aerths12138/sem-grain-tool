import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image


ROOT = Path(__file__).resolve().parent


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def read_gray(path: Path) -> np.ndarray:
    try:
        return normalize_to_uint8(tifffile.imread(path))
    except Exception:
        return np.array(Image.open(path).convert("L"))


def find_footer_y(gray: np.ndarray) -> int:
    h = gray.shape[0]
    row_mean = gray.mean(axis=1)
    rows = np.where((row_mean < 30) & (np.arange(h) > h * 0.65))[0]
    if len(rows) == 0:
        return int(h * 0.75)
    return int(rows[0])


def detect_tick_scale(
    gray: np.ndarray,
    threshold: int,
    min_x_fraction: float,
) -> dict[str, float | int | str] | None:
    h, w = gray.shape
    footer_y = find_footer_y(gray)
    roi = gray[footer_y:]
    binary = (roi >= threshold).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    ticks: list[tuple[float, int, int, int, int]] = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if x < w * min_x_fraction:
            continue
        if y > 18:
            continue
        if not (2 <= bw <= 10 and bh >= 10):
            continue
        if area < bw * bh * 0.6:
            continue
        ticks.append((x + bw / 2.0, x, footer_y + y, bw, bh))

    if len(ticks) < 2:
        return None

    ticks.sort()
    x0 = float(ticks[0][0])
    x1 = float(ticks[-1][0])
    if x1 <= x0:
        return None
    return {
        "method": "vertical_ticks",
        "footer_y_px": footer_y,
        "bar_x0_px": round(x0, 3),
        "bar_x1_px": round(x1, 3),
        "bar_y_px": footer_y,
        "bar_pixel_length": round(x1 - x0, 3),
        "tick_count": len(ticks),
    }


def detect_horizontal_scale(
    gray: np.ndarray,
    threshold: int,
    min_x_fraction: float,
) -> dict[str, float | int | str] | None:
    h, w = gray.shape
    footer_y = find_footer_y(gray)
    roi = gray[footer_y:]
    binary = (roi >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, 8)

    candidates: list[tuple[int, int, int, int, int]] = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if x < w * min_x_fraction:
            continue
        if not (bw >= 60 and bh <= 12):
            continue
        candidates.append((bw, x, footer_y + y, bh, area))

    if not candidates:
        return None

    bw, x, y, bh, _ = max(candidates)
    return {
        "method": "horizontal_bar",
        "footer_y_px": footer_y,
        "bar_x0_px": float(x),
        "bar_x1_px": float(x + bw - 1),
        "bar_y_px": y + bh / 2.0,
        "bar_pixel_length": float(bw - 1),
        "tick_count": 0,
    }


def detect_scale_bar(
    image_path: Path,
    scale_um: float,
    threshold: int,
    min_x_fraction: float,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    gray = read_gray(image_path)
    result = detect_tick_scale(gray, threshold, min_x_fraction)
    if result is None:
        result = detect_horizontal_scale(gray, threshold, min_x_fraction)
    if result is None:
        raise RuntimeError("Could not detect a scale bar in the lower SEM footer.")

    bar_px = float(result["bar_pixel_length"])
    if bar_px <= 0:
        raise RuntimeError("Detected scale bar has invalid pixel length.")

    result["image_path"] = str(image_path)
    result["scale_um"] = float(scale_um)
    result["um_per_px"] = float(scale_um) / bar_px
    result["image_width_px"] = int(gray.shape[1])
    result["image_height_px"] = int(gray.shape[0])
    return result, gray


def save_preview(path: Path, gray: np.ndarray, result: dict[str, float | int | str]) -> None:
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x0 = int(round(float(result["bar_x0_px"])))
    x1 = int(round(float(result["bar_x1_px"])))
    y = int(round(float(result["bar_y_px"])))
    cv2.line(preview, (x0, y), (x1, y), (0, 0, 255), 2)
    cv2.circle(preview, (x0, y), 5, (0, 255, 255), -1)
    cv2.circle(preview, (x1, y), 5, (0, 255, 255), -1)
    label = f'{float(result["scale_um"]):g} um = {float(result["bar_pixel_length"]):.1f} px'
    cv2.putText(preview, label, (max(0, x0 - 220), max(20, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    ok, encoded = cv2.imencode(".png", preview)
    if not ok:
        raise RuntimeError(f"Could not encode preview: {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--sample", default=None)
    parser.add_argument("--scale-um", type=float, default=100.0)
    parser.add_argument("--threshold", type=int, default=170)
    parser.add_argument("--min-x-fraction", type=float, default=0.45)
    parser.add_argument("--output-dir", default=str(ROOT / "particle_analysis"))
    args = parser.parse_args()

    image_path = Path(args.image)
    sample = args.sample or image_path.stem
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    result, gray = detect_scale_bar(
        image_path=image_path,
        scale_um=float(args.scale_um),
        threshold=int(args.threshold),
        min_x_fraction=float(args.min_x_fraction),
    )
    json_path = out_dir / f"{sample}_scale_bar.json"
    preview_path = out_dir / f"{sample}_scale_bar_detection.png"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_preview(preview_path, gray, result)

    print(f"scale_um={result['scale_um']}")
    print(f"bar_pixel_length={result['bar_pixel_length']}")
    print(f"um_per_px={result['um_per_px']:.10g}")
    print(f"saved scale json: {json_path}")
    print(f"saved scale preview: {preview_path}")


if __name__ == "__main__":
    main()
