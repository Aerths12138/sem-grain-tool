"""Local browser UI for the packaged SEM Grain Tool."""

from __future__ import annotations

import json
import math
import mimetypes
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

# Importing this module also preserves the packaged ``--helper`` entry point.
from sem_grain_app_frozen import (
    PROCESS_TIMEOUTS_SECONDS,
    ProcessController,
    TaskCancelled,
    TaskTimedOut,
    add_polygon_mask,
    delete_mask_labels,
    dxf_scale_mm_per_pixel,
    manual_scale_values,
    merge_mask_labels,
    read_sem_image,
    safe_sample_name,
    save_manual_scale_calibration,
    save_png_cv2,
    split_mask_label,
)

import cv2
import numpy as np
import tifffile
from PIL import Image


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


APP_ROOT = application_root()
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", APP_ROOT))
STATIC_ROOT = RESOURCE_ROOT / "web_static"
MODEL_CACHE = RESOURCE_ROOT / "cellpose_cache" / "models"
CELLPOSE_CACHE = APP_ROOT / "cellpose_cache"
UPLOAD_DIR = APP_ROOT / "input_uploads"
DATASET_DIR = APP_ROOT / "cellpose_dataset"
RESULT_DIR = APP_ROOT / "cellpose_results_pretrained"
ANALYSIS_DIR = APP_ROOT / "particle_analysis"
SHAPE_DIR = APP_ROOT / "coordinate_shape_outputs"

MAX_UPLOAD_BYTES = 512 * 1024 * 1024
ALLOWED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
FILE_ROOTS = {
    "dataset": DATASET_DIR,
    "results": RESULT_DIR,
    "analysis": ANALYSIS_DIR,
    "shapes": SHAPE_DIR,
}


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label}必须是正数")
    return number


def _finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label}不能小于 0")
    return number


def _helper_command(command: str, *arguments: str) -> list[str]:
    if getattr(sys, "frozen", False):
        return [str(sys.executable), "--helper", command, *arguments]
    if command == "cellpose":
        return [str(sys.executable), "-m", "cellpose", *arguments]
    return [str(sys.executable), str(APP_ROOT / command), *arguments]


def _file_url(path: Path, token: str) -> str:
    resolved = path.resolve()
    for key, root in FILE_ROOTS.items():
        try:
            relative = resolved.relative_to(root.resolve())
        except ValueError:
            continue
        quoted = "/".join(urllib.parse.quote(part) for part in relative.parts)
        version = resolved.stat().st_mtime_ns if resolved.exists() else int(time.time_ns())
        return f"/files/{key}/{quoted}?token={urllib.parse.quote(token)}&v={version}"
    raise ValueError(f"File is outside a published output directory: {path}")


def _prepare_uploaded_image(path: Path, sample: str) -> tuple[Path, Path]:
    """Save a full-size browser preview and the cropped Cellpose input."""
    gray = read_sem_image(path)
    source_path = DATASET_DIR / f"{sample}_source.png"
    prepared_path = DATASET_DIR / f"{sample}_sem.png"
    save_png_cv2(source_path, gray)

    row_mean = gray.mean(axis=1)
    dark_rows = np.where(row_mean < 20)[0]
    crop_bottom = gray.shape[0]
    if len(dark_rows):
        bottom_dark = dark_rows[dark_rows > gray.shape[0] * 0.75]
        if len(bottom_dark):
            crop_bottom = int(bottom_dark[0])
    save_png_cv2(prepared_path, gray[:crop_bottom])
    return source_path, prepared_path


def _render_editor_preview(
    image_path: Path,
    labels: np.ndarray,
    selected: set[int],
    output_path: Path,
) -> None:
    base = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8).copy()
    max_label = int(labels.max())
    if max_label > 0:
        ids = np.arange(max_label + 1, dtype=np.int64)
        palette = np.column_stack((
            70 + (ids * 47) % 170,
            70 + (ids * 83) % 170,
            70 + (ids * 131) % 170,
        )).astype(np.uint8)
        foreground = labels > 0
        colors = palette[labels]
        base[foreground] = (
            base[foreground].astype(np.float32) * 0.55
            + colors[foreground].astype(np.float32) * 0.45
        ).astype(np.uint8)

        edges = np.zeros(labels.shape, dtype=bool)
        edges[1:, :] |= labels[1:, :] != labels[:-1, :]
        edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
        edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        base[edges & foreground] = (255, 255, 255)

    if selected:
        selected_pixels = np.isin(labels, list(selected))
        base[selected_pixels] = (
            base[selected_pixels].astype(np.float32) * 0.35
            + np.array([255, 224, 60], dtype=np.float32) * 0.65
        ).astype(np.uint8)
    save_png_cv2(output_path, cv2.cvtColor(base, cv2.COLOR_RGB2BGR))


class WebApplication:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(24)
        self.logs: deque[str] = deque(maxlen=800)
        self.controller = ProcessController(self.append_log)
        self.server: ThreadingHTTPServer | None = None
        self.shutdown_started = False

        self.input_path: Path | None = None
        self.sample: str | None = None
        self.um_per_px: float | None = None
        self.preview_path: Path | None = None
        self.status = "请选择一张电子显微镜图像"
        self.busy = False
        self.phase: str | None = None
        self.error: str | None = None
        # Probe Torch in an isolated helper so missing drivers or CUDA DLL failures
        # cannot delay or crash the local web server.
        self.gpu_available = False
        self.device_info: dict[str, Any] = {
            "status": "checking",
            "backend": "cpu",
            "gpu_available": False,
            "reason": "正在检测GPU和驱动",
        }

        self.editor_labels: np.ndarray | None = None
        self.editor_selected: set[int] = set()
        self.editor_undo: list[np.ndarray] = []
        self.editor_redo: list[np.ndarray] = []
        self.editor_dirty = False

        for path in (
            CELLPOSE_CACHE,
            UPLOAD_DIR,
            DATASET_DIR,
            RESULT_DIR,
            ANALYSIS_DIR,
            SHAPE_DIR,
        ):
            path.mkdir(parents=True, exist_ok=True)
        threading.Thread(
            target=self._probe_device,
            name="sem-device-probe",
            daemon=True,
        ).start()

    def append_log(self, text: str) -> None:
        with self.lock:
            self.logs.append(str(text))

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            preview_url = None
            if self.preview_path and self.preview_path.exists():
                preview_url = _file_url(self.preview_path, self.token)
            return {
                "status": self.status,
                "busy": self.busy,
                "phase": self.phase,
                "error": self.error,
                "sample": self.sample,
                "input_name": self.input_path.name if self.input_path else None,
                "um_per_px": self.um_per_px,
                "preview_url": preview_url,
                "source_url": (
                    _file_url(DATASET_DIR / f"{self.sample}_source.png", self.token)
                    if self.sample and (DATASET_DIR / f"{self.sample}_source.png").exists()
                    else None
                ),
                "gpu_available": self.gpu_available,
                "device": dict(self.device_info),
                "active_pid": self.controller.active_pid,
                "logs": list(self.logs),
                "mask_available": self._mask_path().exists() if self.sample else False,
                "editor": {
                    "active": self.editor_labels is not None,
                    "selected": sorted(self.editor_selected),
                    "dirty": self.editor_dirty,
                    "can_undo": bool(self.editor_undo),
                    "can_redo": bool(self.editor_redo),
                },
                "outputs": self._outputs(),
            }

    def _probe_device(self) -> None:
        try:
            completed = subprocess.run(
                _helper_command("device_probe.py", "--json"),
                cwd=APP_ROOT,
                env=self._process_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if os.name == "nt" else 0,
                check=False,
            )
            marker = "DEVICE_PROBE_JSON="
            payload_line = next(
                (line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
                None,
            )
            if completed.returncode != 0 or payload_line is None:
                detail = completed.stdout.strip() or f"exit code {completed.returncode}"
                raise RuntimeError(detail)
            result = json.loads(payload_line[len(marker):])
            if not isinstance(result, dict):
                raise RuntimeError("设备探测返回格式不正确")
            result["status"] = "ready"
        except Exception as exc:
            result = {
                "status": "error",
                "backend": "cpu",
                "gpu_available": False,
                "reason": f"设备探测失败，已使用CPU：{exc}",
            }
        with self.lock:
            self.device_info = result
            self.gpu_available = bool(result.get("gpu_available"))
        if self.gpu_available:
            memory = result.get("total_memory_mb")
            memory_text = f"，显存 {memory} MB" if memory is not None else ""
            self.append_log(f"GPU检测通过：{result.get('device_name')}{memory_text}")
            if result.get("reason"):
                self.append_log(str(result["reason"]))
        else:
            self.append_log(f"GPU不可用，使用CPU：{result.get('reason')}")

    def _outputs(self) -> list[dict[str, str]]:
        if not self.sample:
            return []
        outputs: list[dict[str, str]] = []
        for root_key, root in FILE_ROOTS.items():
            for path in sorted(root.glob(f"{self.sample}*")):
                if path.is_file():
                    outputs.append({
                        "name": path.name,
                        "group": root_key,
                        "url": _file_url(path, self.token),
                    })
        return outputs

    def _mask_path(self) -> Path:
        return RESULT_DIR / f"{self.sample}_sem_cp_masks.tif"

    def upload(self, filename: str, data: bytes) -> None:
        if self.busy:
            raise RuntimeError("当前任务尚未完成")
        clean_name = Path(filename).name
        suffix = Path(clean_name).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError("请选择 TIF、TIFF、PNG、JPG、JPEG 或 BMP 图像")
        if not data:
            raise ValueError("上传的图像为空")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("图像超过 512 MB 限制")

        sample = safe_sample_name(Path(clean_name))
        input_path = UPLOAD_DIR / f"{sample}{suffix}"
        input_path.write_bytes(data)
        try:
            source_preview, prepared = _prepare_uploaded_image(input_path, sample)
        except Exception:
            input_path.unlink(missing_ok=True)
            raise

        with self.lock:
            self.input_path = input_path
            self.sample = sample
            self.um_per_px = None
            self.preview_path = source_preview
            self.status = f"已选择：{clean_name}"
            self.error = None
            self.logs.clear()
            self.logs.append(f"sample = {sample}")
            self.logs.append(f"prepared image: {prepared}")
            self._clear_editor()

    def start_task(
        self,
        phase: str,
        status: str,
        target: Callable[[], tuple[str, Path | None]],
    ) -> None:
        with self.lock:
            if self.busy:
                raise RuntimeError("已有任务正在运行")
            self.busy = True
            self.phase = phase
            self.status = status
            self.error = None
            self.controller.reset_cancel()

        def worker() -> None:
            try:
                final_status, preview = target()
            except TaskCancelled as exc:
                final_status = f"任务已取消：{exc}"
                preview = None
            except TaskTimedOut as exc:
                final_status = "任务超时"
                preview = None
                with self.lock:
                    self.error = str(exc)
                self.append_log(str(exc))
            except Exception as exc:
                final_status = "任务失败"
                preview = None
                with self.lock:
                    self.error = str(exc)
                self.append_log(f"ERROR: {exc}")
            with self.lock:
                if preview is not None and preview.exists():
                    self.preview_path = preview
                self.status = final_status
                self.busy = False
                self.phase = None

        threading.Thread(target=worker, name=f"sem-{phase}", daemon=True).start()

    def cancel(self) -> None:
        with self.lock:
            if not self.busy:
                raise RuntimeError("当前没有可取消的任务")
        self.controller.request_cancel("用户取消")

    def _process_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["USERPROFILE"] = str(CELLPOSE_CACHE)
        environment["CELLPOSE_LOCAL_MODELS_PATH"] = str(MODEL_CACHE)
        return environment

    def _run_helper(self, command: str, arguments: list[str], phase: str) -> None:
        self.controller.run(
            _helper_command(command, *arguments),
            cwd=APP_ROOT,
            env=self._process_environment(),
            phase=phase,
            timeout_seconds=PROCESS_TIMEOUTS_SECONDS[phase],
        )

    def _require_image(self) -> tuple[Path, str]:
        if self.input_path is None or self.sample is None:
            raise RuntimeError("请先选择 SEM 图像")
        return self.input_path, self.sample

    def start_auto_scale(self, scale_um: Any) -> None:
        image_path, sample = self._require_image()
        known_length = _finite_positive(scale_um, "比例尺实际长度")

        def task() -> tuple[str, Path]:
            self._run_helper("detect_scale_bar.py", [
                "--image", str(image_path),
                "--sample", sample,
                "--scale-um", f"{known_length:g}",
                "--output-dir", str(ANALYSIS_DIR),
            ], "scale-bar")
            result = json.loads(
                (ANALYSIS_DIR / f"{sample}_scale_bar.json").read_text(encoding="utf-8")
            )
            with self.lock:
                self.um_per_px = float(result["um_per_px"])
            self.append_log(
                f"比例尺：{float(result['scale_um']):g} um / "
                f"{float(result['bar_pixel_length']):.1f} px = "
                f"{float(result['um_per_px']):.6g} um/px"
            )
            return "比例尺自动检测完成", ANALYSIS_DIR / f"{sample}_scale_bar_detection.png"

        self.start_task("scale-bar", "正在自动检测比例尺", task)

    def start_manual_scale(self, scale_um: Any, points: Any) -> None:
        image_path, sample = self._require_image()
        known_length = _finite_positive(scale_um, "比例尺实际长度")
        parsed = self._parse_points(points, minimum=2, exact=2)
        manual_scale_values(known_length, parsed[0], parsed[1])

        def task() -> tuple[str, Path]:
            result = save_manual_scale_calibration(
                image_path, ANALYSIS_DIR, sample, known_length, parsed[0], parsed[1]
            )
            with self.lock:
                self.um_per_px = float(result["um_per_px"])
            self.append_log(
                f"人工比例尺：{known_length:g} um / "
                f"{float(result['bar_pixel_length']):.1f} px = "
                f"{float(result['um_per_px']):.6g} um/px"
            )
            return "人工两点标定完成", ANALYSIS_DIR / f"{sample}_scale_bar_detection.png"

        self.start_task("manual-scale", "正在保存人工比例尺", task)

    def _ensure_scale(self, scale_um: float) -> float | None:
        with self.lock:
            current = self.um_per_px
        if current is not None:
            return current
        try:
            self._run_helper("detect_scale_bar.py", [
                "--image", str(self.input_path),
                "--sample", str(self.sample),
                "--scale-um", f"{scale_um:g}",
                "--output-dir", str(ANALYSIS_DIR),
            ], "scale-bar")
            result = json.loads(
                (ANALYSIS_DIR / f"{self.sample}_scale_bar.json").read_text(encoding="utf-8")
            )
            current = float(result["um_per_px"])
            with self.lock:
                self.um_per_px = current
            return current
        except (TaskCancelled, TaskTimedOut):
            raise
        except Exception as exc:
            self.append_log(f"自动比例尺检测已跳过：{exc}")
            return None

    def start_detection(self, use_gpu: Any, scale_um: Any) -> None:
        _, sample = self._require_image()
        known_length = _finite_positive(scale_um, "比例尺实际长度")
        gpu_requested = bool(use_gpu) and self.gpu_available

        def task() -> tuple[str, Path]:
            um_per_px = self._ensure_scale(known_length)
            arguments = [
                "--image_path", f"cellpose_dataset\\{sample}_sem.png",
                "--pretrained_model", "cpsam",
                "--diameter", "90",
                "--save_tif",
                "--savedir", "cellpose_results_pretrained",
                "--verbose",
            ]
            fallback_used = False
            if gpu_requested:
                self.append_log("Cellpose device: GPU 0")
                try:
                    self._run_helper(
                        "cellpose",
                        [*arguments, "--use_gpu", "--gpu_device", "0"],
                        "cellpose",
                    )
                except (TaskCancelled, TaskTimedOut):
                    raise
                except Exception as exc:
                    fallback_used = True
                    self.append_log(f"GPU识别失败：{exc}")
                    self.append_log("正在自动切换到CPU并重新识别")
                    with self.lock:
                        self.gpu_available = False
                        self.device_info = {
                            **self.device_info,
                            "status": "fallback",
                            "backend": "cpu",
                            "gpu_available": False,
                            "reason": f"GPU任务失败，当前会话已回退CPU：{exc}",
                        }
                    self._run_helper("cellpose", arguments, "cellpose")
            else:
                self.append_log("Cellpose device: CPU")
                self._run_helper("cellpose", arguments, "cellpose")
            self._refresh_analysis(sample, um_per_px)
            status = (
                "GPU失败后已使用CPU完成识别，可以人工校正或导出圆形 DXF"
                if fallback_used
                else "晶粒识别完成，可以人工校正或导出圆形 DXF"
            )
            return status, (
                RESULT_DIR / f"{sample}_sem_cpsam_overlay.png"
            )

        self.start_task("cellpose", "正在识别晶粒，CPU 环境可能需要较长时间", task)

    def _refresh_analysis(self, sample: str, um_per_px: float | None) -> None:
        self._run_helper(
            "make_cellpose_overlay.py", ["--sample", sample, "--skip-gt"], "overlay"
        )
        arguments = ["--sample", sample, "--bins", "10"]
        if um_per_px is not None:
            arguments.extend(["--um-per-px", f"{um_per_px:.10g}"])
        self._run_helper("analyze_particles.py", arguments, "particle-analysis")

    def start_export(self, area_scale: Any, circle_gap: Any) -> None:
        _, sample = self._require_image()
        with self.lock:
            um_per_px = self.um_per_px
        if um_per_px is None:
            raise RuntimeError("请先自动检测或人工两点标定比例尺")
        area = _finite_positive(area_scale, "面积比例")
        gap = _finite_nonnegative(circle_gap, "圆间距")
        dxf_scale = dxf_scale_mm_per_pixel(um_per_px)

        def task() -> tuple[str, Path]:
            self._run_helper("generate_shapes_from_particles.py", [
                "--sample", sample,
                "--mode", "circle",
                "--area-scale", f"{area:g}",
                "--circle-gap", f"{gap:g}",
                "--dxf-scale", f"{dxf_scale:.12g}",
                "--resolve-iterations", "1200",
                "--resolve-damping", "0.22",
                "--max-center-shift-ratio", "1.0",
            ], "circle-export")
            self.append_log(
                f"DXF physical scale: 1 px = {um_per_px:.6g} um = {dxf_scale:.9g} mm"
            )
            preview = SHAPE_DIR / (
                f"{sample}_circle_nonoverlap_a{area:g}_gap{gap:g}"
                "_area_from_centers_overlay.png"
            )
            return "圆形图和毫米单位 DXF 已输出", preview

        self.start_task("circle-export", "正在生成圆形图和 DXF", task)

    @staticmethod
    def _parse_points(points: Any, minimum: int, exact: int | None = None) -> list[tuple[int, int]]:
        if not isinstance(points, list):
            raise ValueError("点坐标格式不正确")
        parsed: list[tuple[int, int]] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("点坐标格式不正确")
            parsed.append((int(round(float(point[0]))), int(round(float(point[1])))))
        if len(parsed) < minimum or (exact is not None and len(parsed) != exact):
            raise ValueError(f"需要 {exact if exact is not None else minimum} 个点")
        return parsed

    def _clear_editor(self) -> None:
        self.editor_labels = None
        self.editor_selected.clear()
        self.editor_undo.clear()
        self.editor_redo.clear()
        self.editor_dirty = False

    def start_editor(self) -> None:
        _, sample = self._require_image()
        mask_path = self._mask_path()
        image_path = DATASET_DIR / f"{sample}_sem.png"
        if not mask_path.exists():
            raise RuntimeError("请先完成晶粒识别")
        labels = tifffile.imread(mask_path).astype(np.int32)
        with self.lock:
            self.editor_labels = labels
            self.editor_selected.clear()
            self.editor_undo.clear()
            self.editor_redo.clear()
            self.editor_dirty = False
            preview = RESULT_DIR / f"{sample}_web_editor.png"
            _render_editor_preview(image_path, labels, set(), preview)
            self.preview_path = preview
            self.status = "人工校正：点击晶粒进行选择"

    def editor_select(self, x: Any, y: Any, additive: Any) -> None:
        with self.lock:
            labels = self.editor_labels
            sample = self.sample
            if labels is None or sample is None:
                raise RuntimeError("请先进入人工校正")
            ix, iy = int(round(float(x))), int(round(float(y)))
            if not (0 <= ix < labels.shape[1] and 0 <= iy < labels.shape[0]):
                raise ValueError("点击位置超出图像范围")
            label_id = int(labels[iy, ix])
            if not bool(additive):
                self.editor_selected.clear()
            if label_id > 0:
                if bool(additive) and label_id in self.editor_selected:
                    self.editor_selected.remove(label_id)
                else:
                    self.editor_selected.add(label_id)
            preview = RESULT_DIR / f"{sample}_web_editor.png"
            _render_editor_preview(
                DATASET_DIR / f"{sample}_sem.png", labels, self.editor_selected, preview
            )
            self.preview_path = preview
            self.status = f"人工校正：已选择 {len(self.editor_selected)} 个晶粒"

    def editor_apply(self, operation: str, points: Any = None) -> None:
        with self.lock:
            labels = self.editor_labels
            sample = self.sample
            if labels is None or sample is None:
                raise RuntimeError("请先进入人工校正")
            if operation == "undo":
                if self.editor_undo:
                    self.editor_redo.append(labels.copy())
                    self.editor_labels = self.editor_undo.pop()
            elif operation == "redo":
                if self.editor_redo:
                    self.editor_undo.append(labels.copy())
                    self.editor_labels = self.editor_redo.pop()
            else:
                if operation == "delete":
                    if not self.editor_selected:
                        raise ValueError("请先选择至少一个晶粒")
                    changed = delete_mask_labels(labels, self.editor_selected)
                elif operation == "merge":
                    changed = merge_mask_labels(labels, self.editor_selected)
                elif operation == "add":
                    changed = add_polygon_mask(labels, self._parse_points(points, 3))
                elif operation == "split":
                    if len(self.editor_selected) != 1:
                        raise ValueError("切割前必须只选择一个晶粒")
                    changed = split_mask_label(
                        labels,
                        next(iter(self.editor_selected)),
                        self._parse_points(points, 2),
                    )
                else:
                    raise ValueError("未知的校正操作")
                self.editor_undo.append(labels.copy())
                if len(self.editor_undo) > 20:
                    self.editor_undo.pop(0)
                self.editor_redo.clear()
                self.editor_labels = changed
            self.editor_selected.clear()
            self.editor_dirty = bool(self.editor_undo)
            preview = RESULT_DIR / f"{sample}_web_editor.png"
            _render_editor_preview(
                DATASET_DIR / f"{sample}_sem.png",
                self.editor_labels,
                set(),
                preview,
            )
            self.preview_path = preview
            self.status = "人工校正结果尚未保存"

    def save_editor(self) -> None:
        with self.lock:
            if self.editor_labels is None or self.sample is None:
                raise RuntimeError("没有可保存的人工校正")
            labels = self.editor_labels.copy()
            sample = self.sample
            um_per_px = self.um_per_px

        def task() -> tuple[str, Path]:
            mask_path = RESULT_DIR / f"{sample}_sem_cp_masks.tif"
            backup = mask_path.with_name(f"{mask_path.stem}_auto_backup{mask_path.suffix}")
            if not backup.exists():
                shutil.copy2(mask_path, backup)
            dtype = np.uint16 if int(labels.max()) <= np.iinfo(np.uint16).max else np.uint32
            tifffile.imwrite(mask_path, labels.astype(dtype))
            self._refresh_analysis(sample, um_per_px)
            with self.lock:
                self._clear_editor()
            return "人工校正已保存，统计结果已更新", (
                RESULT_DIR / f"{sample}_sem_cpsam_overlay.png"
            )

        self.start_task("mask-save", "正在保存并更新人工校正结果", task)

    def discard_editor(self) -> None:
        with self.lock:
            self._clear_editor()
            if self.sample:
                overlay = RESULT_DIR / f"{self.sample}_sem_cpsam_overlay.png"
                if overlay.exists():
                    self.preview_path = overlay
            self.status = "已放弃未保存的人工校正"

    def shutdown(self) -> None:
        with self.lock:
            if self.shutdown_started:
                return
            self.shutdown_started = True
            server = self.server
        self.controller.shutdown("软件关闭")
        if server is not None:
            server.shutdown()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "SEMGrainLocal/1.0"

    @property
    def app(self) -> WebApplication:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'",
        )
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _authorized(self, query: dict[str, list[str]] | None = None) -> bool:
        supplied = self.headers.get("X-SEM-Token")
        if supplied is None and query is not None:
            supplied = (query.get("token") or [None])[0]
        return bool(supplied) and secrets.compare_digest(str(supplied), self.app.token)

    def _require_api_auth(self) -> bool:
        if self._authorized():
            return True
        self._json({"error": "unauthorized"}, HTTPStatus.FORBIDDEN)
        return False

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            index_path = STATIC_ROOT / "index.html"
            try:
                page = index_path.read_text(encoding="utf-8").replace("__SEM_TOKEN__", self.app.token)
            except OSError as exc:
                self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            if self._require_api_auth():
                self._json(self.app.snapshot())
            return
        if parsed.path.startswith("/files/"):
            query = urllib.parse.parse_qs(parsed.query)
            if not self._authorized(query):
                self._json({"error": "unauthorized"}, HTTPStatus.FORBIDDEN)
                return
            self._serve_file(parsed.path)
            return
        self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _serve_file(self, request_path: str) -> None:
        parts = request_path.split("/", 3)
        if len(parts) != 4 or parts[2] not in FILE_ROOTS:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        root = FILE_ROOTS[parts[2]].resolve()
        relative = Path(urllib.parse.unquote(parts[3]))
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            self._json({"error": "invalid path"}, HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(candidate.stat().st_size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{urllib.parse.quote(candidate.name)}")
        self.end_headers()
        with candidate.open("rb") as source:
            shutil.copyfileobj(source, self.wfile)

    def do_POST(self) -> None:
        if not self._require_api_auth():
            return
        parsed = urllib.parse.urlparse(self.path)
        try:
            if parsed.path == "/api/upload":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_UPLOAD_BYTES:
                    raise ValueError("上传大小无效或超过 512 MB")
                filename = (urllib.parse.parse_qs(parsed.query).get("name") or [""])[0]
                self.app.upload(filename, self.rfile.read(length))
            else:
                payload = self._read_json()
                self._dispatch(parsed.path, payload)
        except (ValueError, RuntimeError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:
            self.app.append_log(f"HTTP ERROR: {exc}")
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._json({"ok": True, "state": self.app.snapshot()})

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1024 * 1024:
            raise ValueError("请求内容过大")
        if length == 0:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("请求格式必须是 JSON 对象")
        return value

    def _dispatch(self, path: str, data: dict[str, Any]) -> None:
        if path == "/api/scale/auto":
            self.app.start_auto_scale(data.get("scale_um"))
        elif path == "/api/scale/manual":
            self.app.start_manual_scale(data.get("scale_um"), data.get("points"))
        elif path == "/api/detect":
            self.app.start_detection(data.get("use_gpu"), data.get("scale_um"))
        elif path == "/api/export":
            self.app.start_export(data.get("area_scale"), data.get("circle_gap"))
        elif path == "/api/cancel":
            self.app.cancel()
        elif path == "/api/editor/start":
            self.app.start_editor()
        elif path == "/api/editor/select":
            self.app.editor_select(data.get("x"), data.get("y"), data.get("additive"))
        elif path == "/api/editor/apply":
            self.app.editor_apply(str(data.get("operation")), data.get("points"))
        elif path == "/api/editor/save":
            self.app.save_editor()
        elif path == "/api/editor/discard":
            self.app.discard_editor()
        elif path == "/api/shutdown":
            threading.Thread(target=self.app.shutdown, daemon=True).start()
        else:
            raise ValueError("未知的操作")


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], app: WebApplication) -> None:
        super().__init__(address, RequestHandler)
        self.app = app


def main() -> int:
    app = WebApplication()
    if not (STATIC_ROOT / "index.html").is_file():
        raise SystemExit(f"missing web interface: {STATIC_ROOT / 'index.html'}")
    server = LocalServer(("127.0.0.1", 0), app)
    app.server = server
    port = int(server.server_address[1])
    url = f"http://127.0.0.1:{port}/"
    app.append_log(f"本地服务已启动：{url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        app.controller.shutdown("服务结束")
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
