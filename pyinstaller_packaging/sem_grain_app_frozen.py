import datetime as dt
import json
import math
import os
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = app_root()
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", ROOT))
HELPER_EXE = Path(sys.executable) if getattr(sys, "frozen", False) else Path(sys.executable)
MODEL_CACHE = RESOURCE_ROOT / "cellpose_cache" / "models"
CELLPOSE_CACHE = ROOT / "cellpose_cache"
DATASET_DIR = ROOT / "cellpose_dataset"
RESULT_DIR = ROOT / "cellpose_results_pretrained"
ANALYSIS_DIR = ROOT / "particle_analysis"
SHAPE_DIR = ROOT / "coordinate_shape_outputs"

PROCESS_TIMEOUTS_SECONDS = {
    "scale-bar": 120.0,
    "cellpose": 1800.0,
    "overlay": 300.0,
    "particle-analysis": 300.0,
    "circle-export": 600.0,
}


def helper_main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("missing helper command")
    command = argv[0]
    rest = argv[1:]
    sys.argv = [command] + rest
    if command == "cellpose":
        runpy.run_module("cellpose", run_name="__main__")
        return 0
    script = RESOURCE_ROOT / command
    if not script.exists():
        raise SystemExit(f"missing helper script: {script}")
    namespace = {
        "__name__": "__main__",
        "__file__": str(ROOT / command),
        "__package__": None,
        "__cached__": None,
    }
    source = script.read_text(encoding="utf-8")
    exec(compile(source, str(script), "exec"), namespace)
    return 0


if len(sys.argv) >= 2 and sys.argv[1] == "--helper":
    raise SystemExit(helper_main(sys.argv[2:]))


import queue
import signal
import threading
import time
from collections.abc import Callable

if getattr(sys, "frozen", False):
    os.environ.setdefault("TCL_LIBRARY", str(RESOURCE_ROOT / "_tcl_data" / "tcl8.6"))
    os.environ.setdefault("TK_LIBRARY", str(RESOURCE_ROOT / "_tcl_data" / "tk8.6"))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageTk

def safe_sample_name(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem).strip("_")
    if not stem:
        stem = "sem"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{stamp}"


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


def find_sem_crop(gray: np.ndarray) -> int:
    row_mean = gray.mean(axis=1)
    dark_rows = np.where(row_mean < 20)[0]
    if len(dark_rows) == 0:
        return gray.shape[0]
    bottom_dark = dark_rows[dark_rows > gray.shape[0] * 0.75]
    if len(bottom_dark) == 0:
        return gray.shape[0]
    return int(bottom_dark[0])


def read_sem_image(path: Path) -> np.ndarray:
    try:
        arr = tifffile.imread(path)
        return normalize_to_uint8(arr)
    except Exception:
        return np.array(Image.open(path).convert("L"))


def save_png_cv2(path: Path, image: np.ndarray) -> None:
    """OpenCV 在中文路径下直接 imwrite 可能失败，用 imencode + tofile 写入。"""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def prepare_input_image(path: Path, sample: str) -> Path:
    DATASET_DIR.mkdir(exist_ok=True)
    gray = read_sem_image(path)
    crop_bottom = find_sem_crop(gray)
    out_path = DATASET_DIR / f"{sample}_sem.png"
    save_png_cv2(out_path, gray[:crop_bottom])
    return out_path


def dxf_scale_mm_per_pixel(um_per_px: float) -> float:
    value = float(um_per_px)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("um_per_px must be a positive finite number")
    return value / 1000.0


def manual_scale_values(
    scale_um: float,
    first_point: tuple[float, float],
    second_point: tuple[float, float],
) -> tuple[float, float]:
    known_length = float(scale_um)
    if not math.isfinite(known_length) or known_length <= 0:
        raise ValueError("Scale bar length must be positive")
    pixel_length = math.dist(first_point, second_point)
    if pixel_length < 2.0:
        raise ValueError("The two scale points are too close")
    return pixel_length, known_length / pixel_length


def save_manual_scale_calibration(
    image_path: Path,
    output_dir: Path,
    sample: str,
    scale_um: float,
    first_point: tuple[float, float],
    second_point: tuple[float, float],
) -> dict[str, float | int | str]:
    pixel_length, um_per_px = manual_scale_values(scale_um, first_point, second_point)
    gray = read_sem_image(image_path)
    x0, y0 = first_point
    x1, y1 = second_point
    result: dict[str, float | int | str] = {
        "method": "manual-two-point",
        "image_path": str(image_path),
        "scale_um": float(scale_um),
        "bar_pixel_length": pixel_length,
        "um_per_px": um_per_px,
        "bar_x0_px": x0,
        "bar_y0_px": y0,
        "bar_x1_px": x1,
        "bar_y1_px": y1,
        "image_width_px": int(gray.shape[1]),
        "image_height_px": int(gray.shape[0]),
    }
    output_dir.mkdir(exist_ok=True)
    json_path = output_dir / f"{sample}_scale_bar.json"
    preview_path = output_dir / f"{sample}_scale_bar_detection.png"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    point0 = (int(round(x0)), int(round(y0)))
    point1 = (int(round(x1)), int(round(y1)))
    cv2.line(preview, point0, point1, (0, 0, 255), 2)
    cv2.circle(preview, point0, 5, (0, 255, 255), -1)
    cv2.circle(preview, point1, 5, (0, 255, 255), -1)
    label = f"{float(scale_um):g} um = {pixel_length:.1f} px"
    label_x = max(0, min(point0[0], point1[0]) - 20)
    label_y = max(24, min(point0[1], point1[1]) - 12)
    cv2.putText(preview, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    save_png_cv2(preview_path, preview)
    return result


def delete_mask_labels(labels: np.ndarray, label_ids: set[int]) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int32).copy()
    selected = {int(label_id) for label_id in label_ids if int(label_id) > 0}
    if selected:
        result[np.isin(result, list(selected))] = 0
    return result


def merge_mask_labels(labels: np.ndarray, label_ids: set[int]) -> np.ndarray:
    selected = sorted({int(label_id) for label_id in label_ids if int(label_id) > 0})
    if len(selected) < 2:
        raise ValueError("Select at least two particles to merge")
    result = np.asarray(labels, dtype=np.int32).copy()
    target = selected[0]
    result[np.isin(result, selected)] = target
    return result


def add_polygon_mask(labels: np.ndarray, points: list[tuple[int, int]]) -> np.ndarray:
    if len(points) < 3:
        raise ValueError("At least three points are required to add a particle")
    result = np.asarray(labels, dtype=np.int32).copy()
    polygon = np.zeros(result.shape, dtype=np.uint8)
    cv2.fillPoly(polygon, [np.asarray(points, dtype=np.int32)], 1)
    new_region = (polygon != 0) & (result == 0)
    if int(new_region.sum()) < 3:
        raise ValueError("The new particle does not contain enough background pixels")
    result[new_region] = int(result.max()) + 1
    return result


def split_mask_label(
    labels: np.ndarray,
    label_id: int,
    points: list[tuple[int, int]],
    line_width: int = 3,
) -> np.ndarray:
    if label_id <= 0 or not np.any(labels == label_id):
        raise ValueError("Select one existing particle to split")
    if len(points) < 2:
        raise ValueError("At least two points are required for a cut line")

    selected = (np.asarray(labels) == int(label_id)).astype(np.uint8)
    cut = np.zeros(selected.shape, dtype=np.uint8)
    cv2.polylines(
        cut,
        [np.asarray(points, dtype=np.int32)],
        isClosed=False,
        color=1,
        thickness=max(1, int(line_width)),
    )
    cut_selected = selected.copy()
    cut_selected[cut != 0] = 0
    component_count, components = cv2.connectedComponents(cut_selected, connectivity=8)
    component_ids = [
        component_id
        for component_id in range(1, component_count)
        if int(np.count_nonzero(components == component_id)) >= 3
    ]
    if len(component_ids) < 2:
        raise ValueError("The cut line must cross the selected particle completely")

    component_ids.sort(
        key=lambda component_id: int(np.count_nonzero(components == component_id)),
        reverse=True,
    )
    distances = np.stack([
        cv2.distanceTransform(
            (components != component_id).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        for component_id in component_ids
    ])
    nearest_component = np.argmin(distances, axis=0)
    result = np.asarray(labels, dtype=np.int32).copy()
    result[selected != 0] = 0
    next_label = int(np.asarray(labels).max()) + 1
    output_ids = [int(label_id)] + list(
        range(next_label, next_label + len(component_ids) - 1)
    )
    for component_index, output_id in enumerate(output_ids):
        result[(selected != 0) & (nearest_component == component_index)] = output_id
    return result


class TaskCancelled(RuntimeError):
    pass


class TaskTimedOut(TimeoutError):
    pass


class ProcessController:
    def __init__(self, log_callback: Callable[[str], None]) -> None:
        self._log_callback = log_callback
        self._cancel_event = threading.Event()
        self._cancel_reason = "cancelled"
        self._process_lock = threading.Lock()
        self._termination_lock = threading.Lock()
        self._active_process: subprocess.Popen[str] | None = None
        self._active_phase: str | None = None

    @property
    def active_pid(self) -> int | None:
        with self._process_lock:
            process = self._active_process
            if process is None or process.poll() is not None:
                return None
            return process.pid

    def reset_cancel(self) -> None:
        with self._process_lock:
            if self._active_process is not None and self._active_process.poll() is None:
                raise RuntimeError("cannot reset cancellation while a process is active")
            self._cancel_reason = "cancelled"
            self._cancel_event.clear()

    def request_cancel(self, reason: str) -> bool:
        with self._process_lock:
            process = self._active_process
            active = process is not None and process.poll() is None
            if self._cancel_event.is_set():
                return active
            self._cancel_reason = reason
            self._cancel_event.set()
        self._emit(f"[control] cancel requested reason={reason}")
        return active

    def shutdown(self, reason: str = "window closed") -> None:
        self.request_cancel(reason)
        with self._process_lock:
            process = self._active_process
            phase = self._active_phase or "unknown"
        if process is not None and process.poll() is None:
            self._terminate_process_tree(process, phase, reason)

    def run(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        phase: str,
        timeout_seconds: float,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self._cancel_event.is_set():
            raise TaskCancelled(self._cancel_reason)

        started_at = dt.datetime.now()
        started_monotonic = time.monotonic()
        self._emit(
            f"[{started_at:%Y-%m-%d %H:%M:%S}] START phase={phase} "
            f"timeout={timeout_seconds:g}s"
        )

        popen_kwargs: dict[str, object] = {}
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
        with self._process_lock:
            self._active_process = process
            self._active_phase = phase
        self._emit(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] PID phase={phase} pid={process.pid}")

        output_queue: queue.Queue[str] = queue.Queue()
        reader_done = threading.Event()

        def read_output() -> None:
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    output_queue.put(line.rstrip("\r\n"))
            finally:
                reader_done.set()

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        try:
            while True:
                self._drain_output(output_queue)
                elapsed = time.monotonic() - started_monotonic
                if self._cancel_event.is_set():
                    reason = self._cancel_reason
                    self._terminate_process_tree(process, phase, reason)
                    raise TaskCancelled(reason)
                if elapsed >= timeout_seconds:
                    reason = f"timeout after {timeout_seconds:g}s"
                    self._terminate_process_tree(process, phase, reason)
                    raise TaskTimedOut(f"{phase}: {reason}")
                if process.poll() is not None and reader_done.is_set():
                    break
                time.sleep(0.05)

            reader.join(timeout=1.0)
            self._drain_output(output_queue)
            code = process.wait(timeout=1.0)
            elapsed = time.monotonic() - started_monotonic
            if code != 0:
                self._emit(
                    f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] FAIL phase={phase} "
                    f"pid={process.pid} reason=exit-code-{code} elapsed={elapsed:.1f}s"
                )
                raise RuntimeError(f"{phase} failed with exit code {code}")
            self._emit(
                f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] DONE phase={phase} "
                f"pid={process.pid} elapsed={elapsed:.1f}s"
            )
        finally:
            if process.poll() is None:
                self._terminate_process_tree(process, phase, "final cleanup")
            reader.join(timeout=1.0)
            self._drain_output(output_queue)
            if process.stdout is not None:
                process.stdout.close()
            with self._process_lock:
                if self._active_process is process:
                    self._active_process = None
                    self._active_phase = None

    def _drain_output(self, output_queue: queue.Queue[str]) -> None:
        while True:
            try:
                self._emit(output_queue.get_nowait())
            except queue.Empty:
                return

    def _emit(self, text: str) -> None:
        try:
            self._log_callback(text)
        except Exception:
            pass

    def _terminate_process_tree(
        self,
        process: subprocess.Popen[str],
        phase: str,
        reason: str,
    ) -> None:
        with self._termination_lock:
            if process.poll() is not None:
                return
            self._emit(
                f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] STOP phase={phase} "
                f"pid={process.pid} reason={reason}"
            )
            try:
                if os.name == "nt":
                    result = subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        errors="replace",
                        timeout=10,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        check=False,
                    )
                    if result.stdout:
                        for line in result.stdout.splitlines():
                            self._emit(f"[control] {line}")
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    self._emit(
                        f"[control] process tree cleanup incomplete pid={process.pid}"
                    )


class ManualScaleDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        image_path: Path,
        on_apply: Callable[[tuple[float, float], tuple[float, float]], None],
    ) -> None:
        super().__init__(parent)
        self.title("人工比例尺标定")
        self.transient(parent)
        self.grab_set()
        self._on_apply = on_apply
        self._source = Image.open(image_path).convert("RGB")
        self._points: list[tuple[float, float]] = []

        max_width = max(480, min(1100, self.winfo_screenwidth() - 180))
        max_height = max(360, min(720, self.winfo_screenheight() - 220))
        display = self._source.copy()
        display.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        self._scale_x = display.width / self._source.width
        self._scale_y = display.height / self._source.height
        self._photo = ImageTk.PhotoImage(display)

        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="依次点击比例尺的两个端点").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="重置", command=self._reset).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(toolbar, text="应用", command=self._apply).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            self,
            width=display.width,
            height=display.height,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(padx=8, pady=(0, 8))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        self.canvas.bind("<Button-1>", self._click)

    def _click(self, event: tk.Event) -> None:
        if len(self._points) == 2:
            self._points.clear()
        self._points.append((event.x / self._scale_x, event.y / self._scale_y))
        self._draw_points()

    def _draw_points(self) -> None:
        self.canvas.delete("scale-selection")
        display_points = [
            (x * self._scale_x, y * self._scale_y)
            for x, y in self._points
        ]
        if len(display_points) == 2:
            self.canvas.create_line(
                *display_points[0],
                *display_points[1],
                fill="#ff3030",
                width=3,
                tags="scale-selection",
            )
        for x, y in display_points:
            self.canvas.create_oval(
                x - 5,
                y - 5,
                x + 5,
                y + 5,
                fill="#ffe45c",
                outline="#c00000",
                width=2,
                tags="scale-selection",
            )

    def _reset(self) -> None:
        self._points.clear()
        self._draw_points()

    def _apply(self) -> None:
        if len(self._points) != 2:
            messagebox.showerror("比例尺", "请先点击比例尺的两个端点。", parent=self)
            return
        try:
            self._on_apply(self._points[0], self._points[1])
        except Exception as exc:
            messagebox.showerror("比例尺", str(exc), parent=self)
            return
        self.destroy()


class MaskEditor(tk.Toplevel):
    HISTORY_LIMIT = 20

    def __init__(
        self,
        parent: tk.Misc,
        image_path: Path,
        mask_path: Path,
        on_save: Callable[[], None],
    ) -> None:
        super().__init__(parent)
        self.title("晶粒人工校正")
        self.geometry("1180x820")
        self.minsize(820, 600)
        self.transient(parent)
        self._mask_path = mask_path
        self._on_save = on_save
        self._source = Image.open(image_path).convert("RGB")
        self._labels = tifffile.imread(mask_path).astype(np.int32)
        if self._labels.ndim != 2 or self._labels.shape != (self._source.height, self._source.width):
            raise ValueError("Mask dimensions do not match the prepared SEM image")

        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self._selected: set[int] = set()
        self._pending_points: list[tuple[int, int]] = []
        self._mode = "select"
        self._dirty = False
        self._photo: ImageTk.PhotoImage | None = None
        self._render_after_id: str | None = None
        self._display_scale_x = 1.0
        self._display_scale_y = 1.0
        self._status = tk.StringVar(value="点击选择晶粒；按 Ctrl 可多选")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._schedule_render()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="选择", command=lambda: self._set_mode("select")).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="补画", command=lambda: self._set_mode("add")).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="切割", command=lambda: self._set_mode("split")).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="完成绘制", command=self._finish_drawing).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(toolbar, text="删除", command=self._delete_selected).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="合并", command=self._merge_selected).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="撤销", command=self._undo_change).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(toolbar, text="重做", command=self._redo_change).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="保存校正", command=self._save).pack(side=tk.RIGHT)

        ttk.Label(self, textvariable=self._status, padding=(8, 0, 8, 6)).pack(fill=tk.X)
        frame = ttk.Frame(self, padding=(8, 0, 8, 8))
        frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(frame, background="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Configure>", lambda _event: self._schedule_render())

    def _schedule_render(self) -> None:
        if self._render_after_id is None:
            self._render_after_id = self.after_idle(self._run_scheduled_render)

    def _run_scheduled_render(self) -> None:
        self._render_after_id = None
        if self.winfo_exists():
            self._render()

    def _cancel_scheduled_render(self) -> None:
        if self._render_after_id is not None:
            self.after_cancel(self._render_after_id)
            self._render_after_id = None

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._pending_points.clear()
        if mode == "select":
            self._status.set("点击选择晶粒；按 Ctrl 可多选")
        elif mode == "add":
            self._status.set("沿新晶粒边界点击至少三个点，然后点击“完成绘制”")
        else:
            self._status.set("先选择一个晶粒，再沿切割方向点击至少两个点")
        self._render()

    def _click(self, event: tk.Event) -> None:
        x = int(round(event.x / self._display_scale_x))
        y = int(round(event.y / self._display_scale_y))
        if not (0 <= x < self._labels.shape[1] and 0 <= y < self._labels.shape[0]):
            return
        if self._mode == "select":
            label_id = int(self._labels[y, x])
            if not (event.state & 0x0004):
                self._selected.clear()
            if label_id > 0:
                if label_id in self._selected and (event.state & 0x0004):
                    self._selected.remove(label_id)
                else:
                    self._selected.add(label_id)
            self._status.set(f"已选择 {len(self._selected)} 个晶粒")
        else:
            self._pending_points.append((x, y))
            self._status.set(f"已记录 {len(self._pending_points)} 个点；完成后点击“完成绘制”")
        self._render()

    def _render(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        if width <= 2 or height <= 2:
            return
        scale = min(width / self._source.width, height / self._source.height)
        display_width = max(1, int(round(self._source.width * scale)))
        display_height = max(1, int(round(self._source.height * scale)))
        self._display_scale_x = display_width / self._source.width
        self._display_scale_y = display_height / self._source.height

        base = np.asarray(
            self._source.resize((display_width, display_height), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        ).copy()
        labels = cv2.resize(
            self._labels,
            (display_width, display_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
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

        if self._selected:
            selected = np.isin(labels, list(self._selected))
            base[selected] = (
                base[selected].astype(np.float32) * 0.35
                + np.array([255, 224, 60], dtype=np.float32) * 0.65
            ).astype(np.uint8)

        self._photo = ImageTk.PhotoImage(Image.fromarray(base))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        self._draw_pending()

    def _draw_pending(self) -> None:
        if not self._pending_points:
            return
        points = [
            (x * self._display_scale_x, y * self._display_scale_y)
            for x, y in self._pending_points
        ]
        if len(points) >= 2:
            coordinates = [coordinate for point in points for coordinate in point]
            self.canvas.create_line(*coordinates, fill="#ff3030", width=3)
        for x, y in points:
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ffe45c", outline="#c00000")

    def _apply_change(self, new_labels: np.ndarray) -> None:
        self._undo.append(self._labels.copy())
        if len(self._undo) > self.HISTORY_LIMIT:
            self._undo.pop(0)
        self._redo.clear()
        self._labels = new_labels.astype(np.int32, copy=False)
        self._selected.clear()
        self._pending_points.clear()
        self._dirty = True
        self._render()

    def _delete_selected(self) -> None:
        if not self._selected:
            messagebox.showerror("人工校正", "请先选择要删除的晶粒。", parent=self)
            return
        self._apply_change(delete_mask_labels(self._labels, self._selected))
        self._status.set("已删除所选晶粒")

    def _merge_selected(self) -> None:
        try:
            new_labels = merge_mask_labels(self._labels, self._selected)
        except ValueError as exc:
            messagebox.showerror("人工校正", str(exc), parent=self)
            return
        self._apply_change(new_labels)
        self._status.set("已合并所选晶粒")

    def _finish_drawing(self) -> None:
        try:
            if self._mode == "add":
                new_labels = add_polygon_mask(self._labels, self._pending_points)
                message = "已补画晶粒"
            elif self._mode == "split":
                if len(self._selected) != 1:
                    raise ValueError("切割前必须只选择一个晶粒")
                new_labels = split_mask_label(
                    self._labels,
                    next(iter(self._selected)),
                    self._pending_points,
                )
                message = "已切割晶粒"
            else:
                raise ValueError("请先选择“补画”或“切割”模式")
        except ValueError as exc:
            messagebox.showerror("人工校正", str(exc), parent=self)
            return
        self._apply_change(new_labels)
        self._status.set(message)

    def _undo_change(self) -> None:
        if not self._undo:
            return
        self._redo.append(self._labels.copy())
        self._labels = self._undo.pop()
        self._selected.clear()
        self._pending_points.clear()
        self._dirty = True
        self._render()

    def _redo_change(self) -> None:
        if not self._redo:
            return
        self._undo.append(self._labels.copy())
        self._labels = self._redo.pop()
        self._selected.clear()
        self._pending_points.clear()
        self._dirty = True
        self._render()

    def _save(self) -> None:
        backup = self._mask_path.with_name(f"{self._mask_path.stem}_auto_backup{self._mask_path.suffix}")
        if not backup.exists():
            shutil.copy2(self._mask_path, backup)
        output_dtype = np.uint16 if int(self._labels.max()) <= np.iinfo(np.uint16).max else np.uint32
        tifffile.imwrite(self._mask_path, self._labels.astype(output_dtype))
        self._dirty = False
        self._cancel_scheduled_render()
        self.destroy()
        self._on_save()

    def _close(self) -> None:
        if self._dirty and not messagebox.askyesno("人工校正", "尚未保存修改，确定关闭吗？", parent=self):
            return
        self._cancel_scheduled_render()
        self.destroy()


class GrainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SEM Grain Tool")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.input_path: Path | None = None
        self.sample: str | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self._closing = False
        self._task_thread: threading.Thread | None = None
        self._ui_queue: queue.Queue[tuple[Callable, tuple]] = queue.Queue()
        self._ui_poll_id: str | None = None
        self._process_controller = ProcessController(self._queue_process_log)

        self.area_scale = tk.StringVar(value="0.95")
        self.circle_gap = tk.StringVar(value="3")
        self.scale_um = tk.StringVar(value="100")
        self.um_per_px: float | None = None
        self.status = tk.StringVar(value="请选择一张电子显微镜图像")

        self._build_ui()
        self._ensure_dirs()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._ui_poll_id = self.after(50, self._drain_ui_queue)

    def _ensure_dirs(self) -> None:
        for path in (MODEL_CACHE, CELLPOSE_CACHE, DATASET_DIR, RESULT_DIR, ANALYSIS_DIR, SHAPE_DIR):
            path.mkdir(exist_ok=True)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        ttk.Button(left, text="选择 SEM 图像", command=self.choose_image).pack(fill=tk.X, pady=(0, 8))
        self.detect_btn = ttk.Button(left, text="识别晶粒", command=self.detect_grains, state=tk.DISABLED)
        self.detect_btn.pack(fill=tk.X, pady=(0, 8))
        self.edit_btn = ttk.Button(left, text="人工校正晶粒", command=self.open_mask_editor, state=tk.DISABLED)
        self.edit_btn.pack(fill=tk.X, pady=(0, 8))
        self.export_btn = ttk.Button(left, text="输出间隔圆形图和 DXF", command=self.export_circles, state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=(0, 8))
        self.cancel_btn = ttk.Button(left, text="取消当前任务", command=self.cancel_current_task, state=tk.DISABLED)
        self.cancel_btn.pack(fill=tk.X, pady=(0, 14))

        scale_params = ttk.LabelFrame(left, text="Scale calibration", padding=8)
        scale_params.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(scale_params, text="Scale bar length (um)").pack(anchor=tk.W)
        ttk.Entry(scale_params, textvariable=self.scale_um, width=12).pack(fill=tk.X, pady=(0, 8))
        self.scale_btn = ttk.Button(scale_params, text="Detect scale bar", command=self.detect_scale_bar, state=tk.DISABLED)
        self.scale_btn.pack(fill=tk.X, pady=(0, 6))
        self.manual_scale_btn = ttk.Button(
            scale_params,
            text="人工两点标定",
            command=self.open_manual_scale_dialog,
            state=tk.DISABLED,
        )
        self.manual_scale_btn.pack(fill=tk.X)

        params = ttk.LabelFrame(left, text="圆形导出参数", padding=8)
        params.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(params, text="面积比例").pack(anchor=tk.W)
        ttk.Entry(params, textvariable=self.area_scale, width=12).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(params, text="圆间距(px)").pack(anchor=tk.W)
        ttk.Entry(params, textvariable=self.circle_gap, width=12).pack(fill=tk.X)

        ttk.Label(left, textvariable=self.status, wraplength=260).pack(fill=tk.X, pady=(0, 10))

        self.log = tk.Text(left, width=42, height=28)
        self.log.pack(fill=tk.BOTH, expand=True)

        self.preview = ttk.Label(right, anchor=tk.CENTER)
        self.preview.pack(fill=tk.BOTH, expand=True)

    def choose_image(self) -> None:
        filename = filedialog.askopenfilename(
            title="选择电子显微镜图像",
            filetypes=[
                ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not filename:
            return
        self.input_path = Path(filename)
        self.sample = safe_sample_name(self.input_path)
        self.um_per_px = None
        self.detect_btn.config(state=tk.NORMAL)
        self.scale_btn.config(state=tk.NORMAL)
        self.manual_scale_btn.config(state=tk.NORMAL)
        self.edit_btn.config(state=tk.DISABLED)
        self.export_btn.config(state=tk.DISABLED)
        self.status.set(f"已选择: {self.input_path.name}")
        self.log.delete("1.0", tk.END)
        self.append_log(f"sample = {self.sample}")
        prepared = prepare_input_image(self.input_path, self.sample)
        self.show_image(prepared)
        self.append_log(f"prepared image: {prepared}")

    def append_log(self, text: str) -> None:
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def show_image(self, path: Path) -> None:
        if not path.exists():
            return
        image = Image.open(path).convert("RGB")
        image.thumbnail((850, 700), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.preview.config(image=self.preview_image)

    def _post_ui(self, callback: Callable, *args) -> None:
        if self._closing:
            return
        self._ui_queue.put((callback, args))

    def _drain_ui_queue(self) -> None:
        if self._closing:
            return
        while True:
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            callback(*args)
        self._ui_poll_id = self.after(50, self._drain_ui_queue)

    def _queue_process_log(self, text: str) -> None:
        self._post_ui(self.append_log, text)

    def run_command(
        self,
        command: list[str],
        *,
        phase: str,
        timeout_seconds: float,
    ) -> None:
        env = os.environ.copy()
        env["USERPROFILE"] = str(CELLPOSE_CACHE)
        env["CELLPOSE_LOCAL_MODELS_PATH"] = str(MODEL_CACHE)
        self._process_controller.run(
            command,
            cwd=ROOT,
            env=env,
            phase=phase,
            timeout_seconds=timeout_seconds,
        )

    def cancel_current_task(self) -> None:
        thread = self._task_thread
        if thread is None or not thread.is_alive():
            return
        self._process_controller.request_cancel("cancelled by user")
        self.status.set("正在取消当前任务...")
        self.cancel_btn.config(state=tk.DISABLED)

    def run_background(
        self,
        label: str,
        task: Callable[[], None],
        on_success: Callable[[], None] | None = None,
    ) -> None:
        if self._task_thread is not None and self._task_thread.is_alive():
            messagebox.showinfo("任务进行中", "请先等待当前任务完成或取消当前任务。")
            return

        button_states = [
            (self.detect_btn, str(self.detect_btn.cget("state"))),
            (self.scale_btn, str(self.scale_btn.cget("state"))),
            (self.manual_scale_btn, str(self.manual_scale_btn.cget("state"))),
            (self.edit_btn, str(self.edit_btn.cget("state"))),
            (self.export_btn, str(self.export_btn.cget("state"))),
        ]
        self._process_controller.reset_cancel()
        self.status.set(label)
        for button, _state in button_states:
            button.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        started = time.monotonic()
        self.append_log(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] TASK START label={label}")

        def worker() -> None:
            outcome = "success"
            error: Exception | None = None
            try:
                task()
            except TaskCancelled as exc:
                outcome = "cancelled"
                error = exc
            except TaskTimedOut as exc:
                outcome = "timeout"
                error = exc
            except Exception as exc:
                outcome = "failed"
                error = exc
            finally:
                elapsed = time.monotonic() - started
                self._post_ui(
                    self._finish_background_task,
                    label,
                    on_success,
                    button_states,
                    outcome,
                    error,
                    elapsed,
                )

        self._task_thread = threading.Thread(target=worker, daemon=True)
        self._task_thread.start()

    def _finish_background_task(
        self,
        label: str,
        on_success: Callable[[], None] | None,
        button_states: list[tuple[ttk.Button, str]],
        outcome: str,
        error: Exception | None,
        elapsed: float,
    ) -> None:
        try:
            for button, state in button_states:
                button.config(state=state)
            self.cancel_btn.config(state=tk.DISABLED)
            if outcome == "success":
                self.append_log(
                    f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] TASK DONE "
                    f"label={label} elapsed={elapsed:.1f}s"
                )
                if on_success is not None:
                    on_success()
            elif outcome == "cancelled":
                reason = str(error) if error else "cancelled"
                self.append_log(
                    f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] TASK STOP "
                    f"label={label} reason={reason} elapsed={elapsed:.1f}s"
                )
                self.status.set("任务已取消")
            elif outcome == "timeout":
                reason = str(error) if error else "timeout"
                self.append_log(
                    f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] TASK STOP "
                    f"label={label} reason={reason} elapsed={elapsed:.1f}s"
                )
                self.status.set("任务超时")
                messagebox.showerror("任务超时", reason)
            else:
                reason = str(error) if error else "unknown error"
                self.append_log(
                    f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] TASK FAIL "
                    f"label={label} reason={reason} elapsed={elapsed:.1f}s"
                )
                self.status.set("任务失败")
                messagebox.showerror("错误", reason)
        except Exception as exc:
            self.status.set("任务失败")
            self.append_log(f"task finalization failed: {exc}")
        finally:
            self._task_thread = None

    def on_close(self) -> None:
        if self._closing:
            return
        self.append_log(
            f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] TASK STOP reason=window-closed"
        )
        self.cancel_btn.config(state=tk.DISABLED)
        self._closing = True
        if self._ui_poll_id is not None:
            self.after_cancel(self._ui_poll_id)
            self._ui_poll_id = None
        self._process_controller.shutdown("window closed")
        self.destroy()

    def detect_scale_bar_for_current_image(self) -> float:
        if not self.input_path or not self.sample:
            raise RuntimeError("No SEM image selected")
        scale_um = float(self.scale_um.get())
        if scale_um <= 0:
            raise RuntimeError("Scale bar length must be positive")

        self.run_command(
            [
                str(HELPER_EXE),
                "--helper",
                "detect_scale_bar.py",
                "--image",
                str(self.input_path),
                "--sample",
                self.sample,
                "--scale-um",
                str(scale_um),
            ],
            phase="scale-bar",
            timeout_seconds=PROCESS_TIMEOUTS_SECONDS["scale-bar"],
        )
        result_path = ANALYSIS_DIR / f"{self.sample}_scale_bar.json"
        data = json.loads(result_path.read_text(encoding="utf-8"))
        self.um_per_px = float(data["um_per_px"])
        self.after(
            0,
            self.append_log,
            f"scale calibration: {float(data['scale_um']):g} um / {float(data['bar_pixel_length']):.1f} px = {self.um_per_px:.6g} um/px",
        )
        return self.um_per_px

    def detect_scale_bar(self) -> None:
        try:
            float(self.scale_um.get())
        except ValueError:
            messagebox.showerror("Scale error", "Scale bar length must be a number")
            return

        def task() -> None:
            self.detect_scale_bar_for_current_image()

        def success() -> None:
            if self.sample:
                preview = ANALYSIS_DIR / f"{self.sample}_scale_bar_detection.png"
                self.show_image(preview)
            self.status.set("Scale bar detected")

        self.run_background("Detecting scale bar", task, success)

    def open_manual_scale_dialog(self) -> None:
        if not self.input_path or not self.sample:
            return
        try:
            scale_um = float(self.scale_um.get())
            if scale_um <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("比例尺", "比例尺实际长度必须是正数。")
            return

        def apply_points(
            first_point: tuple[float, float],
            second_point: tuple[float, float],
        ) -> None:
            result = save_manual_scale_calibration(
                self.input_path,
                ANALYSIS_DIR,
                self.sample,
                scale_um,
                first_point,
                second_point,
            )
            self.um_per_px = float(result["um_per_px"])
            self.append_log(
                f"manual scale calibration: {scale_um:g} um / "
                f"{float(result['bar_pixel_length']):.1f} px = {self.um_per_px:.6g} um/px"
            )
            self.status.set(f"人工比例尺已标定：{self.um_per_px:.6g} um/px")
            self.show_image(ANALYSIS_DIR / f"{self.sample}_scale_bar_detection.png")
            mask_path = RESULT_DIR / f"{self.sample}_sem_cp_masks.tif"
            if mask_path.exists():
                self.after_idle(self.refresh_current_mask_analysis)

        ManualScaleDialog(self, self.input_path, apply_points)

    def open_mask_editor(self) -> None:
        if not self.sample:
            return
        image_path = DATASET_DIR / f"{self.sample}_sem.png"
        mask_path = RESULT_DIR / f"{self.sample}_sem_cp_masks.tif"
        if not image_path.exists() or not mask_path.exists():
            messagebox.showerror("人工校正", "请先完成晶粒识别。")
            return
        try:
            editor = MaskEditor(
                self,
                image_path,
                mask_path,
                self.refresh_current_mask_analysis,
            )
            editor.grab_set()
        except Exception as exc:
            messagebox.showerror("人工校正", str(exc))

    def refresh_current_mask_analysis(self) -> None:
        if not self.sample:
            return

        def task() -> None:
            self.run_command(
                [
                    str(HELPER_EXE),
                    "--helper",
                    "make_cellpose_overlay.py",
                    "--sample",
                    self.sample,
                    "--skip-gt",
                ],
                phase="overlay",
                timeout_seconds=PROCESS_TIMEOUTS_SECONDS["overlay"],
            )
            analyze_command = [
                str(HELPER_EXE),
                "--helper",
                "analyze_particles.py",
                "--sample",
                self.sample,
                "--bins",
                "10",
            ]
            if self.um_per_px is not None:
                analyze_command.extend(["--um-per-px", f"{self.um_per_px:.10g}"])
            self.run_command(
                analyze_command,
                phase="particle-analysis",
                timeout_seconds=PROCESS_TIMEOUTS_SECONDS["particle-analysis"],
            )

        def success() -> None:
            overlay = RESULT_DIR / f"{self.sample}_sem_cpsam_overlay.png"
            self.show_image(overlay)
            self.edit_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            self.status.set("人工校正已保存，统计结果已更新")

        self.run_background("正在更新人工校正后的结果", task, success)

    def detect_grains(self) -> None:
        if not self.sample:
            return

        def task() -> None:
            image_rel = f"cellpose_dataset\\{self.sample}_sem.png"
            um_per_px = self.um_per_px
            if um_per_px is None:
                try:
                    um_per_px = self.detect_scale_bar_for_current_image()
                except (TaskCancelled, TaskTimedOut):
                    raise
                except Exception as exc:
                    self._queue_process_log(f"scale bar detection skipped: {exc}")
            self.run_command(
                [
                    str(HELPER_EXE),
                    "--helper",
                    "cellpose",
                    "--image_path",
                    image_rel,
                    "--pretrained_model",
                    "cpsam",
                    "--diameter",
                    "90",
                    "--save_tif",
                    "--savedir",
                    "cellpose_results_pretrained",
                    "--verbose",
                ],
                phase="cellpose",
                timeout_seconds=PROCESS_TIMEOUTS_SECONDS["cellpose"],
            )
            self.run_command(
                [
                    str(HELPER_EXE),
                    "--helper",
                    "make_cellpose_overlay.py",
                    "--sample",
                    self.sample,
                    "--skip-gt",
                ],
                phase="overlay",
                timeout_seconds=PROCESS_TIMEOUTS_SECONDS["overlay"],
            )
            analyze_command = [str(HELPER_EXE), "--helper", "analyze_particles.py", "--sample", self.sample, "--bins", "10"]
            if um_per_px is not None:
                analyze_command.extend(["--um-per-px", f"{um_per_px:.10g}"])
            self.run_command(
                analyze_command,
                phase="particle-analysis",
                timeout_seconds=PROCESS_TIMEOUTS_SECONDS["particle-analysis"],
            )

        def success() -> None:
            overlay = RESULT_DIR / f"{self.sample}_sem_cpsam_overlay.png"
            self.show_image(overlay)
            self.edit_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            self.status.set("晶粒识别完成，可以人工校正或导出圆形 DXF")

        self.run_background("正在识别晶粒，CPU 环境可能需要较长时间", task, success)

    def export_circles(self) -> None:
        if not self.sample:
            return
        if self.um_per_px is None:
            messagebox.showerror(
                "缺少比例尺",
                "请先自动检测或人工两点标定比例尺。未标定时不会导出 DXF，避免把像素误当成毫米。",
            )
            return
        try:
            area_scale = float(self.area_scale.get())
            circle_gap = float(self.circle_gap.get())
            dxf_scale = dxf_scale_mm_per_pixel(self.um_per_px)
        except ValueError:
            messagebox.showerror("参数错误", "面积比例和圆间距必须是数字")
            return

        def task() -> None:
            self.run_command(
                [
                    str(HELPER_EXE),
                    "--helper",
                    "generate_shapes_from_particles.py",
                    "--sample",
                    self.sample,
                    "--mode",
                    "circle",
                    "--area-scale",
                    str(area_scale),
                    "--circle-gap",
                    str(circle_gap),
                    "--dxf-scale",
                    f"{dxf_scale:.12g}",
                    "--resolve-iterations",
                    "1200",
                    "--resolve-damping",
                    "0.22",
                    "--max-center-shift-ratio",
                    "1.0",
                ],
                phase="circle-export",
                timeout_seconds=PROCESS_TIMEOUTS_SECONDS["circle-export"],
            )

        def success() -> None:
            overlay = SHAPE_DIR / f"{self.sample}_circle_nonoverlap_a{area_scale:g}_gap{circle_gap:g}_area_from_centers_overlay.png"
            self.show_image(overlay)
            self.export_btn.config(state=tk.NORMAL)
            self.append_log(
                f"DXF physical scale: 1 px = {self.um_per_px:.6g} um = {dxf_scale:.9g} mm"
            )
            self.status.set("圆形图和毫米单位 DXF 已输出")

        self.run_background("正在生成圆形图和 DXF", task, success)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--helper":
        raise SystemExit(helper_main(sys.argv[2:]))
    app = GrainApp()
    app.mainloop()
