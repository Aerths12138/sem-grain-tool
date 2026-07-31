import datetime as dt
import json
import os
import re
import runpy
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


import threading

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


class GrainApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("SEM Grain Tool")
        self.geometry("1180x760")
        self.minsize(980, 640)

        self.input_path: Path | None = None
        self.sample: str | None = None
        self.preview_image: ImageTk.PhotoImage | None = None

        self.area_scale = tk.StringVar(value="0.95")
        self.circle_gap = tk.StringVar(value="3")
        self.scale_um = tk.StringVar(value="100")
        self.um_per_px: float | None = None
        self.status = tk.StringVar(value="请选择一张电子显微镜图像")

        self._build_ui()
        self._ensure_dirs()

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
        self.export_btn = ttk.Button(left, text="输出间隔圆形图和 DXF", command=self.export_circles, state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=(0, 14))

        scale_params = ttk.LabelFrame(left, text="Scale calibration", padding=8)
        scale_params.pack(fill=tk.X, pady=(0, 12))
        ttk.Label(scale_params, text="Scale bar length (um)").pack(anchor=tk.W)
        ttk.Entry(scale_params, textvariable=self.scale_um, width=12).pack(fill=tk.X, pady=(0, 8))
        self.scale_btn = ttk.Button(scale_params, text="Detect scale bar", command=self.detect_scale_bar, state=tk.DISABLED)
        self.scale_btn.pack(fill=tk.X)

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

    def run_command(self, command: list[str]) -> None:
        env = os.environ.copy()
        env["USERPROFILE"] = str(CELLPOSE_CACHE)
        env["CELLPOSE_LOCAL_MODELS_PATH"] = str(MODEL_CACHE)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            self.after(0, self.append_log, line.rstrip())
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"command failed with exit code {code}")

    def run_background(self, label: str, task, on_success=None) -> None:
        def worker() -> None:
            self.after(0, self.status.set, label)
            self.after(0, self.detect_btn.config, {"state": tk.DISABLED})
            self.after(0, self.scale_btn.config, {"state": tk.DISABLED})
            self.after(0, self.export_btn.config, {"state": tk.DISABLED})
            try:
                task()
            except Exception as exc:
                self.after(0, messagebox.showerror, "错误", str(exc))
                self.after(0, self.status.set, "任务失败")
            else:
                if on_success:
                    self.after(0, on_success)
            finally:
                self.after(0, self.detect_btn.config, {"state": tk.NORMAL if self.input_path else tk.DISABLED})
                self.after(0, self.scale_btn.config, {"state": tk.NORMAL if self.input_path else tk.DISABLED})

        threading.Thread(target=worker, daemon=True).start()

    def detect_scale_bar_for_current_image(self) -> float:
        if not self.input_path or not self.sample:
            raise RuntimeError("No SEM image selected")
        scale_um = float(self.scale_um.get())
        if scale_um <= 0:
            raise RuntimeError("Scale bar length must be positive")

        self.run_command([
            str(HELPER_EXE),
            "--helper",
            "detect_scale_bar.py",
            "--image",
            str(self.input_path),
            "--sample",
            self.sample,
            "--scale-um",
            str(scale_um),
        ])
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
    def detect_grains(self) -> None:
        if not self.sample:
            return

        def task() -> None:
            image_rel = f"cellpose_dataset\\{self.sample}_sem.png"
            um_per_px = self.um_per_px
            if um_per_px is None:
                try:
                    um_per_px = self.detect_scale_bar_for_current_image()
                except Exception as exc:
                    self.after(0, self.append_log, f"scale bar detection skipped: {exc}")
            self.run_command([
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
            ])
            self.run_command([str(HELPER_EXE), "--helper", "make_cellpose_overlay.py", "--sample", self.sample, "--skip-gt"])
            analyze_command = [str(HELPER_EXE), "--helper", "analyze_particles.py", "--sample", self.sample, "--bins", "10"]
            if um_per_px is not None:
                analyze_command.extend(["--um-per-px", f"{um_per_px:.10g}"])
            self.run_command(analyze_command)

        def success() -> None:
            overlay = RESULT_DIR / f"{self.sample}_sem_cpsam_overlay.png"
            self.show_image(overlay)
            self.export_btn.config(state=tk.NORMAL)
            self.status.set("晶粒识别完成，可以导出圆形 DXF")

        self.run_background("正在识别晶粒，CPU 环境可能需要较长时间", task, success)

    def export_circles(self) -> None:
        if not self.sample:
            return
        try:
            area_scale = float(self.area_scale.get())
            circle_gap = float(self.circle_gap.get())
        except ValueError:
            messagebox.showerror("参数错误", "面积比例和圆间距必须是数字")
            return

        def task() -> None:
            self.run_command([
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
                "--resolve-iterations",
                "1200",
                "--resolve-damping",
                "0.22",
                "--max-center-shift-ratio",
                "1.0",
            ])

        def success() -> None:
            overlay = SHAPE_DIR / f"{self.sample}_circle_nonoverlap_a{area_scale:g}_gap{circle_gap:g}_area_from_centers_overlay.png"
            self.show_image(overlay)
            self.export_btn.config(state=tk.NORMAL)
            self.status.set("圆形图和 DXF 已输出")

        self.run_background("正在生成圆形图和 DXF", task, success)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--helper":
        raise SystemExit(helper_main(sys.argv[2:]))
    app = GrainApp()
    app.mainloop()
