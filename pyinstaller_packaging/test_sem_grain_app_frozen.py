import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

environment_root = Path(sys.executable).resolve().parent
portable_tcl = environment_root / "Library" / "lib" / "tcl8.6"
portable_tk = environment_root / "Library" / "lib" / "tk8.6"
if (portable_tcl / "init.tcl").is_file():
    os.environ.setdefault("TCL_LIBRARY", str(portable_tcl))
if (portable_tk / "tk.tcl").is_file():
    os.environ.setdefault("TK_LIBRARY", str(portable_tk))

import sem_grain_app_frozen as app_module

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import generate_shapes_from_particles as shape_module


def process_tree_command(seconds: int = 30) -> list[str]:
    child_code = f"import time; time.sleep({seconds})"
    parent_code = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', "
        f"{child_code!r}]); "
        "print(f'CHILD_PID={child.pid}', flush=True); "
        f"time.sleep({seconds})"
    )
    return [sys.executable, "-c", parent_code]


def pid_is_running(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        check=False,
    )
    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) >= 2 and row[1].isdigit() and int(row[1]) == pid:
            return True
    return False


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def logged_pid(logs: list[str], prefix: str) -> int:
    for line in logs:
        if prefix in line:
            return int(line.rsplit("=", 1)[-1])
    raise AssertionError(f"missing log prefix: {prefix}")


class CalibrationAndMaskTests(unittest.TestCase):
    def test_dxf_scale_converts_micrometers_to_millimeters(self) -> None:
        self.assertAlmostEqual(app_module.dxf_scale_mm_per_pixel(0.25), 0.00025)
        with self.assertRaises(ValueError):
            app_module.dxf_scale_mm_per_pixel(0)

    def test_manual_scale_writes_compatible_json_and_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "sem.png"
            Image.new("L", (100, 60), color=80).save(image_path)

            result = app_module.save_manual_scale_calibration(
                image_path,
                root,
                "sample",
                20.0,
                (10.0, 30.0),
                (50.0, 30.0),
            )

            self.assertEqual(result["method"], "manual-two-point")
            self.assertAlmostEqual(float(result["um_per_px"]), 0.5)
            self.assertTrue((root / "sample_scale_bar.json").is_file())
            self.assertTrue((root / "sample_scale_bar_detection.png").is_file())

    def test_dxf_geometry_and_metadata_use_millimeters(self) -> None:
        if shape_module.ezdxf is None:
            self.skipTest("ezdxf is not installed")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dxf_path = root / "sample.dxf"
            metadata_path = root / "sample_dxf_metadata.json"
            shapes = [shape_module.CircleShape(center_x=10.0, center_y=20.0, radius=2.0)]

            shape_module.write_dxf(dxf_path, shapes, height=100, scale=0.001)
            shape_module.write_dxf_metadata(
                metadata_path,
                root / "sem.png",
                dxf_path,
                width=200,
                height=100,
                shape_count=1,
                scale=0.001,
            )

            document = shape_module.ezdxf.readfile(dxf_path)
            circle = next(iter(document.modelspace().query("CIRCLE")))
            self.assertEqual(document.units, shape_module.ezdxf.units.MM)
            self.assertAlmostEqual(circle.dxf.center.x, 0.01)
            self.assertAlmostEqual(circle.dxf.center.y, 0.08)
            self.assertAlmostEqual(circle.dxf.radius, 0.002)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["dxf_unit"], "mm")
            self.assertAlmostEqual(metadata["geometry_width_mm"], 0.2)

    def test_mask_edit_operations_delete_merge_add_and_split(self) -> None:
        labels = np.zeros((12, 14), dtype=np.int32)
        labels[2:10, 2:7] = 1
        labels[3:8, 9:12] = 2

        merged = app_module.merge_mask_labels(labels, {1, 2})
        self.assertEqual(set(np.unique(merged)), {0, 1})

        deleted = app_module.delete_mask_labels(labels, {2})
        self.assertFalse(np.any(deleted == 2))

        added = app_module.add_polygon_mask(labels, [(9, 9), (12, 9), (12, 11), (9, 11)])
        self.assertGreater(int(added.max()), 2)

        split = app_module.split_mask_label(labels, 1, [(0, 6), (8, 6)], line_width=1)
        split_ids = set(np.unique(split[labels == 1])) - {0}
        self.assertGreaterEqual(len(split_ids), 2)
        self.assertEqual(np.count_nonzero(split), np.count_nonzero(labels))

    def test_mask_editor_save_keeps_automatic_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_path = Path(temp_dir)
            image_path = root_path / "sem.png"
            mask_path = root_path / "sem_cp_masks.tif"
            Image.new("RGB", (20, 16), color=(80, 80, 80)).save(image_path)
            labels = np.zeros((16, 20), dtype=np.uint16)
            labels[3:10, 4:12] = 1
            app_module.tifffile.imwrite(mask_path, labels)
            saved: list[bool] = []

            parent = app_module.tk.Tk()
            parent.withdraw()
            try:
                editor = app_module.MaskEditor(parent, image_path, mask_path, lambda: saved.append(True))
                editor.withdraw()
                editor._labels = app_module.delete_mask_labels(editor._labels, {1})
                editor._dirty = True
                editor._save()

                backup = root_path / "sem_cp_masks_auto_backup.tif"
                self.assertTrue(backup.is_file())
                self.assertEqual(np.count_nonzero(app_module.tifffile.imread(mask_path)), 0)
                self.assertEqual(np.count_nonzero(app_module.tifffile.imread(backup)), np.count_nonzero(labels))
                self.assertEqual(saved, [True])
            finally:
                parent.destroy()


class ProcessControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logs: list[str] = []
        self.controller = app_module.ProcessController(self.logs.append)
        self.controller.reset_cancel()

    def test_normal_process_logs_output_and_completion(self) -> None:
        self.controller.run(
            [sys.executable, "-c", "print('NORMAL_OUTPUT', flush=True)"],
            cwd=Path.cwd(),
            env=os.environ.copy(),
            phase="normal-test",
            timeout_seconds=5,
        )

        joined = "\n".join(self.logs)
        self.assertIn("START phase=normal-test", joined)
        self.assertIn("PID phase=normal-test", joined)
        self.assertIn("NORMAL_OUTPUT", joined)
        self.assertIn("DONE phase=normal-test", joined)
        self.assertIsNone(self.controller.active_pid)

    def test_timeout_terminates_process_tree_and_logs_reason(self) -> None:
        with self.assertRaises(app_module.TaskTimedOut):
            self.controller.run(
                process_tree_command(),
                cwd=Path.cwd(),
                env=os.environ.copy(),
                phase="timeout-test",
                timeout_seconds=0.8,
            )

        parent_pid = logged_pid(self.logs, "PID phase=timeout-test pid=")
        child_pid = logged_pid(self.logs, "CHILD_PID=")
        self.assertTrue(wait_until(lambda: not pid_is_running(parent_pid)))
        self.assertTrue(wait_until(lambda: not pid_is_running(child_pid)))
        joined = "\n".join(self.logs)
        self.assertIn("STOP phase=timeout-test", joined)
        self.assertIn("reason=timeout after 0.8s", joined)

    def test_cancel_terminates_process_tree_and_logs_reason(self) -> None:
        errors: list[Exception] = []

        def run_process() -> None:
            try:
                self.controller.run(
                    process_tree_command(),
                    cwd=Path.cwd(),
                    env=os.environ.copy(),
                    phase="cancel-test",
                    timeout_seconds=30,
                )
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_process)
        worker.start()
        self.assertTrue(wait_until(lambda: any("CHILD_PID=" in line for line in self.logs)))
        parent_pid = logged_pid(self.logs, "PID phase=cancel-test pid=")
        child_pid = logged_pid(self.logs, "CHILD_PID=")
        self.controller.request_cancel("controlled test cancellation")
        worker.join(timeout=10)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], app_module.TaskCancelled)
        self.assertTrue(wait_until(lambda: not pid_is_running(parent_pid)))
        self.assertTrue(wait_until(lambda: not pid_is_running(child_pid)))
        joined = "\n".join(self.logs)
        self.assertIn("cancel requested reason=controlled test cancellation", joined)
        self.assertIn("STOP phase=cancel-test", joined)


class GrainAppLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(self.temp_dir.name)
        self.original_paths = {
            name: getattr(app_module, name)
            for name in (
                "MODEL_CACHE",
                "CELLPOSE_CACHE",
                "DATASET_DIR",
                "RESULT_DIR",
                "ANALYSIS_DIR",
                "SHAPE_DIR",
            )
        }
        app_module.MODEL_CACHE = temp_root / "models"
        app_module.CELLPOSE_CACHE = temp_root / "cache"
        app_module.DATASET_DIR = temp_root / "dataset"
        app_module.RESULT_DIR = temp_root / "results"
        app_module.ANALYSIS_DIR = temp_root / "analysis"
        app_module.SHAPE_DIR = temp_root / "shapes"
        self.errors: list[tuple[str, str]] = []
        self.original_showerror = app_module.messagebox.showerror
        app_module.messagebox.showerror = (
            lambda title, message: self.errors.append((title, str(message)))
        )

    def tearDown(self) -> None:
        app_module.messagebox.showerror = self.original_showerror
        for name, value in self.original_paths.items():
            setattr(app_module, name, value)
        self.temp_dir.cleanup()

    def make_app(self) -> app_module.GrainApp:
        app = app_module.GrainApp()
        app.withdraw()
        app.input_path = Path(self.temp_dir.name) / "selected.tif"
        app.detect_btn.config(state=app_module.tk.NORMAL)
        app.scale_btn.config(state=app_module.tk.NORMAL)
        app.manual_scale_btn.config(state=app_module.tk.NORMAL)
        app.edit_btn.config(state=app_module.tk.NORMAL)
        app.export_btn.config(state=app_module.tk.NORMAL)
        return app

    def pump_until_finished(self, app: app_module.GrainApp, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.update()
            if app._task_thread is None:
                return
            time.sleep(0.02)
        self.fail("GUI background task did not finish")

    def test_timeout_restores_all_buttons(self) -> None:
        app = self.make_app()
        try:
            app.run_background(
                "controlled GUI timeout",
                lambda: app.run_command(
                    process_tree_command(),
                    phase="gui-timeout-test",
                    timeout_seconds=0.8,
                ),
            )
            self.pump_until_finished(app)

            self.assertEqual(str(app.detect_btn.cget("state")), app_module.tk.NORMAL)
            self.assertEqual(str(app.scale_btn.cget("state")), app_module.tk.NORMAL)
            self.assertEqual(str(app.manual_scale_btn.cget("state")), app_module.tk.NORMAL)
            self.assertEqual(str(app.edit_btn.cget("state")), app_module.tk.NORMAL)
            self.assertEqual(str(app.export_btn.cget("state")), app_module.tk.NORMAL)
            self.assertEqual(str(app.cancel_btn.cget("state")), app_module.tk.DISABLED)
            self.assertEqual(app.status.get(), "任务超时")
            self.assertEqual(len(self.errors), 1)
        finally:
            app.on_close()

    def test_window_close_terminates_running_process_tree(self) -> None:
        app = self.make_app()
        app.run_background(
            "controlled close",
            lambda: app.run_command(
                process_tree_command(),
                phase="close-test",
                timeout_seconds=30,
            ),
        )
        self.assertTrue(
            wait_until(
                lambda: self._pump_for_child_log(app),
                timeout=5,
            )
        )
        log_text = app.log.get("1.0", app_module.tk.END)
        parent_pid = int(
            next(line for line in log_text.splitlines() if "PID phase=close-test pid=" in line).rsplit("=", 1)[-1]
        )
        child_pid = int(
            next(line for line in log_text.splitlines() if "CHILD_PID=" in line).rsplit("=", 1)[-1]
        )
        worker = app._task_thread
        app.on_close()
        if worker is not None:
            worker.join(timeout=10)

        self.assertIsNotNone(worker)
        self.assertFalse(worker.is_alive())
        self.assertTrue(wait_until(lambda: not pid_is_running(parent_pid)))
        self.assertTrue(wait_until(lambda: not pid_is_running(child_pid)))

    def test_circle_export_passes_millimeter_scale(self) -> None:
        app = self.make_app()
        commands: list[list[str]] = []
        app.sample = "sample"
        app.um_per_px = 0.25

        def capture_command(command: list[str], **_kwargs) -> None:
            commands.append(command)

        app.run_command = capture_command  # type: ignore[method-assign]
        app.run_background = (  # type: ignore[method-assign]
            lambda _label, task, _on_success=None: task()
        )
        try:
            app.export_circles()
            self.assertEqual(len(commands), 1)
            scale_index = commands[0].index("--dxf-scale")
            self.assertAlmostEqual(float(commands[0][scale_index + 1]), 0.00025)
        finally:
            app.on_close()

    @staticmethod
    def _pump_for_child_log(app: app_module.GrainApp) -> bool:
        app.update()
        return "CHILD_PID=" in app.log.get("1.0", app_module.tk.END)


if __name__ == "__main__":
    unittest.main(verbosity=2)
