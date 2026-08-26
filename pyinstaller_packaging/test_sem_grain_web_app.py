from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import tifffile
from PIL import Image


PACKAGING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGING_DIR.parent
if str(PACKAGING_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGING_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import device_probe
import sem_grain_web_app as web


class TemporaryApplication(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.stack = ExitStack()
        replacements = {
            "APP_ROOT": root,
            "RESOURCE_ROOT": root,
            "STATIC_ROOT": root / "web_static",
            "MODEL_CACHE": root / "model_cache",
            "CELLPOSE_CACHE": root / "cellpose_cache",
            "UPLOAD_DIR": root / "input_uploads",
            "DATASET_DIR": root / "cellpose_dataset",
            "RESULT_DIR": root / "cellpose_results_pretrained",
            "ANALYSIS_DIR": root / "particle_analysis",
            "SHAPE_DIR": root / "coordinate_shape_outputs",
        }
        for name, value in replacements.items():
            self.stack.enter_context(mock.patch.object(web, name, value))
        self.probe_mock = self.stack.enter_context(
            mock.patch.object(web.WebApplication, "_probe_device", return_value=None)
        )
        web.FILE_ROOTS = {
            "dataset": web.DATASET_DIR,
            "results": web.RESULT_DIR,
            "analysis": web.ANALYSIS_DIR,
            "shapes": web.SHAPE_DIR,
        }
        self.app = web.WebApplication()

    def tearDown(self) -> None:
        self.app.controller.shutdown("test cleanup")
        self.stack.close()
        self.temp.cleanup()

    @staticmethod
    def png_bytes(width: int = 40, height: int = 30) -> bytes:
        output = io.BytesIO()
        Image.fromarray(np.full((height, width), 120, dtype=np.uint8)).save(output, "PNG")
        return output.getvalue()

    def test_upload_creates_source_and_cellpose_input(self) -> None:
        self.app.upload("测试 image.png", self.png_bytes())
        snapshot = self.app.snapshot()

        self.assertTrue(snapshot["sample"])
        self.assertTrue(snapshot["source_url"].startswith("/files/dataset/"))
        self.assertTrue((web.DATASET_DIR / f"{self.app.sample}_source.png").is_file())
        self.assertTrue((web.DATASET_DIR / f"{self.app.sample}_sem.png").is_file())
        with Image.open(web.DATASET_DIR / f"{self.app.sample}_sem.png") as prepared:
            self.assertEqual(prepared.size, (40, 30))

    def test_startup_schedules_isolated_device_probe(self) -> None:
        deadline = time.monotonic() + 1
        while self.probe_mock.call_count == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(self.probe_mock.call_count, 1)
        self.assertEqual(self.app.device_info["status"], "checking")

    def test_device_probe_returns_consistent_backend(self) -> None:
        result = device_probe.probe_device()
        self.assertIn(result["backend"], {"cpu", "cuda"})
        self.assertEqual(result["gpu_available"], result["backend"] == "cuda")
        self.assertTrue(result["torch_version"])

    def test_gpu_failure_retries_same_detection_on_cpu(self) -> None:
        self.app.input_path = Path(self.temp.name) / "input.png"
        self.app.sample = "fallback_sample"
        self.app.um_per_px = 1.0
        self.app.gpu_available = True
        calls: list[tuple[str, list[str], str]] = []

        def run_helper(command: str, arguments: list[str], phase: str) -> None:
            calls.append((command, arguments, phase))
            if command == "cellpose" and "--use_gpu" in arguments:
                raise RuntimeError("simulated CUDA failure")

        self.app._run_helper = run_helper  # type: ignore[method-assign]
        self.app.start_detection(True, 100)
        deadline = time.monotonic() + 3
        while self.app.snapshot()["busy"] and time.monotonic() < deadline:
            time.sleep(0.01)

        cellpose_calls = [arguments for command, arguments, _ in calls if command == "cellpose"]
        self.assertEqual(len(cellpose_calls), 2)
        self.assertIn("--use_gpu", cellpose_calls[0])
        self.assertNotIn("--use_gpu", cellpose_calls[1])
        self.assertFalse(self.app.gpu_available)
        self.assertEqual(self.app.device_info["status"], "fallback")
        self.assertIn("已使用CPU完成识别", self.app.status)

    def test_manual_scale_uses_original_image_coordinates(self) -> None:
        self.app.upload("scale.png", self.png_bytes())
        self.app.start_manual_scale(100, [[5, 10], [25, 10]])
        deadline = time.monotonic() + 3
        while self.app.snapshot()["busy"] and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertFalse(self.app.snapshot()["busy"])
        self.assertAlmostEqual(self.app.um_per_px or 0, 5.0)
        self.assertTrue((web.ANALYSIS_DIR / f"{self.app.sample}_scale_bar.json").is_file())

    def test_mask_selection_merge_and_undo(self) -> None:
        self.app.upload("mask.png", self.png_bytes())
        labels = np.zeros((30, 40), dtype=np.uint16)
        labels[2:12, 2:12] = 1
        labels[2:12, 18:28] = 2
        tifffile.imwrite(web.RESULT_DIR / f"{self.app.sample}_sem_cp_masks.tif", labels)

        self.app.start_editor()
        self.app.editor_select(4, 4, False)
        self.app.editor_select(20, 4, True)
        self.app.editor_apply("merge")
        self.assertEqual(set(np.unique(self.app.editor_labels)), {0, 1})
        self.app.editor_apply("undo")
        self.assertEqual(set(np.unique(self.app.editor_labels)), {0, 1, 2})

    def test_http_state_requires_token(self) -> None:
        web.STATIC_ROOT.mkdir(parents=True)
        (web.STATIC_ROOT / "index.html").write_text("token=__SEM_TOKEN__", encoding="utf-8")
        server = web.LocalServer(("127.0.0.1", 0), self.app)
        self.app.server = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/", timeout=2) as response:
                page = response.read().decode("utf-8")
            self.assertIn(self.app.token, page)

            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + "/api/state", timeout=2)
            self.assertEqual(caught.exception.code, 403)

            request = urllib.request.Request(
                base + "/api/state", headers={"X-SEM-Token": self.app.token}
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(payload["status"], "请选择一张电子显微镜图像")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
