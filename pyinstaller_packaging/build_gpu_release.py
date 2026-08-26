"""Build the CUDA PyInstaller release with automatic CPU fallback."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

from accept_sem_grain_build import (
    file_info,
    finish_process_validation,
    process_step,
    run_acceptance,
    skipped_step,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GPU_ENV = Path(r"E:\conda_envs\sem_sam_env")
SPEC_PATH = PROJECT_ROOT / "pyinstaller_packaging" / "SEMGrainToolGPU.spec"
APP_NAME = "SEMGrainToolGPU"
DEFAULT_BUILD_DIR = PROJECT_ROOT / "dist" / APP_NAME
VERSION_MANIFEST = PROJECT_ROOT / "dist" / f"{APP_NAME}_versions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-env", type=Path, default=DEFAULT_GPU_ENV)
    parser.add_argument("--build-timeout", type=int, default=5400)
    parser.add_argument("--env-check-timeout", type=int, default=180)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--record", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gpu_python = (args.gpu_env / "python.exe").resolve()
    if not gpu_python.is_file():
        raise SystemExit(f"CUDA Python environment is missing: {gpu_python}")

    record_path = args.record
    if record_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        record_path = PROJECT_ROOT / "dist" / f"{APP_NAME}_acceptance_{stamp}-{uuid.uuid4().hex[:8]}.json"

    check_command = [
        str(gpu_python),
        str(PROJECT_ROOT / "check_torch_env.py"),
        "--require-cuda",
        "--expected-python", str(gpu_python),
        "--output", str(VERSION_MANIFEST),
    ]
    env_step = process_step(
        "cuda_environment_gate", check_command, PROJECT_ROOT, args.env_check_timeout
    )
    env_step["outputs"].append(file_info(VERSION_MANIFEST))
    initial_steps = [env_step]

    if env_step["status"] == "passed":
        test_step = process_step(
            "web_application_tests",
            [str(gpu_python), str(PROJECT_ROOT / "pyinstaller_packaging" / "test_sem_grain_web_app.py")],
            PROJECT_ROOT,
            120,
        )
        initial_steps.append(test_step)
    else:
        initial_steps.append(skipped_step("web_application_tests", 120, "CUDA environment gate failed"))

    if all(step["status"] == "passed" for step in initial_steps):
        build_environment = os.environ.copy()
        build_environment["SEM_GRAIN_BUILD_ENV"] = str(args.gpu_env.resolve())
        build_step = process_step(
            "pyinstaller_cuda_build",
            [str(gpu_python), "-m", "PyInstaller", "--clean", "--noconfirm", str(SPEC_PATH)],
            PROJECT_ROOT,
            args.build_timeout,
            env=build_environment,
        )
        exe_path = args.build_dir.resolve() / f"{APP_NAME}.exe"
        build_step["outputs"].extend([file_info(exe_path), file_info(VERSION_MANIFEST)])
        build_step["checks"].append({
            "name": "executable_created", "passed": exe_path.is_file(), "detail": str(exe_path)
        })
        finish_process_validation(build_step)
        initial_steps.append(build_step)
    else:
        initial_steps.append(skipped_step("pyinstaller_cuda_build", args.build_timeout, "GPU precondition failed"))

    exe_path = args.build_dir.resolve() / f"{APP_NAME}.exe"
    if initial_steps[-1]["status"] == "passed":
        probe_step = process_step(
            "packaged_cuda_probe",
            [str(exe_path), "--helper", "device_probe.py", "--json"],
            args.build_dir.resolve(),
            120,
        )
        probe_payload = None
        for line in probe_step["stdout"].splitlines():
            if line.startswith("DEVICE_PROBE_JSON="):
                try:
                    probe_payload = json.loads(line.split("=", 1)[1])
                except json.JSONDecodeError:
                    pass
        probe_step["checks"].append({
            "name": "packaged_cuda_runtime_available",
            "passed": bool(probe_payload and probe_payload.get("cuda_available")),
            "detail": probe_payload,
        })
        finish_process_validation(probe_step)
        initial_steps.append(probe_step)
    else:
        initial_steps.append(skipped_step("packaged_cuda_probe", 120, "GPU build failed"))

    exit_code, final_record = run_acceptance(
        build_dir=args.build_dir,
        record_path=record_path,
        initial_steps=initial_steps,
        app_name=APP_NAME,
        expect_cuda=True,
    )
    print(f"acceptance_record={final_record}")
    print(f"release_exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
