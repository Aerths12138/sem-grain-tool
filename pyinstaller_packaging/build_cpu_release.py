"""Build the CPU PyInstaller release and immediately run product acceptance."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

from accept_sem_grain_build import (
    DEFAULT_BUILD_DIR,
    file_info,
    finish_process_validation,
    process_step,
    run_acceptance,
    skipped_step,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTABLE_PYTHON = PROJECT_ROOT / "SEM_Grain_Tool_Portable" / "env" / "python.exe"
SPEC_PATH = PROJECT_ROOT / "pyinstaller_packaging" / "SEMGrainTool.spec"
VERSION_MANIFEST = PROJECT_ROOT / "dist" / "SEMGrainTool_versions.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-timeout", type=int, default=3600)
    parser.add_argument("--env-check-timeout", type=int, default=120)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--record", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record_path = args.record
    if record_path is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        record_path = PROJECT_ROOT / "dist" / f"SEMGrainTool_acceptance_{stamp}-{uuid.uuid4().hex[:8]}.json"

    check_command = [
        sys.executable, str(PROJECT_ROOT / "check_torch_env.py"), "--require-cpu",
        "--expected-python", str(PORTABLE_PYTHON), "--output", str(VERSION_MANIFEST),
    ]
    env_step = process_step("cpu_environment_gate", check_command, PROJECT_ROOT, args.env_check_timeout)
    env_step["outputs"].append(file_info(VERSION_MANIFEST))
    initial_steps = [env_step]

    if env_step["status"] == "passed":
        web_test_command = [
            sys.executable,
            str(PROJECT_ROOT / "pyinstaller_packaging" / "test_sem_grain_web_app.py"),
        ]
        web_test_step = process_step(
            "web_application_tests", web_test_command, PROJECT_ROOT, 120
        )
        initial_steps.append(web_test_step)
    else:
        initial_steps.append(
            skipped_step("web_application_tests", 120, "CPU environment gate failed")
        )

    if all(step["status"] == "passed" for step in initial_steps):
        build_command = [
            sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", str(SPEC_PATH),
        ]
        build_step = process_step("pyinstaller_cpu_build", build_command, PROJECT_ROOT, args.build_timeout)
        exe_path = args.build_dir.resolve() / "SEMGrainTool.exe"
        build_step["outputs"].extend([file_info(exe_path), file_info(VERSION_MANIFEST)])
        build_step["checks"].append({"name": "executable_created", "passed": exe_path.is_file(), "detail": str(exe_path)})
        finish_process_validation(build_step)
        initial_steps.append(build_step)
    else:
        initial_steps.append(
            skipped_step(
                "pyinstaller_cpu_build",
                args.build_timeout,
                "environment gate or web application tests failed",
            )
        )

    exit_code, final_record = run_acceptance(
        build_dir=args.build_dir, record_path=record_path, initial_steps=initial_steps,
    )
    print(f"acceptance_record={final_record}")
    print(f"release_exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
