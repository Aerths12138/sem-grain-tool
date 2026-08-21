"""Run existing focused and packaged validation routes and record their evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGING_DIR = Path(__file__).resolve().parent
ROUTE_FILES = {
    "lifecycle": [
        PACKAGING_DIR / "sem_grain_app_frozen.py",
        PACKAGING_DIR / "test_sem_grain_app_frozen.py",
        PACKAGING_DIR / "run_final_validation.py",
    ],
    "packaging": [
        PROJECT_ROOT / "check_torch_env.py",
        PACKAGING_DIR / "README.md",
        PACKAGING_DIR / "SEMGrainTool.spec",
        PACKAGING_DIR / "SEMGrainToolConsole.spec",
        PACKAGING_DIR / "spec_common.py",
        PACKAGING_DIR / "accept_sem_grain_build.py",
        PACKAGING_DIR / "build_cpu_release.py",
        PACKAGING_DIR / "run_final_validation.py",
    ],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        item: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
        if path.is_file():
            stat = path.stat()
            item.update(
                {
                    "modified_at": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).astimezone().isoformat(timespec="milliseconds"),
                    "modified_at_ns": stat.st_mtime_ns,
                    "size_bytes": stat.st_size,
                    "sha256": sha256(path),
                }
            )
        result.append(item)
    return result


def terminate_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def run_command(
    argv: list[str],
    timeout_seconds: int,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started_ns = time.time_ns()
    started_monotonic = time.monotonic()
    env = os.environ.copy()
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
    if extra_env:
        env.update(extra_env)
    process = subprocess.Popen(
        argv,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    timed_out = False
    stop_reason = None
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        stop_reason = f"timeout after {timeout_seconds} seconds; process tree terminated"
        terminate_tree(process)
        stdout, _ = process.communicate()
    finished_ns = time.time_ns()
    return {
        "argv": argv,
        "cwd": str(PROJECT_ROOT),
        "pid": process.pid,
        "started_at": datetime.fromtimestamp(
            started_ns / 1_000_000_000, timezone.utc
        ).astimezone().isoformat(timespec="milliseconds"),
        "finished_at": datetime.fromtimestamp(
            finished_ns / 1_000_000_000, timezone.utc
        ).astimezone().isoformat(timespec="milliseconds"),
        "started_at_ns": started_ns,
        "finished_at_ns": finished_ns,
        "duration_seconds": round(time.monotonic() - started_monotonic, 3),
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "exit_code": 124 if timed_out else process.returncode,
        "process_exit_code": process.returncode,
        "stop_reason": stop_reason,
        "stdout": stdout,
    }


def same_snapshot(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> bool:
    return [
        (item["path"], item["exists"], item.get("sha256")) for item in before
    ] == [
        (item["path"], item["exists"], item.get("sha256")) for item in after
    ]


def route_command(route: str) -> tuple[list[str], int]:
    if route == "lifecycle":
        return (
            [sys.executable, str(PACKAGING_DIR / "test_sem_grain_app_frozen.py")],
            180,
        )
    return (
        [sys.executable, str(PACKAGING_DIR / "accept_sem_grain_build.py")],
        1200,
    )


def parse_acceptance_artifact(stdout: str) -> dict[str, Any] | None:
    match = re.search(r"^acceptance_record=(.+)$", stdout, flags=re.MULTILINE)
    if match is None:
        return None
    path = Path(match.group(1).strip())
    return {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "sha256": sha256(path) if path.is_file() else None,
    }


def validate_route(route: str) -> dict[str, Any]:
    relevant_files = ROUTE_FILES[route]
    before = snapshot(relevant_files)
    argv, timeout = route_command(route)
    extra_env = None
    if route == "lifecycle":
        environment_root = Path(sys.executable).resolve().parent
        extra_env = {
            "TCL_LIBRARY": str(environment_root / "Library" / "lib" / "tcl8.6"),
            "TK_LIBRARY": str(environment_root / "Library" / "lib" / "tk8.6"),
        }
    execution = run_command(argv, timeout, extra_env)
    after = snapshot(relevant_files)
    latest_before_ns = max(
        (item.get("modified_at_ns", 0) for item in before), default=0
    )
    unchanged = same_snapshot(before, after)
    result: dict[str, Any] = {
        "route": route,
        "covers": (
            "ProcessController output, timeout/cancel process-tree termination, button "
            "restoration, and window-close cleanup"
            if route == "lifecycle"
            else "packaged CPU resources plus scale JSON, particle CSV, circle preview, and DXF"
        ),
        "relevant_files_before": before,
        "execution": execution,
        "relevant_files_after": after,
        "files_unchanged_during_validation": unchanged,
        "started_after_latest_relevant_modification": (
            execution["started_at_ns"] >= latest_before_ns
        ),
        "acceptance_artifact": (
            parse_acceptance_artifact(execution["stdout"])
            if route == "packaging"
            else None
        ),
    }
    artifact_ok = (
        route != "packaging"
        or bool(result["acceptance_artifact"])
        and result["acceptance_artifact"]["exists"]
    )
    result["status"] = (
        "passed"
        if execution["exit_code"] == 0
        and unchanged
        and result["started_after_latest_relevant_modification"]
        and artifact_ok
        else "failed"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--route",
        choices=("lifecycle", "packaging", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    routes = ["lifecycle", "packaging"] if args.route == "all" else [args.route]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (
        args.output
        or PROJECT_ROOT
        / "dist"
        / f"SEMGrainTool_final_validation_{stamp}-{uuid.uuid4().hex[:8]}.json"
    ).resolve()
    record: dict[str, Any] = {
        "schema_version": 1,
        "created_at": now_iso(),
        "python_executable": str(Path(sys.executable).resolve()),
        "record_path": str(output),
        "requested_route": args.route,
        "results": [],
        "status": "running",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for route in routes:
        record["results"].append(validate_route(route))
        output.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    record["status"] = (
        "passed"
        if all(result["status"] == "passed" for result in record["results"])
        else "failed"
    )
    record["finished_at"] = now_iso()
    record["validation_artifact"] = str(output)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"final_validation_record={output}")
    print(f"final_validation_status={record['status']}")
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
