"""Validate a packaged SEM Grain Tool release through its helper interface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUILD_DIR = PROJECT_ROOT / "dist" / "SEMGrainTool"
RESOURCE_PATHS = {
    "torch_cpu": Path("_internal/torch/lib/torch_cpu.dll"),
    "cpsam": Path("_internal/cellpose_cache/models/cpsam"),
    "tcl_init": Path("_internal/_tcl_data/tcl8.6/init.tcl"),
    "tk_init": Path("_internal/_tcl_data/tk8.6/tk.tcl"),
    "tcl_dll": Path("_internal/tcl86t.dll"),
    "tk_dll": Path("_internal/tk86t.dll"),
    "tkinter_pyd": Path("_internal/_tkinter.pyd"),
    "web_index": Path("_internal/web_static/index.html"),
    "device_probe": Path("_internal/device_probe.py"),
}
STEP_TIMEOUTS = {
    "resource_manifest": 120,
    "detect_scale": 120,
    "analyze_particles": 300,
    "generate_circles": 600,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path, with_hash: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        result.update({"size_bytes": stat.st_size, "modified_at": stat.st_mtime})
        if with_hash:
            result["sha256"] = sha256(path)
    return result


def run_text(command: list[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(
        command, cwd=cwd, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    return completed.returncode, completed.stdout.strip()


def git_metadata(project_root: Path) -> dict[str, Any]:
    head_code, head = run_text(["git", "rev-parse", "HEAD"], project_root)
    status_code, status = run_text(["git", "status", "--short"], project_root)
    diff_code, diff = run_text(["git", "diff", "--binary", "HEAD"], project_root)
    return {
        "head": head if head_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status": status.splitlines() if status else [],
        "working_tree_fingerprint": hashlib.sha256(diff.encode("utf-8")).hexdigest()
        if diff_code == 0 else None,
    }


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    else:
        process.kill()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def process_step(
    name: str,
    command: list[str],
    cwd: Path,
    timeout: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    step: dict[str, Any] = {
        "name": name, "status": "running", "started_at": now_iso(),
        "timeout_seconds": timeout, "command": command, "cwd": str(cwd.resolve()),
        "pid": None, "process_exit_code": None, "exit_code": None,
        "stop_reason": None, "stdout": "", "outputs": [], "checks": [],
    }
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command, cwd=cwd, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            env=env,
        )
        step["pid"] = process.pid
        try:
            stdout, _ = process.communicate(timeout=timeout)
            step["stdout"] = stdout
            step["process_exit_code"] = process.returncode
            step["exit_code"] = process.returncode
            step["status"] = "passed" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                step["stop_reason"] = f"process exited with code {process.returncode}"
        except subprocess.TimeoutExpired as exc:
            terminate_process_tree(process)
            remainder, _ = process.communicate()
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode("utf-8", errors="replace")
            step["stdout"] = partial + remainder
            step["process_exit_code"] = process.returncode
            step["exit_code"] = 124
            step["status"] = "failed"
            step["stop_reason"] = f"timeout after {timeout} seconds; process tree terminated"
    except Exception as exc:
        if process is not None:
            terminate_process_tree(process)
        step["exit_code"] = 1
        step["status"] = "failed"
        step["stop_reason"] = f"runner error: {type(exc).__name__}: {exc}"
    finally:
        step["finished_at"] = now_iso()
        step["duration_seconds"] = round(time.monotonic() - started, 3)
    return step


def internal_step(name: str, timeout: int) -> dict[str, Any]:
    return {
        "name": name, "status": "running", "started_at": now_iso(),
        "timeout_seconds": timeout, "command": None, "cwd": str(PROJECT_ROOT),
        "pid": os.getpid(), "process_exit_code": 0, "exit_code": 0,
        "stop_reason": None, "stdout": "", "outputs": [], "checks": [],
    }


def add_check(step: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    step["checks"].append({"name": name, "passed": bool(passed), "detail": detail})


def finish_internal(step: dict[str, Any], started: float) -> None:
    failed = time.monotonic() - started > step["timeout_seconds"] or not all(
        item["passed"] for item in step["checks"]
    )
    if failed:
        step["status"] = "failed"
        step["exit_code"] = 124 if time.monotonic() - started > step["timeout_seconds"] else 1
        step["stop_reason"] = (
            f"timeout after {step['timeout_seconds']} seconds"
            if step["exit_code"] == 124 else "one or more validation checks failed"
        )
    else:
        step["status"] = "passed"
    step["finished_at"] = now_iso()
    step["duration_seconds"] = round(time.monotonic() - started, 3)


def finish_process_validation(step: dict[str, Any]) -> None:
    if step["status"] != "failed" and not all(item["passed"] for item in step["checks"]):
        step["status"] = "failed"
        step["exit_code"] = 1
        step["stop_reason"] = "one or more validation checks failed"


class AcceptanceRecord:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def append(self, step: dict[str, Any]) -> None:
        self.data["steps"].append(step)
        self.write()


def skipped_step(name: str, timeout: int, reason: str) -> dict[str, Any]:
    timestamp = now_iso()
    return {
        "name": name, "status": "skipped", "started_at": timestamp,
        "finished_at": timestamp, "duration_seconds": 0.0, "timeout_seconds": timeout,
        "command": None, "cwd": str(PROJECT_ROOT), "pid": None,
        "process_exit_code": None, "exit_code": None, "stop_reason": reason,
        "stdout": "", "outputs": [], "checks": [],
    }


def validate_resources(
    build_dir: Path,
    version_manifest: Path,
    fixtures: dict[str, Path],
    *,
    app_name: str = "SEMGrainTool",
    expect_cuda: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    step = internal_step("packaged_resource_manifest", STEP_TIMEOUTS["resource_manifest"])
    version_data: dict[str, Any] = {}
    if version_manifest.is_file():
        try:
            version_data = json.loads(version_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            step["stdout"] = f"Version manifest read failed: {exc}"
    exe_path = build_dir / f"{app_name}.exe"
    step["outputs"].append(file_info(exe_path))
    add_check(step, "executable_exists", exe_path.is_file(), str(exe_path.resolve()))
    step["outputs"].append(file_info(version_manifest))
    add_check(step, "version_manifest_exists", version_manifest.is_file(), str(version_manifest.resolve()))
    if expect_cuda:
        add_check(step, "torch_has_cuda_runtime", bool(version_data.get("torch_cuda")), version_data.get("torch_cuda"))
        add_check(step, "cuda_available_on_build_host", version_data.get("cuda_available") is True, version_data.get("cuda_available"))
    else:
        add_check(step, "torch_is_cpu", version_data.get("torch_cuda") is None, version_data.get("torch_cuda"))
        add_check(step, "cuda_unavailable", version_data.get("cuda_available") is False, version_data.get("cuda_available"))
    add_check(step, "torch_version_present", bool(version_data.get("torch_version")), version_data.get("torch_version"))
    add_check(step, "cellpose_version_present", bool(version_data.get("cellpose_version")), version_data.get("cellpose_version"))

    for resource_name, relative_path in RESOURCE_PATHS.items():
        path = build_dir / relative_path
        info = file_info(path, with_hash=resource_name == "cpsam")
        step["outputs"].append(info)
        add_check(step, f"resource_{resource_name}", path.is_file(), info)
        if resource_name == "cpsam":
            add_check(step, "cpsam_size_plausible", path.is_file() and path.stat().st_size >= 1_000_000_000, info.get("size_bytes"))

    torch_lib = build_dir / "_internal" / "torch" / "lib"
    cuda_markers = ("torch_cuda", "c10_cuda", "cudnn", "cublas", "cusparse", "cufft")
    cuda_files = [str(path.resolve()) for path in torch_lib.glob("*.dll") if any(marker in path.name.lower() for marker in cuda_markers)]
    add_check(
        step,
        "cuda_runtime_dll_policy",
        bool(cuda_files) if expect_cuda else not cuda_files,
        cuda_files,
    )
    for fixture_name, path in fixtures.items():
        add_check(step, f"fixture_{fixture_name}", path.is_file(), file_info(path, with_hash=False))
    finish_internal(step, started)
    return step


def validate_scale_json(path: Path, step: dict[str, Any]) -> float | None:
    step["outputs"].append(file_info(path))
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            step["stdout"] += f"\nScale JSON read failed: {exc}"
    add_check(step, "scale_json_exists", path.is_file(), str(path.resolve()))
    for key in ("bar_pixel_length", "scale_um", "um_per_px"):
        value = data.get(key)
        add_check(step, f"scale_{key}_positive", isinstance(value, (int, float)) and value > 0, value)
    return float(data["um_per_px"]) if isinstance(data.get("um_per_px"), (int, float)) and data["um_per_px"] > 0 else None


def validate_particle_csv(path: Path, step: dict[str, Any]) -> int:
    step["outputs"].append(file_info(path))
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                rows = list(reader)
        except (OSError, csv.Error) as exc:
            step["stdout"] += f"\nParticle CSV read failed: {exc}"
    required = {"particle_id", "center_x_px", "center_y_px", "area_px", "equivalent_diameter_px"}
    add_check(step, "particle_csv_exists", path.is_file(), str(path.resolve()))
    add_check(step, "particle_csv_columns", required.issubset(fieldnames), fieldnames)
    add_check(step, "particle_count_positive", len(rows) > 0, len(rows))
    numeric_ok = bool(rows)
    try:
        numeric_ok = numeric_ok and all(float(row["area_px"]) > 0 and float(row["equivalent_diameter_px"]) > 0 for row in rows)
    except (KeyError, TypeError, ValueError):
        numeric_ok = False
    add_check(step, "particle_sizes_positive", numeric_ok, len(rows))
    return len(rows)


def validate_png(path: Path, step: dict[str, Any]) -> None:
    step["outputs"].append(file_info(path))
    dimensions: tuple[int, int] | None = None
    extrema: Any = None
    try:
        from PIL import Image
        with Image.open(path) as image:
            dimensions = image.size
            extrema = image.convert("L").getextrema()
    except Exception as exc:
        step["stdout"] += f"\nPreview validation failed: {exc}"
    add_check(step, "circle_preview_exists", path.is_file(), str(path.resolve()))
    add_check(step, "circle_preview_nonblank", bool(dimensions and extrema and extrema[0] != extrema[1]), {"dimensions": dimensions, "extrema": extrema})


def validate_dxf(
    path: Path,
    step: dict[str, Any],
    expected_count: int,
) -> None:
    step["outputs"].append(file_info(path))
    entity_types: list[str] = []
    units: int | None = None
    radii: list[float] = []
    try:
        import ezdxf
        document = ezdxf.readfile(path)
        units = int(document.units)
        entities = list(document.modelspace())
        entity_types = [entity.dxftype() for entity in entities]
        radii = [float(entity.dxf.radius) for entity in entities if entity.dxftype() == "CIRCLE"]
    except Exception as exc:
        step["stdout"] += f"\nDXF validation failed: {exc}"
    add_check(step, "dxf_exists", path.is_file(), str(path.resolve()))
    add_check(step, "dxf_has_entities", bool(entity_types), len(entity_types))
    add_check(step, "dxf_only_circles", bool(entity_types) and set(entity_types) == {"CIRCLE"}, sorted(set(entity_types)))
    add_check(step, "dxf_circle_count_matches_particles", len(entity_types) == expected_count, {"circles": len(entity_types), "particles": expected_count})
    add_check(step, "dxf_units_are_millimeters", units == 4, units)
    add_check(step, "dxf_circle_radii_positive", bool(radii) and all(math.isfinite(radius) and radius > 0 for radius in radii), {"count": len(radii), "minimum": min(radii) if radii else None})


def validate_dxf_metadata(
    path: Path,
    step: dict[str, Any],
    expected_scale: float,
    expected_count: int,
) -> None:
    step["outputs"].append(file_info(path))
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            step["stdout"] += f"\nDXF metadata validation failed: {exc}"
    scale = data.get("dxf_scale_mm_per_pixel")
    width = data.get("geometry_width_mm")
    height = data.get("geometry_height_mm")
    add_check(step, "dxf_metadata_exists", path.is_file(), str(path.resolve()))
    add_check(step, "dxf_metadata_unit_mm", data.get("dxf_unit") == "mm", data.get("dxf_unit"))
    add_check(
        step,
        "dxf_metadata_scale_matches_calibration",
        isinstance(scale, (int, float)) and math.isclose(float(scale), expected_scale, rel_tol=1e-9, abs_tol=1e-15),
        {"actual": scale, "expected": expected_scale},
    )
    add_check(step, "dxf_metadata_shape_count", data.get("shape_count") == expected_count, {"actual": data.get("shape_count"), "expected": expected_count})
    add_check(step, "dxf_metadata_extent_positive", isinstance(width, (int, float)) and width > 0 and isinstance(height, (int, float)) and height > 0, {"width_mm": width, "height_mm": height})


def run_acceptance(
    build_dir: Path = DEFAULT_BUILD_DIR,
    record_path: Path | None = None,
    initial_steps: list[dict[str, Any]] | None = None,
    fault_delete_resource: str | None = None,
    app_name: str = "SEMGrainTool",
    expect_cuda: bool = False,
) -> tuple[int, Path]:
    build_dir = build_dir.resolve()
    exe_path = build_dir / f"{app_name}.exe"
    version_manifest = build_dir.parent / f"{app_name}_versions.json"
    acceptance_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    record_path = (record_path or build_dir.parent / f"SEMGrainTool_acceptance_{acceptance_id}.json").resolve()
    fixtures = {
        "scale_image": PROJECT_ROOT / "1_i311.tif",
        "particle_mask": PROJECT_ROOT / "cellpose_results_pretrained" / "od_metaldam_04_sem_cp_masks.tif",
        "particle_image": PROJECT_ROOT / "cellpose_dataset" / "od_metaldam_04_sem.png",
    }
    data: dict[str, Any] = {
        "schema_version": 1, "acceptance_id": acceptance_id, "status": "running",
        "started_at": now_iso(), "finished_at": None,
        "project_root": str(PROJECT_ROOT), "git": git_metadata(PROJECT_ROOT),
        "build": {
            "directory": str(build_dir), "executable": file_info(exe_path),
            "version_manifest": file_info(version_manifest),
        },
        "fixtures": {name: file_info(path) for name, path in fixtures.items()},
        "fault_injection": None, "steps": list(initial_steps or []),
    }
    record = AcceptanceRecord(record_path, data)
    if any(step.get("status") != "passed" for step in data["steps"]):
        reason = "build precondition failed"
        record.append(skipped_step("packaged_resource_manifest", STEP_TIMEOUTS["resource_manifest"], reason))
        record.append(skipped_step("detect_scale_bar_helper", STEP_TIMEOUTS["detect_scale"], reason))
        record.append(skipped_step("analyze_particles_helper", STEP_TIMEOUTS["analyze_particles"], reason))
        record.append(skipped_step("generate_circle_preview_and_dxf_helper", STEP_TIMEOUTS["generate_circles"], reason))
        data["status"] = "failed"
        data["finished_at"] = now_iso()
        record.write()
        return 1, record_path
    backup_path: Path | None = None
    fault_target: Path | None = None
    try:
        if fault_delete_resource:
            fault_target = build_dir / RESOURCE_PATHS[fault_delete_resource]
            backup_path = fault_target.with_name(fault_target.name + ".acceptance-backup")
            if backup_path.exists() and not fault_target.exists():
                os.replace(backup_path, fault_target)
            if backup_path.exists():
                raise RuntimeError(f"stale fault backup already exists: {backup_path}")
            data["fault_injection"] = {
                "resource": fault_delete_resource, "target": str(fault_target),
                "backup": str(backup_path), "deleted": False, "restored": False,
            }
            if not fault_target.is_file():
                raise FileNotFoundError(f"fault target does not exist: {fault_target}")
            os.replace(fault_target, backup_path)
            data["fault_injection"]["deleted"] = True
            record.write()

        manifest_step = validate_resources(
            build_dir,
            version_manifest,
            fixtures,
            app_name=app_name,
            expect_cuda=expect_cuda,
        )
        record.append(manifest_step)
        if manifest_step["status"] != "passed":
            reason = "packaged resource manifest failed"
            record.append(skipped_step("detect_scale_bar_helper", STEP_TIMEOUTS["detect_scale"], reason))
            record.append(skipped_step("analyze_particles_helper", STEP_TIMEOUTS["analyze_particles"], reason))
            record.append(skipped_step("generate_circle_preview_and_dxf_helper", STEP_TIMEOUTS["generate_circles"], reason))
            return 1, record_path

        run_dir = build_dir / "acceptance_outputs" / acceptance_id
        scale_dir = run_dir / "scale"
        shape_dir = run_dir / "shapes"
        scale_dir.mkdir(parents=True, exist_ok=True)
        shape_dir.mkdir(parents=True, exist_ok=True)
        sample = "accept_" + acceptance_id.replace("-", "_")

        scale_json = scale_dir / f"{sample}_scale_bar.json"
        scale_command = [
            str(exe_path), "--helper", "detect_scale_bar.py", "--image", str(fixtures["scale_image"]),
            "--sample", sample, "--scale-um", "100", "--output-dir", str(scale_dir),
        ]
        scale_step = process_step("detect_scale_bar_helper", scale_command, build_dir, STEP_TIMEOUTS["detect_scale"])
        um_per_px = validate_scale_json(scale_json, scale_step)
        finish_process_validation(scale_step)
        record.append(scale_step)
        if scale_step["status"] != "passed" or um_per_px is None:
            reason = "scale helper or JSON validation failed"
            record.append(skipped_step("analyze_particles_helper", STEP_TIMEOUTS["analyze_particles"], reason))
            record.append(skipped_step("generate_circle_preview_and_dxf_helper", STEP_TIMEOUTS["generate_circles"], reason))
            return 1, record_path

        particle_csv = build_dir / "particle_analysis" / f"{sample}_particles.csv"
        particle_command = [
            str(exe_path), "--helper", "analyze_particles.py", "--sample", sample,
            "--mask", str(fixtures["particle_mask"]), "--image", str(fixtures["particle_image"]),
            "--um-per-px", format(um_per_px, ".12g"),
        ]
        particle_step = process_step("analyze_particles_helper", particle_command, build_dir, STEP_TIMEOUTS["analyze_particles"])
        particle_count = validate_particle_csv(particle_csv, particle_step)
        finish_process_validation(particle_step)
        record.append(particle_step)
        if particle_step["status"] != "passed":
            record.append(skipped_step("generate_circle_preview_and_dxf_helper", STEP_TIMEOUTS["generate_circles"], "particle helper or CSV validation failed"))
            return 1, record_path

        prefix = shape_dir / f"{sample}_circle_nonoverlap_a0.95_gap1_area_from_centers"
        preview_path = prefix.with_name(prefix.name + "_filled.png")
        dxf_path = prefix.with_name(prefix.name + ".dxf")
        dxf_metadata_path = prefix.with_name(prefix.name + "_dxf_metadata.json")
        dxf_scale = um_per_px / 1000.0
        shape_command = [
            str(exe_path), "--helper", "generate_shapes_from_particles.py", "--sample", sample,
            "--particles-csv", str(particle_csv), "--image", str(fixtures["particle_image"]),
            "--output-dir", str(shape_dir), "--mode", "circle", "--area-scale", "0.95",
            "--circle-gap", "1", "--dxf-scale", format(dxf_scale, ".12g"),
            "--resolve-iterations", "120",
        ]
        shape_step = process_step("generate_circle_preview_and_dxf_helper", shape_command, build_dir, STEP_TIMEOUTS["generate_circles"])
        validate_png(preview_path, shape_step)
        validate_dxf(dxf_path, shape_step, particle_count)
        validate_dxf_metadata(dxf_metadata_path, shape_step, dxf_scale, particle_count)
        finish_process_validation(shape_step)
        record.append(shape_step)
        return (0 if shape_step["status"] == "passed" else 1), record_path
    except Exception as exc:
        failure = internal_step("acceptance_runner_error", 30)
        failure.update({
            "status": "failed", "exit_code": 1,
            "stop_reason": f"{type(exc).__name__}: {exc}",
            "finished_at": now_iso(), "duration_seconds": 0.0,
        })
        record.append(failure)
        return 1, record_path
    finally:
        if backup_path is not None and fault_target is not None and backup_path.exists():
            if fault_target.exists():
                fault_target.unlink()
            os.replace(backup_path, fault_target)
            if data["fault_injection"] is not None:
                data["fault_injection"]["restored"] = True
                data["fault_injection"]["restored_at"] = now_iso()
        failed = any(step["status"] == "failed" for step in data["steps"])
        incomplete = any(step["status"] == "skipped" for step in data["steps"])
        data["status"] = "failed" if failed or incomplete else "passed"
        data["finished_at"] = now_iso()
        data["build"]["executable_after_acceptance"] = file_info(exe_path)
        record.write()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--fault-delete-resource", choices=sorted(RESOURCE_PATHS))
    parser.add_argument("--app-name", default="SEMGrainTool")
    parser.add_argument("--expect-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exit_code, record_path = run_acceptance(
        build_dir=args.build_dir, record_path=args.record,
        fault_delete_resource=args.fault_delete_resource,
        app_name=args.app_name,
        expect_cuda=args.expect_cuda,
    )
    print(f"acceptance_record={record_path}")
    print(f"acceptance_exit_code={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
