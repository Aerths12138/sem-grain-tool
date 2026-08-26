import os
import site
import subprocess
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def prepare_build(
    spec_path: str,
    dist_path: str,
    app_name: str,
    *,
    environment_root: Path | None = None,
    runtime: str = "cpu",
) -> dict:
    spec_dir = Path(spec_path).resolve()
    project_root = spec_dir.parent
    if runtime not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported runtime: {runtime}")
    environment_root = (
        environment_root.resolve()
        if environment_root is not None
        else project_root / "SEM_Grain_Tool_Portable" / "env"
    )
    expected_python = (environment_root / "python.exe").resolve()
    current_python = Path(sys.executable).resolve()

    user_site = Path(site.getusersitepackages()).resolve()
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry or not _is_inside(Path(entry).resolve(), user_site)
    ]

    manifest_path = Path(dist_path).resolve() / f"{app_name}_versions.json"
    check_environment = os.environ.copy()
    check_environment.pop("PYTHONPATH", None)
    check_environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        [
            str(current_python),
            str(project_root / "check_torch_env.py"),
            "--require-cuda" if runtime == "cuda" else "--require-cpu",
            "--expected-python",
            str(expected_python),
            "--output",
            str(manifest_path),
        ],
        check=True,
        env=check_environment,
    )

    datas = [
        (str(project_root / "pyinstaller_packaging" / "web_static"), "web_static"),
        (str(project_root / "make_cellpose_overlay.py"), "."),
        (str(project_root / "analyze_particles.py"), "."),
        (str(project_root / "detect_scale_bar.py"), "."),
        (str(project_root / "generate_shapes_from_particles.py"), "."),
        (str(project_root / "generate_sintered_agglomerates.py"), "."),
        (str(project_root / "device_probe.py"), "."),
        (str(project_root / "cellpose_cache" / "models" / "cpsam"), "cellpose_cache/models"),
        (str(environment_root / "Lib" / "tkinter"), "tkinter"),
        (str(environment_root / "Library" / "lib" / "tcl8.6"), "_tcl_data/tcl8.6"),
        (str(environment_root / "Library" / "lib" / "tk8.6"), "_tcl_data/tk8.6"),
    ]
    binaries = [
        (str(environment_root / "DLLs" / "_tkinter.pyd"), "."),
        (str(environment_root / "Library" / "bin" / "tcl86t.dll"), "."),
        (str(environment_root / "Library" / "bin" / "tk86t.dll"), "."),
    ]

    required_paths = [Path(source) for source, _ in datas + binaries]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise SystemExit(
            f"{runtime.upper()} build environment is incomplete:\n  "
            + "\n  ".join(str(path) for path in missing_paths)
        )

    hiddenimports = [
        "cellpose.__main__",
        "torch",
        "cv2",
        "tifffile",
        "skimage.measure",
        "skimage.segmentation",
        "ezdxf",
        "PIL.ImageTk",
    ]
    for package in ("cellpose", "torch", "skimage", "cv2", "tifffile", "ezdxf"):
        package_datas, package_binaries, package_hiddenimports = collect_all(package)
        if package == "torch":
            excluded_suffixes = {".lib", ".exp", ".h", ".hpp", ".cuh", ".cpp", ".cc"}
            package_datas = [
                item for item in package_datas
                if Path(item[0]).suffix.lower() not in excluded_suffixes
                and "include" not in {part.lower() for part in Path(item[0]).parts}
            ]
            package_binaries = [
                item for item in package_binaries
                if Path(item[0]).suffix.lower() not in excluded_suffixes
            ]
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hiddenimports
    hiddenimports += collect_submodules("cellpose")

    return {
        "script": str(spec_dir / "sem_grain_web_app.py"),
        "pathex": [str(spec_dir)],
        "datas": datas,
        "binaries": binaries,
        "hiddenimports": hiddenimports,
        "runtime": runtime,
        "environment_root": str(environment_root),
    }
