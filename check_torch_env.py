import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def module_origin(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve()


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cpu", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-python", type=Path)
    args = parser.parse_args()

    python_executable = Path(sys.executable).resolve()
    environment_root = python_executable.parent
    torch_origin = module_origin("torch")

    print(python_executable)
    print("has_torch", torch_origin is not None)
    if torch_origin is None:
        if args.require_cpu:
            print("CPU build stopped: torch is not installed", file=sys.stderr)
            return 2
        return 0

    import torch

    torch_cuda = torch.version.cuda
    cuda_available = bool(torch.cuda.is_available())
    print("torch", torch.__version__)
    print("torch_cuda", torch_cuda)
    print("cuda_available", cuda_available)
    print("device_count", torch.cuda.device_count())
    if cuda_available:
        print("device", torch.cuda.get_device_name(0))

    if args.require_cpu and torch_cuda is not None:
        print(
            f"CPU build stopped: torch.version.cuda is {torch_cuda!r}, expected None",
            file=sys.stderr,
        )
        return 3

    if args.expected_python is not None:
        expected_python = args.expected_python.resolve()
        if python_executable != expected_python:
            print(
                "CPU build stopped: use the project portable Python environment\n"
                f"  expected: {expected_python}\n"
                f"  current:  {python_executable}",
                file=sys.stderr,
            )
            return 6

    cellpose_origin = module_origin("cellpose")
    if args.require_cpu and (
        not is_inside(torch_origin, environment_root)
        or cellpose_origin is None
        or not is_inside(cellpose_origin, environment_root)
    ):
        print(
            "CPU build stopped: torch and cellpose must come from the selected Python environment",
            file=sys.stderr,
        )
        return 4

    try:
        cellpose_version = importlib.metadata.version("cellpose")
    except importlib.metadata.PackageNotFoundError:
        cellpose_version = None
    print("cellpose", cellpose_version)

    if args.require_cpu and cellpose_version is None:
        print("CPU build stopped: cellpose is not installed", file=sys.stderr)
        return 5

    if args.output is not None:
        manifest = {
            "python_version": platform.python_version(),
            "python_executable": str(python_executable),
            "torch_version": torch.__version__,
            "torch_cuda": torch_cuda,
            "cuda_available": cuda_available,
            "cellpose_version": cellpose_version,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("version_manifest", args.output.resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
