"""Report whether the bundled Torch runtime can execute a small CUDA operation."""

from __future__ import annotations

import argparse
import json
from typing import Any


def probe_device() -> dict[str, Any]:
    result: dict[str, Any] = {
        "backend": "cpu",
        "cuda_available": False,
        "gpu_available": False,
        "torch_version": None,
        "cuda_runtime": None,
        "device_name": None,
        "capability": None,
        "total_memory_mb": None,
        "reason": None,
    }
    try:
        import torch

        result["torch_version"] = str(torch.__version__)
        result["cuda_runtime"] = torch.version.cuda
        if torch.version.cuda is None:
            result["reason"] = "当前发布包使用CPU版PyTorch"
            return result
        if not torch.cuda.is_available():
            result["reason"] = "未检测到兼容的NVIDIA显卡驱动或CUDA设备"
            return result

        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        # A real allocation and matrix multiplication catches driver/runtime
        # mismatches that a device-count check alone can miss.
        left = torch.ones((32, 32), dtype=torch.float32, device=device)
        product = left @ left
        torch.cuda.synchronize(device)
        if float(product[0, 0].item()) != 32.0:
            raise RuntimeError("CUDA计算结果校验失败")
        del left, product
        torch.cuda.empty_cache()

        total_memory_mb = int(properties.total_memory // (1024 * 1024))
        enough_memory = total_memory_mb >= 4096
        result.update({
            "backend": "cuda" if enough_memory else "cpu",
            "cuda_available": True,
            "gpu_available": enough_memory,
            "device_name": str(properties.name),
            "capability": f"{capability[0]}.{capability[1]}",
            "total_memory_mb": total_memory_mb,
            "reason": (
                "检测到CUDA，但显存小于4 GB；为避免CPSAM崩溃，已使用CPU兼容模式"
                if not enough_memory
                else None
            ),
        })
    except Exception as exc:
        result["reason"] = f"CUDA探测失败：{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = probe_device()
    if args.json:
        print("DEVICE_PROBE_JSON=" + json.dumps(result, ensure_ascii=False))
    else:
        for key, value in result.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
