from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import platform
import socket
import sys
from typing import Any

import torch


def _get_llmcompressor_version() -> str | None:
    try:
        return version("llmcompressor")
    except PackageNotFoundError:
        return None


def gather_run_metadata(config: Any, *, model_tag: str) -> dict[str, Any]:
    gpu_name = None
    gpu_count = 0
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "torch_version": torch.__version__,
        "llmcompressor_version": _get_llmcompressor_version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "model_tag": model_tag,
        "project": config.project.name,
        "seed": config.project.seed,
        "config": config.as_dict(),
    }
