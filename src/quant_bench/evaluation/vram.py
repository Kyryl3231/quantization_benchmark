from __future__ import annotations

from typing import Any

import torch

def reset_vram_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def measure_vram_usage(model: Any | None = None) -> dict[str, Any]:
    peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    current_allocated_gb = (
        torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    )
    return {
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "current_allocated_gb": current_allocated_gb,
        "model_present": model is not None,
    }
