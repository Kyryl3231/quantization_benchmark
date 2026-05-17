from __future__ import annotations

import time
from typing import Any

import torch

from quant_bench.config import Config


def _get_input_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def evaluate_generation_speed(model: Any, tokenizer: Any, config: Config) -> dict[str, Any]:
    prompts = [
        "Explain the theory of relativity in simple terms:"
    ]
    device = _get_input_device(model)
    generated_tokens = 0
    latencies: list[float] = []

    model.eval()
    if prompts:
        warmup = tokenizer(prompts[0], return_tensors="pt")
        warmup = {key: value.to(device) for key, value in warmup.items()}
        with torch.inference_mode():
            _ = model.generate(
                **warmup,
                max_new_tokens=min(8, config.benchmark.max_new_tokens),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    start_total = time.perf_counter()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=config.benchmark.max_new_tokens,
                do_sample=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        generated_tokens += max(0, output_ids.shape[-1] - inputs["input_ids"].shape[-1])
    total_elapsed = time.perf_counter() - start_total

    return {
        "tokens_per_second": (generated_tokens / total_elapsed) if total_elapsed else 0.0,
        "average_latency_seconds": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": total_elapsed,
        "prompt_count": len(prompts),
    }
