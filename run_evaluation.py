from __future__ import annotations

import argparse
import json
import random
import sys
from dotenv import load_dotenv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quant_bench.config import Config, load_config
from quant_bench.evaluation.mmlu import evaluate_mmlu
from quant_bench.evaluation.speed import evaluate_generation_speed
from quant_bench.evaluation.vram import measure_vram_usage, reset_vram_stats

from quant_bench.utils.system_info import gather_run_metadata

import numpy as np
import torch
import transformers

# For debugging
transformers.logging.set_verbosity_info()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark quantization methods.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--mode", choices=["quick", "full"], default=None, help="Override benchmark mode.")
    parser.add_argument(
        "--quantization-method",
        choices=["all", "awq", "gptq", "bitsandbytes"],
        default=None,
        help="Quantization method to benchmark.",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Force a tiny quick run.")
    return parser.parse_args()


def ensure_parent_dir(path_str: str) -> None:
    Path(path_str).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def write_json(path_str: str, payload: dict[str, Any]) -> None:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compare_metrics(baseline: dict[str, Any], awq: dict[str, Any]) -> dict[str, Any]:
    baseline_speed = float(baseline["speed"]["tokens_per_second"])
    awq_speed = float(awq["speed"]["tokens_per_second"])
    baseline_vram = float(baseline["vram"]["peak_allocated_gb"])
    awq_vram = float(awq["vram"]["peak_allocated_gb"])
    return {
        "mmlu_accuracy_delta": float(awq["mmlu"]["accuracy"]) - float(baseline["mmlu"]["accuracy"]),
        "speedup_ratio": awq_speed / baseline_speed if baseline_speed else None,
        "vram_reduction_gb": baseline_vram - awq_vram,
        "vram_reduction_percent": ((baseline_vram - awq_vram) / baseline_vram * 100.0) if baseline_vram else None,
    }


def run_single_benchmark(config: Config, *, quantization_method: str) -> dict[str, Any]:
    if quantization_method == "bitsandbytes":
        from quant_bench.models.load_bnb import load_bitsandbytes_model_and_tokenizer
        model, tokenizer = load_bitsandbytes_model_and_tokenizer(config)
    elif quantization_method == "gptq":
        from quant_bench.models.load_gptq import load_gptq_model_and_tokenizer
        model, tokenizer = load_gptq_model_and_tokenizer(config)
    elif quantization_method == "awq":
        from quant_bench.models.load_awq import load_awq_model_and_tokenizer
        model, tokenizer = load_awq_model_and_tokenizer(config)
    elif quantization_method == "baseline":  # no quantization is applied and only baseline model is evaluated
        from quant_bench.models.load_baseline import load_baseline_model_and_tokenizer
        model, tokenizer = load_baseline_model_and_tokenizer(config)

    reset_vram_stats()
    mmlu_result = evaluate_mmlu(model, tokenizer, config)
    speed_result = evaluate_generation_speed(model, tokenizer, config)
    vram_result = measure_vram_usage(model)
    metadata = gather_run_metadata(config, model_tag=quantization_method)

    result =  {
        "metadata": metadata,
        "mmlu": mmlu_result,
        "speed": speed_result,
        "vram": vram_result,
    }

    results_path = f"results/{quantization_method}_metrics.json"
    ensure_parent_dir(results_path)
    write_json(results_path, result)
    print_summary(quantization_method, result)

    return result


def print_summary(name: str, result: dict[str, Any]) -> None:
    print(f"\n[{name}]")
    print(f"  MMLU accuracy: {result['mmlu']['accuracy']:.4f}")
    print(f"  Speed: {result['speed']['tokens_per_second']:.2f} tokens/sec")
    print(f"  VRAM: {result['vram']['peak_allocated_gb']:.3f} GB peak allocated")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    load_dotenv(config.project.env_path)

    random.seed(config.project.seed)
    if np is not None:
        np.random.seed(config.project.seed)
    torch.manual_seed(config.project.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.project.seed)

    if args.mode:
        config.benchmark.mode = args.mode
    if args.smoke_test:
        config.benchmark.mode = "quick"
        config.benchmark.quick_limit = min(config.benchmark.quick_limit, 8)
        config.benchmark.prompt_count = min(config.benchmark.prompt_count, 4)
        config.benchmark.max_new_tokens = min(config.benchmark.max_new_tokens, 16)

    run_single_benchmark(config, quantization_method="baseline")

    if args.quantization_method == "all":
        run_single_benchmark(config, quantization_method="bitsandbytes")
        run_single_benchmark(config, quantization_method="gptq")
        run_single_benchmark(config, quantization_method="awq")
    elif args.quantization_method is not None:
        run_single_benchmark(config, quantization_method=args.quantization_method)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
