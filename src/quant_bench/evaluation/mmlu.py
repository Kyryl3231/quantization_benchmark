from __future__ import annotations

from typing import Any

from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator

from quant_bench.config import Config


def evaluate_mmlu(model: Any, tokenizer: Any, config: Config) -> dict[str, Any]:
    lm_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        dtype=model.dtype,
        device="cuda"
    )

    limit = config.benchmark.quick_limit if config.benchmark.mode == "quick" else None
    
    results = evaluator.simple_evaluate(
        model=lm_model,
        tasks=["mmlu"],
        num_fewshot=config.benchmark.few_shot_k,
        batch_size="auto:4",
        limit=limit,
    )

    return {
        "accuracy": results["results"]["mmlu"]["acc,none"],
        "total": results["results"]["mmlu"]["sample_len"],
    }
