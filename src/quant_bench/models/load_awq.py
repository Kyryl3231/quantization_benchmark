from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quant_bench.config import Config

LLMCOMPRESSOR_VERSION = "0.10.0"


# def _import_llmcompressor():
#     try:
#         from llmcompressor import oneshot
#         from llmcompressor.modifiers.awq import AWQModifier
#     except Exception as exc:  # pragma: no cover - import error surface is user-facing
#         raise RuntimeError(
#             f"AWQ loading requires the 'llmcompressor' package. Install llmcompressor=={LLMCOMPRESSOR_VERSION} and ensure CUDA support is available."
#         ) from exc
#     return oneshot, AWQModifier



# def _calibration_texts(config: BenchmarkConfig) -> list[str]:
#     dataset = load_dataset("cais/mmlu", "all", split=config.benchmark.mmlu_split)
#     texts: list[str] = []
#     for row in dataset.select(range(min(len(dataset), config.awq.local_calibration_samples))):
#         choices = row["choices"]
#         texts.append(
#             "Question: "
#             + row["question"]
#             + "\n"
#             + "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", choices))
#             + "\nAnswer:"
#         )
#     return texts


def load_awq_model_and_tokenizer(config: Config):

    # oneshot, AWQModifier = _import_llmcompressor()
    # source = _preferred_awq_source(config) or config.models.baseline_model_id
    source = config.models.baseline_model_id
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    # if config.models.quantization_bits != 4:
    #     raise ValueError(
    #         f"llmcompressor AWQ currently expects 4-bit weight quantization in this benchmark, got {config.models.quantization_bits}."
    #     )

    model = AutoModelForCausalLM.from_pretrained(
        source,
        trust_remote_code=True,
        # safetensors=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    scheme = "W4A16_ASYM" if config.awq.zero_point else "W4A16"
    # calib_data = _calibration_texts(config)
    # calib_dataset = Dataset.from_dict({"text": calib_data})
    calib_dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="train"
    )

    # Keep only non-empty samples
    calib_dataset = calib_dataset.filter(
        lambda x: len(x["text"].strip()) > 0
    )
    recipe: list[Any] = [
        AWQModifier(
            ignore=["lm_head"],
            scheme=scheme,
            targets=["Linear"],
        )
    ]

    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        # max_calib_samples=len(calib_data),
        max_seq_length=config.awq.calibration_max_length,
        # num_calibration_samples=len(calib_data),
        num_calibration_samples=128,
    )

    output_dir = Path(config.project.cache_dir) / f"{config.project.name}-awq"
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True, save_compressed=True)
    tokenizer.save_pretrained(str(output_dir))

    quantized_model = AutoModelForCausalLM.from_pretrained(
        str(output_dir),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        # safetensors=True,
    )
    quantized_model.eval()
    # metadata = {
    #     "model_id": str(output_dir),
    #     "loading_mode": "awq_local",
    #     "quantized_path": str(output_dir),
    #     "backend": "llmcompressor",
    #     "llmcompressor_version": LLMCOMPRESSOR_VERSION,
    #     "awq_scheme": scheme,
    #     "awq_group_size": config.awq.group_size,
    #     "awq_zero_point": config.awq.zero_point,
    # }
    return quantized_model, tokenizer#, metadata
