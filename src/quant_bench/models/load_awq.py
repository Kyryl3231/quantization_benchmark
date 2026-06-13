# from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from quant_bench.config import Config


def load_awq_model_and_tokenizer(config: Config):

    source = config.models.baseline_model_id
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.float16,
        device_map="cuda",
    )

    dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")
    calibration_data = dataset.filter(
        lambda x: len(x["text"].strip()) > 50
    ).take(128)  # TODO: remove
    print(f"Calibration data: {len(calibration_data)} samples")
    
    qwen_awq_mappings = [
        AWQMapping(
            smooth_layer="re:.*input_layernorm",
            balance_layers=[
                "re:.*self_attn\\.q_proj", 
                "re:.*self_attn\\.k_proj", 
                "re:.*self_attn\\.v_proj"
            ],
        ),
        AWQMapping(
            smooth_layer="re:.*post_attention_layernorm",
            balance_layers=[
                "re:.*mlp\\.gate_proj", 
                "re:.*mlp\\.up_proj"
            ],
        ),
    ]

    recipe = [
        AWQModifier(
            mappings=qwen_awq_mappings,
            duo_scaling="both"
        ),
        QuantizationModifier(
            targets=["Linear"], 
            scheme="W4A16", 
            ignore=["lm_head"] # Leave the output head unquantized
        ),
    ]

    # recipe = [
    #     # Step 1: AWQ calibration / scaling (transform stage)
    #     AWQModifier(),

    #     # Step 2: INT4 quantization
    #     QuantizationModifier(
    #         targets=["Linear"],
    #         scheme="W4A16",
    #         ignore=["lm_head"],
    #     ),
    # ]

    output_dir = Path(config.project.cache_dir) / f"{config.project.name}-awq"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model takes {model.get_memory_footprint()/1e9:.2f} GBs of GPU RAM.")

    quantized_model = oneshot(
        model=model,
        recipe=recipe,
        dataset=calibration_data,
        num_calibration_samples=128,
        max_seq_length=config.awq.calibration_max_length,
        output_dir=output_dir,
    )
    
    print(f"Quantized model takes {quantized_model.get_memory_footprint()/1e9:.2f} GBs of GPU RAM.")


    # quantized_model = AutoModelForCausalLM.from_pretrained(
    #     output_dir,
    #     device_map="auto",
    #     dtype=torch.float16,
    #     trust_remote_code=True,
    #     # safetensors=True,
    # )
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
