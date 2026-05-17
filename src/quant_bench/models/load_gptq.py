from __future__ import annotations

import json
import os
import time
from pathlib import Path
import gc
import math

from quant_bench.config import Config
# import torch
# from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
# from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def load_gptq_model_and_tokenizer(config: Config):
    output_dir = Path(config.project.cache_dir) / f"{config.project.name}-gptq"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.models.baseline_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    gptq_config = GPTQConfig(
        bits=config.gptq.bits,
        tokenizer=tokenizer,
        dataset='wikitext2',
        model_seqlen=config.gptq.calibration_max_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.models.baseline_model_id,
        quantization_config=gptq_config,
        device_map=0, # cuda
    )

    model.to("cuda:0")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    model.eval()
    return model, tokenizer
