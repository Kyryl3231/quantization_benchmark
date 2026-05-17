from __future__ import annotations

import torch
import warnings
import logging
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import BitsAndBytesConfig
from datasets import disable_progress_bars

from quant_bench.config import Config


# Suppress noisy warnings and progress bars
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

disable_progress_bars()

# assert torch.cuda.is_available(), "This notebook requires a GPU runtime!"
# print(f"GPU: {torch.cuda.get_device_name(0)}")
# print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

QUANT_CONFIGS = {
    "fp16": None,
    "int8": BitsAndBytesConfig(load_in_8bit=True),
    "nf4": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ),
}


def load_bitsandbytes_model_and_tokenizer(config: Config):

    """Load model with the specified quantization."""
    quant_cfg = QUANT_CONFIGS[config.bitsandbytes.quant_name]
    kwargs = {"trust_remote_code": True, "torch_dtype": torch.float16, "device_map": "auto"}
    if quant_cfg is not None:
        kwargs["quantization_config"] = quant_cfg
    tokenizer = AutoTokenizer.from_pretrained(config.models.baseline_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config.models.baseline_model_id, **kwargs)
    model.eval()
    return model, tokenizer
