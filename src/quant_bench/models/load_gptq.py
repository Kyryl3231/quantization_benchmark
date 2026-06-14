from __future__ import annotations
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from quant_bench.config import Config


def build_calibration_dataset(num_samples: int = 128) -> list[str]:
    base_samples = [
        "The quick brown fox jumps over the lazy dog. This sentence is used for language model calibration.",
        "Large language models generate text by predicting the next token from previous context.",
        "Quantization reduces model memory usage by storing weights with fewer bits.",
        "GPTQ is a post-training quantization method for transformer language models.",
        "A calibration dataset should contain representative text similar to expected model inputs.",
        "Machine learning systems often trade off speed, memory usage, and numerical precision.",
        "Evaluation benchmarks measure accuracy, generation speed, and peak memory consumption.",
        "Natural language processing models are trained on large corpora of human-written text.",
        "The model should preserve fluency and factual consistency after quantization.",
        "Efficient inference is important when deploying large neural networks on limited hardware.",
    ]

    samples = []
    while len(samples) < num_samples:
        for sample in base_samples:
            samples.append(sample)
            if len(samples) >= num_samples:
                break

    return samples


def load_gptq_model_and_tokenizer(config: Config):
    output_dir = Path(config.project.cache_dir) / f"{config.project.name}-gptq"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.models.baseline_model_id,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_dataset = build_calibration_dataset(num_samples=128)

    gptq_config = GPTQConfig(
        bits=config.gptq.bits,
        tokenizer=tokenizer,
        dataset=calibration_dataset,
        model_seqlen=config.gptq.calibration_max_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.models.baseline_model_id,
        quantization_config=gptq_config,
        device_map="auto",
        torch_dtype="auto",
    )

    model.eval()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer
