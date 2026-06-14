from __future__ import annotations
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from quant_bench.config import Config


def patch_optimum_gptq_missing_hf_device_map() -> None:
    from optimum.gptq.quantizer import GPTQQuantizer

    if getattr(GPTQQuantizer, "_quant_bench_hf_device_map_patch", False):
        return

    original_pack_model = GPTQQuantizer.pack_model

    def patched_pack_model(self, model, quantizers):
        if not hasattr(model, "hf_device_map"):
            model.hf_device_map = {"": 0}
            print("Patched missing model.hf_device_map:", model.hf_device_map)

        return original_pack_model(self, model, quantizers)

    GPTQQuantizer.pack_model = patched_pack_model
    GPTQQuantizer._quant_bench_hf_device_map_patch = True


def load_wikitext2_calibration_dataset(num_samples: int = 128) -> list[str]:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train",
    )

    texts: list[str] = []
    for row in dataset:
        text = row["text"].strip()
        if text:
            texts.append(text)
        if len(texts) >= num_samples:
            break

    if not texts:
        raise RuntimeError(
            "WikiText-2 loaded, but no non-empty calibration texts were found.")

    return texts


def load_gptq_model_and_tokenizer(config: Config):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPTQModel inference requires CUDA, but torch.cuda.is_available() is False.")

    patch_optimum_gptq_missing_hf_device_map()

    output_dir = Path(config.project.cache_dir) / f"{config.project.name}-gptq"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.models.baseline_model_id,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calibration_dataset = load_wikitext2_calibration_dataset(num_samples=128)

    print("GPTQ calibration dataset: Salesforce/wikitext / wikitext-2-raw-v1")
    print("GPTQ calibration samples:", len(calibration_dataset))
    print("First sample:", calibration_dataset[0][:200])

    gptq_config = GPTQConfig(
        bits=config.gptq.bits,
        tokenizer=tokenizer,
        dataset=calibration_dataset,
        model_seqlen=config.gptq.calibration_max_length,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.models.baseline_model_id,
        quantization_config=gptq_config,
        device_map={"": 0},
        torch_dtype="auto",
    )

    model.eval()

    # Keep model on CUDA for GPTQModel Triton kernels.
    model.to("cuda:0")

    print("Model device after GPTQ:", next(model.parameters()).device)

    # Optional save. Do not move to CPU before returning.
    try:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Saved GPTQ model to:", output_dir)
    except Exception as exc:
        print("Warning: GPTQ model saving failed, but continuing with CUDA model.")
        print("Save error:", repr(exc))

    model.to("cuda:0")
    model.eval()

    return model, tokenizer
