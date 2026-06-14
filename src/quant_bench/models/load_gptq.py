from __future__ import annotations
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from quant_bench.config import Config


def patch_optimum_gptq_missing_hf_device_map() -> None:
    """
    Workaround for Optimum/GPTQModel packing on models that do not expose
    model.hf_device_map during pack_model().
    """
    from optimum.gptq.quantizer import GPTQQuantizer

    if getattr(GPTQQuantizer, "_quant_bench_hf_device_map_patch", False):
        return

    original_pack_model = GPTQQuantizer.pack_model

    def patched_pack_model(self, model, quantizers):
        if not hasattr(model, "hf_device_map"):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")

            if device.type == "cuda":
                mapped_device = device.index if device.index is not None else 0
            else:
                mapped_device = "cpu"

            model.hf_device_map = {"": mapped_device}
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
        device_map="auto",
        torch_dtype="auto",
    )

    model.eval()

    try:
        model.to("cpu")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print("Saved GPTQ model to:", output_dir)
    except Exception as exc:
        print("Warning: GPTQ model saving failed, but quantized model was created.")
        print("Save error:", repr(exc))

    return model, tokenizer
