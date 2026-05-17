from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ProjectConfig:
    name: str
    seed: int
    device: str
    output_dir: str
    cache_dir: str
    env_path: str


@dataclass
class ModelsConfig:
    baseline_model_id: str


@dataclass
class BenchmarkConfig:
    mode: str
    quick_limit: int
    prompt_count: int
    max_new_tokens: int
    do_sample: bool
    few_shot_k: int


@dataclass
class AWQConfig:
    zero_point: bool
    calibration_max_length: int


@dataclass
class BitsAndBytesConfig:
    quant_name: str


@dataclass
class GPTQConfig:
    bits: int
    calibration_max_length: int


@dataclass
class Config:
    project: ProjectConfig
    models: ModelsConfig
    benchmark: BenchmarkConfig
    awq: AWQConfig
    gptq: GPTQConfig
    bitsandbytes: BitsAndBytesConfig
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        return cls(
            project=ProjectConfig(**data["project"]),
            models=ModelsConfig(**data["models"]),
            benchmark=BenchmarkConfig(**data["benchmark"]),
            awq=AWQConfig(**data["awq"]),
            gptq=GPTQConfig(**data["gptq"]),
            bitsandbytes=BitsAndBytesConfig(**data["bitsandbytes"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    config = Config.from_yaml(config_path)
    return config
