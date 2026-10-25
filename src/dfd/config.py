"""Typed configuration objects, loadable from nested YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class ModelConfig:
    backbone: str = "facebook/dinov2-base"
    feature_layers: tuple[int, ...] = (4, 8, 10, 12)
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_blocks: int = 3
    lora_targets: tuple[str, ...] = ("q_proj", "v_proj", "query", "value")
    hidden_dim: int = 512
    latent_dim: int = 128
    dropout: float = 0.1
    beauty_head: bool = True
    temporal: bool = False
    temporal_dim: int = 768
    temporal_kernels: tuple[int, ...] = (3, 5)
    topk_frac: float = 0.25
    image_size: int = 224


@dataclass
class PTEConfig:
    """Platform Transcoding Emulation parameters (see docs/implementation-architecture.md §3.2)."""

    enabled: bool = True
    min_hops: int = 1
    max_hops: int = 3
    short_side_buckets: tuple[int, ...] = (360, 480, 720, 1080)
    rescale_prob: float = 0.3
    rescale_range: tuple[float, float] = (0.5, 0.9)
    jpeg_prob: float = 0.6
    webp_prob: float = 0.25
    quality_range: tuple[int, int] = (30, 95)
    chroma_444_prob: float = 0.3
    blur_prob: float = 0.3
    blur_sigma: tuple[float, float] = (0.1, 1.2)
    noise_prob: float = 0.2
    noise_std: tuple[float, float] = (1.0, 6.0)
    color_jitter_prob: float = 0.3
    beauty_prob: float = 0.25


@dataclass
class DataConfig:
    source: str = "synthetic"  # synthetic | hf | folder | genimage_arrow
    data_root: str = "/data"  # root of the dfd-data Modal volume (local mirror path for non-Modal runs)
    genimage_generators: tuple[str, ...] = ()  # empty = all 8 canonical generators
    genimage_train_shards_per_generator: int = 4
    genimage_val_shards_per_generator: int | None = None
    align_size: int = 256
    hf_dataset: str = "TheKernel01/Tiny-GenImage"
    hf_data_files: dict[str, Any] | None = None
    hf_train_split: str = "train"
    hf_val_split: str = "validation"
    image_column: str = "image"
    label_column: str = "label"
    generator_column: str | None = "generator"
    holdout_generators: tuple[str, ...] = ()
    selection_generators: tuple[str, ...] = ()
    folder_root: str | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    synthetic_train_size: int = 256
    synthetic_val_size: int = 128
    synthetic_image_size: int = 256
    num_workers: int = 0
    pte: PTEConfig = field(default_factory=PTEConfig)
    dff_root: str | None = None
    dff_generators: tuple[str, ...] = ("inpainting", "insight", "text2img")
    dff_val_fraction: float = 0.2
    dff_format_align: bool = True
    sowaiba_root: str | None = None
    sowaiba_val_fraction: float = 0.2
    sowaiba_eval: bool = False
    sfhq_zip: str | None = None  # SFHQ-T2I zip; adds its fakes to source=paired_faces
    sfhq_generators: tuple[str, ...] = ()  # empty = all 5
    sfhq_val_fraction: float = 0.2
    sfhq_max_per_generator_train: int | None = 8000
    sfhq_max_per_generator_val: int | None = 1000


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    lora_lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    beta_max: float = 1e-3
    beta_min: float = 1e-5
    beta_warmup_frac: float = 0.3
    beauty_weight: float = 0.2
    device: str = "auto"
    amp: bool = True
    seed: int = 0
    output_dir: str = "runs/default"
    log_every: int = 10
    max_steps: int | None = None
    target_fpr: float = 0.01
    target_fnr: float = 0.02
    calibration_method: str = "conformal"  # "conformal" (split-conformal, per-generator FNR) | "empirical"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return from_dict(cls, yaml.safe_load(Path(path).read_text()) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, coercing lists to tuples where annotated."""
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = from_dict(hint, value)
        elif isinstance(value, list) and str(hint).startswith("tuple"):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)
