"""Frozen DINOv2/v3 feature extractor with multi-layer pooling and optional LoRA."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dfd.config import ModelConfig
from dfd.models.lora import inject_lora

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_hf_backbone(name: str) -> nn.Module:
    from transformers import AutoModel, Dinov2Config, Dinov2Model

    if name == "random-tiny":
        return Dinov2Model(
            Dinov2Config(
                hidden_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=128,
                image_size=224,
                patch_size=14,
            )
        )
    return AutoModel.from_pretrained(name)


def _transformer_blocks(model: nn.Module) -> nn.ModuleList:
    # DINOv2: encoder.layer | DINOv3 (transformers 5): model.layer | timm-style: blocks
    for path in ("encoder.layer", "model.layer", "layer", "blocks"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"Cannot locate transformer blocks in {type(model).__name__}")


class GatedAttentionPool(nn.Module):
    """Gated attention-MIL pooling over tokens (Ilse et al., 2018). The scorer's output layer is zero-initialised,
    so it starts as an exact mean pool."""

    def __init__(self, dim: int, attn_dim: int = 128):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(dim, attn_dim), nn.Linear(dim, attn_dim), nn.Linear(attn_dim, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, N, D] -> [B, D]"""
        a = self.w(torch.tanh(self.v(tokens)) * torch.sigmoid(self.u(tokens))).softmax(dim=1)
        return (a * tokens).sum(dim=1)


class FoundationBackbone(nn.Module):
    """Input: RGB in [0, 1], shape [B, 3, H, W]. Output: [B, len(layers) * 2 * hidden]."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.model = _load_hf_backbone(cfg.backbone)
        for p in self.model.parameters():
            p.requires_grad_(False)
        hcfg = self.model.config
        self.hidden_size = hcfg.hidden_size
        self.num_layers = hcfg.num_hidden_layers
        self.num_prefix_tokens = 1 + getattr(hcfg, "num_register_tokens", 0)
        self.layers = tuple(min(i, self.num_layers) for i in cfg.feature_layers)
        self.lora_count = 0
        if cfg.lora_rank > 0 and cfg.lora_blocks > 0:
            for block in _transformer_blocks(self.model)[-cfg.lora_blocks :]:
                self.lora_count += inject_lora(block, cfg.lora_targets, cfg.lora_rank, cfg.lora_alpha)
            if self.lora_count == 0:
                raise ValueError(f"No LoRA targets {cfg.lora_targets} found in {cfg.backbone}")
        if cfg.patch_pool not in ("mean", "attn_mil"):
            raise ValueError(f"Unknown patch_pool {cfg.patch_pool!r}")
        self.patch_pool = (nn.ModuleList(GatedAttentionPool(self.hidden_size) for _ in self.layers)
                           if cfg.patch_pool == "attn_mil" else None)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def out_dim(self) -> int:
        return len(self.layers) * 2 * self.hidden_size

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = (pixels - self.mean) / self.std
        with torch.set_grad_enabled(self.lora_count > 0 and torch.is_grad_enabled()):
            hidden = self.model(pixel_values=x, output_hidden_states=True).hidden_states
        feats = []
        for j, i in enumerate(self.layers):
            h = hidden[i]
            cls_tok = h[:, 0]
            patches = h[:, self.num_prefix_tokens :]
            patch_mean = patches.mean(dim=1) if self.patch_pool is None else self.patch_pool[j](patches)
            feats.append(F.layer_norm(cls_tok, (self.hidden_size,)))
            feats.append(F.layer_norm(patch_mean, (self.hidden_size,)))
        return torch.cat(feats, dim=-1)
