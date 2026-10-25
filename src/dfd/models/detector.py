"""Stage 2 core detector: frozen foundation backbone (+LoRA) -> [temporal adapter] -> VIB head."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dfd.config import ModelConfig, from_dict
from dfd.models.backbone import FoundationBackbone
from dfd.models.heads import VIBHead
from dfd.models.temporal import TemporalAdapter, topk_pool


class DeepfakeDetector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = FoundationBackbone(cfg)
        feat_dim = self.backbone.out_dim
        self.temporal = TemporalAdapter(feat_dim, cfg.temporal_dim, cfg.temporal_kernels) if cfg.temporal else None
        self.head = VIBHead(feat_dim, cfg.hidden_dim, cfg.latent_dim, cfg.dropout, cfg.beauty_head)

    def forward(self, pixels: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """pixels: [B,3,H,W] (image) or [B,T,3,H,W] (video, optional frame mask [B,T])."""
        if pixels.dim() == 4:
            return self.head(self.backbone(pixels))
        b, t = pixels.shape[:2]
        feats = self.backbone(pixels.flatten(0, 1)).view(b, t, -1)
        if self.temporal is not None:
            feats = self.temporal(feats, mask)
        out = self.head(feats.flatten(0, 1))
        frame_logits = out["logit"].view(b, t)
        out["frame_logits"] = frame_logits
        out["logit"] = topk_pool(frame_logits, mask, self.cfg.topk_frac)
        return out

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        lora = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" in n]
        head = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" not in n]
        return {"lora": lora, "head": head}

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        full = self.cfg.backbone == "random-tiny"
        state = {n: t for n, t in self.state_dict().items() if full or "lora_" in n or not n.startswith("backbone.")}
        torch.save({"model_config": dataclasses.asdict(self.cfg), "state_dict": state, **(extra or {})}, path)

    def warm_start(self, path: str | Path) -> dict[str, list[str]]:
        """Copy every checkpoint tensor whose name and shape match this model, except the temporal adapter (it stays
        identity-initialised). Returns {loaded, shape_mismatch, not_in_model, temporal_skipped} key lists."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        own = self.state_dict()
        out: dict[str, list[str]] = {"loaded": [], "shape_mismatch": [], "not_in_model": [], "temporal_skipped": []}
        for k, v in ckpt.items():
            if k.startswith("temporal."):
                out["temporal_skipped"].append(k)
            elif k not in own:
                out["not_in_model"].append(k)
            elif own[k].shape != v.shape:
                out["shape_mismatch"].append(k)
            else:
                out["loaded"].append(k)
        self.load_state_dict({k: ckpt[k] for k in out["loaded"]}, strict=False)
        return out

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> tuple["DeepfakeDetector", dict]:
        # weights_only=True: refuse arbitrary pickled objects (code execution on load). Checkpoints contain only
        # tensors and plain containers/primitives, so no allowlist is needed (CODE_REVIEW 2026-10-24, High).
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(from_dict(ModelConfig, ckpt["model_config"]))
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if unexpected or any(not k.startswith("backbone.") or "lora_" in k for k in missing):
            raise RuntimeError(f"Checkpoint mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
        return model, ckpt
