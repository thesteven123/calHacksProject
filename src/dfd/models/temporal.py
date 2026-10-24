"""Video components: post-backbone temporal adapter, top-k pooling, keyframe selection."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class TemporalAdapter(nn.Module):
    """Residual dual-branch depthwise Conv1d over per-frame features. Identity at init."""

    def __init__(self, in_dim: int, dim: int, kernels: tuple[int, ...] = (3, 5)):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.branches = nn.ModuleList(nn.Conv1d(dim, dim, k, padding=k // 2, groups=dim) for k in kernels)
        self.mix = nn.Linear(dim * len(kernels), in_dim)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        h = h.transpose(1, 2)
        mixed = torch.cat([F.gelu(b(h)) for b in self.branches], dim=1).transpose(1, 2)
        return x + self.mix(mixed)


def topk_pool(logits: torch.Tensor, mask: torch.Tensor | None = None, k_frac: float = 0.25) -> torch.Tensor:
    """Mean of the top-k frame logits per video; k = ceil(k_frac * valid_frames)."""
    if mask is None:
        mask = torch.ones_like(logits, dtype=torch.bool)
    mask = mask.bool()
    valid = mask.sum(dim=1).clamp(min=1)
    k = torch.clamp((valid.float() * k_frac).ceil().long(), min=1)
    sorted_logits, _ = logits.masked_fill(~mask, float("-inf")).sort(dim=1, descending=True)
    ranks = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    keep = ranks < k.unsqueeze(1)
    return (sorted_logits.masked_fill(~keep, 0.0)).sum(dim=1) / k.float()


def select_keyframes(signal: np.ndarray, n: int, anchor_frac: float = 0.5) -> np.ndarray:
    """Pick `n` frame indices: uniform anchors + frames with the largest change vs. their predecessor.

    `signal` is any cheap per-frame descriptor [T, D] (e.g. 32x32 grayscale thumbnails), so selection
    is not a fixed stride an attacker can schedule around (adversarial review §1.3).
    """
    t = len(signal)
    if t <= n:
        return np.arange(t)
    n_anchor = max(1, math.floor(n * anchor_frac))
    anchors = np.linspace(0, t - 1, n_anchor).round().astype(int)
    flat = signal.reshape(t, -1).astype(np.float32)
    delta = np.concatenate([[0.0], np.abs(np.diff(flat, axis=0)).mean(axis=1)])
    delta[anchors] = -np.inf
    extra = np.argsort(-delta)[: n - n_anchor]
    return np.sort(np.unique(np.concatenate([anchors, extra])))
