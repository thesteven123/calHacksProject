"""Variational Information Bottleneck head with auxiliary beauty-filter output."""

from __future__ import annotations

import torch
from torch import nn


class VIBHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, dropout: float, beauty_head: bool):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.classifier = nn.Linear(latent_dim, 1)
        self.beauty = nn.Linear(latent_dim, 1) if beauty_head else None

    def forward(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encoder(feats).chunk(2, dim=-1)
        logvar = logvar.clamp(-10.0, 10.0)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        out = {"logit": self.classifier(z).squeeze(-1), "mu": mu, "logvar": logvar}
        if self.beauty is not None:
            out["beauty_logit"] = self.beauty(z).squeeze(-1)
        return out
