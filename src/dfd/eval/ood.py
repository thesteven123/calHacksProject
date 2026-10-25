"""Real-manifold OOD gate on VIB latents: 'real' verdicts far from the fit-split reals become 'abstain'."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

from dfd.eval.calibration import conformal_upper

MIN_REALS = 20


@dataclass
class OODGate:
    mean: list[float]
    precision: list[list[float]]
    gamma: float  # math.inf when n_cal is too small to certify alpha (nothing is gated)
    alpha: float
    n_fit: int
    n_cal: int
    method: str = "mahalanobis_lw"

    def distance(self, mu) -> np.ndarray:
        """Squared Mahalanobis distance of each row of mu [N, D] to the real-manifold Gaussian."""
        d = np.atleast_2d(np.asarray(mu, dtype=np.float64)) - np.asarray(self.mean)
        return np.einsum("ni,ij,nj->n", d, np.asarray(self.precision), d)

    def save(self, path: str | Path) -> None:
        data = asdict(self)
        data["gamma"] = self.gamma if math.isfinite(self.gamma) else None
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path) -> "OODGate":
        data = json.loads(Path(path).read_text())
        data["gamma"] = math.inf if data["gamma"] is None else float(data["gamma"])
        return cls(**data)


def fit_ood_gate(mu_reals: np.ndarray, alpha: float = 0.05, seed: int = 0) -> OODGate:
    """Random half (seeded) fits mean + LedoitWolf covariance; the other half sets gamma =
    sorted(d_cal)[ceil((n_cal+1)*(1-alpha)) - 1], or inf if that index exceeds n_cal."""
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError(f"OOD gate alpha must be finite and strictly between 0 and 1, got {alpha!r}")
    mu = np.asarray(mu_reals, dtype=np.float64)
    if len(mu) < MIN_REALS:
        raise ValueError(f"OOD gate needs >= {MIN_REALS} reals, got {len(mu)}")
    perm = np.random.default_rng(seed).permutation(len(mu))
    fit, cal = mu[perm[: len(mu) // 2]], mu[perm[len(mu) // 2:]]
    lw = LedoitWolf().fit(fit)
    gate = OODGate(mean=lw.location_.tolist(), precision=lw.precision_.tolist(), gamma=math.inf,
                   alpha=alpha, n_fit=len(fit), n_cal=len(cal))
    gate.gamma = conformal_upper(gate.distance(cal), alpha)
    return gate


def gated_verdicts(cal, logits, mu, gate: OODGate | None) -> np.ndarray:
    """cal.verdict(cal.prob(logits)); then 'real' -> 'abstain' where gate.distance(mu) > gate.gamma."""
    v = np.array([cal.verdict(p) for p in np.atleast_1d(cal.prob(logits))], dtype=object)
    if gate is not None and len(v):
        v[(v == "real") & (gate.distance(mu) > gate.gamma)] = "abstain"
    return v


@torch.no_grad()
def predict_with_latents(model, loader: DataLoader, device: torch.device) -> dict[str, list]:
    """Like dfd.training.trainer.predict but also returns 'mu' (list of latent_dim-float lists)."""
    model.eval()
    out: dict[str, list] = {"logits": [], "labels": [], "generators": [], "severities": [], "mu": []}
    for batch in loader:
        res = model(batch["pixels"].to(device))
        out["logits"] += res["logit"].float().cpu().tolist()
        out["mu"] += res["mu"].float().cpu().tolist()
        out["labels"] += batch["label"].tolist()
        out["generators"] += list(batch["generator"])
        out["severities"] += list(batch["severity"])
    return out
