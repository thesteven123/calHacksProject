"""Temperature scaling and abstain-band thresholds for 3-way moderation verdicts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Calibration:
    temperature: float = 1.0
    tau_real: float = 0.5
    tau_fake: float = 0.5
    target_fpr: float = 0.01
    target_fnr: float = 0.02
    crop_mode: str = "center"  # score distribution these thresholds were fit on ("center" | "multi")
    method: str = "empirical"  # "empirical" (pooled quantiles) | "conformal" (finite-sample split-conformal)
    n_cal_real: int = 0
    tau_real_by_generator: dict | None = None  # conformal lower thresholds per known generator (min is used)

    def prob(self, logit: float | np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.asarray(logit, dtype=np.float64) / self.temperature))

    def verdict(self, prob: float) -> str:
        if prob >= self.tau_fake:
            return "fake"
        if prob < self.tau_real:
            return "real"
        return "abstain"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        # Files written before crop_mode existed: infer it from the conventional filename.
        data.setdefault("crop_mode", "multi" if "multicrop" in Path(path).name else "center")
        return cls(**data)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    x = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def conformal_upper(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal upper threshold q: for a new score exchangeable with `scores`, P(new > q) <= alpha.

    q is the ceil((n+1)(1-alpha))-th smallest score; +inf when n is too small to certify alpha
    (need n >= 1/alpha - 1, e.g. >= 99 reals for 1% FPR), in which case nothing is auto-flagged.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


def conformal_lower(scores: np.ndarray, beta: float) -> float:
    """Split-conformal lower threshold q: for a new exchangeable score, P(new < q) <= beta.

    q is the floor((n+1)beta)-th smallest score; -inf when n is too small (nothing is auto-cleared).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))
    k = math.floor((len(s) + 1) * beta)
    return float("-inf") if k < 1 else float(s[k - 1])


def calibrate(logits, labels, target_fpr: float = 0.01, target_fnr: float = 0.02, method: str = "empirical",
              generators=None) -> Calibration:
    """Fit temperature T, then the two abstain-band thresholds.

    empirical: tau_fake / tau_real = pooled quantiles of real / fake probabilities (no finite-sample guarantee).
    conformal: tau_fake = split-conformal upper threshold on REAL probabilities -> P(real auto-flagged fake) <= FPR
               for reals exchangeable with the calibration reals (they must pass through the same platform
               transcoding emulation as deployment). tau_real = MIN over known generators of the per-generator
               conformal lower threshold -> P(fake auto-cleared real) <= FNR for each *known* generator. There is
               no guarantee for unseen generators (they are not exchangeable with any calibration set).
    """
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels).astype(int)
    if method not in ("empirical", "conformal"):
        raise ValueError(f"Unknown calibration method {method!r}")
    cal = Calibration(temperature=fit_temperature(logits, labels), target_fpr=target_fpr, target_fnr=target_fnr,
                      method=method)
    probs = cal.prob(logits)
    real_p, fake_p = probs[labels == 0], probs[labels == 1]
    cal.n_cal_real = int(len(real_p))
    if method == "empirical":
        cal.tau_fake = float(np.quantile(real_p, 1 - target_fpr, method="higher")) + 1e-9 if len(real_p) else 0.5
        cal.tau_real = float(np.quantile(fake_p, target_fnr, method="lower")) if len(fake_p) else 0.5
    else:
        q = conformal_upper(real_p, target_fpr)
        cal.tau_fake = float(np.nextafter(q, np.inf)) if np.isfinite(q) else 1.0 + 1e-9  # verdict "fake" iff p > q
        gens = np.asarray(generators) if generators is not None else np.array(["fake"] * len(labels))
        fake_gens = gens[labels == 1]
        per_gen = {str(g): conformal_lower(fake_p[fake_gens == g], target_fnr) for g in sorted(set(fake_gens))}
        cal.tau_real_by_generator = {g: (v if np.isfinite(v) else 0.0) for g, v in per_gen.items()}
        cal.tau_real = min(cal.tau_real_by_generator.values()) if per_gen else 0.0  # 0.0 -> never auto-clear
    cal.tau_real = min(cal.tau_real, cal.tau_fake)
    return cal
