"""Temperature scaling and abstain-band thresholds for 3-way moderation verdicts."""

from __future__ import annotations

import json
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
        return cls(**json.loads(Path(path).read_text()))


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


def calibrate(logits, labels, target_fpr: float = 0.01, target_fnr: float = 0.02) -> Calibration:
    """Fit T, then tau_fake = real-score quantile at (1 - target_fpr), tau_real = fake-score quantile at target_fnr."""
    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels).astype(int)
    cal = Calibration(temperature=fit_temperature(logits, labels), target_fpr=target_fpr, target_fnr=target_fnr)
    probs = cal.prob(logits)
    real_p, fake_p = probs[labels == 0], probs[labels == 1]
    cal.tau_fake = float(np.quantile(real_p, 1 - target_fpr, method="higher")) + 1e-9 if len(real_p) else 0.5
    cal.tau_real = float(np.quantile(fake_p, target_fnr, method="lower")) if len(fake_p) else 0.5
    cal.tau_real = min(cal.tau_real, cal.tau_fake)
    return cal
