"""Loss functions: VIB detector objective, annealed beta schedule, distillation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, I)), summed over latent dims and averaged over the batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()


@dataclass
class BetaSchedule:
    """Linear warm-up to beta_max, never below beta_min after warm-up (guards against semantic collapse)."""

    beta_max: float
    beta_min: float
    warmup_steps: int

    def __call__(self, step: int) -> float:
        if self.warmup_steps <= 0 or step >= self.warmup_steps:
            return self.beta_max
        return max(self.beta_max * step / self.warmup_steps, self.beta_min if step > 0 else 0.0)


def detector_loss(
    out: dict[str, torch.Tensor],
    labels: torch.Tensor,
    beta: float,
    beauty_labels: torch.Tensor | None = None,
    beauty_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = F.binary_cross_entropy_with_logits(out["logit"], labels.float())
    kl = kl_standard_normal(out["mu"], out["logvar"])
    total = bce + beta * kl
    logs = {"bce": bce.item(), "kl": kl.item(), "beta": beta}
    if beauty_weight > 0 and beauty_labels is not None and "beauty_logit" in out:
        aux = F.binary_cross_entropy_with_logits(out["beauty_logit"], beauty_labels.float())
        total = total + beauty_weight * aux
        logs["beauty"] = aux.item()
    logs["loss"] = total.item()
    return total, logs


def distillation_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.7,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Hard-label BCE + temperature-scaled binary KL to the teacher (Stage 1 student training)."""
    hard = F.binary_cross_entropy_with_logits(student_logit, labels.float())
    t = temperature
    p_t = torch.sigmoid(teacher_logit / t)
    soft = F.binary_cross_entropy_with_logits(student_logit / t, p_t) - F.binary_cross_entropy(p_t, p_t)
    return (1 - alpha) * hard + alpha * soft * t * t
