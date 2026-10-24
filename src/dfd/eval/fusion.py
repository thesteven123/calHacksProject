"""Logit-level route fusion (HEDGE-style, docs/implementation-architecture.md §5) + paired bootstrap.

Fusion parameters (per-route standardization + logistic-regression weights) are fit on the `fit` split
only — known generators — and then frozen before any held-out generator is scored.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dfd.eval.metrics import tpr_at_fpr


@dataclass
class Fusion:
    routes: list[str]
    mean: list[float]
    std: list[float]
    weights: list[float]
    bias: float

    def __call__(self, features: np.ndarray) -> np.ndarray:
        z = (np.asarray(features, dtype=np.float64) - np.array(self.mean)) / np.array(self.std)
        return z @ np.array(self.weights) + self.bias

    def to_dict(self) -> dict:
        return asdict(self)


def fit_fusion(features: np.ndarray, labels: np.ndarray, routes: list[str], c: float = 1.0) -> Fusion:
    """features: [N, R] route logits. L2-regularized logistic regression on standardized logits."""
    x = np.asarray(features, dtype=np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-9
    clf = LogisticRegression(C=c, max_iter=1000).fit((x - mean) / std, np.asarray(labels).astype(int))
    return Fusion(routes, mean.tolist(), std.tolist(), clf.coef_[0].tolist(), float(clf.intercept_[0]))


def paired_bootstrap(labels, score_a, score_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Stratified bootstrap of (b - a) for AUC and TPR@1%FPR on the same samples."""
    y = np.asarray(labels).astype(int)
    a, b = np.asarray(score_a, dtype=np.float64), np.asarray(score_b, dtype=np.float64)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        s = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        deltas.append((roc_auc_score(y[s], b[s]) - roc_auc_score(y[s], a[s]),
                       tpr_at_fpr(y[s], b[s], 0.01) - tpr_at_fpr(y[s], a[s], 0.01)))
    d = np.array(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {
        "d_auc": float(roc_auc_score(y, b) - roc_auc_score(y, a)), "d_auc_ci": [float(lo[0]), float(hi[0])],
        "d_tpr1": float(tpr_at_fpr(y, b, 0.01) - tpr_at_fpr(y, a, 0.01)), "d_tpr1_ci": [float(lo[1]), float(hi[1])],
        "n_boot": n_boot,
    }
