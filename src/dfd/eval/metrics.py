"""Moderation-oriented metrics: AUC, AP, TPR@fixed-FPR, ECE, and per-slice breakdowns."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def binary_metrics(labels, logits) -> dict[str, Any]:
    labels = np.asarray(labels).astype(int)
    logits = np.asarray(logits, dtype=np.float64)
    probs = 1 / (1 + np.exp(-logits))
    out: dict[str, Any] = {
        "n": int(len(labels)),
        "n_fake": int(labels.sum()),
        "acc@0.5": float(((probs >= 0.5) == labels).mean()) if len(labels) else None,
        "ece": expected_calibration_error(labels, probs) if len(labels) else None,
    }
    if 0 < labels.sum() < len(labels):
        out.update(
            auc=float(roc_auc_score(labels, logits)),
            ap=float(average_precision_score(labels, logits)),
            **{"tpr@1%fpr": tpr_at_fpr(labels, logits, 0.01), "tpr@0.1%fpr": tpr_at_fpr(labels, logits, 0.001)},
        )
    else:
        out.update(auc=None, ap=None, **{"tpr@1%fpr": None, "tpr@0.1%fpr": None})
        out["mean_p_fake"] = float(probs.mean()) if len(probs) else None
    return out


def sliced_report(labels, logits, generators, severities) -> dict[str, Any]:
    """Overall metrics + per-generator (each generator's fakes vs. all reals) + per-severity slices."""
    labels, logits = np.asarray(labels).astype(int), np.asarray(logits, dtype=np.float64)
    generators, severities = np.asarray(generators), np.asarray(severities)
    report: dict[str, Any] = {"overall": binary_metrics(labels, logits), "by_generator": {}, "by_severity": {}}
    reals = labels == 0
    for g in sorted(set(generators[labels == 1])):
        m = reals | ((labels == 1) & (generators == g))
        report["by_generator"][str(g)] = binary_metrics(labels[m], logits[m])
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(severities):
        groups[str(s)].append(i)
    for s, idx in sorted(groups.items()):
        report["by_severity"][s] = binary_metrics(labels[idx], logits[idx])
    return report


def report_to_markdown(report: dict[str, Any], title: str = "Evaluation report") -> str:
    cols = ["n", "n_fake", "auc", "ap", "tpr@1%fpr", "tpr@0.1%fpr", "acc@0.5", "ece"]

    def fmt(v):
        return "-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    def table(rows: dict[str, dict]) -> list[str]:
        lines = ["| slice | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {k} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for k, r in rows.items()]
        return lines

    lines = [f"# {title}", "", "## Overall", *table({"overall": report["overall"]})]
    for section in ("by_generator", "by_severity"):
        if report.get(section):
            lines += ["", f"## {section.replace('_', ' ').title()}", *table(report[section])]
    for section, rows in report.items():
        if section.startswith("eval_") and isinstance(rows, dict):
            lines += ["", f"## {section}", *table({"overall": rows["overall"], **rows.get("by_generator", {})})]
    return "\n".join(lines) + "\n"
