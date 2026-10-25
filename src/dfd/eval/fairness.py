"""Demographic fairness on all-real sets: per-group FPR (verdict 'fake') and review burden (verdict 'abstain')."""

from __future__ import annotations

import math

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (z=1.96) Wilson score interval for k successes in n trials; (nan, nan) when n == 0."""
    if n == 0:
        return math.nan, math.nan
    p, z2 = k / n, z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (0.0 if k == 0 else max(0.0, centre - half)), (1.0 if k == n else min(1.0, centre + half))


def binom_sf(k: int, n: int, p: float) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p), summed in log space (stable for n ~ 1e4)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0 or p >= 1.0:
        return float(p >= 1.0)
    i = np.arange(k, n + 1)
    logs = (math.lgamma(n + 1) - np.array([math.lgamma(j + 1) + math.lgamma(n - j + 1) for j in i])
            + i * math.log(p) + (n - i) * math.log1p(-p))
    m = logs.max()
    return float(min(1.0, math.exp(m) * np.exp(logs - m).sum()))


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (controls family-wise error over the dict's groups)."""
    order = sorted(pvals, key=pvals.get)
    out, running, m = {}, 0.0, len(order)
    for rank, g in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[g]))
        out[g] = running
    return out


def bootstrap_gap_ci(table: dict, groups: list[str], reps: int = 2000, seed: int = 0,
                     level: float = 0.95) -> tuple[float, float] | None:
    """Percentile CI of max-min rate over `groups` by parametric (binomial) bootstrap within each group."""
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    sims = np.stack([rng.binomial(table[g]["n"], table[g]["rate"], size=reps) / table[g]["n"] for g in groups])
    gaps = sims.max(axis=0) - sims.min(axis=0)
    a = (1 - level) / 2
    return float(np.quantile(gaps, a)), float(np.quantile(gaps, 1 - a))


def rate_table(flags, groups) -> dict[str, dict]:
    """{group: {n, k, rate, ci_low, ci_high}} with groups sorted; flags are booleans per row."""
    f, g = np.asarray(flags, dtype=bool), np.asarray(groups).astype(str)
    out: dict[str, dict] = {}
    for name in sorted(set(g.tolist())):
        sel = f[g == name]
        n, k = int(len(sel)), int(sel.sum())
        lo, hi = wilson_interval(k, n)
        out[name] = {"n": n, "k": k, "rate": k / n, "ci_low": lo, "ci_high": hi}
    return out


def disparity_summary(table: dict, target: float | None, min_n: int = 100, alpha: float = 0.05) -> dict:
    """Max/min/gap/ratio over powered groups (n >= min_n); above-target lists cover every group.

    above_target_ci uses each group's own Wilson bound (no multiplicity control); above_target_holm uses exact
    one-sided binomial tests of H0: rate <= target with Holm correction over the table's groups (FWER <= alpha).
    gap_ci is a parametric-bootstrap percentile interval; max-min of noisy rates is biased upward, so a CI that
    excludes 0 is weak evidence of a real gap when groups are many and n is small.
    """
    powered = {g: r for g, r in table.items() if r["n"] >= min_n}
    hi = max(powered, key=lambda g: powered[g]["rate"]) if powered else None
    lo = min(powered, key=lambda g: powered[g]["rate"]) if powered else None
    max_rate = powered[hi]["rate"] if hi else None
    min_rate = powered[lo]["rate"] if lo else None
    holm_p = None if target is None else holm_adjust(
        {g: binom_sf(r["k"], r["n"], target) for g, r in table.items()})
    return {
        "max_group": hi, "max_rate": max_rate, "min_group": lo, "min_rate": min_rate,
        "gap": max_rate - min_rate if powered else None,
        "ratio": max_rate / min_rate if powered and min_rate > 0 else None,
        "above_target_point": None if target is None else [g for g, r in table.items() if r["rate"] > target],
        "above_target_ci": None if target is None else [g for g, r in table.items() if r["ci_low"] > target],
        "p_above_target_holm": holm_p,
        "above_target_holm": None if holm_p is None else [g for g, q in holm_p.items() if q < alpha],
        "gap_ci": bootstrap_gap_ci(table, list(powered)),
        "underpowered": [g for g, r in table.items() if r["n"] < min_n],
    }


def fairness_report(verdicts, groups_by_attr: dict[str, list[str]], target_fpr: float, min_n: int = 100,
                    intersections: tuple[tuple[str, str], ...] = (("race", "gender"),),
                    verdicts_valid: bool = True) -> dict:
    """FPR and review-burden tables + summaries per attribute and per '<a>|<b>' intersection."""
    v = np.asarray(verdicts).astype(str)
    attrs = {a: [str(x) for x in g] for a, g in groups_by_attr.items()}
    for a, b in intersections:
        if a in attrs and b in attrs:
            attrs[f"{a}|{b}"] = [f"{x}|{y}" for x, y in zip(attrs[a], attrs[b])]
    out = {}
    for attr, groups in attrs.items():
        fpr, review = rate_table(v == "fake", groups), rate_table(v == "abstain", groups)
        out[attr] = {"fpr": {"table": fpr, "summary": disparity_summary(fpr, target_fpr, min_n)},
                     "review": {"table": review, "summary": disparity_summary(review, None, min_n)}}
    return {"verdicts_valid": bool(verdicts_valid), "target_fpr": target_fpr, "min_n": min_n, "attributes": out}
