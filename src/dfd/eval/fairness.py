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


AGE3 = {"0-2": "young", "3-9": "young", "10-19": "young", "20-29": "adult", "30-39": "adult", "40-49": "adult",
        "50-59": "adult", "60-69": "senior", "more than 70": "senior"}  # FairFace age bands -> 0-19 / 20-59 / 60+
MIN_K_RATIO = 5  # max/min ratio is suppressed when the min group has fewer events (unstable division)


def _eb_rates(k: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Vectorised Beta-Binomial method-of-moments shrinkage; k is [G] or [G, R] (replicates), n is [G]."""
    k = k.astype(np.float64)
    n2 = n.astype(np.float64).reshape(-1, *([1] * (k.ndim - 1)))
    mu = k.sum(axis=0) / n.sum()
    tau2 = (np.var(k / n2, axis=0, ddof=1) - np.mean(mu * (1 - mu) / n2, axis=0)) if len(n) > 1 else np.zeros_like(mu)
    ok = (tau2 > 0) & (mu > 0) & (mu < 1)
    strength = np.where(ok, np.maximum(mu * (1 - mu) / np.where(ok, tau2, 1.0) - 1, 0.0), 0.0)
    return np.where(ok, (k + mu * strength) / (n2 + strength), mu * np.ones_like(k))


def eb_shrink(table: dict) -> dict[str, float]:
    """Empirical-Bayes Beta-Binomial posterior-mean rates (method of moments over the table's groups).

    Between-group variance = var(raw rates) - mean binomial sampling variance; if it is <= 0, all groups pool to the
    overall rate (no evidence of heterogeneity beyond noise).
    """
    if not table:
        return {}
    k = np.array([r["k"] for r in table.values()])
    n = np.array([r["n"] for r in table.values()])
    return {g: float(x) for g, x in zip(table, _eb_rates(k, n))}


def bootstrap_gap_ci(table: dict, groups: list[str], reps: int = 2000, seed: int = 0, level: float = 0.95,
                     shrink: bool = False) -> tuple[float, float] | None:
    """Percentile CI of max-min rate over `groups` by parametric (binomial) bootstrap within each group.

    shrink=True simulates every group of the table from its EB rate and re-applies eb_shrink to each replicate
    before taking max-min over `groups`, so the interval targets the shrunk gap rather than the raw one.
    """
    if len(groups) < 2:
        return None
    rng = np.random.default_rng(seed)
    names = list(table) if shrink else list(groups)
    n = np.array([table[g]["n"] for g in names])
    p = eb_shrink(table) if shrink else {g: table[g]["rate"] for g in names}
    sims = rng.binomial(n[:, None], np.array([p[g] for g in names])[:, None], size=(len(names), reps))
    rates = _eb_rates(sims, n) if shrink else sims / n[:, None]
    rates = rates[[names.index(g) for g in groups]]
    gaps = rates.max(axis=0) - rates.min(axis=0)
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
    excludes 0 is weak evidence of a real gap when groups are many and n is small. gap_eb / gap_eb_ci use
    empirical-Bayes shrunk rates (eb_shrink), which damp that extreme-value bias; prefer them for gap size. The EB
    interval is conservative (lower end often 0), so use the above-target tests, not gap_eb_ci, to detect disparity.
    Holm has low power at FPR ~1% and n ~100: an empty above_target_holm is not evidence of parity.
    ratio is None when the min group's rate is 0 or its event count is below MIN_K_RATIO.
    """
    powered = {g: r for g, r in table.items() if r["n"] >= min_n}
    hi = max(powered, key=lambda g: powered[g]["rate"]) if powered else None
    lo = min(powered, key=lambda g: powered[g]["rate"]) if powered else None
    max_rate = powered[hi]["rate"] if hi else None
    min_rate = powered[lo]["rate"] if lo else None
    eb = eb_shrink(table)
    eb_powered = [eb[g] for g in powered]
    holm_p = None if target is None else holm_adjust(
        {g: binom_sf(r["k"], r["n"], target) for g, r in table.items()})
    return {
        "max_group": hi, "max_rate": max_rate, "min_group": lo, "min_rate": min_rate,
        "gap": max_rate - min_rate if powered else None,
        "ratio": max_rate / min_rate if powered and min_rate > 0 and powered[lo]["k"] >= MIN_K_RATIO else None,
        "above_target_point": None if target is None else [g for g, r in table.items() if r["rate"] > target],
        "above_target_ci": None if target is None else [g for g, r in table.items() if r["ci_low"] > target],
        "p_above_target_holm": holm_p,
        "above_target_holm": None if holm_p is None else [g for g, q in holm_p.items() if q < alpha],
        "gap_ci": bootstrap_gap_ci(table, list(powered)),
        "rates_eb": eb,
        "gap_eb": max(eb_powered) - min(eb_powered) if powered else None,
        "gap_eb_ci": bootstrap_gap_ci(table, list(powered), shrink=True),
        "underpowered": [g for g, r in table.items() if r["n"] < min_n],
    }


def fairness_report(verdicts, groups_by_attr: dict[str, list[str]], target_fpr: float, min_n: int = 100,
                    intersections: tuple[tuple[str, ...], ...] = (("race", "gender"), ("race", "gender", "age3")),
                    verdicts_valid: bool = True) -> dict:
    """FPR and review-burden tables + summaries per attribute and per '<a>|<b>[|<c>]' intersection.

    When 'age' holds FairFace bands, a derived 'age3' (young 0-19 / adult 20-59 / senior 60+) is added so 3-way
    intersections stay reasonably sized; unknown bands are kept as-is.
    """
    v = np.asarray(verdicts).astype(str)
    attrs = {a: [str(x) for x in g] for a, g in groups_by_attr.items()}
    if "age" in attrs and "age3" not in attrs:
        attrs["age3"] = [AGE3.get(x, x) for x in attrs["age"]]
    for combo in intersections:
        if all(a in attrs for a in combo):
            attrs["|".join(combo)] = ["|".join(xs) for xs in zip(*(attrs[a] for a in combo))]
    out = {}
    for attr, groups in attrs.items():
        fpr, review = rate_table(v == "fake", groups), rate_table(v == "abstain", groups)
        out[attr] = {"fpr": {"table": fpr, "summary": disparity_summary(fpr, target_fpr, min_n)},
                     "review": {"table": review, "summary": disparity_summary(review, None, min_n)}}
    return {"verdicts_valid": bool(verdicts_valid), "target_fpr": target_fpr, "min_n": min_n, "attributes": out}
