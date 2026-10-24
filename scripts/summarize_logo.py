"""Summarize nested-LOGO rotations from scripts/modal_logo.py, and compare two runs rotation by rotation.

    .venv/bin/python scripts/summarize_logo.py runs/modal/logo7_dinov2
    .venv/bin/python scripts/summarize_logo.py runs/modal/logo7_dinov2 --compare runs/modal/logo7_dinov3

Across-rotation uncertainty: mean with a bootstrap 95% CI over rotations. Backbone comparison: per-rotation
paired differences (same test generator, same images, same degradations) with an exact sign-flip test, which
makes no distributional assumption and is valid for small n (7 rotations -> 128 sign patterns).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

METRICS = ("auc", "tpr@1%fpr")


def load(run_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(run_dir.glob("*.json")):
        r = json.loads(f.read_text())
        test = ((r.get("report") or {}).get("eval_test") or {}).get("overall")
        if r.get("returncode") == 0 and test:
            out[r["rotation"]] = {"test": r["test"], "cost": r["est_cost_usd"], "wall_s": r["wall_s"],
                                  **{m: test[m] for m in METRICS},
                                  "fnr": r["report"]["verdicts_heldout"]["fnr_real_verdict"],
                                  "abstain": r["report"]["verdicts_heldout"]["abstain_rate"]}
        else:
            print(f"  ! {f.name}: missing/failed (rc={r.get('returncode')})")
    return out


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p-value for mean(d) != 0."""
    obs = abs(d.mean())
    flips = [abs((d * np.array(s)).mean()) for s in itertools.product((1, -1), repeat=len(d))]
    return float(np.mean([f >= obs - 1e-12 for f in flips]))


def table(name: str, rows: dict[str, dict]) -> None:
    print(f"\n## {name}  ({len(rows)} rotations)")
    print(f"| rotation | test gen | AUC | TPR@1%FPR | FNR (verdict) | abstain | wall min | $ |")
    print("|---|---|---|---|---|---|---|---|")
    for k, r in sorted(rows.items(), key=lambda kv: kv[1]["test"]):
        print(f"| {k} | {r['test']} | {r['auc']:.3f} | {r['tpr@1%fpr']:.3f} | {r['fnr']:.3f} | {r['abstain']:.2f} "
              f"| {r['wall_s']/60:.1f} | {r['cost']:.2f} |")
    for m in METRICS:
        x = np.array([r[m] for r in rows.values()])
        lo, hi = boot_ci(x)
        print(f"- mean {m}: **{x.mean():.3f}** (95% CI over rotations [{lo:.3f}, {hi:.3f}]), min {x.min():.3f}, max {x.max():.3f}")
    print(f"- total est. cost ${sum(r['cost'] for r in rows.values()):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()
    a = load(Path(args.run_dir))
    table(Path(args.run_dir).name, a)
    if args.compare:
        b = load(Path(args.compare))
        table(Path(args.compare).name, b)
        common = sorted(set(a) & set(b))
        print(f"\n## Paired: {Path(args.compare).name} - {Path(args.run_dir).name}  (n={len(common)} rotations)")
        for m in METRICS:
            d = np.array([b[k][m] - a[k][m] for k in common])
            wins = int((d > 0).sum())
            print(f"- d{m}: mean **{d.mean():+.3f}**, wins {wins}/{len(d)}, exact sign-flip p={sign_flip_p(d):.3f}; "
                  f"per rotation: " + ", ".join(f"{b[k]['test']} {v:+.3f}" for k, v in zip(common, d)))


if __name__ == "__main__":
    main()
