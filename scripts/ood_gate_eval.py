"""Experiment: real-manifold OOD gate on VIB latents (nothing is deployed).

    python scripts/ood_gate_eval.py --checkpoint /runs/<tag>/<rotation>/best.pt --alphas 0.05,0.10
    python scripts/ood_gate_eval.py --checkpoint runs/<run>/best.pt --skip-external   # local, no /data

Fits a Ledoit-Wolf Gaussian on fit-split real latents (mu) and a split-conformal threshold gamma, then
turns 'real' verdicts with squared Mahalanobis distance > gamma into 'abstain'. Only 'real' verdicts can
change, so FPR (fake verdict on reals) is untouched. Reports baseline vs gated rates for every alpha.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dfd.config import Config, PTEConfig
from dfd.data.datasets import HFImageDataset, TransformedDataset, build_base_datasets, eval_splits
from dfd.data.transforms import DetectorTransform
from dfd.env import load_dotenv
from dfd.eval.calibration import Calibration
from dfd.eval.ood import fit_ood_gate, gated_verdicts, predict_with_latents
from dfd.models.detector import DeepfakeDetector
from dfd.training.trainer import resolve_device


def _rate(mask: np.ndarray) -> float | None:
    return float(mask.mean()) if len(mask) else None


def verdict_stats(v: np.ndarray, labels: np.ndarray) -> dict:
    reals, fakes = v[labels == 0], v[labels == 1]
    return {
        "n": int(len(v)), "n_real": int(len(reals)), "n_fake": int(len(fakes)),
        "fpr_fake_verdict": _rate(reals == "fake"), "real_abstain_rate": _rate(reals == "abstain"),
        "fnr_real_verdict": _rate(fakes == "real"), "fake_abstain_rate": _rate(fakes == "abstain"),
        "frac_verdict_fake": _rate(v == "fake"), "abstain_rate": _rate(v == "abstain"),
    }


def _set_report(cal, gates: dict, logits, labels, mu) -> dict:
    out = {"baseline": verdict_stats(gated_verdicts(cal, logits, mu, None), labels)}
    for a, gate in gates.items():
        r = verdict_stats(gated_verdicts(cal, logits, mu, gate), labels)
        reals = labels == 0
        r["gate_real_rate"] = _rate(gate.distance(mu[reals]) > gate.gamma) if reals.any() else None
        both = reals.any() and (labels == 1).any()
        r["ood_auc"] = float(roc_auc_score(labels, gate.distance(mu))) if both else None
        out[f"alpha={a}"] = r
    return out


def _eval_set(cal, gates: dict, preds: dict, by_generator: bool) -> dict:
    logits, labels = np.asarray(preds["logits"]), np.asarray(preds["labels"])
    mu, gens = np.asarray(preds["mu"]), np.asarray(preds["generators"])
    report = {"overall": _set_report(cal, gates, logits, labels, mu)}
    if by_generator:
        report["by_generator"] = {str(g): _set_report(cal, gates, logits[gens == g], labels[gens == g], mu[gens == g])
                                  for g in sorted(set(gens))}
    return report


def _fmt(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--alphas", default="0.05,0.10", help="pre-declared; all are reported, none selected")
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--frames-per-clip", type=int, default=8)
    ap.add_argument("--max-per-generator", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip-external", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv()
    ckpt = Path(args.checkpoint)
    run_dir = ckpt.parent
    try:
        alphas = [float(a) for a in args.alphas.split(",")]  # empty or non-numeric entries raise ValueError
    except ValueError:
        alphas = []
    if not alphas or any(not (np.isfinite(a) and 0.0 < a < 1.0) for a in alphas):
        ap.error(f"--alphas must be a comma list of values strictly between 0 and 1, got {args.alphas!r}")
    cfg = Config.from_yaml(run_dir / "config.yaml")
    cal = Calibration.load(run_dir / "calibration.json")
    if cal.crop_mode != "center":
        raise ValueError(f"calibration crop_mode={cal.crop_mode!r}; the OOD experiment needs 'center'")
    model, _ = DeepfakeDetector.load(ckpt)
    device = resolve_device(args.device)
    model.to(device).eval()

    seed, size = cfg.train.seed, cfg.model.image_size
    if cfg.data.source == "hf":
        base_val = HFImageDataset(cfg.data, cfg.data.hf_val_split, max_samples=cfg.data.max_val_samples)
    else:
        base_val = build_base_datasets(cfg.data, seed)[1]
    splits = eval_splits(cfg.data, base_val, seed)
    val_tf = DetectorTransform(cfg.data.pte, size, train=False)
    sets = {"fit": (TransformedDataset(splits["fit"], val_tf, seed + 1), False),
            "select": (TransformedDataset(splits["select"], val_tf, seed + 1), False),
            "test": (TransformedDataset(splits["test"], val_tf, seed + 2), False)}
    if not args.skip_external:
        from dfd.data.external import SFHQT2IDataset, WildDeepfakeDataset

        root = Path(args.data_root)
        ext_tf = DetectorTransform(PTEConfig(enabled=False), size, train=False)
        sets["wilddeepfake"] = (TransformedDataset(
            WildDeepfakeDataset(root / "wilddeepfake", frames_per_clip=args.frames_per_clip), ext_tf, seed), False)
        sets["sfhq_t2i"] = (TransformedDataset(
            SFHQT2IDataset(root / "sfhq_t2i" / "SyntheticFacesHighQuality-T2I.zip",
                           max_per_generator=args.max_per_generator), ext_tf, seed), True)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    preds = {name: predict_with_latents(model, DataLoader(ds, **loader_kw), device) for name, (ds, _) in sets.items()
             if len(ds)}
    fit_labels = np.asarray(preds["fit"]["labels"])
    mu_fit_reals = np.asarray(preds["fit"]["mu"])[fit_labels == 0]
    gates = {a: fit_ood_gate(mu_fit_reals, a, seed=seed) for a in alphas}
    if 0.05 in gates:
        gates[0.05].save(run_dir / "ood_gate_alpha0.05.json")

    report = {
        "checkpoint": str(ckpt), "alphas": alphas, "calibration": {"method": cal.method, "crop_mode": cal.crop_mode,
                                                                    "tau_fake": cal.tau_fake, "tau_real": cal.tau_real},
        "gates": {str(a): {"gamma": g.gamma if np.isfinite(g.gamma) else None, "n_fit": g.n_fit, "n_cal": g.n_cal,
                           "method": g.method} for a, g in gates.items()},
        "sets": {name: _eval_set(cal, gates, preds[name], by_gen) for name, (_, by_gen) in sets.items()
                 if name != "fit" and name in preds},
    }
    out = Path(args.output or run_dir / "ood_gate_eval.json")
    out.write_text(json.dumps(report, indent=2))

    cols = ("n", "fpr_fake_verdict", "real_abstain_rate", "fnr_real_verdict", "fake_abstain_rate",
            "frac_verdict_fake", "abstain_rate", "ood_auc")
    print(f"{'set':28s} {'variant':12s} " + " ".join(f"{c[:10]:>10s}" for c in cols))
    for name, rep in report["sets"].items():
        rows = [(name, rep["overall"])] + [(f"{name}/{g}", r) for g, r in rep.get("by_generator", {}).items()]
        for label, variants in rows:
            for variant, r in variants.items():
                vals = [str(r["n"])] + [_fmt(r.get(c)) for c in cols[1:]]
                print(f"{label:28s} {variant:12s} " + " ".join(f"{v:>10s}" for v in vals))
    for a, g in report["gates"].items():
        print(f"[ood] alpha={a} gamma={g['gamma']} n_fit={g['n_fit']} n_cal={g['n_cal']}")
    print(f"[ood] -> {out}")
    return report


if __name__ == "__main__":
    main()
