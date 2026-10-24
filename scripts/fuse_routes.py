"""Fit core+VLM fusion on `fit`, freeze it, then report fit / select / test with paired bootstrap CIs.

    .venv-tinker/bin/python scripts/fuse_routes.py --config configs/tiny_genimage_nested.yaml \
        --core-ckpt runs/tiny_genimage_nested/best.pt \
        --vlm-fit runs/vlm_tinker_nested/score_fit.json --vlm-select runs/.../score_select_X.json \
        --vlm-test runs/.../score_test.json --output runs/fusion_nested

Core logits are recomputed locally on exactly the degraded images the VLM scored (dfd.eval.paired), so every
comparison is paired. Calibration for each route (and the fused score) is fit on `fit` only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from dfd.config import Config, PTEConfig
from dfd.data.datasets import build_base_datasets, eval_splits
from dfd.data.transforms import DetectorTransform
from dfd.env import load_dotenv
from dfd.eval.calibration import calibrate
from dfd.eval.fusion import fit_fusion, paired_bootstrap
from dfd.eval.metrics import binary_metrics
from dfd.eval.paired import paired_pte_images
from dfd.models.detector import DeepfakeDetector
from dfd.training.trainer import resolve_device


@torch.no_grad()
def core_logits(model, cfg, base, indices, device) -> np.ndarray:
    crop = DetectorTransform(PTEConfig(enabled=False), model.cfg.image_size, train=False)
    out = []
    for _, img, _, _, _ in paired_pte_images(cfg, base, indices):
        x = crop(img, np.random.default_rng(0))[0].unsqueeze(0).to(device)
        out.append(float(model(x)["logit"]))
    return np.array(out)


def verdict_rates(cal, logits, labels) -> dict:
    v = np.array([cal.verdict(p) for p in cal.prob(logits)])
    y = np.asarray(labels)
    return {"fpr_fake": float((v[y == 0] == "fake").mean()), "fnr_real": float((v[y == 1] == "real").mean()),
            "abstain": float((v == "abstain").mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--core-ckpt", required=True)
    ap.add_argument("--vlm-fit", required=True)
    ap.add_argument("--vlm-select", default=None)
    ap.add_argument("--vlm-test", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    load_dotenv()

    cfg = Config.from_yaml(args.config)
    _, base_val = build_base_datasets(cfg.data, cfg.train.seed)
    splits = eval_splits(cfg.data, base_val, cfg.train.seed)
    device = resolve_device(cfg.train.device)
    model, _ = DeepfakeDetector.load(args.core_ckpt)
    model.to(device).eval()

    data = {}
    for name, path in (("fit", args.vlm_fit), ("select", args.vlm_select), ("test", args.vlm_test)):
        if not path:
            continue
        vj = json.loads(Path(path).read_text())
        if vj.get("split") not in (None, name):
            raise SystemExit(f"{path} was scored on split {vj.get('split')!r}, expected {name!r}")
        idx = [p["idx"] for p in vj["preds"]]
        y = np.array([p["label"] for p in vj["preds"]])
        v = np.array([p["logit"] for p in vj["preds"]])
        c = core_logits(model, cfg, splits[name], idx, device)
        data[name] = {"y": y, "core": c, "vlm": v, "gen": np.array([p["generator"] for p in vj["preds"]])}
        print(f"[{name}] n={len(y)} (fakes {int(y.sum())}) core+vlm paired")

    fit = data["fit"]
    fusion = fit_fusion(np.c_[fit["core"], fit["vlm"]], fit["y"], ["core", "vlm"])
    for d in data.values():
        d["fused"] = fusion(np.c_[d["core"], d["vlm"]])
    cals = {r: calibrate(fit[r], fit["y"], cfg.train.target_fpr, cfg.train.target_fnr,
                         method=cfg.train.calibration_method, generators=fit["gen"]) for r in ("core", "vlm", "fused")}
    print(f"[fusion] weights core={fusion.weights[0]:.3f} vlm={fusion.weights[1]:.3f} (standardized, fit on 'fit')")

    report = {"fusion": fusion.to_dict(), "calibration": {r: c.__dict__ for r, c in cals.items()}, "splits": {}}
    for name, d in data.items():
        rep = {r: {**binary_metrics(d["y"], d[r]), "verdicts": verdict_rates(cals[r], d[r], d["y"])}
               for r in ("core", "vlm", "fused")}
        rep["fused_vs_core"] = paired_bootstrap(d["y"], d["core"], d["fused"])
        rep["vlm_vs_core"] = paired_bootstrap(d["y"], d["core"], d["vlm"])
        rep["score_corr"] = float(np.corrcoef(d["core"], d["vlm"])[0, 1])
        report["splits"][name] = rep
        for r in ("core", "vlm", "fused"):
            m, vr = rep[r], rep[r]["verdicts"]
            print(f"[{name}|{r:5s}] AUC={m['auc']:.3f} TPR@1%FPR={m['tpr@1%fpr']:.3f} | verdicts FPR={vr['fpr_fake']:.3f} "
                  f"FNR={vr['fnr_real']:.3f} abstain={vr['abstain']:.2f}")
        b = rep["fused_vs_core"]
        print(f"[{name}] fused-core dAUC={b['d_auc']:+.3f} CI{[round(x, 3) for x in b['d_auc_ci']]} "
              f"dTPR1={b['d_tpr1']:+.3f} CI{[round(x, 3) for x in b['d_tpr1_ci']]} r={rep['score_corr']:.2f}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion_report.json").write_text(json.dumps(report, indent=2))
    (out / "fusion.json").write_text(json.dumps({"fusion": fusion.to_dict(), "calibration": cals["fused"].__dict__}, indent=2))
    print(f"wrote {out}/fusion_report.json and fusion.json")


if __name__ == "__main__":
    main()
