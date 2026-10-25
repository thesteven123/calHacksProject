"""Paired comparison of two 7-rotation nested-LOGO runs (pre-registered: runs/helper/TASK_midjourney_ablation.md).

    # 1. re-score an existing run's best.pt per rotation on its exact test split (eval only; writes a NEW dir)
    .venv/bin/modal run --detach scripts/modal_ablation.py --source-tag full7_dinov2_conformal --tag full7_baseline_rescore
    # 2. (the ablation run itself writes test_preds.json via scripts/train.py; fetch both with `modal volume get`)
    # 3. local analysis: paired stratified bootstrap + the pre-registered decision
    python scripts/modal_ablation.py analyze --baseline DIR --ablation DIR --reports DIR [--recorded DIR]

The test split is rebuilt exactly as scripts/train.py does (nested split of the validation set, seed + 2 eval
transform), so both models are scored on identical images.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
REMOTE = "/root/project"
GPU, CPU, MEM_MB, TIMEOUT_S = "L4", 8.0, 32768, 3600
PRICE_PER_S = 0.000222 + CPU * 0.0000131 + (MEM_MB / 1024) * 0.00000222  # same list prices as modal_logo.py
ROTATIONS = ("ADM-to-Wukong", "BigGAN-to-ADM", "GLIDE-to-BigGAN", "Midjourney-to-GLIDE", "SD15-to-Midjourney",
             "VQDM-to-SD15", "Wukong-to-VQDM")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(  # same pins as scripts/modal_logo.py
        "torch==2.14.1", "torchvision==0.29.1", "transformers==5.20.1", "datasets==5.1.0",
        "huggingface-hub==2.2.0", "timm==1.0.30", "pillow==13.0.0", "numpy==2.5.4", "scikit-learn==1.9.1",
        "pyyaml==6.0.3", "opencv-python-headless==5.0.0.93", "safetensors==0.8.0", "tqdm==4.70.1",
    )
    .env({"PYTHONPATH": f"{REMOTE}/src", "HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(PROJECT / "src", f"{REMOTE}/src")
)
app = modal.App("dfd-ablation", image=image)
hf_cache = modal.Volume.from_name("dfd-hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("dfd-runs", create_if_missing=True)
data_vol = modal.Volume.from_name("dfd-data", create_if_missing=True)


def rescore_test(run_dir: str, num_workers: int = 6, device: str = "auto") -> dict:
    """best.pt of a train.py run scored on that run's test split, rebuilt exactly as train.py builds it."""
    from torch.utils.data import DataLoader

    from dfd.config import Config
    from dfd.data.datasets import HFImageDataset, TransformedDataset, build_base_datasets, eval_splits
    from dfd.data.transforms import DetectorTransform
    from dfd.models.detector import DeepfakeDetector
    from dfd.training.trainer import predict, resolve_device

    run = Path(run_dir)
    cfg = Config.from_yaml(run / "config.yaml")
    seed = cfg.train.seed
    if cfg.data.source == "hf":  # validation split only; the train split is not needed
        base_val = HFImageDataset(cfg.data, cfg.data.hf_val_split, max_samples=cfg.data.max_val_samples)
    else:
        base_val = build_base_datasets(cfg.data, seed)[1]
    test = eval_splits(cfg.data, base_val, seed)["test"]
    dev = resolve_device(device)
    model, _ = DeepfakeDetector.load(run / "best.pt")
    model.to(dev).eval()
    tds = TransformedDataset(test, DetectorTransform(cfg.data.pte, cfg.model.image_size, train=False), seed + 2,
                             cfg.data.video_frames_per_clip)
    p = predict(model, DataLoader(tds, batch_size=cfg.train.batch_size, num_workers=num_workers, shuffle=False), dev)
    return {k: p[k] for k in ("logits", "labels", "generators")}


@app.function(gpu=GPU, cpu=CPU, memory=MEM_MB, timeout=TIMEOUT_S,
              volumes={"/cache": hf_cache, "/runs": runs_vol, "/data": data_vol})
def run_rescore(run_dir: str, out_path: str) -> dict:
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    preds = rescore_test(run_dir)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds))
    runs_vol.commit()
    return {"run_dir": run_dir, "n": len(preds["labels"]), "auc": float(roc_auc_score(preds["labels"], preds["logits"])),
            "est_cost_usd": round((time.time() - t0) * PRICE_PER_S, 3)}


# ---------------------------------------------------------------- local analysis (numpy + sklearn only)

def tpr_at_fpr(labels, scores, fpr: float = 0.01) -> float:
    from dfd.eval.metrics import tpr_at_fpr as _t

    return float(_t(labels, scores, fpr))


def paired_bootstrap(baseline: dict, ablation: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-rotation and mean-over-rotation deltas (ablation - baseline) of AUC and TPR@1%FPR with percentile 95% CIs.

    baseline/ablation: {rotation: {"logits", "labels"}} on identical test images (same order). Each resample draws
    test images with replacement within each rotation, stratified by class, with the same indices for both models.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    rots = sorted(baseline)
    data = {}
    for r in rots:
        y = np.asarray(baseline[r]["labels"])
        if not np.array_equal(y, np.asarray(ablation[r]["labels"])):
            raise ValueError(f"{r}: test labels differ between runs; not the same test split")
        data[r] = (y, np.asarray(baseline[r]["logits"], float), np.asarray(ablation[r]["logits"], float),
                   np.flatnonzero(y == 0), np.flatnonzero(y == 1))

    def deltas(idx_by_rot):
        out = {}
        for r, idx in idx_by_rot.items():
            y, a, b = data[r][0][idx], data[r][1][idx], data[r][2][idx]
            out[r] = (roc_auc_score(y, b) - roc_auc_score(y, a), tpr_at_fpr(y, b) - tpr_at_fpr(y, a))
        return out

    point = deltas({r: np.arange(len(data[r][0])) for r in rots})
    boots = {r: [] for r in rots}
    for _ in range(n_boot):
        idx = {r: np.concatenate([rng.choice(d[3], len(d[3])), rng.choice(d[4], len(d[4]))]) for r, d in data.items()}
        for r, v in deltas(idx).items():
            boots[r].append(v)
    arr = {r: np.asarray(v) for r, v in boots.items()}
    mean_boot = np.mean([arr[r] for r in rots], axis=0)

    def ci(x):
        return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]

    res = {"n_boot": n_boot, "seed": seed, "rotations": {}}
    for r in rots:
        y, a, b = data[r][:3]
        res["rotations"][r] = {"n": int(len(y)), "auc_baseline": float(roc_auc_score(y, a)),
                               "auc_ablation": float(roc_auc_score(y, b)), "d_auc": point[r][0],
                               "d_auc_ci": ci(arr[r][:, 0]), "d_tpr1": point[r][1], "d_tpr1_ci": ci(arr[r][:, 1])}
    res["mean"] = {"d_auc": float(np.mean([point[r][0] for r in rots])), "d_auc_ci": ci(mean_boot[:, 0]),
                   "d_tpr1": float(np.mean([point[r][1] for r in rots])), "d_tpr1_ci": ci(mean_boot[:, 1])}
    return res


def decide(boot: dict, ablation_fpr: dict[str, float], midjourney: str = "SD15-to-Midjourney",
           max_fpr: float = 0.02) -> dict:
    """The pre-registered criterion (runs/helper/TASK_midjourney_ablation.md)."""
    c1 = boot["mean"]["d_auc"] > 0 and boot["mean"]["d_auc_ci"][0] > 0
    mj = boot["rotations"][midjourney]
    c2 = mj["d_auc"] > 0 and mj["d_auc_ci"][0] > 0
    over = {r: f for r, f in ablation_fpr.items() if f is not None and f > max_fpr}
    c3 = not over
    return {"c1_mean_dauc_ci_excludes_0": c1, "c2_midjourney_dauc_ci_excludes_0": c2, "c3_fpr_ok": c3,
            "fpr_over_limit": over, "verdict": "SUCCESS" if c1 and c2 and c3 else "NEGATIVE/INCONCLUSIVE"}


def analyze(baseline_dir: str, ablation_dir: str, reports_dir: str, recorded_dir: str | None = None,
            n_boot: int = 2000) -> dict:
    """Dirs hold <rotation>/test_preds.json (baseline re-score, ablation run); reports_dir holds the ablation's
    modal_logo <rotation>.json (conformal test FPR); recorded_dir holds the baseline's <rotation>.json for the AUC
    sanity check (|re-scored - recorded| <= 0.002)."""
    load = lambda d, r: json.loads((Path(d) / r / "test_preds.json").read_text())  # noqa: E731
    base = {r: load(baseline_dir, r) for r in ROTATIONS}
    abl = {r: load(ablation_dir, r) for r in ROTATIONS}
    boot = paired_bootstrap(base, abl, n_boot)
    sanity = {}
    if recorded_dir:
        for r in ROTATIONS:
            rec = json.loads((Path(recorded_dir) / f"{r}.json").read_text())["report"]["eval_test"]["overall"]["auc"]
            sanity[r] = {"recorded": rec, "rescored": boot["rotations"][r]["auc_baseline"],
                         "ok": abs(rec - boot["rotations"][r]["auc_baseline"]) <= 0.002}
    fpr = {}
    for r in ROTATIONS:
        rep = json.loads((Path(reports_dir) / f"{r}.json").read_text())["report"]
        fpr[r] = rep["verdicts_heldout"]["fpr_fake_verdict"]
    out = {"bootstrap": boot, "ablation_conformal_test_fpr": fpr, "baseline_sanity": sanity,
           "comparison_valid": all(v["ok"] for v in sanity.values()) if sanity else None}
    out["decision"] = decide(boot, fpr) if out["comparison_valid"] is not False else {"verdict": "INVALID (sanity)"}
    return out


@app.local_entrypoint()
def main(source_tag: str = "full7_dinov2_conformal", tag: str = "full7_baseline_rescore"):
    if source_tag == tag:
        raise SystemExit("--tag must be a NEW directory")
    args = [(f"/runs/{source_tag}/{r}", f"/runs/{tag}/{r}/test_preds.json") for r in ROTATIONS]
    print(f"[rescore] {len(args)} rotations of /runs/{source_tag} -> /runs/{tag} (L4, eval only)")
    total, failed = 0.0, []
    for (run_dir, _), r in zip(args, run_rescore.starmap(args, return_exceptions=True)):
        if isinstance(r, BaseException):
            failed.append(run_dir)
            print(f"[rescore] {run_dir}: FAILED {type(r).__name__}: {r}")
            continue
        total += r["est_cost_usd"]
        print(f"[rescore] {run_dir}: n={r['n']} auc={r['auc']:.4f} ~${r['est_cost_usd']:.2f}")
    print(f"[rescore] total ~${total:.2f}")
    if failed:
        raise SystemExit(f"{len(failed)} rotation(s) failed")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "analyze":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--ablation", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--recorded", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    res = analyze(a.baseline, a.ablation, a.reports, a.recorded, a.n_boot)
    if a.output:
        Path(a.output).write_text(json.dumps(res, indent=2))
    print(json.dumps({"decision": res["decision"], "mean": res["bootstrap"]["mean"],
                      "sanity_ok": res["comparison_valid"]}, indent=1))
    for r, v in res["bootstrap"]["rotations"].items():
        print(f"{r:22s} AUC {v['auc_baseline']:.4f} -> {v['auc_ablation']:.4f}  dAUC {v['d_auc']:+.4f} "
              f"[{v['d_auc_ci'][0]:+.4f}, {v['d_auc_ci'][1]:+.4f}]  dTPR1 {v['d_tpr1']:+.3f} "
              f"FPR {res['ablation_conformal_test_fpr'][r]:.4f}")
