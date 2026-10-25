"""Diagnose a collapsed video run (V2: constant output, val AUC 0.5) on Modal. Read-only on existing data.

    .venv/bin/modal run --detach scripts/modal_video_diag.py --run-dir /runs/video_v2/config --tag video_v2_diag
    .venv/bin/modal run scripts/modal_video_diag.py --mode index --tag video_v2_diag     # CPU: validate frame store

(a) frame sanity: decode random train clips from the frame tars; per-frame mean/std, within-clip and cross-clip
    pixel differences, label balance per split.
(b) the run's best.pt on the val clips: logit spread/AUC and the std across clips of each stage's features
    (backbone, temporal adapter output, VIB mu) to locate where the signal disappears.
(c) frozen-backbone linear probe: a fresh model (LoRA is zero-initialised, so this is the pretrained backbone)
    mean-pools frame features over each clip; logistic regression on ~1k train clips, AUC on the val clips. The
    same probe on the trained run's backbone shows whether LoRA training destroyed the features.
Writes <out_dir>/diag.json (new dir only) and a local copy under runs/modal/<tag>/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

PROJECT = Path(__file__).resolve().parents[1]
REMOTE = "/root/project"
GPU, CPU, MEM_MB, TIMEOUT_S = "L4", 8.0, 32768, 3600
PRICE_PER_S = 0.000222 + CPU * 0.0000131 + (MEM_MB / 1024) * 0.00000222  # same list prices as modal_logo.py

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
app = modal.App("dfd-video-diag", image=image)
hf_cache = modal.Volume.from_name("dfd-hf-cache", create_if_missing=True)
runs_vol = modal.Volume.from_name("dfd-runs", create_if_missing=True)
data_vol = modal.Volume.from_name("dfd-data", create_if_missing=True)


def _stats(x) -> dict:
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "max": float(x.max()), "q05": float(np.quantile(x, 0.05)), "q95": float(np.quantile(x, 0.95))}


def frame_sanity(ds, n_clips: int, seed: int) -> dict:
    """Pixel statistics of n random clips from a VideoClipDataset (raw stored frames, no transform)."""
    import pickle

    import numpy as np

    # Read through a pickled clone: file handles opened here must not be inherited by forked DataLoader workers
    # (a shared fd's seek position races across processes and returns garbage bytes).
    ds = pickle.loads(pickle.dumps(ds))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_clips, len(ds)), replace=False)
    clips, firsts = [], []
    for i in idx:
        frames, label, gen = ds[int(i)]
        arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
        within = np.abs(np.diff(arr, axis=0)).mean(axis=(1, 2, 3)) if len(arr) > 1 else np.zeros(1)
        clips.append({"clip_id": ds.clip_id(int(i)), "label": label, "n_frames": len(frames),
                      "size": list(frames[0].size), "frame_mean": float(arr.mean()),
                      "frame_std_min": float(arr.std(axis=(1, 2, 3)).min()),
                      "within_clip_absdiff_mean": float(within.mean()),
                      "identical_consecutive": int((within == 0).sum())})
        firsts.append(arr[0])
    cross = [float(np.abs(firsts[a] - firsts[b]).mean()) for a in range(len(firsts)) for b in range(a + 1, len(firsts))]
    return {"clips": clips, "cross_clip_absdiff": _stats(cross) if cross else None}


def label_counts(ds) -> dict:
    labels = [ds.meta(i)[0] for i in range(len(ds))]
    return {"n": len(labels), "n_fake": int(sum(labels)), "n_real": int(len(labels) - sum(labels))}


def clip_features(model, loader, device, amp: bool) -> dict:
    """Per clip: pooled logit, label, and masked means over frames of backbone / temporal / mu features."""
    import contextlib

    import numpy as np
    import torch

    from dfd.models.temporal import topk_pool

    model.eval()
    out = {"logit": [], "label": [], "backbone": [], "temporal": [], "mu": []}
    ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else contextlib.nullcontext
    with torch.no_grad():
        for batch in loader:
            px, mask = batch["pixels"].to(device), batch["mask"].to(device)
            b, t = px.shape[:2]
            with ctx():
                feats = model.backbone(px.flatten(0, 1)).view(b, t, -1).float()
                temporal = model.temporal(feats, mask).float() if model.temporal is not None else feats
                head = model.head(temporal.flatten(0, 1))
            w = mask.float().unsqueeze(-1)
            pool = lambda x: ((x * w).sum(1) / w.sum(1).clamp(min=1)).cpu().numpy()  # noqa: E731
            out["backbone"].append(pool(feats))
            out["temporal"].append(pool(temporal))
            out["mu"].append(pool(head["mu"].float().view(b, t, -1)))
            out["logit"] += topk_pool(head["logit"].float().view(b, t), mask, model.cfg.topk_frac).cpu().tolist()
            out["label"] += batch["label"].tolist()
    return {k: (np.concatenate(v) if k in ("backbone", "temporal", "mu") else np.asarray(v)) for k, v in out.items()}


def spread(f: dict) -> dict:
    """Std across clips (median over dims) per stage + logit stats + AUC."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels = f["label"]
    auc = float(roc_auc_score(labels, f["logit"])) if 0 < labels.sum() < len(labels) else None
    return {"logit": _stats(f["logit"]), "auc": auc, "n_unique_logits_1e4": int(len(np.unique(np.round(f["logit"], 4)))),
            **{f"{k}_std_median": float(np.median(f[k].std(axis=0))) for k in ("backbone", "temporal", "mu")},
            **{f"{k}_std_max": float(f[k].std(axis=0).max()) for k in ("backbone", "temporal", "mu")}}


def linear_probe(train_feats, train_labels, val_feats, val_labels) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    clf.fit(train_feats, train_labels)
    s_tr, s_va = clf.decision_function(train_feats), clf.decision_function(val_feats)
    return {"train_auc": float(roc_auc_score(train_labels, s_tr)), "val_auc": float(roc_auc_score(val_labels, s_va)),
            "n_train": int(len(train_labels)), "n_val": int(len(val_labels)), "val_score": _stats(s_va),
            "train_label_mean": float(np.mean(train_labels))}


def diagnose(run_dir: str, frames_root: str | None = None, n_sanity: int = 16, n_probe: int = 1000,
             batch_size: int = 8, num_workers: int = 6, device: str = "auto") -> dict:
    """Run (a)-(c) for the run in run_dir (config.yaml + best.pt). Pure local function; used by the Modal job."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from dfd.config import Config
    from dfd.data.datasets import SubsetDataset, TransformedDataset, build_base_datasets
    from dfd.data.transforms import DetectorTransform
    from dfd.models.detector import DeepfakeDetector
    from dfd.training.trainer import resolve_device

    t0 = time.time()
    run = Path(run_dir)
    cfg = Config.from_yaml(run / "config.yaml")
    if frames_root:
        cfg.data.video_frames_root = frames_root
    dev = resolve_device(device)
    amp = dev.type == "cuda"
    seed, size, fpc = cfg.train.seed, cfg.model.image_size, cfg.data.video_frames_per_clip
    base_train, base_val = build_base_datasets(cfg.data, seed)
    report: dict = {"run_dir": str(run), "train": label_counts(base_train), "val": label_counts(base_val),
                    "train_cfg": {k: getattr(cfg.train, k) for k in ("lr", "lora_lr", "batch_size", "epochs",
                                                                     "beta_max", "weight_decay", "grad_clip")}}
    report["a_frame_sanity"] = {"train": frame_sanity(base_train, n_sanity, seed),
                                "val": frame_sanity(base_val, min(n_sanity, 8), seed + 1)}

    tf = DetectorTransform(cfg.data.pte, size, train=False)
    kw = dict(batch_size=batch_size, num_workers=num_workers, shuffle=False)
    probe_idx = sorted(np.random.default_rng(seed).choice(len(base_train), size=min(n_probe, len(base_train)),
                                                          replace=False).tolist())
    train_loader = DataLoader(TransformedDataset(SubsetDataset(base_train, probe_idx), tf, seed + 5, fpc), **kw)
    val_loader = DataLoader(TransformedDataset(base_val, tf, seed + 1, fpc), **kw)

    trained, _ = DeepfakeDetector.load(run / "best.pt")
    trained.to(dev)
    sd = trained.state_dict()
    report["b_weights"] = {"nonfinite_tensors": [k for k, v in sd.items() if v.is_floating_point()
                                                 and not torch.isfinite(v).all()][:10],
                           **{k: float(sd[k].float().norm()) for k in sd if k.startswith(("head.", "temporal.mix"))
                              and k.endswith("weight")}}
    tv = clip_features(trained, val_loader, dev, amp)
    report["b_trained_val"] = spread(tv)
    tt = clip_features(trained, train_loader, dev, amp)
    report["c_probe_trained_backbone"] = linear_probe(tt["backbone"], tt["label"], tv["backbone"], tv["label"])
    del trained

    fresh = DeepfakeDetector(cfg.model).to(dev)  # pretrained backbone, LoRA B=0 -> identical to frozen
    fv = clip_features(fresh, val_loader, dev, amp)
    ft = clip_features(fresh, train_loader, dev, amp)
    report["b_fresh_val"] = spread(fv)
    report["c_probe_frozen_backbone"] = linear_probe(ft["backbone"], ft["label"], fv["backbone"], fv["label"])
    report["wall_s"] = round(time.time() - t0, 1)
    return report


def check_index(root: str) -> dict:
    """Validate <root>/index.json against the tars: JPEG magic at every indexed offset, offsets inside the file,
    and for bad entries the member's real offset from tarfile. Also cross-checks clips.json against the index."""
    import os
    import tarfile

    root_p = Path(root)
    entries = json.loads((root_p / "index.json").read_text())
    clips = json.loads((root_p / "clips.json").read_text())
    by_tar: dict[str, list[dict]] = {}
    for e in entries:
        by_tar.setdefault(e["tar"], []).append(e)
    out: dict = {"n_entries": len(entries), "n_clips": len(clips), "files": {}, "tars": {}}
    for name in ("index.json", "clips.json", *sorted(by_tar)):
        st = os.stat(root_p / name)
        out["files"][name] = {"size": st.st_size, "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(st.st_mtime))}
    for tar, es in sorted(by_tar.items()):
        size = os.path.getsize(root_p / tar)
        bad = []
        with open(root_p / tar, "rb") as f:
            for e in es:
                if e["offset"] + e["size"] > size:
                    bad.append({**e, "why": "past_eof"})
                    continue
                f.seek(e["offset"])
                if f.read(3) != b"\xff\xd8\xff":
                    bad.append({**e, "why": "no_jpeg_magic"})
        info = {"n": len(es), "n_bad": len(bad), "size": size, "bad_examples": bad[:5]}
        if bad:
            want = {b["member"] for b in bad[:5]}
            with tarfile.open(root_p / tar, "r:") as tf:
                info["real_offsets"] = {m.name: [m.offset_data, m.size] for m in tf if m.name in want}
        out["tars"][tar] = info
    keys = {(e["tar"], e["member"].rsplit("/", 1)[0]) for e in entries}
    tar_of = lambda c: c.get("tar") or f"{c['source']}_{c['split']}.tar"  # noqa: E731 (= dfd.data.video.clip_tar)
    missing = [c["clip_id"] for c in clips if (tar_of(c), c["clip_id"]) not in keys]
    out["clips_missing_frames"] = {"n": len(missing), "examples": missing[:5]}
    out["clips_by_source_split"] = {}
    for c in clips:
        k = f"{c['source']}_{c['split']}"
        out["clips_by_source_split"][k] = out["clips_by_source_split"].get(k, 0) + 1
    return out


@app.function(cpu=2.0, memory=4096, timeout=1800, volumes={"/data": data_vol})
def run_check_index(root: str) -> dict:
    return check_index(root)


@app.function(gpu=GPU, cpu=CPU, memory=MEM_MB, timeout=TIMEOUT_S,
              volumes={"/cache": hf_cache, "/runs": runs_vol, "/data": data_vol})
def run_diag(run_dir: str, out_dir: str, n_probe: int) -> dict:
    t0 = time.time()
    report = diagnose(run_dir, n_probe=n_probe)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "diag.json").write_text(json.dumps(report, indent=2))
    runs_vol.commit()
    report["est_cost_usd"] = round((time.time() - t0) * PRICE_PER_S, 3)
    return report


@app.local_entrypoint()
def main(run_dir: str = "/runs/video_v2/config", tag: str = "video_v2_diag", n_probe: int = 1000,
         mode: str = "diag"):
    if mode == "index":  # CPU-only validation of the frame store (read-only)
        r = run_check_index.remote("/data/video_frames")
        local = PROJECT / "runs" / "modal" / tag
        local.mkdir(parents=True, exist_ok=True)
        (local / "index_check.json").write_text(json.dumps(r, indent=2))
        print(json.dumps({k: r[k] for k in ("n_entries", "n_clips", "files", "clips_missing_frames")}, indent=1))
        print({t: (v["n"], v["n_bad"]) for t, v in r["tars"].items()})
        print(f"[index] -> {local / 'index_check.json'}")
        return
    out_dir = f"/runs/{tag}"
    if out_dir.rstrip("/") == str(Path(run_dir).parent).rstrip("/"):
        raise SystemExit("--tag must name a NEW directory, not the run's own")
    print(f"[diag] run_dir={run_dir} -> {out_dir}/diag.json (L4, timeout {TIMEOUT_S // 60} min, "
          f"~${PRICE_PER_S * 3600:.2f}/h list)")
    r = run_diag.remote(run_dir, out_dir, n_probe)
    local = PROJECT / "runs" / "modal" / tag
    local.mkdir(parents=True, exist_ok=True)
    (local / "diag.json").write_text(json.dumps(r, indent=2))
    print(json.dumps({k: r[k] for k in ("train", "val", "b_trained_val", "b_fresh_val", "c_probe_trained_backbone",
                                        "c_probe_frozen_backbone", "wall_s", "est_cost_usd")}, indent=2))
    print(f"[diag] -> {local / 'diag.json'}")
