"""Score a trained checkpoint on external eval datasets (never used for training/calibration).

    python scripts/eval_external.py --dataset wilddeepfake --checkpoint /runs/<run>/best.pt
    python scripts/eval_external.py --dataset sfhq_t2i --checkpoint best.pt --max-per-generator 2000
    python scripts/eval_external.py --dataset sfhq_t2i --checkpoint best.pt --sfhq-split val
    python scripts/eval_external.py --dataset fairface --checkpoint best.pt --max-items 2000

Reuses DetectionPipeline's model, eval transform, and calibration so preprocessing is identical to
deployment; nothing is fit or saved. WildDeepfake reports frame- and clip-level metrics plus calibrated
verdict rates (the real-image FPR is the key real-world check of the conformal bound). SFHQ-T2I is
fakes-only, so it reports recall/abstain/FNR rates per generator instead of AUC. FairFace is reals-only
(HF cache, no /data): it reports real-face false-alarm rates by race, gender and age, plus a fairness
block (FPR and review burden per group with Wilson CIs; see dfd.eval.fairness).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dfd.data.external import SFHQT2IDataset, WildDeepfakeDataset
from dfd.eval.fairness import fairness_report
from dfd.eval.metrics import binary_metrics
from dfd.inference.pipeline import DetectionPipeline


class _Transformed(Dataset):
    def __init__(self, base: Dataset, transform):
        self.base, self.transform = base, transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        img, label, gen = self.base[idx]
        pixels, _ = self.transform(img, np.random.default_rng(idx))
        return {"pixels": pixels, "label": label, "generator": gen, "idx": idx}


class FairFaceDataset(Dataset):
    """HuggingFaceM4/FairFace (config 1.25, loose crops), all real; per-row race/gender/age names. No alignment."""

    def __init__(self, split: str = "validation", max_items: int | None = None, seed: int = 0):
        from datasets import load_dataset

        ds = load_dataset("HuggingFaceM4/FairFace", "1.25", split=split)
        if max_items is not None and len(ds) > max_items:
            ds = ds.select(sorted(np.random.default_rng(seed).choice(len(ds), size=max_items, replace=False).tolist()))
        self.ds = ds
        attrs = ds.select_columns(["race", "gender", "age"])
        self.race, self.gender, self.age = ([ds.features[c].int2str(int(x)) for x in attrs[c]]
                                            for c in ("race", "gender", "age"))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple:
        return self.ds[idx]["image"].convert("RGB"), 0, "real"


def _verdicts(cal, logits, status: str) -> np.ndarray:
    if status == "matched":
        return np.array([cal.verdict(p) for p in cal.prob(np.asarray(logits))], dtype=object)
    # Thresholds were not fit on this score distribution: force abstain like DetectionPipeline._finalize.
    return np.array(["abstain"] * len(np.asarray(logits)), dtype=object)


def _verdict_rates(cal, logits, labels, status: str) -> dict:
    labels = np.asarray(labels)
    reals, fakes = labels == 0, labels == 1
    v = _verdicts(cal, logits, status)
    return {
        "n": int(len(labels)),
        "calibration_status": status,
        "verdicts_valid": status == "matched",
        "abstain_rate": float((v == "abstain").mean()) if len(v) else None,
        "fpr_fake_verdict": float((v[reals] == "fake").mean()) if reals.any() else None,
        "fnr_real_verdict": float((v[fakes] == "real").mean()) if fakes.any() else None,
        "frac_verdict_fake": float((v == "fake").mean()) if len(v) else None,
    }


@torch.no_grad()
def _predict(pipe, ds, batch_size: int, num_workers: int) -> dict:
    loader = DataLoader(_Transformed(ds, pipe.transform), batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    out = {"logits": [], "labels": [], "generators": [], "indices": []}
    for batch in loader:
        logits = pipe.model(batch["pixels"].to(pipe.device))["logit"].float().cpu()
        out["logits"] += logits.tolist()
        out["labels"] += batch["label"].tolist()
        out["generators"] += list(batch["generator"])
        out["indices"] += batch["idx"].tolist()
    return out


def _group_rates(verdicts: np.ndarray, probs: np.ndarray, groups: list[str]) -> dict:
    """Per-group {n, fpr_fake_verdict, abstain_rate, mean_p_fake} for all-real rows."""
    v, p, g = np.asarray(verdicts), np.asarray(probs, dtype=np.float64), np.asarray(groups)
    return {str(k): {"n": int((g == k).sum()), "fpr_fake_verdict": float((v[g == k] == "fake").mean()),
                     "abstain_rate": float((v[g == k] == "abstain").mean()), "mean_p_fake": float(p[g == k].mean())}
            for k in sorted(set(g.tolist()))}


def _eval_fairface(pipe, ds, batch_size: int, num_workers: int, min_n: int = 100) -> dict:
    preds = _predict(pipe, ds, batch_size, num_workers)
    logits, labels, idx = np.asarray(preds["logits"]), np.asarray(preds["labels"]), preds["indices"]
    v, probs = _verdicts(pipe.cal, logits, pipe.calibration_status), pipe.cal.prob(logits)
    groups = {a: [getattr(ds, a)[i] for i in idx] for a in ("race", "gender", "age")}
    report = {"overall": {**_verdict_rates(pipe.cal, logits, labels, pipe.calibration_status),
                          "mean_p_fake": float(probs.mean())},
              **{f"by_{a}": _group_rates(v, probs, g) for a, g in groups.items()}}
    report["fairness"] = fairness_report(v, groups, target_fpr=pipe.cal.target_fpr, min_n=min_n,
                                         verdicts_valid=pipe.calibration_status == "matched")
    return report


def _fmt_ci(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _print_fairness(fair: dict) -> None:
    if not fair["verdicts_valid"]:
        print("[fairness] WARNING: verdicts not valid (calibration not matched, all forced to abstain); "
              "FPR numbers below are NOT meaningful")
    for attr in ("race", "race|gender"):
        fpr = fair["attributes"][attr]["fpr"]
        s, t = fpr["summary"], fpr["table"]
        if s["max_group"] is None:
            print(f"[fairness] {attr}: no group with n >= {fair['min_n']}")
            continue
        w = t[s["max_group"]]
        print(f"[fairness] {attr}: worst FPR {s['max_group']} {w['rate']:.4f} [{w['ci_low']:.4f}, {w['ci_high']:.4f}] "
              f"n={w['n']} | gap={s['gap']:.4f} (EB {s['gap_eb']:.4f} CI={_fmt_ci(s['gap_eb_ci'])}) "
              f"| above target {fair['target_fpr']}: "
              f"Wilson {s['above_target_ci'] or 'none'}, Holm {s['above_target_holm'] or 'none'}")


class _ClipSequences(Dataset):
    """WildDeepfake frames grouped per clip into one [T,3,H,W] sequence (+ mask) for temporal checkpoints."""

    def __init__(self, base: Dataset, transform, frames_per_clip: int):
        self.base, self.transform, self.t = base, transform, frames_per_clip
        by_clip: dict[str, list[int]] = {}
        for i in range(len(base)):
            by_clip.setdefault(base.clip_id(i), []).append(i)
        self.clips = sorted(by_clip.items())

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict:
        clip_id, rows = self.clips[idx]
        rows = rows[: self.t]
        pixels = [self.transform(self.base[i][0], np.random.default_rng(idx))[0] for i in rows]
        mask = [True] * len(pixels) + [False] * (self.t - len(pixels))
        pixels += [pixels[-1]] * (self.t - len(pixels))
        return {"pixels": torch.stack(pixels), "mask": torch.tensor(mask), "label": self.base.meta(rows[0])[0],
                "clip_id": clip_id}


@torch.no_grad()
def _eval_wilddeepfake_temporal(pipe, ds, frames_per_clip: int, batch_size: int, num_workers: int) -> dict:
    """Temporal checkpoint: one pooled logit per clip; its calibration was fit on clip logits, so verdicts apply."""
    # batch_size counts frames, as on the image path, so clips per batch = batch_size // T.
    loader = DataLoader(_ClipSequences(ds, pipe.transform, frames_per_clip),
                        batch_size=max(1, batch_size // frames_per_clip), num_workers=num_workers, shuffle=False)
    logits, labels = [], []
    for batch in loader:
        out = pipe.model(batch["pixels"].to(pipe.device), batch["mask"].to(pipe.device))
        logits += out["logit"].float().cpu().tolist()
        labels += batch["label"].tolist()
    logits, labels = np.asarray(logits), np.asarray(labels)
    return {"frame": None,
            "clip": {**binary_metrics(labels, logits), "n_clips": len(labels),
                     "verdicts": _verdict_rates(pipe.cal, logits, labels, pipe.calibration_status),
                     "note": "temporal checkpoint: clips scored as T-frame sequences (top-k pooled logit)"}}


def _eval_wilddeepfake(pipe, ds, batch_size: int, num_workers: int) -> dict:
    preds = _predict(pipe, ds, batch_size, num_workers)
    logits, labels = np.asarray(preds["logits"]), np.asarray(preds["labels"])
    by_clip: dict[str, list[int]] = {}
    for pos, i in enumerate(preds["indices"]):
        by_clip.setdefault(ds.clip_id(i), []).append(pos)
    clip_logits = np.array([float(logits[idx].mean()) for _, idx in sorted(by_clip.items())])
    clip_labels = np.array([int(round(float(labels[idx].mean()))) for _, idx in sorted(by_clip.items())])
    return {
        "frame": {**binary_metrics(labels, logits),
                  "verdicts": _verdict_rates(pipe.cal, logits, labels, pipe.calibration_status)},
        "clip": {**binary_metrics(clip_labels, clip_logits), "n_clips": len(by_clip),
                 "verdicts": None,
                 "note": "clip-level verdicts omitted: conformal thresholds are calibrated on single-image scores"},
    }


def _eval_sfhq(pipe, ds, batch_size: int, num_workers: int) -> dict:
    preds = _predict(pipe, ds, batch_size, num_workers)
    logits, labels = np.asarray(preds["logits"]), np.asarray(preds["labels"])
    gens = np.asarray(preds["generators"])
    per_gen = {g: {"n": int((gens == g).sum()),
                   **_verdict_rates(pipe.cal, logits[gens == g], labels[gens == g], pipe.calibration_status),
                   "mean_p_fake": float(pipe.cal.prob(logits[gens == g]).mean())}
               for g in sorted(set(gens))}
    return {"overall": {"n": int(len(labels)),
                        **_verdict_rates(pipe.cal, logits, labels, pipe.calibration_status),
                        "mean_p_fake": float(pipe.cal.prob(logits).mean())},
            "by_generator": per_gen}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["wilddeepfake", "sfhq_t2i", "fairface"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--frames-per-clip", type=int, default=8)
    ap.add_argument("--max-per-generator", type=int, default=None)
    ap.add_argument("--sfhq-split", choices=["all", "train", "val"], default="all",
                    help="SFHQ-T2I md5 split (train/val match the face_t2i training split; all = whole zip)")
    ap.add_argument("--max-items", type=int, default=None, help="FairFace: seeded subset for quick runs")
    ap.add_argument("--fairness-min-n", type=int, default=100,
                    help="FairFace: groups with fewer rows are reported but left out of gap/ratio")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    pipe = DetectionPipeline(args.checkpoint, None, args.device, crop_mode="center")
    root = Path(args.data_root)
    if args.dataset == "wilddeepfake":
        ds = WildDeepfakeDataset(root / "wilddeepfake", frames_per_clip=args.frames_per_clip)
        if pipe.model.temporal is not None:
            report = _eval_wilddeepfake_temporal(pipe, ds, args.frames_per_clip, args.batch_size, args.num_workers)
        else:
            report = _eval_wilddeepfake(pipe, ds, args.batch_size, args.num_workers)
    elif args.dataset == "fairface":
        ds = FairFaceDataset("validation", max_items=args.max_items)
        report = _eval_fairface(pipe, ds, args.batch_size, args.num_workers, args.fairness_min_n)
    else:
        ds = SFHQT2IDataset(root / "sfhq_t2i" / "SyntheticFacesHighQuality-T2I.zip",
                            max_per_generator=args.max_per_generator,
                            split=None if args.sfhq_split == "all" else args.sfhq_split)
        report = _eval_sfhq(pipe, ds, args.batch_size, args.num_workers)

    report = {"dataset": args.dataset, "checkpoint": args.checkpoint, "n_items": len(ds),
              **({"sfhq_split": args.sfhq_split} if args.dataset == "sfhq_t2i" else {}),
              "calibration": {"method": pipe.cal.method, "status": pipe.calibration_status,
                              "crop_mode": pipe.cal.crop_mode,
                              "tau_fake": pipe.cal.tau_fake, "tau_real": pipe.cal.tau_real},
              **report}
    out = Path(args.output or Path(args.checkpoint).with_name(f"eval_external_{args.dataset}.json"))
    out.write_text(json.dumps(report, indent=2))

    if pipe.calibration_status != "matched":
        print(f"[eval_external] WARNING: calibration_status={pipe.calibration_status!r} — "
              "thresholds were not fit on this score distribution; all verdicts forced to abstain")
    if args.dataset == "wilddeepfake" and report["frame"] is None:
        co, cv = report["clip"], report["clip"]["verdicts"]
        print(f"[wilddeepfake] temporal clips n={co['n_clips']} auc={co['auc']} tpr@1%fpr={co['tpr@1%fpr']} "
              f"| real FPR(verdict=fake)={cv['fpr_fake_verdict']} abstain={cv['abstain_rate']}")
    elif args.dataset == "wilddeepfake":
        fv = report["frame"]["verdicts"]
        fo, co = report["frame"], report["clip"]
        print(f"[wilddeepfake] frames n={fo['n']} auc={fo['auc']} tpr@1%fpr={fo['tpr@1%fpr']} "
              f"| real FPR(verdict=fake)={fv['fpr_fake_verdict']} abstain={fv['abstain_rate']}")
        print(f"[wilddeepfake] clips  n={co['n_clips']} auc={co['auc']} tpr@1%fpr={co['tpr@1%fpr']} "
              "(uncalibrated discrimination; clip verdicts omitted)")
    elif args.dataset == "fairface":
        ov = report["overall"]
        print(f"[fairface] n={ov['n']} real FPR(verdict=fake)={ov['fpr_fake_verdict']} "
              f"abstain={ov['abstain_rate']} mean_p_fake={ov['mean_p_fake']:.4f}")
        for g, r in report["by_race"].items():
            print(f"  {g:16s} n={r['n']:5d} fpr={r['fpr_fake_verdict']:.4f} abstain={r['abstain_rate']:.4f} "
                  f"mean_p={r['mean_p_fake']:.4f}")
        _print_fairness(report["fairness"])
    else:
        ov = report["overall"]
        print(f"[sfhq_t2i] n={ov['n']} verdict=fake {ov['frac_verdict_fake']} abstain={ov['abstain_rate']} "
              f"verdict=real (FNR)={ov['fnr_real_verdict']} mean_p_fake={ov['mean_p_fake']:.4f}")
        for g, r in report["by_generator"].items():
            print(f"  {g:15s} n={r['n']} fake={r['frac_verdict_fake']} abstain={r['abstain_rate']} "
                  f"real={r['fnr_real_verdict']} mean_p={r['mean_p_fake']:.4f}")
    print(f"[eval_external] calibration={report['calibration']['method']}/"
          f"{report['calibration']['status']} -> {out}")


if __name__ == "__main__":
    main()
