"""Score a trained checkpoint on external eval datasets (never used for training/calibration).

    python scripts/eval_external.py --dataset wilddeepfake --checkpoint /runs/<run>/best.pt
    python scripts/eval_external.py --dataset sfhq_t2i --checkpoint best.pt --max-per-generator 2000
    python scripts/eval_external.py --dataset sfhq_t2i --checkpoint best.pt --sfhq-split val

Reuses DetectionPipeline's model, eval transform, and calibration so preprocessing is identical to
deployment; nothing is fit or saved. WildDeepfake reports frame- and clip-level metrics plus calibrated
verdict rates (the real-image FPR is the key real-world check of the conformal bound). SFHQ-T2I is
fakes-only, so it reports recall/abstain/FNR rates per generator instead of AUC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dfd.data.external import SFHQT2IDataset, WildDeepfakeDataset
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


def _verdict_rates(cal, logits, labels, status: str) -> dict:
    labels = np.asarray(labels)
    reals, fakes = labels == 0, labels == 1
    if status == "matched":
        v = np.array([cal.verdict(p) for p in cal.prob(np.asarray(logits))])
    else:
        # Thresholds were not fit on this score distribution: force abstain like DetectionPipeline._finalize.
        v = np.array(["abstain"] * len(labels))
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
    ap.add_argument("--dataset", required=True, choices=["wilddeepfake", "sfhq_t2i"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--frames-per-clip", type=int, default=8)
    ap.add_argument("--max-per-generator", type=int, default=None)
    ap.add_argument("--sfhq-split", choices=["all", "train", "val"], default="all",
                    help="SFHQ-T2I md5 split (train/val match the face_t2i training split; all = whole zip)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    pipe = DetectionPipeline(args.checkpoint, None, args.device, crop_mode="center")
    root = Path(args.data_root)
    if args.dataset == "wilddeepfake":
        ds = WildDeepfakeDataset(root / "wilddeepfake", frames_per_clip=args.frames_per_clip)
        report = _eval_wilddeepfake(pipe, ds, args.batch_size, args.num_workers)
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
    if args.dataset == "wilddeepfake":
        fv = report["frame"]["verdicts"]
        fo, co = report["frame"], report["clip"]
        print(f"[wilddeepfake] frames n={fo['n']} auc={fo['auc']} tpr@1%fpr={fo['tpr@1%fpr']} "
              f"| real FPR(verdict=fake)={fv['fpr_fake_verdict']} abstain={fv['abstain_rate']}")
        print(f"[wilddeepfake] clips  n={co['n_clips']} auc={co['auc']} tpr@1%fpr={co['tpr@1%fpr']} "
              "(uncalibrated discrimination; clip verdicts omitted)")
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
