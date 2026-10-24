"""Score a trained checkpoint on Deepfake-Eval-2024 images (in-the-wild, EVALUATION ONLY; gated dataset).

    .venv/bin/python scripts/eval_deepfake_eval_2024.py --checkpoint runs/<run>/best.pt [--limit 200]

Uses the checkpoint's saved calibration (conformal by default) unchanged — nothing is fit on this data — and reports
AUC / TPR@1%FPR plus verdict rates. The FPR on real in-the-wild images is the key check: the conformal bound only
holds if deployment reals resemble the (transcoding-emulated) calibration reals, and this measures whether they do.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dfd.data.deepfake_eval import download, parse_metadata
from dfd.env import load_dotenv
from dfd.eval.metrics import binary_metrics
from dfd.inference.pipeline import DetectionPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--limit", type=int, default=0, help="Score at most N images (0 = all), seeded sample")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    load_dotenv()

    csv_path, data_dir = download("image", token=os.environ.get("HF_TOKEN"))
    items, stats = parse_metadata(csv_path, data_dir)
    print(f"[deepfake-eval-2024] {stats['usable']} usable images ({stats['n_fake']} fake / {stats['n_real']} real); "
          f"skipped {stats['unknown_label']} unknown-label, {stats['missing_file']} missing; labels {stats['label_values']}")
    if args.limit and len(items) > args.limit:
        keep = np.random.default_rng(0).choice(len(items), args.limit, replace=False)
        items = [items[i] for i in sorted(keep)]

    pipe = DetectionPipeline(args.checkpoint, args.calibration, args.device, crop_mode="center")
    logits, labels, verdicts, failures = [], [], [], 0
    for path, y in items:
        try:
            r = pipe.predict_image(path)
        except Exception as e:  # corrupt / unsupported files in a scraped set: count, don't crash
            failures += 1
            print(f"  ! {path.name}: {type(e).__name__}")
            continue
        if r.logit is None:  # stage0 hash hit: no score to evaluate
            continue
        logits.append(r.logit); labels.append(y); verdicts.append(r.verdict)
    y, v = np.array(labels), np.array(verdicts)
    report = {
        "dataset": "Deepfake-Eval-2024 (images, eval-only)", "checkpoint": args.checkpoint,
        "calibration": pipe.cal.__dict__, "calibration_status": pipe.calibration_status,
        "metadata": {k: stats[k] for k in ("rows", "usable", "n_fake", "n_real", "unknown_label", "missing_file",
                                            "filename_column", "label_column")},
        "scored": len(y), "decode_failures": failures,
        "metrics": binary_metrics(y, logits),
        "verdicts": {"fpr_fake_verdict": float((v[y == 0] == "fake").mean()) if (y == 0).any() else None,
                     "fnr_real_verdict": float((v[y == 1] == "real").mean()) if (y == 1).any() else None,
                     "abstain_rate": float((v == "abstain").mean()) if len(v) else None},
    }
    m, vr = report["metrics"], report["verdicts"]
    print(f"[deepfake-eval-2024] n={len(y)} AUC={m['auc']} TPR@1%FPR={m['tpr@1%fpr']} | verdicts FPR={vr['fpr_fake_verdict']} "
          f"FNR={vr['fnr_real_verdict']} abstain={vr['abstain_rate']} (calibration {pipe.cal.method}, {pipe.calibration_status})")
    out = Path(args.output or Path(args.checkpoint).with_name("deepfake_eval_2024_report.json"))
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
