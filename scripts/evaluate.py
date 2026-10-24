"""Evaluate a checkpoint under fixed PTE severity presets, centre-crop vs. multi-crop.

    python scripts/evaluate.py --checkpoint runs/smoke/best.pt --config configs/smoke.yaml
    python scripts/evaluate.py --checkpoint runs/x/best.pt --config cfg.yaml --split test --crop-mode both
    python scripts/evaluate.py --checkpoint runs/x/best.pt --config cfg.yaml --split fit --crop-mode multi \
        --severities light --save-calibration    # writes calibration_multicrop.json next to the checkpoint

Splits follow scripts/train.py exactly (same seed), so `test` is the untouched final-generator split and
`fit` contains only known generators. Calibration may only be fit on `fit` (or `select` without nesting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from dfd.config import Config
from dfd.env import load_dotenv
from dfd.data.datasets import TransformedDataset, build_base_datasets, eval_splits
from dfd.data.pte import SEVERITY_PRESETS
from dfd.data.transforms import DetectorTransform
from dfd.eval.calibration import calibrate
from dfd.eval.metrics import report_to_markdown, sliced_report
from dfd.inference.crops import predict_multicrop
from dfd.models.detector import DeepfakeDetector
from dfd.training.trainer import predict, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, help="Config providing the data section")
    ap.add_argument("--severities", nargs="+", default=list(SEVERITY_PRESETS),
                    help=f"Any of {list(SEVERITY_PRESETS)} or 'train_mix' (the config's training PTE mix)")
    ap.add_argument("--split", choices=["all", "fit", "select", "test"], default="all")
    ap.add_argument("--crop-mode", choices=["center", "multi", "both"], default="center")
    ap.add_argument("--save-calibration", action="store_true",
                    help="Fit calibration_multicrop.json (requires --split fit, --crop-mode multi, one severity)")
    ap.add_argument("--save-preds", action="store_true", help="Also write per-sample logits (<output>_preds.jsonl)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.save_calibration and (args.split != "fit" or args.crop_mode != "multi" or len(args.severities) != 1):
        raise SystemExit("--save-calibration needs --split fit --crop-mode multi and exactly one severity")

    load_dotenv()
    cfg = Config.from_yaml(args.config)
    presets = {**SEVERITY_PRESETS, "train_mix": cfg.data.pte}
    unknown = set(args.severities) - set(presets)
    if unknown:
        raise SystemExit(f"Unknown severities: {sorted(unknown)}")
    device = resolve_device(cfg.train.device)
    model, _ = DeepfakeDetector.load(args.checkpoint)
    model.to(device).eval()
    _, base_val = build_base_datasets(cfg.data, cfg.train.seed)
    base = eval_splits(cfg.data, base_val, cfg.train.seed)[args.split]
    if len(base) == 0:
        raise SystemExit(f"Split {args.split!r} is empty for this config")
    modes = ["center", "multi"] if args.crop_mode == "both" else [args.crop_mode]

    out = Path(args.output or Path(args.checkpoint).with_name(f"eval_report_{args.split}"))
    preds_path = out.with_name(out.name + "_preds.jsonl")
    if args.save_preds:
        preds_path.write_text("")
    report, last_preds = {}, None
    for sev in args.severities:
        for mode in modes:
            if mode == "center":
                ds = TransformedDataset(base, DetectorTransform(presets[sev], model.cfg.image_size, False), 1234)
                preds = predict(model, DataLoader(ds, batch_size=cfg.train.batch_size), device)
            else:
                preds = predict_multicrop(model, base, presets[sev], device, seed=1234)
            key = f"eval_{sev}_{mode}"
            report[key] = sliced_report(preds["labels"], preds["logits"], preds["generators"], preds["severities"])
            if mode == "multi":
                report[key]["mean_crops"] = sum(preds["n_crops"]) / len(preds["n_crops"])
            ov = report[key]["overall"]
            print(f"[{args.split}|{sev}|{mode}] n={ov['n']} auc={ov['auc']} tpr@1%fpr={ov['tpr@1%fpr']}")
            last_preds = preds
            if args.save_preds:
                n_crops = preds.get("n_crops", [1] * len(preds["logits"]))
                with preds_path.open("a") as f:
                    for i, (lg, y, g, nc) in enumerate(zip(preds["logits"], preds["labels"], preds["generators"], n_crops)):
                        f.write(json.dumps({"severity": sev, "mode": mode, "idx": i, "logit": lg,
                                            "label": y, "generator": g, "n_crops": nc}) + "\n")

    if args.save_calibration:
        cal = calibrate(last_preds["logits"], last_preds["labels"], cfg.train.target_fpr, cfg.train.target_fnr)
        cal_path = Path(args.checkpoint).with_name("calibration_multicrop.json")
        cal.save(cal_path)
        print(f"[calibration] wrote {cal_path} (fit split, {args.severities[0]}, multi-crop)")

    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    md = report_to_markdown({"overall": {}, **report}, f"Severity sweep ({args.split}): {args.checkpoint}")
    out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
