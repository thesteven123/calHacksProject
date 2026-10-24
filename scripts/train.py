"""Train the Stage 2 detector, calibrate it, and write a report.

    python scripts/train.py --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from torch.utils.data import DataLoader

from dfd.config import Config
from dfd.env import load_dotenv
from dfd.data.datasets import TransformedDataset, build_base_datasets, split_nested_holdouts, split_val_holdout
from dfd.data.transforms import DetectorTransform
from dfd.eval.calibration import calibrate
from dfd.eval.metrics import report_to_markdown
from dfd.models.detector import DeepfakeDetector
from dfd.training.trainer import Trainer, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    load_dotenv()  # HF_TOKEN for gated backbones/datasets
    cfg = Config.from_yaml(args.config)
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    out = Path(cfg.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg.to_dict()))))
    seed_everything(cfg.train.seed)

    base_train, base_val = build_base_datasets(cfg.data, cfg.train.seed)
    holdout = tuple(cfg.data.holdout_generators)
    selection = tuple(cfg.data.selection_generators)
    if selection:
        base_fit, base_sel, base_test = split_nested_holdouts(base_val, selection, holdout, cfg.train.seed)
    else:
        base_sel, base_test = split_val_holdout(base_val, holdout, seed=cfg.train.seed)
        base_fit = base_sel
    size = cfg.model.image_size
    train_ds = TransformedDataset(base_train, DetectorTransform(cfg.data.pte, size, train=True), cfg.train.seed)
    val_ds = TransformedDataset(base_sel, DetectorTransform(cfg.data.pte, size, train=False), cfg.train.seed + 1)
    fit_ds = TransformedDataset(base_fit, DetectorTransform(cfg.data.pte, size, train=False), cfg.train.seed + 1)
    test_ds = TransformedDataset(base_test, DetectorTransform(cfg.data.pte, size, train=False), cfg.train.seed + 2)
    loader_kw = dict(batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
                     persistent_workers=cfg.data.num_workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > cfg.train.batch_size, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    fit_loader = DataLoader(fit_ds, shuffle=False, **loader_kw) if selection else None
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw) if len(test_ds) else None
    print(f"[data] train={len(train_ds)} fit={len(fit_ds)} select={len(val_ds)} test={len(test_ds)} "
          f"selection={selection} holdout={holdout}")

    model = DeepfakeDetector(cfg.model)
    best = Trainer(model, cfg, train_loader, val_loader, test_loader, fit_loader).train()

    # Calibrate on known generators, not either OOD generator.
    fit_preds = best.get("fit_preds", best["preds"])
    logits = np.asarray(fit_preds["logits"])
    labels = np.asarray(fit_preds["labels"])
    cal = calibrate(logits, labels, cfg.train.target_fpr, cfg.train.target_fnr)
    cal.save(out / "calibration.json")
    test_logits = np.asarray(best["test_preds"]["logits"]) if "test_preds" in best else logits
    held = np.asarray(best["test_preds"]["labels"]) if "test_preds" in best else labels
    probs = cal.prob(test_logits)
    verdicts = np.array([cal.verdict(p) for p in probs])
    reals, fakes = held == 0, held == 1
    best["report"]["verdicts_heldout"] = {
        "split": "test" if "test_preds" in best else "selection",
        "abstain_rate": float((verdicts == "abstain").mean()),
        "fpr_fake_verdict": float((verdicts[reals] == "fake").mean()) if reals.any() else None,
        "fnr_real_verdict": float((verdicts[fakes] == "real").mean()) if fakes.any() else None,
        "calibration": cal.__dict__,
    }
    if "test_report" in best:
        best["report"]["eval_test"] = best["test_report"]
    (out / "report.json").write_text(json.dumps(best["report"], indent=2))
    md = report_to_markdown(best["report"], f"Validation report: {out.name}")
    md += "\n## Verdicts (final test)\n```json\n" + json.dumps(best["report"]["verdicts_heldout"], indent=2) + "\n```\n"
    (out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
