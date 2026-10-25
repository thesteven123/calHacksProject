"""Evaluate extra named validation splits (beyond all/fit/select/test) after training."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

from dfd.data.datasets import TransformedDataset, eval_splits
from dfd.data.transforms import DetectorTransform
from dfd.eval.metrics import sliced_report
from dfd.training.trainer import predict

_CORE_SPLITS = ("all", "fit", "select", "test")


def verdict_rates(cal, logits, labels, split_name: str) -> dict:
    labels = np.asarray(labels)
    reals, fakes = labels == 0, labels == 1
    v = np.array([cal.verdict(p) for p in cal.prob(np.asarray(logits))])
    return {
        "split": split_name,
        "abstain_rate": float((v == "abstain").mean()),
        "fpr_fake_verdict": float((v[reals] == "fake").mean()) if reals.any() else None,
        "fnr_real_verdict": float((v[fakes] == "real").mean()) if fakes.any() else None,
        "calibration": cal.__dict__,
    }


def extra_eval_splits(cfg_data, base_val: Dataset, seed: int) -> dict[str, Dataset]:
    """Validation splits returned by eval_splits beyond the core ones."""
    return {k: v for k, v in eval_splits(cfg_data, base_val, seed).items()
            if k not in _CORE_SPLITS and len(v)}


def evaluate_extra_splits(model, cfg, base_val: Dataset, cals: dict, device,
                          loader_kw: dict) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, ds in extra_eval_splits(cfg.data, base_val, cfg.train.seed).items():
        tds = TransformedDataset(ds, DetectorTransform(cfg.data.pte, cfg.model.image_size, train=False),
                                 cfg.train.seed + 3, cfg.data.video_frames_per_clip)
        preds = predict(model, DataLoader(tds, shuffle=False, **loader_kw), device)
        out[name] = {
            "report": sliced_report(preds["labels"], preds["logits"], preds["generators"], preds["severities"]),
            "verdicts_by_method": {m: verdict_rates(c, preds["logits"], preds["labels"], name)
                                   for m, c in cals.items()},
        }
    return out
