"""Tests for the script hooks: modal_logo _rotations and train-time extra-split evaluation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from dfd.config import Config, ModelConfig
from dfd.data.datasets import SyntheticDeepfakeDataset
from dfd.eval.calibration import calibrate
from dfd.models.detector import DeepfakeDetector

TINY = dict(backbone="random-tiny", feature_layers=(2, 4), lora_rank=2, lora_blocks=2, hidden_dim=32, latent_dim=8)
MODAL_LOGO = Path(__file__).resolve().parents[1] / "scripts" / "modal_logo.py"


def _modal_logo():
    spec = importlib.util.spec_from_file_location("modal_logo", MODAL_LOGO)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rotations_config_returns_single_placeholder():
    assert _modal_logo()._rotations("config") == [(None, None)]


def test_rotations_all_returns_seven_pairs():
    pairs = _modal_logo()._rotations("all")
    assert len(pairs) == 7
    assert all(sel != test for sel, test in pairs)


def test_rotations_bad_pair_exits():
    with pytest.raises(SystemExit):
        _modal_logo()._rotations("GLIDE:NotAGenerator")


def test_evaluate_extra_splits(monkeypatch):
    import dfd.eval.extra as extra

    base_val = SyntheticDeepfakeDataset(8)
    splits = {"all": base_val, "fit": base_val, "select": base_val, "test": base_val,
              "extra": SyntheticDeepfakeDataset(16)}
    monkeypatch.setattr(extra, "eval_splits", lambda cfg_data, val, seed: splits)

    cfg = Config(model=ModelConfig(**TINY))
    cfg.train.batch_size = 8
    model = DeepfakeDetector(cfg.model).eval()
    rng = np.random.default_rng(0)
    cals = {m: calibrate(rng.normal(size=64), np.tile([0, 1], 32), method=m) for m in ("conformal", "empirical")}

    out = extra.evaluate_extra_splits(model, cfg, base_val, cals, torch.device("cpu"),
                                      dict(batch_size=8, num_workers=0, persistent_workers=False))
    assert set(out) == {"extra"}
    entry = out["extra"]
    assert entry["report"]["overall"]["n"] == 16
    assert set(entry["verdicts_by_method"]) == {"conformal", "empirical"}
    for v in entry["verdicts_by_method"].values():
        assert v["split"] == "extra"
        assert 0.0 <= v["abstain_rate"] <= 1.0


def test_evaluate_extra_splits_empty(monkeypatch):
    import dfd.eval.extra as extra

    base_val = SyntheticDeepfakeDataset(8)
    splits = {"all": base_val, "fit": base_val, "select": base_val, "test": base_val}
    monkeypatch.setattr(extra, "eval_splits", lambda cfg_data, val, seed: splits)
    cfg = Config(model=ModelConfig(**TINY))
    assert extra.evaluate_extra_splits(DeepfakeDetector(cfg.model), cfg, base_val, {},
                                       torch.device("cpu"), dict(batch_size=8)) == {}
