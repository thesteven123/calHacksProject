"""Offline tests for scripts/modal_ablation.py: exact test-split re-scoring, paired bootstrap, pre-registered decision."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from test_external_data import _script

TINY = dict(backbone="random-tiny", feature_layers=[2, 4], lora_rank=2, lora_blocks=2, hidden_dim=32, latent_dim=8,
            image_size=56)
ROOT = Path(__file__).resolve().parents[1]


def test_rescore_reproduces_train_test_preds(tmp_path):
    cfg = {"model": {**TINY, "patch_pool": "attn_mil"},
           "data": {"source": "synthetic", "synthetic_train_size": 32, "synthetic_val_size": 48,
                    "synthetic_image_size": 64,
                    "holdout_generators": ["synth_gan"], "pte": {"enabled": True}},
           "train": {"epochs": 1, "batch_size": 8, "device": "cpu", "output_dir": str(tmp_path / "run")}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(tmp_path / "cfg.yaml")],
                   check=True, capture_output=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    recorded = json.loads((tmp_path / "run" / "test_preds.json").read_text())
    rescored = _script("modal_ablation").rescore_test(str(tmp_path / "run"), num_workers=0, device="cpu")
    assert rescored["labels"] == recorded["labels"] and set(recorded["generators"]) == {"real", "synth_gan"}
    assert np.allclose(rescored["logits"], recorded["logits"], atol=1e-5)


def _rot(rng, n, shift):
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    base = rng.normal(0, 1, 2 * n) + y * 1.0
    return y, base, base + y * shift


def test_paired_bootstrap_and_decision():
    mod = _script("modal_ablation")
    rng = np.random.default_rng(0)
    base, better, same = {}, {}, {}
    for r in mod.ROTATIONS:
        y, a, b = _rot(rng, 300, 0.8)
        base[r] = {"labels": y.tolist(), "logits": a.tolist()}
        better[r] = {"labels": y.tolist(), "logits": b.tolist()}
        same[r] = {"labels": y.tolist(), "logits": a.tolist()}
    up = mod.paired_bootstrap(base, better, n_boot=200)
    assert up["mean"]["d_auc"] > 0 and up["mean"]["d_auc_ci"][0] > 0
    fpr_ok = {r: 0.01 for r in mod.ROTATIONS}
    assert mod.decide(up, fpr_ok)["verdict"] == "SUCCESS"
    assert mod.decide(up, {**fpr_ok, "ADM-to-Wukong": 0.03})["verdict"] == "NEGATIVE/INCONCLUSIVE"
    flat = mod.paired_bootstrap(base, same, n_boot=200)
    assert flat["mean"]["d_auc"] == 0 and mod.decide(flat, fpr_ok)["verdict"] == "NEGATIVE/INCONCLUSIVE"
