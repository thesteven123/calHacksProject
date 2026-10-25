"""Offline tests for the real-manifold OOD gate (dfd.eval.ood) and scripts/ood_gate_eval.py."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from torch.utils.data import DataLoader

from dfd.config import Config, DataConfig, ModelConfig, PTEConfig, TrainConfig
from dfd.data.datasets import SyntheticDeepfakeDataset, TransformedDataset
from dfd.data.transforms import DetectorTransform
from dfd.eval.calibration import Calibration
from dfd.eval.ood import OODGate, fit_ood_gate, gated_verdicts, predict_with_latents
from dfd.models.detector import DeepfakeDetector
from dfd.training.trainer import resolve_device

TINY = dict(backbone="random-tiny", feature_layers=(2, 4), lora_rank=2, lora_blocks=2, hidden_dim=32, latent_dim=8)
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ood_gate_eval.py"


def _script():
    spec = importlib.util.spec_from_file_location("ood_gate_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("alpha", [0.05, 0.10])
def test_gate_rates_in_and_out_of_distribution(alpha):
    rng = np.random.default_rng(0)
    gate = fit_ood_gate(rng.normal(size=(2000, 128)), alpha, seed=0)
    assert gate.n_fit + gate.n_cal == 2000 and math.isfinite(gate.gamma)
    assert (gate.distance(rng.normal(size=(4000, 128))) > gate.gamma).mean() <= alpha + 0.03
    assert (gate.distance(rng.normal(4.0, 1.0, size=(1000, 128))) > gate.gamma).mean() > 0.95


def test_tiny_cal_gives_inf_gamma_and_roundtrips(tmp_path):
    rng = np.random.default_rng(1)
    gate = fit_ood_gate(rng.normal(size=(20, 4)), alpha=0.05)  # n_cal=10 -> ceil(11*0.95)=11 > 10
    assert gate.gamma == math.inf
    cal = Calibration(temperature=1.0, tau_real=0.4, tau_fake=0.6)
    v = gated_verdicts(cal, np.full(5, -5.0), rng.normal(10.0, 1.0, size=(5, 4)), gate)
    assert (v == "real").all()
    gate.save(tmp_path / "g.json")
    assert json.loads((tmp_path / "g.json").read_text())["gamma"] is None
    loaded = OODGate.load(tmp_path / "g.json")
    assert loaded.gamma == math.inf and loaded.n_cal == gate.n_cal
    np.testing.assert_allclose(loaded.distance(np.ones((2, 4))), gate.distance(np.ones((2, 4))))


def test_gated_verdicts_only_turns_real_into_abstain():
    cal = Calibration(temperature=1.0, tau_real=0.3, tau_fake=0.7)
    logits = np.array([-3.0, -3.0, 0.0, 0.0, 3.0, 3.0])  # real, real, abstain, abstain, fake, fake
    gate = OODGate(mean=[0.0, 0.0], precision=[[1.0, 0.0], [0.0, 1.0]], gamma=1.0, alpha=0.05, n_fit=10, n_cal=10)
    mu = np.array([[0.1, 0.0], [5.0, 0.0]] * 3)  # alternate in / out of distribution
    base = gated_verdicts(cal, logits, mu, None)
    assert base.tolist() == ["real", "real", "abstain", "abstain", "fake", "fake"]
    assert gated_verdicts(cal, logits, mu, gate).tolist() == ["real", "abstain", "abstain", "abstain", "fake", "fake"]


def test_too_few_reals_raises():
    with pytest.raises(ValueError):
        fit_ood_gate(np.zeros((19, 8)))


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.05, 1.5, float("nan"), float("inf")])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError, match="alpha"):
        fit_ood_gate(np.random.default_rng(0).normal(size=(40, 4)), alpha)


@pytest.mark.parametrize("alphas", ["0", "1", "-0.1", "0.05,2", "nan", ","])
def test_cli_rejects_invalid_alphas(alphas, tmp_path):
    with pytest.raises(SystemExit):
        _script().main(["--checkpoint", str(tmp_path / "best.pt"), "--alphas", alphas, "--skip-external"])


def test_predict_with_latents_shapes():
    model = DeepfakeDetector(ModelConfig(**TINY, image_size=56))
    ds = TransformedDataset(SyntheticDeepfakeDataset(6, image_size=64),
                            DetectorTransform(PTEConfig(enabled=False), 56, train=False))
    preds = predict_with_latents(model, DataLoader(ds, batch_size=4), resolve_device("cpu"))
    assert np.asarray(preds["mu"]).shape == (6, TINY["latent_dim"])
    assert len(preds["logits"]) == len(preds["labels"]) == len(preds["generators"]) == 6


def test_script_smoke(tmp_path):
    cfg = Config(model=ModelConfig(**TINY, image_size=56),
                 data=DataConfig(source="synthetic", synthetic_val_size=160, synthetic_image_size=64,
                                 holdout_generators=("synth_gan",)),
                 train=TrainConfig(device="cpu"))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(cfg.to_dict()))))
    DeepfakeDetector(cfg.model).save(tmp_path / "best.pt")
    Calibration(temperature=1.0, tau_real=0.6, tau_fake=0.99, method="conformal").save(tmp_path / "calibration.json")
    report = _script().main(["--checkpoint", str(tmp_path / "best.pt"), "--skip-external", "--device", "cpu",
                             "--num-workers", "0", "--batch-size", "32"])
    out = json.loads((tmp_path / "ood_gate_eval.json").read_text())
    assert set(out["gates"]) == {"0.05", "0.1"} and (tmp_path / "ood_gate_alpha0.05.json").exists()
    for name in ("select", "test"):
        ov = report["sets"][name]["overall"]
        assert {"alpha=0.05", "alpha=0.1"} <= set(ov)
        for a in ("alpha=0.05", "alpha=0.1"):
            assert ov[a]["fpr_fake_verdict"] == ov["baseline"]["fpr_fake_verdict"]
            assert ov[a]["frac_verdict_fake"] == ov["baseline"]["frac_verdict_fake"]
            assert ov[a]["abstain_rate"] >= ov["baseline"]["abstain_rate"]
