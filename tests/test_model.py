import numpy as np
import torch

from dfd.config import ModelConfig
from dfd.eval.calibration import calibrate
from dfd.eval.metrics import binary_metrics, sliced_report, tpr_at_fpr
from dfd.models.detector import DeepfakeDetector
from dfd.models.temporal import select_keyframes, topk_pool
from dfd.training.losses import BetaSchedule, detector_loss, distillation_loss

TINY = dict(backbone="random-tiny", feature_layers=(2, 4), lora_rank=2, lora_blocks=2, hidden_dim=32, latent_dim=8)


def test_image_forward_shapes_and_lora_only_trainable_in_backbone():
    model = DeepfakeDetector(ModelConfig(**TINY))
    out = model(torch.rand(2, 3, 224, 224))
    assert out["logit"].shape == (2,) and out["mu"].shape == (2, 8) and "beauty_logit" in out
    assert model.backbone.lora_count == 4
    bb_trainable = [n for n, p in model.backbone.named_parameters() if p.requires_grad]
    assert bb_trainable and all("lora_" in n for n in bb_trainable)


def test_video_forward_with_temporal_adapter_and_mask():
    model = DeepfakeDetector(ModelConfig(**TINY, temporal=True, temporal_dim=16)).eval()
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    out = model(torch.rand(2, 4, 3, 224, 224), mask)
    assert out["logit"].shape == (2,) and out["frame_logits"].shape == (2, 4)


def test_loss_backward_reaches_lora():
    model = DeepfakeDetector(ModelConfig(**TINY))
    out = model(torch.rand(2, 3, 224, 224))
    loss, logs = detector_loss(out, torch.tensor([0, 1]), 1e-3, torch.tensor([1, 0]), 0.2)
    loss.backward()
    lora_grads = [p.grad for n, p in model.named_parameters() if "lora_b" in n]
    assert all(g is not None for g in lora_grads) and {"bce", "kl", "beauty"} <= logs.keys()


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = DeepfakeDetector(ModelConfig(**TINY)).eval()
    x = torch.rand(1, 3, 224, 224)
    model.save(tmp_path / "m.pt")
    loaded, _ = DeepfakeDetector.load(tmp_path / "m.pt")
    assert torch.allclose(model(x)["logit"], loaded.eval()(x)["logit"], atol=1e-5)


def test_topk_pool_respects_mask():
    logits = torch.tensor([[5.0, 1.0, 0.0, 9.0]])
    assert topk_pool(logits, torch.tensor([[1, 1, 1, 0]]), k_frac=0.25).item() == 5.0


def test_select_keyframes_includes_scene_change():
    sig = np.zeros((50, 4))
    sig[30:] = 10.0
    idx = select_keyframes(sig, 6)
    assert len(idx) == 6 and 30 in idx


def test_beta_schedule_and_distillation():
    s = BetaSchedule(1e-3, 1e-5, 10)
    assert s(0) == 0.0 and s(1) >= 1e-5 and s(20) == 1e-3
    loss = distillation_loss(torch.zeros(4), torch.zeros(4), torch.tensor([0, 1, 0, 1]))
    assert torch.isfinite(loss)


def test_metrics_and_calibration():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(500), np.ones(500)].astype(int)
    logits = rng.normal(0, 1, 1000) + 3 * labels
    m = binary_metrics(labels, logits)
    assert 0.95 < m["auc"] <= 1.0 and 0 <= m["tpr@0.1%fpr"] <= m["tpr@1%fpr"] <= 1
    assert tpr_at_fpr(labels, labels.astype(float), 0.01) == 1.0
    cal = calibrate(logits, labels, target_fpr=0.01, target_fnr=0.02)
    probs = cal.prob(logits)
    assert (probs[labels == 0] >= cal.tau_fake).mean() <= 0.011 and cal.tau_real <= cal.tau_fake
    rep = sliced_report(labels, logits, ["real"] * 500 + ["g1"] * 250 + ["g2"] * 250, ["light"] * 1000)
    assert set(rep["by_generator"]) == {"g1", "g2"}


def test_transformer_blocks_found_for_dinov3_layout():
    """DINOv3 (transformers 5) nests blocks at `model.layer`, not `encoder.layer`."""
    from torch import nn

    from dfd.models.backbone import _transformer_blocks

    inner = nn.Module()
    inner.layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
    outer = nn.Module()
    outer.model = inner
    assert len(_transformer_blocks(outer)) == 3
