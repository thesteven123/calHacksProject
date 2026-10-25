"""Offline tests for scripts/modal_logo.py failure propagation helpers (nothing remote runs)."""

from test_external_data import _script


def test_classify_exit():
    mod = _script("modal_logo")
    assert mod._classify_exit(0) == "ok"
    assert mod._classify_exit(-9) == mod._classify_exit(-15) == "retryable"
    assert mod._classify_exit(1) == mod._classify_exit(2) == "failed"


def test_failed_rotations_order_and_kinds():
    mod = _script("modal_logo")
    pairs = [("ADM-to-BigGAN", {"returncode": 0}), ("BigGAN-to-GLIDE", {"returncode": 1}),
             ("GLIDE-to-Midjourney", RuntimeError("killed by signal 9")), ("SD15-to-VQDM", {"returncode": 0})]
    assert mod._failed_rotations(pairs) == ["BigGAN-to-GLIDE", "GLIDE-to-Midjourney"]
    assert mod._failed_rotations([("config", {"returncode": 0})]) == []


def test_rotation_name():
    mod = _script("modal_logo")
    assert mod._rotation_name(None, None) == "config"
    assert mod._rotation_name("GLIDE", "Midjourney") == "GLIDE-to-Midjourney"
