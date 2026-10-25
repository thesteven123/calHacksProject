"""Offline tests for the FairFace demographic fairness check (dfd.eval.fairness + scripts)."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from dfd.eval.calibration import Calibration
from dfd.eval.fairness import disparity_summary, fairness_report, rate_table, wilson_interval
from test_external_data import _script


def test_wilson_interval():
    assert wilson_interval(0, 10) == (0.0, pytest.approx(0.2775, abs=1e-4))
    assert wilson_interval(10, 10) == (pytest.approx(0.7225, abs=1e-4), 1.0)
    lo, hi = wilson_interval(5, 10)
    assert lo + hi == pytest.approx(1.0) and lo < 0.5 < hi
    assert all(math.isnan(x) for x in wilson_interval(0, 0))


def test_rate_table_exact():
    t = rate_table([True, False, False, True, True, False], ["b", "a", "a", "b", "a", "c"])
    assert list(t) == ["a", "b", "c"]
    assert (t["a"]["n"], t["a"]["k"], t["a"]["rate"]) == (3, 1, pytest.approx(1 / 3))
    assert (t["b"]["n"], t["b"]["k"], t["b"]["rate"]) == (2, 2, 1.0)
    assert (t["c"]["n"], t["c"]["k"], t["c"]["rate"]) == (1, 0, 0.0)
    assert t["c"]["ci_low"] == 0.0 and t["b"]["ci_high"] == 1.0


def _row(k, n):
    lo, hi = wilson_interval(k, n)
    return {"n": n, "k": k, "rate": k / n, "ci_low": lo, "ci_high": hi}


def test_disparity_summary():
    table = {"hi": _row(30, 200), "mid": _row(3, 200), "zero": _row(0, 200), "tiny": _row(5, 10)}
    s = disparity_summary(table, target=0.01, min_n=100)
    assert s["max_group"] == "hi" and s["min_group"] == "zero" and s["underpowered"] == ["tiny"]
    assert s["gap"] == pytest.approx(0.15) and s["ratio"] is None
    assert "hi" in s["above_target_ci"] and "mid" not in s["above_target_ci"]
    assert {"hi", "mid"} <= set(s["above_target_point"])
    s2 = disparity_summary({"hi": _row(30, 200), "mid": _row(3, 200)}, target=None)
    assert s2["ratio"] == pytest.approx(10.0) and s2["above_target_point"] is None and s2["above_target_ci"] is None
    assert disparity_summary({"tiny": _row(1, 5)}, 0.01)["max_group"] is None


def test_fairness_report_intersections_and_review():
    v = ["fake", "abstain", "real", "abstain"]
    rep = fairness_report(v, {"race": ["A", "A", "B", "B"], "gender": ["M", "F", "M", "F"]}, target_fpr=0.01,
                          min_n=1, verdicts_valid=False)
    assert rep["verdicts_valid"] is False and rep["target_fpr"] == 0.01
    assert set(rep["attributes"]) == {"race", "gender", "race|gender"}
    assert set(rep["attributes"]["race|gender"]["fpr"]["table"]) == {"A|M", "A|F", "B|M", "B|F"}
    assert rep["attributes"]["gender"]["review"]["table"]["F"]["k"] == 2
    assert rep["attributes"]["race"]["fpr"]["table"]["A"]["k"] == 1
    assert rep["attributes"]["race"]["review"]["summary"]["above_target_ci"] is None


class _FakeFace(Dataset):
    def __init__(self, n=8):
        from PIL import Image

        self.imgs = [Image.new("RGB", (32, 32), (i * 20, 0, 0)) for i in range(n)]
        self.race = ["White", "Black"] * (n // 2)
        self.gender = ["Male"] * (n // 2) + ["Female"] * (n // 2)
        self.age = ["20-29"] * n

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], 0, "real"


def test_eval_fairface_includes_fairness():
    mod = _script("eval_external")

    class _Model:
        def __call__(self, x):
            return {"logit": torch.linspace(-4, 4, len(x))}

    pipe = SimpleNamespace(model=_Model(), device="cpu", calibration_status="matched",
                           cal=Calibration(temperature=1.0, tau_real=0.3, tau_fake=0.7, target_fpr=0.02),
                           transform=lambda img, rng: (torch.zeros(3, 8, 8), None))
    rep = mod._eval_fairface(pipe, _FakeFace(), batch_size=4, num_workers=0, min_n=1)
    assert {"overall", "by_race", "by_gender", "by_age", "fairness"} <= set(rep)
    fair = rep["fairness"]
    assert fair["verdicts_valid"] and fair["target_fpr"] == 0.02
    assert set(fair["attributes"]["race|gender"]["fpr"]["table"]) == {"White|Male", "Black|Male",
                                                                      "White|Female", "Black|Female"}
    pipe.calibration_status = "mismatched"
    off = mod._eval_fairface(pipe, _FakeFace(), batch_size=4, num_workers=0)
    assert off["fairness"]["verdicts_valid"] is False
    assert off["fairness"]["attributes"]["race"]["review"]["table"]["White"]["rate"] == 1.0


def _synthetic_report(fpr_flags):
    v = np.where(fpr_flags, "fake", "real")
    groups = {"race": ["White", "Black"] * 100, "gender": ["Male", "Female"] * 100, "age": ["20-29"] * 200}
    return {"checkpoint": "x/best.pt", "calibration": {"status": "matched"},
            "overall": {"n": 200, "fpr_fake_verdict": float(v.tolist().count("fake") / 200), "abstain_rate": 0.0},
            "by_race": {"White": {"mean_p_fake": 0.1}, "Black": {"mean_p_fake": 0.2}},
            "fairness": fairness_report(v, groups, target_fpr=0.01, min_n=50)}


def test_fairness_report_script_renders(tmp_path):
    mod = _script("fairness_report")
    rng = np.random.default_rng(0)
    raw, wrapped = tmp_path / "b1.json", tmp_path / "b3.json"
    raw.write_text(json.dumps(_synthetic_report(rng.random(200) < 0.05)))
    wrapped.write_text(json.dumps({"returncode": 0, "report": _synthetic_report(rng.random(200) < 0.02)}))
    out = tmp_path / "r.md"
    md = mod.main(["--inputs", str(raw), str(wrapped), "--labels", "B1,B3", "--output", str(out)])
    assert out.read_text() == md
    assert "## B1" in md and "## B3" in md and "## Comparison" in md and "| White\\|Male |" in md
    rows = [line for line in md.splitlines() if line.startswith("| White\\|") or line.startswith("| Black\\|")]
    assert rows and all(line.replace("\\|", "").count("|") == 5 for line in rows)  # 4 columns, pipes escaped
    assert "| White |" in md and "0.1000" in md
    with pytest.raises(SystemExit):
        mod.main(["--inputs", str(raw), "--labels", "B1,B3"])


def test_binom_sf_and_holm():
    from dfd.eval.fairness import binom_sf, holm_adjust

    assert binom_sf(0, 5, 0.1) == 1.0 and binom_sf(6, 5, 0.1) == 0.0
    assert binom_sf(5, 10, 0.5) == pytest.approx(638 / 1024)
    assert binom_sf(10, 10, 0.3) == pytest.approx(0.3 ** 10)
    assert holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_disparity_summary_holm_and_gap_ci():
    table = {"hi": _row(30, 200), "edge": _row(5, 200), "ok": _row(2, 200)}
    s = disparity_summary(table, target=0.01, min_n=100)
    assert s["above_target_holm"] == ["hi"] and set(s["p_above_target_holm"]) == set(table)
    assert "edge" in s["above_target_point"]
    lo, hi = s["gap_ci"]
    assert 0 <= lo <= s["gap"] <= hi
    assert disparity_summary({"hi": _row(30, 200)}, target=0.01)["gap_ci"] is None
    assert disparity_summary(table, target=None)["above_target_holm"] is None
