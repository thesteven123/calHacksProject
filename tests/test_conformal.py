"""Split-conformal threshold tests: finite-sample bounds hold empirically on exchangeable data."""

import numpy as np

from dfd.eval.calibration import Calibration, calibrate, conformal_lower, conformal_upper


def test_conformal_upper_bounds_exceedance_rate_over_many_draws():
    rng = np.random.default_rng(0)
    alpha, n = 0.05, 200
    rates = []
    for _ in range(400):
        cal_scores, new = rng.normal(size=n), rng.normal(size=2000)
        rates.append((new > conformal_upper(cal_scores, alpha)).mean())
    assert np.mean(rates) <= alpha + 0.005  # marginal guarantee: E[exceedance] <= alpha


def test_conformal_lower_bounds_undershoot_rate():
    rng = np.random.default_rng(1)
    beta, n = 0.05, 200
    rates = [(rng.normal(size=2000) < conformal_lower(rng.normal(size=n), beta)).mean() for _ in range(400)]
    assert np.mean(rates) <= beta + 0.005


def test_too_few_calibration_points_never_auto_decide():
    assert conformal_upper(np.arange(50.0), 0.01) == float("inf")   # need >= 99 reals for 1%
    assert conformal_lower(np.arange(10.0), 0.02) == float("-inf")  # need >= 49 fakes for 2%


def test_conformal_calibration_uses_worst_known_generator_and_saves_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    real = rng.normal(-3, 1, 600)
    easy = rng.normal(4, 1, 300)   # well-separated generator
    hard = rng.normal(0.5, 1, 300)  # hard generator: its low tail must set tau_real
    logits = np.r_[real, easy, hard]
    labels = np.r_[np.zeros(600), np.ones(600)].astype(int)
    gens = np.array(["real"] * 600 + ["easy"] * 300 + ["hard"] * 300)
    cal = calibrate(logits, labels, 0.01, 0.02, method="conformal", generators=gens)
    assert cal.method == "conformal" and cal.n_cal_real == 600
    assert cal.tau_real == min(cal.tau_real_by_generator.values()) == cal.tau_real_by_generator["hard"]
    assert cal.tau_real <= cal.tau_fake
    # FPR on fresh exchangeable reals stays near the 1% target
    fresh = cal.prob(rng.normal(-3, 1, 20000))
    assert (fresh >= cal.tau_fake).mean() <= 0.02
    cal.save(tmp_path / "c.json")
    loaded = Calibration.load(tmp_path / "c.json")
    assert loaded.method == "conformal" and loaded.tau_real_by_generator == cal.tau_real_by_generator


def test_conformal_never_flags_fake_when_reals_too_few():
    logits = np.r_[np.full(20, -2.0), np.full(200, 3.0)]
    labels = np.r_[np.zeros(20), np.ones(200)].astype(int)
    cal = calibrate(logits, labels, 0.01, 0.02, method="conformal")
    assert cal.tau_fake > 1.0 and cal.verdict(0.999999) != "fake"
