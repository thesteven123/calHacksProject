import numpy as np
from sklearn.metrics import roc_auc_score

from dfd.eval.fusion import fit_fusion, paired_bootstrap


def _two_noisy_routes(n, seed):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    a = 1.0 * y + rng.normal(0, 1, 2 * n)          # route A
    b = 3.0 * (1.0 * y + rng.normal(0, 1, 2 * n))  # route B: independent noise, different scale
    return y, np.c_[a, b]


def test_fusion_of_independent_routes_beats_each_route():
    y_fit, x_fit = _two_noisy_routes(400, 0)
    y_test, x_test = _two_noisy_routes(400, 1)
    fusion = fit_fusion(x_fit, y_fit, ["a", "b"])
    fused = fusion(x_test)
    assert roc_auc_score(y_test, fused) > max(roc_auc_score(y_test, x_test[:, 0]), roc_auc_score(y_test, x_test[:, 1]))
    # standardization makes the weights comparable despite route B's 3x scale
    assert abs(fusion.weights[0] - fusion.weights[1]) < 0.5 * max(fusion.weights)


def test_paired_bootstrap_detects_real_improvement_and_brackets_zero_for_identical():
    y, x = _two_noisy_routes(300, 2)
    better = x[:, 0] + x[:, 1] / 3
    res = paired_bootstrap(y, x[:, 0], better, n_boot=300)
    assert res["d_auc"] > 0 and res["d_auc_ci"][0] > 0
    same = paired_bootstrap(y, x[:, 0], x[:, 0], n_boot=100)
    assert same["d_auc"] == 0 and same["d_auc_ci"] == [0.0, 0.0]
