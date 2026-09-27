import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from calibrate_parent import tune_parent_threshold  # noqa: E402


def test_threshold_sweep_finds_optimum():
    # Parent gets everything right with high conf and everything wrong with low conf; child ok=0.86.
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    conf = np.array([0.9, 0.8, 0.85, 0.2, 0.1, 0.15])
    child_ok = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    best, curve = tune_parent_threshold(y_true, y_pred, conf, child_ok)
    assert best["leaf_proxy"] == 0.5  # 3/6 recoverable leaves
    assert len(curve) > 0
    # With a perfect fallback the proxy never drops below the no-threshold value
    fb = np.ones(6)
    best2, _ = tune_parent_threshold(y_true, y_pred, conf, child_ok, fallback_correct=fb)
    assert best2["leaf_proxy"] >= best["leaf_proxy"]


def test_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        tune_parent_threshold([], [], [], [])
