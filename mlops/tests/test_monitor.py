import numpy as np
import pandas as pd
import pytest

from mlops.monitor import _psi, _share_diff, _try_evidently_drift, compute_drift


def test_psi_zero_on_identical():
    x = np.random.RandomState(0).normal(size=2000)
    assert _psi(x, x.copy()) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.RandomState(1)
    ref = rng.normal(size=5000)
    cur = ref + 1.0  # shift de média: drift inequívoco
    assert _psi(ref, cur) > 0.25


def test_share_diff_detects_category_change():
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    cur = pd.Series(["a"] * 20 + ["b"] * 80)
    assert _share_diff(ref, cur) == pytest.approx(0.6)


def test_compute_drift_flags_shifted_numeric():
    rng = np.random.RandomState(2)
    ref = pd.DataFrame({"lag_1": rng.normal(size=1000), "cat": ["x", "y"] * 500})
    cur = ref.copy()
    cur["lag_1"] = cur["lag_1"] + 1.0
    out = compute_drift(ref, cur)
    assert out["max_psi"] > 0.25
    assert out["drifted_features"] >= 1


def test_evidently_helper_never_raises():
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"]})
    out = _try_evidently_drift(ref, ref.copy())
    assert out is None or out.get("method") == "evidently"
