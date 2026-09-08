import pandas as pd
import pytest

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from multiseed import summarize_multiseed  # noqa: E402


def _frame(seed, rows):
    d = pd.DataFrame(rows, columns=["method", "best_cv", "best_feats", "full_cv", "test_score"])
    d["seed"] = seed
    return d


def test_multiseed_mean_std():
    f1 = _frame(0, [["GAAP (NSGA-II)", 0.69, 23, 0.71, 0.68], ["SelectKBest", 0.71, 44, 0.71, 0.70]])
    f2 = _frame(1, [["GAAP (NSGA-II)", 0.70, 25, 0.71, 0.69], ["SelectKBest", 0.70, 44, 0.71, 0.69]])
    out = summarize_multiseed([f1, f2])
    g = out.set_index("method").to_dict("index")
    assert g["GAAP (NSGA-II)"]["best_cv_mean"] == pytest.approx(0.695)
    assert g["GAAP (NSGA-II)"]["seeds"] == 2
    assert g["SelectKBest"]["best_cv_std"] == pytest.approx(0.005)


def test_empty_raises():
    with pytest.raises(ValueError):
        summarize_multiseed([])
