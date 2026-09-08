import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval_detection import (  # noqa: E402
    classification_metrics, detection_map, face_verification_metrics)


def test_classification_metrics_perfect():
    out = classification_metrics([0, 1, 1, 0], [0, 1, 1, 0])
    assert out["accuracy"] == 1.0 and out["f1_macro"] == 1.0


def test_classification_metrics_empty_raises():
    with pytest.raises(ValueError):
        classification_metrics([], [])


def test_face_verification_finds_threshold():
    out = face_verification_metrics([0.2, 0.3, 0.8, 0.9], [True, True, False, False])
    assert out["best"]["accuracy"] == 1.0
    assert len(out["curve"]) > 0


def test_detection_map_perfect_and_empty():
    pb = [np.array([[0, 0, 10, 10]])]
    ps = [np.array([0.99])]
    tb = [np.array([[0, 0, 10, 10]])]
    assert detection_map(pb, ps, tb)["map"] == pytest.approx(1.0, abs=0.05)
    assert detection_map([np.zeros((0, 4))], [np.zeros(0)], tb)["map"] == 0.0
