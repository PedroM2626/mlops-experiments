import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pose_classifier import N_FEATURES, PoseClassifier  # noqa: E402

TASK_BUNDLE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "models", "holistic_landmarker.task")


@pytest.fixture
def classifier(tmp_path, monkeypatch):
    """A classifier that neither loads a stray model nor writes ./mlruns."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    return PoseClassifier()


def fake_landmarks(seed=0):
    return np.random.default_rng(seed).random((33, 3))


def test_classifier_initialization(classifier):
    assert classifier.model is not None
    assert len(classifier.pose_labels) == 4


def test_feature_count_is_19_for_real_and_empty_poses(classifier):
    """The recorded datasets and logged models use 19 features; the no-pose
    placeholder used to be 20 zeros, a shape no model accepts."""
    assert N_FEATURES == 19
    assert len(classifier.extract_features(fake_landmarks())) == N_FEATURES
    empty = classifier.extract_features(None)
    assert len(empty) == N_FEATURES
    assert np.all(empty == 0)


def test_angle_calculation(classifier):
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([0, 1, 0])

    angle = classifier._calculate_angle(p1, p2, p3)
    assert np.isclose(angle, 90.0)


def test_no_pose_is_reported_as_such_not_as_a_missing_model(classifier):
    X = np.vstack([classifier.extract_features(fake_landmarks(i)) for i in range(40)])
    classifier.model.fit(X, np.arange(40) % 4)
    assert classifier.predict(None) == ("No Pose", 0.0)
    assert classifier.get_all_confidences(None) == {}
    name, conf = classifier.predict(fake_landmarks(1))
    assert name in classifier.pose_labels.values() and 0.0 < conf <= 1.0


def test_feature_mismatch_fails_loudly(classifier):
    """A model trained on a different feature count must raise, not read as
    "No Model" (the old blanket `except Exception` hid exactly this)."""
    classifier.model.fit(np.random.default_rng(0).random((40, N_FEATURES + 1)),
                         np.arange(40) % 4)
    with pytest.raises(ValueError):
        classifier.predict(fake_landmarks())


@pytest.mark.skipif(not os.path.exists(TASK_BUNDLE),
                    reason="models/holistic_landmarker.task is not in git (see README)")
def test_detector_initialization():
    pytest.importorskip("mediapipe", reason="the detector wrapper needs MediaPipe installed")
    from holistic_detector import HolisticDetector
    detector = HolisticDetector()
    assert detector.holistic is not None
    detector.release()
