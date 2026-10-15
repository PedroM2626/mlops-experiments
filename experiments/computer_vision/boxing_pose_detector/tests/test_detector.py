import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pose_classifier import PoseClassifier  # noqa: E402

pytest.importorskip("mediapipe", reason="the detector wrapper needs MediaPipe installed")
from holistic_detector import HolisticDetector  # noqa: E402

TASK_BUNDLE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "models", "holistic_landmarker.task")


def test_classifier_initialization():
    classifier = PoseClassifier()
    assert classifier.model is not None
    assert len(classifier.pose_labels) == 4


def test_feature_extraction_empty():
    classifier = PoseClassifier()
    features = classifier.extract_features(None)
    assert len(features) == 20
    assert np.all(features == 0)


def test_angle_calculation():
    classifier = PoseClassifier()
    p1 = np.array([1, 0, 0])
    p2 = np.array([0, 0, 0])
    p3 = np.array([0, 1, 0])

    angle = classifier._calculate_angle(p1, p2, p3)
    assert np.isclose(angle, 90.0)


@pytest.mark.skipif(not os.path.exists(TASK_BUNDLE),
                    reason="models/holistic_landmarker.task is not in git (see README)")
def test_detector_initialization():
    detector = HolisticDetector()
    assert detector.holistic is not None
    detector.release()
