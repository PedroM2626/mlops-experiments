import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from yolo_inference import YoloDetector, build_detector_from_env  # noqa: E402


def test_detector_download_and_detect():
    det = build_detector_from_env()
    assert len(det.classes) == 80  # COCO
    rng = np.random.RandomState(0)
    img = (rng.rand(416, 416, 3) * 255).astype(np.uint8)
    dets = det.detect(img)
    assert isinstance(dets, list)
    for d in dets:
        assert set(d) == {"class_name", "confidence", "box"}
        assert d["class_name"] in det.classes
    out = det.draw(img, dets)
    assert out.shape == img.shape


def test_env_override_tmp(tmp_path):
    import shutil
    src_weights = os.path.join(os.path.expanduser("~"), ".cache", "mlops_yolo",
                               "yolov3-tiny.weights")
    assert os.path.exists(src_weights), "rode test_detector_download_and_detect antes"
    for name in ["yolov3-tiny.weights", "yolov3-tiny.cfg", "coco.names"]:
        shutil.copy(os.path.join(os.path.dirname(src_weights), name), tmp_path / name)
    os.environ["YOLO_WEIGHTS"] = str(tmp_path / "yolov3-tiny.weights")
    os.environ["YOLO_CFG"] = str(tmp_path / "yolov3-tiny.cfg")
    os.environ["YOLO_NAMES"] = str(tmp_path / "coco.names")
    try:
        det = build_detector_from_env()
        assert len(det.classes) == 80
    finally:
        for k in ["YOLO_WEIGHTS", "YOLO_CFG", "YOLO_NAMES"]:
            os.environ.pop(k, None)
