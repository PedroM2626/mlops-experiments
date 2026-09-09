"""Detector YOLO via OpenCV DNN (usado pelo `yolo_notebook.ipynb`).

`build_detector_from_env()` monta o detector a partir de `.env` ou do
fallback YOLOv3-tiny COCO (pesos baixados automaticamente p/ o cache local
`~/.cache/mlops_yolo/`). API minima usada pelo notebook e pelos testes:

    detector = build_detector_from_env()
    dets = detector.detect(img_bgr)   # [{'class_name', 'confidence', 'box'}]
    out = detector.draw(img_bgr, dets)

Variaveis `.env` (todas opcionais):
    YOLO_WEIGHTS, YOLO_CFG, YOLO_NAMES, YOLO_SIZE (default 416),
    YOLO_CONF (default 0.5), YOLO_NMS (default 0.4)
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

COCO_URLS = {
    # pesos: original Joseph Redmon (pjreddie fora do ar -> tentar proximos)
    "weights": [
        "https://pjreddie.com/media/files/yolov3-tiny.weights",
    ],
    "cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3-tiny.cfg",
    "names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
}
CACHE = Path.home() / ".cache" / "mlops_yolo"


def _fetch(url: str | list, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    urls = [url] if isinstance(url, str) else url
    last = None
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
                f.write(r.read())
            return dest
        except Exception as e:
            last = e
            if dest.exists():
                dest.unlink()
    raise RuntimeError(f"download falhou ({urls[0]}...): {last}")


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v or default


class YoloDetector:
    def __init__(self, weights: str, cfg: str, names: str, size=416,
                 conf=0.5, nms=0.4):
        with open(names, encoding="utf-8") as f:
            self.classes = [l.strip() for l in f if l.strip()]
        self.net = cv2.dnn.readNetFromDarknet(cfg, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.size = int(size)
        self.conf = float(conf)
        self.nms = float(nms)
        layers = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        self.out_layers = [layers[i - 1] for i in np.array(unconnected).flatten()]

    def detect(self, img_bgr: np.ndarray) -> list:
        h, w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(img_bgr, 1 / 255.0, (self.size, self.size),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_layers)
        boxes, confs, cls_ids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                c = float(scores[cid])
                if c > self.conf:
                    cx, cy, bw, bh = det[0:4] * np.array([w, h, w, h])
                    boxes.append([int(cx - bw / 2), int(cy - bh / 2),
                                  int(bw), int(bh)])
                    confs.append(c)
                    cls_ids.append(cid)
        idxs = cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.nms)
        idxs = np.array(idxs).flatten() if len(idxs) else []
        return [{"class_name": self.classes[cls_ids[i]], "confidence": confs[i],
                 "box": boxes[i]} for i in idxs]

    def draw(self, img_bgr: np.ndarray, dets: list) -> np.ndarray:
        out = img_bgr.copy()
        for d in dets:
            x, y, bw, bh = d["box"]
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(out, f"{d['class_name']} {d['confidence']:.2f}",
                        (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)
        return out


def build_detector_from_env() -> YoloDetector:
    weights = _env("YOLO_WEIGHTS", "")
    cfg = _env("YOLO_CFG", "")
    names = _env("YOLO_NAMES", "")
    if not (weights and cfg and names):
        weights = str(_fetch(COCO_URLS["weights"], CACHE / "yolov3-tiny.weights"))
        cfg = str(_fetch(COCO_URLS["cfg"], CACHE / "yolov3-tiny.cfg"))
        names = str(_fetch(COCO_URLS["names"], CACHE / "coco.names"))
    return YoloDetector(weights, cfg, names, size=_env("YOLO_SIZE", "416"),
                        conf=_env("YOLO_CONF", "0.5"), nms=_env("YOLO_NMS", "0.4"))
