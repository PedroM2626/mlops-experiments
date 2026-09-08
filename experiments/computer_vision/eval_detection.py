"""Harness de avaliação para CV: classificação, verificação facial e detecção.

Cobre o gap documentado no README (face_recognition_app e YOLO avaliados
só por inspeção visual). Funções puras (numpy + sklearn), sem OpenCV
obrigatório — o chamador extrai embeddings/boxes e passa arrays.

- `classification_metrics`: accuracy, macro-F1, relatório por classe.
- `face_verification_metrics`: a partir de distâncias de pares
  (mesma pessoa ou não), varre limiares e retorna melhor acc + curva.
- `detection_map`: mAP@IoU simples para caixas [x1,y1,x2,y2] por imagem.
"""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("y_true vazio")
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def face_verification_metrics(distances, same_person, thresholds=None) -> dict:
    """Varre limiares de distância e retorna o melhor ponto.

    `distances`: menor = mais similar. `same_person`: bool por par.
    """
    d = np.asarray(distances, dtype=float)
    s = np.asarray(same_person, dtype=bool)
    if d.shape != s.shape or d.size == 0:
        raise ValueError("distances e same_person devem ter o mesmo shape não-vazio")
    if thresholds is None:
        thresholds = np.unique(np.quantile(d, np.linspace(0, 1, 21)))
    best = {"threshold": None, "accuracy": -1.0}
    curve = []
    for t in thresholds:
        pred_same = d <= float(t)
        acc = float((pred_same == s).mean())
        curve.append({"threshold": round(float(t), 4), "accuracy": round(acc, 4)})
        if acc > best["accuracy"]:
            best = {"threshold": round(float(t), 4), "accuracy": round(acc, 4)}
    return {"best": best, "curve": curve}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.array([max(a[0], b[0]), max(a[1], b[1]),
                      min(a[2], b[2]), min(a[3], b[3])])
    iw, ih = max(0.0, inter[2] - inter[0]), max(0.0, inter[3] - inter[1])
    inter_area = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def detection_map(pred_boxes, pred_scores, true_boxes, iou_thr: float = 0.5) -> dict:
    """mAP simplificado (uma classe) sobre listas por imagem.

    Cada elemento é um array (n,4) de boxes [x1,y1,x2,y2].
    """
    aps = []
    for pb, ps, tb in zip(pred_boxes, pred_scores, true_boxes):
        pb, ps, tb = np.asarray(pb, dtype=float), np.asarray(ps, dtype=float), np.asarray(tb, dtype=float)
        if tb.size == 0:
            aps.append(1.0 if pb.size == 0 else 0.0)
            continue
        if pb.size == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-ps)
        matched = np.zeros(len(tb), dtype=bool)
        tp = []
        for i in order:
            ious = [_iou(pb[i], t) for t in tb]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched[j]:
                matched[j] = True
                tp.append(1)
            else:
                tp.append(0)
        tp = np.asarray(tp)
        prec = np.cumsum(tp) / (np.arange(len(tp)) + 1)
        rec = np.cumsum(tp) / len(tb)
        # AP: área sob a curva PR com envelope (precisão interpolada)
        prec_env = np.maximum.accumulate(prec[::-1])[::-1]
        rec_prev = np.r_[0.0, rec[:-1]]
        ap = float(np.sum((rec - rec_prev) * prec_env))
        aps.append(ap)
    return {"map": round(float(np.mean(aps)), 4) if aps else 0.0, "per_image_ap": [round(float(a), 4) for a in aps]}
