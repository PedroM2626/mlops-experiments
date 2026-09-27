"""Parent-level calibration for hierarchical classification (20 Newsgroups).

Addresses the README "next steps": the parent is the bottleneck (an error at
the parent loses the leaf in cascade). This module sweeps a confidence threshold over
the parent classifier `decision_function`/`predict_proba`:

- confidence >= t → keep the parent prediction;
- confidence < t → fallback (here: predict the majority parent class of the training
  set, configurable via `fallback_correct`, a per-sample boolean mask saying
  whether the fallback would be right).

The leaf proxy metric is:
    leaf_proxy(t) = mean(parent_ok(t) * child_ok)
where `child_ok` = 1 if the child classifier would get the leaf right given the
correct parent (measured on val). Returns best t + the full curve.

Usage:
    from calibrate_parent import tune_parent_threshold
    best, curve = tune_parent_threshold(y_true, y_pred, conf, child_ok)
"""

from __future__ import annotations

import numpy as np


def tune_parent_threshold(y_true, y_pred, conf, child_ok, fallback_correct=None,
                          thresholds=None) -> tuple[dict, list]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    conf = np.asarray(conf, dtype=float)
    child_ok = np.asarray(child_ok, dtype=float)
    if not (len(y_true) == len(y_pred) == len(conf) == len(child_ok)):
        raise ValueError("inputs with divergent sizes")
    if len(y_true) == 0:
        raise ValueError("empty inputs")
    if fallback_correct is None:
        # dumb fallback: always misses the parent (honest lower bound)
        fallback_correct = np.zeros_like(y_true, dtype=float)
    fallback_correct = np.asarray(fallback_correct, dtype=float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(conf, np.linspace(0, 1, 21)))
    parent_ok_base = (y_pred == y_true).astype(float)
    best = {"threshold": None, "parent_acc": -1.0, "leaf_proxy": -1.0}
    curve = []
    for t in thresholds:
        t = float(t)
        keep = conf >= t
        parent_ok = np.where(keep, parent_ok_base, fallback_correct)
        parent_acc = float(parent_ok.mean())
        leaf_proxy = float((parent_ok * child_ok).mean())
        curve.append({"threshold": round(t, 4), "parent_acc": round(parent_acc, 4),
                      "leaf_proxy": round(leaf_proxy, 4), "kept": round(float(keep.mean()), 4)})
        if leaf_proxy > best["leaf_proxy"]:
            best = {"threshold": round(t, 4), "parent_acc": round(parent_acc, 4),
                    "leaf_proxy": round(leaf_proxy, 4)}
    return best, curve
