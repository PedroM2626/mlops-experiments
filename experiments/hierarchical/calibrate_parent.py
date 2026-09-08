"""Calibração do nível pai para classificação hierárquica (20 Newsgroups).

Endereça os "próximos passos" do README: o pai é o gargalo (erro no pai
perde a folha em cadeia). Este módulo varre um limiar de confiança sobre
`decision_function`/`predict_proba` do classificador do pai:

- confiança >= t → mantém predição do pai;
- confiança < t → fallback (aqui: prediz a classe pai majoritária do treino,
  configurável via `fallback_correct`, máscara booleana por amostra que diz
  se o fallback acertaria).

A métrica proxy de folha é:
    leaf_proxy(t) = mean(parent_ok(t) * child_ok)
onde `child_ok` = 1 se o classificador filho acertaria a folha dado o pai
correto (medido no val). Retorna melhor t + curva completa.

Uso:
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
        raise ValueError("entradas com tamanhos divergentes")
    if len(y_true) == 0:
        raise ValueError("entradas vazias")
    if fallback_correct is None:
        # fallback burro: sempre erra o pai (limite inferior honesto)
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
