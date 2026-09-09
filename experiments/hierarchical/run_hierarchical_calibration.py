"""Calibracao do nivel pai em dados REAIS (20 Newsgroups).

Reproduz o setup V3 do README (TF-IDF word+char, LinearSVC pai C=0.5
balanced, filhos C=0.15) e aplica `calibrate_parent.tune_parent_threshold`
sobre `decision_function` real do pai, com 2 fallbacks:
  (a) zeros (limite inferior honesto);
  (b) pai implicito do classificador FLAT 20-classes (a alternativa real —
      "na duvida, delega pro flat").

Metrica proxy de folha: mean(pai_ok(t) x filho_ok). Tambem reporta
exact-match real de folha dos 3 sistemas (flat, hierarquico puro,
hierarquico com limiar otimo).

Salva em `experiments/artifacts/hier_calibrate_<ts>/metrics.json`.
~15-30 min de CPU.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
SEED = 42
sys.path.insert(0, os.path.dirname(__file__))
from calibrate_parent import tune_parent_threshold  # noqa: E402


def parent_of(name: str) -> str:
    return name.split(".")[0]


def main() -> int:
    t0 = time.time()
    tr = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                            random_state=SEED)
    te = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"),
                            random_state=SEED)
    ytr_leaf = np.array(tr.target)
    yte_leaf = np.array(te.target)
    names = np.array(tr.target_names)
    ytr_par = np.array([parent_of(n) for n in names[ytr_leaf]])
    yte_par = np.array([parent_of(n) for n in names[yte_leaf]])
    parents = sorted(set(ytr_par))
    print(f"[hier] treino={len(ytr_leaf)} teste={len(yte_leaf)} pais={parents}", flush=True)

    vec_w = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=80_000,
                            max_df=0.9)
    vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                            min_df=2, max_features=150_000, max_df=0.9)
    Xtr = hstack([vec_w.fit_transform(tr.data), vec_c.fit_transform(tr.data)]).tocsr()
    Xte = hstack([vec_w.transform(te.data), vec_c.transform(te.data)]).tocsr()
    print(f"[hier] matriz: {Xtr.shape} ({time.time()-t0:.0f}s)", flush=True)

    # --- flat 20 classes (baseline + fallback) ---
    flat = LinearSVC(C=0.15, random_state=SEED)
    flat.fit(Xtr, ytr_leaf)
    pred_flat = flat.predict(Xte)
    acc_flat = accuracy_score(yte_leaf, pred_flat)
    fb_parent_ok = (np.array([parent_of(n) for n in names[pred_flat]]) == yte_par).astype(float)
    print(f"[hier] flat leaf-acc={acc_flat:.4f}", flush=True)

    # --- pai ---
    clf_p = LinearSVC(C=0.5, class_weight="balanced", random_state=SEED)
    clf_p.fit(Xtr, ytr_par)
    pred_par = clf_p.predict(Xte)
    conf = clf_p.decision_function(Xte).max(axis=1)
    acc_par = accuracy_score(yte_par, pred_par)
    print(f"[hier] pai acc={acc_par:.4f}", flush=True)

    # --- filhos por pai ---
    child_pred = np.empty(len(yte_leaf), dtype=int)
    for p in parents:
        mtr = ytr_par == p
        leaves = sorted(set(ytr_leaf[mtr]))
        if len(leaves) == 1:
            child_pred[yte_par == p] = leaves[0]
            continue
        clf = LinearSVC(C=0.15, random_state=SEED)
        clf.fit(Xtr[mtr], ytr_leaf[mtr])  # labels = folhas globais
        mte = yte_par == p
        child_pred[mte] = clf.predict(Xte[mte])
    child_ok = ((child_pred == yte_leaf) & (pred_par == yte_par)).astype(float)
    leaf_hier = float(((pred_par == yte_par) & (child_pred == yte_leaf)).mean())
    print(f"[hier] hierarquico puro leaf-acc={leaf_hier:.4f} "
          f"| filho-dado-pai={child_ok[pred_par == yte_par].mean():.4f}", flush=True)

    out = {"parent_acc": round(float(acc_par), 4),
           "flat_leaf_acc": round(float(acc_flat), 4),
           "hier_leaf_acc": round(float(leaf_hier), 4)}
    for fb_name, fb in {"zeros": None,
                        "flat": fb_parent_ok}.items():
        best, curve = tune_parent_threshold(yte_par, pred_par, conf, child_ok,
                                            fallback_correct=fb)
        out[f"threshold_{fb_name}"] = best
        out[f"n_thresholds_{fb_name}"] = len(curve)
        print(f"[hier] fallback={fb_name}: {best}", flush=True)

    out["elapsed_min"] = round((time.time() - t0) / 60, 1)
    d = ART / f"hier_calibrate_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("artefatos em", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
