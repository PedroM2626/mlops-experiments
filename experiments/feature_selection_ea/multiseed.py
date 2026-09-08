"""Agregação multi-seed para o comparativo EA vs clássicos.

Endereça a limitação do README (resultados de 1 seed; NSGA-II/DE são
estocásticos): dada uma lista de DataFrames de resumo por seed
(colunas `method,best_cv,best_feats,full_cv,test_score` + coluna `seed`),
retorna média ± desvio por método. CLI:

    python multiseed.py outputs/summary_cal_seed*.csv
"""

from __future__ import annotations

import glob
import sys

import pandas as pd


SUMMARY_COLS = ["method", "best_cv", "best_feats", "full_cv", "test_score"]


def summarize_multiseed(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("nenhum frame fornecido")
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in SUMMARY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"colunas ausentes: {missing}")
    if "seed" not in df.columns:
        df = df.copy()
        df["seed"] = 0
    g = df.groupby("method")
    out = pd.DataFrame({
        "seeds": g["seed"].nunique(),
        "best_cv_mean": g["best_cv"].mean().round(4),
        "best_cv_std": g["best_cv"].std(ddof=0).fillna(0.0).round(4),
        "test_mean": g["test_score"].mean().round(4),
        "test_std": g["test_score"].std(ddof=0).fillna(0.0).round(4),
        "feats_median": g["best_feats"].median(),
    }).reset_index().sort_values("best_cv_mean", ascending=False)
    return out


def main(paths: list[str]) -> int:
    files: list[str] = []
    for p in paths:
        files.extend(glob.glob(p) or [p])
    frames = []
    for i, f in enumerate(files):
        d = pd.read_csv(f)
        d["seed"] = i
        frames.append(d)
    print(summarize_multiseed(frames).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
