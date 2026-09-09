"""Cold-start com metadados: prever series novas SEM historico.

Trabalho futuro do README do sales-forecast: embeddings de series exigiam
historico (vazamento/causalidade). Aqui a via e outra — so metadados
categoricos (pdv/produto) + calendario, sem nenhum lag. Split por COMBO
(pdv x produto): 80% dos combos treinam, 20% sao "novos" (cold).

Baselines: media global, media por (categoria_pdv, categoria).
Modelo: LightGBM com categoricas nativas, objective L1 (MAE).

Salva metricas em `experiments/artifacts/sales_coldstart_<ts>/metrics.json`.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "raw"
ART = HERE.parent / "artifacts"
SEED = 42
CATS = ["categoria_pdv", "premise", "categoria", "subcategoria", "tipos",
        "label", "marca", "fabricante"]


def load_panel():
    fato = pd.read_parquet(DATA / "fato_vendas.parquet")
    pdvs = pd.read_parquet(DATA / "dim_pdvs.parquet")
    prods = pd.read_parquet(DATA / "dim_produtos.parquet")
    df = fato.merge(pdvs, left_on="internal_store_id", right_on="pdv", how="left")
    df = df.merge(prods, left_on="internal_product_id", right_on="produto", how="left")
    assert df["categoria"].notna().mean() > 0.9, "join de produtos falhou"
    assert df["categoria_pdv"].notna().mean() > 0.9, "join de pdvs falhou"
    dt = pd.to_datetime(df["transaction_date"])
    iso = dt.dt.isocalendar()
    df["semana"] = iso.week.astype(int)
    df["preco"] = df["gross_value"] / df["quantity"].clip(lower=1e-9)
    g = df.groupby(["internal_store_id", "internal_product_id", "semana"],
                   observed=True)
    panel = g.agg(q=("quantity", "sum"), preco=("preco", "mean"),
                  **{c: (c, "first") for c in CATS}).reset_index()
    panel["sem_sin"] = np.sin(2 * np.pi * panel["semana"] / 52)
    panel["sem_cos"] = np.cos(2 * np.pi * panel["semana"] / 52)
    return panel


def main() -> int:
    t0 = time.time()
    panel = load_panel()
    print(f"[cold] panel: {panel.shape} ({time.time()-t0:.0f}s)", flush=True)
    panel["combo"] = (panel["internal_store_id"].astype(str) + "_" +
                      panel["internal_product_id"].astype(str))
    combos = panel["combo"].unique()
    tr_combos, cold_combos = train_test_split(combos, test_size=0.2, random_state=SEED)
    tr = panel[panel["combo"].isin(tr_combos)].copy()
    te = panel[panel["combo"].isin(cold_combos)].copy()
    print(f"[cold] combos treino={len(tr_combos)} cold={len(cold_combos)}", flush=True)

    feats = CATS + ["sem_sin", "sem_cos", "preco", "semana"]
    for c in CATS:
        tr[c] = tr[c].astype("category")
        te[c] = pd.Categorical(te[c], categories=tr[c].cat.categories)
    # categorias nao vistas no cold -> NaN -> categoria 'desconhecida'
    for c in CATS:
        if te[c].isna().any():
            tr[c] = tr[c].cat.add_categories("__novo__")
            te[c] = te[c].cat.add_categories("__novo__")
            te[c] = te[c].fillna("__novo__")

    gmean = float(tr["q"].mean())
    catmean = tr.groupby(["categoria_pdv", "categoria"], observed=True)["q"].mean()
    pred_g = np.full(len(te), gmean)
    pred_c = te.set_index(["categoria_pdv", "categoria"]).index.map(catmean)
    pred_c = pred_c.fillna(gmean).values
    mae = lambda a, b: float(np.abs(a - b).mean())
    res = {"mae_global": round(mae(te["q"].values, pred_g), 4),
           "mae_categoria": round(mae(te["q"].values, pred_c), 4)}

    tr2, va2 = train_test_split(tr, test_size=0.15, random_state=SEED)
    dtr = lgb.Dataset(tr2[feats], tr2["q"], categorical_feature=CATS)
    dva = lgb.Dataset(va2[feats], va2["q"], categorical_feature=CATS, reference=dtr)
    params = {"objective": "regression_l1", "metric": "mae", "verbosity": -1,
              "seed": SEED, "learning_rate": 0.05, "num_leaves": 63,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1}
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    pred_m = model.predict(te[feats])
    res["mae_metadata_lgbm"] = round(mae(te["q"].values, pred_m), 4)
    res["best_iter"] = int(model.best_iteration)
    imp = dict(zip(model.feature_name(),
                   (np.array(model.feature_importance()) /
                    max(1, np.array(model.feature_importance()).max())).round(3).tolist()))
    res["importance_rel"] = imp
    res.update({"n_train": int(len(tr)), "n_cold": int(len(te)),
                "elapsed_min": round((time.time() - t0) / 60, 1)})

    d = ART / f"sales_coldstart_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("artefatos em", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
