"""
Feature Selection Evolucionaria (DEAP: GAAP-NSGA-II + MO-DE) vs metodos classicos.

Datasets:
  * California Housing  -> regressao      (metrica: R2)
  * Twitter sentiment   -> classificacao  (metrica: F1-macro)

Metodos comparados:
  * SelectKBest        (univariado: f_regression / f_classif)
  * RandomForest       (ranking por feature importance)
  * Boruta             (shadow features)
  * GAAP (NSGA-II)     (DEAP) - otimiza via Pareto: minimiza (1-score) e n-features
  * MO-DE              (DE multi-objetivo em vetor real continuo + threshold 0.5)

A metrica reportada e o score CV (R2 / F1-macro) em funcao do numero de features
selecionadas; o melhor subset de cada metodo tambem e validado no holdout (test).

Uso:
    python feature_selection_ea.py                 # completo
    python feature_selection_ea.py --quick         # config menor p/ validar rapido
    python feature_selection_ea.py --no-mlflow
    python feature_selection_ea.py --only-cal

Artefatos: outputs/curves_*.csv, outputs/summary_*.csv, outputs/*.png
"""
import argparse
import os
import random
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from deap import base, creator, tools, algorithms

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer, f1_score, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 42
OUT = Path(__file__).resolve().parent / "outputs"


# --------------------------------------------------------------------------- #
# 1. DADOS
# --------------------------------------------------------------------------- #
def load_california(poly_degree=2):
    """California Housing (regressao). log1p nas variaveis assimetricas + polynomial."""
    data = fetch_california_housing()
    X = data.data.astype(np.float64)
    names = list(data.feature_names)
    # log1p em vars com cauda longa (MedInc, AveRooms, AveBedrms, Population, AveOccup)
    asim = [0, 3, 4, 5, 6]
    X = np.column_stack([np.log1p(X[:, i]) if i in asim else X[:, i]
                         for i in range(X.shape[1])])
    if poly_degree and poly_degree > 1:
        from sklearn.preprocessing import StandardScaler as _StdScaler
        Xs = _StdScaler().fit_transform(X)
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        X = poly.fit_transform(Xs)
        names = [str(n).replace(" ", "_") for n in poly.get_feature_names_out(names)]
    return X, data.target.astype(np.float64), np.array(names, dtype=object)


def _clean_tweet(t):
    import re
    t = re.sub(r"http\S+|www\S+|https\S+", "", str(t).lower())
    t = re.sub(r"@\w+|#\w+", "", t)
    t = re.sub(r"[^a-z0-9\s!?.,'\&-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_twitter(max_features=400, n_rows=2000):
    """Twitter Entity Sentiment -> classificacao multiclasse."""
    here = Path(__file__).resolve().parent
    train = here / ".." / "senti-pred-variations" / "Senti-Pred-remake2" / "data" / "raw" / "twitter_training.csv"
    if not train.exists():
        raise FileNotFoundError(f"Twitter CSV not found: {train}")
    df = pd.read_csv(train, header=None, names=["id", "entity", "label", "text"])
    df = df.dropna(subset=["text", "label"])
    if n_rows and len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=SEED)
    labels = sorted(df["label"].unique())
    y = df["label"].map({l: i for i, l in enumerate(labels)}).to_numpy()
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                          max_features=max_features, min_df=2)
    X = vec.fit_transform(df["text"].map(_clean_tweet)).toarray()
    return X, y, np.array(vec.get_feature_names_out(), dtype=object), labels


# --------------------------------------------------------------------------- #
# 2. AVALIADOR (CV interno + holdout)
# --------------------------------------------------------------------------- #
class Evaluator:
    def __init__(self, X_tr, y_tr, X_te, y_te, task, cv_folds=3):
        self.X_tr, self.y_tr, self.X_te, self.y_te = X_tr, y_tr, X_te, y_te
        self.task = task
        self.n_features = X_tr.shape[1]
        if task == "regression":
            self.scoring, self.metric_name = "r2", "R2"
            self._model = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        else:
            self.scoring = make_scorer(f1_score, average="macro")
            self.metric_name = "F1-macro"
            self._model = lambda: LogisticRegression(C=1.0, max_iter=3000,
                                                     class_weight="balanced")
            self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

    def cv_score(self, mask):
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        return float(cross_val_score(self._model(), self.X_tr[:, mask], self.y_tr,
                                     cv=self.cv, scoring=self.scoring,
                                     n_jobs=1).mean())

    def test_score(self, mask):
        mask = np.asarray(mask, dtype=bool)
        m = self._model()
        m.fit(self.X_tr[:, mask], self.y_tr)
        yp = m.predict(self.X_te[:, mask])
        if self.task == "regression":
            return float(r2_score(self.y_te, yp))
        return float(f1_score(self.y_te, yp, average="macro"))

    def full_score(self):
        return self.cv_score(np.ones(self.n_features, dtype=bool))


# --------------------------------------------------------------------------- #
# 3. GAAP - NSGA-II (DEAP)
# --------------------------------------------------------------------------- #
def _ensure_creator():
    if not hasattr(creator, "FitnessFS"):
        creator.create("FitnessFS", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "IndividualFS"):
        creator.create("IndividualFS", list, fitness=creator.FitnessFS)


def run_ga(evaler, pop=20, ngen=30, seed=SEED, verbose=False):
    random.seed(seed)
    np.random.seed(seed)
    n = evaler.n_features
    _ensure_creator()

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IndividualFS,
                     toolbox.attr_bool, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n)
    toolbox.register("select", tools.selNSGA2)

    def evaluate(ind):
        mask = np.array(ind, dtype=bool)
        k = int(mask.sum())
        sc = evaler.cv_score(mask)
        return (1.0 - sc, k)

    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=pop)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    hof = tools.ParetoFront()
    hof.update(population)

    for gen in range(ngen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.3)
        for ind in offspring:
            del ind.fitness.values
        fits = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        # NSGA-II: selecao do conjunto uniao (parentes + filhotes)
        population = toolbox.select(population + offspring, k=pop)
        hof.update(population)
        if verbose and gen % 5 == 0:
            best = min(i.fitness.values[0] for i in population)
            print(f"   [GA] gen {gen:3d} best-loss {best:.4f}")

    pareto = {}
    best_mask_by_k = {}
    for ind in hof:
        loss, k = ind.fitness.values
        k = int(k)
        sc = 1.0 - loss
        if k not in pareto or sc > pareto[k]:
            pareto[k] = sc
            best_mask_by_k[k] = np.array(ind, dtype=bool)
    return pareto, best_mask_by_k


# --------------------------------------------------------------------------- #
# 4. MO-DE (DE multi-objetivo, vetor real continuo + threshold)
# --------------------------------------------------------------------------- #
def _obj(evaler, mask):
    k = int(mask.sum())
    if k == 0:
        return (np.inf, 0)
    return (1.0 - evaler.cv_score(mask), k)


def _dominates(a, b):
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def _nondom_idxs(costs):
    return [i for i in range(len(costs))
            if not any(_dominates(costs[j], costs[i]) for j in range(len(costs)) if j != i)]


def run_de(evaler, pop=20, ngen=40, cr=0.5, fw=0.7, seed=SEED, verbose=False):
    n = evaler.n_features
    rng = np.random.default_rng(seed)
    pop_v = rng.uniform(0.0, 1.0, (pop, n))
    costs = np.array([_obj(evaler, v > 0.5) for v in pop_v])

    for g in range(ngen):
        trial = np.empty_like(pop_v)
        for i in range(pop):
            idx = rng.choice(pop, 3, replace=False)
            r = rng.integers(n)
            mutant = np.clip(pop_v[idx[0]] + fw * (pop_v[idx[1]] - pop_v[idx[2]]), 0, 1)
            cross = (rng.random(n) < cr) | (np.arange(n) == r)
            trial[i] = np.where(cross, mutant, pop_v[i])
        t_costs = np.array([_obj(evaler, v > 0.5) for v in trial])
        for i in range(pop):
            if _dominates(t_costs[i], costs[i]) or (
                    not _dominates(costs[i], t_costs[i]) and rng.random() < 0.35):
                pop_v[i] = trial[i]
                costs[i] = t_costs[i]
        if verbose and g % 10 == 0:
            nd = _nondom_idxs(costs)
            print(f"   [DE] gen {g:3d} front={len(nd)} best-loss={min(costs[i][0] for i in nd):.4f}")

    pareto = {}
    best_mask_by_k = {}
    for i in _nondom_idxs(costs):
        k = int(costs[i][1])
        sc = 1.0 - costs[i][0]
        if k not in pareto or sc > pareto[k]:
            pareto[k] = sc
            best_mask_by_k[k] = pop_v[i] > 0.5
    return pareto, best_mask_by_k


# --------------------------------------------------------------------------- #
# 5. BASELINES - curvas top-k
# --------------------------------------------------------------------------- #
def _topk_mask(evaler, ranking, k):
    m = np.zeros(evaler.n_features, dtype=bool)
    m[ranking[:k]] = True
    return m


def curve_topk(evaler, ranking, ks):
    out = {}
    for k in ks:
        out[int(k)] = evaler.cv_score(_topk_mask(evaler, ranking, k))
    out[0] = 0.0
    return out


def baselines(evaler, X, y, task, max_steps=8):
    n = evaler.n_features
    ks = np.unique(np.round(np.linspace(1, n, min(max_steps, n))).astype(int))
    bl = {}

    sel = SelectKBest(f_regression if task == "regression" else f_classif, k="all")
    sel.fit(X, y)
    rank_sel = np.argsort(sel.scores_)[::-1]
    bl["SelectKBest"] = curve_topk(evaler, rank_sel, ks)
    bl["_rank_SelectKBest"] = rank_sel

    if task == "regression":
        rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=120, random_state=SEED, n_jobs=-1,
                                    class_weight="balanced")
    rf.fit(X, y)
    rank_rf = np.argsort(rf.feature_importances_)[::-1]
    bl["RandomForest"] = curve_topk(evaler, rank_rf, ks)
    bl["_rank_RandomForest"] = rank_rf

    try:
        from boruta import BorutaPy
        # subsample para rapidez; Boruta eh caro em espacos grandes
        if X.shape[0] > 1600 or X.shape[1] > 100:
            idx_bor = np.random.RandomState(SEED).choice(X.shape[0],
                                                         min(1600, X.shape[0]),
                                                         replace=False)
            Xb, yb = X[idx_bor], y[idx_bor]
        else:
            Xb, yb = X, y
        est = rf.__class__(n_estimators=40, random_state=SEED, n_jobs=-1,
                           max_depth=8, **({"class_weight": "balanced"}
                                           if task != "regression" else {}))
        bor = BorutaPy(est, n_estimators=40, random_state=SEED, perc=90,
                       max_iter=40, verbose=0)
        bor.fit(Xb, yb)
        bor_rank = np.argsort(bor.ranking_)[::-1]
        bl["Boruta"] = curve_topk(evaler, bor_rank, ks)
        bl["_rank_Boruta"] = bor_rank
        bl["Boruta_nconfirm"] = int(bor.support_.sum())
    except Exception as e:
        print(f"[warn] Boruta skipped: {e}")

    return bl


# --------------------------------------------------------------------------- #
# 6. PLOT
# --------------------------------------------------------------------------- #
def plot_curves(df, title, fname):
    plt.figure(figsize=(9, 5.5))
    for method, g in df.groupby("method"):
        g = g.sort_values("n_feats")
        plt.plot(g["n_feats"], g["cv_score"], marker="o", label=method, ms=3)
    plt.xlabel("numero de features selecionadas")
    plt.ylabel("CV score (R2 / F1-macro)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=140)
    plt.close()


# --------------------------------------------------------------------------- #
# 7. ORQUESTRADOR
# --------------------------------------------------------------------------- #
def run_one(task, X, y, cfg):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    ev = Evaluator(Xtr, ytr, Xte, yte, task)

    t0 = time.time()
    base = baselines(ev, Xtr, ytr, task)
    print(f"   [baselines] {time.time()-t0:.1f}s")
    t_base = time.time() - t0

    t0 = time.time()
    p_ga, masks_ga = run_ga(ev, cfg["ga_pop"], cfg["ga_gen"])
    print(f"   [GA NSGA-II] {time.time()-t0:.1f}s front={len(p_ga)}")

    t0 = time.time()
    p_de, masks_de = run_de(ev, cfg["de_pop"], cfg["de_gen"])
    print(f"   [MO-DE     ] {time.time()-t0:.1f}s front={len(p_de)}")

    rows = []
    for method, curve in base.items():
        if isinstance(curve, dict) and "_" not in method:
            for k, s in curve.items():
                rows.append(dict(method=method, n_feats=int(k), cv_score=s))
    for k, s in p_ga.items():
        rows.append(dict(method="GAAP (NSGA-II)", n_feats=int(k), cv_score=s))
    for k, s in p_de.items():
        rows.append(dict(method="MO-DE", n_feats=int(k), cv_score=s))
    df = pd.DataFrame(rows).sort_values(["method", "n_feats"])

    # resumo: melhor ponto (best_cv) de cada metodo
    summary = []
    for method, g in df.groupby("method"):
        g = g.sort_values("n_feats")
        best = g.loc[g["cv_score"].idxmax()]
        summary.append(dict(method=method, best_cv=round(float(best["cv_score"]), 4),
                            best_feats=int(best["n_feats"])))
    s = pd.DataFrame(summary).sort_values("best_cv", ascending=False).reset_index(drop=True)
    s["full_cv"] = round(ev.full_score(), 4)

    # validacao no holdout (test) do melhor subset de cada metodo
    test_rows = []
    for _, r in s.iterrows():
        k = int(r["best_feats"])
        if r["method"] == "GAAP (NSGA-II)":
            mask = masks_ga.get(k)
        elif r["method"] == "MO-DE":
            mask = masks_de.get(k)
        else:
            rank = base.get("_rank_" + r["method"])
            mask = _topk_mask(ev, rank, k) if rank is not None else None
        row = dict(r)
        row["test_score"] = round(ev.test_score(mask), 4) if mask is not None else None
        test_rows.append(row)
    s_test = pd.DataFrame(test_rows)
    return dict(df=df, summary=s_test, ev=ev, base=base,
                pareto_ga=p_ga, pareto_de=p_de,
                masks_ga=masks_ga, masks_de=masks_de)


# --------------------------------------------------------------------------- #
# 8. MAIN
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--only-cal", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    np.random.seed(SEED)

    # ---------- California Housing (regressao) ----------
    print("=" * 95)
    print("1) CALIFORNIA HOUSING - R2 | polynomial features (44)")
    X, y, names_cal = load_california()
    print(f"   shape: {X.shape}")
    cfg = dict(ga_pop=14 if args.quick else 24,
               ga_gen=10 if args.quick else 35,
               de_pop=14 if args.quick else 30,
               de_gen=10 if args.quick else 40)
    res_cal = run_one("regression", X, y, cfg)
    df_cal, sum_cal, ev_cal = res_cal["df"], res_cal["summary"], res_cal["ev"]

    # ---------- Twitter (classificacao) ----------
    if not args.only_cal:
        print("=" * 95)
        print("2) TWITTER - F1-macro | TF-IDF")
        Xt, yt, tnames, labels = load_twitter(max_features=150 if args.quick else 400,
                                              n_rows=1500 if args.quick else 5000)
        print(f"   shape: {Xt.shape}  classes: {labels}")
        cfg = dict(ga_pop=6 if args.quick else 18,
                   ga_gen=6 if args.quick else 25,
                   de_pop=8 if args.quick else 22,
                   de_gen=8 if args.quick else 30)
        res_tw = run_one("classification", Xt, yt, cfg)
        df_tw, sum_tw, ev_tw = res_tw["df"], res_tw["summary"], res_tw["ev"]
    else:
        sum_tw = None

    # persiste e plota
    df_cal.to_csv(OUT / "curves_cal.csv", index=False)
    sum_cal.to_csv(OUT / "summary_cal.csv", index=False)
    plot_curves(df_cal, "Feature Selection EA - California Housing", "curves_cal.png")
    if sum_tw is not None:
        df_tw.to_csv(OUT / "curves_twitter.csv", index=False)
        sum_tw.to_csv(OUT / "summary_twitter.csv", index=False)
        plot_curves(df_tw, "Feature Selection EA - Twitter", "curves_twitter.png")

    for tag, sm_row in {"California": sum_cal, "twitter": sum_tw}.items():
        if sm_row is None:
            continue
        print(f"\n----- RESUMO {tag.upper()} -----")
        print(sm_row.to_string(index=False))

    # MLflow
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("Feature_Selection_EA")
            for tag, sm_row in {"California": sum_cal, "twitter": sum_tw}.items():
                if sm_row is None:
                    continue
                with mlflow.start_run(run_name=f"{tag}_{time.strftime('%H%M%S')}"):
                    for _, r in sm_row.iterrows():
                        mlflow.log_metric(f"{r['method']}_cv", r["best_cv"])
                        mlflow.log_metric(f"{r['method']}_feats", r["best_feats"])
                    mlflow.log_metric("full_cv", r["full_cv"])
                    mlflow.log_artifact(str(OUT / f"curves_{tag.lower()}.csv"))
        except Exception as e:
            print(f"[mlflow] skip: {e}")

    print(f"\n>> done. artefactos em {OUT}")


if __name__ == "__main__":
    main()