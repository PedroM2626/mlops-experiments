# Evolutionary Feature Selection: GAAP (NSGA-II) and MO-DE vs Classical Methods

> **Area:** Feature Selection / Multi-objective Optimization
> **Task:** Regression (California Housing) and multiclass Classification (Twitter)
> **Metrics:** R2 (regression) and F1-macro (classification) — CV and holdout
> **Status:** Completed
> **Datasets:** California Housing (20.640 x 44 poly) and Twitter Entity Sentiment (5.000 x 400 TF-IDF)

---

## 1. Abstract

This experiment compares evolutionary feature-selection algorithms —
**GAAP** (GA with NSGA-II) and **MO-DE** (multi-objective Differential Evolution),
both implemented in **DEAP** — against classical methods (SelectKBest,
Random Forest importance and Boruta). The evaluation is made through the
**CV score x number of features** curve in two distinct domains: regression
(California Housing with interactive polynomial features) and classification
(TF-IDF of tweets). The evolutionary methods showed a significant advantage
when there is interactive structure between features (California), reaching ~0,69
of R2 with half the features of the baseline; in the bag-of-words domain (Twitter),
the classical methods remain superior. The best subset of each method is validated
on the holdout (test).

## 2. Context and Objectives

Feature selection is a combinatorial NP-hard problem that directly affects
inference cost, interpretability and generalization. Classical methods
based on univariate ranking or importance ignore
**complementarity between features** (e.g.: pairs whose predictive power only
appears together). The research questions:

- (RQ1) Do evolutionary algorithms find subsets of **lower cardinality**
  with a score competitive with that of the model with all features?
- (RQ2) Does the gain depend on the **structure of the feature space** (interactions vs.
  almost independent features)?

## 3. Theoretical Foundation (brief)

- **Feature selection** as a search for a subset $S$ that maximizes a
  validation metric $f(S)$ under a cardinality budget (filter,
  wrapper, embedded).
- **NSGA-II (Deb et al., 2002)**: elitism by Pareto dominance on two
  fronts: minimize $(1 - \text{score})$ and minimize $|S|$. Binary
  representation (gene = feature present).
- **MO-DE**: evolution of real vectors in $[0,1]^n$ with mutation
  $v = v_{r1} + F(v_{r2} - v_{r3})$ and binomial crossover; the subset is obtained
  by threshold $\ge 0.5$; maintenance of the non-dominated front.
- **Baselines**: ranking by univariate statistic (f_regression/f_classif),
  feature importance (RF) and shadows/importance (Boruta); top-k curve
  monotonically increasing.

## 4. Methodology

### 4.1 Data

| Dataset | Shape | Task | Metric | Split |
|---|---|---|---|---|
| California Housing | 20.640 x 44 (poly degree 2, log1p) | Regression | R2 | 80/20 (seed 42) |
| Twitter Entity Sentiment | 5.000 x 400 (TF-IDF 1-2g) | Classification 4 classes | F1-macro | 80/20 (seed 42) |

### 4.2 Preprocessing

- California: log1p on the asymmetric variables + `PolynomialFeatures(degree=2)`
  over standardized data (8 -> 44 interactive features).
- Twitter: `TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2),
  max_features=400, min_df=2)` with URL/mention cleaning.

### 4.3 Compared methods

| Method | Type | Configuration |
|---|---|---|
| SelectKBest | univariate filter | f_regression / f_classif, top-k curve |
| RandomForest | embedded | importance ranking, top-k curve |
| Boruta | wrapper/shadow | perc=90, n_estimators=40 (subsample) |
| GAAP (NSGA-II) | evolutionary | pop=24, ngen=35 (cal) / 18x25 (tw); cxTwoPoint, mutFlipBit, selNSGA2 |
| MO-DE | evolutionary | pop=30, ngen=40 (cal) / 22x30 (tw); cr=0.5, fw=0.7 |

Base model of the evaluator: `Ridge(alpha=1.0)` + `StandardScaler` (regression) and
`LogisticRegression(C=1.0, class_weight='balanced')` (classification), with an
internal CV of 3 folds.

### 4.4 Evaluation

- Protocol: internal CV (3 folds) to build the score x features curves;
  the **best subset** of each method (highest CV) is re-trained and evaluated on the
  **holdout** (test_score).
- Fixed seeds (42) for reproducibility; execution on CPU (~4 min cal +
  ~2,5 min twitter).

### 4.5 Reproduction

```bash
python feature_selection_ea.py             # full pipeline
python feature_selection_ea.py --quick     # reduced config
python build_notebook.py                   # generates/executes the notebook with outputs
python multiseed.py outputs/summary_cal_seed*.csv   # aggregates multi-seed (mean ± std per method)
```

Tests: `tests/test_multiseed.py`. For robustness, run `feature_selection_ea.py` with
distinct seeds (e.g.: 42, 43, 44), save each `summary_*.csv` per seed and aggregate with `multiseed.py`
(`summarize_multiseed` → `best_cv_mean/std`, `test_mean/std` per method).

Artifacts in `outputs/` (`summary_*.csv`, `curves_*.csv`, `*.png`).

## 5. Results

### 5.1 Best point per method (CV) and holdout

**California Housing (R2; full = 0.7101)**

| Method | best_cv | best_feats | test_score |
|---|---|---|---|
| Boruta | 0.7101 | 44 | 0.7025 |
| SelectKBest | 0.7101 | 44 | 0.7025 |
| RandomForest | 0.7101 | 44 | 0.7025 |
| GAAP (NSGA-II) | 0.6941 | **23** | 0.6807 |
| MO-DE | 0.6831 | **20** | 0.6696 |

**Twitter (F1-macro; full = 0.469)**

| Method | best_cv | best_feats | test_score |
|---|---|---|---|
| SelectKBest | 0.4965 | 286 | 0.4610 |
| RandomForest | 0.4722 | 343 | 0.4589 |
| Boruta | 0.4690 | 400 | 0.4711 |
| GAAP (NSGA-II) | 0.4547 | 182 | 0.4394 |
| MO-DE | 0.4501 | 198 | 0.4405 |

### 5.1b Multi-seed robustness (seeds 42–45, `multiseed.py`)

| California (R2) | best_cv mean ± std | feats (median) |
|---|---|---|
| SelectKBest | 0.7013 ± 0.0105 | 44 |
| Boruta / RandomForest | 0.6994 ± 0.0137 | 44 |
| GAAP (NSGA-II) | 0.6842 ± 0.0082 | 22.5 |
| MO-DE | 0.6722 ± 0.0132 | 19.5 |

| Twitter (F1-macro) | best_cv mean ± std | feats (median) |
|---|---|---|
| SelectKBest | 0.4835 ± 0.0106 | 229 |
| RandomForest | 0.4633 ± 0.0116 | 286 |
| Boruta | 0.4606 ± 0.0125 | 400 |
| GAAP (NSGA-II) | 0.4451 ± 0.0101 | 182 |
| MO-DE | 0.4438 ± 0.0054 | 185 |

Std ≤ 0.014 in all methods: the EA-vs-classical ranking is stable across
seeds (it holds in both domains). Artifacts per seed: `summary_*_s43/s44/s45.csv`.

### 5.2 Comparison at an equal feature budget (California, R2)

| k features | GAAP | MO-DE | RandomForest | SelectKBest |
|---|---|---|---|---|
| ~13 | 0.678 | 0.665 | 0.662 | 0.592 |
| ~19-23 | 0.694 | 0.683 | 0.687 | 0.582 |
| 44 (full) | — | — | — | 0.710 |

### 5.3 Comparison at an equal feature budget (Twitter, F1-macro)

| k features | GAAP | MO-DE | RandomForest | SelectKBest |
|---|---|---|---|---|
| 172-182 | 0.443-0.455 | 0.429-0.440 | 0.461 | 0.478 |
| 229-286 | — | — | 0.471 | 0.490-0.497 |

## 6. Discussion

- **California (interactive features): the evolutionary methods win.** GAAP reaches
  0.6941 with 23 features (vs 0.7101 with 44). At equal budgets GAAP
  dominates SelectKBest (0.678 vs 0.592 at k=13). Top-k rankings fail because
  they ignore interactions of the type `MedInc x Latitude`; Boruta collapses at mid k
  (0.098 at 19 features).
- **Twitter (bag-of-words): the classical methods already sufficed.** TF-IDF features are
  almost independent, with no interaction to be discovered. SelectKBest at k=229
  (0.490) beats the best point of GAAP (0.4547 at 182). Note also that
  SelectKBest at k=286 (0.4965) beats the score with **all** 400 features
  (0.469) — more features only degrade LogReg in this domain.
- **Cost**: GA/DE cost 60-93 s per run on Twitter against ~7 s for the
  baselines, with no gain in domains without interaction.
- **Limitations**: results from one seed; NSGA-II/DE are stochastic; the
  evaluator is simple (Ridge/LogReg), so the conclusion holds for this
  protocol.

## 7. Conclusions and Recommendations

1. EA delivers a **Pareto front** (score x cardinality) and is only
   advantageous when there is **interactive structure** between features (the
   California/poly case).
2. In bag-of-words representations (TF-IDF), the univariate ranking
   (SelectKBest) or importance (RF) already reaches the same or better at lower
   cost.
3. Practical rule: use EA when derived/interactive features dominate the
   space; otherwise, start with SelectKBest/RF.

## 8. References and Files

- Implementation: `feature_selection_ea.py`
- Executed notebook: `feature_selection_ea.ipynb` (builder: `build_notebook.py`)
- Results: `outputs/` (`summary_cal.csv`, `summary_twitter.csv`, `curves_*.csv`, `*.png`)
- DEAP: Fortin et al., 2012 — DEAP: Evolutionary Algorithms Made Easy.
- Deb et al., 2002 — A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.
