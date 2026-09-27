# Machine Learning & MLOps Experiment Repository

A portfolio of ML/MLOps experiments. Each experiment lives in its own folder
under `experiments/<experiment>/` with its **own academic README** (a compact
article: abstract, context, methodology, results, discussion, conclusions and
reproduction). This document is only the **index** that connects everything.

For the documentation standard, see
[`docs/modelo-academico-readme.md`](docs/modelo-academico-readme.md).

---

## Experiment Index

### 🧪 NLP — Sentiment Analysis, Topics and Representations

| Experiment | What it does | Main result | Read more |
|---|---|---|---|
| **NLP group** (senti-pred, A/B/C pipelines, Twitter Methods, multiclass Logistic, MMoE, AG News, FE NLP) | Sentiment/topic classification and text representations | TF-IDF + n-grams ~0.98 F1; transformers win in low-data | [see README](experiments/nlp/README.md) |
| **NLP in Regression — Wine (Kaggle)** | wine scoring from text | Ridge MAE 1.33 / R² 0.69 vs LightGBM 1.47 / 0.63 | [see](experiments/nlp-regression-wine/README.md) |
| **Senti-Pred variations** | Variations of the sentiment pipeline | record 97.80% (TF-IDF 100k, 4-grams) | [see](experiments/nlp/twitter-entity-sentiment/senti-pred-variations/README.md) |
| **Hierarchical 20 Newsgroups** | flat vs hierarchical classification, clustering | flat acc 0.7188 vs hierarchical 0.6953 | [see](experiments/hierarchical/README.md) |

### 🤖 Reinforcement Learning / AutoML

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **Q-Learning for AutoML** | RL agent optimizes hyperparameters | RL proxy on sales-forecast: MAE 1.4297 vs Optuna 1.4218 | [see](experiments/reinforcement_learning/README.md) |

### 📈 Time Series and Forecasting

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **Time Series group** (Prophet, 4×4 benchmark, 6-paradigm classification, TS+NLP, forecast-classification, distillation, anomalies, probabilistic/generative DeepAR, VAR, hierarchical forecast) | TS forecasting, classification and analysis | SARIMA wins 2/4 in the benchmark; ROCKET 3/3; Logistic 0.958 on forecast-direction | [see](experiments/time_series/README.md) |
| **5 Feature Engineering Phases (TS)** | manual vs automatic vs signals vs embeddings | DWT + manual: MAE 54.19 (best) | [see](experiments/ts_fe/README.md) |
| **Sales Forecast (Hackathon)** | weekly sales forecasting | LightGBM V2.2 MAE 1.4218 | [see](experiments/sales-forecast/README.md) |
| **Databricks Forecast (cloud)** | Prophet/DeepAR imported (Databricks) | local equivalents in time_series | [see](experiments/databricks-forecast/README.md) |

### 🖥️ Cloud → Local Equivalents (Watsonx & Databricks)

| Experiment | Objective | Read more |
|---|---|---|
| **IBM Watsonx (cloud originals)** | Boston Housing, Electric_Production, sentiment | [see](experiments/ibm-experiments/README.md) |
| **Local open-source equivalents** | replicate cloud AutoML/forecast (FLAML, TPOT, Prophet+Optuna, GluonTS) | [time_series](experiments/time_series/README.md) · [tabular](experiments/tabular_regression/README.md) |

### 🐱🖼️ Computer Vision

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **CV Methods (CIFAR-10)** | HOG+SVM vs ResNet18 vs ViT | ViT 0.9805 vs ResNet 0.9362 vs HOG 0.3970 | [see](experiments/computer_vision/README.md) |
| **Animal multi-label** | 4 approaches (pets Dime/Frida) | ResNet18+aug Exact Match 1.000 | same as above |
| **Face detection/recognition** | LBPH, CNN, YuNet (app in the notebook) | — | [see](experiments/computer_vision/README.md) |

### 🎬 RecSys

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **MovieLens RecSys — 8 approaches** | MF, neural networks, similarity, heuristic | Two-Tower RMSE 0.9297; SVD most efficient | [see](experiments/recommender_systems/README.md) |

### 🏎️ Tabular Regression & local AutoML

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **Tabular Regression group** (tabular FE, Price Prediction v1–v3, IBM Watsonx local) | impact of feature engineering; pipeline evolution | R² 0.9489 (Random Forest); per-model asymmetric FE | [see](experiments/tabular_regression/README.md) |

### 🔬 Evolutionary Feature Selection

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **GAAP (NSGA-II) and MO-DE vs classics** | multi-objective feature selection (R²/F1 × number of features) | advantage on interactive features (California); classics are already enough on Twitter | [see](experiments/feature_selection_ea/README.md) |

### 🔢 Ordinal Classification

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **Ordinal vs Nominal (Wine Quality)** | nominal LogReg/RF vs ordinal LogisticAT/IT (`mord`) | RF nominal acc 0.66 / MAE 0.36; the ordinal models tie on acc±1 ~0.9775 | [see](experiments/ordinal_classification/README.md) |

### 🧭 Causal Inference

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **Causal ML + NLP (Olist, real data)** | causal effect of delivery delay on review sentiment (ATE/CATE: LPM, Logit-AME, Matching, IPW, AIPW, S/T-learners, honest tree) | delay raises P(negative review) by ~+42 p.p. after adjustment (AIPW); stable to bootstrap/placebo/trimming; positive effect in all leaves | [see](experiments/causal_nlp_olist/README.md) |

---

## Other artifacts and standalone files

- **Standalone notebooks at the root of `experiments/`** (`anomaly_detection_comparison.ipynb`,
  `anomaly_detection_enhanced.ipynb`, `ensemble_pyramid.ipynb` + `ensemble_pyramid_best.pkl`)
  — supporting the READMEs above.
- **Experiment dashboard**: `dashboard/index.html` (open in the browser).

## Repository standards

- **Reproduction**: run each script/notebook from its own folder;
  artifacts in `experiments/artifacts/<experiment>_<timestamp>_<sha>/`
  (`model.pkl` / `model.joblib` / `SavedModel/` / `pip_freeze.txt`); fixed
  seeds recorded in MLflow (`seed`, `git_sha`, `run_timestamp`).
- **Structural validation** of notebooks: `python scripts/validate_notebooks.py`
  (external notebooks marked as `EXT`).
- **Runtime conventions** (CPU vs GPU) are in each group README.

## How to navigate

1. Open the README of the experiment folder (index links above).
2. For the full technical details (code, executed notebooks),
   enter the corresponding folder: `experiments/<experiment>/`.
3. In the aggregated history: dashboard (`dashboard/index.html`) and `mlflow ui`.

---

*This repository is a living diary of Data Science discoveries.*