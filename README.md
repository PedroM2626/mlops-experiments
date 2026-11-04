# Machine Learning & MLOps Experiment Repository

A portfolio of ML/MLOps experiments. Each experiment lives in its own folder
under `experiments/<experiment>/` with its **own academic README** (a compact
article: abstract, context, methodology, results, discussion, conclusions and
reproduction). This document is only the **index** that connects everything.

For the documentation standard, see
[`docs/academic-readme-template.md`](docs/academic-readme-template.md).

---

## Experiment Index

### 🧪 NLP — Sentiment Analysis, Topics and Representations

| Experiment | What it does | Main result | Read more |
|---|---|---|---|
| **NLP group** (senti-pred, A/B/C pipelines, Twitter Methods, multiclass Logistic, MMoE, AG News, FE NLP) | Sentiment/topic classification and text representations | TF-IDF + n-grams ~0.98 F1; transformers win in low-data | [see README](experiments/nlp/README.md) |
| **NLP in Regression — Wine (Kaggle)** | wine scoring from text | Ridge MAE 1.33 / R² 0.70 vs LightGBM 1.48 / 0.63 | [see](experiments/nlp-regression-wine/README.md) |
| **Senti-Pred variations** | Variations of the sentiment pipeline | record 97.80% (TF-IDF 100k, 4-grams) | [see](experiments/nlp/twitter-entity-sentiment/senti-pred-variations/README.md) |
| **Twitter Entity Sentiment** (sub-index) | the folder that holds every pipeline of that dataset | A/B/C compared with significance tests | [see](experiments/nlp/twitter-entity-sentiment/README.md) |
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

### 🖥️ Local open-source AutoML / forecast pipelines

| Experiment | Objective | Read more |
|---|---|---|
| **Local equivalents of cloud AutoML** | open-source stand-ins for managed AutoML/forecast services (FLAML, TPOT, Prophet+Optuna, GluonTS); the cloud originals (Watsonx, Databricks) were removed because they could not be executed or reproduced here | [time_series](experiments/time_series/README.md) · [tabular](experiments/tabular_regression/README.md) |

### 🐱🖼️ Computer Vision

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **CV Methods (CIFAR-10)** | HOG+SVM vs ResNet18 vs ViT | ViT 0.9805 vs ResNet 0.9362 vs HOG 0.3970 (0.5381 with the full 50k train set) | [see](experiments/computer_vision/README.md) |
| **Animal multi-label** | 4 approaches (pets Dime/Frida) | ResNet18+aug Exact Match 1.000 | same as above |
| **Face detection/recognition** | LBPH, CNN, YuNet (app in the notebook) | — | [see](experiments/computer_vision/README.md) |
| **Boxing stance detector** | MediaPipe Holistic pose → 19 geometric features → RF/XGBoost/LightGBM, 4 stances, live webcam | XGBoost 0.9875 (79/80), one sample above RF and LightGBM (0.9750) | [see](experiments/computer_vision/boxing_pose_detector/README.md) |

### 🎬 RecSys

| Experiment | Objective | Main result | Read more |
|---|---|---|---|
| **MovieLens RecSys — 10 approaches** | MF, neural networks, autoencoders (AutoRec), similarity, heuristic | Item-AutoRec RMSE 0.9054; Two-Tower 0.9297; SVD 0.9352 | [see](experiments/recommender_systems/README.md) |

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
  `anomaly_detection_enhanced.ipynb`, `ensemble_pyramid.ipynb`) — supporting the READMEs
  above. `ensemble_pyramid.ipynb` also dumps `ensemble_pyramid_best.pkl` (model + TF-IDF +
  label encoder); `*.pkl` is gitignored, so that file comes back only by re-running it.
- **Experiment dashboard**: `dashboard/index.html` (open in the browser).
- **Serving / MLOps stack** for the sales-forecast champion: `mlops/` (FastAPI
  service, drift monitor, retrain, MLflow registry; see
  [`mlops/README_mlops.md`](mlops/README_mlops.md)), packaged by `Dockerfile` /
  `docker-compose.yml` (`ml_api`, `mlflow_ui`, `mlops_monitor`; needs a `.env`
  copied from `.env.example`). `Dockerfile_ensemble` is the ensemble variant.
  `k8s-deployment.yaml` is only a cluster-ready **example**: it points at a
  container-local SQLite file, so its 3 replicas would not share tracking state
  until `MLFLOW_TRACKING_URI` is set to a shared server.

## Environments, and what a fresh clone does not have

Four dependency layers exist on purpose; they do **not** pin the same versions,
because each one records a different moment:

| File | What it is for |
|---|---|
| `requirements.txt` | the analysis environment that produced the recorded results (kept in sync with the working venv; `pip freeze`-verified) |
| `requirements-mlops.txt` | the serving image (`Dockerfile`): FastAPI + MLflow + monitor/retrain, no research stack |
| `requirements_ensemble.txt` | `Dockerfile_ensemble`, the ensemble-serving variant |
| `experiments/sales-forecast/requirements.txt`, `.../senti-pred-exp1/requirements.txt`, `experiments/computer_vision/boxing_pose_detector/requirements.txt` | the project environment **at the time that experiment ran** (e.g. the hackathon ran on MLflow 2.17.2 / numpy 1.24.3, while the root env is on MLflow 3.11.1 / numpy 1.26.4) |

Where a freeze was recorded, it is the ground truth for that result:
`experiments/artifacts/<experiment>_<timestamp>_<sha>/pip_freeze.txt`. Only the
`ag_news_classification_*` and `image_recommender_*` bundles carry one; the other bundles
(`kd_cifar10`, `pipelines_abc`, `hog_cifar10`, `deepar_conformal`, `hier_calibrate`,
`sales_coldstart`, `tsnlp_edgar`) record `seed`/`git_sha`/`run_timestamp` but no
freeze, so for those the root `requirements.txt` is the closest reference. The
`boxing_pose_detector` bundle never froze its environment either; it carries
`logged_model_env_record.txt` instead — the dependency set MLflow recorded
alongside the shipped model.

**Not in git** (`.gitignore` keeps them out; download or regenerate before
reproducing):

- `experiments/sales-forecast/data/` (~134 MB of raw/processed parquet; the Google
  Drive folder is linked from `data/raw/Path_to_normalized_data.txt`);
- `experiments/causal_nlp_olist/data/` (the Olist datasets, from
  https://www.kaggle.com/olistbr/brazilian-ecommerce);
- `experiments/computer_vision/data/cifar-10-python.tar.gz` (downloaded by the
  notebooks on first use) and `experiments/time_series/.tsnlp_cache/`;
- `experiments/computer_vision/boxing_pose_detector/models/holistic_landmarker.task`
  (13.7 MB MediaPipe bundle; the download command is in that experiment's README);
- serialized models: `*.joblib`, `*.pkl` (`sales_forecaster_v2_final.joblib`,
  `ensemble_pyramid_best.pkl`, `tfidf_vectorizer.pkl`, `champion.joblib`,
  `boxing_pose_detector/pose_classifier_model.pkl`);

The one exception to the last two rules is
`experiments/artifacts/boxing_pose_detector_*/model_artifacts/`, where the MLflow
SQLite store and the three logged model pickles are re-included by path and
tracked with Git LFS — that project's standalone repository was deleted, so
without them the recorded runs would have no loadable model or logged
parameters left anywhere.
- MLflow stores: `experiments/mlruns/` (legacy file store, kept browsable by the
  `mlflow_ui` service) and `experiments/mlops_tracking.db` (the SQLite backend
  used by `mlops/`, `MLFLOW_TRACKING_URI` overrides it).

**Libraries some notebooks import but this environment does not have** are listed
commented-out at the bottom of `requirements.txt`, next to the experiment that
needs them (Sentence-BERT, `mord`, `tsfresh`, `aeon`, `scikit-surprise`,
`Boruta-Shap`, `yfinance`, `open-clip-torch`, `mediapipe` for the
`boxing_pose_detector` webcam path, Django + DRF for the `senti-pred-exp1`
API, `auto-sklearn` which needs Linux/WSL). Install them before running those
notebooks; otherwise they fail at import.

Credentials are never committed: copy `.env.example` to `.env` (DagsHub, W&B,
Hugging Face, AWS). Only the DagsHub/W&B tracking paths need them.

## Repository standards

- **License**: MIT, a single `LICENSE` at the repository root; sub-projects carry no
  licenses of their own.
- **Continuous integration**: `.github/workflows/tests.yml` runs the repository guards
  (`scripts/tests`), notebook validation, `mlops/tests` and the RecSys tests on every push
  and pull request, installing only `requirements-mlops.txt`; tests that need the research
  stack (torch, mediapipe, ...) run locally.
- **Docker build context**: `.dockerignore` keeps environments, tracking stores, datasets and
  notebooks out of the images (the ensemble image re-includes the one notebook it executes).
  `Dockerfile.dockerignore` replaces it for the API image only and also drops the rest of
  `experiments/`, keeping just `sales-forecast/` and `artifacts/mlops_sales/`.

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