# Senti-Pred — Sentiment Analysis Pipeline Variations

> **Domain:** NLP
> **Task:** Sentiment classification (4 classes: Irrelevant, Negative, Neutral, Positive)
> **Primary metric:** Macro-F1 / Accuracy
> **Status:** Completed
> **Datasets:** Twitter (Twitter Entity Sentiment — `twitter_training.csv` / `twitter_validation.csv`)

## 1. Abstract

This folder consolidates the **variations of the Senti-Pred project**, unifying results, optimizations and lessons learned from Transformer baselines through high-performance ensembles and AutoML. On the tweet dataset (4 classes), the journey showed that data refinement and robust linear models outperform complex Deep Learning architectures. The record was achieved by **Senti-Pred-remake2** (voting LinearSVC + LogisticRegression over TF-IDF 100k + 4-grams) with **97.80%** Macro-F1/Accuracy.

## 2. Context and Objectives

The same tweet dataset was explored through multiple approaches — pre-trained Transformers (RoBERTa), deep networks, linear models, ensembles, AutoML (FLAML) and "Data-Centric AI" data engineering — to investigate:

- Which text representation (n-grams, vocabulary, cleaning) maximizes F1.
- How much preprocessing matters vs. the model architecture.
- How MLOps (MLflow/DagsHub, modularity, persistence) supports the evolution of the variations.

Central hypothesis: for tweets, **a feature pipeline + robust linear models outperform transformer fine-tuning** when massive hardware is unavailable.

## 3. Theoretical Background (brief)

- **TF-IDF / n-grams** — sparse representation with up to 4-grams and vocabularies of up to 100k features to capture sentiment context.
- **Voting Ensemble** — democratic combination of classifiers (LinearSVC + LogisticRegression, or Passive Aggressive) to eliminate individual errors.
- **Passive Aggressive** — online algorithm that learns quickly from errors, ideal for large scale.
- **FLAML** — fast AutoML framework (300s) for prototyping.
- **MLflow / DagsHub** — traceability of hyperparameters, metrics and artifacts; persistent wrappers (`Pipeline` + `LabelEncoder`) so inference matches training.

## 4. Methodology

### 4.1 Data
- `senti-pred-exp1/data/raw/twitter_training.csv` (training) and `twitter_validation.csv` (validation training).
- 4 classes: *Irrelevant*, *Negative*, *Neutral*, *Positive*.

### 4.2 Preprocessing (Data-Centric AI)
- **Sentiment-aware cleaning:** preservation of emotional punctuation (`!`, `?`) and expansion of contractions.
- **Noise normalization:** Regex removes URLs and mentions and handles repeated characters (e.g.: `"loooove"` → `"love"`).
- **Extreme vectorization:** n-grams up to 4-grams, vocabularies of up to 100k features.
- **Parallelization:** `joblib.Parallel` (15 cores) for lemmatization and large-scale cleaning.

### 4.3 Compared methods
From the TF-IDF 10k + LR baseline model through KNN, LinearSVC, MultinomialNB, Random Forest (Optuna), stacking (Chi2 + feature sel.), FLAML AutoML and voting ensembles; as well as RoBERTa (transformer baseline). Variations isolated in two subfolders:

- `Senti-pred-exp1/` — complete pipeline: `src/scripts/01_eda.py` →
  `src/scripts/04_evaluation.py`, the Django API (`src/api/views.py`,
  `src/api/urls.py`) and containerization (`Dockerfile`, `docker-compose.yml`).
- `Senti-Pred-remake2/` — Pipeline C as a package: `src/data/preprocess.py`
  (vectorizer-centric cleaning) and `src/models/train.py` / `src/models/predict.py`.
- `Senti-Pred-remake2/` — remake with modular `src/` + `data/raw/`.

### 4.4 Evaluation / MLOps
- Primary metric: Macro-F1/Accuracy; **MLflow/DagsHub** integration.
- **Persistence:** wrappers (`Pipeline` + `LabelEncoder`) saved via `joblib` for identical inference.
- **Modularization**: each variation isolated in its own directory to avoid dependency conflicts.

### 4.5 Reproduction
- See `EXPERIMENTS_SUMMARY.md` (consolidated summary) and the structure of each subfolder (`senti-pred-exp1/`, `Senti-Pred-remake2/`).
- Pipelines: `senti-pred-exp1/src/scripts/01_eda.py` … `04_evaluation.py`; Docker instructions in `senti-pred-exp1/Dockerfile`/`docker-compose.yml`.
- Versioned training logs: `senti-pred-exp1/training_log{,_v2..v7}.txt`.

## 5. Results

| Model / Experiment | Text technique | Primary metric (Macro-F1/Acc) | Notes/Config |
| :--- | :--- | :--- | :--- |
| **🏆 Senti-Pred-remake2** | TF-IDF (100k) + 4-grams | **97.80%** | Record: Voting (LinearSVC + LR) |
| God Mode (Remake 1) | TF-IDF (50k) + Punct | 97.50% | Voting (Passive Aggressive + LR) |
| Ultimate (Remake 1) | TF-IDF (40k) + Char Rep | 97.00% | Aggressive error correction |
| FLAML (AutoML) V3 | TF-IDF (30k) + 1-2 n-grams | 96.73% | Best AutoML: RandomForest in 5 min |
| Insane Mode | Chi2 Feature Selection | 96.20% | Stacking Classifier (mild overfitting) |
| Logistic Regression | TF-IDF (20k) + Regex | 96.00% | Stable linear baseline |
| LinearSVC | TF-IDF (Standard) | 95.00% | Excellent for sparse spaces |
| KNN | TF-IDF (Standard) | 95.00% | Non-parametric, fast |
| MultinomialNB | Trigrams + Sublinear TF | 92.06% | Logarithmic search of alpha |
| Random Forest | Optuna (deep search) | 91.00% | Jump from 71% → 91% after HPO |
| Classic (LR Baseline) | TF-IDF (10k) | 87.20% | First robust model (full dataset) |
| Baseline RoBERTa | Transformer (pre-trained) | ~60.00% | Slow and little data (1k sample) |

### Highlights by approach

- **AutoML (FLAML):** 96.73% in 300 seconds; selected `RandomForestClassifier`.
- **Voting ensembles:** the LinearSVC + LogisticRegression combination (or Passive Aggressive) is the most stable.
- **Passive Aggressive:** learns quickly from errors, ideal for large scale (*Ultimate* mode).
- **RoBERTa:** without massive hardware and time for fine-tuning on the full dataset, classical statistical models are more efficient for this task.

## 6. Discussion

The comparison shows a clear hierarchy: **more n-grams + more vocabulary + good cleaning** lift classical systems from 87.2% (baseline) to **97.8%** (record), while RoBERTa stayed at ~60% for lack of data/hardware. Voting among robust linear models (SVC + LR) was the key factor for the record. The *Data-Centric* choices (URL removal, character n-grams and a 100k vocabulary) outperformed modeling complex architectures. Limitations: FLAMB picks RandomForest, but the linear ensembles won with more features; mild overfitting was reported in *Insane Mode* (stacking with Chi2).

## 7. Conclusions and Recommendations

- **Prioritize rich sparse features (TF-IDF, up to 4-grams, 100k) + voting of linear models** as the best cost vs. return for this tweet domain.
- For **rapid prototyping**, `AutoML (FLAML)` is sufficient in 5 min (96.73%).
- **Transformers** are only worth it with hardware and the full dataset (see `../nlp/README.md` for fine-tuning).
- **Next steps:** Streamlit interface to compare models in real time; deploy via Docker for reproducibility; test zero-shot LLMs (API/quantized).

## 8. References and Files

- `EXPERIMENTS_SUMMARY.md` — consolidated summary (source of this documentation).
- `senti-pred-exp1/` — original pipeline (scripts 01–04, Docker, local MLflow, training logs).
- `Senti-Pred-remake2/` — modular remake with `src/` and raw data.
- Rigorous **A vs B vs C (remake2)** comparison with what-ifs: `../nlp/pipelines_abc_comparison/README.md`
- Similar cases (representation/ensembles, logistic multi): `../nlp/README.md`.