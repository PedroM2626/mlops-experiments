# Consolidated Experiment Summary: Sentiment Analysis (Senti-Pred)

This document unifies the results, optimizations and learnings obtained across all variations of the Senti-Pred project, from Transformer baselines to high-performance Ensembles and AutoML.

## 1. Context and Evolution
We explored the Twitter dataset (4 classes: *Irrelevant*, *Negative*, *Neutral*, *Positive*) through multiple approaches. The journey showed that, for this dataset, data refinement and robust linear models outperform complex Deep Learning architectures.

## 2. Performance Comparison (Consolidated Results)

| Model / Experiment | Text technique | Primary metric (Macro-F1 / Acc) | Notes |
| :--- | :--- | :--- | :--- |
| **🏆 Senti-Pred-remake2** | **TF-IDF (100k) + 4-grams** | **97.80%** | **Record**: Voting (LinearSVC + LR). |
| **God Mode (Remake 1)** | TF-IDF (50k) + Punct | 97.50% | Voting (Passive Aggressive + LR). |
| **Ultimate (Remake 1)** | TF-IDF (40k) + Char Rep | 97.00% | Focus on aggressive error correction. |
| **FLAML (AutoML) V3** | TF-IDF (30k) + 1-2 n-grams | 96.73% | Best AutoML: RandomForest in 5 min. |
| **Insane Mode** | Chi2 Feature Selection | 96.20% | Stacking Classifier (mild overfitting). |
| **Logistic Regression** | TF-IDF (20k) + Regex | 96.00% | Extremely stable linear baseline. |
| **LinearSVC** | TF-IDF (Standard) | 95.00% | Excellent for sparse spaces. |
| **KNN** | TF-IDF (Standard) | 95.00% | Fast non-parametric approach. |
| **MultinomialNB** | Trigrams + Sublinear TF | 92.06% | Optimized via logarithmic search of Alpha. |
| **Random Forest** | Optuna (Deep search) | 91.00% | Jump from 71% -> 91% after HPO. |
| **Classic (LR Baseline)** | TF-IDF (10k) | 87.20% | First robust model with the full dataset. |
| **Baseline RoBERTa** | Transformer (Pre-trained) | ~60.00% | Slow and little data (1k sample). |

## 3. Data Engineering Optimizations
The biggest performance differentiator came from preprocessing ("Data-Centric AI"):

- **Sentiment-Aware Cleaning**: Preservation of emotional punctuation (!, ?) and expansion of contractions.
- **Noise Normalization**: Use of Regex to remove URLs and mentions and handle repeated characters (e.g: "loooove" -> "love").
- **Extreme Vectorization**: Use of n-grams (up to 4-grams) and vocabularies of up to 100k features to capture contextual nuances.
- **Parallelization**: Use of `joblib.Parallel` (15 cores) for lemmatization and large-scale cleaning.

## 4. Highlights by Approach

### 🤖 AutoML (FLAML)
The FLAML framework proved to be the best tool for rapid prototyping, reaching **96.73%** in just 300 seconds and automatically selecting `RandomForestClassifier` as the winner.

### 🏛️ Linear Models and Ensembles
- **Voting Ensemble**: The combination of `LinearSVC` and `LogisticRegression` (or Passive Aggressive) proved to be the most stable, eliminating individual errors through "democratic voting".
- **Passive Aggressive**: Used in *Ultimate* mode for its ability to learn quickly from classification errors, making it ideal for large-scale datasets.

### 🧠 Deep Learning vs. Classic
The initial attempt with **RoBERTa** showed that, without massive hardware and time for fine-tuning on the full dataset, classical statistical models are more efficient and accurate for this specific tweet domain.

## 5. Architecture and MLOps
- **Modularization**: Each variation isolated in its own directory to avoid dependency conflicts.
- **Traceability**: Integration with **MLflow** and **DagsHub** for logging hyperparameters, metrics and artifacts.
- **Persistence**: Use of custom wrappers (`Pipeline` + `LabelEncoder`) saved via `joblib` to guarantee inference identical to training.

## 6. Next Steps
- Implement a unified Streamlit interface to compare the models in real time.
- Carry out deployment via Docker to guarantee reproducibility in any environment.
- Explore LLMs (via API or quantized) for zero-shot level sentiment analysis.

## 7. Rigorous comparison A vs B vs C (what-ifs)
- See `experiments/nlp/pipelines_abc_comparison/README.md`: canonical reproduction of the three
  pipelines and controlled ablations of n-grams, vocabulary, preprocessing and model.
- Key findings: removing bigrams costs −2.6 to −4.8 pp; reducing C from 4-grams to
  bigrams improves +0.33 pp; C's vectorizer (100k) is the best component, and
  A preprocessing + C vectorizer reaches 0.9857 Macro-F1; differences < 1 pp between
  pipelines are not significant (McNemar).
