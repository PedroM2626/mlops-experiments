# Feature Engineering for Time Series: 5 Phases

> **Area:** Time Series
> **Task:** Forecasting (regression) with feature engineering
> **Primary metric:** MAE
> **Status:** Completed
> **Datasets:** Daily Minimum Temperatures (univariate), Beijing PM2.5 (multivariate), derived series for DL embeddings, seasonal decomposition + wavelets.

## 1. Abstract

A 5-phase journey in search of the lowest MAE in time series forecasting, answering: **what works better — human intuition, statistical brute force (tsfresh) or advanced algorithms (Deep Learning)?** The work evolves from univariate to multivariate and to signal representations: the final win goes to the **Manual Features + Wavelets (DWT)** combination with MAE **54.19**; in Phase 5, **time embeddings (sine/cosine)** + **Optuna** improved all isolated feature sets, but — paradoxically — made the hybrid model worse through underfitting induced by cross-validation.

## 2. Context and Objectives

The objective was to systematize the evolution of time series feature engineering into a single comparable pipeline, starting from univariate and reaching advanced representations (Deep Learning, wavelets, circular embeddings, HPO). At each phase, the feature representation was decided from the absolute error (MAE) obtained on holdout, checking whether "more automatic features" really helps against controlled manual feature engineering.

## 3. Theoretical Background (brief)

- **tsfresh:** mass automatic extraction of statistical features from series; here it showed a performance drop on short series.
- **LSTM Autoencoder (PyTorch):** learns a compressed latent representation (16-dim vector) that can be used as a tabular feature.
- **Discrete Wavelet Transform (DWT, `pywt`):** extracts the shocks in windows (7 days), separating the magnitude structure from the trend/seasonality.
- **Circular embeddings (Sine/Cosine):** encode months/days so that week 52 is close to week 1.
- **Optuna (TPE)** + `TimeSeriesSplit` (cross-temporal validation) as the Evaluation protocol.

## 4. Methodology

### 4.1 Data
| Phase | Series | Approaches compared |
|---|---|---|
| 1 | Daily Minimum Temperatures (univariate) | tsfresh (automated) vs Manual FE |
| 2 | Beijing PM2.5 (multivariate) | tsfresh (automated, 313 features) vs Manual FE (moving average) |
| 3 | Phase 2 series + latent representation | Manual, DL Embeddings, Hybrid |
| 4 | Phase 2 series + decomposition | Manual + Trend/Seasonality → Wavelet (DWT) |
| 5 | Phase 4 with time embeddings | 4 approaches (Manual, Seasonal Decomp., Wavelets, Full Hybrid) + Optuna |

### 4.2 Preprocessing
- Signal decomposition into Trend/Seasonality before wavelet extraction.
- Lags, rolling windows (moving average), circular embeddings (sine/cosine) for month/day.
- DWT transforms over 7-day windows to capture shocks.

### 4.3 Compared methods (Phase 5)
| Approach | Description |
|---|---|
| 1. Manual FE only | Classic manual features (lags, moving average) |
| 2. Seasonal Decomp. only | Trend + seasonality separated, no wavelets |
| 3. Wavelets only (DWT) | Seasonal decomposition + DWT, no manual features |
| 4. Full Hybrid | Manual + Seasonal Decomp. + Wavelets (62 features) |

### 4.4 Evaluation
- Metric: **MAE** on a time-based holdout; `TimeSeriesSplit` cross-validation (3 folds) in Optuna.
- Optuna ran hyperparameter searches (Random Forest: `n_estimators`, `max_depth`, `min_samples_split`).

### 4.5 Reproduction
Notebooks (in this same folder, outputs already included):
- `automated_vs_manual_fe_ts.ipynb` → Phase 1
- `multivariate_auto_vs_manual_fe.ipynb` → Phase 2
- `dl_embeddings_fe_ts.ipynb` → Phase 3
- `advanced_signal_fe_ts.ipynb` → Phase 4
- `hpo_time_embeddings_ts.ipynb` → Phase 5

```powershell
# Optional run (re-executes the notebook in place):
jupyter nbconvert --to notebook --execute hpo_time_embeddings_ts.ipynb --inplace
```

Output pattern: `experiments/artifacts/<experiment>_<timestamp>_<sha>/`.

## 5. Results

| Phase | Winner (Phase) | MAE | Details |
|---|---|---|---|
| 1 | Random Forest + Manual FE | **1.76** | tsfresh: hundreds of features, ~30 s, MAE 1.79 |
| 2 | Random Forest + Manual FE | **46.07** | tsfresh: 313 features destroying performance; moving average wins |
| 3 | Hybrid (Manual + DL) | **57.24** | LSTM Autoencoder compresses into a 16-dim latent vector |
| 4 | Full Hybrid (Manual + Wavelets) | **54.19** | DWT extracts shocks in 7-day windows |

### Final Duel (Phase 5 — approaches with Optuna vs Phase 4)

| Approach (Phase 5) | Best Params (Optuna) | MAE Phase 5 (with Optuna) | MAE Phase 4 (without Optuna) |
|---|---|---|---|
| **3. Wavelets only (DWT)** | `n_est: 50, depth: 5, min_split: 5` | **56.74** | 57.23 |
| **1. Manual FE only** | `n_est: 200, depth: 5, min_split: 5` | **57.14** | 57.88 |
| **2. Seasonal Decomp. only** | `n_est: 200, depth: 5, min_split: 4` | **59.63** | 60.82 |
| **4. Full Hybrid (Manual + Signals)** | `n_est: 150, depth: 5, min_split: 2` | 55.25 | **54.19** (wins without HPO) |

> [!WARNING]
> The optimization kept the simple models from overfitting; in Optuna's cross-validation, however, it imposed hard regularization on the Hybrid model, which then underfit on the final test.

## 6. Discussion

- **tsfresh is not a silver bullet:** it produces hundreds of features (313 on PM2.5), explodes the dimensionality and worsens the MAE (1.79 vs 1.76; complete destruction on PM2.5). Automatic features without regularized selection do not help short series and degrade the useful information.
- **Deep Learning and wavelets:** the latent embedding of the LSTM Autoencoder (16 dims) is effective but inferior to explicit signal representations; DWT (54.19) beat every previous tactic, capturing the shocks in 7-day windows that static features lose.
- **Cross-validation paradox:** Optuna chose `max_depth = 5` to lower the mean error over the 3 folds, which is enough for the ~20-feature sets but causes underfitting in the 62-feature Hybrid, which needs deeper trees to relate the moving average and the wavelet shock (lost 54.19 → 55.25).
- **Time embeddings + HPO rescue the simple models:** sine/cosine representations + Optuna improved all isolated feature sets (Wavelet 57.23→56.74; Manual 57.88→57.14), showing that regularization (the chosen `max_depth`) avoids memorizing the past and improves generalization.

## 7. Conclusions and Recommendations

- **Use wavelets (DWT) + time embeddings (sine/cosine) as the heart of feature engineering for time series**; this was the crowning result of the project. Signal science > "black box" algorithms.
- **Do not rely on tsfresh as automatic extraction** without dimensionality/regularization control.
- **If you train the Full Hybrid (62 features), do not restrict `max_depth`** (nor use a very high `min_samples_split`) — or use **far more than 10 Optuna trials** so the optimizer discovers that the complexity of the feature set requires deeper trees.
- For simple feature sets, Optuna + time embeddings work well and ensure good generalization in the future.

## 8. References and Files

- [`automated_vs_manual_fe_ts.ipynb`](automated_vs_manual_fe_ts.ipynb) — Phase 1 (univariate).
- [`multivariate_auto_vs_manual_fe.ipynb`](multivariate_auto_vs_manual_fe.ipynb) — Phase 2 (PM2.5).
- [`dl_embeddings_fe_ts.ipynb`](dl_embeddings_fe_ts.ipynb) — Phase 3 (LSTM Autoencoder).
- [`advanced_signal_fe_ts.ipynb`](advanced_signal_fe_ts.ipynb) — Phase 4 (DWT).
- [`hpo_time_embeddings_ts.ipynb`](hpo_time_embeddings_ts.ipynb) — Phase 5 (Optuna + time embeddings).

References: Christ et al., "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh)", 2018 (brief); planned: use of `pywt` (wavelets).