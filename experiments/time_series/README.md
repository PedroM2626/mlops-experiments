# Time Series and Forecasting (Forecast)

> **Area:** Time Series
> **Task:** Forecasting (regression), direction classification, anomaly detection and knowledge distillation
> **Primary metric:** MAE (and RMSE/MAPE/sMAPE, F1, Accuracy, macro-F1 depending on the experiment)
> **Status:** Completed
> **Datasets:** temperature (UCI), weekly per-store sales (`fato_vendas` — ~6.5M transactions), hourly electricity consumption (MWh), daily Melbourne temperature, CO₂ Mauna Loa, Nile, Sunspots, GunPoint/ArrowHead/ECG5000 (UEA), synthetic GBM price + headlines (TS+NLP), Electric Production (Watsonx) and synthetic sales (Databricks).

## 1. Abstract

This folder gathers **12 experiments** of forecasting and time-series analysis: **Prophet optimization with Optuna**, the **Prophet vs LightGBM** head-to-head (MAE 1.7344 vs 1.96), the evolution of the **sales forecast** V2→V2.1→V2.2 (MAE 1.4218), **knowledge distillation** (LSTM→TCN at 103.9% of the Teacher's performance), **anomaly detection** (Z-Score F1 0.9954), **time-series classification** in 6 paradigms, a **benchmark of 4 paradigms × 4 scenarios**, the **TS+NLP** fusion for market targeting, the conversion of the forecast into **direction classification** and **probabilistic forecasting with DeepAR** (GluonTS/PyTorch). Central conclusion: no paradigm dominates universally — each model family wins on series whose structure favours it (SARIMA for smooth series, ROCKET for classification, Z-Score/Prophet for conservative anomalies); DeepAR does not beat baselines on point accuracy in short univariate series, but it offers native probabilistic forecasting (100 sampled trajectories, confidence intervals).

## 2. Context and Objectives

The study was born from the need to answer **which approach forecasts time-series data best** — classical statistics (SARIMA/ETS), tree-based Machine Learning (LightGBM) or Deep Learning (TCN, LSTM) — under different series structures (trend, seasonality, noise, structural breaks). The local notebooks were also created as **open-source equivalents** of the commercial cloud pipelines of IBM Watsonx AutoAI/autoai-ts-libs and Databricks AutoML, enabling local experimentation without cloud credentials.

Research questions:
- Can a calendar-based additive model (Prophet) compete with a Gradient Boosting fed by lags/rolling?
- Which paradigm wins per series type (smooth trend, noisy trend, long cycle, regime with jumps)?
- Does knowledge distillation work with both neural-network and tree teachers?
- How to convert a forecasting problem into a direction-classification problem with interpretable metrics?

## 3. Theoretical Background (brief)

- **Prophet (Meta):** additive model (GAM) decomposed into trend (changepoints), seasonality (Fourier) and holidays; tunable via `changepoint_prior_scale` and `seasonality_prior_scale`.
- **SARIMA/SARIMAX:** parametric statistical model that captures autocorrelation and seasonality; here with fixed order (1,1,1) to avoid 12 seasonal fits.
- **TCN (Temporal Convolutional Network):** dilated convolution at multiple scales, effective for long periods.
- **LightGBM:** gradient boosting of trees with lags, rolling windows and calendar (feature engineering).
- **ROCKET:** 10k random convolution kernels + a linear Ridge classifier; **DTW + 1-NN**: elastic baseline.
- **Knowledge Distillation:** transfer of smoothed *soft targets* from a Teacher to a smaller Student.
- **Diebold–Mariano (DM):** statistical test of equal predictive accuracy (small-sample / Newey-West correction).
- **Anomaly detection:** Z-Score, Prophet loss bands, Isolation Forest, Elliptic Envelope, LOF on the residuals.

## 4. Methodology

### 4.1 Data
- **Forecast:** daily temperatures (UCI); weekly per-store sales (`sales`), ~6.5M transactions from 2022; hourly electricity consumption.
- **Anomalies:** 3.650 days of daily Melbourne temperature, with 3% simulated contamination = 109 real anomalies.
- **UEA classification:** GunPoint (2 classes, 200 samples), ArrowHead (3 classes, 211 samples), ECG5000 (5 classes, 5.000 samples).
- **4×4 benchmark:** Mauna Loa CO₂ (weekly, trend+seasonality; H=30), Nile (annual, decline; H=8), Sunspots (annual, cyclic ~11 years, H=25), Synthetic (regime changes + jumps, H=30).
- **TS+NLP:** synthetic price via Geometric Brownian Motion (drift 12%, volatility 25%, 1.260 trading sessions) + financial headlines.
- **Forecast→Classification:** `fato_vendas.parquet` aggregated into a daily series, 80/20 time split (training until Oct 21, test Oct 22–Dec 31, 71 days).

### 4.2 Preprocessing
- Prophet: Bayesian tuning (Optuna) of `changepoint_prior_scale` and `seasonality_prior_scale`, `multiplicative` mode, guided by MAE via Time Series Cross-Validation.
- LightGBM: feature engineering — lags of 1/4/52 weeks, rolling windows, sine/cosine cyclic features.
- Forecast→Classification: winsorization at P99 (a 29.4M daily-sale anomaly on 11/09/2022; median ≈ 72K), binary target `qtd(t+1) > qtd(t)`, lag/momentum/moving-average/calendar features.
- Anomalies: techniques applied to the **residuals** of the decomposition.

### 4.3 Methods compared
| Experiment | Methods |
|---|---|
| Prophet and Optuna (forecast) | Prophet baseline vs Prophet+Optuna |
| Prophet vs LightGBM | LightGBM with FE vs Prophet |
| Sales forecast V2→V2.1→V2.2 | LightGBM with ~21 → 23 → 32 features, Optuna with pruning, MLflow (the LightGBM+CatBoost attempt failed) |
| Distillation | LSTM+Attention (Teacher, 1.44M) → TCN (Student, 228k); LGBM Deep (1.500 est.) → LGBM Shallow (50 est.) |
| Anomalies | Z-Score, Prophet (99.9% interval), Isolation Forest, Elliptic Envelope, LOF |
| Classification 6 paradigms | 1-NN+DTW, ROCKET, InceptionTime, TSFresh+RF, Transformer Encoder, LightGBM+FE |
| Benchmark 4×4 | SARIMA, Prophet, TCN, LightGBM |
| TS+NLP | LightGBM: TS-only, NLP-only, TS+NLP |
| Forecast→Classification | Logistic, Random Forest, XGBoost, LightGBM |
| Local equivalents | Prophet+Optuna, SARIMA, ETS, Naive (local Watsonx/Databricks) |
| Probabilistic DeepAR | DeepAR (GluonTS/PyTorch) vs SARIMA, Prophet, LightGBM + probabilistic metrics (Coverage, CRPS) |

### 4.4 Evaluation
- Chronological splits (no shuffle), 80/20 training/test.
- Metrics: MAE, RMSE, MAPE, sMAPE, Accuracy, F1, macro-F1, AUC-ROC, Precision, Recall; DM with Newey-West correction (MSE).
- Benchmark: seed 42 (numpy/torch), proportional H by power; hardware: Intel i7, 16GB RAM, RTX 4070 Laptop (CUDA 12.1).

### 4.5 Reproduction
The notebooks already contain the embedded outputs. To re-run them (avoid this, they are deterministic analyses already executed):

```powershell
# From experiments/
jupyter nbconvert --to notebook --execute time_series/ibm-watsonx-local-timeseries.ipynb --inplace
jupyter nbconvert --to notebook --execute time_series/databricks-forecast-local-equivalent.ipynb --inplace
```

Output pattern: `experiments/artifacts/<experiment>_<timestamp>_<sha>/`.

## 5. Results

### 5.1 Prophet vs LightGBM (daily temperatures, holdout)
| Model | MAE |
|---|---|
| **LightGBM (lags+rolling+calendar)** | **1.7344** |
| Prophet | 1.96 |

### 5.2 Sales Forecast evolution (weekly per-store sales)
| Version | MAE | Δ (vs V2) | Main changes |
|---|---|---|---|
| V2 (objective) | 2.5769 | — | ~21 features, only 2 categorical |
| V2.1 (MLOps) | 2.2340 | −13.3% | 23 features, 5 categorical, MLflow |
| V2.2 (current) | **1.4218** | **−44.8%** (−36.3% vs V2.1) | 32 features, 10 dimensional categorical, price/trend/volatility, 32 pruned trials, MLflow+Docker+10 Pytest tests |

Frustration: `log1p` on the target made the MAE worse, at **2.7094** (log scale optimizes relative error → conservative model); the CatBoost ensemble consumed >23GB RAM and >30h without finishing.

### 5.3 Knowledge Distillation (hourly electricity consumption)
| Approach | Teacher | Student-KD | Student without KD | Result |
|---|---|---|---|---|
| Neural (LSTM→TCN) | 893.20 MW (1.44M params) | **858.72 MW** (228k) | 939.31 MW | Student-KD keeps **103.9%** of the Teacher |
| Trees (LGBM 1.500→50 est.) | — | 148.28 MW | **146.13 MW** | Distillation failed |

The simple LightGBM (50 estimators), with MAE **146.13 MW**, beat the LSTM network (893.20 MW) by ~6× at an almost null computational cost.

### 5.4 Anomaly Detection (Melbourne temperature, 109 anomalies)
| Technique | F1 | Precision | Recall (anomalies) |
|---|---|---|---|
| **Z-Score (residuals)** | **0.9954** | 100.0% | 99.1% (108/109, 0 false alarms) |
| Isolation Forest (residuals) | 0.9863 | 98.2% | 99.1% (108/109, 2 false) |
| Elliptic Envelope (residuals) | 0.9863 | 98.2% | 99.1% (108/109, 2 false) |
| Local Outlier Factor (LOF) | 0.0183 | — | 2/109 (108 false) |

Note: Prophet with a 99.9% interval obtained F1 0.9860, Precision 100.0% and Recall 97.2% (106 correct anomalies, no false positive).

### 5.5 Paradigm Benchmark: MAE per Dataset (v2) — winners
| Dataset | SARIMA | LightGBM | Prophet | TCN | Winner |
|---|---|---|---|---|---|
| CO₂ | **0.40** | 0.53 | 1.47 | 0.58 | **SARIMA** |
| Nile | **95.09** | 110.23 | 137.35 | 101.61 | **SARIMA** |
| Sunspots | 45.76 | 56.05 | 43.62 | **25.02** | **TCN** |
| Synthetic | 7.59 | 5.94 | **5.27** | 5.71 | **Prophet** |

Training time (s): SARIMA **34.87** (CO₂)/79.49 (Synthetic); LightGBM 0.48–0.64; **Prophet 0.14–0.45**; TCN 0.16–2.44. Ranking: SARIMA 1st on CO₂ and Nile; TCN 1st on Sunspots; Prophet 1st on Synthetic.

### 5.6 Diebold–Mariano significant (p<0.05)
Prophet vs TCN: significant on **3/4** datasets; SARIMA vs LightGBM on Sunspots (p<0.001); SARIMA vs Prophet on CO₂ and Synthetic; p=1.000 indicates nearly identical errors (e.g. CO₂ SARIMA vs LightGBM 0.40 vs 0.53, but the error correlation makes the test inconclusive).

### 5.7 Time-Series Classification (6 paradigms)
- **GunPoint (2 cls, 200 samples):** ROCKET **1.000**/1.000 in 1.2s; Transformer 0.967; 1-NN+DTW 0.917; LightGBM+FE 0.833; TSFresh+RF 0.783; InceptionTime 0.767 (14s).
- **ArrowHead (3 cls, 211 samples):** ROCKET **0.953**/F1 0.657 in 1.8s; InceptionTime 0.766; 1-NN+DTW 0.578; TSFresh+RF 0.516; LightGBM+FE 0.484; Transformer collapsed to 0.094.
- **ECG5000 (5 cls, 5.000 samples):** ROCKET **0.889**/F1 0.487 in 33.1s; Transformer 0.878; 1-NN+DTW 0.841 (36min!); LightGBM+FE 0.820; InceptionTime 0.751; TSFresh+RF 0.720.

### 5.8 TS + NLP (synthetic market, 80/20 time split)
| Model | Accuracy | F1 |
|---|---|---|
| NLP-only | **0.730** | 0.683 |
| TS+NLP | 0.718 | **0.720** |
| TS-only | 0.492 | 0.504 |

### 5.8b TS+NLP on real data — 8-K × D+1 direction (`run_tsnlp_edgar.py`)

Real pilot (the synthetic one had signal by construction): 179 8-K filings
(SEC EDGAR, no auth) from AAPL/MSFT/NVDA/AMZN/META/TSLA/JPM, 2024-01–2026-08;
FinBERT sentiment on the body of the filing (512 tokens); target = up/down of the
ticker in the next trading session; 178 valid events, 70/30 time split
(test ≈ 54 events — wide CI, read with caution):

| Model | Accuracy | F1 | AUC |
|---|---|---|---|
| TS/logreg | 0.537 | 0.000 (single class) | 0.500 |
| TS/lgbm | 0.482 | 0.482 | 0.484 |
| NLP/logreg | 0.500 | 0.542 | 0.510 |
| NLP/lgbm | 0.500 | 0.342 | 0.485 |
| TS+NLP/logreg | 0.500 | 0.542 | 0.510 |
| TS+NLP/lgbm | 0.444 | 0.423 | 0.444 |

Everything ≈ coin-flip — the opposite of the synthetic case (NLP-only 0.730). Interpretation:
in the synthetic series the signal existed by construction (lagged news → return);
in real large-cap filings, sentiment + 5-day momentum fail to beat the
market — consistent with informational efficiency at D+1. Limits: small test
(n≈54), FinBERT truncated at 512 tokens, no control of the filing time
(after-close vs intraday). Artifacts:
`experiments/artifacts/tsnlp_edgar_20260908_194545/` (+ `run_tsnlp_real.py`
documents the GDELT attempt, blocked by the 429 rate-limit).

### 5.9 Forecast→Classification (out-of-sample test, 71 days) — higher = better
| Model | Accuracy | Bal. Acc | Prec | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---:|
| **Logistic** | **0.958** | 0.957 | 0.917 | 0.957 | 0.936 | **0.967** |
| Random Forest | 0.944 | 0.936 | 0.913 | 0.913 | 0.913 | 0.952 |
| XGBoost | 0.944 | 0.936 | 0.913 | 0.913 | 0.913 | 0.942 |
| LightGBM | 0.930 | 0.914 | 0.909 | 0.870 | 0.889 | 0.945 |

Baselines: majority class 66.9% | persistence (d+1) 67.7% | same day last week 42.2%. Top features: `is_weekend`, `dow` and `lag_7` concentrate ~59% of the importance in the Random Forest.

### 5.10 Local equivalents (open-source vs cloud)
**Watsonx local** (`ibm-watsonx-local-timeseries.ipynb`, Electric Production, 20-month holdout, MAPE):
| Method | RMSE | MAPE | Time |
|---|---|---:|---:|
| Prophet+Optuna (100 trials) | **3.5583** | 3.90% | 21.6s |
| SARIMA | 3.5648 | 3.90% | 0.91s |
| Prophet (baseline) | 3.6134 | 4.04% | 0.09s |
| ETS | 3.6412 | 4.08% | 0.05s |
| Naive | 19.0495 | 20.55% | 0.00s |

**Databricks local** (`databricks-forecast-local-equivalent.ipynb`, synthetic sales, 14-day holdout, sMAPE):
| Method | RMSE | sMAPE | Time |
|---|---|---:|---:|
| Prophet+Optuna (50 trials) | **7.7510** | **5.66%** | 11.3s |
| Prophet (baseline) | 8.0511 | 6.39% | 0.17s |
| ETS | 9.6679 | 6.51% | 0.29s |
| SARIMA | 9.9620 | 6.71% | 3.11s |

### 5.11 Probabilistic DeepAR (4 datasets from the benchmark, 100 samples, CPU)

Re-run on 08/09/2026 (GluonTS 0.17, seed 42; `deepar-probabilistic-forecast.ipynb`
with updated outputs):

**Point forecast (MAE):**
| Dataset | SARIMA | Prophet | LightGBM | DeepAR | Winner |
|---|---|---|---|---|---|
| CO₂ | 4.27 | **0.61** | 1.19 | 2.01 | Prophet |
| Nile | 123.01 | **120.12** | 127.24 | 137.94 | Prophet |
| Sunspots | 44.90 | — (failed: `Overflow in int64 addition` on annual dates) | 16.89 | **15.22** | DeepAR |
| Synthetic | 8.33 | 4.20 | 4.68 | **4.09** | DeepAR |

**Probabilistic metrics (DeepAR):**
| Dataset | Coverage(90%) | AvgWidth | CRPS |
|---|---:|---:|---:|
| CO₂ | 90.0% | 6.35 | 2.41 |
| Nile | 75.0% | 370.52 | 164.97 |
| Sunspots | 80.0% | 87.16 | 26.43 |
| Synthetic | 93.3% | 18.37 | 6.35 |

DeepAR won 2/4 on MAE in this re-run (Sunspots, Synthetic) — stochastic
training varies between runs (in the previous run: 0/4). Coverage follows the
pattern: good on long weekly series (90–93%), weak on the short annual ones
(75–80%). Cost: 10–54 s per dataset (GPU available for Lightning; Prophet
0.1–2 s). Conclusion maintained: DeepAR is relevant for probabilistic
forecasting on long/multiple series, it does not replace baselines on short
univariate series — but with re-training it can tie/win on point MAE.

### 5.12 Conformal recalibration — measured on real series (`run_deepar_conformal.py`)

Split-conformal with a rolling pool (6 origins; `n_cal_points` below), nominal 90%:

| Dataset | n_cal | Coverage before | After (global) | After (per-h) | Width before → per-h |
|---|---|---|---|---|---|
| CO₂ | 180 | 1.00 | 1.00 | 1.00 | 8.1 → 13.9 |
| Nile | 48 | 0.375 | 0.75 | 0.75 | 326.7 → 429.2 |
| Sunspots | 150 | 0.12 | 0.96 | **0.92** | 31.2 → 613.0 |
| Synthetic | 90 | 0.30 | 1.00 | 0.77 | 11.6 → 137.2 |

Measured lessons (not only the protocol):
1. **A single window of H points is not enough** (first attempt: CO₂ 0.43→0.57) — the rolling pool (48–180 points) is what unlocked the correction.
2. **Per-horizon ≈ nominal with less width than global** (Sunspots 0.92 with 613 vs 0.96 with 671; Synthetic 0.77 with 137 vs 1.00 with 192).
3. **Conformal is a band-aid, not a cure**: the q_global of 27–35× on Sunspots/Synthetic shows that the native σ is useless in these regimes — the model needs a review, not only a rescaling.
4. **Test granularity limits**: Nile H=8 only allows coverages in steps of 1/8 (0.75 = 6/8) — it will never hit exactly 0.90.
Artifacts: `experiments/artifacts/deepar_conformal_20260908_130049/metrics.json`.

## 6. Discussion

- **Prophet vs LightGBM challenge:** in abrupt daily noise, trees that "read" the immediate lags react better than additive equations based only on the static calendar (MAE 1.73 vs 1.96). High-cardinality categoricals, `preco_medio_unitario` and price features concentrated ~65% of the gain in the V2.2 pipeline.
- **Benchmark:** SARIMA wins 2/4 scenarios even with fixed order (1,1,1), but it costs 35–80s on long series; TCN only wins on long cycles (Sunspots), where 6 dilated blocks capture the ~11-year periodicity; LightGBM never wins but is consistent (2nd–3rd) and good when there are exogenous features; Prophet wins regime changes because the changepoint detection absorbs the structural jumps that break the ARIMA memory.
- **Classification:** ROCKET dominates through the high-dimensional random projection (max + ppv of 10k kernels), linear Ridge; DTW is a robust baseline but impractical at scale (O(N²·L²); 36 min on ECG5000); Transformer collapses on ArrowHead (0.094) for lack of data (147 training × 251 steps); the low macro F1 on the multiclass datasets indicates confusion in the minority class.
- **Distillation:** the transfer of soft targets works for dense models (Student-KD up to 103.9% of the Teacher), but it fails when the Teacher is a tree, because the teacher overfits and produces point predictions identical to the ground truth, cancelling the smoothing (146.13 vs 148.28 MW).
- **Anomalies:** statistical methods (Z-Score/Prophet) are conservative and ideal when false alarms are costly; Isolation Forest is better on Recall (≈108/109) with contamination calibration; LOF fails because of spatial density in a 1D cloud of residuals clustered close to zero — it requires a previous decomposition into residuals or lag windows.
- **TS+NLP and Forecast→Classification:** the lagged causal feature (news of the day → tomorrow's return) dominates in the synthetic case; in real series, TS+NLP tends to beat both modalities in isolation. Classifying direction converts error metrics into interpretable F1/AUC, with strong calendar dominance (weekend ≈ 2.8% of daily sales).
- **Local equivalents:** SARIMA reproduced the Prophet+Optuna result on Electric Production (identical MAPE 3.90%, 24× faster); Prophet+Optuna wins on the synthetic sales (sMAPE 5.66%, +11.4% vs baseline), beating SARIMA/ETS on complex weekly patterns.
- **Probabilistic DeepAR:** the autoregressive deep-learning model (LSTM + Student-T) did not beat baselines on point accuracy in any of the 4 datasets — Prophet won 3/4, LightGBM won 1/4. The DeepAR differentiator is native probabilistic forecasting: 100 sampled trajectories, calibrated confidence intervals (Coverage 90% ≈ 93–100% on long weekly series; undercoverage on short annual series with <350 obs). Computational cost 50–600× higher than the baselines (CPU). Recommended for multiple correlated series (cross-learning) or when the predictive distribution is a business requirement.

## 7. Conclusions and Recommendations

- **Pure statistical forecast:** use SARIMA for smooth series (noise <5%); Prophet when there are structural breaks / holidays; TCN for long cyclic patterns (10k+ points); LightGBM with FE when the parametric assumption does not hold and exogenous features exist (e.g. sales).
- Although LightGBM MAE 1.7344 > Prophet 1.96 on daily temperature, the sales model went from 1.7344 → 2.5769 → 2.2340 → 1.4218 via 32 features + Bayesian HPO with pruning; `log1p` must **not** be applied to an L1/MAPE target.
- Prefer Z-Score/Prophet for costly alerts; IsolationForest for maximizing recall; avoid LOF on the raw series.
- For classifying series: ROCKET is a fast and dominant baseline; InceptionTime for >10k samples; DTW for <200; LightGBM+FE with SHAP when interpretability matters.
- In a forecasting problem, consider converting it into direction classification when the natural scale makes `MAE` hard to interpret.
- The open-source equivalents replace Watsonx/Databricks on the experimental line with comparable metrics.

## 8. References and Files

Notebooks (in this same folder):
- [`temperature_forecasting_prophet.ipynb`](temperature_forecasting_prophet.ipynb) and [`property-sales-time-series.ipynb`](property-sales-time-series.ipynb) — Prophet and Optuna, sales V2→V2.2.
- [`knowledge_distillation-time_series.ipynb`](knowledge_distillation-time_series.ipynb) — neural/tabular distillation.
- [`exp4_anomaly_detection.ipynb`](exp4_anomaly_detection.ipynb) — 5 anomaly techniques.
- [`benchmark-ts-paradigms.ipynb`](benchmark-ts-paradigms.ipynb) — 4 scenarios × 4 architectures.
- [`time-series-classification.ipynb`](time-series-classification.ipynb) — 6 paradigms, UEA datasets.
- [`stock-sentiment-ts-nlp.ipynb`](stock-sentiment-ts-nlp.ipynb) — TS+NLP.
- [`forecast-classification.ipynb`](forecast-classification.ipynb) — direction forecasting.
- [`ibm-watsonx-local-timeseries.ipynb`](ibm-watsonx-local-timeseries.ipynb) — Watsonx equivalent.
- [`databricks-forecast-local-equivalent.ipynb`](databricks-forecast-local-equivalent.ipynb) — Databricks equivalent.
- [`multivariate-time-series-var.ipynb`](multivariate-time-series-var.ipynb) — Vector Autoregression (VAR) and Impulse Response Functions.
- [`hierarchical_forecast.ipynb`](hierarchical_forecast.ipynb) — hierarchical time-series forecasting with bottom-up reconciliation.
- [`deepar-probabilistic-forecast.ipynb`](deepar-probabilistic-forecast.ipynb) — probabilistic DeepAR (GluonTS/PyTorch) vs point baselines.
- [`deepar-generative/deepar-generative-futures.ipynb`](deepar-generative/deepar-generative-futures.ipynb) — DeepAR as a generative model: 500 trajectories, scenarios, probabilities.
- `sktime_vs_hybrid_ts.ipynb` — comparison of the library vs a custom reference hybrid TS.

References: Taylor & Letham, "Forecasting at Scale" (Prophet, 2018); Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (2020); Bromet et al., ROCKET (2020), Diebold & Mariano (1995); Hinton et al., Distilling the Knowledge (2015); UEA Archive, GunPoint/ArrowHead/ECG5000.