# Sales Forecasting Model - Hackathon 2025

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completo-success)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2.svg)

This repository contains the complete solution for the sales forecasting challenge of Hackathon 2025. The project implements a state-of-the-art Machine Learning pipeline to forecast weekly product demand per store, using a Gradient Boosting model (LightGBM) meticulously optimized for maximum accuracy and robustness.

---

## Project Objective

The main objective of this project is to develop a sales forecasting system (`forecast`) for the first five weeks of 2023, based on the 2022 transaction history. The solution aims to optimize stock replenishment, minimizing stockouts and excess, and to provide a solid data basis for the company's strategic decision-making.

---

## Applied Methodology (V2.2 Architecture with MLOps)

The solution was developed iteratively, evolving from a base model into a sophisticated pipeline that incorporates best practices from the Data Science industry:

1.  **Comprehensive Feature Engineering (32 features):** 32 features were created from the raw data, exhaustively exploring all available dimensional tables:
    * **Dimensional Categorical Features (10):** `pdv`, `sku`, `categoria_pdv`, `premise` (On/Off), `categoria`, `subcategoria`, `tipos`, `label`, `marca`, `fabricante`.
    * **Cyclic and Calendar Features (4):** `semana`, `trimestre`, `seno_semana`, `cosseno_semana`.
    * **Temporal Lag Features (7):** Quantity lags at 1, 2, 3, 4, 12 and 52 weeks, and a lag of the average unit price.
    * **Trend Features (2):** Difference between consecutive lags (`lag_diff_1`) to capture short-term momentum, and the coefficient of variation (`coef_variacao_4`) for relative volatility.
    * **Rolling Window Features (11):** Moving mean, standard deviation, maximum and minimum over windows of 4, 12 and 52 weeks, with `min_periods=1` to avoid data loss.
    * **Monetary Value Feature (1):** `preco_medio_unitario` -- gross revenue divided by quantity sold, capturing the product's price positioning.

2.  **MLOps Traceability (MLflow):** The entire training cycle is logged in MLflow, including:
    * Individual hyperparameters (model type, learning rate, num_leaves, etc.).
    * Performance metrics (MAE, training time, dataset size).
    * Artifacts (`.joblib` model, feature importance chart `.png`).

3.  **Bayesian Optimization with Early Pruning (Optuna):** Bayesian search with `MedianPruner` and `LightGBMPruningCallback`. Unpromising trials are aborted after 5 iterations, drastically saving computational time. In V2.2, 20 of the 30 trials were pruned automatically, completing the search in ~6 minutes (vs ~6 minutes of the final training).

4.  **Containerization and Automated Tests:** Pipeline packaged in Docker and covered by 10 unit tests via Pytest, covering: feature engineering, training, forecasting, persistence (round-trip) and chart generation.

5.  **Predictive Submission Strategy:** The final file respects the 1.5 million row limit by selecting the (store, product) combinations with the highest future sales potential predicted by the optimized model itself.

---

## Repository Structure

The project is organized as follows to guarantee modularity and clarity:

```
/
├── artifacts/                  # Trained model (.joblib) and charts (.png)
├── mlruns/                     # MLflow metadata and logs
├── data/
│   ├── raw/                    # Raw input data (.parquet)
│   └── processed/              # Generated final forecasts (.parquet)
├── scripts/
│   ├── forecaster_class.py     # Main pipeline class (SalesForecasterV2)
│   ├── train.py                # Training script with Optuna and MLflow
│   ├── predict.py              # Forecast generation script
│   ├── forecaster_sktime.py    # sktime flavour of the same forecaster
│   ├── train_sktime.py         # sktime training driver
│   ├── coldstart_metadata.py   # cold-start features (category/brand metadata)
│   ├── ae_valid.py             # AE exp.: baseline vs naive embeddings
│   ├── ae_valid2.py            # AE exp.: baseline vs naive vs causal
│   └── ae_cluster.py           # AE exp.: clustering of series (k=3,5,8)
├── strategic_visualization/    # Post-hoc charts over the saved forecasts
│   ├── 01_momentum_analysis.py
│   ├── 02_performance_by_category.py
│   └── 03_global_heatmap.py    # geocodes PDVs, then plots the global heatmap
├── Predictive_Sales_Pipeline.ipynb      # end-to-end pipeline walkthrough
├── ae_embedding_experiments.ipynb       # AE experiment write-up (markdown only)
├── decomposition_vs_regression.ipynb    # decomposition vs LightGBM
├── imputation_experiments.ipynb         # lag/rolling imputation + missing-data robustness
├── rl_proxy_sales_full.ipynb            # RL hyper-parameter search on the full proxy
├── tests/
│   └── test_forecaster.py      # 10 automated tests with Pytest
├── Dockerfile                  # Docker image for environment isolation
└── requirements.txt            # Dependencies with exact versions
```

---

## How to Run the Pipeline

The process is divided into two main stages: training and forecasting. Run the scripts from the terminal, in the project root folder.

**1. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run Automated Tests:**
```bash
pytest tests/ -v
```

**3. Train the Model:**
```bash
# Train the LightGBM model with Optuna (30 trials with Pruning)
python scripts/train.py --n_trials 30
```
At the end, the `sales_forecaster_v2_final.joblib` file and the `feature_importance.png` chart will be created in the `artifacts/` folder.

**4. Generate the Final Submission File:**

* **To generate the SUBMISSION file (limited to 1.5M rows):**
    ```bash
    python scripts/predict.py
    ```

* **To generate the COMPLETE forecast (Optional):**
    ```bash
    python scripts/predict.py --full_forecast
    ```

**5. (Optional) Run via Docker:**
```bash
docker build -t sales-forecaster .
docker run -v $(pwd)/data:/app/data -v $(pwd)/artifacts:/app/artifacts sales-forecaster
```

---

## Results and Academic Comparison (V2 vs V2.1 vs V2.2)

The model was evaluated on a temporal hold-out validation set (weeks >= 48 of 2022), simulating the forecast of unknown future data. The table below documents the quantitative evolution across the three versions of the architecture:

| Metric / Architecture | V2 (Base) | V2.1 (MLOps) | V2.2 (Current) |
|---|---|---|---|
| **MAE (Loss)** | 2.5769 | 2.2340 | **1.4218** |
| **Relative MAE reduction** | -- | -13.3% vs V2 | **-44.8% vs V2** |
| **Incremental reduction** | -- | -- | **-36.3% vs V2.1** |
| **Number of features** | ~21 | 23 | **32** |
| **Categorical features** | 2 (pdv, sku) | 5 (+categ, marca, categ_pdv) | **10** (+subcateg, tipos, label, premise, fabricante) |
| **Price features** | 0 | 0 | **2** (preco_medio_unitario, lag_1_preco) |
| **Trend features** | 0 | 0 | **2** (lag_diff_1, coef_variacao_4) |
| **n_estimators (final)** | 1000 | 500 | **1000** (with early_stopping=50) |
| **Training time** | Hours | ~10 min | **~12 min** |
| **Optuna trials** | 100 (no pruning) | 20 (with pruning) | **30** (with pruning, 20 pruned) |
| **MLflow tracking** | No | Basic | **Complete** (params, metrics, artifacts, plots) |
| **Automated tests** | 0 | 2 | **10** |

### Analysis of the Improvement Factors

The **44.8%** MAE reduction between V2 and V2.2 is attributed to the following factors, in estimated order of impact:

1. **Complete dimensional categorical features** (~40% of the gain): the inclusion of `subcategoria` (42 values), `tipos` (22 values), `label` (14 values), `premise` (On/Off) and `fabricante` (343 values) allowed LightGBM to learn demand patterns specific to each product segment and store type. LightGBM handles categoricals natively via histogram-based splitting, avoiding the need for one-hot encoding.

2. **Average unit price feature** (~25% of the gain): the `preco_medio_unitario` variable (gross_value / quantity) captures the product's price positioning, a strong predictor of sales volume according to the price elasticity of demand theory.

3. **Trend and volatility features** (~20% of the gain): `lag_diff_1` (short-term momentum) and `coef_variacao_4` (relative volatility) give the model information about the direction and stability of recent demand, complementing the level features (lags and moving averages).

4. **Expanded Optuna search space** (~15% of the gain): the inclusion of `min_split_gain` as a hyperparameter and the wider range of `n_estimators` (200-800) and `max_depth` (5-15) allowed Optuna to find configurations better suited to the new feature space.

### The Failed Experiment with the Logarithmic Transformation (log1p)

During the experiments toward V2.3, we tested applying the `log1p` transformation (natural logarithm of 1 + x) to the `target` (`quantidade`), a common technique for extremely skewed data (median=2, but max>90000). The idea was to stabilize the gradient.

However, when evaluating the model on the original scale (via `expm1`), we observed that the MAE jumped drastically (worsened) from ~1.42 to **2.7094**. 
**Why does this happen?** When optimizing `regression_l1` (MAE) over `log(y)`, the model essentially minimizes the *Percentage Error* (MAPE). This makes the model extremely conservative, punishing deviations on small sales while underestimating large sales (e.g.: an error from 1000 to 900 yields a small log deviation, but a giant absolute deviation of 100). Since the official metric is absolute MAE, this transformation was tested, mapped and deliberately **excluded** from the final solution. The flag `use_log_target=False` guarantees that the model always trains on the original scale.

### The CatBoost Bottleneck and the Focus on LightGBM (V2.2)

An attempt was made to scale the model into an Ensemble mixing **LightGBM** with **CatBoost** (V2.3). However, including CatBoost proved unfeasible for a pipeline without hardware acceleration (GPU). Due to the enormous data volume (5.6 million rows) and the high cardinality of 10 categorical variables (e.g.: `fabricante` has 343 unique categories), CatBoost consumed **more than 23 GB of RAM** and monopolized the CPU for more than 30 hours/core without even finishing the initial baseline.

For this reason, the process was aborted in favor of our V2.2 model (purely LightGBM). LightGBM demonstrated a daunting superiority in computational efficiency in this project, managing to process the same thick categoricals via *histogram-based splitting* and to generate a 100% optimized model with Optuna in only **~12 to 15 minutes**, keeping the state of the art and saving infrastructure.

### The Architectural Showdown: Sktime vs High Cardinality (OOM Crash)

During the testing and framework evaluation phase, we conducted a rigorous experiment to compare our manual Feature Engineering based on `pandas.groupby().rolling()` (native in C/Cython) against the automated `WindowSummarizer` solution from the acclaimed **`sktime`** library.

**The Stress Test (Panel Data):**
`sktime` was instantiated using the hierarchical MultiIndex structure (`['pdv', 'sku', 'semana']`) on our base of 5.6 million transactional records from 2022. The empirical result revealed a critical vulnerability of the library for high-cardinality data:
1. **Out Of Memory (OOM) Crash:** the internal *split-apply-combine* method of `sktime` multiplies and inflates matrices in memory when instantiating each temporal grouping. The execution consumed 100% of the RAM (exceeding Hypervisor limits and collapsing Python instantly) while trying to process the hundreds of thousands of store and product combinations, proving itself **Not-Scalable**.
2. **Time Assessment:** under a severe artificial subsampling of **only 50.000 rows**, `sktime` required ~3 minutes to extract the features. Extrapolating linearly (although the memory complexity is superlinear), the full 5.6M base would require more than 5,5 uninterrupted hours just for the Feature Engineering stage, contrasting with the few minutes of our Pandas solution.
3. **Mathematical Equivalence:** extracting only the "Best-Selling Store" (a single store generating ~6.554 perfectly serial records), the showdown was fair. Sktime was very fast (1.37s) and produced the same statistical result as Pandas (MAE of **2.516** in Pandas vs **2.544** in Sktime), proving that the bottleneck is purely architectural (memory management of High-Cardinality Panel Data), and not algorithmic.

**Academic Conclusion:** `sktime` is the state of the art for univariate, low-cardinality time series. However, for massive hierarchical Transactional Dataframes (MLOps in corporate production), the optimized C/Cython vector routine of Pandas that we architected in **V2.2** is unquestionably superior and shielded against hardware bottlenecks.

### Autoencoder Experiments (V2.4 - exploration)

We tested the use of **Autoencoders (AE)** to extract latent representations of the time series and use them as features or for clustering. The objective was to assess whether a non-linear compression of the temporal profile of each series (pdv, sku) would add predictive information to LightGBM. Full details in the `ae_embedding_experiments.ipynb` notebook.

**AE Architecture:** series matrix (709.667 series × 47 weeks) of `log1p(quantidade)` → StandardScaler → MLP (47 → 32 → 8 → 32 → 47), 8-dimensional bottleneck extracted manually via forward pass.

**Results (reproduced baseline: MAE = 1.4247):**

| Approach | MAE val | Δ vs baseline |
|-----------|---------|---------------|
| + AE naive (leaky) | 1.7131 | **+20.24%** (worse) |
| + AE causal | 1.4227 | -0.14% (neutral) |
| k=3 global+cluster_id | 1.5087 | +5.89% (worse) |
| k=5 global+cluster_id | 1.4653 | +2.85% (worse) |
| k=8 global+cluster_id | 1.4786 | +3.78% (worse) |
| k=3 per-cluster | 1.5188 | +6.60% (worse) |
| k=5 per-cluster | 1.5276 | +7.22% (worse) |
| k=8 per-cluster | 1.5376 | +7.92% (worse) |

**Why it did not work:**

1. **Data leakage in the naive variant:** using the embedding of the complete weeks 1-47 as a feature for all rows meant that a week-30 row started to "see" weeks 31-47 (the future). `best_iter` collapsed from ~1000 to 38 — a classic sign of a model learning future information during training. **Lessons: temporal embeddings require a causal mask to be legitimate features.**
2. **Redundancy with existing features:** with the causal variant (correct, without leakage), the result was neutral (-0.14%). The champion LightGBM already captures the temporal profile via `lag_4`, `lag_52`, `rolling_mean_4/12/52` — the AE only compresses the same information.
3. **Clustering adds nothing:** in all configurations tested (k=3, 5, 8 × cluster_id feature or per-cluster models), the result worsened. `cluster_id` is redundant with the dimensional categoricals already present (`categoria`, `marca`, etc.), and per-cluster models fragment the training sample — small clusters (e.g.: 1.849 series in k=8) produced weak models (MAE 8.91 in the worst cluster).

**Conclusion:** AE embeddings add no predictive value to an already well feature-engineered model. The promising path that remains is the use of AEs on **categorical metadata** for **cold-start** (forecasting new series with no history) — implemented below.

### Cold-start with metadata (`scripts/coldstart_metadata.py`, 08/09/2026)

No lags, no history: split by combo (835k training / 209k new combos), only
categoricals + calendar + price. MAE on the new combos (1,25M rows):

| Approach | MAE cold |
|---|---|
| Global average | 11,07 |
| Average per (categoria_pdv, categoria) | 9,68 |
| **LightGBM metadata** | **5,87** |

−47% vs the global average; far from the champion with lags (1,42) — expected, it is another
task (new series, zero history). Top features: `marca`, `preco`,
`categoria_pdv`. Use: initial estimate for a SKU/store with no history until it accumulates
lags (then it migrates to V2.2). Limit: `best_iter` hit the cap (2000, still
improving slowly). Artifacts: `experiments/artifacts/sales_coldstart_20260908_131710/metrics.json`.

---

## Technologies Used

* **Language:** Python 3.8+
* **Packaging Environment:** Docker & Pytest
* **Main Libraries:**
    * Pandas 2.0.3 (Vectorized Data Manipulation)
    * **LightGBM 4.6.0** (main and optimized Gradient Boosting)
    * **Optuna 4.5.0 & PruningCallback** (Bayesian Optimization with pruning)
    * **MLflow 2.17.2** (Tracking, Model Management and MLOps Governance)
    * Scikit-learn 1.3.2 (Metrics)
    * Matplotlib 3.7.2 (Feature Importance Visualization)
    * Joblib 1.4.2 (Artifact Serialization)

---

## Authors - Team: BSB Data 01

* **Erick Cardoso Mendes (developer)**
* **Pedro Morato Lahoz (reporter)**

---

## License

This project is licensed under the MIT License.
