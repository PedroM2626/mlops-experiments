# Tabular Regression: Feature Engineering and Price Prediction

> **Area:** Tabular regression / local AutoML
> **Task:** Continuous-variable prediction (real-estate price, car price)
> **Main metric:** R2, MAE, RMSE
> **Status:** Completed
> **Datasets:** California Housing (20.640 samples), price-prediction-multiple-linear-regression (205 samples)

---

## 1. Abstract

This group brings together three complementary tabular regression studies:
(i) a systematic study of **10 feature engineering techniques** on California
Housing with LinearRegression, LightGBM and RandomForest; (ii) the evolution of
a **car price prediction** pipeline (v1 -> v2 -> v3) up to the practical
plateau of R² = 0,9489; and (iii) a local open-source equivalent to the
IBM Watsonx AutoML. The cross-cutting conclusion: feature engineering has
**model-specific value** (large for linear models, marginal for
tree models) and PCA was largely harmful.

## 2. Context and Objectives

- Quantify the **isolated effect** of feature engineering on models from
  different families (linear vs. tree) on the same dataset.
- Show the evolution of a simple price pipeline into a pipeline with
  regularization, encoding, target transformation and tuning (v1 -> v3).
- Provide a 100% open-source alternative to the IBM Watsonx AutoML running
  locally.

## 3. Theoretical Background (brief)

- Linear models (OLS, Ridge) capture only linear relationships; polynomial
  features/transformations extend expressiveness without switching model and,
  by affine invariance, scaling does not change the OLS result.
- Trees (RF/LightGBM) learn nonlinearities natively and are scale
  invariant; domain-knowledge (geo) features add what
  univariate splits cannot derive.
- PCA maximizes variance, not correlation with the target -- risk of destroying
  directional information in collinear features.

## 4. Methodology

### 4.1 Tabular Feature Engineering (California Housing)

| Factor | Detail |
|---|---|
| Dataset | California Housing, 20.640 x 8 features |
| Techniques | Raw, Standardized, MinMax, Polynomial(d=2), Interactions, Log, Binning, PCA(95%), Geo, Combined |
| Models | LinearRegression, LightGBM, RandomForest |
| Metrics | R², MAE |
| Seed / HW | 42; Intel i7, 16 GB |

### 4.2 Price Prediction (205 samples)

- v2: drops `ID`, one-hot of 9 categories (23 -> 42 features), `log1p` of the
  target (skewness 1,78 -> 0,46), winsorization, GridSearchCV on 6 models,
  CV 5-folds.
- v3: polynomial (741 feats), ExtraTrees, RF variant, GradientBoosting to
  try to beat the plateau.

### 4.3 IBM Watsonx local (California Housing, holdout 10%)

- Baselines (Ridge, Lasso, ElasticNet, RF, ET, GB, AdaBoost, SVR, XGB) + FLAML
  (Bayesian AutoML) + TPOT (genetic AutoML).

### 4.4 Reproduction

```bash
jupyter nbconvert --to notebook --execute feature-engineering-tabular.ipynb
jupyter nbconvert --to notebook --execute price-prediction-multiple-linear-regression.ipynb
jupyter nbconvert --to notebook --execute ibm-watsonx-local-automl.ipynb
```

## 5. Results

### 5.1 Feature Engineering - R² by model

| Technique | LinearRegression | LightGBM | RandomForest |
|---|:--:|:--:|:--:|
| Raw | 0,5758 | 0,8360 | 0,8051 |
| Polynomial (d=2) | 0,6457 | 0,8346 | 0,7968 |
| Log transform | 0,6114 | 0,8360 | 0,8053 |
| Geo features | 0,5945 | **0,8418** | **0,8205** |
| Combined | **0,7112** | 0,8375 | 0,8045 |
| PCA (95%) | 0,4877 | 0,6583 | 0,6422 |

### 5.2 Price Prediction (test)

| Model | Test R² | MAE | CV R² | Overfit |
|---|---|--:|--:|--:|
| Random Forest (GS) | **0,9489** | 1.043,7 | 0,8897 | 0,0372 |
| XGBoost (GS) | 0,9391 | 1.316,2 | 0,8931 | 0,0576 |
| ElasticNet (GS) | 0,8978 | 1.424,3 | 0,8801 | 0,0194 |
| Ridge (GS) | 0,8968 | 1.461,6 | 0,8823 | 0,0188 |
| Linear Regression | 0,8900 | 1.676,8 | 0,8423 | 0,0478 |

v1 -> v2 evolution: R² 0,8517 -> 0,9489; MAE -56,7%. v3 (poly/ExtraTrees):
no approach beat the v2 plateau (limiting factor = dataset size).

### 5.3 IBM Watsonx local (holdout)

| Method | RMSE | R² | Time |
|---|--:|--:|--:|
| XGBoost | 0,4618 | 0,8401 | 1,58s |
| FLAML (CatBoost) | 0,4780 | 0,8286 | 63,9s |
| TPOT | 0,4817 | 0,8260 | 199,1s |
| Extra Trees | 0,4997 | 0,8128 | 1,12s |

## 6. Discussion

- Feature engineering has model-specific value: LinearRegression gained +13,5 pp
  (R²) with Combined; LightGBM only +0,6 pp (Geo). Scaling does not change OLS
  (affine invariance). PCA lost 9-18 pp on every model.
- Price prediction: log of the target + encoding + CV took Linear Regression
  from 0,8517 to 0,8900; ensembles beat linear models by ~5 pp; v3 confirmed
  that the limiting factor is the dataset size, not model complexity.
  The RF residuals are normal (Shapiro p=0,09; Jarque-Bera p=0,48).
- Local AutoML: manual XGBoost beat FLAML/TPOT by a small margin; AutoML
  is a good automatic baseline.

## 7. Conclusions and Recommendations

1. Make the FE effort proportional to the model family: linear models justify
   hours, trees minutes (focus on domain knowledge).
2. For price-prediction, Random Forest (v2) is the recommended final model;
   more data would be the next step.
3. Use PCA with caution: it optimizes variance, not correlation with the target.
4. AutoML (FLAML/TPOT) is an automatic baseline; a well-tuned XGBoost remains
   competitive and much faster.

## 8. References and Files

- `feature-engineering-tabular.ipynb` -- tabular FE study.
- `price-prediction-multiple-linear-regression.ipynb` -- v1->v3 pipeline.
- `california-house-regression.ipynb` -- California Housing baselines and FE.
- `ibm-watsonx-local-automl.ipynb` -- local equivalent of the Watsonx AutoML.
- Model x FE cross-study in the root README (Feature Engineering section).