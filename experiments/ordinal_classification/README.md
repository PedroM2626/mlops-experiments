# Ordinal vs. Nominal Classification — Wine Quality (Red)

> **Area:** Ordinal classification / Tabular
> **Task:** Predict wine quality (wine scores 3–8, 6 ordered classes) as an ordinal vs. nominal problem
> **Metrics:** Accuracy, MAE (ordinal distance), Cohen's Kappa, within-one accuracy (±1)
> **Status:** Done
> **Dataset:** Wine Quality Red (UCI) — 1.599 rows × 12 columns; split 1.199 training / 400 test (seed 42). Synthetic ordinal fallback if the download fails.

## 1. Abstract

Compares 4 approaches on the same split: nominal LogReg, nominal RF, ordinal LogisticAT (`mord`) and ordinal LogisticIT (`mord`). The **nominal RF wins on accuracy (0.6600) and MAE (0.3600)**, but the ordinal models tie on **within-one accuracy (~0.9775)** — their errors are "close". The lesson: for ordered labels, MAE/Kappa/acc±1 count more than exact accuracy; an ordinal model rarely beats the RF on pure accuracy, yet it produces less severe errors.

## 2. Context and Objectives

Nominal classification ignores the order (a 5→8 miss costs the same as 5→6). Ordinal classification penalizes the distance. Questions: (RQ1) does the ordinal model beat the nominal one on MAE/Kappa? (RQ2) does acc±1 reveal practical equivalence?

## 3. Theoretical Background (brief)

- **Nominal (LogReg OvR, RF):** per-class boundaries, no notion of ordinal neighborhood.
- **Ordinal (LogisticAT/IT, `mord`):** cumulative thresholds over a latent score; AT (all-threshold) vs IT (immediate-threshold).
- **OrdinalRandomForest (notebook):** decomposition into K−1 binary classifiers `P(y ≥ k)`, final class = sum of the predictions.
- **Ordinal metrics:** MAE = mean |ŷ−y|; Kappa weights agreement beyond chance; acc±1 = fraction with error ≤ 1 level.

## 4. Methodology

### 4.1 Data

Wine Quality Red: 1.599 samples; distribution by quality: 3:10, 4:53, 5:681, 6:638, 7:199, 8:18 (strong imbalance, rare extreme classes). `y −= y.min()` to 0..5 (`mord` requires 0..K−1).

### 4.2 Compared methods

| Model | Type | Config |
|---|---|---|
| LogReg (Nominal) | nominal | `StandardScaler + LogisticRegression(max_iter=2000)` |
| RF (Nominal) | nominal | `RandomForestClassifier(n_estimators=200, random_state=42)` |
| LogisticAT (Ordinal) | ordinal | `StandardScaler + mord.LogisticAT(alpha=1.0)` |
| LogisticIT (Ordinal) | ordinal | `StandardScaler + mord.LogisticIT(alpha=1.0)` |

### 4.3 Evaluation

Single stratified 75/25 holdout (1.199/400). Metrics: accuracy, MAE, Kappa, acc±1. Figures: bars per metric, confusion matrices, distribution of errors |ŷ−y|.

### 4.4 Reproduction

```bash
jupyter nbconvert --to notebook --execute experiments/ordinal_classification/ordinal_classification.ipynb --inplace
pip install mord scikit-learn pandas matplotlib seaborn
```

## 5. Results

| Model | Type | Accuracy | MAE | Kappa | Acc ±1 |
|---|---|---|---|---|---|
| **RF (Nominal)** | nominal | **0.6600** | **0.3600** | **0.4451** | **0.9800** |
| LogisticIT (Ordinal) | ordinal | 0.5975 | 0.4275 | 0.3285 | 0.9775 |
| LogReg (Nominal) | nominal | 0.5950 | 0.4375 | 0.3286 | 0.9700 |
| LogisticAT (Ordinal) | ordinal | 0.5850 | 0.4400 | 0.3082 | 0.9775 |

## 6. Discussion

- **RF dominates across the board** (acc +0.06, MAE −0.07 vs 2nd): trees capture chemical non-linearities that linear thresholds do not capture.
- **Ordinal models do not beat their nominal counterpart** (LogReg 0.5950 vs LogisticIT 0.5975 — tie; MAE 0.4375 vs 0.4275 — marginal gain). The ordinal gain shows up in acc±1 (0.9775 vs 0.9700): "1-level" errors.
- **Rare classes (3, 8) are almost never predicted correctly** — Kappa 0.31–0.45 reflects this; without rebalancing or weighted ordinal loss, the model collapses to 5/6.
- **Limitations:** single holdout (no CV); 1 seed; linear `mord` (no kernel); no per-class threshold calibration.

## 7. Conclusions and Recommendations

- If the business metric tolerates an error of ±1 level (e.g.: a quality band), **any model works** (≥0.97) — pick the simplest.
- If severe errors are costly, **nominal RF + MAE monitoring** is the best cost-benefit here; a linear ordinal model is only worth it under a monotonic interpretability constraint.
- Next: stratified CV ×5, `class_weight=balanced`, per-class threshold tuning, ordinal RF (K−1 binary) as a middle ground.

## 8. References and Files

- Notebook: `./ordinal_classification.ipynb` (executed, with figures).
- References: Pedregosa et al. (`mord`); UCI Wine Quality (Cortez et al., 2009); Cohen (1960) Kappa; see `docs/modelo-academico-readme.md`.
