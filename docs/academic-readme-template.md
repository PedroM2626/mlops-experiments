# Academic Per-Experiment README Template

> Reference template for experiment documentation. Each experiment
> must have its own `README.md` immediately inside its folder in
> `experiments/<experiment>/`, following the structure below (compact
> article, in Portuguese).

---

## Required structure

```markdown
# <Experiment Title>

> **Area:** <NLP | Time Series | Computer Vision | Regression | RecSys | ...>
> **Task:** <Classification | Regression | ...>
> **Primary metric:** <F1-macro, R², MAE, AUC-ROC, ...>
> **Status:** <Completed | In progress>
> **Datasets:** <source and size>

## 1. Abstract
<3-5 lines: the problem investigated, the proposed method, the main result
and a one-sentence conclusion. No tables.>

## 2. Context and Objectives

<What motivated the study; research questions; hypotheses (if any);
reference to earlier work/problems that motivated it.>

## 3. Theoretical Background (brief)

<key concepts needed to understand the experiment: representations, algorithms,
metrics. No extensions; only connecting theory to the study's decision.>

## 4. Methodology

### 4.1 Data
- Source, size, number of classes/features, validation split.

### 4.2 Preprocessing
- Cleaning, transformations, feature engineering, outlier handling.

### 4.3 Compared methods
- Table with the model/paradigm, strategy and relevant configuration.

### 4.4 Evaluation
- Metrics, validation protocol (holdout/CV/temporal), seeds, hardware.

### 4.5 Reproduction
- Command(s) and/or path to the notebooks.
- Output pattern: `experiments/artifacts/<experiment>_<timestamp>_<sha>/`.

## 5. Results

<comparative tables and figures with the REAL values obtained; state the seed
and the run date. Never invent values.>

## 6. Discussion

<interpretation of the results, comparison between methods, limitations and
possible sources of bias.>

## 7. Conclusions and Recommendations

<practical bullet points + a choice recommendation for usage scenarios.>

## 8. References and Files

- Link to the notebook/scripts/artifacts (relative paths).
- Bibliographic references where applicable (short APIBT/APA).
```

---

## Composition rules

1. **Truthfulness**: numbers/figures must reflect real runs present in the
   notebooks/outputs. If a value is not available, write "to be
   measured/TBD" and **never** invent one.
2. **Language**: Brazilian Portuguese.
3. **Paths**: always relative to the README's directory (e.g. `por-ramal: ./feature_selection_ea.py`).
4. **Tables**: use Markdown tables for comparative results.
5. **Reproducibility**: take care of the "Reproduce" section — it must let
   anyone run the experiment locally.
6. **Index links**: the root README index points to
   `experiments/<experiment>/README.md`.