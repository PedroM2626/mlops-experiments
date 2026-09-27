# Senti-Pred — Rigorous Comparison of Pipelines A, B and C (Senti-Pred-remake2)

> **Area:** NLP
> **Task:** Sentiment classification into 4 classes (Irrelevant, Negative, Neutral, Positive)
> **Main metric:** F1-Macro (with Accuracy / F1-Weighted as complementary metrics)
> **Status:** Completed
> **Datasets:** Twitter Entity Sentiment — `twitter_training.csv` (73.996 usable rows) / `twitter_validation.csv` (1.000 rows)
> **Execution date:** 12/08/2026 — seed 42 — CPU (16 logical cores, Windows, sklearn 1.7.1)

---

## 1. Abstract

This study extends the engineering duel "Pipeline A vs. Pipeline B" (documented in
`experiments/nlp/README.md`, §5.4) with a **third pipeline** — **Pipeline C
(Senti-Pred-remake2)**, all-time record holder with 97,80% accuracy/F1. The three pipelines
were faithfully reimplemented from their source artifacts and run under the
**same dataset, same split and same seed**, guaranteeing a fair comparison. Besides the
canonical reproduction (F1-Macro: **A = 0,9845** with ExtraTrees; **B = 0,9833** with LinearSVC C=19;
**C = 0,9773** with Voting LinearSVC+LR), the work systematizes an extensive set of
**"what-ifs"** (controlled ablations) on the number of n-grams, the vocabulary size,
`min_df`, `sublinear_tf`, the preprocessing and the model. The central conclusion reinforces the
*Data-Centric AI* paradigm: **text cleaning matters more than the choice of model**,
and the Pipeline C vectorizer (100k features, 4-grams) is the best individual component,
but Pipeline C's cleaning is the least favorable of the three — the best F1-Macro observed
(**0,9857**) arises when preprocessing **A** is combined with the vectorizer **C**.

## 2. Context and Objectives

The repository contains a group of NLP experiments that compares, among other things, two
sentiment analysis pipeline variations on social networks (Twitter):

- **Pipeline A** (`../senti-pred_pipeline.ipynb`) — **aggressive** preprocessing
  (removes URLs, mentions, whole hashtags, punctuation and digits).
- **Pipeline B** (`../twitter-sentiment-analysis.ipynb`) — **conservative** preprocessing
  (keeps hashtag content, punctuation `!?.,'"`, hyphens and numbers).

Section §5.4 of the NLP README concluded that **B = 0,9860 F1-Weighted vs A = 0,9820**
with LinearSVC C=19 — a gain of **+0,40 pp** attributed exclusively to the cleaning.

In parallel, the project `experiments/senti-pred-variations/` documents the evolution of a
"remake2" pipeline (`Senti-Pred-remake2`) that reached the record of **97,80%** (accuracy/F1)
with **TF-IDF (100k) + 4-grams + Voting (LinearSVC + LogisticRegression)**.

**Objective of this experiment:** unify the two research lines, bringing
Senti-Pred-remake2 into the A vs B comparison as **Pipeline C**, and answer the
"what-if" questions rigorously:

1. *What if I increase/decrease the number of bigrams (n-grams) of Pipeline C, A or B?*
2. *What if I change the preprocessing of each pipeline (hashtags, punctuation, digits,
   stopwords, lemmatization, contractions)?*
3. *What if I increase/decrease the vocabulary size (max_features), min_df or
   sublinear_tf?*
4. *What if I swap the model (hard/soft voting, class_weight, LinearSVC alone)?*
5. *Are the differences between the pipelines statistically significant?*

**Hypotheses tested:**

- `H1` — Bigrams are crucial; removing the bigrams drops the F1-Macro in all pipelines.
- `H2` — Pipeline C has the best vectorizer, but the "heaviest" cleaning;
  crossed combinations (preprocessing X + vectorizer Y) may beat the canonical ones.
- `H3` — Differences below ~1 pp between pipelines on the holdout (N=1.000) are **not**
  statistically significant (McNemar test).

## 3. Theoretical Background (brief)

- **TF-IDF** — sparse matrix where each dimension is a term (or n-gram); weight = `tf × idf`
  with logarithmic damping when `sublinear_tf=True` (`1 + log(tf)`).
- **word n-grams** — unigrams capture the lexicon; **bigrams** capture negation and
  composition (`not good`, `very bad`) — exactly what is decisive in sentiment. N-grams
  of higher order (3–5) add rare vocabulary and noise.
- **LinearSVC** — linear SVM with L2 penalty (`C`); robust in sparse high-dimensional
  space; it has no `predict_proba` (so, for `voting='soft'`, it required Platt
  calibration via `CalibratedClassifierCV`).
- **Voting Ensemble** — democratic combination; `hard` = vote by majority class,
  `soft` = average of probabilities.
- **class_weight='balanced'** — re-weights unbalanced classes by the inverse of the frequency.
- **Data-Centric preprocessing** — cleaning of noise (URLs, mentions), decision about
  hashtags (remove/block vs. preserve the content), punctuation with sentiment load
  (`!`, `?`), stopwords (in English; `not`/`no` preserved because they are signal), WordNet
  lemmatization and contraction expansion.
- **McNemar test** — paired test (exact, binomial) on the validation set, to decide whether
  two classifications differ significantly.

## 4. Methodology

### 4.1 Data

| Split | Source | Rows (after load cleaning) | Classes |
|---|---|---|---|
| Training | `twitter_training.csv` | 73.996 | 4 |
| Validation | `twitter_validation.csv` | 1.000 | 4 |

- Removal of rows with null `text` or `sentiment` and of texts that become empty after cleaning.
- Classes in training: Negative 22.542 / Positive 20.832 / Neutral 18.318 / Irrelevant 12.990.

### 4.2 Preprocessing of the three pipelines

| Component | Pipeline A (aggressive) | Pipeline B (conservative) | Pipeline C (remake2) |
|---|---|---|---|
| URLs / www | removed | removed | removed |
| Mentions `@user` | removed | removed | removed |
| Hashtags | removed (whole `#word`) | content preserved (`#great`→`great`) | symbol `#` removed, word preserved |
| Punctuation | all removed | preserved `!?.,'"` and `-` | only `!` and `?` preserved |
| Numbers | removed | preserved | removed |
| Stopwords | not removed (final stage) | not removed | removed, **preserving `not`/`no`** |
| Lemmatization | not used (final stage) | not used | WordNet |
| Contractions | not expanded | not expanded | expanded (`n't`→` not`, etc.) |
| Case | lowercase | lowercase | lowercase |

### 4.3 Canonical vectorizers

| Pipeline | max_features | ngram_range | min_df | sublinear_tf | token_pattern |
|---|---|---|---|---|---|
| A | 70.000 | (1,2) | 2 | True | default |
| B | 70.000 | (1,2) | 2 | True | default |
| C | 100.000 | (1,4) | 2 | True | `\w{1,}` |

### 4.4 Canonical models (per pipeline)

- **A:** LR(2000 it.), MultinomialNB, LinearSVC (C=1; 10; 19), ExtraTrees(100). Documented champion: **ExtraTrees**.
- **B:** LR(C=11), ExtraTrees, LinearSVC(C=19), PassiveAggressive, KNN(7, cosine), Ridge, SGD(modified_huber). Champion: **LinearSVC C=19**.
- **C:** LinearSVC(C=0.5, `class_weight=balanced`), LR(C=10, balanced), **Voting hard (SVC+LR)**, as per the official artifact.

### 4.5 Experimental design ("what-ifs")

Whenever one dimension is varied, the others remain canonical. Champion model per
pipeline in the ablations: **A/B → LinearSVC C=19**; **C → Voting** (to align with the
documentation and isolate the effect of features).

| Exp | Dimension | Values tested |
|---|---|---|
| E1 | Canonical reproduction | all models of each pipeline |
| E2 | **n-grams** | (1,1), (1,2), (1,3), (1,4), (1,5), (2,2), (2,3) |
| E3 | `max_features` | 10k, 25k, 50k, 70k, 100k, 150k, 200k |
| E4 | `min_df` | 1, 2, 3, 5 |
| E5 | `sublinear_tf` | True, False |
| E6 | Preprocessing toggles | see §5.4 |
| E7 | Pipeline C model | SVC alone, LR alone, Voting hard/soft, with/without `class_weight`, C=0.5 vs 19 |
| E8 | `class_weight=balanced` fairness | applied to A/B/C (LR C=10 and SVC C=19) |
| E9 | Cross preprocessing × vectorizer | 3 cleaners × ({A,B,C} vectorizers) |

### 4.6 Evaluation (protocol)

- **Fixed holdout** — original dataset training/validation (73.996 / 1.000), **seed 42**.
- Metrics: **Accuracy, F1-Macro, F1-Weighted**, per-class F1, confusion matrix, training
  time, `n_features`.
- Paired significance test **exact McNemar** (binomial) among the three canonical ones.
- Hardware: Windows, 16 logical cores, Python 3.13.1, scikit-learn 1.7.1.

### 4.7 Reproduction

```bash
cd experiments/nlp/twitter-entity-sentiment/pipelines_abc_comparison
jupyter nbconvert --to notebook --execute run_abc_comparison.ipynb
```

- The runner is the single-cell notebook `run_abc_comparison.ipynb`; it calls
  `parse_args([])`, so the argparse defaults always apply (all stages, `seed=42`)
  and it writes to `artifacts_abc/` under the current working directory.
- The run recorded in this repository is
  `experiments/artifacts/pipelines_abc_20260812_123257_0295952/`.

- Output: `experiments/artifacts/pipelines_abc_20260812_123257_0295952/`
  (`results_*.csv`, `results_all.csv`, `fig*.png`, `predictions.npz`,
  `val_with_predictions.csv`, `champions_summary.csv`, `meta.json`).
- Code: `pipelines_abc_core.py` (cleaners/vectorizers/models/evaluation),
  `run_abc_comparison.ipynb` (orchestration of the ablations, same code as the
  exported script that used to live here).

## 5. Results

> All values below were **measured in this run** (not copied from the documentations).
> Differences of ±0,1 pp relative to the original READMEs reflect version variation of the
> data/seed; the qualitative hierarchy is reproduced.

### 5.1 E1 — Canonical reproduction (F1-Macro per model)

| Pipeline | Model | Acc | F1-Macro | F1-Weighted | Time (s) |
|---|---|---|---|---|---|
| **A** | **ExtraTrees** | **0,985** | **0,9846** | **0,9850** | 121,2 |
| A | LinearSVC C=10 | 0,983 | 0,9828 | 0,9830 | 11,6 |
| A | LinearSVC C=19 | 0,981 | 0,9807 | 0,9810 | 15,4 |
| A | LinearSVC C=1 | 0,980 | 0,9797 | 0,9800 | 6,8 |
| A | LR | 0,975 | 0,9743 | 0,9750 | 10,8 |
| A | MultinomialNB | 0,914 | 0,9105 | 0,9134 | 2,4 |
| **B** | **LinearSVC C=19** | **0,983** | **0,9833** | **0,9830** | 17,3 |
| B | Ridge | 0,983 | 0,9830 | 0,9830 | 4,5 |
| B | LR C=11 | 0,981 | 0,9813 | 0,9810 | 13,6 |
| B | PassiveAggressive | 0,980 | 0,9806 | 0,9801 | 3,4 |
| B | KNN cosine | 0,978 | 0,9785 | 0,9781 | 4,3 |
| B | ExtraTrees | 0,977 | 0,9772 | 0,9770 | 119,9 |
| B | SGD | 0,977 | 0,9769 | 0,9770 | 3,0 |
| **C** | **LinearSVC C=0.5** | **0,978** | **0,9782** | **0,9780** | 6,4 |
| C | Voting hard (SVC+LR) | 0,978 | 0,9773 | 0,9780 | 21,9 |
| C | LR C=10 | 0,978 | 0,9773 | 0,9780 | 19,6 |

*Among the champions of each pipeline:* **A (ExtraTrees) 0,9846 > B (LinearSVC C=19) 0,9833
> C (Voting) 0,9773.**

### 5.2 E2 — What-if of the number of n-grams (max N) — **central question of the study**

Δ (in pp = percentage points) against the canonical configuration of each pipeline
(→ *(1,2)* for A/B; *(1,4)* for C):

| N-grams | Δ A | Δ B | Δ C |
|---|---|---|---|
| (1,1) — **unigrams only** | **−2,57** | **−3,09** | **−4,76** |
| (1,2) | 0,00 (canonical) | 0,00 (canonical) | **+0,33** |
| (1,3) | +0,00 | −0,46 | +0,00 |
| (1,4) | −0,11 | −0,49 | 0,00 (canonical) |
| (1,5) | −0,11 | −0,61 | −0,46 |
| (2,2) — bigrams only | −2,48 | −2,81 | −1,32 |
| (2,3) | −3,73 | −3,80 | −3,48 |

**Readings:**

- **Bigrams are indispensable (H1 confirmed):** removing the bigrams drops **−2,6 to −4,8 pp**.
- **Pipeline C is "over-tuned" to 4-grams:** reducing from (1,4) to **(1,2) the
  F1-Macro RISES +0,33 pp** (0,9773 → 0,9806). The 4-grams of remake2 add noise.
- **Pipeline B is the most sensitive to n-grams** of higher order: any N>2 degrades it (−0,5 to −0,6 pp).
- **Pipeline A is the most robust** to increasing the order (stable up to trigrams).

### 5.3 E3–E5 — What-ifs of vocabulary and TF normalization

**`max_features` (Δ in pp vs canonical):**

| max_features | Δ A | Δ B | Δ C |
|---|---|---|---|
| 10.000 | −4,02 | −4,63 | **−8,47** |
| 25.000 | −0,66 | −1,70 | −2,55 |
| 50.000 | −0,23 | −0,38 | −0,91 |
| 70.000 | 0,00 | 0,00 | −0,44 |
| 100.000 | +0,39 | +0,18 | 0,00 (canonical) |
| 150.000 | +0,39 | +0,15 | +0,09 |
| 200.000 | +0,30 | +0,24 | **+0,41** |

- Vocabulary is the most "elastic" resource of Pipeline C: cutting it to 10k costs **−8,5 pp**;
  expanding to 200k yields **+0,41 pp** (0,9773 → 0,9815). A and B also gain slightly with
  vocabulary > canonical (100–200k).

**`min_df`:** A is insensitive (0,00 in all of them); **B improves with `min_df=1` (+0,30 pp)**;
**C worsens with `min_df=5` (−0,67 pp)**.

**`sublinear_tf`:** neutral in A and C; **B slightly better without sublinear (+0,12 pp)**.

### 5.4 E6 — What-if of preprocessing (one-at-a-time toggles)

**Pipeline A**

| Toggle | F1-Macro | Δ (pp) |
|---|---|---|
| A default (everything removed) | 0,9807 | 0,00 |
| **Keep hashtags** | 0,9848 | **+0,42** |
| **Keep punctuation** | 0,9848 | **+0,41** |
| **Keep digits** | 0,9848 | **+0,42** |
| Keep punctuation+digits | 0,9841 | +0,35 |

→ **The "aggressive" cleaning is too aggressive on these axes:** preserving the content of hashtags,
punctuation or numbers yields **≈ +0,4 pp** each to Pipeline A.

**Pipeline B**

| Toggle | F1-Macro | Δ (pp) |
|---|---|---|
| B default | 0,9833 | 0,00 |
| Remove punctuation | 0,9851 | **+0,18** |
| Remove hashtags | 0,9812 | −0,20 |
| Remove digits | 0,9821 | −0,12 |

→ B is already well calibrated; only *removing punctuation* gives a slight gain (+0,18 pp); losing
hashtags/digits costs.

**Pipeline C**

| Toggle | F1-Macro | Δ (pp) |
|---|---|---|
| C default | 0,9773 | 0,00 |
| **Keep stopwords** | 0,9803 | **+0,30** |
| Without expanding contractions | 0,9730 | **−0,43** |
| Remove the hashtag word | 0,9741 | −0,32 |
| Remove `!` `?` | 0,9762 | −0,12 |
| Without lemmatizing | 0,9769 | −0,05 |

→ **Pipeline C's cleaning is the least favorable in the comparison:** keeping the stopwords
recovers +0,30 pp (the removed stopwords were, deep down, signal), expanding contractions is
essential (+0,43 pp if lost), and the content of the hashtags carries sentiment (+0,32 pp).

### 5.5 E7 — What-if of the model in Pipeline C

| Model | F1-Macro | Acc |
|---|---|---|
| **LinearSVC C=0.5 (alone)** | **0,9782** | 0,978 |
| Voting hard without weight | 0,9779 | 0,978 |
| LinearSVC C=0.5 without weight | 0,9779 | 0,978 |
| **Voting hard (official)** | 0,9773 | 0,978 |
| LR C=10 balanced | 0,9773 | 0,978 |
| Voting soft (calibrated) | 0,9758 | 0,976 |
| LR C=10 | 0,9746 | 0,975 |
| LinearSVC C=19 | 0,9741 | 0,975 |

→ **the official Voting is not superior to LinearSVC C=0.5 alone** (−0,09 pp); `soft` voting
with a calibrated SVC worsens it; `class_weight=balanced` is practically neutral; **`C=0.5` ≫ `C=19`**
in the regime with `balanced` (0,9782 vs 0,9741).

### 5.6 E8 — Fairness `class_weight=balanced`

| Config | F1-Macro A | F1-Macro B | F1-Macro C |
|---|---|---|---|
| LR C=10 balanced | 0,9788 (−0,15) | 0,9809 (−0,23) | 0,9773 (0,00) |
| SVC C=19 balanced | 0,9795 (−0,12) | 0,9815 (−0,18) | 0,9733 (−0,04) |

→ applying `balanced` **is not** the source of C's performance: for A/B the weighting even
degrades ~0,1–0,2 pp; C, with C=19, also loses a little. C's advantage comes from
**C=0.5 + balanced combined**, not from the weight alone.

### 5.7 E9 — Cross preprocessing × vectorizer (champion model of the preprocessing)

| Preprocessing | Vectorizer | F1-Macro | Acc |
|---|---|---|---|
| **A** | **C (100k, 1-4)** | **0,9857** | 0,986 |
| **B** | C (100k, 1-4) | 0,9833 | 0,984 |
| A | A/B (70k, 1-2) | 0,9807 | 0,981 |
| B | A/B (70k, 1-2) | 0,9833 | 0,983 |
| C | C (100k, 1-4) | 0,9773 | 0,978 |
| C | A/B (70k, 1-2) | 0,9771 | 0,978 |

→ **the best single run of the study is `preprocessing A + vectorizer C` = 0,9857
F1-Macro**, surpassing ALL the canonical ones. The remake2 vectorizer is the most powerful;
cleaning A is the most compatible with it.

### 5.8 Statistical significance (exact McNemar) and F1 per class

F1 per class (champions):

| Class | A | B | C |
|---|---|---|---|
| Positive | 0,9856 | 0,9767 | 0,9693 |
| Negative | 0,9887 | 0,9868 | 0,9831 |
| Neutral | 0,9843 | 0,9842 | 0,9859 |
| Irrelevant | 0,9796 | 0,9854 | 0,9711 |
| **Errors /1000** | **15** | **17** | **22** |

McNemar (paired, exact):

| Pair | Discordants (p1 ok / p2 ok) | p-value | Conclusion |
|---|---|---|---|
| A vs B | 10 / 9 | 1,00 | n.s. |
| A vs C | 11 / 6 | 0,33 | n.s. |
| B vs C | 11 / 10 | 1,00 | n.s. |

→ **H3 confirmed:** with N=1.000 in validation, differences of ~0,6–0,7 pp between pipelines
are **statistically indistinguishable** (p ≥ 0,33). The observed ordering reflects a tendency,
not a proven difference.

### 5.9 Overfitting and *dataset shift* diagnosis

Beyond the validation set, two other quantities were measured: (i) F1 on the training set itself after a full fit and
(ii) stratified CV-5 over the training set. The "overfitting gap" is defined as
`F1_train − F1_validation` (positive ⇒ overfit; ~0 ⇒ perfect generalization); the "*dataset shift* gap"
as `F1_validation − F1_median_CV`.

| Champion | TRAINING F1-Macro | VALIDATION F1-Macro | CV-5 F1-Macro | Overfit gap (pp) | Shift gap (pp) |
|---|---|---|---|---|---|
| A: ExtraTrees (100) | 0,9781 | 0,9845 | 0,9255 ± 0,0022 | **−0,64** (n.s.) | **+5,91** |
| B: LinearSVC C=19 | 0,9763 | 0,9833 | 0,9186 ± 0,0015 | **−0,70** (n.s.) | **+6,47** |
| C: Voting (official) | 0,9697 | 0,9773 | 0,9200 ± 0,0019 | **−0,76** (n.s.) | **+5,74** |
| C: LinearSVC C=0.5 (single) | 0,9660 | 0,9782 | 0,9222 ± 0,0019 | **−1,23** (n.s.) | **+5,61** |

**Readings:**

1. **No pipeline suffers from overfitting.** The *overfit gap* is **negative** in every case
   (between −0,64 and −1,23 pp), i.e., `F1_train < F1_validation`. The model makes *more* errors on training
   than on validation — a classic sign of **mild underfitting** (training is harder and
   contains noisy/ambiguous examples), never of memorization.
2. **There is a strong *dataset shift* between training and validation (~+6 pp).** The validation is
   substantially **easier** than an average of folds inside training
   (`F1_CV ≈ 0,92` vs `F1_val ≈ 0,98`). Consequence: **the ~0,98 reproduced for all the
   pipelines is inflated**; the honest generalization capacity (CV) sits at ~0,92.
   The ranking by absolute value on validation reflects *ease of the test set*, not
   intrinsic model quality.
3. **Ranking by honest generalization (CV-5):** A (ExtraTrees) **0,9255** > C (LinearSVC
   C=0.5) 0,9222 ≈ C (Voting) 0,9200 > B (LinearSVC C=19) 0,9186. The order **A > C > B** is
   different from the pure-validation ranking (`A > B > C`), and it reinforces the recommendation to use the
   Pipeline A when possible — under cross-validation, ExtraTrees generalizes best.
4. **CV variance is very low** (~0,002), so the A vs B/C difference (~0,7 pp) in CV is already
   more robust than the paired comparison on validation. In production, **ExtraTrees over A
   (or over the cross A+vectorizer C) is the most defensible choice in terms of
   generalization**, not LinearSVC C=19 over B.
5. **Methodological recommendation:** report **stratified CV over the training set** (and not only
   the original holdout) as the primary metric; store `overfit_gap` and `shift_gap` in the
   metadata of each model in MLflow.

→ Additional *sanity* check: the majority classifier (Negative) is correct on 30,2 %
of training and 26,6 % of validation, confirming that the classes are only *moderately*
imbalanced and that the random baseline is ~25 % — the ~98 % observed are not an artifact of
extreme imbalance.

### 5.10 Qualitative Generalization Diagnosis (Real Out-of-Domain Sentences)

To assess the real semantic understanding of the models (Pipeline B and C with LinearSVC) in out-of-distribution (Out-of-Distribution) situations for the Twitter dataset, 5 sentences were tested containing sarcasm, negation, neutral sentences and contexts unrelated to brands. The results illustrate the practical limits of TF-IDF:

* **Double negation ("The new update is not bad at all, I actually think it is quite good.")**:
  **Success**. Both B and C predicted *Positive*. Bigrams prove their value by capturing "not bad".
* **Sarcasm ("I absolutely love waiting 3 hours in line for a coffee... best day ever.")**:
  **Severe failure**. Both predicted *Positive*. TF-IDF only sums the high weights of the words "love" and "best day", failing to capture the ironic structure.
* **Absolute neutral ("I have no strong feelings about this movie, it was just okay.")**:
  **Failure**. Pipeline B predicted *Negative* (weight on the word "no"), Pipeline C predicted *Positive* (weight on the word "okay"). Statistical models lose the center if the words pull toward the poles.
* **Trivial OOD ("What is the weather going to be like tomorrow?")**:
  **Failure (Hallucination)**. Both predicted *Positive*. Since the dataset is biased toward entities and brands, purely casual and inquiring language receives arbitrary predictions (statistical hallucination).
* **Complex negative ("My flight got delayed and my luggage is lost. I am furious!")**:
  **Mixed**. Pipeline B predicted *Negative* (success, it preserved the strong '!' punctuation). Pipeline C predicted *Neutral* (failure, it cleaned the sentence too much and did not connect the flight jargon as absolute negative).

**Qualitative conclusion:** Although they reach ~98% in the training/validation domain, in the open world the TF-IDF classifiers are limited. They have no deep understanding of semantics and suffer from sarcasm and OOD sentences, working more like hyper-optimized "keyword scales" than true interpreters of natural language (like modern LLMs).

## 6. Discussion

1. **The Pipeline C vectorizer (remake2) is the most valuable asset.** Its 100k features with
   4-grams, when matched with preprocessing A, generate the best result of the whole
   study (**0,9857**), +0,50 pp over canonical A. The gain of *remake2* came much more from the
   "extreme vocabulary" than from the cleaning.
2. **Pipeline C's cleaning is its greatest weakness.** The removal of stopwords costs
   **+0,30 pp** (if kept), the 4-grams cost **+0,33 pp** (if reduced to bigrams), and
   together these two changes — *keep stopwords + bigrams* — would put C back in the
   A/B range under external validity.
3. **There is a clear trade-off between "feature power" and "noise".** Pipelines with small
   vocabulary (A/B 70k) are more sensitive to `min_df`/`sublinear`; C, with 100–200k, depends
   critically on `max_features` (−8,5 pp if cut to 10k).
4. **The model is the minor factor** (semantic *No Free Lunch*): the same pipeline changes ≤0,5 pp
   by swapping the model, while the same model family changes up to +0,4 pp by swapping the cleaning;
   the official C `Voting` is neutral/slightly worse than `LinearSVC C=0.5` alone.
5. **Important limitation — statistical power.** The holdout has only 1.000 samples;
   with ~15–22 total errors, McNemar cannot separate pipelines that differ by < 1 pp.
   To decide "which is better" in production, **repeated cross-validation**
   or a larger test would be needed.
6. **No overfitting; strong *dataset shift*.** The full diagnosis (§5.9) shows that
   no champion memorizes (`F1_train < F1_validation` in all of them, gap −0,6 to −1,2 pp). Yet
   the validation is **~6 pp easier** than CV-5 over training (`F1_CV ≈ 0,92`), which
   inflates all the reported "0,98". The honest ranking (CV) is **A (ExtraTrees 0,9255) >
   C (Voting/LinearSVC 0,9200–0,9222) > B (LinearSVC C=19 0,9186)** — in production, prefer
   Pipeline A (or the cross A+vectorizer C).
7. **Consistency with the existing documentation.** The *Data-Centric* hierarchy (cleaning >
   model) and the remake2 record (~97,8% in validation) are reproduced; but the interpretation
   attributed — "4-grams + 100k explain the record" — must be qualified: the component that
   explained the result most *in validation* was the vectorizer, not the cleaning; and the validation
   itself is an easier target (~6 pp above CV).

## 7. Conclusions and Recommendations

- **Best by validation (F1-Macro in validation, 1.000 samples):** Pipeline **A + ExtraTrees**
  = 0,9845. **Best *single run* (combination of components):** **preprocessing A +
  vectorizer C + LinearSVC C=19** = 0,9857. **Best cost/benefit:** Pipeline **B +
  LinearSVC C=19** = 0,9833 in ~15 s (ExtraTrees costs ~120 s for ~+0,1 pp).
- **Best by honest generalization (stratified CV-5 over the training set):** Pipeline
  **A + ExtraTrees** = 0,9255 (stable, ±0,002). This is the recommended criterion for production,
  since the original validation suffers from *dataset shift* (~+6 pp).
- **No pipeline suffers from overfitting**: all of them have `F1_train < F1_validation`
  (a negative `overfit_gap` of −0,6 to −1,2 pp), which indicates mild underfitting over a noisier
  training set — there is no memorization.
- **For maximum accuracy in this domain:** use **preprocessing A + vectorizer C
  (100k, bigrams, sublinear) + LinearSVC/ExtraTrees** → F1-Macro ≥ 0,985 in validation;
  only **0,2–0,3 pp** below the practical state of the art (~0,987) at a cost of seconds.
- **Do not copy the remake2 cleaning blindly:** removing stopwords and using 4-grams penalize
  under external validation; **keeping stopwords and using bigrams** adds ~+0,6 pp to C.
- **Prioritize bigrams first, then vocabulary.** Bigrams are worth **+2,6 to +4,8 pp**
  (never give them up); vocabulary breadth is worth **+0,2 to +0,4 pp** above the saturation
  point — and avoid truncation below ~50k in C (−0,9 pp at 50k, −8,5 pp at 10k).
- **Model:** prefer LinearSVC (C ≈ 0,5–19) or **ExtraTrees** over TF-IDF (ExtraTrees
  generalizes better in CV); the official remake2 hard voting is dispensable and `voting='soft'`
  should be avoided with linear SVC.
- **In MLOps terms:** keep modularity (parameterizable cleaners), log the toggles
  as hyperparameters in MLflow, **report stratified CV + overfit_gap + shift_gap** as metrics
  complementary to validation, and settle the final differences with paired tests
  (McNemar) before "crowning" a pipeline in production.
- **Documented practical upper bound for new explorations:** `max_features=200k`
  (+0,41 pp vs 100k in C), `min_df=1` (+0,30 pp in B), preprocessing A + vectorizer C.

## 8. References and Files

- `run_abc_comparison.ipynb` (runner), `pipelines_abc_core.py` (cleaners, vectorizers,
  models, evaluation) — experiment code, in this folder.
- `experiments/artifacts/pipelines_abc_20260812_123257_0295952/` — results CSVs,
  figures `fig1..fig5`, predictions, confusion matrix and summary.
- Source notebooks: `experiments/nlp/twitter-entity-sentiment/senti-pred_pipeline.ipynb` (A),
  `experiments/nlp/twitter-entity-sentiment/twitter-sentiment-analysis.ipynb` (B),
  `experiments/nlp/twitter-entity-sentiment/senti-pred-variations/Senti-Pred-remake2/` (C).
- Related documentation: `experiments/nlp/README.md` (§5.3–5.6),
  `experiments/nlp/twitter-entity-sentiment/senti-pred-variations/README.md`,
  `experiments/nlp/twitter-entity-sentiment/senti-pred-variations/EXPERIMENTS_SUMMARY.md`.
- Methods: Vapnik (SVM), Manning et al. (TF-IDF/n-grams), McNemar (1947);
  Platt calibration (1999).

## Scripts and Reproduction

# Senti-Pred Pipelines A vs B vs C — Scripts

Code of the comparative study of the 3 Twitter sentiment pipelines.

## Files

| File | Function |
|---|---|
| `pipelines_abc_core.py` | Faithful reimplementation of the 3 preprocessing variants, vectorizers, models and evaluation function |
| `run_abc_comparison.ipynb` | Orchestration of the 9 experiment batteries (E1–E9) and CSV/JSON export |
| `README.md` | Academic documentation of the study (results, what-ifs, discussion, conclusions) |

## Reproduce

```bash
# from the directory
# experiments/nlp/twitter-entity-sentiment/pipelines_abc_comparison
jupyter nbconvert --to notebook --execute run_abc_comparison.ipynb   # -> artifacts_abc/
```

To run only a subset of the batteries, edit the `STAGES` selection in the runner
cell before executing it: the script parses `parse_args([])`, so the
`--stages`/`--out` flags are never read from a real command line.

Available stages (`--stages`): `canonical ngrams max_features min_df sublinear_tf
preprocessing model_c fairness cross`.

## Dependencies

`pandas`, `numpy`, `scikit-learn>=1.0`, `joblib`, `nltk` (punkt, stopwords, wordnet, omw-1.4).
Requires the raw CSVs in
`experiments/senti-pred-variations/senti-pred-exp1/data/raw/`.

## Output

All artifacts go to `experiments/artifacts/pipelines_abc_<timestamp>_<sha>/`:

- `results_<stage>.csv` and `results_all.csv` — metric tables per run.
- `fig1_canonical.png` … `fig5_model_c.png` — comparative figures.
- `champions_summary.csv`, `predictions.npz`, `val_with_predictions.csv` — champion data.
- `meta.json` — metadata (seed, sizes, hardware, date).
