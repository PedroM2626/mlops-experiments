# Causal ML in NLP with Real Data — Effect of Delivery Delay on Customer Sentiment (Olist)

> **Domain:** Causal inference + NLP
> **Task:** Causal effect estimation (ATE/CATE) of a binary treatment on text
> **Primary metric:** ATE as a risk difference (percentage points of P(negative sentiment))
> **Status:** Completed
> **Dataset:** Brazilian E-Commerce Public Dataset by Olist (2016–2018, ~100k orders, CC BY-NC-SA 4.0 license) — 7 linked tables; public mirror `github.com/Mylinear/Brazilian_E_Commerce_Public_Dataset_by_Olist` (downloaded automatically by the notebook).

## 1. Abstract

A **causal** (not predictive) question: does delivering after the estimated date **cause** negative sentiment in the review text? We formalize it with Potential Outcomes (Rubin) + a DAG (Pearl): treatment $T$ = delivery delay (`delay_days > 0`), outcome $Y$ = negative sentiment extracted from the text by a PT-BR lexicon (validated against the human rating), confounders $X$ strictly pre-treatment (price, freight, category, UF, payment, weight, promised delivery window, month). On N = 39,068 orders with text, the raw association is enormous (+43.5 p.p., RR 3.3×) and **survives every adjustment**: LPM +44.4 p.p., Logit (AME) +32.5 p.p., Matching +44.0, IPW-Hajek +43.3, AIPW +41.7, S-learner +35.9, T-learner +41.8. The honest causal tree confirms a positive effect in **all** leaves. Conclusion: the delivery delay is a causal lever of textual negative sentiment (not mere correlation).

## 2. Context and Objectives

The NLP experiments in this repository address sentiment **prediction**; here the question is **interventionist**: if logistics operations eliminate the delay, by how much does the probability of a negative review fall? This is the repository's first experiment devoted entirely to causal inference on real public data, with no counterfactual *ground truth* — credibility comes from the triangulation of methods, diagnostics and refutations.

Research questions:

- **RQ1:** is the naive contrast (+43.5 p.p.) confounded by price/category/UF? (Partly yes — but the adjusted effect stays ≥ +32 p.p. across all estimators.)
- **RQ2:** does the effect replicate on outcomes independent of the lexicon (`review_score ≤ 2`, continuous `sent_score`)? (Yes.)
- **RQ3:** is there heterogeneity (CATE) by category, UF, price band and period? (Yes, moderate.)
- **RQ4:** do the refutations (placebo, trimming, alternative definition of T, sensitivity to an omitted confounder) overturn the conclusion? (No.)

## 3. Theoretical Foundation (brief)

- **Potential outcomes (Rubin):** $Y_i = T_iY_i(1) + (1-T_i)Y_i(0)$; target $\tau = E[Y(1)-Y(0)]$ on the risk-difference scale (p.p.). Assumptions: SUTVA/consistency, ignorability given $X$, positivity $0 < e(X) < 1$, temporality (the delay precedes the review — verified: delivery ≤ review in 88.6% of cases).
- **Back-door (Pearl):** adjusting for pre-treatment $X$ blocks $T \leftarrow X \to Y$; post-treatment variables (review dates, text length) deliberately excluded.
- **Estimators:** LPM (OLS with HC1 robust SE), Logit with average marginal effect (AME), propensity score + 1-NN Matching on the logit, IPW-Hajek, AIPW (doubly robust, nuisances via RandomForest), S/T-learners (Kunzel et al., 2019).
- **Heterogeneity:** CATE via the T-learner and an **honest causal tree** (Athey & Imbens, 2016) implemented by hand: DR pseudo-outcomes + honest 50/50 split (sample A builds the structure, sample B estimates per leaf).
- **Credibility without ground truth:** bootstrap B=200, placebo (permuted T), dummy outcome (`n_palavras`), propensity trimming, alternative definition of T (>1 day) and sensitivity to a simulated omitted confounder.

## 4. Methodology

### 4.1 Data

Seven Olist tables linked by `order_id`/`customer_id`/`product_id` (reviews ⨝ orders ⨝ aggregated items ⨝ first product ⨝ category translation ⨝ customers ⨝ aggregated payments): merged frame 99,224 × 25. Distribution of `review_score`: 5★ 57.8%, 4★ 19.3%, 3★ 8.2%, 2★ 3.2%, 1★ 11.5%. **Analysis sample:** `order_status = delivered` + known `delay_days` + non-empty text → **N = 39,068** (treated = 4,275, 10.9%).

### 4.2 Building T, Y and X

- **T** = `1{delay_days > 0}` (days between delivery and the estimate; zero tolerance; sensitivity with >1 day in Section 8 of the notebook).
- **Y_neg** = `1{sent_score < 0}`; `sent_score = (p − n)/max(1, p+n)` via a PT-BR substring lexicon (auditable, ~80 terms); secondary outcomes: continuous `sent_score` and `Y_low = 1{review_score ≤ 2}` (human label).
- **X (22 columns after dummies):** `log_price`, `log_freight`, `n_items`, `log_weight`, `installments`, `est_days` (promised delivery window), `purchase_yearmonth_int` (trend), `cat_group` (top-8 + others), `uf_group` (top-5 + others), `pay_group`.

### 4.3 NLP Validation

Lexicon × human rating: correlation **0.661**; accuracy of `Y_neg` against `score ≤ 2` = **0.8375**; a TF-IDF classifier → `Y_neg` reaches AUC **0.985** (strong linguistic signal). Descriptive TF-IDF shows the vocabulary of delay (Portuguese phrases meaning "I did not receive it", "still not", "delay", "it did not arrive") vs. without delay ("ahead of the deadline", "I recommend", "very good").

### 4.4 Evaluation and Reproduction

- Global seed 42; probabilities via `cross_val_predict` for the propensity AUC; non-parametric bootstrap B=200 with fast estimators (logit).
- Hardware of the reference run: CPU x86-64, Python 3.13, scikit-learn 1.7.1, statsmodels 0.14.5 (17/09/2026).
- To reproduce:

```bash
cd experiments/causal_nlp_olist
python -m nbconvert --to notebook --execute causal_nlp_olist.ipynb --inplace
```

The 7 CSVs (~65 MB) are downloaded automatically into `data/` on the first run (public GitHub mirror; CC BY-NC-SA 4.0 license — non-commercial use with attribution). For data already present, the notebook also looks in `../datasets/olist/`.

## 5. Results

### 5.1 Selection (who writes a review differs)

| Group | Mean score | P(score ≤ 2) | P(delay) | Mean price | n |
|---|---|---|---|---|---|
| No text (59.5%) | 4.417 | 0.053 | 0.060 | R$ 130.26 | 57,285 |
| With text (40.5%) | 3.773 | 0.238 | 0.109 | R$ 146.03 | 39,068 |

Those who write have more extremes and more delay → we estimate the **SATE** (effect on the subpopulation with text), not the PATE.

### 5.2 Estimated causal effect (primary outcome `Y_neg`)

Run 17/09/2026, seed 42:

| Estimator | ATE (risk difference) |
|---|---|
| Naive (unadjusted) | 0.4348 |
| Adjusted LPM (OLS + HC1) | 0.4441 (95% CI [0.4289; 0.4594]) |
| Logit — average marginal effect | 0.3249 (95% CI [0.3155; 0.3343]) |
| 1-NN Matching (propensity) | 0.4399 |
| IPW-Hajek | 0.4332 |
| **Doubly robust AIPW (RF)** | **0.4175** |
| S-learner (RF) | 0.3591 |
| T-learner (RF) | 0.4179 |

**Independent replication of the lexicon:** `Y_low = score ≤ 2` (human label): naive 0.4878 → AIPW **0.4749**; continuous `sent_score`: naive −0.7975 → AIPW **−0.7598** (worse sentiment).

### 5.3 Diagnostics and refutations

| Check | Result |
|---|---|
| Propensity AUC (in-sample / CV-5) | 0.687 / 0.683 (moderate discrimination = no perfect separation) |
| Overlap (e_hat ∈ [0.02; 0.5]) | 98.2% of the sample |
| Mean SMD \|·\| raw → IPW | 0.085 → 0.024 (Love plot) |
| Bootstrap B=200 (AIPW-logit) | mean 0.4367, 95% CI [0.4208; 0.4547] — excludes 0 |
| Placebo (permuted T) | +0.0007 (p = 0.922) — null as expected |
| Dummy outcome (`n_palavras`) | +3.36 words (14.9 vs 11.5) — small style effect |
| Trimming e ∈ [0.02; 0.98] / [0.05; 0.95] | IPW 0.4343 / 0.4428 (stable) |
| Alternative T (> 1 day) | naive 0.4800 (the effect grows under the stricter definition) |
| Simulated omitted confounder (γ = 0.05) | τ rises to 0.5044 vs 0.4486 corrected — it would take a very strong U to nullify it |

### 5.4 Heterogeneity (CATE, T-learner)

- **Category:** sports_leisure 0.446 > ... > telephony 0.366 — a span of ~8 p.p. across categories.
- **UF:** RJ 0.456 > RS 0.432 > MG 0.419 > SP 0.394 / PR 0.393.
- **Price band:** Q2 0.432 ≈ Q3 0.430 > Q4 0.409 > Q1 0.401 — an effect in every band.
- **Honest causal tree** (depth 3, 7 leaves, honest 50/50 split): effect **positive in all leaves** (0.302 to 0.467; range 0.1644); weighted mean 0.4228; correlation with the per-leaf T-learner = 0.759. Root split: `purchase_yearmonth_int` (period), followed by `pay_group`/`est_days`/`log_freight`/`n_items`.

## 6. Discussion

- **Association ≠ confounding here:** in data with strong selection (raw SMDs up to 0.285 on UF), adjustment barely moves the effect (43.5 → 42–44 p.p. under IPW/Matching) because the delay has low prevalence (10.9%) and the outcome is extremely reactive to it. Logit-AME (+32.5) and the S-learner (+35.9) are more conservative because of smoothing by the outcome model.
- **Attenuation from measurement error:** the lexicon errs in a plausibly non-differential way (it does not "see" the delivery date), so the expected bias is **attenuation** — the real effect tends to be ≥ the estimated one. Triangulation with `review_score` (human label, AIPW +47.5 p.p.) confirms magnitude and direction.
- **Selection:** SATE ≠ PATE; those who do not write have a mean rating of 4.42 and 6% delay — plausibly a smaller effect in the total population. Natural extension: selection IPW / Heckman.
- **Limitations:** no counterfactual ground truth; approximate SUTVA (a binary delay collapses 1 vs 30 days; possible regional interference); unobserved confounders (product quality, expectation, strikes); aggregation by the first item in multi-item orders; GoT anonymization and the 2016–2018 period limit generalization; text length (a possible mediator) excluded from X.

## 7. Conclusions and Recommendations

- **Meeting the estimated delivery date is a causal lever on sentiment:** eliminating the delay in the subpopulation that writes reviews would reduce the probability of a negative review by ~32–44 percentage points (central estimator AIPW: ~42 p.p.).
- **Operational prioritization:** larger CATE in RJ, in the sports_leisure/watches_gifts categories and in orders not paid with a card; the leaves of the honest tree suggest period and promised delivery window as moderators.
- **For research:** continuous dose-response (GPS) in `delay_days`; Double ML + Causal Forest (EconML) with cross-fitting; BERTimbau for sentiment calibrated by aspect (delay vs quality vs service); correction for text selection.

## 8. References and Files

- Executed notebook: [`./causal_nlp_olist.ipynb`](./causal_nlp_olist.ipynb) (37 original cells + data bootstrap; figures and outputs included).
- Data: `data/` (ignored in git; downloaded automatically by the notebook from the `github.com/Mylinear/Brazilian_E_Commerce_Public_Dataset_by_Olist` mirror).
- References: Rubin (1974); Rosenbaum & Rubin (1983); Pearl (2009); Kunzel et al. (2019, metalearners); Athey & Imbens (2016, honest trees); Chernozhukov et al. (2018, Double ML); Austin (2009, balancing); Egami et al. (2018, text-as-outcome); Feder et al. (2022, causal inference in NLP); Olist (2018, dataset, CC BY-NC-SA 4.0).
