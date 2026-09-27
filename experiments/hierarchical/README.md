# Hierarchical Experiments — 20 Newsgroups

> **Area:** NLP + class hierarchy
> **Task:** Hierarchical classification (supervised) and hierarchical clustering (unsupervised)
> **Main metric:** Leaf exact-match accuracy; Purity / NMI / ARI / F (clustering); HF (hierarchical)
> **Status:** Completed
> **Datasets:** 20 Newsgroups (scikit-learn) — ~18.000 real Usenet messages, 1993, 7 parents / 20 classes

## 1. Abstract

Two complementary experiments explore tree structures (`parent → leaf`) in the **20 Newsgroups** dataset (~18.000 real Usenet messages; 11.314 training / 7.532 test; 7 parents / 20 classes):
1. **Hierarchical classification** — compares the flat baseline vs. a node-wise local hierarchical classifier (word + char TF-IDF with LinearSVC): leaf acc **0.7188** (flat) vs. **0.6953** (hierarchical V3), but the hierarchical HF metric shows a closer picture (0.7668 vs. 0.7516) — the hierarchy wins on interpretability, not on exact-match.
2. **Flat vs. hierarchical clustering** (KMeans, agglomerative Ward and top-down in 2 levels, sample of 3.000 docs seed 42): **top-down** beats flat at leaf level (**Purity 0.398**, **NMI 0.360**, F 0.381) — above the typical literature range (NMI 0.25–0.45).

The central lesson is that the hierarchy **helps most when it is explored in levels** (top-down / cascade), not with a single plain cut; and classification must not be compared directly with unsupervised clustering.

## 2. Context and Objectives

In many problems classes are not independent but structured as a tree (`parent → child`): in 20 Newsgroups every label has the form `parent.leaf` (e.g.: `sci.med`), the first part is the parent and the group is the leaf — a natural 2-level hierarchy. The objectives:

**Part 1 (classification):** compare a flat model (20 classes at once) with a node-wise local hierarchical classifier (level 1 predicts the parent; level 2 predicts the leaf inside the parent) and measure whether the structure helps, hurts or is equivalent; find out where the hierarchy errs (parent vs. child level).

**Part 2 (clustering):** measure how much a **flat** clustering (one level) and a **hierarchical** one (nested groups) rediscover the real structure of the dataset, both at leaf level (20) and at parent level (7), without access to the labels.

## 3. Theoretical Background (brief)

- **TF-IDF (word + char n-grams)** — sparse representation; character n-grams (`char_wb` 2–5) capture complementary morphological patterns. Vectorizers fitted **on training data only** (avoids leakage).
- **LinearSVC** — linear SVM with hyperplane margin and L2 penalty; together with character n-grams it generalized better than LogisticRegression with unigrams.
- **Hierarchical metrics HP/HR/HF** — consider the label and all its ancestors (augmented sets `T* = leaf + parent` and `P* = leaf + parent`): `HP = |P*∩T*|/|P*|`, `HR = |P*∩T*|/|T*|`, `HF = 2·HP·HR/(HP+HR)`. They capture *partial* errors (correct parent, wrong leaf ⇒ 0.5) that exact-match does not see.
- **External clustering validation:** Purity (fraction in the majority cluster), NMI (mutual information normalized for chance), ARI (adjusted Rand index), cluster F. For external validity on 20 Newsgroups, typical NMI values for classical methods over TF-IDF fall between **0.25 and 0.45**.

## 4. Methodology

### 4.1 Data

| Item | Classification | Clustering |
|------|------------------------|------------|
| Dataset | 20 Newsgroups (`fetch_20newsgroups`) | 20 Newsgroups (test set) |
| Train / test | **11.314** / **7.532** docs | fixed sample of **3.000** docs (seed 42) |
| Leaves | 20 | 20 |
| Parents (level 1) | 7 (`alt`, `comp`, `misc`, `rec`, `sci`, `soc`, `talk`) | 7 |

Headers, footers and quoted passages removed (avoids label leakage from e-mail metadata). Training distribution per parent: alt 480, comp 2.936, misc 585, rec 2.389, sci 2.373, soc 599, talk 1.952. Mean text ~1.218 characters (median 491); ~218 empty documents after cleaning. Final matrix: **7.532 × 189.423 features**.

Structure of the tree (from the labels):

```
├─ alt    → alt.atheism
├─ comp   → comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware,
│           comp.sys.mac.hardware, comp.windows.x
├─ misc   → misc.forsale
├─ rec    → rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey
├─ sci    → sci.crypt, sci.electronics, sci.med, sci.space
├─ soc    → soc.religion.christian
└─ talk   → talk.politics.guns, talk.politics.mideast, talk.politics.misc,
            talk.religion.misc
```

Note: `alt`, `misc`, `soc` have only one leaf → at level 2 the prediction is trivial.

### 4.2 Preprocessing and features

```python
TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=100_000)   # word 1-1 (V1)
```

Improved version (V2/V3) — **word + char** concatenation:

| Group | Analyzer | n-gram | max_features | min_df | max_df | sublinear_tf |
|-------|-----------|--------|--------------|--------|--------|--------------|
| Words | `word` | (1, 1) | 80.000 | 2 | 0.9 | yes |
| Characters | `char_wb` | (2, 5) | 150.000 | 2 | 0.9 | yes |

For clustering: sublinear word + char TF-IDF concatenated (≈189k) → **SVD of 100 components** + normalization, fitted on training data only.

### 4.3 Methods compared

**Classification:**
- **Flat** (baseline): a single multiclass model (20 classes).
- **Hierarchical local per node:** level 1 predicts the parent, level 2 predicts the leaf inside each parent; single leaves → trivial prediction.
- V1: `LogisticRegression(C=1.0, max_iter=2000)` on TF-IDF word 1-1.
- V2: `LinearSVC(C=0.15)` word+char (C chosen by a sweep over {0.02–2.0}).
- V3: level 1 tuned via `GridSearchCV`(cv=3) over `C ∈ {0.15,0.5,1.0,2.0}` × `class_weight ∈ {None, balanced}` → best `{'C':0.5,'class_weight':'balanced'}` (CV score 0.8353); children keep LinearSVC(C=0.15).

**Clustering:**
| # | Strategy | Detail | n_clusters |
|---|-----------|---------|-----------|
| 1 | **Flat** | `KMeans(k=20)` (leaf) / `KMeans(k=7)` (parent) | 20 / 7 |
| 2 | **Hierarchical agglomerative** | `AgglomerativeClustering` (Ward), cut at 7 and 20 | 20 |
| 3 | **Hierarchical top-down (2 levels)** | `KMeans(k=7)` → re-clusters each group (number of subgroups = number of real leaves); label = pair (parent, child) | 45 |

### 4.4 Evaluation

Classification: leaf exact-match accuracy, macro-F1, parent accuracy, leaf accuracy given the correct parent, and the hierarchical HP/HR/HF metrics; confusion matrices normalized at parent level.
Clustering: Purity, NMI, ARI, F for leaves; Purity and NMI for parents. The fixed sample with seed 42 guarantees a fair comparison and the O(n²) cost of the agglomerative method.

### 4.5 Reproduction

```bash
# Classification
pip install scikit-learn numpy pandas matplotlib jupyter nbconvert
python -m nbconvert --to notebook --execute --inplace classificacao_hierarquica.ipynb --ExecutePreprocessor.timeout=1200
# Clustering
python -m nbconvert --to notebook --execute --inplace clustering_flat_vs_hierarquico.ipynb --ExecutePreprocessor.timeout=1800
```

Or interactive: `jupyter notebook <notebook>.ipynb`. The dataset downloads automatically on first use (scikit-learn cache).

Real environment: Windows, Python 3.13.5, pip 25.1.1, scikit-learn 1.9.0, jupyter/nbconvert/nbformat; direct deps: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `scipy`.

## 5. Results

### 5.1 Classification — V1 (TF-IDF word 1-1; LogisticRegression)

| Metric | Flat | Hierarchical |
|---------|------|-------------|
| Leaf accuracy (exact) | **0.6835** | 0.6516 |
| Leaf macro-F1 | 0.6680 | 0.6401 |
| Parent accuracy | 0.7889 | 0.7740 |
| Leaf accuracy given correct parent | — | 0.8419 |

### 5.2 Classification — V2 (word+char; LinearSVC C=0.15)

| Metric | Flat | Hierarchical | Δ Flat | Δ Hier. |
|---------|------|-------------|--------|---------|
| Leaf accuracy (exact) | **0.7188** | 0.6889 | +0.0353 (3.53 pp) | +0.0373 (3.73 pp) |
| Leaf macro-F1 | 0.7045 | 0.6786 | +0.0365 | +0.0385 |
| Parent accuracy | 0.8148 | 0.8054 | +0.0259 | +0.0314 |
| Leaf accuracy given correct parent | — | 0.8554 | — | +0.0135 |

Errors V1→V2: 2.624 → **2.343** (test 7.532).

### 5.3 Classification — V3 (tuned level 1: balanced + C=0.5)

| Metric | Flat | Hierarchical (V3) |
|---------|------|------------------|
| Leaf accuracy (exact) | **0.7188** | 0.6953 |
| Leaf macro-F1 | 0.7045 | 0.6848 |
| Parent accuracy | 0.8148 | **0.8079** |
| Leaf accuracy given correct parent | — | **0.8606** |
| **HF (hierarchical)** | 0.7668 | 0.7516 |

Hierarchical evolution: leaf 0.6516 (V1) → 0.6889 (V2) → **0.6953** (V3); parent 0.7740 → 0.8054 → **0.8079**. Hierarchical errors: 2.343 → **2.295** (V3).

**Confusions (hierarchical, V3):**

| True class | Predicted | Number of errors |
|-------------------|---------|-----------|
| `talk.politics.misc` | `talk.politics.guns` | 93 |
| `alt.atheism` | `talk.religion.misc` | 44 |
| `comp.windows.x` | `comp.graphics` | 44 |
| `talk.religion.misc` | `soc.religion.christian` | 43 |
| `soc.religion.christian` | `talk.religion.misc` | 38 |

Politics/religion classes are the hardest (lowest F1: `talk.religion.misc` ~0.38, `talk.politics.misc` ~0.48).

### 5.4 Hyperparameter exploration (flat, leaf acc)

| Configuration | Leaf accuracy |
|--------------|----------------|
| word 1-1 + LogisticRegression (V1) | 0.6835 |
| word 1-1 + LinearSVC(C=0.5) | 0.7010 |
| char_wb 2-5 + LinearSVC(C=0.5) | 0.7080 |
| **char_wb 2-5 + word 1-1 + LinearSVC(C=0.15)** | **0.7188** (final choice) |

| Parent configuration | Parent accuracy |
|---------------------|------|
| LinearSVC(C=0.15), no rebalancing | 0.8054 |
| LinearSVC(C=0.5), `class_weight='balanced'` | **0.8079** (final choice) |

### 5.5 Clustering — leaf level

| Strategy | n_clusters | Purity | NMI | ARI | F |
|------------|-----------|--------|-----|-----|---|
| **Flat (KMeans k=20)** | 20 | 0.332 | 0.315 | 0.144 | 0.374 |
| Hierarchical agglomerative (cut 20) | 20 | 0.289 | 0.280 | 0.102 | 0.335 |
| **Hierarchical top-down (2 levels)** | 45 | **0.398** | **0.360** | 0.102 | **0.381** |

### 5.6 Clustering — parent level

| Strategy | Purity | NMI |
|-----------|--------|-----|
| **Flat (KMeans k=7)** | **0.568** | **0.268** |
| Agglomerative (cut 7) | 0.487 | 0.214 |

Notes: **top-down produces 45 leaf clusters** (not 20) because the real groups are not perfectly separated at parent level; the agglomerative method has the same number of clusters as flat (20), compared on equal terms.

## 6. Discussion

**Classification:** level 1 (parent) is the bottleneck of the hierarchy — when it errs, the leaf is lost. Rebalancing (`class_weight='balanced'`) + `C=0.5` raised the parent from 0.805→0.808 and, in cascade, the leaf from 0.689→0.695. Flat still wins exact-match (0.7188) because errors accumulate going down the tree. However the hierarchical HF shows a fairer picture (Flat 0.7668 vs 0.7516): hitting the parent but missing the leaf counts 0.5; when the parent is correct, the child gets the leaf right **86.1%** of the time. Advantages of the hierarchy: **interpretability** (where the error occurs: parent or child), **scalability** (models per node, useful for thousands of classes), robustness in problems with many classes.

**Clustering**
- Flat KMeans: reasonable baseline (ARI 0.144), but the agglomerative method with a single cut (20) stays **below** flat (NMI 0.280 vs 0.315) — a "tree" is not the same as an "explored hierarchy".
- **Top-down wins at leaf level** (Purity 0.398, NMI 0.360, F 0.381): by grouping the topics first and then the groups inside each topic, it exploits the real `topic → group` structure.
- At **parent** level, Flat KMeans still beats the agglomerative method (Purity 0.568 vs 0.487).
- As for whether the result is good: clustering **cannot** come close to classification (acc ≈ 0.72) — it sees no labels at all; typical NMI values (on the 20) are 0.25–0.45; our flat (0.315) is expected, top-down (0.360) and purity (0.398) are **above expectation**. Purity 0.40/NMI 0.36 = the clusters recover the main topics (comp, sport, science, religion/politics) with considerable mixing, since some groups are almost inseparable by vocabulary (`talk.politics.*`, `talk.religion.misc`, `soc.religion.christian`, `alt.atheism`).
- **Common limitation:** politics/religion classes remain the most confused and the minutes in these cells never improve the hierarchies.

## 7. Conclusions and Recommendations

- **Feature representation matters**: word + char n-grams + LinearSVC (C=0.15) raised flat from ~0.68 to ~0.72.
- **Hierarchical classification**: more interpretable and scalable, with ~2.4 pp of exact-match loss to flat; the error chain concentrates in the parent — strengthening level 1 (rebalancing + C) raised the leaf accuracy and HF brought flat/hierarchical closer.
- **Top-down clustering (2 levels) is superior to flat at leaf level** (Purity 0.398, NMI 0.360) and **above the text-clustering literature on 20 Newsgroups**.
- **Practical recommendation:** for data with a known taxonomy (2+ levels), **cascade (top-down) clustering** is superior to a single KMeans; for classification, if exact-match is critical prefer flat — if interpretability and scalability matter, use a hierarchy with a well-trained parent.
- **Next steps** (class): calibrate the parent level via `calibrate_parent.py::tune_parent_threshold` — **measured on real data** (`run_hierarchical_calibration.py`, reproduces the exact V3: flat 0,7188 / parent 0,8079 / leaf 0,6953 / child-given-parent 0,8606): the optimal threshold is "keep everything" under both fallbacks (zeros and flat), proxy 0,6953 = baseline. That is, parent errors spread over the whole confidence range — thresholding does not solve it; the way forward is a better parent (rebalancing/C, already done in V3) or non-linear models on the politics/religion classes.

## 8. References and Files

**Classification** (`classificacao_hierarquica.ipynb`)
- Main notebook (executed, contains tables, matrices, plots, predictions).
- Code: `calibrate_parent.py` + `tests/test_calibrate_parent.py` (parent threshold tuning).

**Clustering** (`clustering_flat_vs_hierarquico.ipynb`)
- Main notebook (executed, with dendrogram and heatmap).
- Standalone comparisons: `clustering_comparison.ipynb`, `supervised_clustering.ipynb`.

Source: repository root README (summary table "Hierarchical Experiments — 20 Newsgroups") and the scikit-learn documentation of the 20 Newsgroups dataset.