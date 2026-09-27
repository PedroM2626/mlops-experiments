# Recommender Systems — MovieLens and Visual Recommendation

> **Area:** RecSys
> **Task:** Rating prediction (matrix completion) and top-K / similarity-based recommendation
> **Primary metric:** RMSE (rating prediction)
> **Status:** Completed
> **Datasets:** MovieLens 100k (100,000 ratings, 943 users, 1,682 movies); custom image dataset (visual recommendation)

## 1. Abstract

This folder compares **10 recommendation approaches** on MovieLens 100k (sparsity 93.7%), combining the 8-paradigm comparison of `movielens-recsys.ipynb` with **AutoRec** (item/user) from `movielens-autorec.ipynb`. **Item-AutoRec** won with RMSE 0.9054, followed by **Two-Tower (0.9297)** and **SVD (0.9352)**; BPR, despite its last place in RMSE (1.1138), is the suitable choice for ranking tasks. The folder also includes `image_recommender.ipynb`, a visual-similarity recommendation system (ResNet embeddings + cosine), with no built-in quality metrics (**TBD**).

## 2. Context and Objectives

The experiment seeks to answer which recommendation paradigm predicts ratings better under **high sparsity** (93.7% of the user-item matrix empty) and which one generalizes to production:
1. **Movielens recsys (8 paradigms):** popularity heuristic, similarity (user/item KNN), matrix factorization (SVD, BPR) and neural models (NCF, Two-Tower), plus tabular Gradient Boosting with manual features (LightGBM+FE).
2. **AutoRec (2 additional approaches, 10 in total):** autoencoders that reconstruct partially observed vectors (they test whether the non-linear representation wins over low-rank MF).
3. **Cold-start:** simulate a new user (3 ratings) and assess the coherence of the recommendations.
4. **Visual recommendation:** content-based similarity (not collaborative) through image embeddings.

## 3. Theoretical Background (brief)

- **Colaborative filtering:** uses user-item interactions; suffers from cold-start and sparsity.
- **Matrix Factorization (SVD, `surprise`):** approximates the matrix as R ≈ U·Vᵀ (100 factors) + biases, minimizing MSE over the observed ratings; fast Cython implementation.
- **KNN user/item:** cosine similarity between profiles; degrades under high sparsity (distance matrix dominated by zeros).
- **BPR (pairwise MF):** optimizes ranking through negative sampling (one (pos, neg) pair at a time); the choice for top-K, the baseline is not evaluated by RMSE.
- **NCF (NeuMF):** concatenation of 32-d embeddings + MLP [64,32,16] + Dropout.
- **Two-Tower (DLRM-style):** independent MLP towers for user and item + dot product; precomputes item embeddings and makes approximate retrieval (ANN) feasible on massive catalogs.
- **LightGBM + FE:** tabular features (mean, std, count per user/item + interactions, 8 features, 500 trees/6k leaves).
- **AutoRec (Sedhain et al., 2015):** autoencoder (input→tanh→hidden→sigmoid) that reconstructs sparse vectors; **masked** MSE loss (only observed entries (`M==1`) generate gradient); item-based variant (vector = ratings the item received, dim N_USERS) and user-based variant (dim N_ITEMS).

## 4. Methodology

### 4.1 Data

- **MovieLens 100k:** 100,000 ratings (1–5 scale) from 943 users over 1,682 movies; sparsity of **93.7%**. Loaded through the surprise `u.data` file (`~/.surprise_data/ml-100k/ml-100k`).
- Split of the neural comparison: `train_test_split` 80/20 with `random_state=42` (80,000 training / 20,000 test) — the same split for NCF, LightGBM and Two-Tower.
- **Anti-leakage (AutoRec):** test ratings never enter the training matrix (zeroed when building `R_tr`).
- **Images (image_recommender):** local dataset of ~30 images (embedding dim 2048) used to index/demonstrate.

### 4.2 Preprocessing

- Normalization of the ratings to **[0,1]** (`rating/5`) for AutoRec (as in the paper); observed entries stay 0.
- Binary matrix M (mask) — loss only over the observed entries.
- In rating prediction/recsys: no excessive feature engineering; LightGBM+FE depends on 8 manual features.
- Visual representation: extraction of pretrained ResNet embeddings → L2 normalization → cosine similarity.

### 4.3 Compared methods

| Model | Paradigm | Strategy | Parameters (~) |
|--------|-----------|-----------|---------------|
| **Popularity** | Heuristic | Global mean per item (min 5 ratings) | 0 |
| **KNN User-based** | Similarity (user) | Cosine between users | 0 |
| **KNN Item-based** | Similarity (item) | Cosine between items | 0 |
| **SVD** | MF (MSE) | 100 latent factors + biases, 20 epochs | ~200k |
| **NCF (NeuMF)** | Neural (concat) | 32-d embs + MLP [64,32,16] + Dropout | ~2.2M |
| **LightGBM + FE** | GB Tabular | 8 user/item features + interactions, 500 trees | ~6k leaves |
| **BPR** | MF (pairwise) | 64-d factors, BPR loss, negative sampling | ~165k |
| **Two-Tower** | Neural (dot) | 32-d embs + MLP towers [64,32] + dot product | ~150k |
| **Item-AutoRec** | Autoencoder | MLP N_USERS→hidden(500)→N_USERS, Tanh+Sigmoid, masked MSE | ~500×2 |
| **User-AutoRec** | Autoencoder | MLP N_ITEMS→hidden(500)→N_ITEMS, masked MSE | ~500×2 |

### 4.4 Evaluation

- **Metric:** RMSE on the 80/20 split with `random_state=42` (holdout).
- **AutoRec:** masked MSE per epoch; tuning of `hidden ∈ {200, 500, 800}` with a training/validation split (90/10), ranked by `val RMSE`; 100 epochs, Adam.
- **Cold-start:** simulation of a new user (Star Wars 5, Fargo 4, Shining 3) → SVD recommendation list.
- **Reproducibility:** experiment registered in local MLflow (`./mlruns`, experiment `MovieLens_AutoRec`).

### 4.5 Reproduction

- Notebooks relative to this folder:
  - `./movielens-recsys.ipynb` — the 8 base paradigms (Popularity, KNN, SVD, NCF, LightGBM, BPR, Two-Tower)
  - `./movielens-autorec.ipynb` — adds Item-/User-AutoRec (the 9th and 10th approaches), compares the 10 models and registers them in MLflow
  - `./image_recommender.ipynb` — visual recommendation pipeline (CLI + interactive demo)
- Dependencies: `surprise` (MovieLens data), PyTorch, LightGBM, pandas, numpy, MLflow.
- Artifact pattern: `experiments/artifacts/<experiment>_<timestamp>_<sha>/`.

## 5. Results

### 5.1 Run (10 models) — AutoRec included (movielens-autorec.ipynb)

| Model | RMSE | Paradigm |
|---|---|---|
| **Item-AutoRec** | **0.9054** | Autoencoder (item) |
| Two-Tower | 0.9297 | Neural (dot) |
| SVD | 0.9352 | MF (MSE) |
| LightGBM+FE | 0.9406 | GB Tabular |
| NCF | 0.9462 | Neural (concat) |
| User-AutoRec | 0.9611 | Autoencoder (user) |
| Popularity | 1.0171 | Heuristic |
| KNN User | 1.0194 | Similarity |
| KNN Item | 1.0264 | Similarity |
| BPR | 1.1138 | MF (pairwise) |

Note: in the source notebook the 8 original approaches follow the reported RMSE pattern (Two-Tower 0.929712, SVD 0.935171, LightGBM+FE 0.940597, NCF 0.946228, Popularity 1.017112, KNN User 1.019354, KNN Item 1.026430, BPR 1.113827).

**AutoRec (details):**
- Item-based: converging masked MSE 0.0230 (epoch 100); **test RMSE 0.9054**; training **16.0s**.
- User-based: masked MSE 0.0239; **test RMSE 0.9611**; training **10.7s**.
- Tuning (val RMSE): item `hidden=500` → 0.9065 (peak), `hidden=800` → 0.9080, `hidden=200` → 0.9124; user `hidden=500` → 0.9614, `hidden=200` → 0.9649, `hidden=800` → 0.9665.

**Run analysis:**
- Item-AutoRec beats Two-Tower by ~0.024 in RMSE, and SVD by ~0.030 — the compressed non-linear representation (item-side) outperforms low-rank factorization and the neural disciples.
- User-AutoRec (0.9611) lands behind LightGBM (0.9406) and NCF (0.9462), but ahead of the heuristics; **item-side >> user-side** on this high-sparsity matrix (the heuristic improves on mean per-item sparsity rather than per user).
- BPR in last place (1.1138): RMSE is not a fair metric for pairwise ranking (evaluate precision/recall@K).

### 5.2 Cold-Start (movielens-recsys.ipynb)

A new user rated Star Wars 5, Fargo 4, Shining 3 → SVD recommended **Empire Strikes Back (4.97)**, **Dr. Strangelove (4.94)**, **Cuckoo's Nest (4.94)** — well-rated classics of similar profile, indicating coherence without user training.

### 5.3 Visual Recommendation (image_recommender.ipynb)

- Pipeline: collection → ResNet embedding extraction → L2 normalization → cosine top-K → JSONL output.
- Observed indexing benchmark: **30 images indexed in 4.229 s, dim 2048**.
- Ranking evaluation: use `ranking_metrics.py` (`ranking_report(y_score, y_relevant, ks=(5,10,20))` → precision/recall@K, hit-rate@K, nDCG@K). Mask training items with score −inf before ranking.

### 5.4 Ranking evaluation (BPR and others) — `ranking_metrics.py`

RMSE measures the absolute rating; BPR optimizes pairs (ranking). For a fair top-K comparison:

```bash
python -c "from ranking_metrics import ranking_report; print(ranking_report(scores, rel))"
```

with `scores` (n_users × n_items) and binary `rel` (1 = relevant in the test set, e.g. rating ≥ 4). Tests: `tests/test_ranking_metrics.py` (7 asserts, includes the case where the BPR ordering beats the heuristic in nDCG).

## 6. Discussion

- **Item-based AutoRec is the new local champion:** with the masked (non-linear) MLP the item represents compact information that the low-rank latent-space SVD does not exploit; the cost is ~16s of training vs the SVD's ~11 Cython seconds — a trade-off still favorable on a small dataset.
- **Two-Tower vs SVD** keeps a near tie locally (~0.005–0.006), but Two-Tower enables ANN/max recovery for production with millions of items (Google/Meta/Pinterest).
- **LightGBM+FE** proves that manual FE (u_std, u_mean) competes with the neural models (0.9406), besides being interpretable (SHAP/feature importance).
- **Sparsity hurts similarity methods** (KNN ~1.02); the model is also a sample of the wins of tabular Feature Engineering: trees gain less from FE than linear models (see the `tabular_regression` folder).
- **BPR should not be evaluated by RMSE** — its function is top-K; the ranking measurement is implemented in `ranking_metrics.py` (§5.4).
- **Limitations:** single dataset (100k); `image_recommender` evaluated through the ranking protocol (`ranking_metrics.py`), with no external quality baseline — a demonstration of the pipeline.

## 7. Conclusions and Recommendations

- **Rating prediction** on datasets ≤ 100k: **AutoRec item-based** (RMSE 0.9054) or **SVD** (0.9352) as the best cost/simplicity (Cython, seconds).
- **Best production trade-off:** **Two-Tower** for scale (ANN) — the ~0.93 RMSE cost is the best option when retrieval over a massive catalog is needed.
- **Interpretability:** **LightGBM+m3** with 8 features (u_mean/u_std) is strong (0.9406) and provides SHAP.
- **Ranking (top-K):** **BPR** (pairwise) — measured by precision/recall@K and nDCG through `ranking_metrics.py`, not by RMSE.
- Suggestion: version the score matrix per model to re-evaluate ranking without retraining, and evaluate `image_recommender` with precision@K on a labeled dataset.
- **Cold-start:** popularity + content hybrid until the user accumulates interactions.

## 8. References and Files

- Notebooks: `./movielens-recsys.ipynb`, `./movielens-autorec.ipynb`, `./image_recommender.ipynb`.
- Code: `./ranking_metrics.py` + `./tests/test_ranking_metrics.py`.
- References: Sedhain et al. (2015) *AutoRec: Autoencoders Meet Collaborative Filtering*; Koren et al. (2009) *Matrix Factorization Techniques for Recommender* (SVD); Rendle et al. (2009) *BPR*; He et al. (2017) *Neural Collaborative Filtering* (NeuMF); Grafer et al. for Two-Tower/DLRM; Harley et al. (2022) for visual recommendation embeddings (ResNet).
- Group reference document: `docs/academic-readme-template.md`.