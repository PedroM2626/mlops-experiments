"""Ranking metrics for RecSys (top-K).

Covers the gap documented in the README: BPR optimizes ranking, not
rating — judging it only by RMSE is unfair. This module computes
Precision@K, Recall@K, HitRate@K and nDCG@K from predicted scores +
binary test relevance, without heavy dependencies (numpy only).

Usage:
    from ranking_metrics import ranking_report
    report = ranking_report(y_score, y_relevant, ks=(5, 10, 20))

where `y_score` (n_users, n_items) holds the predicted scores and
`y_relevant` (n_users, n_items) is 1 for an item relevant in the test
set (e.g. rating >= 4), 0 otherwise. Training items must already be
masked with -infinity score in `y_score` by the caller.
"""

from __future__ import annotations

import numpy as np


def _topk_idx(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    # argpartition for efficient top-k, then sorts the top-k
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    return part[rows, order]


def precision_recall_at_k(y_score: np.ndarray, y_relevant: np.ndarray, k: int) -> tuple[float, float, float]:
    """Returns the (precision@k, recall@k, hitrate@k) means per user.

    Users without any relevant item are ignored in the denominator.
    """
    y_score = np.asarray(y_score, dtype=float)
    y_rel = np.asarray(y_relevant, dtype=int)
    if y_score.shape != y_rel.shape:
        raise ValueError(f"divergent shapes: {y_score.shape} vs {y_rel.shape}")
    if k < 1:
        raise ValueError("k must be >= 1")
    topk = _topk_idx(y_score, k)
    hits = y_rel[np.arange(y_rel.shape[0])[:, None], topk].sum(axis=1)
    n_rel = y_rel.sum(axis=1)
    mask = n_rel > 0
    if not mask.any():
        return 0.0, 0.0, 0.0
    precision = (hits[mask] / min(k, y_score.shape[1])).mean()
    recall = (hits[mask] / np.clip(n_rel[mask], 1, None)).mean()
    hitrate = (hits[mask] > 0).mean()
    return float(precision), float(recall), float(hitrate)


def ndcg_at_k(y_score: np.ndarray, y_relevant: np.ndarray, k: int) -> float:
    """Mean binary nDCG@K per user (0/1 relevance)."""
    y_score = np.asarray(y_score, dtype=float)
    y_rel = np.asarray(y_relevant, dtype=int)
    if y_score.shape != y_rel.shape:
        raise ValueError(f"divergent shapes: {y_score.shape} vs {y_rel.shape}")
    topk = _topk_idx(y_score, k)
    ranked_rel = y_rel[np.arange(y_rel.shape[0])[:, None], topk].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, ranked_rel.shape[1] + 2))
    dcg = (ranked_rel * discounts).sum(axis=1)
    # IDCG: all relevant items first
    n_rel = y_rel.sum(axis=1)
    ideal_len = np.minimum(n_rel, ranked_rel.shape[1]).astype(int)
    idcg = np.array([discounts[:n].sum() if n > 0 else 0.0 for n in ideal_len])
    mask = idcg > 0
    if not mask.any():
        return 0.0
    return float((dcg[mask] / idcg[mask]).mean())


def ranking_report(y_score: np.ndarray, y_relevant: np.ndarray, ks: tuple[int, ...] = (5, 10, 20)) -> dict:
    """Report {k: {precision, recall, hit_rate, ndcg}}."""
    out: dict = {}
    for k in ks:
        p, r, h = precision_recall_at_k(y_score, y_relevant, k)
        out[k] = {"precision": round(p, 4), "recall": round(r, 4),
                  "hit_rate": round(h, 4), "ndcg": round(ndcg_at_k(y_score, y_relevant, k), 4)}
    return out
