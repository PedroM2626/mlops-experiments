import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ranking_metrics import (  # noqa: E402
    ndcg_at_k, precision_recall_at_k, ranking_report)


def test_perfect_ranking_scores_one():
    y_score = np.array([[0.9, 0.8, 0.1, 0.0]])
    y_rel = np.array([[1, 1, 0, 0]])
    p, r, h = precision_recall_at_k(y_score, y_rel, k=2)
    assert p == 1.0 and r == 1.0 and h == 1.0
    assert ndcg_at_k(y_score, y_rel, k=2) == 1.0


def test_bpr_style_ranking_beats_rating_heuristic():
    # Dois usuários; BPR ordena relevantes primeiro, heurística não.
    y_rel = np.array([[1, 0, 1, 0, 0], [0, 0, 1, 1, 0]])
    bpr_scores = np.array([[5.0, 0.0, 4.0, 1.0, 0.5], [0.0, 0.1, 5.0, 4.0, 0.2]])
    heur_scores = np.array([[0.0, 5.0, 0.1, 4.0, 3.0], [5.0, 4.0, 0.1, 0.0, 3.0]])
    rep_bpr = ranking_report(bpr_scores, y_rel, ks=(2,))
    rep_heur = ranking_report(heur_scores, y_rel, ks=(2,))
    assert rep_bpr[2]["ndcg"] > rep_heur[2]["ndcg"]
    assert rep_bpr[2]["recall"] >= rep_heur[2]["recall"]


def test_users_without_relevant_are_ignored():
    y_score = np.array([[0.5, 0.2], [0.9, 0.1]])
    y_rel = np.array([[0, 0], [1, 0]])
    p, r, h = precision_recall_at_k(y_score, y_rel, k=1)
    assert p == 1.0 and r == 1.0 and h == 1.0
