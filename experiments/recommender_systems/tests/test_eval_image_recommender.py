import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from eval_image_recommender import (  # noqa: E402
    cosine_scores, relevance_matrix, stratified_subset)
from ranking_metrics import ranking_report  # noqa: E402


def test_stratified_subset_is_balanced_sorted_and_reproducible():
    labels = np.repeat(np.arange(4), 50)
    a = stratified_subset(labels, 10, seed=1)
    assert np.array_equal(a, stratified_subset(labels, 10, seed=1))
    assert np.array_equal(a, np.sort(a))
    assert np.bincount(labels[a]).tolist() == [10, 10, 10, 10]


def test_relevance_excludes_the_query_itself():
    rel = relevance_matrix(np.array([0, 0, 1, 1, 1]))
    assert rel.diagonal().sum() == 0
    assert rel.sum(axis=1).tolist() == [1, 1, 2, 2, 2]


def test_cosine_masks_self_and_ranks_the_nearest_first():
    emb = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    s = cosine_scores(emb)
    assert np.isneginf(s.diagonal()).all()
    assert s[0].argmax() == 1


def test_perfect_embedding_scores_one_and_random_sits_at_chance():
    labels = np.repeat(np.arange(5), 20)
    onehot = np.eye(5)[labels] + 1e-3 * np.random.default_rng(0).random((100, 5))
    rel = relevance_matrix(labels)
    assert ranking_report(cosine_scores(onehot), rel, (5,))[5]["precision"] == 1.0
    rand = np.random.default_rng(0).random(rel.shape)
    np.fill_diagonal(rand, -np.inf)
    assert abs(ranking_report(rand, rel, (5,))[5]["precision"] - 19 / 99) < 0.05
