"""Unit tests for the A/B/C pipeline core (the three preprocessing variants).

No dataset, model or training: the cleaners are pure string functions, and they
are what the ablation study in this folder compares, so their documented
behaviour is pinned here. The module imports NLTK data at import time, so the
whole file skips when the corpora are not installed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CORE = (HERE.parent / "twitter-entity-sentiment"
        / "pipelines_abc_comparison" / "pipelines_abc_core.py")


def _load():
    spec = importlib.util.spec_from_file_location("pipelines_abc_core", CORE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("pipelines_abc_core", mod)
    spec.loader.exec_module(mod)
    return mod


if not CORE.exists():                            # moved without updating this test
    raise RuntimeError(f"pipelines_abc_core.py not found at {CORE}")
try:
    import nltk                                    # noqa: F401
except ImportError as exc:                          # missing nltk itself
    pytest.skip(f"nltk is not installed ({exc})", allow_module_level=True)
try:
    abc_core = _load()
except LookupError as exc:                       # corpora not downloaded
    pytest.skip(f"NLTK data not downloaded ({exc}); see the bootstrap step in "
                "experiments/nlp/README.md", allow_module_level=True)


@pytest.mark.parametrize("cleaner", [abc_core.clean_a, abc_core.clean_b,
                                     abc_core.clean_c])
def test_non_string_input_returns_empty(cleaner):
    assert cleaner(None) == ""
    assert cleaner(12345) == ""


def test_pipeline_a_is_aggressive_by_default():
    out = abc_core.clean_a("Verizon #fellover https://t.co/x @user great 2 deal!!")
    assert out == "verizon great deal"
    for gone in ("http", "@user", "#fellover", "!", "2"):
        assert gone not in out


def test_pipeline_a_toggles_keep_the_tokens_they_name():
    text = "Verizon #fellover @user great 2 deal!"
    kept = abc_core.clean_a(text, keep_hashtags=True, keep_punct=True,
                            keep_digits=True)
    assert "fellover" in kept and "2" in kept and "!" in kept
    assert "@" not in kept                       # mentions are always removed


def test_pipeline_b_is_conservative_by_default():
    out = abc_core.clean_b("Verizon #fellover @user great 2 deal!!")
    assert out == "verizon fellover great 2 deal!!"
    assert "fellover" in out                     # hashtag content kept as a word
    assert "2" in out and "!" in out
    assert "@user" not in out


def test_pipeline_b_drop_toggles_remove_what_they_name():
    text = "Verizon #fellover great 2 deal!"
    assert "fellover" not in abc_core.clean_b(text, drop_hashtags=True)
    assert "2" not in abc_core.clean_b(text, drop_digits=True)
    assert "!" not in abc_core.clean_b(text, drop_punct=True)


def test_pipeline_c_drops_stopwords_and_keeps_negation():
    assert abc_core.clean_c("The movie was great and the plot was awesome",
                            lemmatize=False) == "movie great plot awesome"
    # 'not' is deliberately kept out of the stopword list: negation carries
    # the sentiment, and NLTK's tokenizer splits "can't" into "can" + "not".
    assert abc_core.clean_c("this is not good", lemmatize=False) == "not good"
    assert abc_core.clean_c("Movie was n't great!", lemmatize=False) == "movie not great !"


def test_pipeline_c_keeps_only_bang_and_question_punctuation():
    out = abc_core.clean_c("Great!!! Why? $5.00 @verizon", keep_question_mark=True,
                           lemmatize=False)
    assert out == "great ! ! ! ?"
    assert "$" not in out and "@" not in out and "verizon" not in out


def test_cleaners_are_idempotent_on_already_clean_text():
    for cleaner in (abc_core.clean_a, abc_core.clean_b):
        once = cleaner("simple words only")
        assert cleaner(once) == once


def test_cleaner_registry_matches_the_three_documented_pipelines():
    assert set(abc_core.CLEANERS) == {"A", "B", "C"}
    assert abc_core.CLEANERS["A"] is abc_core.clean_a
    assert abc_core.CLEANERS["B"] is abc_core.clean_b
    assert abc_core.CLEANERS["C"] is abc_core.clean_c


def test_documented_sentiment_labels_are_the_four_original_ones():
    assert abc_core.VALID_SENTIMENTS == ["Positive", "Negative", "Neutral",
                                         "Irrelevant"]
    assert abc_core.COLUMNS == ["id", "entity", "sentiment", "text"]
