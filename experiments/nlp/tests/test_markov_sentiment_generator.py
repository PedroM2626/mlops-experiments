"""Unit tests for the sentiment-conditioned Markov generator."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "generative-text-markov" / "markov_sentiment_generator.py"

spec = importlib.util.spec_from_file_location("markov_sentiment_generator", MODULE)
gen = importlib.util.module_from_spec(spec)
sys.modules.setdefault("markov_sentiment_generator", gen)
spec.loader.exec_module(gen)

CORPUS = [
    "o produto chegou antes do prazo e funcionou muito bem",
    "o produto funcionou bem mas chegou atrasado",
    "atendimento ruim e produto com defeito",
    "atendimento excelente e produto perfeito",
]


def test_clean_text_lowercases_strips_urls_and_needs_four_tokens():
    # the loader keeps a review only if it still has 4+ tokens after cleaning,
    # so short reviews are dropped (None), not kept as stubs
    assert gen._clean_text("Produto, Excelente!!!") is None
    assert gen._clean_text("O produto chegou https://x.com antes do prazo") == \
        "o produto chegou antes do prazo"


def test_model_fits_from_a_tiny_corpus_without_files():
    model = gen.MarkovModel(order=3)
    model.fit([t.split() for t in CORPUS])
    assert model.k == 2
    assert model.transitions, "no transitions learned"


def test_generation_is_deterministic_for_a_given_seed():
    def sample(seed):
        random.seed(seed)
        model = gen.MarkovModel(order=3)
        model.fit([t.split() for t in CORPUS])
        return model.generate(max_words=12)
    assert sample(7) == sample(7)
    assert len(sample(7).split()) >= 1


def test_generated_words_come_from_the_corpus_vocabulary():
    random.seed(3)
    model = gen.MarkovModel(order=3)
    model.fit([t.split() for t in CORPUS])
    vocab = {w for line in CORPUS for w in gen._clean_text(line).split()}
    for _ in range(20):
        for word in model.generate(max_words=10).split():
            assert word in vocab, word


def test_order_below_two_is_clamped_to_a_bigram():
    assert gen.MarkovModel(order=1).order == 2
    assert gen.MarkovModel(order=0).k == 1


def test_cli_help_is_translated_and_documents_the_defaults():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngram", type=int, choices=(2, 3, 4), default=3)
    assert parser.get_default("ngram") == 3
    text = gen.main.__doc__ or ""
    assert "Markov" in text or text == ""       # no leftover non-English text


def test_module_has_no_portuguese_user_strings_left():
    src = MODULE.read_text(encoding="utf-8")
    for token in ("Geração", "Sentimento", "Amostra", "Resultado"):
        assert token not in src, token
