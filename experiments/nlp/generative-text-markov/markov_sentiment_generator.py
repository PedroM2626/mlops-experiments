#!/usr/bin/env python3
"""Gerador de frases por sentimento usando cadeias de Markov / n-gramas.

Treina um modelo de Markov separado para cada classe de sentimento
(Positive, Negative, Neutral) e gera frases novas amostrando a proxima palavra.

Datasets suportados:
    b2w      B2W-Reviews01 (reviews da Americanas em PT-BR; nota 1-5 mapeada
             em Negative <=2, Neutral ==3, Positive >=4)  [padrao]
    twitter  Senti-Pred / Twitter Entity Sentiment Analysis (ingles)

Uso:
    python markov_sentiment_generator.py                          # B2W, todas as categorias
    python markov_sentiment_generator.py --dataset twitter        # dataset original (ingles)
    python markov_sentiment_generator.py --sentiment negative     # so negativa
    python markov_sentiment_generator.py --ngram 3 --samples 5 --seed 42
"""
import argparse
import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR.parent.parent / "datasets"
TWITTER_DATA_DIR = (BASE_DIR.parent / "twitter-entity-sentiment"
                    / "senti-pred-variations" / "Senti-Pred-remake2" / "data" / "raw")
TWITTER_FILES = ("twitter_training.csv", "twitter_validation.csv")
B2W_PATH = DATASETS_DIR / "B2W-Reviews01.csv"
SENTIMENTS = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}
START, END = "<s>", "</s>"
URL_MENTION_RE = re.compile(r"(https?://\S+|www\.\S+|@\w+)")
TOKEN_RE = re.compile(r"[a-z\u00e0-\u00fa']+")


def _clean_text(text: str):
    tokens = TOKEN_RE.findall(URL_MENTION_RE.sub(" ", text).lower())
    return " ".join(tokens) if len(tokens) >= 4 else None


def load_b2w_texts(label: str):
    texts = set()
    with B2W_PATH.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        idx_rating = header.index("overall_rating")
        idx_title = header.index("review_title")
        idx_text = header.index("review_text")
        for row in reader:
            if len(row) <= max(idx_rating, idx_text):
                continue
            try:
                rating = int(float(row[idx_rating]))
            except ValueError:
                continue
            mapped = "Negative" if rating <= 2 else "Neutral" if rating == 3 else "Positive"
            if mapped != label:
                continue
            parts = [row[idx_title], row[idx_text]]
            cleaned = _clean_text(". ".join(p for p in parts if isinstance(p, str) and p.strip()))
            if cleaned:
                texts.add(cleaned)
    return sorted(texts)


def load_twitter_texts(label: str):
    texts = set()
    for name in TWITTER_FILES:
        path = TWITTER_DATA_DIR / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for row in csv.reader(fh):
                if len(row) < 4 or row[2].strip().lower() != label.lower():
                    continue
                cleaned = _clean_text(row[3])
                if cleaned:
                    texts.add(cleaned)
    return sorted(texts)


LOADERS = {
    "b2w": load_b2w_texts,
    "twitter": load_twitter_texts,
}


class MarkovModel:
    def __init__(self, order: int = 3):
        self.order = max(2, order)
        self.k = self.order - 1
        self.transitions = defaultdict(Counter)

    def fit(self, token_lists):
        for tokens in token_lists:
            seq = [START] * self.k + list(tokens) + [END]
            for i in range(self.k, len(seq)):
                ctx = tuple(seq[i - self.k:i])
                self.transitions[ctx][seq[i]] += 1
                self.transitions[()][seq[i]] += 1

    def _sample_from(self, ctx, ban=None):
        for cut in range(len(ctx) + 1):
            counter = self.transitions.get(ctx[cut:])
            if not counter:
                continue
            items = [(w, c) for w, c in counter.items() if w != ban]
            if not items:
                continue
            words, weights = zip(*items)
            return random.choices(words, weights=weights)[0]
        return END

    def generate(self, max_words: int = 25, max_repeat: int = 1) -> str:
        ctx = (START,) * self.k
        words = []
        while len(words) < max_words:
            nxt = self._sample_from(ctx)
            if nxt == END:
                break
            if len(words) >= max_repeat and all(w == nxt for w in words[-max_repeat:]):
                nxt = self._sample_from(ctx, ban=nxt)
                if nxt == END:
                    break
            words.append(nxt)
            ctx = ctx[1:] + (nxt,)
        return " ".join(words)


def train_models(dataset: str, order: int):
    loader = LOADERS[dataset]
    models = {}
    for key, label in SENTIMENTS.items():
        raw_texts = loader(label)
        token_lists = [t.split() for t in raw_texts]
        model = MarkovModel(order)
        model.fit(token_lists)
        vocab = {w for tokens in token_lists for w in tokens}
        models[key] = {
            "model": model,
            "n_tweets": len(token_lists),
            "vocab_size": len(vocab),
            "originals": set(raw_texts),
        }
    return models


def sample_sentence(entry, max_words: int, attempts: int = 25):
    for _ in range(attempts):
        sentence = entry["model"].generate(max_words=max_words)
        if len(sentence.split()) >= 4 and sentence not in entry["originals"]:
            return sentence
    return sentence


def main():
    parser = argparse.ArgumentParser(description="Gera frases com Markov/n-gramas treinado em dataset de sentimentos.")
    parser.add_argument("--dataset", choices=list(LOADERS), default="b2w",
                        help="Dataset de treino (padrao: b2w, reviews da Americanas em PT-BR).")
    parser.add_argument("--sentiment", choices=[*SENTIMENTS, "all"], default="all",
                        help="Categoria desejada (padrao: todas).")
    parser.add_argument("--ngram", type=int, choices=(2, 3, 4), default=3,
                        help="Ordem do n-grama (2=bigrama, 3=trigrama). Padrao: 3.")
    parser.add_argument("--samples", type=int, default=1, help="Frases geradas por categoria (padrao: 1).")
    parser.add_argument("--max-words", type=int, default=25, help="Tamanho maximo da frase gerada.")
    parser.add_argument("--seed", type=int, default=None, help="Seed para reprodutibilidade.")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    targets = SENTIMENTS.keys() if args.sentiment == "all" else [args.sentiment]
    models = train_models(args.dataset, args.ngram)

    print(f"Dataset: {args.dataset} | Modelo Markov de ordem {args.ngram} "
          f"(contexto de {args.ngram - 1} palavra(s))")
    print("=" * 70)
    for key in SENTIMENTS:
        info = models[key]
        status = "OK " if key in targets else "-- "
        print(f"[{status}] {key:>8}: {info['n_tweets']:>6} textos | vocabulario de {info['vocab_size']:>6} palavras")
    print("=" * 70)

    for key in SENTIMENTS:
        if key not in targets:
            continue
        print(f"\n--- Frase(s) gerada(s) [{key.upper()}] ---")
        for i in range(args.samples):
            print(f"  {i + 1}. {sample_sentence(models[key], args.max_words)}")


if __name__ == "__main__":
    main()
