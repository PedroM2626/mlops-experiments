"""Retrieval quality of the image recommender (`image_recommender.ipynb`).

The notebook only reports an indexing benchmark; it has no ground truth. This
script gives the same pipeline (ResNet50 ImageNet-V2 embedding, 2048-d, L2
normalisation, cosine top-K) a measurable target by using CIFAR-10 class labels
as relevance: an item is "relevant" to a query when it shares the query's
class. Every gallery image is used once as the query (leave-one-out, the query
itself is masked with -inf), and the ranking is scored with `ranking_metrics`.

Two reference points are reported next to the ResNet50 embedding: raw pixels
(flattened, L2-normalised, cosine) and a seeded random ranking, so the numbers
can be read against a floor.

Data: CIFAR-10 test split (HF `cifar10`; local tarball as fallback).
ResNet50 weights come from torchvision (downloaded to the torch cache on first
use, exactly as the notebook does).

Usage:
    python eval_image_recommender.py [--per-class 200] [--batch-size 64]

Output: `experiments/artifacts/image_recommender_<timestamp>_<sha>/`
(`metrics.json`, `pip_freeze.txt`).
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ranking_metrics import ranking_report  # noqa: E402

TARBALL = HERE.parent / "computer_vision" / "data" / "cifar-10-python.tar.gz"
ART = HERE.parent / "artifacts"
SEED = 42
KS = (5, 10, 20)


def load_cifar10_test():
    """(images uint8 [N,32,32,3], labels [N]) of the CIFAR-10 test split.

    Same source as `computer_vision/run_hog_full.py` (HF `cifar10`), with the
    local tarball as fallback.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("cifar10", split="test")
        x = np.stack([np.array(im) for im in ds["img"]])
        return x, np.asarray(ds["label"])
    except Exception as e:
        print(f"[eval] HF cifar10 failed ({e}); trying the local tarball...")
    with tempfile.TemporaryDirectory(prefix="cifar10_") as tmp:
        with tarfile.open(TARBALL) as tf:
            tf.extractall(tmp, filter="data")
        with open(Path(tmp) / "cifar-10-batches-py" / "test_batch", "rb") as f:
            d = pickle.load(f, encoding="bytes")
    x = np.asarray(d[b"data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return x, np.asarray(d[b"labels"])


def stratified_subset(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Sorted indices with exactly `per_class` items of every class."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in np.unique(labels):
        pool = np.flatnonzero(labels == c)
        idx.append(rng.choice(pool, size=per_class, replace=False))
    return np.sort(np.concatenate(idx))


def relevance_matrix(labels: np.ndarray) -> np.ndarray:
    """1 where two different items share a class (the query itself is 0)."""
    rel = (labels[:, None] == labels[None, :]).astype(int)
    np.fill_diagonal(rel, 0)
    return rel


def cosine_scores(emb: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair, self-similarity masked with -inf."""
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n[n == 0] = 1.0
    e = emb / n
    s = e @ e.T
    np.fill_diagonal(s, -np.inf)
    return s


def resnet50_embeddings(images: np.ndarray, batch_size: int) -> np.ndarray:
    import torch
    from PIL import Image
    from torchvision import models

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()          # 2048-d pooled features, as in the notebook
    model.eval()
    tf = weights.transforms()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack([tf(Image.fromarray(a)) for a in images[i:i + batch_size]])
            out.append(model(batch).numpy())
            print(f"[embed] {min(i + batch_size, len(images))}/{len(images)}", flush=True)
    return np.vstack(out).astype(np.float32)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=HERE, text=True).strip()
    except Exception:
        return "nogit"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    x, y = load_cifar10_test()
    sel = stratified_subset(y, args.per_class, SEED)
    x, y = x[sel], y[sel]
    rel = relevance_matrix(y)
    print(f"[eval] gallery={len(y)} images, {len(np.unique(y))} classes, "
          f"{int(rel.sum(axis=1)[0])} relevant per query")

    t0 = time.time()
    emb = resnet50_embeddings(x, args.batch_size)
    embed_s = round(time.time() - t0, 1)

    rng = np.random.default_rng(SEED)
    rand = rng.random(rel.shape)
    np.fill_diagonal(rand, -np.inf)

    results = {
        "resnet50_cosine": ranking_report(cosine_scores(emb), rel, KS),
        "raw_pixels_cosine": ranking_report(
            cosine_scores(x.reshape(len(x), -1).astype(np.float32)), rel, KS),
        "random": ranking_report(rand, rel, KS),
    }
    for name, rep in results.items():
        print(name, json.dumps(rep))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ART / f"image_recommender_{stamp}_{git_sha()}"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({
        "experiment": "image_recommender",
        "protocol": "leave-one-out retrieval on a stratified CIFAR-10 test subset; "
                    "relevant = same class",
        "gallery_size": int(len(y)), "per_class": args.per_class,
        "relevant_per_query": int(rel.sum(axis=1)[0]),
        "embedding": "torchvision ResNet50 IMAGENET1K_V2, fc removed (2048-d), cosine",
        "embedding_seconds": embed_s, "device": "cpu",
        "seed": SEED, "git_sha": git_sha(), "run_timestamp": stamp,
        "results": {k: {str(kk): vv for kk, vv in v.items()} for k, v in results.items()},
    }, indent=2), encoding="utf-8")
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (out / "pip_freeze.txt").write_text(freeze, encoding="utf-8")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
