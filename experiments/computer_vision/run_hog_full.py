"""HOG+SVM no CIFAR-10 COMPLETO (50k treino / 10k teste).

Comparacao justa com ResNet18/ViT (que usaram 50k/10k): mesma receita do
notebook `cv-methods-comparison.ipynb` (gray -> 64x64 -> HOG 9ori/8x8/3x3 =
2.916 feats -> StandardScaler -> LinearSVC C=1), mas sem a subamostra de
10k/2k. Extracao paralelizada com joblib (o notebook era serial).

Dados: `computer_vision/data/cifar-10-python.tar.gz` local (sem download).
Features intermediarias (float32) vao p/ TEMP (nao commitadas); metricas em
`experiments/artifacts/hog_cifar10_<timestamp>/metrics.json`.

Uso:
    python run_hog_full.py [--jobs -1]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
TARBALL = HERE / "data" / "cifar-10-python.tar.gz"
ART = HERE.parent / "artifacts"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SEED = 42


def load_cifar10():
    """HF `cifar10` (mesma fonte do notebook). Fallback: tarball local."""
    try:
        from datasets import load_dataset
        ds = load_dataset("cifar10")
        Xtr = np.stack([np.array(ds["train"][i]["img"]) for i in range(len(ds["train"]))])
        ytr = np.array(ds["train"]["label"])
        Xte = np.stack([np.array(ds["test"][i]["img"]) for i in range(len(ds["test"]))])
        yte = np.array(ds["test"]["label"])
        print("[hog] fonte: HuggingFace cifar10")
        return Xtr, ytr, Xte, yte
    except Exception as e:
        print(f"[hog] HF falhou ({e}); tentando tarball local...")
    tmp = Path(tempfile.mkdtemp(prefix="cifar10_"))
    with tarfile.open(TARBALL) as tf:
        tf.extractall(tmp, filter="data")
    base = tmp / "cifar-10-batches-py"
    Xtr, ytr = [], []
    for i in range(1, 6):
        with open(base / f"data_batch_{i}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        Xtr.append(d[b"data"])
        ytr += d[b"labels"]
    with open(base / "test_batch", "rb") as f:
        d = pickle.load(f, encoding="bytes")
    Xte, yte = d[b"data"], d[b"labels"]
    Xtr = np.vstack(Xtr).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    Xte = np.array(Xte).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return Xtr, np.array(ytr), Xte, np.array(yte)


def hog_one(img):
    g = resize(rgb2gray(img), (64, 64), anti_aliasing=True)
    return hog(g, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(3, 3), block_norm="L2-Hys").astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    t0 = time.time()
    Xtr, ytr, Xte, yte = load_cifar10()
    print(f"[hog] dados: treino={Xtr.shape} teste={Xte.shape} "
          f"({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    Xtr_h = np.array(Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(hog_one)(im) for im in Xtr))
    Xte_h = np.array(Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(hog_one)(im) for im in Xte))
    print(f"[hog] features {Xtr_h.shape}/dim={Xtr_h.shape[1]} "
          f"({time.time()-t1:.0f}s, jobs={args.jobs})", flush=True)
    np.savez_compressed(os.path.join(tempfile.gettempdir(), "hog_cifar10_full.npz"),
                        Xtr=Xtr_h, Xte=Xte_h)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_h)
    Xte_s = scaler.transform(Xte_h)
    svm = LinearSVC(C=1.0, max_iter=5000, random_state=SEED, dual="auto")
    svm.fit(Xtr_s, ytr)
    pred = svm.predict(Xte_s)
    acc = accuracy_score(yte, pred)
    elapsed = time.time() - t0
    print(f"[hog] HOG+SVM FULL Accuracy: {acc:.4f} | Time: {elapsed:.0f}s")
    print(classification_report(yte, pred, target_names=CLASSES, zero_division=0))

    outdir = ART / f"hog_cifar10_{datetime.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps({
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "hog_dim": int(Xtr_h.shape[1]), "jobs": args.jobs,
        "accuracy": round(float(acc), 4),
        "elapsed_s": round(elapsed, 1), "seed": SEED,
        "report": classification_report(yte, pred, target_names=CLASSES,
                                        zero_division=0, output_dict=True),
    }, indent=2), encoding="utf-8")
    print(f"[hog] artefatos em {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
