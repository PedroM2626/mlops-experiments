"""TS+NLP com dados REAIS: S&P 500 (yfinance) + manchetes (GDELT DOC 2.1).

Mesmo desenho do notebook sintetico (`stock-sentiment-ts-nlp.ipynb`):
direcao do dia seguinte (up/down) com LightGBM/LogReg em 3 conjuntos
(TS-only, NLP-only, TS+NLP). Sentimento via FinBERT (ProsusAI/finbert).

Janela default: 2026-03-02 a 2026-08-31 (~130 pregoes). GDELT tem rate
limit agressivo: 20 s entre requests + cache incremental em JSONL (resume
seguro se interrompido).

Uso:
    python run_tsnlp_real.py [--start 2026-03-02] [--end 2026-08-31]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
CACHE = HERE / ".tsnlp_cache"
SEED = 42


def setup_cache():
    CACHE.mkdir(exist_ok=True)
    return CACHE / "headlines.jsonl", CACHE / "scores.jsonl"


def read_jsonl(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                out[r["k"]] = r["v"]
            except Exception:
                continue
    return out


def append_jsonl(path, k, v):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def gdelt_day(day: str) -> list:
    """Manchetes de um dia (YYYY-MM-DD). Lista de (seendate, title)."""
    q = urllib.parse.urlencode({
        "query": "stock market", "mode": "artlist", "maxrecords": 50,
        "format": "json", "startdatetime": day.replace("-", ""),
        "enddatetime": day.replace("-", "")})
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?{q}"
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.load(r)
            return [(a.get("seendate", ""), (a.get("title") or "").strip())
                    for a in d.get("articles", []) if a.get("title")]
        except Exception as e:
            last = e
            time.sleep(15 * (attempt + 1))
    print(f"[gdelt] {day} falhou apos retries: {str(last)[:100]}", flush=True)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-03-02")
    ap.add_argument("--end", default="2026-08-31")
    args = ap.parse_args()
    hpath, spath = setup_cache()
    headlines = read_jsonl(hpath)
    cached_scores = read_jsonl(spath)

    import yfinance as yf
    px = yf.download("^GSPC", start=args.start, end=args.end, progress=False,
                     auto_adjust=True)
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(0)
    px = px[["Close"]].dropna()
    days = [d.strftime("%Y-%m-%d") for d in px.index]
    print(f"[tsnlp] pregoes: {len(days)} ({days[0]}..{days[-1]})", flush=True)

    for i, day in enumerate(days):
        if day not in headlines or not headlines[day]:
            fetched = gdelt_day(day)
            if fetched:  # nao cacheia falha/vazio: permite retry futuro
                headlines[day] = fetched
                append_jsonl(hpath, day, fetched)
            else:
                headlines[day] = []
            print(f"[gdelt] {day}: {len(headlines[day])} manchetes "
                  f"({i+1}/{len(days)})", flush=True)
            time.sleep(20)
    n_titles = sum(len(v) for v in headlines.values())
    print(f"[tsnlp] total manchetes: {n_titles}", flush=True)

    # --- FinBERT (cache por titulo) ---
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    mdl.eval()
    all_titles = sorted({t for v in headlines.values() for _, t in v if t})
    todo = [t for t in all_titles if t not in cached_scores]
    print(f"[finbert] device={device} titulos={len(all_titles)} novos={len(todo)}", flush=True)
    import torch.nn.functional as F
    BS = 64
    with torch.no_grad():
        for i in range(0, len(todo), BS):
            enc = tok(todo[i:i + BS], truncation=True, padding=True,
                      max_length=64, return_tensors="pt").to(device)
            probs = F.softmax(mdl(**enc).logits, dim=1).cpu().numpy()
            for t, p in zip(todo[i:i + BS], probs):
                cached_scores[t] = {"neg": round(float(p[0]), 4),
                                    "neu": round(float(p[1]), 4),
                                    "pos": round(float(p[2]), 4)}
                append_jsonl(spath, t, cached_scores[t])
    print("[finbert] ok", flush=True)

    # --- Painel diario ---
    px["ret"] = np.log(px["Close"] / px["Close"].shift(1))
    rows = []
    for j, day in enumerate(days):
        if j < 1:
            continue
        ts = headlines.get(day, [])
        sc = [cached_scores[t] for _, t in ts if t in cached_scores]
        sent = float(np.mean([s["pos"] - s["neg"] for s in sc])) if sc else 0.0
        rows.append({"date": day, "ret": float(px["ret"].iloc[j]),
                     "n_news": len(sc), "sent": sent})
    panel = pd.DataFrame(rows).dropna().reset_index(drop=True)
    panel["target"] = (panel["ret"].shift(-1) > 0).astype(int)
    panel = panel.iloc[:-1].reset_index(drop=True)
    for L in (1, 2, 3, 5):
        panel[f"ret_lag{L}"] = panel["ret"].shift(L)
        panel[f"sent_lag{L}"] = panel["sent"].shift(L)
    panel["ret_ma5"] = panel["ret"].rolling(5).mean()
    panel["ret_std5"] = panel["ret"].rolling(5).std()
    panel["dow"] = pd.to_datetime(panel["date"]).dt.dayofweek
    panel = panel.dropna().reset_index(drop=True)
    print(f"[tsnlp] painel: {len(panel)} dias", flush=True)

    TS = [c for c in panel.columns if c.startswith(("ret_lag", "ret_ma", "ret_std"))] + ["dow"]
    NLP = ["sent"] + [c for c in panel.columns if c.startswith("sent_lag")] + ["n_news"]
    y = panel["target"].values
    cut = int(len(panel) * 0.7)
    print(f"[tsnlp] split temporal: treino={cut} teste={len(panel)-cut} "
          f"(base={y[cut:].mean():.3f})", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import lightgbm as lgb
    res = {}
    for fname, cols in {"TS": TS, "NLP": NLP, "TS+NLP": TS + NLP}.items():
        Xtr, Xte = panel[cols].values[:cut], panel[cols].values[cut:]
        ytr, yte = y[:cut], y[cut:]
        lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        dtr = lgb.Dataset(Xtr, ytr)
        bm = lgb.train({"objective": "binary", "verbosity": -1, "seed": SEED},
                       dtr, num_boost_round=200)
        for mname, pred in {f"{fname}/logreg": lr.predict(Xte),
                            f"{fname}/lgbm": (bm.predict(Xte) > 0.5).astype(int)}:
            res[mname] = {"acc": round(float(accuracy_score(yte, pred)), 4),
                          "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
                          "auc": round(float(roc_auc_score(yte, pred)), 4)}
    for k, v in res.items():
        print(f"  {k}: {v}", flush=True)

    d = ART / f"tsnlp_real_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    panel.to_csv(d / "panel.csv", index=False)
    (d / "metrics.json").write_text(json.dumps({
        "start": args.start, "end": args.end, "n_days": len(panel),
        "n_headlines": n_titles, "features": {"TS": TS, "NLP": NLP},
        "results": res}, indent=2), encoding="utf-8")
    print("artefatos em", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
