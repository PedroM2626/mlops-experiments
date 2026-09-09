"""TS+NLP com dados REAIS: 8-K filings (SEC EDGAR) + precos (yfinance).

Desenho: cada 8-K e um evento datado com texto real e relevante ao mercado.
Alvo: direcao do ticker no pregao seguinte ao filing. Comparacao TS-only /
NLP-only / TS+NLP (LogReg + LightGBM), mesmo protocolo do notebook sintetico.

Texto: corpo do .htm principal do filing (BeautifulSoup, truncado em 512
tokens p/ FinBERT — documentado como limite). Sentimento: ProsusAI/finbert.
Precos: yfinance (ticker + ^GSPC como mercado).

Sem auth, rate policido (0.5 s/req) + cache incremental em JSONL (resume).
 tickers default: AAPL MSFT NVDA AMZN META TSLA JPM (2024-01-01 a 2026-08-31).

Uso:
    pip install yfinance  # + torch transformers lightgbm scikit-learn bs4
    python run_tsnlp_edgar.py [--tickers AAPL,MSFT] [--start 2024-01-01]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
CACHE = HERE / ".tsnlp_cache"
UA = {"User-Agent": "mlops-experiments research pedro@example.com"}
SEED = 42
CIKS = {"AAPL": "0000320193", "MSFT": "0000789019", "NVDA": "0001045810",
        "AMZN": "0001018724", "META": "0001326801", "TSLA": "0001318605",
        "JPM": "0000019617"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def filing_docs(ticker, start, end):
    cik = CIKS[ticker]
    d = get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    rec = d["filings"]["recent"]
    out = []
    for f, dt, acc in zip(rec["form"], rec["filingDate"], rec["accessionNumber"]):
        if f == "8-K" and start <= dt <= end:
            out.append((dt, acc))
    return out


def filing_text(ticker, acc):
    from bs4 import BeautifulSoup
    cik = CIKS[ticker].lstrip("0")
    nodash = acc.replace("-", "")
    idx = get_json(f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/index.json")
    items = idx.get("directory", {}).get("item", [])
    cands = [it["name"] for it in items
             if it["name"].endswith(".htm") and "index" not in it["name"].lower()
             and "xsl" not in it["name"].lower()]
    if not cands:
        return ""
    doc = max(cands, key=lambda n: next(
        (it.get("size", "0") for it in items if it["name"] == n), "0"))
    req = urllib.request.Request(
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{doc}", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        html = r.read()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    return text[:20000]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN,META,TSLA,JPM")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-08-31")
    args = ap.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    CACHE.mkdir(exist_ok=True)
    tpath = CACHE / "edgar_texts.jsonl"
    texts = {}
    if tpath.exists():
        for line in tpath.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                texts[r["k"]] = r["v"]
            except Exception:
                continue

    events = []
    for t in tickers:
        docs = filing_docs(t, args.start, args.end)
        print(f"[edgar] {t}: {len(docs)} 8-K", flush=True)
        for dt, acc in docs:
            k = f"{t}|{acc}"
            if k not in texts or not texts[k]:
                try:
                    txt = filing_text(t, acc)
                    time.sleep(0.5)
                except Exception as e:
                    print(f"[edgar] {k} falhou: {str(e)[:100]}", flush=True)
                    continue
                if txt:
                    texts[k] = {"date": dt, "ticker": t, "text": txt}
                    with open(tpath, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"k": k, "v": texts[k]},
                                           ensure_ascii=False) + "\n")
            if k in texts and texts[k]:
                events.append((t, dt, texts[k]["text"]))
    print(f"[edgar] eventos com texto: {len(events)}", flush=True)
    if len(events) < 30:
        print("[edgar] poucos eventos — abortando.")
        return 2

    # --- FinBERT ---
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
    mdl.eval()
    spath = CACHE / "edgar_scores.jsonl"
    scores = {}
    if spath.exists():
        for line in spath.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                scores[r["k"]] = r["v"]
            except Exception:
                continue
    todo = [(t, dt, tx) for t, dt, tx in events
            if f"{t}|{dt}|{hash(tx)}" not in scores]
    print(f"[finbert] device={device} novos={len(todo)}", flush=True)
    with torch.no_grad():
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
            enc = tok([c[2][:4000] for c in chunk], truncation=True, padding=True,
                      max_length=512, return_tensors="pt").to(device)
            probs = F.softmax(mdl(**enc).logits, dim=1).cpu().numpy()
            for (t, dt, tx), p in zip(chunk, probs):
                k = f"{t}|{dt}|{hash(tx)}"
                scores[k] = {"pos": round(float(p[2]), 4), "neg": round(float(p[0]), 4)}
                with open(spath, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": scores[k]}) + "\n")

    # --- Precos + painel ---
    import yfinance as yf
    px = yf.download(tickers + ["^GSPC"], start="2023-12-01", end=args.end,
                     progress=False, auto_adjust=True)["Close"]
    rows = []
    for t, dt, tx in events:
        try:
            closes = px[t].dropna()
            loc = closes.index.searchsorted(pd.Timestamp(dt))
            if loc + 1 >= len(closes) or loc < 21:
                continue
            r_next = float(np.log(closes.iloc[loc + 1] / closes.iloc[loc]))
            mom5 = float(np.log(closes.iloc[loc] / closes.iloc[loc - 5]))
            vol20 = float(np.log(closes.iloc[loc - 19:loc + 1] /
                                 closes.iloc[loc - 19:loc + 1].shift(1)).dropna().std())
            m = px["^GSPC"].dropna()
            mloc = m.index.searchsorted(pd.Timestamp(dt))
            m_mom5 = float(np.log(m.iloc[mloc] / m.iloc[mloc - 5])) if mloc >= 5 else 0.0
            s = scores.get(f"{t}|{dt}|{hash(tx)}", {})
            rows.append({"ticker": t, "date": dt, "target": int(r_next > 0),
                         "mom5": mom5, "vol20": vol20, "mkt_mom5": m_mom5,
                         "sent": float(s.get("pos", 0) - s.get("neg", 0)),
                         "doclen": len(tx)})
        except Exception:
            continue
    panel = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    print(f"[tsnlp] eventos validos: {len(panel)} "
          f"(base up={panel['target'].mean():.3f})", flush=True)
    if len(panel) < 40:
        print("[tsnlp] poucos eventos — abortando.")
        return 2

    TS, NLP = ["mom5", "vol20", "mkt_mom5"], ["sent", "doclen"]
    y = panel["target"].values
    cut = int(len(panel) * 0.7)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    import lightgbm as lgb
    res = {}
    for fname, cols in {"TS": TS, "NLP": NLP, "TS+NLP": TS + NLP}.items():
        Xtr, Xte = panel[cols].values[:cut], panel[cols].values[cut:]
        ytr, yte = y[:cut], y[cut:]
        lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        bm = lgb.train({"objective": "binary", "verbosity": -1, "seed": SEED},
                       lgb.Dataset(Xtr, ytr), num_boost_round=200)
        for mname, pred in {f"{fname}/logreg": lr.predict(Xte),
                            f"{fname}/lgbm": (bm.predict(Xte) > 0.5).astype(int)}.items():
            res[mname] = {"acc": round(float(accuracy_score(yte, pred)), 4),
                          "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
                          "auc": round(float(roc_auc_score(yte, pred)), 4)}
    for k, v in res.items():
        print(f"  {k}: {v}", flush=True)

    d = ART / f"tsnlp_edgar_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    panel.drop(columns=["doclen"]).to_csv(d / "panel.csv", index=False)
    (d / "metrics.json").write_text(json.dumps({
        "n_events": len(panel), "features": {"TS": TS, "NLP": NLP},
        "results": res}, indent=2), encoding="utf-8")
    print("artefatos em", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
