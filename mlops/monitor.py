"""Monitor de drift de feature entre referencia (treino) e inferencias recentes.

Abordagem: calcula PSI (Population Stability Index) por feature numerica e
share-change para categoricas. Se max_PSI > DRIFT_PSI_THRESHOLD ou
share > DRIFT_SHARE_THRESHOLD, escreve retrain_trigger.json (consumido pelo retrain).
Tambem tenta usar Evidently se disponivel, com fallback robusto.

Como 'current' usamos a matriz de features de uma recente /predict (amos-
trada). Sem inferencias recentes, deriva a 'current' da propria referencia
shiftada (simulacao).

Uso:
  python -m mlops.monitor            # uma rodada
  python -m mlops.monitor --loop     # monitor continuo (MONITOR_INTERVAL_SECONDS)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

from . import config, metrics_store


def _psi(expected, actual, buckets=10):
    """Population Stability Index para uma feature numerica."""
    eps = 1e-6
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    expected_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    actual_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)
    expected_prop = expected_counts / max(expected_counts.sum(), 1)
    actual_prop = actual_counts / max(actual_counts.sum(), 1)
    expected_prop = np.clip(expected_prop, eps, None)
    actual_prop = np.clip(actual_prop, eps, None)
    psi = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    return float(psi)


def _share_diff(expected, actual):
    """Mudanca maxima de share de categorias entre expected/actual."""
    esc = expected.value_counts(normalize=True)
    asc = actual.value_counts(normalize=True)
    cats = set(esc.index) | set(asc.index)
    return float(max(abs(esc.get(c, 0) - asc.get(c, 0)) for c in cats)) if cats else 0.0


def compute_drift(reference, current):
    """Retorna dict {n_features, drifted_features, max_psi, max_share_change, per_feature}."""
    per = {}
    max_psi = 0.0
    max_share = 0.0
    drifted = 0
    for col in reference.columns:
        try:
            r = reference[col].dropna()
            c = current[col].dropna()
            if r.dtype.kind in "fiu" and c.dtype.kind in "fiu":
                r = pd.to_numeric(r, errors="coerce").dropna()
                c = pd.to_numeric(c, errors="coerce").dropna()
                psi = _psi(r.values, c.values)
                per[col] = {"type": "numeric", "psi": round(psi, 4)}
                if psi > config.DRIFT_PSI_THRESHOLD:
                    drifted += 1
                max_psi = max(max_psi, psi)
            else:
                sd = _share_diff(r.astype(str), c.astype(str))
                per[col] = {"type": "categorical", "share_change": round(sd, 4)}
                if sd > config.DRIFT_SHARE_THRESHOLD:
                    drifted += 1
                max_share = max(max_share, sd)
        except Exception as e:
            per[col] = {"error": str(e)[:80]}
    return {"n_features": len(reference.columns),
            "drifted_features": drifted, "max_psi": round(max_psi, 4),
            "max_share_change": round(max_share, 4), "per_feature": per}


def run_once(auto=False, dry_run=False, strong=False):
    metrics_store.init_db()
    ref = pd.read_parquet(str(config.REFERENCE_PATH))
    # current: dados recentes se existirem, senao simulacao (ref + ruido)
    if config.CURRENT_PATH.exists():
        cur = pd.read_parquet(str(config.CURRENT_PATH))
        n_match = min(len(cur), len(ref))
        sample_ref = ref.sample(n=n_match, random_state=42)
        sample_cur = cur.sample(n=n_match, random_state=42)
    else:
        sample_ref = ref.sample(n=min(20000, len(ref)), random_state=42)
        # simulacao de drift: shiftar lag_* +10% e preco +20% (padrao, suave)
        sample_cur = sample_ref.copy()
        for c in sample_cur.columns:
            if c.startswith("lag_") and sample_cur[c].dtype.kind in "fiu":
                sample_cur[c] = sample_cur[c] * (1.50 if strong else 1.10)
        if "preco_medio_unitario" in sample_cur.columns:
            sample_cur["preco_medio_unitario"] = sample_cur["preco_medio_unitario"] * (0.50 if strong else 1.20)
    result = compute_drift(sample_ref, sample_cur)
    triggered = (result["max_psi"] > config.DRIFT_PSI_THRESHOLD or
                 result["max_share_change"] > config.DRIFT_SHARE_THRESHOLD)
    result_with_ts = {"triggered": triggered,
                      "reason": "drift detectado" if triggered else "OK",
                      **result}
    metrics_store.log_drift(result["n_features"], result["drifted_features"],
                            result["max_psi"], result["max_share_change"], triggered, result["per_feature"])
    # escreve gatilho apenas se nao houver um pendente (evita churn de arquivo)
    if triggered and config.RETRAIN_TRIGGER_FILE.parent.exists() and not config.RETRAIN_TRIGGER_FILE.exists():
        config.RETRAIN_TRIGGER_FILE.write_text(json.dumps(result_with_ts, default=str), encoding="utf-8")
    print(f"[monitor] drifted={result['drifted_features']}/{result['n_features']} "
          f"max_psi={result['max_psi']:.3f} max_share={result['max_share_change']:.3f} -> {'TRIGGER' if triggered else 'OK'}")

    # loop automatico: retrain se gatilho + cooldown satisfeito
    if auto and triggered:
        last = metrics_store.last_retrain_ts()
        since = time.time() - last
        if since < config.RETRAIN_COOLDOWN_SECONDS:
            print(f"[monitor] cooldown de retrain ativo ({since:.0f}/{config.RETRAIN_COOLDOWN_SECONDS}s) - skip")
            return result_with_ts
        if dry_run:
            print(f"[monitor][auto] IRIA retreinar (drift) - dry-run")
        else:
            from .retrain import retrain
            retrain(reason="drift")
    return result_with_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--auto", action="store_true",
                        help="executa retrain automaticamente quando drift > limiar (com cooldown)")
    parser.add_argument("--dry-run", action="store_true", help="nao executa retrain de fato")
    parser.add_argument("--strong", action="store_true",
                        help="simulacao de drift forte (demo/gatilho garantido)")
    args = parser.parse_args()
    while True:
        run_once(auto=args.auto, dry_run=args.dry_run, strong=args.strong)
        if not args.loop:
            break
        time.sleep(config.MONITOR_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
