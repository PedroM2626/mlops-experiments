"""Recalibracao conformal dos intervalos do DeepAR (series reais, §5.12 do README).

Re-treina o DeepAR por dataset (mesma config do notebook
`deepar-probabilistic-forecast.ipynb`), coleta as 100 trajetorias e aplica
split-conformal: metade final do treino como calibracao (residuos
padronizados pelo desvio das trajetorias), q_hat global e por horizonte,
avaliados no holdout. Salva JSON em
`experiments/artifacts/deepar_conformal_<ts>/metrics.json`.

~5 min de CPU no total.

Dependencias (pip): gluonts torch statsmodels scikit-learn pandas numpy
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
SEED = 42


def load_co2():
    from statsmodels.datasets.co2 import load_pandas
    df = load_pandas().data.resample("W").mean().interpolate().reset_index()
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def load_nile():
    from statsmodels.datasets.nile import load_pandas as _l
    v = _l().data.iloc[:, 1].dropna().values.astype(float)
    return pd.DataFrame({"date": pd.date_range("1871-01-01", periods=len(v), freq="YS"),
                         "value": v})


def load_sunspots():
    from statsmodels.datasets.sunspots import load_pandas as _l
    v = _l().data.iloc[:, 1].dropna().values.astype(float)
    return pd.DataFrame({"date": pd.date_range("1700-01-01", periods=len(v), freq="YS"),
                         "value": v})


def load_synthetic():
    rng = np.random.RandomState(SEED)
    n = 200
    y = np.zeros(n)
    for t in range(1, n):
        jump = rng.normal(0, 8) if rng.random() < 0.05 else 0.0
        y[t] = 50 + 0.15 * t + 10 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 3) + jump
    return pd.DataFrame({"date": pd.date_range("2020-01-06", periods=n, freq="W"),
                         "value": y})


DATASETS = {
    "CO2": (load_co2, 30, "W"),
    "Nile": (load_nile, 8, "YS"),
    "Sunspots": (load_sunspots, 25, "YS"),
    "Synthetic": (load_synthetic, 30, "W"),
}
FREQ_MAP = {"YS": "Y"}


def run_deepar_samples(values, dates, horizon, freq, num_samples=100):
    from gluonts.torch import DeepAREstimator
    from gluonts.torch.distributions import StudentTOutput
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.evaluation import make_evaluation_predictions
    from lightning.pytorch.callbacks import EarlyStopping
    import tempfile
    import torch

    torch.manual_seed(SEED)
    df_ts = pd.DataFrame({"date": pd.to_datetime(dates), "y": values}).set_index("date").sort_index()
    gf = FREQ_MAP.get(freq, freq)
    if gf == "Y":
        df_ts.index = df_ts.index.to_period("Y")
    train_df = df_ts.iloc[:-horizon]
    train_ds = PandasDataset({"train": train_df}, target="y")
    context_length = max(horizon * 2, 10)
    est = DeepAREstimator(freq=gf if gf != "W" else "W", context_length=context_length,
                          prediction_length=horizon, num_layers=2, hidden_size=40,
                          distr_output=StudentTOutput(), lr=1e-3, batch_size=32,
                          trainer_kwargs={
                              "max_epochs": 30, "enable_progress_bar": False,
                              "enable_model_summary": False, "logger": False,
                              "default_root_dir": tempfile.mkdtemp(prefix="deepar_"),
                              "callbacks": [EarlyStopping(monitor="train_loss", patience=10,
                                                          mode="min")]})
    predictor = est.train(train_ds)
    forecast_it, _ = make_evaluation_predictions(
        dataset=PandasDataset({"test": df_ts}, target="y"),
        predictor=predictor, num_samples=num_samples)
    fc = next(iter(forecast_it))
    return np.asarray(fc.samples)  # (100, horizon)


def coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean())


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=6,
                    help="origens rolling p/ calibracao (default 6)")
    args = ap.parse_args()

    out = {"seed": SEED, "nominal": 0.9, "windows": args.windows, "datasets": {}}
    for name, (loader, horizon, freq) in DATASETS.items():
        t0 = time.time()
        df = loader()
        y = df["value"].values.astype(float)
        dates = df["date"].values
        N = len(y)
        y_test = y[-horizon:]
        # Pool de calibracao: W origens rolling antes do holdout. Cada origem
        # treina em y[:end], prevê `horizon` e coleta resíduos padronizados.
        pool = []  # lista de (residuos_padronizados [horizon])
        context_length = max(horizon * 2, 10)
        for w in range(args.windows, 0, -1):
            end = N - horizon * w
            if end < context_length + horizon:
                continue
            try:
                s = run_deepar_samples(y[:end], dates[:end], horizon, freq)
            except Exception as e:
                print(f"[{name}] janela w={w} pulada: {str(e)[:100]}", flush=True)
                continue
            mu_c, sd_c = s.mean(0), s.std(0).clip(min=1e-9)
            pool.append(np.abs(y[end:end + horizon] - mu_c) / sd_c)
        if not len(pool):
            print(f"[{name}] sem janelas de calibracao — pulando dataset", flush=True)
            continue
        pool = np.array(pool)
        s_test = run_deepar_samples(y, dates, horizon, freq)
        mu, sd = s_test.mean(0), s_test.std(0).clip(min=1e-9)

        z = 1.645
        cov_before = coverage(y_test, mu - z * sd, mu + z * sd)
        w_before = float(2 * z * sd.mean())
        scores = pool.reshape(-1)
        q_g = float(np.quantile(scores, 0.9))
        q_h = np.quantile(pool, 0.9, axis=0)
        cov_g = coverage(y_test, mu - q_g * sd, mu + q_g * sd)
        cov_h = coverage(y_test, mu - q_h * sd, mu + q_h * sd)
        out["datasets"][name] = {
            "horizon": horizon,
            "n_cal_windows": int(len(pool)),
            "n_cal_points": int(pool.size),
            "coverage_antes": round(cov_before, 4),
            "largura_antes": round(w_before, 4),
            "q_global": round(q_g, 4),
            "coverage_depois_global": round(cov_g, 4),
            "largura_depois_global": round(float(2 * q_g * sd.mean()), 4),
            "coverage_depois_por_h": round(cov_h, 4),
            "largura_depois_por_h": round(float((2 * q_h * sd).mean()), 4),
            "treino_s": round(time.time() - t0, 1),
        }
        print(f"[{name}] antes={cov_before:.3f} -> global={cov_g:.3f} "
              f"por_h={cov_h:.3f} | {time.time()-t0:.0f}s", flush=True)

    d = ART / f"deepar_conformal_{datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("artefatos em", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
