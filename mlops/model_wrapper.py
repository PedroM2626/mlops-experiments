"""pyfunc wrapper for SalesForecasterV2 for the MLflow model registry.

The pyfunc encapsulates:
  - the LightGBM artifact (booster + feature_names + categorical + flags)
  - the SalesForecasterV2 class (feature_engineering + generate_forecasts)
  - the historical data (2022) that feeds generate_forecasts

Expected input (df or dict): {"weeks_to_forecast": int, "top_n": int|None}
Output: DataFrame with columns [semana, pdv, sku, quantidade_prevista]
"""
from __future__ import annotations
import os
import sys
import threading
import warnings
import joblib
import numpy as np
import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel

SALES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "sales-forecast")
sys.path.insert(0, os.path.join(SALES_DIR, "scripts"))
from forecaster_class import SalesForecasterV2  # noqa: E402

DATA_PATHS = {
    "vendas": os.path.join(SALES_DIR, "data", "raw", "fato_vendas.parquet"),
    "pdvs": os.path.join(SALES_DIR, "data", "raw", "dim_pdvs.parquet"),
    "produtos": os.path.join(SALES_DIR, "data", "raw", "dim_produtos.parquet"),
}


_EMPTY_FC = pd.DataFrame(columns=["pdv", "sku", "semana", "quantidade_prevista"])


def build_forecast_state(forecaster, df_historical):
    """Invariant part of the forecast (depends only on the static history):
    sliding matrix of 53 offsets per (pdv, sku), prices/parity, week of the
    last observation, categorical dims and the model categories. Cacheable."""
    model = forecaster.model
    feature_names = list(forecaster.feature_names)
    cat_features = list(forecaster.categorical_features)
    use_log = bool(forecaster.use_log_target)
    DIFF = 53

    arr = df_historical.sort_values(["pdv", "sku", "ano", "semana"])
    if arr.empty:
        return None

    gb = arr.groupby(["pdv", "sku"], sort=False)
    arr = arr.assign(_code=gb.ngroup(), _seq=gb.cumcount())
    sizes = gb.size().to_numpy()
    n = len(sizes)
    if n == 0:
        return None

    codes = arr["_code"].to_numpy()
    seq = arr["_seq"].to_numpy()
    sizes_c = sizes[codes]
    idx = sizes_c - 1 - seq  # 0 = most recent observation of each combo

    qty = arr["quantidade"].to_numpy(dtype=np.float32)
    keep = idx < DIFF
    Q = np.full((n, DIFF), np.nan, dtype=np.float32)
    Q[codes[keep], idx[keep]] = qty[keep]

    P0 = np.full(n, np.nan, dtype=np.float32)  # price of the last real week
    P1 = np.full(n, np.nan, dtype=np.float32)  # price of the second-to-last real week
    m_last = seq == (sizes_c - 1)
    m_prev = seq == (sizes_c - 2)
    P0[codes[m_last]] = arr.loc[m_last, "preco_medio_unitario"].to_numpy(dtype=np.float32)
    P1[codes[m_prev]] = arr.loc[m_prev, "preco_medio_unitario"].to_numpy(dtype=np.float32)

    S0 = np.full(n, np.nan, dtype=np.float64)
    S0[codes[m_last]] = arr.loc[m_last, "semana"].to_numpy(dtype=np.float64)

    last_rows = arr[m_last].sort_values("_code")
    combos = last_rows[["pdv", "sku"]].reset_index(drop=True)
    dim_cols = [c for c in df_historical.columns
                if c not in ("ano", "semana", "pdv", "sku", "quantidade", "preco_medio_unitario")]
    for c in dim_cols:
        combos[c] = last_rows[c].reset_index(drop=True)

    cats = {}
    for col in cat_features:
        cats[col] = pd.CategoricalDtype(
            categories=model.booster_.pandas_categorical[cat_features.index(col)])
    cat_ok = np.ones(n, dtype=bool)
    for col in cat_features:
        cat_ok &= ~pd.isna(pd.Categorical(combos[col].values, dtype=cats[col]))
    if not cat_ok.all():
        Q = Q[cat_ok]
        P0, P1, S0 = P0[cat_ok], P1[cat_ok], S0[cat_ok]
        combos = combos[cat_ok].reset_index(drop=True)

    return {"model": model, "feature_names": feature_names,
            "cat_features": cat_features, "use_log": use_log,
            "combos": combos, "cats": cats, "Q": Q, "P0": P0, "P1": P1, "S0": S0}


def _yield_week_predictions(state, weeks_to_forecast):
    """Generator: yields (wk, pred int64) for each week of the horizon.

    Reproduces exactly the features/quirks of the original generate_forecasts
    (week=w with sine/cosine/previous quarter, lag_diff_1 pre-fillna, fp32
    prices, confirm_matrix etc), but 100% vectorized from the state."""
    if weeks_to_forecast <= 0 or state is None:
        return
    Q = state["Q"].copy()  # the recursion mutates Q locally; the original state stays intact (cache)
    P0, P1, S0 = state["P0"], state["P1"], state["S0"]
    combos, cats = state["combos"], state["cats"]
    model, use_log = state["model"], state["use_log"]
    feature_names, cat_features = state["feature_names"], state["cat_features"]
    n = Q.shape[0]

    for wk in range(1, weeks_to_forecast + 1):
        V = Q[:, 1:] if Q.shape[1] > 1 else np.empty((n, 0), dtype=np.float32)
        lags = {}
        _zero = lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        for k in (1, 2, 3, 4, 12, 52):
            lags[k] = V[:, k - 1].astype(np.float64) if V.shape[1] >= k else np.full(n, np.nan)

        roll = {}
        for w in (4, 12, 52):
            if V.shape[1] == 0:
                roll[w] = {"mean": np.full(n, np.nan), "std": np.full(n, np.nan),
                           "max": np.full(n, np.nan), "min": np.full(n, np.nan)}
            else:
                W = V[:, :w].astype(np.float64)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    if w in (4, 12):
                        roll[w] = {"mean": np.nanmean(W, axis=1), "std": np.nanstd(W, axis=1, ddof=1),
                                   "max": np.nanmax(W, axis=1), "min": np.nanmin(W, axis=1)}
                    else:
                        roll[w] = {"mean": np.nanmean(W, axis=1), "std": np.nanstd(W, axis=1, ddof=1),
                                   "max": np.nanmax(W, axis=1), "min": np.full(n, np.nan)}

        # original quirk: 'semana'=w, but sine/cosine/quarter of the previous week
        # (w-1; on the last real week when w==1)
        if wk == 1:
            sem_ref = S0
        else:
            sem_ref = np.full(n, float(wk - 1))

        lag1_preco = _zero(np.where(wk % 2 == 1, P1, P0).astype(np.float64))
        preco_feat = _zero(np.where(wk % 2 == 1, P0, P1).astype(np.float64))
        seno = np.sin(2 * np.pi * sem_ref / 52)
        cosseno = np.cos(2 * np.pi * sem_ref / 52)
        trim = ((sem_ref - 1) // 13 + 1).astype(np.int64)

        coef4 = np.zeros(n, dtype=np.float64)
        m4 = roll[4]["mean"] > 0
        coef4[m4] = roll[4]["std"][m4] / roll[4]["mean"][m4]

        cols = {c: combos[c].values for c in cat_features}
        cols.update({
            "semana": np.full(n, wk, dtype=np.int64),
            "trimestre": trim,
            "seno_semana": seno.astype(np.float64),
            "cosseno_semana": cosseno.astype(np.float64),
            "lag_1_semanas": _zero(lags[1]), "lag_2_semanas": _zero(lags[2]),
            "lag_3_semanas": _zero(lags[3]), "lag_4_semanas": _zero(lags[4]),
            "lag_12_semanas": _zero(lags[12]), "lag_52_semanas": _zero(lags[52]),
            "lag_1_preco": lag1_preco,
            "lag_diff_1": _zero(lags[1] - lags[2]),
            "rolling_mean_4_semanas": _zero(roll[4]["mean"]),
            "rolling_std_4_semanas": _zero(roll[4]["std"]),
            "rolling_max_4_semanas": _zero(roll[4]["max"]),
            "rolling_min_4_semanas": _zero(roll[4]["min"]),
            "rolling_mean_12_semanas": _zero(roll[12]["mean"]),
            "rolling_std_12_semanas": _zero(roll[12]["std"]),
            "rolling_max_12_semanas": _zero(roll[12]["max"]),
            "rolling_min_12_semanas": _zero(roll[12]["min"]),
            "rolling_mean_52_semanas": _zero(roll[52]["mean"]),
            "rolling_std_52_semanas": _zero(roll[52]["std"]),
            "rolling_max_52_semanas": _zero(roll[52]["max"]),
            "coef_variacao_4": coef4,
            "preco_medio_unitario": preco_feat,
        })

        X = pd.DataFrame(cols)
        for col in cat_features:
            X[col] = pd.Categorical(X[col].values, dtype=cats[col])
        X = X[feature_names]

        pred = model.predict(X)
        if use_log:
            pred = np.expm1(pred)
        pred = np.maximum(0, np.round(pred)).astype(np.int64)

        yield wk, pred

        new_col = pred.astype(np.float32)[:, None]
        Q = np.concatenate([new_col, Q[:, :-1]], axis=1)


def forecast_from_state(state, weeks_to_forecast):
    """Week-by-week recursion using the pre-computed state (cacheable)."""
    if weeks_to_forecast <= 0 or state is None:
        return _EMPTY_FC.copy()
    n = state["Q"].shape[0]
    pdv = state["combos"]["pdv"].to_numpy()
    sku = state["combos"]["sku"].to_numpy()
    frames = []
    for wk, pred in _yield_week_predictions(state, weeks_to_forecast):
        frames.append(pd.DataFrame({
            "pdv": pdv, "sku": sku,
            "semana": np.full(n, wk, dtype=np.int64),
            "quantidade_prevista": pred,
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[["pdv", "sku", "semana", "quantidade_prevista"]]


def precompute_forecasts(state, horizon):
    """Pre-computes the full forecast of `horizon` weeks in memory (numpy).

    Returns dict {"pdv", "sku", "preds"} with preds of shape (n_combos, horizon)
    in int64 — raw material for answering /predict in milliseconds."""
    if state is None or horizon <= 0:
        return None
    n = state["Q"].shape[0]
    pdv = state["combos"]["pdv"].to_numpy()
    sku = state["combos"]["sku"].to_numpy()
    preds = np.zeros((n, horizon), dtype=np.int64)
    for wk, pred in _yield_week_predictions(state, horizon):
        preds[:, wk - 1] = pred
    return {"pdv": pdv, "sku": sku, "preds": preds, "horizon": horizon}


def fast_forecast(forecaster, df_historical, weeks_to_forecast):
    """Vectorized forecast, mathematically equivalent to the recursive
    generate_forecasts (which re-ran feature_engineering on the whole table every week).

    Caching build_forecast_state() across calls is recommended (see pyfunc)."""
    return forecast_from_state(build_forecast_state(forecaster, df_historical),
                               weeks_to_forecast)


class SalesForecasterPyfunc(PythonModel):
    def load_context(self, context):
        self.artifacts = joblib.load(context.artifacts["model"])
        self.forecaster = SalesForecasterV2()
        self.forecaster.model = self.artifacts["model"]
        self.forecaster.feature_names = self.artifacts["feature_names"]
        self.forecaster.categorical_features = self.artifacts["categorical_features"]
        self.forecaster.use_log_target = self.artifacts.get("use_log_target", False)
        self._init_caches()

    def _init_caches(self):
        self._df_hist = None
        self._fstate = None
        self._precomputed = None
        self._load_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._precompute_lock = threading.Lock()

    def _load_hist(self):
        with self._load_lock:
            if self._df_hist is None:
                df_full = self.forecaster.load_data(DATA_PATHS)
                self._df_hist = df_full[df_full["ano"] == 2022].copy()
        return self._df_hist

    def _state(self):
        with self._state_lock:
            if self._fstate is None:
                self._fstate = build_forecast_state(self.forecaster, self._load_hist())
        return self._fstate

    def ensure_precomputed(self, horizon):
        """Builds (once) the full pre-compute of `horizon` weeks when there is not
        yet enough coverage. Returns the pre-computed dict."""
        if horizon is None or horizon <= 0:
            return self._precomputed
        with self._precompute_lock:
            if self._precomputed is None or self._precomputed["horizon"] < horizon:
                self._precomputed = precompute_forecasts(self._state(), horizon)
        return self._precomputed

    @staticmethod
    def _parse_input(model_input):
        if isinstance(model_input, dict):
            return int(model_input.get("weeks_to_forecast", 5)), model_input.get("top_n")
        if hasattr(model_input, "iloc"):
            weeks = int(model_input.iloc[0].get("weeks_to_forecast", 5))
            top_n = model_input.iloc[0].get("top_n")
            top_n = int(top_n) if pd.notna(top_n) else None
            return weeks, top_n
        return 5, None

    def _predict_from_cache(self, weeks, top_n):
        """Answer from the numpy pre-compute (ms). Orders the rows
        exactly like the live path (weeks outer, combos in the original state
        order; candidate combos = top_n by total over `weeks`)."""
        pc = self._precomputed
        n = pc["preds"].shape[0]
        if n == 0 or weeks <= 0:
            return _EMPTY_FC.copy()
        preds = pc["preds"][:, :weeks]
        totals = preds.sum(axis=1)
        if top_n and n:
            k = min(int(top_n), n)
            # top-k by importance (desc rank); stable = ties keep the
            # original order (same as nlargest keep='first' in the live path)
            part = np.argsort(-totals, kind="stable")[:k]
            combo_ids = np.sort(part)
        else:
            combo_ids = np.arange(n)
        sem = np.repeat(np.arange(1, weeks + 1, dtype=np.int64), len(combo_ids))
        cid = np.tile(combo_ids, weeks)
        out = pd.DataFrame({
            "pdv": pc["pdv"][cid],
            "sku": pc["sku"][cid],
            "semana": sem,
            "quantidade_prevista": preds[cid, sem - 1],
        })
        return out[["pdv", "sku", "semana", "quantidade_prevista"]]

    def predict(self, context, model_input):
        weeks, top_n = self._parse_input(model_input)
        pc = self._precomputed
        if pc is not None and weeks > 0 and pc["horizon"] >= weeks:
            try:
                return self._predict_from_cache(weeks, top_n)
            except Exception:
                pass  # any cache failure falls back to the live path
        df_hist = self._load_hist()
        try:
            fc = forecast_from_state(self._state(), weeks_to_forecast=weeks)
        except Exception:
            fc = self.forecaster.generate_forecasts(df_hist, weeks_to_forecast=weeks)
        if top_n and not fc.empty:
            importance = fc.groupby(["pdv", "sku"], observed=True)["quantidade_prevista"].sum().reset_index()
            top = importance.nlargest(top_n, "quantidade_prevista")[["pdv", "sku"]]
            fc = fc.merge(top, on=["pdv", "sku"], how="inner")
        return fc


def get_production_model():
    """Loads the Production model from the MLflow registry (fallback: committed joblib)."""
    from . import config
    uri = config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(uri)
    versions = client.search_model_versions(f"name='{config.MLFLOW_MODEL_NAME}'")
    prod = [v for v in versions if v.current_stage == config.MLFLOW_MODEL_STAGE]
    if prod:
        v = sorted(prod, key=lambda x: x.last_updated_timestamp)[-1]
        return mlflow.pyfunc.load_model(f"models:/{config.MLFLOW_MODEL_NAME}/{v.version}")
    # fallback: build the pyfunc from the committed joblib
    joblib_path = os.path.join(SALES_DIR, "artifacts", "sales_forecaster_v2_final.joblib")
    if os.path.exists(joblib_path):
        from .model_wrapper_local import load_local_pyfunc
        return load_local_pyfunc(joblib_path)
    raise RuntimeError(f"No model '{config.MLFLOW_MODEL_NAME}' in Production and no joblib fallback.")
