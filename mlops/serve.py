"""FastAPI serving API for sales-forecast v2.2 with an MLflow registry.

Endpoints:
  GET  /health         -> service status + version of the model in Production
  POST /predict        -> generates the forecast (weeks) and logs latency/cost
  GET  /metrics?window=3600 -> aggregate summary of the window (arrivals, cost, latency, drift)
  GET  /recent         -> latest predictions + drift checks (dashboard feed)
  GET  /drift          -> last drift state + retrain trigger
  GET  /dashboard      -> live HTML dashboard (AJAX polling)

Usage:
  python -m mlops.serve            # embedded uvicorn
  uvicorn mlops.serve:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import sys
import time
import json
import threading
import joblib
import pandas as pd
import mlflow
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

from . import config, metrics_store
from .model_wrapper import SalesForecasterPyfunc, DATA_PATHS, SALES_DIR

sys.path.insert(0, os.path.join(SALES_DIR, "scripts"))
from forecaster_class import SalesForecasterV2  # noqa: E402


class PredictRequest(BaseModel):
    weeks_to_forecast: int = 5
    top_n: Optional[int] = 50


def _load_production_predictor():
    """Loads the production artifact from MLflow and caches the historical data.

    Primary: alias `models:/<name>@production` (MLflow 3.x). Fallback:
    the legacy Production stage and, as a last resort, the committed joblib.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(config.MLFLOW_TRACKING_URI)
    try:
        v = client.get_model_version_by_alias(
            name=config.MLFLOW_MODEL_NAME, alias=config.MLFLOW_MODEL_ALIAS)
        model = mlflow.pyfunc.load_model(
            f"models:/{config.MLFLOW_MODEL_NAME}@{config.MLFLOW_MODEL_ALIAS}")
        return model, f"v{v.version}@{config.MLFLOW_MODEL_ALIAS}", v.run_id
    except Exception:
        pass
    from .registry import resolve_production_version
    try:
        v = resolve_production_version(client, config.MLFLOW_MODEL_NAME,
                                       config.MLFLOW_MODEL_ALIAS,
                                       config.MLFLOW_MODEL_STAGE)
    except RuntimeError:
        # fallback: joblib committed
        p = os.path.join(SALES_DIR, "artifacts", "sales_forecaster_v2_final.joblib")
        if not os.path.exists(p):
            raise RuntimeError("No Production model and no joblib fallback.")
        from .model_wrapper_local import load_local_pyfunc
        return load_local_pyfunc(p), "fallback_joblib", None
    # resolve via the registry (style-independent: v.source can be an artifact
    # path OR a 'models:/...' locator depending on the MLflow version)
    model = mlflow.pyfunc.load_model(f"models:/{config.MLFLOW_MODEL_NAME}/{v.version}")
    return model, f"v{v.version}", v.run_id


app = FastAPI(title="MLOps Sales-Forecast v2.2", version="2.0")
_predictor, _model_version, _model_run_id = None, None, None
_forecaster_cache = None  # reuse the historical data
_precompute_thread = None
_precompute_done = False


def _ensure_predictor():
    global _predictor, _model_version, _model_run_id, _forecaster_cache
    if _predictor is None:
        _predictor, _model_version, _model_run_id = _load_production_predictor()
    if _forecaster_cache is None:
        # loads the raw 2022 data once to feed generate_forecasts
        fc = SalesForecasterV2()
        df_full = fc.load_data(DATA_PATHS)
        _forecaster_cache = df_full[df_full["ano"] == 2022].copy()
    return _predictor


def _python_model():
    """The pyfunc instance behind the MLflow PyFuncModel (or the local
    pyfunc itself in the joblib fallback, when there is no Production model)."""
    impl = getattr(_predictor, "_model_impl", None)
    if impl is not None:
        return getattr(impl, "python_model", _predictor)
    return _predictor


def _kick_precompute():
    """Pre-computes the full forecast in the background; /predict does not block."""
    global _precompute_thread, _precompute_done
    if _predictor is None or (_precompute_thread and _precompute_thread.is_alive()):
        return
    def _run():
        global _precompute_done
        try:
            _python_model().ensure_precomputed(config.PRECOMPUTE_HORIZON)
            _precompute_done = True
        except Exception as e:  # noqa: BLE001
            print(f"[precompute] error: {e}", flush=True)
    _precompute_thread = threading.Thread(target=_run, name="precompute", daemon=True)
    _precompute_thread.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern startup/shutdown (replaces the deprecated `@app.on_event`)."""
    metrics_store.init_db()
    _ensure_predictor()
    _kick_precompute()
    yield


app.router.lifespan_context = lifespan


@app.get("/health")
def health():
    return {"status": "ok", "model": config.MLFLOW_MODEL_NAME,
            "version": _model_version, "run_id": _model_run_id}


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.time()
    try:
        _ensure_predictor()
        model = _predictor
        inp = pd.DataFrame([{"weeks_to_forecast": req.weeks_to_forecast, "top_n": req.top_n}])
        fc = model.predict(inp)
        latency_ms = (time.time() - t0) * 1000
        n = int(len(fc)) if fc is not None and hasattr(fc, "__len__") else 0
        cost = metrics_store.log_prediction(req.weeks_to_forecast, n, latency_ms)
        rows = []
        if n:
            for _, r in fc.head(min(req.top_n or 20, n)).iterrows():
                rows.append({c: (r[c] if not pd.isna(r[c]) else None) for c in fc.columns})
        return {"weeks_to_forecast": req.weeks_to_forecast, "n_predictions": n,
                "latency_ms": round(latency_ms, 2), "cost_usd": round(cost, 6),
                "model_version": _model_version, "top_predictions": rows}
    except Exception as e:
        metrics_store.log_prediction(req.weeks_to_forecast, 0, (time.time() - t0) * 1000, status="error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/precompute")
def precompute_status():
    pc = _python_model()._precomputed if _predictor else None
    ready = _precompute_done and pc is not None
    return {
        "ready": ready,
        "building": bool(_precompute_thread and _precompute_thread.is_alive()),
        "configured_horizon": config.PRECOMPUTE_HORIZON,
        "cached_horizon": pc["horizon"] if pc else None,
        "combos": int(pc["preds"].shape[0]) if pc else None,
        "forecast_path": "cache" if ready else "live",
    }


@app.post("/precompute")
def precompute_refresh():
    """Forces a (blocking) rebuild of the pre-computed forecast."""
    _kick_precompute()
    if _precompute_thread and _precompute_thread.is_alive():
        _precompute_thread.join(timeout=config.PRECOMPUTE_HORIZON * 30)
    return precompute_status()


@app.get("/metrics")
def metrics(window: int = Query(3600, ge=60)):
    s = metrics_store.get_summary(window)
    s["model_version"] = _model_version
    s["model_runid"] = _model_run_id
    return s


@app.get("/recent")
def recent(limit: int = Query(20, le=100)):
    return {"predictions": metrics_store.recent_predictions(limit),
            "drift_checks": metrics_store.recent_drift(limit)}


@app.get("/drift")
def drift_status():
    p = config.RETRAIN_TRIGGER_FILE
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"triggered": False, "message": "no active trigger"}


@app.get("/dashboard")
def dashboard():
    html = os.path.join(os.path.dirname(__file__), "dashboard.html")
    return __import__("fastapi").responses.HTMLResponse(open(html, encoding="utf-8").read())


# helper to detect the local fallback (pyfunc)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
