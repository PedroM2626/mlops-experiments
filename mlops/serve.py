"""API FastAPI de serving do sales-forecast v2.2 com MLflow registry.

Endpoints:
  GET  /health         -> status do servico + versao do modelo em Producao
  POST /predict        -> gera forecast (semanas) e loga latencia/custo
  GET  /metrics?window=3600 -> resumo agregado da janela (chegadas, custo, latencia, drift)
  GET  /recent         -> ultimas predicoes + drift checks (feed do dashboard)
  GET  /drift          -> ultimo estado de drift + gatilho de retrain
  GET  /dashboard      -> dashboard HTML vivo (polling AJAX)

Uso:
  python -m mlops.serve            # uvicorn embutido
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
    """Carrega artefato Production do MLflow e cacheia dados historicos."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(config.MLFLOW_TRACKING_URI)
    versions = client.search_model_versions(f"name='{config.MLFLOW_MODEL_NAME}'")
    prod = [v for v in versions if v.current_stage == config.MLFLOW_MODEL_STAGE]
    if not prod:
        # fallback: joblib committed
        p = os.path.join(SALES_DIR, "artifacts", "sales_forecaster_v2_final.joblib")
        if not os.path.exists(p):
            raise RuntimeError("Sem modelo Production e sem fallback joblib.")
        from .model_wrapper_local import load_local_pyfunc
        return load_local_pyfunc(p), "fallback_joblib", None
    v = sorted(prod, key=lambda x: x.last_updated_timestamp)[-1]
    # resolve pelo registry (estilo-independente: v.source pode ser caminho
    # de artefato OU locator 'models:/...' dependendo da versao do MLflow)
    model = mlflow.pyfunc.load_model(f"models:/{config.MLFLOW_MODEL_NAME}/{v.version}")
    return model, f"v{v.version}", v.run_id


app = FastAPI(title="MLOps Sales-Forecast v2.2", version="1.0")
_predictor, _model_version, _model_run_id = None, None, None
_forecaster_cache = None  # para reusar dados historicos
_precompute_thread = None
_precompute_done = False


def _ensure_predictor():
    global _predictor, _model_version, _model_run_id, _forecaster_cache
    if _predictor is None:
        _predictor, _model_version, _model_run_id = _load_production_predictor()
    if _forecaster_cache is None:
        # carrega dados brutos 2022 uma vez para alimentar generate_forecasts
        fc = SalesForecasterV2()
        df_full = fc.load_data(DATA_PATHS)
        _forecaster_cache = df_full[df_full["ano"] == 2022].copy()
    return _predictor


def _python_model():
    """Instancia do pyfunc por tras do PyFuncModel do MLflow (ou o proprio
    pyfunc local no fallback joblib, quando nao ha modelo Production)."""
    impl = getattr(_predictor, "_model_impl", None)
    if impl is not None:
        return getattr(impl, "python_model", _predictor)
    return _predictor


def _kick_precompute():
    """Pre-computa o forecast completo em background; /predict nao bloqueia."""
    global _precompute_thread, _precompute_done
    if _predictor is None or (_precompute_thread and _precompute_thread.is_alive()):
        return
    def _run():
        global _precompute_done
        try:
            _python_model().ensure_precomputed(config.PRECOMPUTE_HORIZON)
            _precompute_done = True
        except Exception as e:  # noqa: BLE001
            print(f"[precompute] erro: {e}", flush=True)
    _precompute_thread = threading.Thread(target=_run, name="precompute", daemon=True)
    _precompute_thread.start()


@app.on_event("startup")
def _startup():
    metrics_store.init_db()
    _ensure_predictor()
    _kick_precompute()


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
    """Forca (bloqueante) o rebuild do forecast pre-computado."""
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
    return {"triggered": False, "message": "sem gatilho ativo"}


@app.get("/dashboard")
def dashboard():
    html = os.path.join(os.path.dirname(__file__), "dashboard.html")
    return __import__("fastapi").responses.HTMLResponse(open(html, encoding="utf-8").read())


# helper p/ detectar o fallback local (pyfunc)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
