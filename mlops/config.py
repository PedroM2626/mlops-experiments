"""Configuracoes do sistema de MLOps (serving + drift + retrain + dashboard)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SALES_DIR = REPO_ROOT / "experiments" / "sales-forecast"
DATA_DIR = SALES_DIR / "data"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "artifacts" / "mlops_sales"
DB_PATH = ARTIFACTS_DIR / "mlops_prod.db"
REFERENCE_PATH = ARTIFACTS_DIR / "reference_features.parquet"
CURRENT_PATH = ARTIFACTS_DIR / "current_features.parquet"

MLFLOW_TRACKING_URI = (REPO_ROOT / "experiments" / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "sales_forecast_v22_prod"
MLFLOW_MODEL_NAME = "sales_forecaster_v22"
MLFLOW_MODEL_STAGE = "Production"

API_HOST = "0.0.0.0"
API_PORT = 8000

# horizonte (semanas) do forecast pre-computado em memoria; /predict abaixo
# disso passa a ser um simples lookup (warm em milissegundos). Acima disso,
# cai no forecast "live" (ainda rapido, ~7s/2 semanas).
PRECOMPUTE_HORIZON = 12

COST_PER_1000_PREDICTIONS = 0.0009

DRIFT_PSI_THRESHOLD = 0.25
DRIFT_SHARE_THRESHOLD = 0.40
DRIFT_CHECK_INTERVAL_SECONDS = 60
RETRAIN_TRIGGER_FILE = ARTIFACTS_DIR / "retrain_trigger.json"

MONITOR_INTERVAL_SECONDS = 30
# cooldown entre retrains automaticos (evita feedback loop em drift persistente)
RETRAIN_COOLDOWN_SECONDS = 1800

for _d in (ARTIFACTS_DIR,):
    _d.mkdir(parents=True, exist_ok=True)
