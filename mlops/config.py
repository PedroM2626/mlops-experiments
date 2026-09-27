"""Configuration of the MLOps system (serving + drift + retrain + dashboard)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SALES_DIR = REPO_ROOT / "experiments" / "sales-forecast"
DATA_DIR = SALES_DIR / "data"
ARTIFACTS_DIR = REPO_ROOT / "experiments" / "artifacts" / "mlops_sales"
DB_PATH = ARTIFACTS_DIR / "mlops_prod.db"
REFERENCE_PATH = ARTIFACTS_DIR / "reference_features.parquet"
CURRENT_PATH = ARTIFACTS_DIR / "current_features.parquet"

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    (REPO_ROOT / "experiments" / "mlops_tracking.db").as_uri().replace(
        "file:///", "sqlite:///"))
MLFLOW_EXPERIMENT = "sales_forecast_v22_prod"
MLFLOW_MODEL_NAME = "sales_forecaster_v22"
MLFLOW_MODEL_STAGE = "Production"  # legacy (fallback); the primary one is the alias below
MLFLOW_MODEL_ALIAS = "production"  # `models:/<name>@production` (MLflow 3.x)

API_HOST = "0.0.0.0"
API_PORT = 8000

# horizon (weeks) of the forecast pre-computed in memory; /predict below
# that becomes a simple lookup (warm in milliseconds). Above that,
# it falls back to the "live" forecast (still fast, ~7s/2 weeks).
PRECOMPUTE_HORIZON = 12

COST_PER_1000_PREDICTIONS = 0.0009

DRIFT_PSI_THRESHOLD = 0.25
DRIFT_SHARE_THRESHOLD = 0.40
DRIFT_CHECK_INTERVAL_SECONDS = 60
RETRAIN_TRIGGER_FILE = ARTIFACTS_DIR / "retrain_trigger.json"

MONITOR_INTERVAL_SECONDS = 30
# cooldown between automatic retrains (avoids a feedback loop under persistent drift)
RETRAIN_COOLDOWN_SECONDS = 1800

for _d in (ARTIFACTS_DIR,):
    _d.mkdir(parents=True, exist_ok=True)
