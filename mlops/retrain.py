"""Retrain automatico: reusa o pipeline do campeao, registra nova versao MLflow
e promove como Production (convertendo a versao auto-registrada do log_model).

Pode ser chamado:
  - manualmente:  python -m mlops.retrain --reason manual
  - pelo monitor: monitor.py --auto consome o gatilho e chama retrain(reason='drift')
"""
import os
import sys
import time
import argparse
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from . import config, metrics_store
from .register_model import train_champion
from .model_wrapper import SalesForecasterPyfunc, DATA_PATHS, SALES_DIR

sys.path.insert(0, os.path.join(SALES_DIR, "scripts"))
from forecaster_class import SalesForecasterV2  # noqa: E402


def retrain(reason="manual"):
    metrics_store.init_db()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    # consome gatilho (se existir) para nao re-acionar
    if config.RETRAIN_TRIGGER_FILE.exists():
        config.RETRAIN_TRIGGER_FILE.unlink()

    print(f"[retrain] iniciando ({reason})...")
    t0 = time.time()
    forecaster, df_full = train_champion()
    val_mae = forecaster.performance_metrics.get("validation_mae")
    elapsed = time.time() - t0
    print(f"[retrain] concluido em {elapsed:.1f}s | val_mae={val_mae:.4f}")

    model_joblib = os.path.join(str(config.ARTIFACTS_DIR), f"champion_{int(time.time())}.joblib")
    forecaster.save_model(model_joblib)

    dummy = pd.DataFrame([{"weeks_to_forecast": 1, "top_n": 5}])
    example_out = pd.DataFrame({"semana": [1], "pdv": ["x"], "sku": ["y"], "quantidade_prevista": [0]})
    signature = infer_signature(dummy, example_out)

    with mlflow.start_run(run_name=f"retrain_{reason}_{int(time.time())}") as run:
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("use_log_target", False)
        mlflow.log_param("use_optuna", False)
        mlflow.log_param("retrain_reason", reason)
        mlflow.log_metric("validation_mae", val_mae)
        mlflow.log_metric("training_time_seconds", elapsed)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SalesForecasterPyfunc(),
            artifacts={"model": model_joblib},
            signature=signature,
            registered_model_name=config.MLFLOW_MODEL_NAME,
        )
        run_id = run.info.run_id

    client = MlflowClient(config.MLFLOW_TRACKING_URI)
    latest = client.get_latest_versions(config.MLFLOW_MODEL_NAME, stages=["None"])
    if not latest:
        raise RuntimeError(f"log_model não registrou versão de '{config.MLFLOW_MODEL_NAME}'")
    mv = sorted(latest, key=lambda x: x.last_updated_timestamp)[-1]
    client.transition_model_version_stage(
        name=config.MLFLOW_MODEL_NAME,
        version=mv.version,
        stage=config.MLFLOW_MODEL_STAGE,
        archive_existing_versions=True,
    )
    metrics_store.log_retrain(reason, run_id, val_mae, "ok")
    print(f"[retrain] v{mv.version} -> Production | run_id={run_id} | val_mae={val_mae:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", default="manual")
    args = parser.parse_args()
    retrain(args.reason)