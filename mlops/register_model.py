"""Registers the champion model V2.2 (use_log_target=False) in the MLflow registry.

Retrains without Optuna (fast), logs it, creates the model 'sales_forecaster_v22' in
the local MLflow Tracking Server, moves the version to Production and saves the
reference dataset (training features) for the drift monitor.

Usage:
    python -m mlops.register_model
"""
import os
import sys
import time
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from . import config
from .model_wrapper import SalesForecasterPyfunc, DATA_PATHS, SALES_DIR

sys.path.insert(0, os.path.join(SALES_DIR, "scripts"))
from forecaster_class import SalesForecasterV2  # noqa: E402


def train_champion():
    forecaster = SalesForecasterV2()
    forecaster.use_log_target = False
    df_full = forecaster.load_data(DATA_PATHS)
    forecaster.train(df_full, validation_split_week=48, use_optuna=False, n_trials=1)
    return forecaster, df_full


def save_reference(forecaster, df_full):
    df_2022 = df_full[df_full["ano"] == 2022].copy()
    feats = forecaster.feature_engineering(df_2022)
    X, _ = forecaster._prepare_data_for_model(feats)
    X.to_parquet(str(config.REFERENCE_PATH), index=False)
    print(f"[register] reference saved: {config.REFERENCE_PATH} ({len(X)} rows)")
    return X


def main():
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    print("[register] retraining the champion (use_log_target=False, no Optuna)...")
    t0 = time.time()
    forecaster, df_full = train_champion()
    elapsed = time.time() - t0
    val_mae = forecaster.performance_metrics.get("validation_mae")
    print(f"[register] training finished in {elapsed:.1f}s | val_mae = {val_mae:.4f}")

    ref_X = save_reference(forecaster, df_full)

    model_joblib = os.path.join(str(config.ARTIFACTS_DIR), "champion.joblib")
    forecaster.save_model(model_joblib)

    # example forecast for infer_signature
    from .model_wrapper import SalesForecasterPyfunc as PM
    dummy = pd.DataFrame([{"weeks_to_forecast": 1, "top_n": 5}])
    example_out = pd.DataFrame({"semana": [1], "pdv": ["x"], "sku": ["y"], "quantidade_prevista": [0]})
    signature = infer_signature(dummy, example_out)

    with mlflow.start_run(run_name=f"v22_champion_{int(time.time())}") as run:
        mlflow.log_param("model_type", "LightGBM")
        mlflow.log_param("use_log_target", False)
        mlflow.log_param("use_optuna", False)
        mlflow.log_param("n_features", len(forecaster.feature_names))
        for k, v in forecaster.performance_metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_metric("training_time_seconds", elapsed)

        # log the model as pyfunc (with the joblib artifact)
        import shutil
        model_artifact_dir = "model"
        mlflow.pyfunc.log_model(
            artifact_path=model_artifact_dir,
            python_model=SalesForecasterPyfunc(),
            artifacts={"model": model_joblib},
            signature=signature,
            registered_model_name=config.MLFLOW_MODEL_NAME,
        )
        run_id = run.info.run_id

    # promote using the version AUTO-registered by log_model
    # (source points at the real run's artifact; a manual create_model_version
    #  with mlflow.get_artifact_uri() produced a broken source in mlruns/0).
    # The alias is the primary one (MLflow 3.x); stage kept for compatibility.
    from .registry import latest_unstaged_version, promote_version
    client = MlflowClient(config.MLFLOW_TRACKING_URI)
    mv = latest_unstaged_version(client, config.MLFLOW_MODEL_NAME)
    res = promote_version(client, config.MLFLOW_MODEL_NAME, mv.version,
                          config.MLFLOW_MODEL_ALIAS, config.MLFLOW_MODEL_STAGE)
    print(f"[register] model '{config.MLFLOW_MODEL_NAME}' v{mv.version} -> "
          f"@{res['alias']} (stage_ok={res['stage_ok']})")
    print(f"[register] source={mv.source}")
    print(f"[register] run_id={run_id} | val_mae={val_mae:.4f}")


if __name__ == "__main__":
    main()
