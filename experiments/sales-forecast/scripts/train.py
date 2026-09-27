import argparse
import logging
import os
import time
from forecaster_class import SalesForecasterV2

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import mlflow


def main(data_path: str, artifacts_path: str, n_trials: int):
    """
    Main function to orchestrate model training.
    """
    logging.info("Starting the Training Pipeline V2.2.")

    # Define the paths for the data files and model output
    file_paths = {
        'vendas': os.path.join(data_path, 'raw/fato_vendas.parquet'),
        'pdvs': os.path.join(data_path, 'raw/dim_pdvs.parquet'),
        'produtos': os.path.join(data_path, 'raw/dim_produtos.parquet')
    }
    model_output_path = os.path.join(artifacts_path, 'sales_forecaster_v2_final.joblib')
    fi_plot_path = os.path.join(artifacts_path, 'feature_importance.png')

    # Configure MLflow tracking
    mlflow.set_experiment("Sales_Forecaster_Hackathon")

    # Instantiate and run the pipeline
    forecaster = SalesForecasterV2()

    with mlflow.start_run(run_name="V2.2_training"):
        try:
            start_time = time.time()

            # Log params
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_param("objective", "regression_l1")
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("validation_split_week", 48)
            mlflow.log_param("use_optuna", True)
            mlflow.log_param("version", "V2.2")

            df_full_data = forecaster.load_data(file_paths)

            forecaster.train(
                df_full_data,
                validation_split_week=48,
                use_optuna=True,
                n_trials=n_trials
            )

            elapsed = time.time() - start_time

            # Log metrics
            for metric_name, metric_val in forecaster.performance_metrics.items():
                mlflow.log_metric(metric_name, metric_val)
            mlflow.log_metric("training_time_seconds", elapsed)

            # Log best params individually
            for param_name, param_val in forecaster.best_params.items():
                mlflow.log_param(f"best_{param_name}", param_val)

            # Generate and log the feature importance plot
            forecaster.plot_feature_importance(fi_plot_path)
            mlflow.log_artifact(fi_plot_path, artifact_path="plots")

            # Save and log the model
            forecaster.save_model(path=model_output_path)
            mlflow.log_artifact(model_output_path, artifact_path="model")

            logging.info(f"Training completed in {elapsed:.1f}s. MAE: {forecaster.performance_metrics.get('validation_mae', 'N/A')}")

        except Exception as e:
            logging.error(f"The training pipeline failed with error: {e}")
            raise e

    logging.info("Training Pipeline finished successfully!")

if __name__ == "__main__":
    # Configure the arguments the script can receive via the command line
    parser = argparse.ArgumentParser(description="Train the sales forecasting model.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the 'data' folder.")
    parser.add_argument("--artifacts_path", type=str, default="artifacts", help="Path to save the trained model.")
    parser.add_argument("--n_trials", type=int, default=30, help="Number of trials for the Optuna optimization.")

    args = parser.parse_args()

    main(args.data_path, args.artifacts_path, args.n_trials)