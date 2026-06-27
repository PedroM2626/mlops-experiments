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
    Funcao principal para orquestrar o treinamento do modelo.
    """
    logging.info("Iniciando o Pipeline de Treinamento V2.2.")

    # Define os caminhos para os arquivos de dados e saida do modelo
    file_paths = {
        'vendas': os.path.join(data_path, 'raw/fato_vendas.parquet'),
        'pdvs': os.path.join(data_path, 'raw/dim_pdvs.parquet'),
        'produtos': os.path.join(data_path, 'raw/dim_produtos.parquet')
    }
    model_output_path = os.path.join(artifacts_path, 'sales_forecaster_v2_final.joblib')
    fi_plot_path = os.path.join(artifacts_path, 'feature_importance.png')

    # Configurando tracking do MLflow
    mlflow.set_experiment("Sales_Forecaster_Hackathon")

    # Instancia e executa o pipeline
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

            # Gerar e logar grafico de feature importance
            forecaster.plot_feature_importance(fi_plot_path)
            mlflow.log_artifact(fi_plot_path, artifact_path="plots")

            # Salvar e logar modelo
            forecaster.save_model(path=model_output_path)
            mlflow.log_artifact(model_output_path, artifact_path="model")

            logging.info(f"Treinamento completo em {elapsed:.1f}s. MAE: {forecaster.performance_metrics.get('validation_mae', 'N/A')}")

        except Exception as e:
            logging.error(f"O pipeline de treinamento falhou com o erro: {e}")
            raise e

    logging.info("Pipeline de Treinamento finalizado com sucesso!")

if __name__ == "__main__":
    # Configura os argumentos que o script pode receber via linha de comando
    parser = argparse.ArgumentParser(description="Treina o modelo de previsao de vendas.")
    parser.add_argument("--data_path", type=str, default="data", help="Caminho para a pasta 'data'.")
    parser.add_argument("--artifacts_path", type=str, default="artifacts", help="Caminho para salvar o modelo treinado.")
    parser.add_argument("--n_trials", type=int, default=30, help="Numero de trials para a otimizacao com Optuna.")

    args = parser.parse_args()

    main(args.data_path, args.artifacts_path, args.n_trials)