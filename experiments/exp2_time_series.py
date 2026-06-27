import os
import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.prophet
from mlflow.models.signature import infer_signature
import dagshub
import optuna
from dotenv import load_dotenv
import numpy as np
import warnings

# Suprimir avisos excessivos do Prophet/CmdStanPy durante a execucao
warnings.filterwarnings("ignore")

from run_context import create_run_context, log_reproducibility, first_existing_path

# Configuracao DagsHub
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")

dagshub.init(repo_owner=repo_owner, repo_name=repo_name)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
BASE_DIR = Path(__file__).resolve().parent

def objective(trial, df_train):
    """Funcao objetivo para otimizar os hiperparametros do Prophet com Optuna."""
    # Sugerir hiperparametros
    params = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True),
        "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
        "daily_seasonality": True,
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "interval_width": 0.95
    }
    
    # Suprimir logs para o trial
    import logging
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    model = Prophet(**params)
    model.fit(df_train)
    
    # Executar cross-validation simplificada para economizar tempo durante os trials
    # initial 3 anos, period 1 ano, prevendo 90 dias
    try:
        df_cv = cross_validation(
            model, 
            initial='1095 days', 
            period='365 days', 
            horizon='90 days', 
            parallel='processes' # acelerar com processos paralelos
        )
        df_p = performance_metrics(df_cv)
        return df_p['mae'].mean()
    except Exception as e:
        # Se algo falhar na otimizacao (ex: instabilidade numerica)
        return float('inf')

def run_experiment():
    mlflow.set_experiment("Temperature_Forecasting")
    
    with mlflow.start_run() as run:
        context = create_run_context(BASE_DIR, "temperature_forecasting")
        log_reproducibility(mlflow, context, SEED)
        print("Iniciando Experimento 2: Temperature Forecasting V2 (Optuna Optimized)")
        
        # Carregar dados usando o helper de caminhos relativos
        data_path = first_existing_path([
            BASE_DIR / "datasets" / "daily-minimum-temperatures-in-me.csv",
            BASE_DIR.parent / "datasets" / "daily-minimum-temperatures-in-me.csv",
        ])
        df = pd.read_csv(data_path)
        
        # Limpeza basica
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna()
        
        # Dividir em treino e teste (ultimos 365 dias para teste hold-out)
        train = df.iloc[:-365]
        test = df.iloc[-365:]
        
        # Logar metadados essenciais do dataset
        mlflow.log_param("dataset_start_date", df['ds'].min().strftime('%Y-%m-%d'))
        mlflow.log_param("dataset_end_date", df['ds'].max().strftime('%Y-%m-%d'))
        mlflow.log_param("train_size", len(train))
        mlflow.log_param("test_size", len(test))
        
        # --- BUSCA BAYESIANA (OPTUNA) ---
        print("Iniciando Otimizacao Bayesiana (Optuna) com 10 trials...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda trial: objective(trial, train), n_trials=10, n_jobs=1)
        
        print(f"Melhor Trial (MAE): {study.best_value:.4f}")
        print("Melhores Parametros:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Juntar hiperparametros otimizados com os fixos
        best_params = study.best_params
        best_params.update({
            "daily_seasonality": True,
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "interval_width": 0.95
        })
        
        mlflow.log_params(best_params)
        mlflow.log_metric("optuna_best_cv_mae", study.best_value)
        
        # --- TREINAMENTO DO MODELO FINAL ---
        print("Treinando Modelo Prophet com Melhores Parametros...")
        final_model = Prophet(**best_params)
        final_model.fit(train)
        
        # Previsao na janela de hold-out
        future = final_model.make_future_dataframe(periods=365)
        forecast = final_model.predict(future)
        
        # Calcular e logar metricas do hold-out (validação final externa)
        y_true = test['y'].values
        y_pred = forecast.iloc[-365:]['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"Metricas Finais Hold-out -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        mlflow.log_metric("holdout_mae", mae)
        mlflow.log_metric("holdout_rmse", rmse)
        
        # Salvar graficos no diretorio versionado de artefatos
        forecast_path = context.artifact_dir / "temp_forecast.png"
        components_path = context.artifact_dir / "temp_components.png"
        
        fig1 = final_model.plot(forecast)
        plt.title("Forecast de Temperatura (Prophet + Optuna)")
        plt.savefig(forecast_path)
        plt.close(fig1)
        
        fig2 = final_model.plot_components(forecast)
        plt.savefig(components_path)
        plt.close(fig2)
        
        mlflow.log_artifact(str(forecast_path))
        mlflow.log_artifact(str(components_path))
        
        # Salvar modelo nativo do Prophet no MLflow com a assinatura exata
        signature = infer_signature(train[['ds']], forecast)
        
        mlflow.prophet.log_model(
            final_model,
            artifact_path="prophet-model",
            signature=signature
        )
        
        print("Experimento 2 V2 concluido e artefatos salvos com MLOps completo.")

if __name__ == "__main__":
    run_experiment()
