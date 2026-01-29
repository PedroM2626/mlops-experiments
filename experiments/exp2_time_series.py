import os
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import dagshub
from dotenv import load_dotenv
import numpy as np

# Configuração DagsHub
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")

dagshub.init(repo_owner=repo_owner, repo_name=repo_name)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def run_experiment():
    mlflow.set_experiment("Temperature_Forecasting")
    
    with mlflow.start_run():
        print("🚀 Iniciando Experimento 2: Temperature Forecasting")
        
        # Carregar dados
        data_path = r"c:\Users\pedro\Downloads\experiments\experiments\datasets\daily-minimum-temperatures-in-me.csv"
        df = pd.read_csv(data_path)
        
        # Limpeza básica (remover erros de parsing se houver)
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna()
        
        # Dividir em treino e teste (últimos 365 dias para teste)
        train = df.iloc[:-365]
        test = df.iloc[-365:]
        
        # Modelo Prophet
        model = Prophet(daily_seasonality=True)
        model.fit(train)
        
        # Previsão
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        
        # Métricas
        y_true = test['y'].values
        y_pred = forecast.iloc[-365:]['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"MAE: {mae}, RMSE: {rmse}")
        
        # Logar métricas
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        
        # Salvar gráficos como artefatos
        fig1 = model.plot(forecast)
        plt.title("Forecast de Temperatura")
        plt.savefig("temp_forecast.png")
        
        fig2 = model.plot_components(forecast)
        plt.savefig("temp_components.png")
        
        mlflow.log_artifact("temp_forecast.png")
        mlflow.log_artifact("temp_components.png")
        
        # Salvar modelo (como pkl)
        import joblib
        joblib.dump(model, "prophet_model.pkl")
        mlflow.log_artifact("prophet_model.pkl")
        
        print("✅ Experimento 2 concluído e logado no DagsHub!")

if __name__ == "__main__":
    run_experiment()
