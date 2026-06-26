import logging
import os
import sys
import shutil
from typing import Dict, List, Tuple
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import optuna
import mlflow
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
import dagshub
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class SalesForecasterV2:
    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Configuração DagsHub/MLflow
        self.repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
        self.repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")
        
        try:
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name)
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        except Exception as e:
            logging.warning(f"Falha ao inicializar DagsHub/MLflow: {e}")

    def load_data(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        logging.info("Iniciando o carregamento dos dados.")
        # Verifica se todos os arquivos existem
        all_exists = True
        for key, path in file_paths.items():
            if not os.path.exists(path):
                logging.warning(f"Arquivo {path} não encontrado.")
                all_exists = False
        
        if not all_exists:
            logging.warning("Alguns arquivos não foram encontrados. Gerando dados sintéticos para teste.")
            return self._generate_synthetic_data()

        try:
            df_vendas = pd.read_parquet(file_paths['vendas'])
            df_pdvs = pd.read_parquet(file_paths['pdvs'])
            df_produtos = pd.read_parquet(file_paths['produtos'])
            
            df_merged = pd.merge(df_vendas, df_pdvs, left_on='internal_store_id', right_on='pdv', how='inner')
            df_merged = pd.merge(df_merged, df_produtos, left_on='internal_product_id', right_on='produto', how='inner')
            df_merged['transaction_date'] = pd.to_datetime(df_merged['transaction_date'])
            df_merged['ano'] = df_merged['transaction_date'].dt.isocalendar().year
            df_merged['semana'] = df_merged['transaction_date'].dt.isocalendar().week
            
            agg_vendas = df_merged.groupby(['ano', 'semana', 'pdv', 'produto']).agg(total_quantity=('quantity', 'sum')).reset_index()
            return agg_vendas.rename(columns={'produto': 'sku', 'total_quantity': 'quantidade'})
        except Exception as e:
            logging.error(f"Erro no carregamento: {e}")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Gera dados sintéticos para possibilitar o teste do pipeline sem os arquivos gigantes do Drive."""
        logging.info("Gerando 10.000 linhas de dados sintéticos...")
        np.random.seed(42)
        data = {
            'ano': [2022] * 10000,
            'semana': np.random.randint(1, 53, 10000),
            'pdv': np.random.randint(1, 50, 10000),
            'sku': np.random.randint(1, 100, 10000),
            'quantidade': np.random.randint(0, 100, 10000)
        }
        return pd.DataFrame(data)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df_featured = df.copy()
        df_featured.sort_values(['pdv', 'sku', 'ano', 'semana'], inplace=True)
        
        # Otimização: Uso de transformações vetorizadas
        df_featured['trimestre'] = (df_featured['semana'] - 1) // 13 + 1
        df_featured['seno_semana'] = np.sin(2 * np.pi * df_featured['semana'] / 52)
        df_featured['cosseno_semana'] = np.cos(2 * np.pi * df_featured['semana'] / 52)
        
        # Otimização: Agrupar shifts e rollings para evitar múltiplas operações de groupby caras
        grouped = df_featured.groupby(['pdv', 'sku'])['quantidade']
        
        lags = [1, 2, 3, 4, 12, 52] 
        for lag in lags:
            df_featured[f'lag_{lag}_semanas'] = grouped.shift(lag)
            
        windows = [4, 12, 52]
        for window in windows:
            rolled = grouped.shift(1).rolling(window=window, min_periods=1)
            df_featured[f'rolling_mean_{window}_semanas'] = rolled.mean()
            df_featured[f'rolling_std_{window}_semanas'] = rolled.std()
            df_featured[f'rolling_max_{window}_semanas'] = rolled.max()
        
        df_featured.fillna(0, inplace=True)
        return df_featured

    def _prepare_data_for_model(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.categorical_features = ['pdv', 'sku']
        df_model = df.copy()
        for col in self.categorical_features:
            df_model[col] = df_model[col].astype('category')
            
        self.feature_names = [c for c in df_model.columns if c not in ['quantidade', 'ano']]
        return df_model[self.feature_names], df_model['quantidade']

    def train(self, df: pd.DataFrame, validation_split_week: int = 48, n_trials: int = 20):
        mlflow.set_experiment("Hackathon_Forecast_2025_Optimized")
        
        with mlflow.start_run() as run:
            logging.info("🚀 Iniciando Treinamento Otimizado com MLflow...")
            
            df_featured = self.feature_engineering(df)
            train_set = df_featured[df_featured['semana'] < validation_split_week]
            val_set = df_featured[df_featured['semana'] >= validation_split_week]
            
            X_train, y_train = self._prepare_data_for_model(train_set)
            X_val, y_val = self._prepare_data_for_model(val_set)

            def objective(trial):
                params = {
                    'objective': 'regression_l1',
                    'metric': 'mae',
                    'verbosity': -1,
                    'boosting_type': 'gbdt',
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 128),
                    'n_estimators': 200, 
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                          callbacks=[lgb.early_stopping(5)])
                return mean_absolute_error(y_val, model.predict(X_val))

            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            
            # Melhor modelo
            self.model = lgb.LGBMRegressor(**study.best_params, n_estimators=500)
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                          callbacks=[lgb.early_stopping(10)])
            
            predictions = self.model.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)
            self.performance_metrics['validation_mae'] = mae
            
            # LOGGING MLFLOW
            mlflow.log_params(study.best_params)
            mlflow.log_metric("val_mae", mae)
            
            # Signature e Input Example
            signature = infer_signature(X_val, predictions)
            input_example = X_val.iloc[:5]
            
            # Salvar e Logar Modelo Completo (MLflow Format)
            # Salvando temporariamente para logar como artefato completo
            temp_model_dir = f"artifacts/run_{run.info.run_id}"
            os.makedirs(temp_model_dir, exist_ok=True)
            
            mlflow.lightgbm.log_model(
                lgb_model=self.model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name="SalesForecaster_LGBM"
            )
            
            # Salva também o joblib para compatibilidade legado, mas dentro da pasta da run
            legacy_path = os.path.join(temp_model_dir, "model.joblib")
            self.save_model(legacy_path)
            mlflow.log_artifact(legacy_path, artifact_path="legacy_model")
            
            logging.info(f"✅ Treinamento finalizado. MAE: {mae:.4f}")
            logging.info(f"📦 Modelo e metadados salvos na Run ID: {run.info.run_id}")
            
            # Cleanup local artifacts after successful logging
            try:
                if os.path.exists("artifacts"):
                    shutil.rmtree("artifacts")
                    logging.info("🧹 Artefatos locais limpos com sucesso (armazenados no DagsHub).")
            except Exception as e:
                logging.warning(f"⚠️ Erro ao limpar artefatos locais: {e}")

    def generate_forecasts(self, df_historical: pd.DataFrame, weeks_to_forecast: int) -> pd.DataFrame:
        if not self.model: raise RuntimeError("O modelo não foi treinado.")
        forecast_df = df_historical.copy()
        all_forecasts = []
        for i in range(1, weeks_to_forecast + 1):
            current_week = i
            features_base = self.feature_engineering(forecast_df)
            latest_entries = features_base.sort_values(by=['ano', 'semana']).drop_duplicates(subset=['pdv', 'sku'], keep='last')
            if latest_entries.empty: continue
            X_pred = latest_entries.copy()
            X_pred['semana'] = current_week
            X_pred['ano'] = 2023
            
            # Ajuste de categorias para o modelo
            for col in self.categorical_features:
                X_pred[col] = X_pred[col].astype('category')
            
            predictions = self.model.predict(X_pred[self.feature_names])
            predictions = np.maximum(0, np.round(predictions)).astype(int)
            
            week_forecast = X_pred[['pdv', 'sku']].copy()
            week_forecast['semana'] = current_week
            week_forecast['quantidade_prevista'] = predictions
            all_forecasts.append(week_forecast)
            
            new_data = week_forecast.rename(columns={'quantidade_prevista': 'quantidade'})
            new_data['ano'] = 2023
            forecast_df = pd.concat([forecast_df, new_data], ignore_index=True)
            
        return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "features": self.feature_names,
            "categorical": self.categorical_features
        }, path)
