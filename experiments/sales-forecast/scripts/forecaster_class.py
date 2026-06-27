import logging
import os
from typing import Dict, List, Tuple
import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import optuna
from optuna.integration import LightGBMPruningCallback


class SalesForecasterV2:
    """Classe principal do pipeline de previsao de vendas (Arquitetura V2.2).

    Encapsula carregamento de dados, engenharia de features, treinamento
    com otimizacao Bayesiana (Optuna) e geracao de forecasts semanais.
    """

    def __init__(self):
        self.model = None
        self.feature_names: List[str] = []
        self.categorical_features: List[str] = []
        self.performance_metrics: Dict[str, float] = {}
        self.best_params: Dict = {}
        self.use_log_target: bool = False

    # ------------------------------------------------------------------
    # 1. CARREGAMENTO E ENRIQUECIMENTO DE DADOS
    # ------------------------------------------------------------------
    def load_data(self, file_paths: Dict[str, str]) -> pd.DataFrame:
        """Carrega, funde e agrega os dados brutos em granularidade semanal."""
        logging.info("Iniciando o carregamento dos dados normalizados.")
        try:
            df_vendas = pd.read_parquet(file_paths['vendas'])
            df_pdvs = pd.read_parquet(file_paths['pdvs'])
            df_produtos = pd.read_parquet(file_paths['produtos'])
        except (FileNotFoundError, KeyError) as e:
            logging.error(f"Erro ao carregar os arquivos. Erro: {e}")
            raise

        df_merged = pd.merge(
            df_vendas, df_pdvs,
            left_on='internal_store_id', right_on='pdv', how='inner'
        )
        df_merged = pd.merge(
            df_merged, df_produtos,
            left_on='internal_product_id', right_on='produto', how='inner'
        )

        df_merged['transaction_date'] = pd.to_datetime(df_merged['transaction_date'])
        df_merged['ano'] = df_merged['transaction_date'].dt.isocalendar().year
        df_merged['semana'] = df_merged['transaction_date'].dt.isocalendar().week

        logging.info("Agregando dados de vendas por semana/pdv/produto com dimensoes enriquecidas.")

        dim_cols = [
            'categoria_pdv', 'premise',
            'categoria', 'subcategoria', 'tipos', 'label', 'marca', 'fabricante'
        ]
        group_cols = ['ano', 'semana', 'pdv', 'produto'] + dim_cols

        agg_vendas = df_merged.groupby(group_cols).agg(
            total_quantity=('quantity', 'sum'),
            total_gross_value=('gross_value', 'sum'),
        ).reset_index()

        agg_vendas = agg_vendas.rename(columns={
            'produto': 'sku',
            'total_quantity': 'quantidade',
        })

        agg_vendas['preco_medio_unitario'] = np.where(
            agg_vendas['quantidade'] > 0,
            agg_vendas['total_gross_value'] / agg_vendas['quantidade'],
            0.0
        )
        agg_vendas.drop(columns=['total_gross_value'], inplace=True)

        logging.info(f"Dados agregados e enriquecidos. DataFrame final com {agg_vendas.shape[0]} registros.")
        return agg_vendas

    # ------------------------------------------------------------------
    # 2. ENGENHARIA DE FEATURES
    # ------------------------------------------------------------------
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features temporais, ciclicas e de tendencia a partir do historico."""
        df_featured = df.copy()
        df_featured.sort_values(['pdv', 'sku', 'ano', 'semana'], inplace=True)
        df_featured.reset_index(drop=True, inplace=True)

        df_featured['trimestre'] = (df_featured['semana'] - 1) // 13 + 1
        df_featured['seno_semana'] = np.sin(2 * np.pi * df_featured['semana'] / 52)
        df_featured['cosseno_semana'] = np.cos(2 * np.pi * df_featured['semana'] / 52)

        grouped_qty = df_featured.groupby(['pdv', 'sku'])['quantidade']
        lags = [1, 2, 3, 4, 12, 52]
        for lag in lags:
            df_featured[f'lag_{lag}_semanas'] = grouped_qty.shift(lag)

        grouped_price = df_featured.groupby(['pdv', 'sku'])['preco_medio_unitario']
        df_featured['lag_1_preco'] = grouped_price.shift(1)

        df_featured['lag_diff_1'] = df_featured['lag_1_semanas'] - df_featured['lag_2_semanas']

        shifted = grouped_qty.shift(1)
        tmp = pd.DataFrame({
            'val': shifted,
            'pdv': df_featured['pdv'],
            'sku': df_featured['sku']
        })
        tmp_grouped = tmp.groupby(['pdv', 'sku'])['val']

        windows = [4, 12, 52]
        for window in windows:
            roll = tmp_grouped.rolling(window=window, min_periods=1)
            df_featured[f'rolling_mean_{window}_semanas'] = roll.mean().reset_index(level=[0, 1], drop=True)
            df_featured[f'rolling_std_{window}_semanas'] = roll.std().reset_index(level=[0, 1], drop=True)
            df_featured[f'rolling_max_{window}_semanas'] = roll.max().reset_index(level=[0, 1], drop=True)

        for window in [4, 12]:
            roll = tmp_grouped.rolling(window=window, min_periods=1)
            df_featured[f'rolling_min_{window}_semanas'] = roll.min().reset_index(level=[0, 1], drop=True)

        mean_4 = df_featured['rolling_mean_4_semanas']
        std_4 = df_featured['rolling_std_4_semanas']
        df_featured['coef_variacao_4'] = np.where(mean_4 > 0, std_4 / mean_4, 0.0)

        df_featured.fillna(0, inplace=True)
        return df_featured

    # ------------------------------------------------------------------
    # 3. PREPARACAO PARA O MODELO
    # ------------------------------------------------------------------
    def _prepare_data_for_model(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Seleciona features, define tipos categoricos e separa X/y."""
        df_model = df.copy()

        self.categorical_features = [
            'pdv', 'sku',
            'categoria_pdv', 'premise',
            'categoria', 'subcategoria', 'tipos', 'label', 'marca', 'fabricante'
        ]
        for col in self.categorical_features:
            df_model[col] = df_model[col].astype('category')

        self.feature_names = [
            'semana', 'trimestre', 'seno_semana', 'cosseno_semana',
            'pdv', 'sku',
            'categoria_pdv', 'premise',
            'categoria', 'subcategoria', 'tipos', 'label', 'marca', 'fabricante',
            'lag_1_semanas', 'lag_2_semanas', 'lag_3_semanas', 'lag_4_semanas',
            'lag_12_semanas', 'lag_52_semanas',
            'lag_1_preco', 'lag_diff_1',
            'rolling_mean_4_semanas', 'rolling_std_4_semanas', 'rolling_max_4_semanas',
            'rolling_mean_12_semanas', 'rolling_std_12_semanas', 'rolling_max_12_semanas',
            'rolling_mean_52_semanas', 'rolling_std_52_semanas', 'rolling_max_52_semanas',
            'rolling_min_4_semanas', 'rolling_min_12_semanas',
            'coef_variacao_4',
            'preco_medio_unitario',
        ]

        X = df_model[self.feature_names]
        y = df_model['quantidade']
        return X, y

    # ------------------------------------------------------------------
    # 4. TREINAMENTO
    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, validation_split_week: int = 48,
              use_optuna: bool = True, n_trials: int = 100):
        """Treina o modelo LightGBM com otimizacao Bayesiana via Optuna."""
        df_train_raw = df[df['ano'] == 2022].copy()
        if df_train_raw.empty:
            raise ValueError("Nao ha dados historicos de 2022 para treinar o modelo.")

        df_featured = self.feature_engineering(df_train_raw)
        train_set = df_featured[df_featured['semana'] < validation_split_week]
        val_set = df_featured[df_featured['semana'] >= validation_split_week]

        X_train, y_train_raw = self._prepare_data_for_model(train_set)
        X_val, y_val_raw = self._prepare_data_for_model(val_set)

        if self.use_log_target:
            y_train = np.log1p(np.maximum(0, y_train_raw))
            y_val = np.log1p(np.maximum(0, y_val_raw))
        else:
            y_train = y_train_raw
            y_val = y_val_raw

        for col in self.categorical_features:
            all_categories = pd.concat([X_train[col], X_val[col]]).astype('category').cat.categories
            X_train[col] = pd.Categorical(X_train[col], categories=all_categories)
            X_val[col] = pd.Categorical(X_val[col], categories=all_categories)

        fit_params = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "mae",
            "callbacks": [lgb.early_stopping(30, verbose=False)],
        }

        if use_optuna:
            def objective(trial):
                params = {
                    'objective': 'regression_l1',
                    'metric': 'mae',
                    'verbosity': -1,
                    'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 256),
                    'max_depth': trial.suggest_int('max_depth', 5, 15),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                    'random_state': 42,
                    'n_jobs': -1,
                }
                pruning_callback = LightGBMPruningCallback(trial, "l1")
                cv_fit_params = fit_params.copy()
                cv_fit_params["callbacks"] = [
                    lgb.early_stopping(20, verbose=False),
                    pruning_callback,
                ]
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_train, y_train,
                    **cv_fit_params,
                    categorical_feature=self.categorical_features,
                )
                preds_log = model.predict(X_val)
                if self.use_log_target:
                    preds_orig = np.expm1(preds_log)
                    mae = mean_absolute_error(y_val_raw, preds_orig)
                else:
                    mae = mean_absolute_error(y_val, preds_log)
                return mae

            study = optuna.create_study(
                direction='minimize',
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            self.best_params = study.best_params
            logging.info(f"Melhores hiperparametros encontrados: {self.best_params}")

            final_params = {k: v for k, v in self.best_params.items()}
            final_params['n_estimators'] = 1000
            self.model = lgb.LGBMRegressor(
                objective='regression_l1', random_state=42,
                verbosity=-1, n_jobs=-1,
                **final_params,
            )
        else:
            self.model = lgb.LGBMRegressor(
                objective='regression_l1', random_state=42,
                n_estimators=500, verbosity=-1, n_jobs=-1,
            )

        final_fit_params = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "mae",
            "callbacks": [lgb.early_stopping(50, verbose=False)],
        }
        self.model.fit(
            X_train, y_train,
            **final_fit_params,
            categorical_feature=self.categorical_features,
        )
        val_preds_log = self.model.predict(X_val)
        if self.use_log_target:
            val_preds = np.expm1(val_preds_log)
            mae = mean_absolute_error(y_val_raw, val_preds)
        else:
            mae = mean_absolute_error(y_val, val_preds_log)
            
        self.performance_metrics['validation_mae'] = mae
        self.performance_metrics['n_features'] = len(self.feature_names)
        self.performance_metrics['train_size'] = len(X_train)
        self.performance_metrics['val_size'] = len(X_val)
        self.performance_metrics['best_iteration'] = self.model.best_iteration_
        logging.info(f"Treinamento concluido. MAE no set de validacao: {mae:.4f}")

    # ------------------------------------------------------------------
    # 5. FEATURE IMPORTANCE PLOT
    # ------------------------------------------------------------------
    def plot_feature_importance(self, output_path: str) -> str:
        """Gera e salva grafico de importancia de features."""
        if not self.model:
            raise RuntimeError("O modelo nao foi treinado.")

        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
        }).sort_values('importance', ascending=True)

        fig, ax = plt.subplots(figsize=(10, max(8, len(self.feature_names) * 0.3)))
        ax.barh(feature_imp['feature'], feature_imp['importance'], color='#2196F3')
        ax.set_xlabel('Importance (split count)')
        ax.set_title('Feature Importance - Sales Forecaster V2')
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"Grafico de feature importance salvo em: '{output_path}'")
        return output_path

    # ------------------------------------------------------------------
    # 6. PREVISAO
    # ------------------------------------------------------------------
    def generate_forecasts(self, df_historical: pd.DataFrame, weeks_to_forecast: int) -> pd.DataFrame:
        """Gera previsoes iterativas semana a semana."""
        if not self.model:
            raise RuntimeError("O modelo nao foi treinado.")

        forecast_df = df_historical.copy()
        all_forecasts = []

        for i in range(1, weeks_to_forecast + 1):
            current_week = i
            features_base = self.feature_engineering(forecast_df)
            latest_entries = features_base.sort_values(
                by=['ano', 'semana']
            ).drop_duplicates(subset=['pdv', 'sku'], keep='last')

            if latest_entries.empty:
                continue

            X_pred = latest_entries.copy()
            X_pred['semana'] = current_week
            X_pred['ano'] = 2023

            for col in self.categorical_features:
                idx = self.categorical_features.index(col)
                model_categories = self.model.booster_.pandas_categorical[idx]
                X_pred[col] = pd.Categorical(X_pred[col], categories=model_categories)

            X_pred.dropna(subset=self.categorical_features, inplace=True)
            if X_pred.empty:
                continue

            predictions_raw = self.model.predict(X_pred[self.feature_names])
            if self.use_log_target:
                predictions_raw = np.expm1(predictions_raw)
            predictions = np.maximum(0, np.round(predictions_raw)).astype(int)

            week_forecast = X_pred[['pdv', 'sku']].copy()
            week_forecast['semana'] = current_week
            week_forecast['quantidade_prevista'] = predictions
            all_forecasts.append(week_forecast)

            new_data = week_forecast.rename(columns={'quantidade_prevista': 'quantidade'})
            new_data['ano'] = 2023

            dim_cols_to_copy = [
                c for c in forecast_df.columns
                if c not in ['ano', 'semana', 'pdv', 'sku', 'quantidade', 'preco_medio_unitario']
            ]
            for col in dim_cols_to_copy:
                if col in X_pred.columns:
                    new_data[col] = X_pred[col].values

            if 'lag_1_preco' in X_pred.columns:
                new_data['preco_medio_unitario'] = X_pred['lag_1_preco'].values
            else:
                new_data['preco_medio_unitario'] = 0.0

            forecast_df = pd.concat([forecast_df, new_data], ignore_index=True)

        return pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

    # ------------------------------------------------------------------
    # 7. PERSISTENCIA
    # ------------------------------------------------------------------
    def save_model(self, path: str):
        """Salva modelo treinado e todos os metadados associados."""
        if not self.model:
            raise RuntimeError("O modelo nao foi treinado. Impossivel salvar.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        artifacts = {
            "model": self.model,
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "performance_metrics": self.performance_metrics,
            "best_params": self.best_params,
            "use_log_target": self.use_log_target,
        }
        joblib.dump(artifacts, path)
        logging.info(f"Modelo e artefatos V2.2 salvos em: '{path}'")
