import logging
import pandas as pd
import numpy as np
import time
from sktime.transformations.series.summarize import WindowSummarizer
from .forecaster_class import SalesForecasterV2

class SalesForecasterSktime(SalesForecasterV2):
    """Classe de previsao de vendas que herda da V2 mas usa sktime para extrair features temporais."""
    
    def __init__(self):
        super().__init__()
        self.fe_time = 0.0

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gera features temporais usando sktime WindowSummarizer paralelizado."""
        start_time = time.time()
        
        df_featured = df.copy()
        
        # 1. Features Temporais Básicas
        df_featured['trimestre'] = (df_featured['semana'] - 1) // 13 + 1
        df_featured['seno_semana'] = np.sin(2 * np.pi * df_featured['semana'] / 52)
        df_featured['cosseno_semana'] = np.cos(2 * np.pi * df_featured['semana'] / 52)
        
        # 2. Sktime WindowSummarizer para Lags e Janelas Moveis
        # Configurar index hierárquico exigido para Panel Data
        df_featured.set_index(['pdv', 'sku', 'semana'], inplace=True, drop=False)
        
        kwargs = {
            "lag_feature": {
                "lag": [1, 2, 3, 4, 12, 52],
                "mean": [[1, 4], [1, 12], [1, 52]],
                "std": [[1, 4], [1, 12], [1, 52]],
                "max": [[1, 4], [1, 12], [1, 52]],
                "min": [[1, 4], [1, 12]]
            }
        }
        
        kwargs_preco = {
            "lag_feature": {
                "lag": [1]
            }
        }
        
        transformer_qty = WindowSummarizer(**kwargs, target_cols=["quantidade"], n_jobs=1)
        transformer_price = WindowSummarizer(**kwargs_preco, target_cols=["preco_medio_unitario"], n_jobs=1)
        
        # Transformando e mesclando de volta (passando apenas a coluna alvo para evitar erro de dtype 'object')
        df_qty_feats = transformer_qty.fit_transform(df_featured[['quantidade']])
        df_price_feats = transformer_price.fit_transform(df_featured[['preco_medio_unitario']])
        
        # Juntar features geradas
        df_featured = pd.concat([df_featured, df_qty_feats, df_price_feats], axis=1)
        
        # Voltar a flat data
        df_featured.reset_index(drop=True, inplace=True)
        
        # 3. Features Derivadas
        df_featured['lag_diff_1'] = df_featured['quantidade_lag_1'] - df_featured['quantidade_lag_2']
        
        mean_4 = df_featured['quantidade_mean_1_4']
        std_4 = df_featured['quantidade_std_1_4']
        df_featured['coef_variacao_4'] = np.where(mean_4 > 0, std_4 / mean_4, 0.0)
        
        df_featured.fillna(0, inplace=True)
        
        self.fe_time += (time.time() - start_time)
        return df_featured

    def _prepare_data_for_model(self, df: pd.DataFrame):
        """Seleciona as features recem criadas pelo sktime."""
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
            
            # Sktime Lags (Quantidade)
            'quantidade_lag_1', 'quantidade_lag_2', 'quantidade_lag_3', 'quantidade_lag_4',
            'quantidade_lag_12', 'quantidade_lag_52',
            
            # Sktime Lag (Preço)
            'preco_medio_unitario_lag_1', 
            
            # Derivadas
            'lag_diff_1', 'coef_variacao_4',
            
            # Sktime Janelas (Média, Std, Max, Min)
            'quantidade_mean_1_4', 'quantidade_std_1_4', 'quantidade_max_1_4', 'quantidade_min_1_4',
            'quantidade_mean_1_12', 'quantidade_std_1_12', 'quantidade_max_1_12', 'quantidade_min_1_12',
            'quantidade_mean_1_52', 'quantidade_std_1_52', 'quantidade_max_1_52',
            
            # Outros
            'preco_medio_unitario',
        ]

        X = df_model[self.feature_names]
        y = df_model['quantidade']
        return X, y
