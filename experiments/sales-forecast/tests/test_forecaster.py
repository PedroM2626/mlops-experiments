import os
import sys
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest
from scripts.forecaster_class import SalesForecasterV2


@pytest.fixture
def dummy_data():
    """Gera DataFrame sintetico que reproduz a estrutura completa de load_data."""
    n = 10
    df = pd.DataFrame({
        'ano': [2022] * n,
        'semana': [1, 2, 3, 4, 10, 1, 2, 3, 4, 48],
        'pdv': [1]*5 + [2]*5,
        'sku': [1]*5 + [1]*5,
        'quantidade': [10, 15, 20, 25, 30, 12, 14, 16, 18, 22],
        'preco_medio_unitario': [5.0]*n,
        'categoria_pdv': ['Restaurant']*5 + ['Convenience']*5,
        'premise': ['On Premise']*5 + ['Off Premise']*5,
        'categoria': ['Distilled Spirits']*n,
        'subcategoria': ['Bourbon Whiskey']*n,
        'tipos': ['Distilled Spirits']*n,
        'label': ['Core']*n,
        'marca': ['Marca X']*n,
        'fabricante': ['Fabricante Y']*n,
    })
    return df


class TestFeatureEngineering:
    def test_output_not_empty(self, dummy_data):
        f = SalesForecasterV2()
        df_feat = f.feature_engineering(dummy_data)
        assert not df_feat.empty

    def test_lag_columns_exist(self, dummy_data):
        f = SalesForecasterV2()
        df_feat = f.feature_engineering(dummy_data)
        for lag in [1, 2, 3, 4, 12, 52]:
            assert f'lag_{lag}_semanas' in df_feat.columns

    def test_rolling_columns_exist(self, dummy_data):
        f = SalesForecasterV2()
        df_feat = f.feature_engineering(dummy_data)
        for w in [4, 12, 52]:
            assert f'rolling_mean_{w}_semanas' in df_feat.columns
        for w in [4, 12]:
            assert f'rolling_min_{w}_semanas' in df_feat.columns

    def test_new_features_exist(self, dummy_data):
        f = SalesForecasterV2()
        df_feat = f.feature_engineering(dummy_data)
        assert 'lag_diff_1' in df_feat.columns
        assert 'coef_variacao_4' in df_feat.columns
        assert 'lag_1_preco' in df_feat.columns

    def test_no_nans_after_fillna(self, dummy_data):
        f = SalesForecasterV2()
        df_feat = f.feature_engineering(dummy_data)
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
        assert df_feat[numeric_cols].isna().sum().sum() == 0

    def test_index_is_unique(self, dummy_data):
        f = SalesForecasterV2()
        df_feat = f.feature_engineering(dummy_data)
        assert df_feat.index.is_unique


class TestTrainingAndPrediction:
    def test_model_trains_without_optuna(self, dummy_data):
        f = SalesForecasterV2()
        f.train(dummy_data, validation_split_week=48, use_optuna=False)
        assert f.model is not None
        assert 'validation_mae' in f.performance_metrics
        assert f.performance_metrics['validation_mae'] >= 0

    def test_forecast_output_shape(self, dummy_data):
        f = SalesForecasterV2()
        f.train(dummy_data, validation_split_week=48, use_optuna=False)
        forecast = f.generate_forecasts(dummy_data, weeks_to_forecast=2)
        assert not forecast.empty
        assert 'quantidade_prevista' in forecast.columns
        assert forecast['semana'].isin([1, 2]).all()


class TestPersistence:
    def test_save_and_load_roundtrip(self, dummy_data):
        f = SalesForecasterV2()
        f.train(dummy_data, validation_split_week=48, use_optuna=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.joblib')
            f.save_model(path)
            assert os.path.exists(path)

            import joblib
            loaded = joblib.load(path)
            assert 'model' in loaded
            assert 'feature_names' in loaded
            assert 'categorical_features' in loaded
            assert 'performance_metrics' in loaded
            assert 'best_params' in loaded


class TestFeatureImportancePlot:
    def test_plot_generates_file(self, dummy_data):
        f = SalesForecasterV2()
        f.train(dummy_data, validation_split_week=48, use_optuna=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'fi.png')
            result = f.plot_feature_importance(path)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0
