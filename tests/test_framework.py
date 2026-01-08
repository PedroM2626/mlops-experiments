import pytest
import os
import pandas as pd
import numpy as np
from train_and_save_professional import MLOpsEnterprise

@pytest.fixture
def framework():
    return MLOpsEnterprise()

@pytest.fixture
def dummy_data():
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })
    df.to_csv("test_data.csv", index=False)
    yield "test_data.csv"
    if os.path.exists("test_data.csv"):
        os.remove("test_data.csv")

# --- TESTES UNITÁRIOS ---
def test_initialization(framework):
    assert framework.repo_owner == 'PedroM2626'
    assert framework.repo_name == 'experiments'

def test_data_loading(dummy_data):
    df = pd.read_csv(dummy_data)
    assert len(df) == 100
    assert 'target' in df.columns

# --- TESTES DE INTEGRAÇÃO ---
def test_mlflow_connection(framework):
    import mlflow
    assert "dagshub.com" in mlflow.get_tracking_uri()

@pytest.mark.parametrize("engine", ["tpot", "flaml"])
def test_automl_engines(framework, dummy_data, engine):
    """Testa se as diferentes engines de AutoML iniciam sem erros."""
    try:
        model = framework.train_automl(dummy_data, engine=engine, timeout=10)
        assert model is not None
    except Exception as e:
        pytest.fail(f"AutoML engine {engine} falhou: {e}")

# --- TESTES DE ACEITAÇÃO ---
def test_full_flow_acceptance(framework, dummy_data):
    """Verifica se o fluxo completo de treinamento e explicação funciona."""
    # 1. Treino
    model = framework.train_automl(dummy_data, generations=1, population_size=5)
    
    # 2. Explicação
    df = pd.read_csv(dummy_data)
    X_sample = df.drop(columns=['target']).iloc[:5]
    try:
        framework.explain_model(model, X_sample, method='shap')
        assert os.path.exists("shap_summary.png")
    except Exception as e:
        pytest.fail(f"Fluxo de aceitação falhou na explicação: {e}")
    finally:
        if os.path.exists("shap_summary.png"):
            os.remove("shap_summary.png")
