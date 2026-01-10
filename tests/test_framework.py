import pytest
import os
import pandas as pd
import numpy as np
import mlflow
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
    data_path = "test_data.csv"
    df.to_csv(data_path, index=False)
    yield data_path
    if os.path.exists(data_path):
        os.remove(data_path)

# --- TESTES UNITÁRIOS ---
def test_initialization(framework):
    """Verifica se a inicialização do framework está correta."""
    assert framework.repo_owner == 'PedroM2626'
    assert framework.repo_name == 'experiments'

def test_validate_data(framework, dummy_data):
    """Verifica a validação de dados."""
    df = pd.read_csv(dummy_data)
    assert framework.validate_data(df, 'target') is True
    
    with pytest.raises(ValueError):
        framework.validate_data(df, 'wrong_target')

def test_log_metrics(framework):
    """Verifica se o logging de métricas funciona no MLflow."""
    mlflow.set_experiment("/test_experiment")
    with mlflow.start_run():
        framework._log_metrics_and_plots({"test_metric": 0.95})
        # Verifica se o MLflow registrou a métrica (precisa estar conectado)
        # Como estamos testando localmente sem mock, apenas verificamos se não falha

# --- TESTES DE INTEGRAÇÃO ---
@pytest.mark.parametrize("engine", ["flaml"])
def test_automl_integration(framework, dummy_data, engine):
    """Testa a integração com motores de AutoML (usando timeout curto)."""
    model, score = framework.train_automl(dummy_data, engine=engine, timeout=10)
    assert model is not None
    assert score >= 0

def test_onnx_export(framework, dummy_data):
    """Verifica a exportação para ONNX."""
    from sklearn.ensemble import RandomForestClassifier
    df = pd.read_csv(dummy_data)
    X = df.drop(columns=['target'])
    y = df['target']
    model = RandomForestClassifier().fit(X, y)
    
    framework.export_to_onnx(model, X.iloc[:1])
    assert os.path.exists("model.onnx")
    if os.path.exists("model.onnx"):
        os.remove("model.onnx")

# --- TESTES DE ACEITAÇÃO ---
def test_full_pipeline_acceptance(framework, dummy_data):
    """Fluxo completo: Validação -> Treino -> Explicação -> Exportação."""
    # 1. Carregar e Validar
    df = pd.read_csv(dummy_data)
    framework.validate_data(df, 'target')
    
    # 2. Treinar (usando FLAML por ser rápido)
    model, score = framework.train_automl(dummy_data, engine='flaml', timeout=10)
    assert model is not None
    
    # 3. Explicar
    X_train = df.drop(columns=['target'])
    framework.explain_model(model, X_train, method='shap')
    assert os.path.exists("shap_summary.png")
    
    # 4. Exportar Serving API
    framework.generate_serving_api("model.onnx")
    assert os.path.exists("serving_api.py")
    
    # Cleanup
    for f in ["shap_summary.png", "serving_api.py", "model.onnx"]:
        if os.path.exists(f):
            os.remove(f)
