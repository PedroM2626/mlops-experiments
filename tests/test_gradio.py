import pytest
import os
import pandas as pd
import numpy as np
from gradio_app import run_training, run_drift_analysis

@pytest.fixture
def dummy_csv():
    df = pd.DataFrame({
        'feat1': np.random.rand(10),
        'feat2': np.random.rand(10),
        'target': np.random.randint(0, 2, 10)
    })
    path = "test_gradio_data.csv"
    df.to_csv(path, index=False)
    # Criar um objeto de arquivo mock que o Gradio esperaria
    class MockFile:
        def __init__(self, name):
            self.name = name
    yield MockFile(path)
    if os.path.exists(path):
        os.remove(path)

def test_gradio_run_training_no_file():
    status, shap, api = run_training(None, "classification", "flaml", 30)
    assert "Por favor, faça o upload" in status

def test_gradio_run_drift_no_files():
    result = run_drift_analysis(None, None)
    assert "Por favor, envie os dois arquivos" in result

@pytest.mark.skipif(os.getenv("CI") == "true", reason="Pula teste pesado de AutoML no CI")
def test_gradio_run_training_integration(dummy_csv):
    # Testa uma execução curta para verificar se a integração funciona
    status, shap, api = run_training(dummy_csv, "classification", "flaml", 30)
    assert "✅ Treino Concluído" in status
    assert "Score:" in status
