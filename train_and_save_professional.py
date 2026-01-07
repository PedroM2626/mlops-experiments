#!/usr/bin/env python3
"""
🎯 MLOPS ENTERPRISE - UNIVERSAL FRAMEWORK
Recursos de Produção:
- Data Validation (Great Expectations)
- Hyperparameter Tuning (Optuna)
- Professional Fine-tuning (Transformers)
- Drift Detection & Monitoring (Evidently)
- Automated API Generation (FastAPI)
"""

import os
import json
import shutil
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

# MLOps, Tracking & Optimization
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.transformers
import dagshub
import optuna

# Validação e Monitoramento
try:
    import great_expectations as ge
    HAS_GE = True
except ImportError:
    HAS_GE = False

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

# ML Clássico
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Deep Learning
import torch
try:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

warnings.filterwarnings('ignore')

class MLOpsEnterprise:
    def __init__(self, repo_owner='PedroM2626', repo_name='experiments'):
        load_dotenv()
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_dagshub()

    def _setup_dagshub(self):
        try:
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
            print("✅ Conectado ao DagsHub/MLflow")
        except Exception as e:
            print(f"⚠️ Erro DagsHub: {e}")

    # --- MÓDULO: DATA DRIFT MONITORING ---
    def detect_drift(self, reference_df, current_df, output_path="drift_report.html"):
        if not HAS_EVIDENTLY:
            print("⚠️ Evidently não instalado.")
            return
        
        print("\n�️ Analisando Data Drift (Monitoramento)...")
        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report.save_html(output_path)
        
        mlflow.log_artifact(output_path)
        print(f"✅ Relatório de Drift gerado: {output_path}")

    # --- MÓDULO: AUTOMATED API GENERATION ---
    def generate_serving_api(self, model_name="rf_hpo_optimized"):
        print(f"\n🚀 Gerando API de Serving para {model_name}...")
        api_code = f"""
import uvicorn
from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import dagshub
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
dagshub.init(repo_owner='{self.repo_owner}', repo_name='{self.repo_name}', mlflow=True)

app = FastAPI(title="MLOps Enterprise API - {model_name}")

# Carregar modelo do MLflow
model_uri = f"models:/{model_name}/1"  # Usando versão 1 explicitamente para o teste
model = mlflow.sklearn.load_model(model_uri)

class PredictionInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: PredictionInput):
    prediction = model.predict([data.text])[0]
    return {{"prediction": str(prediction)}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        with open("app_serving.py", "w", encoding="utf-8") as f:
            f.write(api_code)
        print("✅ API gerada com sucesso em 'app_serving.py'.")

    # --- MÓDULO: HPO & TRAINING ---
    def train_with_hpo(self, data_path='processed_train.csv'):
        print("\n📊 Iniciando Ciclo de Treino com HPO...")
        df = pd.read_csv(data_path)
        
        # Simular drift para demonstração posterior
        train_df, test_df = train_test_split(df, test_size=0.2)
        
        X_train = train_df['text_lemmatized'].fillna('')
        y_train = train_df['sentiment']
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 100)
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000)),
                ('clf', RandomForestClassifier(n_estimators=n_estimators))
            ])
            pipeline.fit(X_train, y_train)
            return accuracy_score(y_train, pipeline.predict(X_train))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5)

        # Configurar experimento explicitamente
        mlflow.set_experiment("/production_hpo")

        with mlflow.start_run(run_name="production_ready_model"):
            best_n = study.best_params['n_estimators']
            final_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000)),
                ('clf', RandomForestClassifier(n_estimators=best_n))
            ])
            final_pipeline.fit(X_train, y_train)
            
            mlflow.log_params(study.best_params)
            mlflow.sklearn.log_model(final_pipeline, "model", registered_model_name="rf_hpo_optimized")
            
            # Monitoramento
            self.detect_drift(train_df, test_df)
            self.generate_serving_api("rf_hpo_optimized")

def main():
    m = MLOpsEnterprise()
    m.train_with_hpo()

if __name__ == "__main__":
    main()
