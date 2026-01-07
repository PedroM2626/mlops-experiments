#!/usr/bin/env python3
"""
🎯 MLOPS ENTERPRISE - UNIVERSAL FRAMEWORK
Recursos avançados:
- Data Validation (Great Expectations)
- Hyperparameter Tuning (Optuna)
- Professional Fine-tuning (Transformers/Deep Learning)
- ML Clássico, CV, Time Series e Clustering
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
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

# MLOps, Tracking & Optimization
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.transformers
import dagshub
import optuna
try:
    import great_expectations as ge
    HAS_GE = True
except ImportError:
    HAS_GE = False

# ML Clássico
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Deep Learning & Transformers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
try:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

warnings.filterwarnings('ignore')

class MLOpsEnterprise:
    """
    Framework de MLOps de nível corporativo.
    """
    
    def __init__(self, repo_owner='PedroM2626', repo_name='experiments'):
        load_dotenv()
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.dagshub_token = os.getenv('DAGSHUB_TOKEN', '')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = "temp_mlops_enterprise"
        
        self._setup_dagshub()
        print(f"🏢 MLOps Enterprise Inicializado | Device: {self.device}")

    def _setup_dagshub(self):
        try:
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
            if self.dagshub_token:
                os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.s3"
                dagshub.auth.add_app_token(self.dagshub_token)
            print("✅ Conectado ao DagsHub/MLflow")
        except Exception as e:
            print(f"⚠️ Erro DagsHub: {e}")

    def validate_data(self, df, context="training"):
        """Validação de dados com Great Expectations."""
        if not HAS_GE:
            print("⚠️ Great Expectations não instalado. Pulando validação.")
            return True
        
        print(f"🔍 Validando dados de {context}...")
        gdf = ge.from_pandas(df)
        
        # Expectativas básicas
        results = [
            gdf.expect_table_row_count_to_be_between(min_value=10),
            gdf.expect_column_values_to_not_be_null("sentiment" if "sentiment" in df.columns else df.columns[0]),
        ]
        
        success = all(r.success for r in results)
        if success:
            print("✅ Dados validados com sucesso.")
        else:
            print("❌ Falha na validação dos dados!")
        return success

    # --- MÓDULO: HYPERPARAMETER TUNING (OPTUNA) ---
    def optimize_nlp(self, X_train, y_train, X_val, y_val, n_trials=10):
        print(f"🧪 Iniciando Otimização de Hiperparâmetros (Optuna - {n_trials} trials)...")
        
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 10, 200)
            max_depth = trial.suggest_int('max_depth', 2, 32)
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=trial.suggest_int('max_features', 1000, 5000))),
                ('clf', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth))
            ])
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            return accuracy_score(y_val, preds)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"🏆 Melhores Parâmetros: {study.best_params}")
        return study.best_params

    # --- MÓDULO: FINE-TUNING (TRANSFORMERS) ---
    def fine_tune_transformer(self, model_name="distilbert-base-uncased", data_path='processed_train.csv'):
        if not HAS_TRANSFORMERS: return
        
        print(f"\n🤗 Iniciando Fine-tuning de {model_name}...")
        df = pd.read_csv(data_path).sample(200) # Exemplo rápido
        if not self.validate_data(df, "transformers"): return

        labels = sorted(df['sentiment'].unique())
        label2id = {l: i for i, l in enumerate(labels)}
        id2label = {i: l for i, l in enumerate(labels)}
        df['label'] = df['sentiment'].map(label2id)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        def tokenize(batch): return tokenizer(batch["text_clean"], truncation=True, padding=True)
        
        ds = Dataset.from_pandas(df[['text_clean', 'label']]).map(tokenize, batched=True)
        ds = ds.train_test_split(test_size=0.2)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
        )

        mlflow.set_experiment("transformer_fine_tuning")
        with mlflow.start_run(run_name=f"finetune_{model_name.split('/')[-1]}"):
            args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=2,
                per_device_train_batch_size=8,
                evaluation_strategy="epoch",
                logging_steps=10,
                report_to="none"
            )

            trainer = Trainer(
                model=model, args=args,
                train_dataset=ds["train"], eval_dataset=ds["test"],
                tokenizer=tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer)
            )

            print("🚀 Treinando...")
            trainer.train()
            
            # Log Model
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="fine_tuned_model",
                registered_model_name="sentiment_transformer_ft"
            )
            print("✅ Fine-tuning concluído e modelo registrado.")

    # --- MÓDULO: ML CLÁSSICO COM HPO ---
    def train_with_hpo(self, data_path='processed_train.csv'):
        print("\n📊 Treinando ML Clássico com Otimização (HPO)...")
        df = pd.read_csv(data_path)
        if not self.validate_data(df, "classic_ml"): return

        X = df['text_lemmatized'].fillna('')
        y = df['sentiment']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

        # HPO
        best_params = self.optimize_nlp(X_train, y_train, X_val, y_val)

        mlflow.set_experiment("classic_ml_hpo")
        with mlflow.start_run(run_name="rf_optimized"):
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=best_params['max_features'])),
                ('clf', RandomForestClassifier(
                    n_estimators=best_params['n_estimators'], 
                    max_depth=best_params['max_depth']
                ))
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            acc = accuracy_score(y_val, y_pred)
            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", acc)
            
            mlflow.sklearn.log_model(pipeline, "model", registered_model_name="rf_hpo_optimized")
            print(f"✅ Modelo Otimizado Registrado (Acc: {acc:.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["hpo", "finetune", "all"], default="all")
    args = parser.parse_args()

    m = MLOpsEnterprise()
    
    if args.task in ["hpo", "all"]:
        m.train_with_hpo()
    
    if args.task in ["finetune", "all"]:
        m.fine_tune_transformer()

if __name__ == "__main__":
    main()
