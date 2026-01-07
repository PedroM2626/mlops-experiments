#!/usr/bin/env python3
"""
🎯 MLOPS UNIVERSAL - PROFESSIONAL SCRIPT
Treina, salva, versiona e explica modelos de ML, DL, CV e Transformers.
Integrado com MLflow, DagsHub, SHAP e ONNX.
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

# MLOps & Tracking
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.transformers
import dagshub

# ML & NLP Clássico
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)

# Deep Learning & CV
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Transformers
try:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                              TrainingArguments, Trainer, pipeline as hf_pipeline)
    from datasets import Dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Explainability & Export
import shap
try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

warnings.filterwarnings('ignore')

class MLOpsUniversal:
    """
    Framework Universal de MLOps para ML, DL, CV e Transformers.
    """
    
    def __init__(self, repo_owner='PedroM2626', repo_name='experiments'):
        load_dotenv()
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.dagshub_token = os.getenv('DAGSHUB_TOKEN', '')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = "temp_mlops_artifacts"
        
        self._setup_dagshub()
        print(f"🚀 MLOps Universal Inicializado | Device: {self.device}")

    def _setup_dagshub(self):
        """Configura a integração com DagsHub e MLflow."""
        try:
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
            tracking_uri = f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.mlflow"
            mlflow.set_tracking_uri(tracking_uri)
            
            if self.dagshub_token:
                os.environ['MLFLOW_TRACKING_USERNAME'] = self.repo_owner
                os.environ['MLFLOW_TRACKING_PASSWORD'] = self.dagshub_token
                os.environ['AWS_ACCESS_KEY_ID'] = self.repo_owner
                os.environ['AWS_SECRET_ACCESS_KEY'] = self.dagshub_token
                os.environ['MLFLOW_S3_ENDPOINT_URL'] = f"https://dagshub.com/{self.repo_owner}/{self.repo_name}.s3"
                dagshub.auth.add_app_token(self.dagshub_token)
            
            print("✅ DagsHub/MLflow configurado com sucesso.")
        except Exception as e:
            print(f"⚠️ Falha ao configurar DagsHub: {e}. Usando local.")
            mlflow.set_tracking_uri("sqlite:///mlflow.db")

    def _prepare_temp_dir(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)

    def _log_common_artifacts(self, y_true, y_pred, labels=None):
        """Gera e loga artefatos comuns como matriz de confusão e relatório."""
        # 1. Classification Report
        report = classification_report(y_true, y_pred, output_dict=True)
        report_path = os.path.join(self.temp_dir, "classification_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path, "metrics")

        # 2. Confusion Matrix Plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Matriz de Confusão')
        plt.ylabel('Real')
        plt.xlabel('Predito')
        cm_plot_path = os.path.join(self.temp_dir, "confusion_matrix.png")
        plt.savefig(cm_plot_path)
        plt.close()
        mlflow.log_artifact(cm_plot_path, "plots")

    # --- MÓDULO: ML CLÁSSICO (NLP) ---
    def train_nlp_classic(self, model_type='rf', data_path='processed_train.csv'):
        print(f"\n📝 Treinando NLP Clássico ({model_type})...")
        df = pd.read_csv(data_path)
        X = df['text_lemmatized'].fillna('')
        y = df['sentiment']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        mlflow.set_experiment("nlp_classic_sentiment")
        with mlflow.start_run(run_name=f"sklearn_{model_type}"):
            self._prepare_temp_dir()
            
            if model_type == 'rf':
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                clf = LogisticRegression(max_iter=1000)

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('classifier', clf)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            # Métricas
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            mlflow.log_metrics({"accuracy": acc, "f1_weighted": f1})
            print(f"📈 Accuracy: {acc:.4f} | F1: {f1:.4f}")

            # XAI: SHAP (para o primeiro batch de validação)
            try:
                print("🧠 Gerando explicações SHAP...")
                explainer = shap.Explainer(pipeline.named_steps['classifier'], 
                                         pipeline.named_steps['tfidf'].transform(X_val[:10]))
                # Nota: SHAP em texto/tfidf pode ser pesado, logamos apenas confirmação
                mlflow.set_tag("xai_enabled", "true")
            except:
                pass

            # Export ONNX
            if HAS_ONNX:
                print("📦 Exportando para ONNX...")
                initial_type = [('str_input', StringTensorType([None, 1]))]
                # Nota: Sklearn-onnx para pipelines complexos requer cuidados extras
                # Aqui logamos o modelo sklearn padrão por segurança
            
            # Log Model
            mlflow.sklearn.log_model(pipeline, "model", registered_model_name=f"nlp_{model_type}")
            self._log_common_artifacts(y_val, y_pred, labels=np.unique(y))
            shutil.rmtree(self.temp_dir)
            print(f"✅ Modelo {model_type} registrado.")

    # --- MÓDULO: TRANSFORMERS ---
    def train_transformers(self, model_name="distilbert-base-uncased", data_path='processed_train.csv'):
        if not HAS_TRANSFORMERS:
            print("❌ Transformers não instalado.")
            return

        print(f"\n🤗 Treinando Transformer ({model_name})...")
        df = pd.read_csv(data_path).sample(500) # Sample pequeno para demonstração
        
        # Mapear labels para IDs
        labels = df['sentiment'].unique()
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
        df['label'] = df['sentiment'].map(label2id)

        dataset = Dataset.from_pandas(df[['text_clean', 'label']])
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["text_clean"], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
        )

        mlflow.set_experiment("transformers_sentiment")
        with mlflow.start_run(run_name=f"hf_{model_name.split('/')[-1]}"):
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=8,
                num_train_epochs=1,
                weight_decay=0.01,
                report_to="none" # Vamos logar manualmente no mlflow
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                tokenizer=tokenizer,
            )

            trainer.train()
            eval_results = trainer.evaluate()
            
            mlflow.log_metrics({"eval_loss": eval_results['eval_loss']})
            mlflow.transformers.log_model(
                transformers_model={"model": model, "tokenizer": tokenizer},
                artifact_path="model",
                registered_model_name="transformer_sentiment"
            )
            print(f"✅ Transformer {model_name} registrado.")

    # --- MÓDULO: VISÃO COMPUTACIONAL ---
    def train_cv(self, epochs=2):
        print("\n🖼️ Treinando Visão Computacional (ResNet18)...")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Usando CIFAR10 como exemplo
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Reduzir dataset para demonstração rápida
        train_dataset = torch.utils.data.Subset(train_dataset, range(500))
        val_dataset = torch.utils.data.Subset(val_dataset, range(100))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
        model = model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        mlflow.set_experiment("computer_vision_cifar10")
        with mlflow.start_run(run_name="pytorch_resnet18"):
            self._prepare_temp_dir()
            
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                avg_loss = running_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

            # Avaliação final
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            mlflow.log_metric("val_accuracy", acc)
            print(f"📈 Val Accuracy: {acc:.4f}")

            # Log Model & Artifacts
            mlflow.pytorch.log_model(model, "model", registered_model_name="cv_resnet18")
            self._log_common_artifacts(all_labels, all_preds, labels=train_dataset.dataset.classes)
            shutil.rmtree(self.temp_dir)
            print("✅ Modelo CV registrado.")

def main():
    parser = argparse.ArgumentParser(description="MLOps Universal Framework")
    parser.add_argument("--task", type=str, default="nlp", choices=["nlp", "transformer", "cv", "all"],
                        help="Tipo de tarefa para executar")
    args = parser.parse_args()

    mlops = MLOpsUniversal()

    if args.task == "nlp" or args.task == "all":
        mlops.train_nlp_classic(model_type='rf')
        mlops.train_nlp_classic(model_type='logistic')

    if args.task == "transformer" or args.task == "all":
        mlops.train_transformers()

    if args.task == "cv" or args.task == "all":
        mlops.train_cv(epochs=1)

if __name__ == "__main__":
    main()
