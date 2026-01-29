import os

# Forçar o uso de PyTorch e desativar TensorFlow no transformers
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import mlflow
import dagshub
from dotenv import load_dotenv

# Configuração DagsHub
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")

dagshub.init(repo_owner=repo_owner, repo_name=repo_name)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def run_experiment():
    mlflow.set_experiment("AG_News_Classification")
    
    with mlflow.start_run():
        print("🚀 Iniciando Experimento 1: AG News Classification")
        
        # Carregar dados
        train_path = r"c:\Users\pedro\Downloads\experiments\experiments\datasets\AG_News-train.csv"
        test_path = r"c:\Users\pedro\Downloads\experiments\experiments\datasets\AG_News-test.csv"
        
        train_df = pd.read_csv(train_path).sample(1000, random_state=42) # Subset para velocidade
        test_df = pd.read_csv(test_path).sample(200, random_state=42)
        
        # Preparar dados
        train_df['text'] = train_df['Title'] + " " + train_df['Description']
        test_df['text'] = test_df['Title'] + " " + test_df['Description']
        
        # Mapear labels (1-4 -> 0-3)
        train_df['label'] = train_df['Class Index'] - 1
        test_df['label'] = test_df['Class Index'] - 1
        
        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        test_ds = Dataset.from_pandas(test_df[['text', 'label']])
        
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        
        tokenized_train = train_ds.map(tokenize_function, batched=True)
        tokenized_test = test_ds.map(tokenize_function, batched=True)
        
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        
        training_args = TrainingArguments(
            output_dir="./results_ag_news",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_dir='./logs',
            report_to="none" # Vamos logar manualmente no MLflow
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            compute_metrics=compute_metrics,
        )
        
        # Treinar
        trainer.train()
        
        # Avaliar
        eval_results = trainer.evaluate()
        print(f"Resultados da Avaliação: {eval_results}")
        
        # Logar métricas no MLflow
        mlflow.log_metrics(eval_results)
        mlflow.log_params(training_args.to_dict())
        
        # Salvar modelo e artefatos
        model_dir = "./ag_news_model"
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        mlflow.log_artifacts(model_dir, artifact_path="model")
        
        print("✅ Experimento 1 concluído e logado no DagsHub!")

if __name__ == "__main__":
    run_experiment()
