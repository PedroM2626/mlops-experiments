import os
import random
from pathlib import Path
import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import dagshub
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from run_context import create_run_context, log_reproducibility, first_existing_path

# Configuração DagsHub
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")

dagshub.init(repo_owner=repo_owner, repo_name=repo_name)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
BASE_DIR = Path(__file__).resolve().parent

def run_experiment():
    mlflow.set_experiment("Twitter_Sentiment_Analysis")
    
    with mlflow.start_run():
        context = create_run_context(BASE_DIR, "twitter_sentiment_analysis")
        log_reproducibility(mlflow, context, SEED)
        print("--- Iniciando Experimento 3: Twitter Sentiment Analysis ---")
        
        # Carregar dados (sem header originalmente)
        data_path = first_existing_path([
            BASE_DIR / "senti-pred" / "data" / "raw" / "twitter_validation.csv",
            BASE_DIR / "senti-pred-variations" / "Senti-Pred-Remake" / "data" / "raw" / "twitter_validation.csv",
            BASE_DIR / "datasets" / "twitter_validation.csv",
        ])
        df = pd.read_csv(data_path, header=None, names=['id', 'entity', 'sentiment', 'text'])
        
        # Amostra para teste rápido
        df = df.dropna(subset=['text']).sample(100, random_state=SEED)
        
        # Modelo Zero-Shot ou Sentiment Analysis específico
        # Vamos usar um modelo pronto para predição direta
        sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        results = []
        for text in df['text']:
            # Truncar texto se for muito longo para o modelo
            res = sentiment_pipeline(text[:512])[0]
            results.append(res['label'])
            
        # Mapear labels do modelo para as labels do dataset
        # O modelo SST-2 retorna 'POSITIVE' ou 'NEGATIVE'
        # O dataset tem 'Positive', 'Negative', 'Neutral', 'Irrelevant'
        
        # Vamos simplificar para binário para métricas se necessário, 
        # ou apenas logar a distribuição das predições
        
        df['pred_sentiment'] = results
        
        # Logar distribuição
        dist = df['pred_sentiment'].value_counts().to_dict()
        for label, count in dist.items():
            mlflow.log_metric(f"count_{label}", count)
            
        # Salvar resultados
        results_path = context.artifact_dir / "twitter_results.csv"
        df.to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path))
        
        # Plotar distribuição
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='pred_sentiment')
        plt.title("Distribuição de Sentimentos Preditos")
        plot_path = context.artifact_dir / "sentiment_dist.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(str(plot_path))
        mlflow.log_param("artifact_version", context.run_id)
        
        print("Experimento 3 concluido e logado no DagsHub!")

if __name__ == "__main__":
    run_experiment()
