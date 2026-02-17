import os
import pandas as pd
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import dagshub
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração DagsHub
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")

dagshub.init(repo_owner=repo_owner, repo_name=repo_name)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

def run_experiment():
    mlflow.set_experiment("Twitter_Sentiment_Analysis")
    
    with mlflow.start_run():
        print("--- Iniciando Experimento 3: Twitter Sentiment Analysis ---")
        
        # Carregar dados (sem header originalmente)
        data_path = r"c:\Users\pedro\Downloads\experiments\experiments\senti-pred\data\raw\twitter_validation.csv"
        df = pd.read_csv(data_path, header=None, names=['id', 'entity', 'sentiment', 'text'])
        
        # Amostra para teste rápido
        df = df.dropna(subset=['text']).sample(100, random_state=42)
        
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
        df.to_csv("twitter_results.csv", index=False)
        mlflow.log_artifact("twitter_results.csv")
        
        # Plotar distribuição
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='pred_sentiment')
        plt.title("Distribuição de Sentimentos Preditos")
        plt.savefig("sentiment_dist.png")
        mlflow.log_artifact("sentiment_dist.png")
        
        print("Experimento 3 concluido e logado no DagsHub!")

if __name__ == "__main__":
    run_experiment()
