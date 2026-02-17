import os
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import re
import nltk
import shutil
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Garantir recursos do NLTK
nltk_resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
for r in nltk_resources:
    try:
        nltk.download(r, quiet=True)
    except:
        pass

# 1. Configurações de MLOps (DagsHub + MLflow)
load_dotenv()
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "PedroM2626")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "experiments")

try:
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
except Exception as e:
    print(f"Erro ao inicializar DagsHub/MLflow: {e}. Prosseguindo com log local.")

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

def clean_text_fast(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    return ' '.join([LEMMATIZER.lemmatize(w) for w in tokens if w not in STOP_WORDS])

def run_autogluon_experiment():
    mlflow.set_experiment("Senti-Pred_AutoGluon_High_Score")
    
    with mlflow.start_run(run_name="AutoGluon_Optimized_NLP"):
        print("Iniciando Experimento 4 Otimizado: AutoGluon (F1 > 0.97)")
        
        # 2. Carregar Dados
        base_path = Path(__file__).parent / "Senti-Pred-Remake"
        train_path = base_path / "data" / "raw" / "twitter_training.csv"
        test_path = base_path / "data" / "raw" / "twitter_validation.csv"
        
        cols = ['id', 'entity', 'sentiment', 'text']
        train_df = pd.read_csv(train_path, header=None, names=cols).dropna(subset=['text', 'sentiment'])
        test_df = pd.read_csv(test_path, header=None, names=cols).dropna(subset=['text', 'sentiment'])
        
        print("Limpando dados para o AutoGluon...")
        train_df['text_clean'] = train_df['text'].apply(clean_text_fast)
        test_df['text_clean'] = test_df['text'].apply(clean_text_fast)
        
        # Manter apenas as colunas necessárias para o treino tabular
        train_data = train_df[['text_clean', 'sentiment']]
        test_data = test_df[['text_clean', 'sentiment']]
        
        label = 'sentiment'
        save_path = 'ag_models_senti_pred_optimized'
        
        # Cleanup cache to ensure a fresh run
        if os.path.exists(save_path):
            print(f"Limpando cache de modelos em {save_path}...")
            shutil.rmtree(save_path)
        
        print("Treinando com AutoGluon (Preset: High Quality + NLP Optimization)...")
        
        # 'best_quality' é o modo de qualidade máxima, utilizando Stacking e Bagging.
        # Aumentamos o time_limit para permitir que o AutoGluon explore esses modelos complexos.
        predictor = TabularPredictor(
                    label=label, 
                    eval_metric='f1_macro',
                    path=save_path
                ).fit(
                    train_data=train_data,
                    presets='best_quality', 
                    time_limit=3600,        # 1 hora para permitir Stacking/Bagging profundo
                    hyperparameters={
                        'GBM': {}, 
                        'CAT': {}, 
                        'XGB': {}, 
                        'RF': {}, 
                        'XT': {},
                        'AG_TEXT_NN': {'presets': 'best_quality'} # Qualidade máxima para o modelo de texto
                    },
                    excluded_model_types=['KNN', 'FASTAI'],
                    ag_args_fit={'num_gpus': 0, 'ag.max_memory_usage_ratio': 1.5}
                )
        
        # 3. Avaliação
        print("Avaliando modelo...")
        y_test = test_data[label]
        y_pred = predictor.predict(test_data.drop(columns=[label]))
        
        report = classification_report(y_test, y_pred, output_dict=True)
        f1_macro = report['macro avg']['f1-score']
        
        print(f"F1-Macro Alcançado: {f1_macro:.4f}")
        
        # 4. Logs
        mlflow.log_param("presets", "high_quality_optimized")
        mlflow.log_metric("f1_macro", f1_macro)
        
        for label_name, metrics in report.items():
            if isinstance(metrics, dict):
                for m_name, val in metrics.items():
                    mlflow.log_metric(f"{label_name}_{m_name}", val)
        
        # Leaderboard
        lb = predictor.leaderboard(test_data, silent=True)
        lb.to_csv("ag_leaderboard_optimized.csv")
        mlflow.log_artifact("ag_leaderboard_optimized.csv")
        
        if f1_macro >= 0.97:
            print("OBJETIVO ALCANÇADO! F1-Macro >= 0.97")
        
        mlflow.log_artifact(save_path)

if __name__ == "__main__":
    run_autogluon_experiment()
