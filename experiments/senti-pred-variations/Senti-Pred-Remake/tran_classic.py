import os
import random
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import mlflow
import dagshub
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from run_context import create_run_context, log_reproducibility

# Configuração MLOps
load_dotenv()
dagshub.init(repo_owner="PedroM2626", repo_name="experiments")
mlflow.set_tracking_uri("https://dagshub.com/PedroM2626/experiments.mlflow")

SEED = 42
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
RUN_CONTEXT = create_run_context(BASE_DIR, "senti_pred_classic")
DATA_DIR = BASE_DIR / "data" / "raw"

def load_data():
    cols = ['id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(os.path.join(DATA_DIR, "twitter_training.csv"), header=None, names=cols).dropna(subset=['text'])
    df_val = pd.read_csv(os.path.join(DATA_DIR, "twitter_validation.csv"), header=None, names=cols).dropna(subset=['text'])
    return df_train, df_val

def train_classic():
    mlflow.set_experiment("Twitter_Sentiment_Classic_ML")
    
    with mlflow.start_run(run_name="Logistic_Regression_TFIDF"):
        log_reproducibility(mlflow, RUN_CONTEXT, SEED)
        df_train, df_val = load_data()
        
        print("--- Vetorizando com TF-IDF ---")
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
        X_train = vectorizer.fit_transform(df_train['text'])
        X_val = vectorizer.transform(df_val['text'])
        y_train = df_train['sentiment']
        y_val = df_val['sentiment']
        
        print("--- Treinando Logistic Regression ---")
        model = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
        model.fit(X_train, y_train)
        
        print("--- Avaliando ---")
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        
        # Logs
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("artifact_version", RUN_CONTEXT.run_id)
        
        # Matriz de Confusão
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
        cm_path = RUN_CONTEXT.artifact_dir / "cm_classic.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(str(cm_path))
        
        # Salvar Artefatos
        model_path = RUN_CONTEXT.artifact_dir / "classic_model.pkl"
        vectorizer_path = RUN_CONTEXT.artifact_dir / "tfidf_vectorizer.pkl"
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(vectorizer_path))
        
        print(f"Treino Clássico Concluído! Acurácia: {acc:.4f}")

if __name__ == "__main__":
    train_classic()