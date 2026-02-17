import pandas as pd
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
from flaml import AutoML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import sys

# Adicionar src ao path para importar o preprocessador
sys.path.append(str(Path(__file__).parent.parent))
from data.preprocess import clean_text

# Carregar variáveis de ambiente
load_dotenv()

def train_flaml_fast():
    """
    Treina o modelo usando FLAML com configurações otimizadas para velocidade e estabilidade.
    Tempo limite: 10 minutos (600 segundos).
    Foco em ExtraTrees e LGBM sem ensemble (evita travamentos no final).
    """
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / os.getenv('DATA_RAW_PATH', 'data/raw')
    models_dir = project_root / os.getenv('MODELS_PATH', 'models')
    
    # Criar diretório de modelos se não existir
    models_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = raw_dir / 'twitter_training.csv'
    val_path = raw_dir / 'twitter_validation.csv'
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Dados brutos não encontrados em {raw_dir}")
    
    columns = ['id', 'topic', 'sentiment', 'text']
    
    print("Carregando dados brutos originais...")
    train_df = pd.read_csv(train_path, names=columns, header=None)
    val_df = pd.read_csv(val_path, names=columns, header=None)
    
    print("Limpando dados...")
    train_df = train_df.dropna(subset=['text', 'sentiment'])
    val_df = val_df.dropna(subset=['text', 'sentiment'])
    
    print("Processando textos (limpeza e lemmatização)...")
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    val_df['cleaned_text'] = val_df['text'].apply(clean_text)
    
    # Remover linhas vazias
    train_df = train_df[train_df['cleaned_text'] != ""]
    val_df = val_df[val_df['cleaned_text'] != ""]
    
    # Vetorização TF-IDF Otimizada (N-grams 1-2, 30k features)
    print("Vetorizando textos (30k features, n-grams 1-2)...")
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents='unicode',
        min_df=5,
        analyzer='word',
        token_pattern=r'\w{1,}'
    )
    
    X_train = vectorizer.fit_transform(train_df['cleaned_text'])
    
    # Codificar labels
    print("Codificando labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(train_df['sentiment'])
    
    X_val = vectorizer.transform(val_df['cleaned_text'])
    y_val = le.transform(val_df['sentiment'])
    
    # Configuração do FLAML Turbo
    automl = AutoML()
    
    automl_settings = {
        "time_budget": 300, # Reduzido para 5 minutos para dar resposta rápida
        "metric": 'accuracy',
        "task": 'classification',
        "log_file_name": "flaml_training.log",
        "estimator_list": ['lgbm', 'rf', 'extra_tree'], 
        "ensemble": False,
        "seed": 42,
        "n_jobs": -1
    }
    
    print(f"Iniciando treinamento FLAML Turbo (Time budget: {automl_settings['time_budget']}s)...")
    automl.fit(X_train=X_train, y_train=y_train, **automl_settings)
    
    print("\n--- Treinamento Concluído! ---")
    
    # Avaliação
    print("\nAvaliando o melhor modelo encontrado pelo FLAML...")
    y_pred = automl.predict(X_val)
    
    print("\nResultados da Validação:")
    print(f"Melhor Estimador: {automl.best_estimator}")
    print(f"Melhor Acurácia de Validação: {accuracy_score(y_val, y_pred):.4f}")
    print("\nRelatório de Classificação:")
    # Usar target_names para o relatório ficar legível
    print(classification_report(y_val, y_pred, target_names=le.classes_))
    
    # Salvar artefatos
    print(f"Salvando artefatos em {models_dir}...")
    joblib.dump(automl, models_dir / 'flaml_model.pkl')
    joblib.dump(vectorizer, models_dir / 'tfidf_vectorizer.pkl')
    joblib.dump(le, models_dir / 'label_encoder.pkl')
    
    # Salvar métricas em JSON para o resumo
    import json
    metrics = {
        "best_estimator": automl.best_estimator,
        "accuracy": accuracy_score(y_val, y_pred),
        "report": classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    }
    
    metrics_path = project_root / 'reports/metrics/flaml_optimized_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    train_flaml_fast()
