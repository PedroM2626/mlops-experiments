"""03_modeling.py

Treina os modelos a partir dos dados pré-processados (pickle gerado por
`02_preprocessing.py`), calcula métricas, gera os gráficos comparativos (ROC,
PR e matrizes de confusão) e salva o melhor pipeline em
`src/models/sentiment_model.pkl`. Salva métricas detalhadas em JSON em
`reports/metrics/model_metrics.json`.
"""

from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import optuna
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize, LabelEncoder

sns.set(style='whitegrid')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
METRICS_DIR = PROJECT_ROOT / 'reports' / 'metrics'
VIS_DIR = PROJECT_ROOT / 'reports' / 'visualizacoes'
MODEL_DIR = PROJECT_ROOT / 'src' / 'models'
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class NBModelWrapper:
    def __init__(self, pipeline, le):
        self.pipeline = pipeline
        self.le = le
        self.classes_ = le.classes_
    
    def predict(self, X):
        preds_idx = self.pipeline.predict(X)
        return self.le.inverse_transform(preds_idx)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

def load_processed():
    p = PROCESSED_DIR / 'processed_data.pkl'
    if not p.exists():
        raise FileNotFoundError(f'Processed data not found: {p}. Execute 02_preprocessing.py first')
    obj = joblib.load(p)
    return obj['train'], obj.get('validation', pd.DataFrame())

def objective(trial, X_train, y_train, X_val, y_val):
    """Função objetivo para o Optuna otimizar o MultinomialNB (Otimização Nível 2)."""
    # Parâmetros do TF-IDF (Espaço expandido)
    max_features = trial.suggest_int('tfidf__max_features', 10000, 40000)
    ngram_range = trial.suggest_categorical('tfidf__ngram_range', [(1, 1), (1, 2), (1, 3)])
    use_idf = trial.suggest_categorical('tfidf__use_idf', [True, False])
    sublinear_tf = trial.suggest_categorical('tfidf__sublinear_tf', [True, False])
    min_df = trial.suggest_int('tfidf__min_df', 1, 5)
    
    # Parâmetros do MultinomialNB
    alpha = trial.suggest_float('nb__alpha', 1e-4, 1.0, log=True)
    fit_prior = trial.suggest_categorical('nb__fit_prior', [True, False])
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=max_features, 
            ngram_range=ngram_range, 
            use_idf=use_idf,
            sublinear_tf=sublinear_tf,
            min_df=min_df
        )),
        ('nb', MultinomialNB(alpha=alpha, fit_prior=fit_prior))
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    return f1_score(y_val, preds, average='macro')

def train_and_evaluate():
    df_train, df_val = load_processed()
    
    # Limpeza básica
    df_train = df_train[df_train['text_lemmatized'].astype(str).str.strip() != ''].copy()
    df_val = df_val[df_val['text_lemmatized'].astype(str).str.strip() != ''].copy()
    
    X_train = df_train['text_lemmatized'].astype(str)
    y_train_raw = df_train['sentiment']
    X_val = df_val['text_lemmatized'].astype(str)
    y_val_raw = df_val['sentiment']

    # Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    classes = le.classes_

    print(f'Iniciando otimização Nível 2 com Optuna (MultinomialNB)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30, timeout=600)
    
    print(f'Melhores hiperparâmetros: {study.best_params}')
    
    # Treina o modelo final
    best_params = study.best_params
    final_nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=best_params['tfidf__max_features'],
            ngram_range=best_params['tfidf__ngram_range'],
            use_idf=best_params['tfidf__use_idf'],
            sublinear_tf=best_params['tfidf__sublinear_tf'],
            min_df=best_params['tfidf__min_df']
        )),
        ('nb', MultinomialNB(
            alpha=best_params['nb__alpha'],
            fit_prior=best_params['nb__fit_prior']
        ))
    ])
    
    print('Treinando modelo final...')
    t0 = time.time()
    final_nb_pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    
    preds = final_nb_pipeline.predict(X_val)
    probs = final_nb_pipeline.predict_proba(X_val)
    
    # Métricas
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    report = classification_report(y_val, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_val, preds)
    
    y_val_b = label_binarize(y_val, classes=range(len(classes)))
    roc_auc = roc_auc_score(y_val_b, probs, average='macro', multi_class='ovr')
    
    wrapped_model = NBModelWrapper(final_nb_pipeline, le)
    joblib.dump(wrapped_model, MODEL_DIR / 'sentiment_model.pkl')
    print(f'[OK] Modelo MultinomialNB (wrapped) salvo em {MODEL_DIR / "sentiment_model.pkl"}')
    
    metrics_out = {
        'model_name': 'MultinomialNB',
        'best_params': best_params,
        'accuracy': acc,
        'f1_macro': f1,
        'roc_auc_macro': roc_auc,
        'train_time_seconds': train_time,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    with open(METRICS_DIR / 'model_metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)
    
    # Gráficos
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix — MultinomialNB (f1={f1:.4f})')
    plt.xlabel('Predito'); plt.ylabel('Real')
    plt.tight_layout(); plt.savefig(VIS_DIR / 'evaluation_confusion_matrix.png'); plt.close()
    
    plt.figure(figsize=(8, 6))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_val_b[:, i], probs[:, i])
        plt.plot(fpr, tpr, label=f'Class {classes[i]} (AUC={roc_auc_score(y_val_b[:, i], probs[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve — MultinomialNB')
    plt.legend(); plt.tight_layout(); plt.savefig(VIS_DIR / 'comparison_roc.png'); plt.close()
    
    print(f'[RESULT] MultinomialNB — acc={acc:.4f} f1_macro={f1:.4f}')
    return metrics_out

if __name__ == '__main__':
    train_and_evaluate()