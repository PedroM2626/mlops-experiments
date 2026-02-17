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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc
)
from sklearn.preprocessing import LabelEncoder

sns.set(style='whitegrid')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
METRICS_DIR = PROJECT_ROOT / 'reports' / 'metrics'
VIS_DIR = PROJECT_ROOT / 'reports' / 'visualizacoes'
MODEL_DIR = PROJECT_ROOT / 'src' / 'models'
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class RFModelWrapper:
    """Wrapper para o pipeline do Random Forest com suporte ao LabelEncoder."""
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
    """Objetivo do Optuna para otimizar Random Forest (Nível 2)."""
    # Hiperparâmetros do TF-IDF
    max_features = trial.suggest_int('tfidf__max_features', 5000, 25000)
    ngram_range = trial.suggest_categorical('tfidf__ngram_range', [(1, 1), (1, 2), (1, 3)])
    use_idf = trial.suggest_categorical('tfidf__use_idf', [True, False])

    # Hiperparâmetros do Random Forest (Espaço expandido)
    n_estimators = trial.suggest_int('rf__n_estimators', 100, 500)
    max_depth = trial.suggest_categorical('rf__max_depth', [None, 30, 50, 70, 100])
    min_samples_split = trial.suggest_int('rf__min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('rf__min_samples_leaf', 1, 10)
    max_features_rf = trial.suggest_categorical('rf__max_features', ['sqrt', 'log2'])
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, use_idf=use_idf)),
        ('rf', RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features_rf,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    return f1_score(y_val, preds, average='macro')

def train_and_evaluate():
    df_train, df_val = load_processed()
    
    # Limpeza básica e remoção de nulos
    df_train = df_train[df_train['text_lemmatized'].astype(str).str.strip() != ''].copy()
    df_val = df_val[df_val['text_lemmatized'].astype(str).str.strip() != ''].copy()

    X_train = df_train['text_lemmatized'].astype(str)
    y_train_raw = df_train['sentiment']
    X_val = df_val['text_lemmatized'].astype(str)
    y_val_raw = df_val['sentiment']

    # Codificação dos rótulos
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    print('Iniciando otimização com Optuna (Random Forest - Nível 2)...')
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30, timeout=600)

    print(f'Melhores hiperparâmetros: {study.best_params}')

    # Treinar modelo final com os melhores parâmetros
    best_params = study.best_params
    final_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=best_params['tfidf__max_features'],
            ngram_range=best_params['tfidf__ngram_range'],
            use_idf=best_params['tfidf__use_idf']
        )),
        ('rf', RandomForestClassifier(
            n_estimators=best_params['rf__n_estimators'],
            max_depth=best_params['rf__max_depth'],
            min_samples_split=best_params['rf__min_samples_split'],
            min_samples_leaf=best_params['rf__min_samples_leaf'],
            max_features=best_params['rf__max_features'],
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])

    print('Treinando modelo final...')
    t0 = time.time()
    final_pipeline.fit(X_train, y_train)
    train_time = time.time() - t0

    # Avaliação
    t0p = time.time()
    preds_idx = final_pipeline.predict(X_val)
    predict_time = time.time() - t0p
    
    preds = le.inverse_transform(preds_idx)
    y_val_orig = le.inverse_transform(y_val)

    acc = accuracy_score(y_val_orig, preds)
    f1 = f1_score(y_val_orig, preds, average='macro')
    report = classification_report(y_val_orig, preds, output_dict=True)
    cm = confusion_matrix(y_val_orig, preds)

    # Salvar modelo embrulhado (wrapped)
    wrapped_model = RFModelWrapper(final_pipeline, le)
    joblib.dump(wrapped_model, MODEL_DIR / 'sentiment_model.pkl')
    print(f'[OK] Modelo Random Forest (wrapped) salvo em {MODEL_DIR / "sentiment_model.pkl"}')
    print(f'[RESULT] Random Forest — acc={acc:.4f} f1_macro={f1:.4f}')

    # Salvar métricas JSON
    metrics_out = {
        'best_model': 'Random Forest',
        'results': {
            'Random Forest': {
                'accuracy': acc,
                'f1_macro': f1,
                'train_time_seconds': train_time,
                'predict_time_seconds': predict_time,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'best_params': best_params
            }
        },
        'model_classes': le.classes_.tolist()
    }
    
    with (METRICS_DIR / 'model_metrics.json').open('w') as f:
        json.dump(metrics_out, f, indent=2)
    print(f'[OK] Métricas salvas em {METRICS_DIR / "model_metrics.json"}')

    # Matriz de Confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Matriz de Confusão — Random Forest')
    plt.xlabel('Predito'); plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'evaluation_confusion_matrix.png')
    plt.close()

    # Curva ROC (Multiclasse One-vs-Rest)
    probs = final_pipeline.predict_proba(X_val)
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(le.classes_):
        y_true_bin = (y_val == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC={roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
    plt.title('Multiclass ROC Curves — Random Forest')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'comparison_roc.png')
    plt.close()

    # Curva Precision-Recall (Multiclasse One-vs-Rest)
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(le.classes_):
        y_true_bin = (y_val == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_bin, probs[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f'{class_name} (AUC={pr_auc:.3f})')
    
    plt.title('Multiclass Precision-Recall Curves — Random Forest')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'comparison_pr.png')
    plt.close()

if __name__ == '__main__':
    train_and_evaluate()