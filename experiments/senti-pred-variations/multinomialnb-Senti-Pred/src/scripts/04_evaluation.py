"""04_evaluation.py

Avaliação modular: carrega o modelo salvo em `src/models/sentiment_model.pkl` e
os dados pré-processados (pickle em `data/processed/processed_data.pkl`) para
gerar métricas e imagens de avaliação (matriz de confusão, ROC/PR quando
disponível).
"""

from pathlib import Path
import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve

sns.set(style='whitegrid')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / 'src' / 'models' / 'sentiment_model.pkl'
PROCESSED_PATH = PROJECT_ROOT / 'data' / 'processed' / 'processed_data.pkl'
METRICS_DIR = PROJECT_ROOT / 'reports' / 'metrics'
VIS_DIR = PROJECT_ROOT / 'reports' / 'visualizacoes'
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

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

def run_evaluation():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}. Run 03_modeling.py first')
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f'Processed data not found: {PROCESSED_PATH}. Run 02_preprocessing.py first')

    model = joblib.load(MODEL_PATH)
    data = joblib.load(PROCESSED_PATH)
    
    # Para avaliação, usa o conjunto de validação se presente
    val = data.get('validation')
    if val is None or val.empty:
        print('[WARN] Validation set empty — using training set for evaluation')
        df_eval = data.get('train')
    else:
        df_eval = val
    
    # Limpeza básica
    df_eval = df_eval[df_eval['text_lemmatized'].astype(str).str.strip() != ''].copy()
    
    X_eval = df_eval['text_lemmatized'].astype(str)
    y_eval = df_eval['sentiment']

    preds = model.predict(X_eval)
    try:
        probs = model.predict_proba(X_eval)
        has_probs = True
    except Exception:
        probs = None; has_probs = False

    report = classification_report(y_eval, preds, output_dict=True)
    acc = accuracy_score(y_eval, preds)
    cm = confusion_matrix(y_eval, preds)

    # Salva matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title('Matriz de Confusão — MultinomialNB')
    plt.xlabel('Predito'); plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'evaluation_confusion_matrix.png')
    plt.close()

    # Estrutura padronizada
    metrics = {
        'best_model': 'MultinomialNB',
        'results': {
            'MultinomialNB': {
                'accuracy': acc,
                'f1_macro': report['macro avg']['f1-score'],
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }
        },
        'model_classes': model.classes_.tolist()
    }

    if has_probs and probs is not None:
        # Gráfico ROC para multiclasse
        plt.figure(figsize=(10, 6))
        for i, class_name in enumerate(model.classes_):
            # Binariza o target para a classe atual (One-vs-Rest)
            y_true_bin = (y_eval == class_name).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC={roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
        plt.title('Multiclass ROC Curves — MultinomialNB')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(VIS_DIR / 'comparison_roc.png')
        plt.close()

    # Escreve métricas json
    with (METRICS_DIR / 'model_metrics.json').open('w') as f:
        json.dump(metrics, f, indent=2)
    print(f'[OK] Evaluation metrics and images saved (see {METRICS_DIR} and {VIS_DIR})')

if __name__ == '__main__':
    run_evaluation()