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

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

sns.set(style='whitegrid')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
METRICS_DIR = PROJECT_ROOT / 'reports' / 'metrics'
VIS_DIR = PROJECT_ROOT / 'reports' / 'visualizacoes'
MODEL_DIR = PROJECT_ROOT / 'src' / 'models'
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_processed():
    p = PROCESSED_DIR / 'processed_data.pkl'
    if not p.exists():
        raise FileNotFoundError(f'Processed data not found: {p}. Execute 02_preprocessing.py first')
    obj = joblib.load(p)
    return obj['train'], obj.get('validation', pd.DataFrame())

def train_and_evaluate():
    df_train, df_val = load_processed()
    X_train = df_train['text_lemmatized'].astype(str)
    y_train = df_train['sentiment']
    X_val = df_val['text_lemmatized'].astype(str)
    y_val = df_val['sentiment']

    # remove empty
    mask_train = X_train.str.strip().replace('', np.nan).notna()
    mask_val = X_val.str.strip().replace('', np.nan).notna()
    X_train = X_train[mask_train]; y_train = y_train[mask_train]
    X_val = X_val[mask_val]; y_val = y_val[mask_val]

    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'MultinomialNB': MultinomialNB(),
        'LinearSVC': LinearSVC(max_iter=20000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(max_iter=100, hidden_layer_sizes=(50,), random_state=42)
    }

    results = {}
    for name, clf in models.items():
        print(f'Training {name}...')
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1,2))),
            ('clf', clf)
        ])
        t0 = time.time(); pipe.fit(X_train, y_train); t1 = time.time()
        train_time = t1 - t0
        t0p = time.time(); preds = pipe.predict(X_val); t1p = time.time(); predict_time = t1p - t0p

        acc = accuracy_score(y_val, preds)
        f1 = f1_score(y_val, preds, average='macro')
        report = classification_report(y_val, preds, output_dict=True)
        cm = confusion_matrix(y_val, preds)

        # scores for ROC/PR
        classes = np.unique(y_val)
        y_val_b = label_binarize(y_val, classes=classes)
        y_score = None
        try:
            y_score = pipe.predict_proba(X_val)
        except Exception:
            try:
                decision = pipe.decision_function(X_val)
                if decision.ndim == 1:
                    decision = np.vstack([-decision, decision]).T
                y_score = decision
            except Exception:
                y_score = None

        roc_auc_macro = None; avg_precision_macro = None
        if y_score is not None and y_score.shape[1] == y_val_b.shape[1]:
            try:
                roc_auc_macro = roc_auc_score(y_val_b, y_score, average='macro', multi_class='ovr')
            except Exception:
                roc_auc_macro = None
            try:
                avg_precision_macro = average_precision_score(y_val_b, y_score, average='macro')
            except Exception:
                avg_precision_macro = None

        results[name] = {
            'pipeline': pipe,
            'accuracy': acc,
            'f1_macro': f1,
            'roc_auc_macro': roc_auc_macro,
            'average_precision_macro': avg_precision_macro,
            'train_time_seconds': train_time,
            'predict_time_seconds': predict_time,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'y_score': y_score
        }
        print(f'[RESULT] {name} — acc={acc:.4f} f1_macro={f1:.4f}')

    # choose best
    best = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_pipeline = results[best]['pipeline']
    joblib.dump(best_pipeline, MODEL_DIR / 'sentiment_model.pkl')
    print(f'[OK] Best model: {best} saved to {MODEL_DIR / "sentiment_model.pkl"}')

    # save metrics JSON
    metrics_out = {'best_model': best, 'results': {}}
    for k in results:
        metrics_out['results'][k] = {
            'accuracy': results[k]['accuracy'],
            'f1_macro': results[k]['f1_macro'],
            'roc_auc_macro': results[k].get('roc_auc_macro'),
            'average_precision_macro': results[k].get('average_precision_macro'),
            'train_time_seconds': results[k].get('train_time_seconds'),
            'predict_time_seconds': results[k].get('predict_time_seconds'),
            'classification_report': results[k]['report'],
            'confusion_matrix': results[k]['confusion_matrix']
        }
    import json
    out_file = METRICS_DIR / 'model_metrics.json'
    out_file.write_text(json.dumps(metrics_out, indent=2))
    print(f'[OK] Metrics saved to {out_file.absolute()}')

    # comparison plots (ROC, PR, confusion side-by-side)
    classes_all = np.unique(y_val)
    y_val_b_all = label_binarize(y_val, classes=classes_all)

    # ROC
    plt.figure(figsize=(8, 6))
    any_plot = False
    for name in results:
        ys = results[name].get('y_score')
        if ys is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_val_b_all.ravel(), ys.ravel())
            auc_val = results[name].get('roc_auc_macro')
            label = name + (f' (AUC={auc_val:.3f})' if auc_val is not None else '')
            plt.plot(fpr, tpr, lw=2, label=label)
            any_plot = True
        except Exception:
            continue
    if any_plot:
        plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Comparative ROC Curves (all models)')
        plt.legend(loc='lower right'); plt.tight_layout(); plt.savefig(VIS_DIR / 'comparison_roc.png'); plt.close()

    # PR
    plt.figure(figsize=(8, 6))
    any_plot = False
    for name in results:
        ys = results[name].get('y_score')
        if ys is None:
            continue
        try:
            precision, recall, _ = precision_recall_curve(y_val_b_all.ravel(), ys.ravel())
            ap = results[name].get('average_precision_macro')
            label = name + (f' (AP={ap:.3f})' if ap is not None else '')
            plt.plot(recall, precision, lw=2, label=label)
            any_plot = True
        except Exception:
            continue
    if any_plot:
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Comparative Precision-Recall Curves (all models)')
        plt.legend(loc='lower left'); plt.tight_layout(); plt.savefig(VIS_DIR / 'comparison_pr.png'); plt.close()

    # Confusion matrices side-by-side
    model_names = list(results.keys())
    cms = [np.array(results[nm]['confusion_matrix']) for nm in model_names]
    if cms:
        vmax = max(cm.max() for cm in cms)
        fig, axes = plt.subplots(1, len(model_names), figsize=(6 * len(model_names), 5))
        if len(model_names) == 1:
            axes = [axes]
        for ax, nm, cm in zip(axes, model_names, cms):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_all, yticklabels=classes_all, vmin=0, vmax=vmax, ax=ax)
            ax.set_title(f'Confusion — {nm}'); ax.set_xlabel('Predito'); ax.set_ylabel('Real')
        plt.tight_layout(); plt.savefig(VIS_DIR / 'comparison_confusion_matrices.png'); plt.close()

    return results

if __name__ == '__main__':
    train_and_evaluate()