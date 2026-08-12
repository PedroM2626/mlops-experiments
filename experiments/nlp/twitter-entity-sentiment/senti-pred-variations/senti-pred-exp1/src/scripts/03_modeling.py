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
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

def load_processed():
    p = PROCESSED_DIR / 'processed_data.pkl'
    if not p.exists():
        raise FileNotFoundError(f'Processed data not found: {p}. Execute 02_preprocessing.py first')
    obj = joblib.load(p)
    return obj['train'], obj.get('validation', pd.DataFrame())

import sys

# Custom print that also writes to a file
def log_print(message, flush=True):
    print(message, flush=flush)
    with open(PROJECT_ROOT / 'training_internal.log', 'a', encoding='utf-8') as f:
        f.write(str(message) + '\n')

def train_and_evaluate():
    # Clear internal log
    with open(PROJECT_ROOT / 'training_internal.log', 'w', encoding='utf-8') as f:
        f.write('Starting training session...\n')
    
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

    # Combine into dataframes for AutoML frameworks that prefer it
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)

    manual_models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'MultinomialNB': MultinomialNB(),
        'LinearSVC': LinearSVC(max_iter=2000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=20, random_state=42),
        'MLP': MLPClassifier(max_iter=50, hidden_layer_sizes=(30,), random_state=42)
    }

    results = {}
    
    # MLflow setup
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_experiment("Sentiment_Analysis_Experiment")

    # 1. Manual Trainings
    results = {}
    for name, clf in manual_models.items():
        try:
            with mlflow.start_run(run_name=f"Manual_{name}"):
                log_print(f'Training Manual {name}...', flush=True)
                pipe = Pipeline([
                    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,1))),
                    ('clf', clf)
                ])
                log_print(f"Fitting {name}...", flush=True)
                t0 = time.time()
                pipe.fit(X_train, y_train)
                train_time = time.time() - t0
                
                t0p = time.time()
                preds = pipe.predict(X_val)
                predict_time = time.time() - t0p
        
                acc = accuracy_score(y_val, preds)
                f1 = f1_score(y_val, preds, average='macro')
                
                # Log to MLflow
                mlflow.log_param("model_type", "manual")
                mlflow.log_param("algorithm", name)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_macro", f1)
                mlflow.log_metric("train_time", train_time)
                mlflow.sklearn.log_model(pipe, "model")
        
                results[name] = {
                    'pipeline': pipe,
                    'accuracy': acc,
                    'f1_macro': f1,
                    'train_time_seconds': train_time,
                    'training_type': 'manual',
                    'y_score': get_y_score(pipe, X_val, y_val)
                }
                log_print(f'[RESULT] {name} — acc={acc:.4f} f1_macro={f1:.4f}', flush=True)
        except Exception as e:
            log_print(f'[ERROR] Manual model {name} failed: {e}', flush=True)
            import traceback
            log_print(traceback.format_exc(), flush=True)
    
    # 2. AutoML Trainings
    automl_frameworks = ['FLAML', 'TPOT', 'LightAutoML', 'PyCaret', 'H2O', 'AutoGluon', 'Auto-sklearn']
    
    for framework in automl_frameworks:
        try:
            log_print(f'\n--- Starting AutoML with {framework} ---', flush=True)
            t0 = time.time()
            model, acc, f1, y_score = train_automl(framework, X_train, y_train, X_val, y_val)
            train_time = time.time() - t0
            
            if model is not None:
                with mlflow.start_run(run_name=f"AutoML_{framework}"):
                    mlflow.log_param("model_type", "automl")
                    mlflow.log_param("framework", framework)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("f1_macro", f1)
                    mlflow.log_metric("train_time", train_time)
                    
                    # Log model artifact
                    model_path = MODEL_DIR / f"model_{framework}.joblib"
                    joblib.dump(model, model_path)
                    mlflow.log_artifact(str(model_path))
                    
                    results[framework] = {
                        'pipeline': model,
                        'accuracy': acc,
                        'f1_macro': f1,
                        'train_time_seconds': train_time,
                        'training_type': framework.lower(),
                        'y_score': y_score
                    }
                    log_print(f'[RESULT] {framework} — acc={acc:.4f} f1_macro={f1:.4f} time={train_time:.2f}s', flush=True)
            else:
                log_print(f'[SKIP] {framework} returned no model (likely not installed or incompatible).', flush=True)
        except Exception as e:
            log_print(f'[ERROR] {framework} failed: {e}', flush=True)

    if not results:
        log_print("No models were trained successfully.")
        return

    # choose best
    best_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_data = results[best_name]
    best_pipeline = best_data['pipeline']
    
    # Save best model
    joblib.dump(best_pipeline, MODEL_DIR / 'sentiment_model.pkl')
    log_print(f'[OK] Best model: {best_name} saved to {MODEL_DIR / "sentiment_model.pkl"}')

    # Save metrics and ranking
    save_metrics_and_ranking(results, best_name)
    log_print("\n[SUCCESS] Pipeline de treinamento finalizado com sucesso!")
    
    # Visualizations
    generate_visualizations(results, y_val, best_name)

    return results

def get_y_score(pipe, X_val, y_val):
    classes = np.unique(y_val)
    y_val_b = label_binarize(y_val, classes=classes)
    try:
        return pipe.predict_proba(X_val)
    except:
        try:
            decision = pipe.decision_function(X_val)
            if decision.ndim == 1:
                decision = np.vstack([-decision, decision]).T
            return decision
        except:
            return None

def train_automl(framework, X_train, y_train, X_val, y_val):
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)
    
    # Encode labels for frameworks that require numeric targets
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    
    time_budget = 60 # 1 minute

    try:
        if framework == 'FLAML':
            from flaml import AutoML
            log_print(f"Running FLAML with budget {time_budget}s...", flush=True)
            automl = AutoML()
            automl.fit(X_train_vec, y_train_enc, task="classification", time_budget=time_budget, verbose=0)
            log_print("FLAML finished fitting.", flush=True)
            preds_enc = automl.predict(X_val_vec)
            preds = le.inverse_transform(preds_enc)
            pipe = Pipeline([('tfidf', tfidf), ('clf', automl)])
            y_score = None
            try:
                y_score = automl.predict_proba(X_val_vec)
            except:
                pass
            return pipe, accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), y_score
            
        elif framework == 'TPOT':
            from tpot import TPOTClassifier
            log_print("Starting TPOT fit...", flush=True)
            # Increased time and population for better chance of finding a pipeline
            tpot = TPOTClassifier(generations=3, population_size=15, verbosity=2, random_state=42, max_time_mins=5)
            tpot.fit(X_train_vec.toarray(), y_train_enc)
            
            if not hasattr(tpot, 'fitted_pipeline_'):
                log_print("TPOT could not find a valid pipeline in the allotted time.", flush=True)
                return None, 0, 0, None
                
            log_print("TPOT finished fitting.", flush=True)
            preds_enc = tpot.predict(X_val_vec.toarray())
            preds = le.inverse_transform(preds_enc)
            pipe = Pipeline([('tfidf', tfidf), ('clf', tpot.fitted_pipeline_)])
            y_score = None
            try:
                y_score = tpot.predict_proba(X_val_vec.toarray())
            except:
                pass
            return pipe, accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), y_score
            
        elif framework == 'PyCaret':
            try:
                log_print("Starting PyCaret...", flush=True)
                from pycaret.classification import setup, compare_models, finalize_model, predict_model
                import pandas as pd
                # PyCaret might struggle with many columns from TFIDF in some versions
                train_df = pd.DataFrame(X_train_vec.toarray())
                train_df['target'] = y_train_enc
                s = setup(train_df, target='target', verbose=False, html=False, session_id=42)
                best = compare_models(budget_time=time_budget, verbose=False)
                final_model = finalize_model(best)
                
                val_df = pd.DataFrame(X_val_vec.toarray())
                preds_enc = predict_model(final_model, data=val_df)['prediction_label']
                preds = le.inverse_transform(preds_enc)
                
                pipe = Pipeline([('tfidf', tfidf), ('clf', final_model)])
                y_score = None
                try:
                    y_score_df = predict_model(final_model, data=val_df, raw_score=True)
                    y_score = y_score_df[[c for c in y_score_df.columns if c.startswith('prediction_score_')]].values
                except:
                    pass
                return pipe, accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), y_score
            except Exception as e:
                log_print(f"PyCaret error: {e}", flush=True)
                return None, 0, 0, None
            
        elif framework == 'H2O':
            try:
                log_print("Starting H2O...", flush=True)
                import h2o
                from h2o.automl import H2OAutoML
                # Check if java is available
                if os.system("java -version") != 0:
                    log_print("H2O requires Java but it is not found.", flush=True)
                    return None, 0, 0, None
                
                h2o.init()
                train_h2o = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
                val_h2o = h2o.H2OFrame(pd.concat([X_val, y_val], axis=1))
                
                target = y_train.name if hasattr(y_train, 'name') else 'sentiment'
                features = [X_train.name] if hasattr(X_train, 'name') else X_train.columns.tolist() if hasattr(X_train, 'columns') else [X_train.name if hasattr(X_train, 'name') else 'text_lemmatized']
                
                aml = H2OAutoML(max_runtime_secs=time_budget, seed=42)
                aml.train(x=features, y=target, training_frame=train_h2o)
                
                preds_h2o = aml.leader.predict(val_h2o)
                preds = preds_h2o['predict'].as_data_frame()
                
                y_score = preds_h2o.as_data_frame().iloc[:, 1:].values
                return aml.leader, accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), y_score
            except Exception as e:
                log_print(f"H2O error: {e}", flush=True)
                return None, 0, 0, None

        elif framework == 'LightAutoML':
            try:
                log_print("Starting LightAutoML...", flush=True)
                from lightautoml.automl.presets.tabular_presets import TabularAutoML
                from lightautoml.tasks import Task
                import pandas as pd
                
                # LightAutoML works better with its own encoding or numeric targets
                train_data = pd.DataFrame({'text': X_train, 'target': y_train_enc})
                task = Task('multiclass')
                roles = {'target': 'target', 'text': ['text']}
                
                automl = TabularAutoML(task=task, timeout=time_budget)
                oof_pred = automl.fit_predict(train_data, roles=roles, verbose=0)
                
                val_data = pd.DataFrame({'text': X_val})
                preds_proba = automl.predict(val_data).data
                preds_enc = np.argmax(preds_proba, axis=1)
                preds = le.inverse_transform(preds_enc)
                
                return automl, accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), preds_proba
            except Exception as e:
                log_print(f"LightAutoML error: {e}", flush=True)
                return None, 0, 0, None

        elif framework == 'AutoGluon':
            try:
                from autogluon.tabular import TabularPredictor
                import pandas as pd
                train_data = pd.DataFrame({'text': X_train, 'target': y_train})
                predictor = TabularPredictor(label='target', verbosity=0).fit(train_data, time_limit=time_budget)
                
                val_data = pd.DataFrame({'text': X_val})
                preds = predictor.predict(val_data)
                preds_proba = predictor.predict_proba(val_data).values
                
                return predictor, accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), preds_proba
            except Exception as e:
                log_print(f"AutoGluon error: {e}")
                return None, 0, 0, None

        elif framework == 'Auto-sklearn':
            if os.name == 'nt':
                log_print("Auto-sklearn is not supported on Windows.")
                return None, 0, 0, None
            import autosklearn.classification
            cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=time_budget, per_run_time_limit=10)
            cls.fit(X_train_vec.toarray(), y_train)
            preds = cls.predict(X_val_vec.toarray())
            preds_proba = cls.predict_proba(X_val_vec.toarray())
            pipe = Pipeline([('tfidf', tfidf), ('clf', cls)])
            return pipe, accuracy_score(y_val, preds), f1_score(y_val, preds, average='macro'), preds_proba
            
    except ImportError:
        log_print(f"Framework {framework} not installed.")
    except Exception as e:
        log_print(f"Error training {framework}: {e}")
        
    return None, 0, 0, None

def save_metrics_and_ranking(results, best_name):
    metrics_out = {
        'best_model': best_name,
        'ranking': [],
        'results': {}
    }
    
    # Create ranking based on f1_macro and train_time
    sorted_models = sorted(results.items(), key=lambda x: (-x[1]['f1_macro'], x[1]['train_time_seconds']))
    
    for i, (name, data) in enumerate(sorted_models):
        metrics_out['ranking'].append({
            'rank': i + 1,
            'model': name,
            'f1_macro': data['f1_macro'],
            'accuracy': data['accuracy'],
            'train_time_seconds': data['train_time_seconds'],
            'training_type': data['training_type']
        })
        
        metrics_out['results'][name] = {
            'accuracy': data['accuracy'],
            'f1_macro': data['f1_macro'],
            'train_time_seconds': data['train_time_seconds'],
            'training_type': data['training_type']
        }

    import json
    (METRICS_DIR / 'model_metrics.json').write_text(json.dumps(metrics_out, indent=2))
    log_print(f'[OK] Metrics and ranking saved to {METRICS_DIR / "model_metrics.json"}')

def generate_visualizations(results, y_val, best_name):
    classes_all = np.unique(y_val)
    y_val_b_all = label_binarize(y_val, classes=classes_all)

    # 1. ROC Curve comparison
    plt.figure(figsize=(10, 8))
    for name, data in results.items():
        ys = data.get('y_score')
        if ys is not None:
            try:
                fpr, tpr, _ = roc_curve(y_val_b_all.ravel(), ys.ravel())
                plt.plot(fpr, tpr, lw=2, label=f"{name} (f1={data['f1_macro']:.3f})")
            except:
                continue
    plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(VIS_DIR / 'comparison_roc.png'); plt.close()

    # 2. Confusion Matrix for BEST model
    best_pipe = results[best_name]['pipeline']
    # Use the original X_val (text) because the pipeline includes TFIDF
    df_train, df_val = load_processed()
    X_val_raw = df_val['text_lemmatized'].astype(str)
    mask_val = X_val_raw.str.strip().replace('', np.nan).notna()
    X_val_raw = X_val_raw[mask_val]
    
    try:
        preds = best_pipe.predict(X_val_raw)
        cm = confusion_matrix(y_val, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes_all, yticklabels=classes_all)
        plt.title(f'Confusion Matrix - BEST MODEL: {best_name}')
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(VIS_DIR / 'best_model_confusion.png'); plt.close()
    except Exception as e:
        log_print(f"Could not generate confusion matrix for best model: {e}")

    # 3. Score vs Time Ranking Plot
    names = list(results.keys())
    scores = [results[n]['f1_macro'] for n in names]
    times = [results[n]['train_time_seconds'] for n in names]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(names, scores, color='skyblue', label='F1 Macro')
    ax1.set_ylabel('F1 Macro Score', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticklabels(names, rotation=45)
    
    ax2 = ax1.twinx()
    ax2.plot(names, times, color='red', marker='o', label='Train Time')
    ax2.set_ylabel('Train Time (s)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Model Performance vs Training Time')
    fig.tight_layout()
    plt.savefig(VIS_DIR / 'score_vs_time.png'); plt.close()

if __name__ == '__main__':
    train_and_evaluate()
