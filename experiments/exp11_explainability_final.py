"""
Experimento 11: Explicabilidade com LIME (versao sem SHAP)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import mlflow
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
import lime
import lime.lime_text

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent

def run_explainability_pipeline():
    print("\n" + "="*80)
    print("EXPERIMENTO 11: EXPLICABILIDADE COM LIME")
    print("="*80 + "\n")
    
    mlflow.set_experiment("Explainability")
    
    with mlflow.start_run(run_name="explainability_lime"):
        
        print("1 Gerando dataset...")
        
        texts = [
            "This product is excellent and amazing",
            "I love this item, highly recommend",
            "Terrible quality, very disappointed",
            "Awful experience, waste of money",
            "Great customer service and delivery",
            "Poor packaging, item damaged",
        ] * 100
        
        labels = [1, 1, 0, 0, 1, 0] * 100
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        print(f"   Total: {len(df)} textos\n")
        
        results = {
            'model_info': {
                'total_samples': len(df),
                'classes': [0, 1]
            },
            'lime_analysis': {},
            'feature_importance': {}
        }
        
        print("2 Treinando modelo...")
        
        le = LabelEncoder()
        y = le.fit_transform(df['label'])
        
        tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        X_tfidf = tfidf.fit_transform(df['text'])
        
        voting = VotingClassifier([
            ('lr1', LogisticRegression(random_state=SEED, max_iter=200)),
            ('lr2', LogisticRegression(C=0.5, random_state=SEED, max_iter=200))
        ], voting='soft')
        
        voting.fit(X_tfidf, y)
        
        y_pred = voting.predict(X_tfidf)
        accuracy = (y_pred == y).mean()
        
        print(f"   Acuracia: {accuracy:.3f}\n")
        results['model_info']['accuracy'] = float(accuracy)
        results['model_info']['n_features'] = X_tfidf.shape[1]

        def predict_proba_texts(texts):
            return voting.predict_proba(tfidf.transform(texts))
        
        print("3 Analisando exemplos com LIME...")
        
        explainer_lime = lime.lime_text.LimeTextExplainer(
            class_names=['Negative', 'Positive'],
            verbose=False
        )
        
        lime_results = []
        example_indices = [0, len(df)//2, len(df)-1]
        
        for idx in example_indices:
            text = df.iloc[idx]['text']
            true_label = df.iloc[idx]['label']
            
            exp = explainer_lime.explain_instance(
                text_instance=text,
                classifier_fn=predict_proba_texts,
                num_features=5
            )
            
            exp_list = exp.as_list()
            pred_label = voting.predict(tfidf.transform([text]))[0]
            
            lime_results.append({
                'text': text[:50],
                'true_label': int(true_label),
                'pred_label': int(pred_label),
                'correct': int(true_label) == int(pred_label),
                'explanation': [{'word': w, 'weight': float(wt)} for w, wt in exp_list]
            })
            
            print(f"   Exemplo: {text[:40]}...")
            print(f"      Real: {true_label}, Pred: {pred_label}, Correto: {true_label == pred_label}")
        
        results['lime_analysis']['examples'] = lime_results
        
        print("\n4 Feature Importance (Coeficientes)...")
        
        feature_names = tfidf.get_feature_names_out()
        feature_imp_lr = np.abs(voting.estimators_[0].coef_[0])
        top_features_lr = sorted(
            zip(feature_names, feature_imp_lr),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        print("   Top 10 features:")
        for i, (feat, imp) in enumerate(top_features_lr, 1):
            print(f"      {i:2d}. {feat:15s} -> {imp:.4f}")
        
        results['feature_importance']['logistic_regression'] = [
            {'feature': feat, 'coefficient': float(imp)} for feat, imp in top_features_lr
        ]
        
        output_dir = BASE_DIR / "artifacts" / "explainability"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"explainability_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nOK Resultados: {results_file}")
        
        mlflow.log_param("seed", SEED)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(str(results_file))
        
        print("\n" + "="*80)
        print("OK EXPERIMENTO 11 CONCLUIDO")
        print("="*80 + "\n")
        
        return results

if __name__ == "__main__":
    run_explainability_pipeline()
