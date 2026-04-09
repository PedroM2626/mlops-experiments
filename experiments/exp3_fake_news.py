"""
Experimento 3: Detecção de Fake News / Confiabilidade em Textos
================================================================
Cria dataset sintético e treina modelos para detectar fake news baseado em:
- Padrões linguísticos (CAPS, exclamações, URLs)
- Análise de credibilidade textual
- Ensemble de modelos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import mlflow
import re
import warnings

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent

# ============================================================================
# GERADOR DE DATASET SINTÉTICO
# ============================================================================

def generate_fake_news_dataset(n_samples=1000):
    """Gera dataset sintético de fake news."""
    
    # Patterns para fake news
    fake_patterns = [
        "SHOCKING! You won't believe what happened next!",
        "Breaking NEWS!!! Scientists HATE this ONE WEIRD TRICK!!!",
        "EXCLUSIVE: Celebrity reveals SHOCKING truth!!!",
        "THIS WILL BLOW YOUR MIND... Click here NOW!",
        "URGENT WARNING: Government hiding TRUTH about...",
        "Celebrities HATE her for discovering this...",
        "PROOF that mainstream media is LYING...",
        "Everyone is talking about this... DO YOU KNOW?",
        "CLICK TO SEE: What they DON'T want you to know",
    ]
    
    # Patterns para real news
    real_patterns = [
        "According to recent studies, researchers found that",
        "The study, published in a peer-reviewed journal, shows",
        "Scientists at the university conducted experiments demonstrating",
        "Data from the latest analysis indicates that market trends",
        "The report, based on extensive research, concludes that",
        "Recent developments in the field suggest possible improvements",
        "Experts believe that careful analysis of the data reveals",
        "Research institutions conducted a comprehensive review of available information",
        "The statistical analysis demonstrates a correlation between",
    ]
    
    texts = []
    labels = []
    
    # Fake news
    for i in range(n_samples // 2):
        base = np.random.choice(fake_patterns)
        # Adiciona detalhes
        detail = f" at {np.random.choice(['link.com', 'viral-truth.xyz', 'shocking-news.net'])}. " + \
                 f"Share before {'DELETED' if np.random.random() > 0.5 else 'BANNED'}!!!"
        texts.append(base + detail)
        labels.append(1)  # Fake
    
    # Real news
    for i in range(n_samples // 2):
        base = np.random.choice(real_patterns)
        detail = f" in the domain of {'economics' if i % 2 else 'biology'}. " + \
                 f"Further investigation is needed to validate the findings."
        texts.append(base + detail)
        labels.append(0)  # Real
    
    return texts, labels

def extract_linguistic_features(text):
    """Extrai features linguísticas associadas a fake news."""
    
    features = {}
    
    # Contadores básicos
    features['caps_ratio'] = len(re.findall(r'[A-Z]', text)) / max(len(text), 1)
    features['exclamation_marks'] = text.count('!')
    features['question_marks'] = text.count('?')
    features['urls'] = len(re.findall(r'http\S+|www\S+', text))
    features['caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text))
    features['emoji_count'] = len(re.findall(r'[😀-🙏🌀-🗿]', text))
    features['repeated_chars'] = len(re.findall(r'(.)\1{2,}', text))
    
    # Features baseadas em palavras
    suspicious_words = ['shocking', 'unbelievable', 'exclusive', 'breaking', 'proof', 
                       'guaranteed', 'secret', 'exposed', 'truth', 'banned', 'deleted']
    features['suspicious_words'] = sum(1 for w in suspicious_words if w.lower() in text.lower())
    
    credible_words = ['research', 'study', 'analysis', 'data', 'found', 'demonstrates',
                      'evidence', 'published', 'journal', 'peer-reviewed']
    features['credible_words'] = sum(1 for w in credible_words if w.lower() in text.lower())
    
    return features

def run_fake_news_detection():
    """Pipeline de detecção de fake news."""
    
    print("\n" + "="*80)
    print("🚨 EXPERIMENTO 3: DETECÇÃO DE FAKE NEWS")
    print("="*80 + "\n")
    
    mlflow.set_experiment("Fake_News_Detection")
    
    with mlflow.start_run(run_name="fake_news_complete"):
        
        # Gera dataset
        print("1️⃣  Gerando dataset sintético...")
        texts, labels = generate_fake_news_dataset(n_samples=2000)
        
        df = pd.DataFrame({'text': texts, 'label': labels})
        df['label_name'] = df['label'].map({0: 'Real', 1: 'Fake'})
        
        print(f"   Total: {len(df)} textos")
        print(f"   Real: {(df['label']==0).sum()}, Fake: {(df['label']==1).sum()}\n")
        
        results = {
            'dataset': {
                'total': len(df),
                'real': int((df['label']==0).sum()),
                'fake': int((df['label']==1).sum())
            },
            'models': {}
        }
        
        # Split teste/treino
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label'],
            test_size=0.2, random_state=SEED, stratify=df['label']
        )
        
        # ====================================================================
        # MODEL 1: TF-IDF + Logistic Regression
        # ====================================================================
        print("2️⃣  Treinando Model 1: TF-IDF + Logistic Regression...")
        
        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        lr = LogisticRegression(random_state=SEED, max_iter=200)
        lr.fit(X_train_tfidf, y_train)
        
        y_pred_lr = lr.predict(X_test_tfidf)
        y_proba_lr = lr.predict_proba(X_test_tfidf)[:, 1]
        
        f1_lr = f1_score(y_test, y_pred_lr)
        auc_lr = roc_auc_score(y_test, y_proba_lr)
        
        print(f"   F1-Score: {f1_lr:.3f}, AUC: {auc_lr:.3f}\n")
        
        results['models']['logistic_regression'] = {
            'f1_score': float(f1_lr),
            'auc_score': float(auc_lr),
            'classification_report': classification_report(y_test, y_pred_lr, output_dict=True)
        }
        
        # ====================================================================
        # MODEL 2: Linguistic Features + Random Forest
        # ====================================================================
        print("3️⃣  Treinando Model 2: Linguistic Features + Random Forest...")
        
        # Extrai features linguísticas
        df['linguistic_features'] = df['text'].apply(extract_linguistic_features)
        df_features = pd.DataFrame(list(df['linguistic_features']))
        df_features['label'] = df['label'].values
        
        X_train_ling = df_features[df.index.isin(X_train.index)].drop('label', axis=1)
        X_test_ling = df_features[df.index.isin(X_test.index)].drop('label', axis=1)
        
        # Normaliza
        scaler = StandardScaler()
        X_train_ling_scaled = scaler.fit_transform(X_train_ling)
        X_test_ling_scaled = scaler.transform(X_test_ling)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
        rf.fit(X_train_ling_scaled, y_train)
        
        y_pred_rf = rf.predict(X_test_ling_scaled)
        y_proba_rf = rf.predict_proba(X_test_ling_scaled)[:, 1]
        
        f1_rf = f1_score(y_test, y_pred_rf)
        auc_rf = roc_auc_score(y_test, y_proba_rf)
        
        print(f"   F1-Score: {f1_rf:.3f}, AUC: {auc_rf:.3f}")
        
        # Feature importance
        feature_names = X_train_ling.columns
        feature_importance = dict(zip(feature_names, rf.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("   Top features:")
        for feat, imp in top_features:
            print(f"      {feat:20s} → {imp:.3f}")
        
        results['models']['random_forest'] = {
            'f1_score': float(f1_rf),
            'auc_score': float(auc_rf),
            'classification_report': classification_report(y_test, y_pred_rf, output_dict=True),
            'top_features': [{'feature': f, 'importance': float(i)} for f, i in top_features]
        }
        
        print()
        
        # ====================================================================
        # MODEL 3: Ensemble Voting
        # ====================================================================
        print("4️⃣  Treinando Model 3: Ensemble (Voting)...")
        
        # Combina predictions dos dois modelos
        voting = VotingClassifier([
            ('lr', LogisticRegression(random_state=SEED, max_iter=200)),
            ('rf_ling', RandomForestClassifier(n_estimators=50, random_state=SEED))
        ], voting='soft')
        
        # Para voting, precisa de uma entrada única
        # Vamos criar features combinadas
        X_train_combined = np.hstack([X_train_tfidf.toarray(), X_train_ling_scaled])
        X_test_combined = np.hstack([X_test_tfidf.toarray(), X_test_ling_scaled])
        
        voting.fit(X_train_combined, y_train)
        
        y_pred_voting = voting.predict(X_test_combined)
        y_proba_voting = voting.predict_proba(X_test_combined)[:, 1]
        
        f1_voting = f1_score(y_test, y_pred_voting)
        auc_voting = roc_auc_score(y_test, y_proba_voting)
        
        print(f"   F1-Score: {f1_voting:.3f}, AUC: {auc_voting:.3f}\n")
        
        results['models']['ensemble'] = {
            'f1_score': float(f1_voting),
            'auc_score': float(auc_voting),
            'classification_report': classification_report(y_test, y_pred_voting, output_dict=True)
        }
        
        # ====================================================================
        # SALVA RESULTADOS
        # ====================================================================
        output_dir = BASE_DIR / "artifacts" / "fake_news_detection"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"fake_news_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Resultados salvos: {results_file}")
        
        # MLflow logging
        mlflow.log_param("seed", SEED)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_metric("lr_f1", f1_lr)
        mlflow.log_metric("rf_f1", f1_rf)
        mlflow.log_metric("ensemble_f1", f1_voting)
        mlflow.log_metric("best_f1", max(f1_lr, f1_rf, f1_voting))
        mlflow.log_artifact(str(results_file))
        
        print("\n" + "="*80)
        print("✅ EXPERIMENTO 3 CONCLUÍDO - Detecção Fake News")
        print("="*80 + "\n")
        
        return results

if __name__ == "__main__":
    run_fake_news_detection()
