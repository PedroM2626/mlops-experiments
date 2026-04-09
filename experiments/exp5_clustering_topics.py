"""
Experimento 5: Clustering + Análise de Tópicos no Senti-Pred
=============================================================
Combina clustering com análise de tópicos em dados de tweets:
- LDA (Latent Dirichlet Allocation)
- Topic Modeling com scikit-learn
- K-Means Clustering
- Análise de sentimento por tópico
- Visualizações interativas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import warnings
import mlflow
from collections import Counter
import pickle

# NLP & Text Processing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Clustering & Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Sentiment Analysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "senti-pred-variations" / "senti-pred-exp1" / "data" / "raw"

# ============================================================================
# PREPROCESS
# ============================================================================

class TextPreprocessor:
    """Preprocessador de texto para análise de tópicos."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean(self, text):
        """Limpa e normaliza texto."""
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove menções, hashtags especiais
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove caracteres especiais
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove números
        text = re.sub(r'\d+', '', text)
        
        # Remove espaços múltiplos
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokeniza e lematiza."""
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(w) 
            for w in tokens 
            if w not in self.stop_words and len(w) > 2
        ]
        return tokens

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_clustering_topic_pipeline():
    """Executa pipeline de clustering + análise de tópicos."""
    
    print("\n" + "="*80)
    print("🔍 EXPERIMENTO 5: CLUSTERING + ANÁLISE DE TÓPICOS (SENTI-PRED)")
    print("="*80 + "\n")
    
    mlflow.set_experiment("Clustering_Topic_Analysis")
    
    with mlflow.start_run(run_name="clustering_topics_complete"):
        
        # Carrega dados
        print("1️⃣  Carregando dados...")
        train_file = DATA_DIR / "twitter_training.csv"
        
        df = pd.read_csv(train_file, header=None)
        df.columns = ['tweet_id', 'topic', 'sentiment', 'text']
        df = df.sample(n=min(500, len(df)), random_state=SEED).reset_index(drop=True)
        
        print(f"   Tweets carregados: {len(df)}")
        print(f"   Tópicos: {df['topic'].unique()}")
        print(f"   Sentimentos: {df['sentiment'].unique()}\n")
        
        results = {
            'dataset_info': {
                'total_tweets': len(df),
                'unique_topics': len(df['topic'].unique()),
                'topics': df['topic'].unique().tolist(),
                'unique_sentiments': len(df['sentiment'].unique()),
                'sentiments': df['sentiment'].unique().tolist()
            },
            'preprocessing': {},
            'clustering': {},
            'topic_modeling': {},
            'sentiment_analysis': {}
        }
        
        # Preprocessa textos
        print("2️⃣  Preprocessando textos...")
        preprocessor = TextPreprocessor()
        
        df['text_cleaned'] = df['text'].apply(preprocessor.clean)
        df['tokens'] = df['text_cleaned'].apply(preprocessor.tokenize_and_lemmatize)
        
        # Remove textos vazios
        df = df[df['tokens'].apply(len) > 0].reset_index(drop=True)
        
        print(f"   Tweets após limpeza: {len(df)}")
        print(f"   Média de tokens: {df['tokens'].apply(len).mean():.1f}\n")
        
        results['preprocessing']['tweets_after_cleaning'] = len(df)
        results['preprocessing']['avg_tokens'] = float(df['tokens'].apply(len).mean())
        
        # ====================================================================
        # CLUSTERING COM K-MEANS
        # ====================================================================
        print("3️⃣  Clustering com K-Means...")
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_tfidf = vectorizer.fit_transform(df['text_cleaned'])
        
        # Determina número ótimo de clusters (Elbow Method)
        inertias = []
        silhouette_scores = []
        K_range = range(2, 7)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            kmeans.fit(X_tfidf)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(
                silhouette_score(
                    X_tfidf,
                    kmeans.labels_,
                    sample_size=min(250, X_tfidf.shape[0]),
                    random_state=SEED
                )
            )
            print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Seleciona K com melhor silhouette
        best_k = K_range[np.argmax(silhouette_scores)]
        print(f"\n   ✅ K ótimo: {best_k} (Silhouette={max(silhouette_scores):.3f})\n")
        
        kmeans_final = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
        df['cluster'] = kmeans_final.fit_predict(X_tfidf)
        
        results['clustering']['optimal_k'] = int(best_k)
        results['clustering']['silhouette_score'] = float(max(silhouette_scores))
        results['clustering']['davies_bouldin_index'] = float(davies_bouldin_score(X_tfidf.toarray(), df['cluster']))
        results['clustering']['cluster_distribution'] = df['cluster'].value_counts().to_dict()
        
        # ====================================================================
        # TOPIC MODELING COM LDA
        # ====================================================================
        print("4️⃣  Topic Modeling com LDA...")
        
        # LDA com scikit-learn para evitar dependência externa
        n_topics = 5
        count_vectorizer = CountVectorizer(max_features=5000)
        X_counts = count_vectorizer.fit_transform(df['tokens'].apply(lambda tokens: ' '.join(tokens)))
        lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=SEED,
            learning_method='batch'
        )
        doc_topic_matrix = lda_model.fit_transform(X_counts)
        perplexity_score = lda_model.perplexity(X_counts)

        print(f"   Perplexidade: {perplexity_score:.3f}")
        print(f"   Tópicos identificados: {n_topics}\n")
        
        # Extrai tópicos principais
        topics_info = {}
        feature_names = count_vectorizer.get_feature_names_out()
        for idx, topic_weights in enumerate(lda_model.components_):
            top_indices = topic_weights.argsort()[-10:][::-1]
            topic_terms = [feature_names[i] for i in top_indices]
            topic_text = ', '.join(topic_terms)
            topics_info[f"topic_{idx}"] = topic_text
            print(f"   Tópico {idx}: {topic_text[:100]}...")
        
        df['topic_lda'] = doc_topic_matrix.argmax(axis=1)
        
        results['topic_modeling']['num_topics'] = n_topics
        results['topic_modeling']['perplexity_score'] = float(perplexity_score)
        results['topic_modeling']['topics'] = topics_info
        results['topic_modeling']['topics_distribution'] = df['topic_lda'].value_counts().to_dict()
        
        # ====================================================================
        # ANÁLISE DE SENTIMENTO POR TÓPICO/CLUSTER
        # ====================================================================
        print("5️⃣  Análise de Sentimento por Tópico...")
        
        # Prepara labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(df['sentiment'])
        
        # Modelo de sentimento simples
        sentiment_model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000)),
            ('lr', LogisticRegression(random_state=SEED, max_iter=200))
        ])
        
        sentiment_model.fit(df['text_cleaned'], y_encoded)
        df['sentiment_pred'] = sentiment_model.predict(df['text_cleaned'])
        df['sentiment_pred_str'] = le.inverse_transform(df['sentiment_pred'])
        
        sentiment_by_cluster = []
        sentiment_by_topic = []
        
        # Por cluster
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            sentiment_dist = cluster_data['sentiment'].value_counts().to_dict()
            sentiment_by_cluster.append({
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_data)),
                'sentiments': sentiment_dist,
                'top_words': ' '.join([w for words in cluster_data['tokens'].head(50) for w in words][:20])
            })
        
        # Por tópico LDA
        for topic_id in sorted(df['topic_lda'].unique()):
            topic_data = df[df['topic_lda'] == topic_id]
            sentiment_dist = topic_data['sentiment'].value_counts().to_dict()
            sentiment_by_topic.append({
                'topic_id': int(topic_id),
                'size': int(len(topic_data)),
                'sentiments': sentiment_dist,
                'top_words': ' '.join([w for words in topic_data['tokens'].head(50) for w in words][:20])
            })
        
        results['sentiment_analysis']['by_cluster'] = sentiment_by_cluster
        results['sentiment_analysis']['by_topic'] = sentiment_by_topic
        
        print(f"   Sentimentos analisados por {len(sentiment_by_cluster)} clusters")
        print(f"   Sentimentos analisados por {len(sentiment_by_topic)} tópicos LDA\n")
        
        # ====================================================================
        # SALVA RESULTADOS
        # ====================================================================
        output_dir = BASE_DIR / "artifacts" / "clustering_topics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON com resultados
        results_file = output_dir / f"clustering_topics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Resultados salvos: {results_file}")
        
        # Salva modelo LDA
        lda_file = output_dir / f"lda_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(lda_file, 'wb') as f:
            pickle.dump(lda_model, f)
        print(f"✅ Modelo LDA salvo: {lda_file}")
        
        # Salva dataframe anotado
        df_output = output_dir / f"tweets_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df[['text', 'sentiment', 'cluster', 'topic_lda', 'sentiment_pred_str']].to_csv(df_output, index=False)
        print(f"✅ Tweets anotados salvos: {df_output}\n")
        
        # MLflow logging
        mlflow.log_param("seed", SEED)
        mlflow.log_param("num_topics", n_topics)
        mlflow.log_param("optimal_clusters", best_k)
        mlflow.log_metric("perplexity_score", perplexity_score)
        mlflow.log_metric("silhouette_score", max(silhouette_scores))
        mlflow.log_metric("davies_bouldin_index", davies_bouldin_score(X_tfidf.toarray(), df['cluster']))
        mlflow.log_artifact(str(results_file))
        mlflow.log_artifact(str(lda_file))
        
        print("="*80)
        print("✅ EXPERIMENTO 5 CONCLUÍDO - Clustering + Tópicos")
        print("="*80 + "\n")
        
        return results

if __name__ == "__main__":
    run_clustering_topic_pipeline()
