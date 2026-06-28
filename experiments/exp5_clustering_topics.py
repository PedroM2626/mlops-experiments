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
    print(" EXPERIMENTO 5: CLUSTERING + ANÁLISE DE TÓPICOS (SENTI-PRED)")
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
        # ====================================================================
        # CLUSTERING COM UMAP E HDBSCAN (SOTA)
        # ====================================================================
        print("3️⃣  Clustering com UMAP + HDBSCAN...")
        
        from sentence_transformers import SentenceTransformer
        import umap
        import hdbscan
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics import silhouette_score
        
        print("   Gerando Embeddings Densos com all-MiniLM-L6-v2...")
        model_sbert = SentenceTransformer('all-MiniLM-L6-v2')
        X_dense = model_sbert.encode(df['text_cleaned'].tolist())
        
        print("   Reduzindo dimensionalidade com UMAP...")
        umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=SEED)
        X_umap = umap_model.fit_transform(X_dense)
        
        print("   Clusterizando com HDBSCAN...")
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean', cluster_selection_method='eom')
        df['cluster'] = hdbscan_model.fit_predict(X_umap)
        
        n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
        n_noise = list(df['cluster']).count(-1)
        
        print(f"   ✅ Clusters encontrados: {n_clusters}")
        print(f"   ⚠️  Tweets classificados como ruído (-1): {n_noise} de {len(df)}")
        
        try:
            sil_score = silhouette_score(X_umap[df['cluster'] != -1], df['cluster'][df['cluster'] != -1], random_state=SEED)
        except ValueError:
            sil_score = 0.0
            
        print(f"   Silhouette Score (sem ruído): {sil_score:.3f}\n")
        
        results['clustering']['optimal_k'] = n_clusters
        results['clustering']['silhouette_score'] = float(sil_score)
        results['clustering']['noise_points'] = n_noise
        results['clustering']['cluster_distribution'] = df['cluster'].value_counts().to_dict()
        
        # ====================================================================
        # EXTRAÇÃO DE TÓPICOS (c-TF-IDF simplificado)
        # ====================================================================
        print("4️⃣  Extraindo Tópicos dos Clusters...")
        
        topics_info = {}
        for cluster_id in sorted(df['cluster'].unique()):
            if cluster_id == -1:
                continue
            
            cluster_texts = df[df['cluster'] == cluster_id]['text_cleaned']
            
            if len(cluster_texts) > 0:
                vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
                try:
                    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                    feature_names = vectorizer.get_feature_names_out()
                    word_scores = tfidf_matrix.sum(axis=0).A1
                    top_indices = word_scores.argsort()[-10:][::-1]
                    topic_terms = [feature_names[i] for i in top_indices]
                    topic_text = ', '.join(topic_terms)
                except ValueError:
                    topic_text = "Sem palavras suficientes"
            else:
                topic_text = "Cluster Vazio"
                
            topics_info[f"topic_{cluster_id}"] = topic_text
            print(f"   Cluster {cluster_id}: {topic_text[:100]}...")
            
        df['topic_lda'] = df['cluster']
        
        results['topic_modeling']['num_topics'] = n_clusters
        results['topic_modeling']['topics'] = topics_info
        results['topic_modeling']['topics_distribution'] = df['topic_lda'].value_counts().to_dict()
        
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
        
        # Salva dataframe anotado
        df_output = output_dir / f"tweets_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df[['text', 'sentiment', 'cluster', 'topic_lda', 'sentiment_pred_str']].to_csv(df_output, index=False)
        print(f"✅ Tweets anotados salvos: {df_output}\n")
        
        # MLflow logging
        mlflow.log_param("seed", SEED)
        mlflow.log_param("num_topics", n_clusters)
        mlflow.log_param("optimal_clusters", n_clusters)
        mlflow.log_metric("silhouette_score", float(sil_score))
        mlflow.log_metric("noise_points", n_noise)
        mlflow.log_artifact(str(results_file))
        print("="*80)
        print("✅ EXPERIMENTO 5 CONCLUÍDO - Clustering + Tópicos")
        print("="*80 + "\n")
        
        return results

if __name__ == "__main__":
    run_clustering_topic_pipeline()
