import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import logging

# Adicionar o diretório pai ao sys.path para importar o framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from train_and_save_professional import MLOpsEnterprise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentiPredExperiment(MLOpsEnterprise):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'twitter_training.csv')
        self.val_path = os.path.join(data_dir, 'twitter_validation.csv')
        self.processed_path = 'processed_senti_data.csv'
        self.vectorized_path = 'vectorized_senti_data.csv'

    def prepare_data(self, vectorize=False):
        """Carrega e limpa o dataset do Twitter."""
        logger.info("📊 Carregando e processando dados de sentimento...")
        cols = ['id', 'entity', 'sentiment', 'text']
        
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {self.train_path}")
            
        df_train = pd.read_csv(self.train_path, names=cols)
        df_train = df_train.dropna(subset=['text', 'sentiment'])
        
        # Reduzir tamanho para garantir execução rápida
        df_train = df_train.sample(n=min(5000, len(df_train)), random_state=42)
        
        if vectorize:
            logger.info("⚙️ Vetorizando texto para engines tabulares (Auto-sklearn, FLAML, TPOT)...")
            tfidf = TfidfVectorizer(max_features=1000)
            X_tfidf = tfidf.fit_transform(df_train['text']).toarray()
            df_vectorized = pd.DataFrame(X_tfidf)
            df_vectorized['target'] = df_train['sentiment'].values
            df_vectorized.to_csv(self.vectorized_path, index=False)
            return self.vectorized_path
        else:
            df_processed = df_train[['text', 'sentiment']].copy()
            df_processed.to_csv(self.processed_path, index=False)
            return self.processed_path

    def train_manual_baseline(self):
        """Treino manual usando TF-IDF + RandomForest para baseline."""
        logger.info("\n🧠 Iniciando Treino Manual (Baseline TF-IDF + RF)...")
        df = pd.read_csv(self.processed_path)
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)
        
        mlflow.set_experiment("/senti_manual_baseline")
        with mlflow.start_run(run_name=f"manual_rf_{datetime.now().strftime('%H%M%S')}"):
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000)),
                ('clf', RandomForestClassifier(n_estimators=100))
            ])
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            self._log_metrics_and_plots({"accuracy": acc})
            mlflow.sklearn.log_model(pipeline, "model")
            logger.info(f"✅ Baseline Manual concluído. Accuracy: {acc:.4f}")
            return acc

    def run_full_comparison(self):
        """Executa a comparação entre todas as engines."""
        # Data file para engines que lidam com texto (AutoGluon)
        raw_data = self.prepare_data(vectorize=False)
        # Data file para engines tabulares (FLAML, Auto-sklearn, H2O, TPOT)
        vec_data = self.prepare_data(vectorize=True)
        
        results = {}
        
        # 1. Manual Baseline
        results['manual'] = self.train_manual_baseline()
        
        # 2. FLAML
        try:
            _, score = self.train_automl(vec_data, engine='flaml', timeout=60)
            results['flaml'] = score
        except Exception as e:
            logger.error(f"❌ Erro no FLAML: {e}")

        # 3. AutoGluon (Lida com texto bruto)
        try:
            _, score = self.train_automl(raw_data, engine='autogluon', timeout=120)
            results['autogluon'] = score
        except Exception as e:
            logger.error(f"❌ Erro no AutoGluon: {e}")

        # 4. Auto-sklearn
        try:
            _, score = self.train_automl(vec_data, engine='autosklearn', timeout=120)
            results['autosklearn'] = score
        except Exception as e:
            logger.error(f"❌ Erro no Auto-sklearn: {e}")

        # 5. H2O AutoML
        try:
            _, score = self.train_automl(vec_data, engine='h2o', timeout=120)
            results['h2o'] = score
        except Exception as e:
            logger.error(f"❌ Erro no H2O: {e}")

        logger.info("\n🏆 Comparação finalizada!")
        logger.info(results)
        
        # Logar resultados finais como artefato
        with open("comparison_results.json", "w") as f:
            import json
            json.dump(results, f)
        mlflow.log_artifact("comparison_results.json")

if __name__ == "__main__":
    # Caminho absoluto para o dataset
    DATA_DIR = r"c:\Users\pedro\Downloads\experiments\experiments\senti-pred\dataset"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        logger.warning(f"⚠️ Diretório de dados criado em {DATA_DIR}. Certifique-se de colocar os arquivos CSV lá.")
    
    experiment = SentiPredExperiment(DATA_DIR)
    try:
        experiment.run_full_comparison()
    except Exception as e:
        logger.error(f"💥 Erro fatal na execução do experimento: {e}")
