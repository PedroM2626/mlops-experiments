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

# Adicionar o diretório pai ao sys.path para importar o framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from train_and_save_professional import MLOpsEnterprise

class SentiPredExperiment(MLOpsEnterprise):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'twitter_training.csv')
        self.val_path = os.path.join(data_dir, 'twitter_validation.csv')
        self.processed_path = 'processed_senti_data.csv'

    def prepare_data(self):
        """Carrega e limpa o dataset do Twitter."""
        print("📊 Carregando e processando dados de sentimento...")
        cols = ['id', 'entity', 'sentiment', 'text']
        df_train = pd.read_csv(self.train_path, names=cols)
        
        # Limpeza básica
        df_train = df_train.dropna(subset=['text', 'sentiment'])
        
        # Reduzir tamanho para garantir execução em ambiente com pouco disco
        df_train = df_train.sample(n=min(10000, len(df_train)), random_state=42)
        
        # Vamos manter apenas o necessário para o AutoML
        df_processed = df_train[['text', 'sentiment']].copy()
        
        # Para o AutoML (TPOT/FLAML) funcionar melhor com texto sem transformers, 
        # vamos salvar o texto bruto e deixar que eles lidem ou fazer um TF-IDF básico.
        # No caso do AutoGluon, ele lida bem com texto se especificado.
        df_processed.to_csv(self.processed_path, index=False)
        return self.processed_path

    def train_manual_baseline(self):
        """Treino manual usando TF-IDF + RandomForest para baseline."""
        print("\n🧠 Iniciando Treino Manual (Baseline TF-IDF + RF)...")
        df = pd.read_csv(self.processed_path)
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)
        
        mlflow.set_experiment("/senti_manual_baseline")
        with mlflow.start_run(run_name=f"manual_rf_{datetime.now().strftime('%H%M%S')}"):
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=5000)),
                ('clf', RandomForestClassifier(n_estimators=100))
            ])
            
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            
            self._log_metrics_and_plots({"accuracy": acc})
            mlflow.sklearn.log_model(pipeline, "model", registered_model_name="senti_manual_baseline")
            print(f"✅ Baseline Manual concluído. Accuracy: {acc:.4f}")
            return acc

    def run_full_comparison(self):
        """Executa a comparação entre todas as engines."""
        data_file = self.prepare_data()
        
        results = {}
        
        # 1. Manual Baseline
        results['manual'] = self.train_manual_baseline()
        
        # 2. FLAML (Rápido)
        try:
            model_flaml, score_flaml = self.train_automl(data_file, engine='flaml', timeout=120)
            results['flaml'] = score_flaml
        except Exception as e:
            print(f"❌ Erro no FLAML: {e}")

        # 3. TPOT
        try:
            # Nota: TPOT com texto bruto precisa de pipeline manual ou ser tratado como tabular 
            # de features numéricas. Para esse script, vamos focar no AutoML Tabular do AutoGluon/FLAML.
            print("⏩ Pulando TPOT para texto bruto (requer pré-vetorização)...")
        except Exception as e:
            print(f"❌ Erro no TPOT: {e}")

        # 4. AutoGluon (Performance Máxima)
        try:
            model_ag, score_ag = self.train_automl(data_file, engine='autogluon', timeout=300)
            results['autogluon'] = score_ag
        except Exception as e:
            print(f"❌ Erro no AutoGluon: {e}")

        # 5. Auto-sklearn
        try:
            model_as, score_as = self.train_automl(data_file, engine='autosklearn', timeout=300)
            results['autosklearn'] = score_as
        except Exception as e:
            print(f"❌ Erro no Auto-sklearn: {e}")

        # 6. H2O AutoML
        try:
            model_h2o, score_h2o = self.train_automl(data_file, engine='h2o', timeout=300)
            results['h2o'] = score_h2o
        except Exception as e:
            print(f"❌ Erro no H2O: {e}")

        print("\n🏆 Comparação finalizada! Verifique o DagsHub para detalhes completos.")
        print(results)

if __name__ == "__main__":
    DATA_DIR = r"c:\Users\pedro\Downloads\experiments\experiments\senti-pred\dataset"
    experiment = SentiPredExperiment(DATA_DIR)
    experiment.run_full_comparison()
