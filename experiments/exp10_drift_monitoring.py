"""
Experimento 10: Data Drift Monitoring + Concept Drift
======================================================
Detecta mudanÃ§as em distribuiÃ§Ãµes de dados:
- Kolmogorov-Smirnov test
- Wasserstein distance
- Monitoramento de Features
- Alertas de Concept Drift
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import mlflow
import warnings
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "senti-pred-variations" / "senti-pred-exp1" / "data" / "raw"

# ============================================================================
# DETECTORES DE DRIFT
# ============================================================================

class DriftDetector:
    """Detector de data/concept drift."""
    
    def __init__(self, window_size=100, threshold=0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.baseline = None
        self.history = []
    
    def ks_test(self, baseline, current):
        """Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(baseline, current)
        return statistic, p_value
    
    def wasserstein_distance_test(self, baseline, current):
        """Wasserstein distance."""
        return wasserstein_distance(baseline, current)
    
    def check_drift(self, baseline, current):
        """Verifica se hÃ¡ drift."""
        
        ks_stat, ks_p = self.ks_test(baseline, current)
        wd = self.wasserstein_distance_test(baseline, current)
        
        # Detecta drift se p-value Ã© significante ou Wasserstein Ã© alto
        drift_detected = (ks_p < self.threshold) or (wd > np.percentile(np.abs(baseline - current.mean()), 75))
        
        return {
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p),
            'wasserstein_distance': float(wd),
            'drift_detected': bool(drift_detected)
        }

def run_drift_monitoring():
    """Pipeline de monitoramento de drift."""
    
    print("\\n" + "="*80)
    print("âš ï¸  EXPERIMENTO 10: DATA DRIFT MONITORING")
    print("="*80 + "\\n")
    
    mlflow.set_experiment("Data_Drift_Monitoring")
    
    with mlflow.start_run(run_name="drift_monitoring_complete"):
        
        # Carrega dados
        print("1ï¸âƒ£  Carregando dados de Senti-Pred...")
        
        train_file = DATA_DIR / "twitter_training.csv"
        val_file = DATA_DIR / "twitter_validation.csv"
        
        df_train = pd.read_csv(train_file, header=None)
        df_train.columns = ['tweet_id', 'topic', 'sentiment', 'text']
        
        df_val = pd.read_csv(val_file, header=None)
        df_val.columns = ['tweet_id', 'topic', 'sentiment', 'text']
        
        print(f"   Treino: {len(df_train)} tweets")
        print(f"   ValidaÃ§Ã£o: {len(df_val)} tweets\\n")
        
        results = {
            'dataset_comparison': {
                'train_size': len(df_train),
                'val_size': len(df_val),
                'sentiments_train': df_train['sentiment'].unique().tolist(),
                'sentiments_val': df_val['sentiment'].unique().tolist()
            },
            'drift_analysis': {},
            'feature_drift': {}
        }
        
        # ====================================================================
        # ExtraÃ§Ã£o de Features NumÃ©ricas
        # ====================================================================
        print("2ï¸âƒ£  Extraindo features das textos...")
        
        def extract_text_features(text):
            """Extrai features dos textos."""
            text = "" if pd.isna(text) else str(text)
            return {
                'length': len(text),
                'word_count': len(text.split()),
                'avg_word_length': np.mean([len(w) for w in text.split()]) if len(text.split()) > 0 else 0,
                'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
                'punctuation_count': sum(1 for c in text if c in '!?.,:;'),
                'unique_words_ratio': len(set(text.lower().split())) / max(len(text.split()), 1)
            }
        
        df_train['text_features'] = df_train['text'].apply(extract_text_features)
        df_val['text_features'] = df_val['text'].apply(extract_text_features)
        
        features_train = pd.DataFrame(list(df_train['text_features']))
        features_val = pd.DataFrame(list(df_val['text_features']))
        
        print(f"   Features extraÃ­das: {features_train.shape}")
        print(f"   Colunas: {list(features_train.columns)}\\n")
        
        # ====================================================================
        # COMPARAÃ‡ÃƒO DE DISTRIBUIÃ‡ÃƒO: SENTIMENTOS
        # ====================================================================
        print("3ï¸âƒ£  Monitorando Concept Drift (Sentimentos)...")
        
        detector = DriftDetector(threshold=0.05)
        
        sentiment_dist_train = df_train['sentiment'].value_counts(normalize=True).sort_index()
        sentiment_dist_val = df_val['sentiment'].value_counts(normalize=True).sort_index()
        
        print(f"   DistribuiÃ§Ã£o Treino: {dict(sentiment_dist_train)}")
        print(f"   DistribuiÃ§Ã£o ValidaÃ§Ã£o: {dict(sentiment_dist_val)}")
        
        # Testa com codificaÃ§Ã£o numÃ©rica
        le = LabelEncoder()
        sentiment_encoded_train = le.fit_transform(df_train['sentiment'])
        sentiment_encoded_val = le.transform(df_val['sentiment'])
        
        sentiment_drift = detector.check_drift(
            sentiment_encoded_train,
            sentiment_encoded_val
        )
        
        print(f"   KS Statistic: {sentiment_drift['ks_statistic']:.4f}")
        print(f"   KS p-value: {sentiment_drift['ks_p_value']:.4f}")
        print(f"   Wasserstein Distance: {sentiment_drift['wasserstein_distance']:.4f}")
        print(f"   Drift Detectado: {'âš ï¸ SIM' if sentiment_drift['drift_detected'] else 'âœ… NÃ£o'}\\n")
        
        results['drift_analysis']['sentiment_drift'] = sentiment_drift
        results['drift_analysis']['sentiment_drift']['drift_detected'] = bool(sentiment_drift['drift_detected'])
        results['drift_analysis']['sentiment_drift']['ks_statistic'] = float(sentiment_drift['ks_statistic'])
        results['drift_analysis']['sentiment_drift']['ks_p_value'] = float(sentiment_drift['ks_p_value'])
        results['drift_analysis']['sentiment_drift']['wasserstein_distance'] = float(sentiment_drift['wasserstein_distance'])
        
        # ====================================================================
        # MONITORAMENTO DE FEATURES DE TEXTO
        # ====================================================================
        print("4ï¸âƒ£  Monitorando Data Drift (Features de Texto)...")
        
        feature_drift_report = {}
        
        for feature_col in features_train.columns:
            feature_train = features_train[feature_col].values
            feature_val = features_val[feature_col].values
            
            # Normaliza
            scaler = StandardScaler()
            feature_train_scaled = scaler.fit_transform(feature_train.reshape(-1, 1)).flatten()
            feature_val_scaled = scaler.transform(feature_val.reshape(-1, 1)).flatten()
            
            feat_drift = detector.check_drift(feature_train_scaled, feature_val_scaled)
            
            feature_drift_report[feature_col] = {
                'ks_statistic': float(feat_drift['ks_statistic']),
                'ks_p_value': float(feat_drift['ks_p_value']),
                'wasserstein_distance': float(feat_drift['wasserstein_distance']),
                'drift_detected': bool(feat_drift['drift_detected']),
                'train_mean': float(feature_train.mean()),
                'train_std': float(feature_train.std()),
                'val_mean': float(feature_val.mean()),
                'val_std': float(feature_val.std())
            }
            
            status = 'âš ï¸ Drift' if feat_drift['drift_detected'] else 'âœ… OK'
            print(f"   {feature_col:20s}: {status} (KS={feat_drift['ks_statistic']:.4f}, WD={feat_drift['wasserstein_distance']:.4f})")
        
        results['feature_drift'] = feature_drift_report
        
        print()
        
        # ====================================================================
        # RESUMO DE ANOMALIAS
        # ====================================================================
        print("5ï¸âƒ£  Resumo de DetecÃ§Ãµes...")
        
        drifted_features = [f for f, v in feature_drift_report.items() if v['drift_detected']]
        
        print(f"   Total de features: {len(feature_drift_report)}")
        print(f"   Features com drift: {len(drifted_features)}")
        
        if drifted_features:
            print(f"   Features afetadas: {', '.join(drifted_features)}\\n")
        
        results['summary'] = {
            'total_features': len(feature_drift_report),
            'drifted_features_count': len(drifted_features),
            'drifted_features': drifted_features,
            'concept_drift_detected': sentiment_drift['drift_detected'],
            'overall_status': 'WARNING' if (sentiment_drift['drift_detected'] or len(drifted_features) > 0) else 'OK'
        }
        
        print(f"   Status Geral: {results['summary']['overall_status']}")
        print("   RecomendaÃ§Ã£o: Considere retreinar o modelo" if results['summary']['overall_status'] == 'WARNING' else "   âœ… Modelo estÃ¡ estÃ¡vel")
        print()
        
        # ====================================================================
        # SALVA RESULTADOS
        # ====================================================================
        output_dir = BASE_DIR / "artifacts" / "drift_monitoring"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"drift_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"âœ… Resultados salvos: {results_file}")
        
        # MLflow logging
        mlflow.log_param("seed", SEED)
        mlflow.log_param("train_size", len(df_train))
        mlflow.log_param("val_size", len(df_val))
        mlflow.log_metric("sentiment_ks_statistic", sentiment_drift['ks_statistic'])
        mlflow.log_metric("drifted_features_count", len(drifted_features))
        mlflow.log_artifact(str(results_file))
        
        print("\\n" + "="*80)
        print("âœ… EXPERIMENTO 10 CONCLUÃDO - Data Drift Monitoring")
        print("="*80 + "\\n")
        
        return results

if __name__ == "__main__":
    run_drift_monitoring()

