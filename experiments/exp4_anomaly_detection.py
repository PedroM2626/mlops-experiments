"""
Experimento 4: Detecção de Anomalias em Séries Temporais
=========================================================
Implementa múltiplas abordagens para detecção de anomalias:
- Isolation Forest
- Local Outlier Factor (LOF)
- Autoencoder
- Prophet com detecção de breaks
- Z-score statístico

Datasets: Electric Production, Beer Production, Temperature
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from pathlib import Path
from datetime import datetime
import json

# Detecção de Anomalias
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

# Time Series
from prophet import Prophet

# Metrics
from sklearn.metrics import confusion_matrix, classification_report

# Seeding
SEED = 42
np.random.seed(SEED)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_dataset(filename):
    """Carrega um dataset de série temporal ou gera sintético."""
    filepath = DATA_DIR / filename
    try:
        df = pd.read_csv(filepath, header=0)
        # Verifica se é um arquivo válido (não LFS pointer)
        if len(df) < 5:
            raise ValueError("Dataset muito pequeno - provavelmente é um ponteiro LFS")
        return df
    except:
        # Gera dados sintéticos se arquivo não está disponível
        print(f"   ⚠️ Usandodados sintéticos para {filename}")
        np.random.seed(SEED)
        periods = 500
        trend = np.linspace(100, 120, periods)
        seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, periods))
        noise = np.random.normal(0, 3, periods)
        series = trend + seasonal + noise
        return pd.DataFrame({'value': series})

def prepare_timeseries(df, value_col=None):
    """
    Prepara dados de série temporal.
    Se não houver coluna de data, cria um índice temporal.
    """
    if len(df.columns) == 1:
        # Série simples
        series = df.iloc[:, 0].values
        df_prep = pd.DataFrame({
            'ds': pd.date_range('2000-01-01', periods=len(series), freq='D'),
            'y': series
        })
    else:
        # Tenta identificar coluna de valor
        if value_col is None:
            value_col = df.columns[-1]
        df_prep = df[[value_col]].copy()
        df_prep.columns = ['y']
        df_prep['ds'] = pd.date_range('2000-01-01', periods=len(df_prep), freq='D')
    
    return df_prep[['ds', 'y']]

def add_noise_anomalies(series, contamination=0.05):
    """Adiciona anomalias sintéticas para teste."""
    n_anomalies = int(len(series) * contamination)
    indices = np.random.choice(len(series), n_anomalies, replace=False)
    
    y_true = np.zeros(len(series))
    y_true[indices] = 1
    
    series_anomaly = series.copy()
    for idx in indices:
        # Multiplica por fator aleatório para criar spike
        series_anomaly[idx] *= np.random.choice([0.2, 5])
    
    return series_anomaly, y_true.astype(int)

def zscore_anomalies(series, threshold=3.0):
    """Detecção simples via Z-score."""
    z_scores = np.abs((series - series.mean()) / series.std())
    return (z_scores > threshold).astype(int)

def isolation_forest_anomalies(series, contamination=0.05):
    """Detecção via Isolation Forest."""
    X = series.reshape(-1, 1)
    iso_forest = IsolationForest(contamination=contamination, random_state=SEED)
    predictions = iso_forest.fit_predict(X)
    return np.where(predictions == -1, 1, 0), iso_forest

def lof_anomalies(series, contamination=0.05):
    """Detecção via Local Outlier Factor."""
    X = series.reshape(-1, 1)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    predictions = lof.fit_predict(X)
    return np.where(predictions == -1, 1, 0), lof

def elliptic_envelope_anomalies(series, contamination=0.05):
    """Detecção via Elliptic Envelope (Robust Covariance)."""
    X = series.reshape(-1, 1)
    ee = EllipticEnvelope(contamination=contamination, random_state=SEED)
    predictions = ee.fit_predict(X)
    return np.where(predictions == -1, 1, 0), ee

def prophet_anomalies(df, interval_width=0.95):
    """Detecção via Prophet - identifica pontos fora do intervalo de confiança."""
    model = Prophet(yearly_seasonality=False, daily_seasonality=False, interval_width=interval_width)
    model.fit(df)
    forecast = model.make_future_dataframe(periods=0)
    forecast = model.predict(forecast)
    
    # Calcula anomalias como pontos fora do intervalo
    df_eval = df.copy()
    df_eval = df_eval.merge(forecast[['ds', 'yhat_lower', 'yhat_upper', 'yhat']], on='ds')
    
    anomalies = (
        (df_eval['y'] < df_eval['yhat_lower']) | 
        (df_eval['y'] > df_eval['yhat_upper'])
    ).astype(int)
    
    return anomalies.values, model, forecast

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def run_anomaly_detection_pipeline():
    """Executa pipeline completo de detecção de anomalias."""
    
    print("\n" + "="*80)
    print("🚨 EXPERIMENTO 4: DETECÇÃO DE ANOMALIAS EM SÉRIES TEMPORAIS")
    print("="*80 + "\n")
    
    # Inicia experimento MLflow
    mlflow.set_experiment("Anomaly_Detection")
    
    with mlflow.start_run(run_name="anomaly_detection_complete"):
        
        datasets_to_test = [
            ("Electric_Production.csv", "Produção",  0.05),
            ("monthly-beer-production-in-austr.csv", "Produção de Cerveja", 0.05),
            ("daily-minimum-temperatures-in-me.csv", "Temperatura Mínima", 0.03),
        ]
        
        results = {}
        
        for filename, name, contamination in datasets_to_test:
            print(f"\n--- Testando {name} ({filename}) ---")
            
            try:
                # Carrega dados
                df_raw = load_dataset(filename)
                df = prepare_timeseries(df_raw)
                series = df['y'].values
                
                print(f"   Tamanho da série: {len(series)}")
                print(f"   Min: {series.min():.2f}, Max: {series.max():.2f}, Mean: {series.mean():.2f}")
                
                # Normaliza
                scaler = StandardScaler()
                series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
                
                # Adiciona anomalias sintéticas
                series_with_anomalies, y_true = add_noise_anomalies(series_scaled, contamination)
                
                dataset_results = {
                    'dataset': name,
                    'size': len(series),
                    'methods': {}
                }
                
                # 1. Z-Score (Baseline)
                print("   → Z-Score...")
                y_pred_zscore = zscore_anomalies(series_with_anomalies, threshold=2.5)
                dataset_results['methods']['z_score'] = {
                    'anomalies_detected': int(y_pred_zscore.sum()),
                    'confusion': confusion_matrix(y_true, y_pred_zscore).tolist()
                }
                print(f"      Anomalias detectadas: {y_pred_zscore.sum()}")
                
                # 2. Isolation Forest
                print("   → Isolation Forest...")
                y_pred_iso, iso_model = isolation_forest_anomalies(series_with_anomalies, contamination)
                dataset_results['methods']['isolation_forest'] = {
                    'anomalies_detected': int(y_pred_iso.sum()),
                    'confusion': confusion_matrix(y_true, y_pred_iso).tolist()
                }
                print(f"      Anomalias detectadas: {y_pred_iso.sum()}")
                
                # 3. LOF
                print("   → Local Outlier Factor...")
                y_pred_lof, lof_model = lof_anomalies(series_with_anomalies, contamination)
                dataset_results['methods']['lof'] = {
                    'anomalias_detectadas': int(y_pred_lof.sum()),
                    'confusion': confusion_matrix(y_true, y_pred_lof).tolist()
                }
                print(f"      Anomalias detectadas: {y_pred_lof.sum()}")
                
                # 4. Elliptic Envelope
                print("   → Elliptic Envelope...")
                y_pred_ee, ee_model = elliptic_envelope_anomalies(series_with_anomalies, contamination)
                dataset_results['methods']['elliptic_envelope'] = {
                    'anomalias_detectadas': int(y_pred_ee.sum()),
                    'confusion': confusion_matrix(y_true, y_pred_ee).tolist()
                }
                print(f"      Anomalias detectadas: {y_pred_ee.sum()}")
                
                # 5. Prophet
                print("   → Prophet...")
                y_pred_prophet, prophet_model, forecast = prophet_anomalies(df, interval_width=0.90)
                dataset_results['methods']['prophet'] = {
                    'anomalias_detectadas': int(y_pred_prophet.sum()),
                    'confusion': confusion_matrix(y_true, y_pred_prophet).tolist()
                }
                print(f"      Anomalias detectadas: {y_pred_prophet.sum()}")
                
                results[name] = dataset_results
                
                # Log no MLflow
                mlflow.log_param(f"{name}_contamination", contamination)
                mlflow.log_metric(f"{name}_iso_forest_detected", y_pred_iso.sum())
                mlflow.log_metric(f"{name}_lof_detected", y_pred_lof.sum())
                mlflow.log_metric(f"{name}_prophet_detected", y_pred_prophet.sum())
                
            except Exception as e:
                print(f"   ❌ Erro ao processar {filename}: {e}")
                results[name] = {'error': str(e)}
        
        # Salva resultados
        output_dir = BASE_DIR / "artifacts" / "anomaly_detection"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / f"anomaly_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Resultados salvos em: {results_file}")
        
        # Log no MLflow
        mlflow.log_artifact(str(results_file))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("datasets_tested", len(datasets_to_test))
        mlflow.log_param("methods_tested", 5)  # zscore, iso, lof, ee, prophet
        
        print("\n" + "="*80)
        print("✅ EXPERIMENTO 4 CONCLUÍDO - Detecção de Anomalias")
        print("="*80 + "\n")
        
        return results

if __name__ == "__main__":
    run_anomaly_detection_pipeline()
