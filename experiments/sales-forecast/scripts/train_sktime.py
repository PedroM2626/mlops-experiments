import os
import sys
import logging
import time

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.forecaster_sktime import SalesForecasterSktime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_paths = {
        'vendas': os.path.join(base_dir, 'data', 'raw', 'fato_vendas.parquet'),
        'pdvs': os.path.join(base_dir, 'data', 'raw', 'dim_pdvs.parquet'),
        'produtos': os.path.join(base_dir, 'data', 'raw', 'dim_produtos.parquet')
    }

    logging.info("Initializing Sktime pipeline...")
    pipeline = SalesForecasterSktime()
    
    # 1. Load data
    df_agregado = pipeline.load_data(file_paths)
    
    # 2. Training
    # Drastic subsampling to test the OOM crash in Sktime
    df_agregado = df_agregado.head(50000)
    
    # For a quick comparison without spending hours on optimization,
    # we use n_trials=5. The main focus is to measure the Feature Engineering time and whether the model breaks
    logging.info("Starting Training with Sktime WindowSummarizer (50k records)")
    
    start_total = time.time()
    pipeline.train(df_agregado, validation_split_week=48, use_optuna=True, n_trials=20)
    total_time = time.time() - start_total
    
    mae = pipeline.performance_metrics.get('validation_mae', -1)
    
    print("\n" + "="*50)
    print(" 🚀 SKTIME BENCHMARK RESULTS (SALES FORECAST)")
    print("="*50)
    print(f"Total FE time: {pipeline.fe_time:.2f} seconds")
    print(f"Total training time: {total_time - pipeline.fe_time:.2f} seconds")
    print(f"MAE on Test (Week 48+): {mae:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
