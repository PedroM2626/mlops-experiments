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

    logging.info("Inicializando Pipeline Sktime...")
    pipeline = SalesForecasterSktime()
    
    # 1. Carregar dados
    df_agregado = pipeline.load_data(file_paths)
    
    # 2. Treinamento
    # Subamostragem drástica para testar OOM crash no Sktime
    df_agregado = df_agregado.head(50000)
    
    # Para fins de comparacao rapida sem perder horas de otimizacao, 
    # vamos usar n_trials=5. O foco principal é medir o tempo do Feature Engineering e se o modelo quebra
    logging.info("Iniciando Treinamento com Sktime WindowSummarizer (50k registros)")
    
    start_total = time.time()
    pipeline.train(df_agregado, validation_split_week=48, use_optuna=True, n_trials=20)
    total_time = time.time() - start_total
    
    mae = pipeline.performance_metrics.get('validation_mae', -1)
    
    print("\n" + "="*50)
    print(" 🚀 RESULTADOS DO SKTIME BENCHMARK (SALES FORECAST)")
    print("="*50)
    print(f"Tempo total de FE: {pipeline.fe_time:.2f} segundos")
    print(f"Tempo total de Treinamento: {total_time - pipeline.fe_time:.2f} segundos")
    print(f"MAE no Teste (Semana 48+): {mae:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
