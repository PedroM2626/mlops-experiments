import argparse
import logging
import os
from forecaster_class import SalesForecasterV2

# logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(data_path: str, artifacts_path: str, n_trials: int):
    """
    Função principal para orquestrar o treinamento do modelo.
    """
    logging.info("Iniciando o Pipeline de Treinamento.")
    
    # Define os caminhos para os arquivos de dados e saída do modelo
    file_paths = {
        'vendas': os.path.join(data_path, 'raw/fato_vendas.parquet'),
        'pdvs': os.path.join(data_path, 'raw/dim_pdvs.parquet'),
        'produtos': os.path.join(data_path, 'raw/dim_produtos.parquet')
    }
    model_output_path = os.path.join(artifacts_path, 'sales_forecaster_v2_final.joblib')

    # Instancia e executa o pipeline
    forecaster = SalesForecasterV2()
    
    try:
        df_full_data = forecaster.load_data(file_paths)
        
        forecaster.train(
            df_full_data,
            validation_split_week=48,
            n_trials=n_trials
        )
        
    except Exception as e:
        logging.error(f"O pipeline de treinamento falhou com o erro: {e}")
        raise e

    logging.info("Pipeline de Treinamento finalizado com sucesso!")

if __name__ == "__main__":
    # Configura os argumentos que o script pode receber via linha de comando
    parser = argparse.ArgumentParser(description="Treina o modelo de previsão de vendas.")
    parser.add_argument("--data_path", type=str, default="data", help="Caminho para a pasta 'data'.")
    parser.add_argument("--artifacts_path", type=str, default="artifacts", help="Caminho para salvar o modelo treinado.")
    parser.add_argument("--n_trials", type=int, default=100, help="Número de trials para a otimização com Optuna.")
    
    args = parser.parse_args()
    
    main(args.data_path, args.artifacts_path, args.n_trials)