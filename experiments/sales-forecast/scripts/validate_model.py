import mlflow
import pandas as pd
import os
from dotenv import load_dotenv
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carrega variáveis de ambiente
load_dotenv()

def validate_best_model(run_id: str):
    """
    Valida o melhor modelo gerado pelo experimento usando o MLflow.
    """
    try:
        # Configura o tracking URI do DagsHub
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        
        model_uri = f'runs:/{run_id}/model'
        logging.info(f"Carregando modelo da URI: {model_uri}")
        
        # Exemplo de dados de entrada (deve ser ajustado conforme a estrutura do modelo)
        # Baseado no SalesForecasterV2, as features incluem lag e rolling features.
        # Aqui criamos um exemplo genérico para teste de carregamento.
        input_data = pd.DataFrame({
            'semana': [1],
            'pdv': [1],
            'sku': [1],
            'trimestre': [1],
            'seno_semana': [0.12],
            'cosseno_semana': [0.99],
            'lag_1_semanas': [10.0],
            'lag_2_semanas': [12.0],
            'lag_3_semanas': [11.0],
            'lag_4_semanas': [13.0],
            'lag_12_semanas': [15.0],
            'lag_52_semanas': [20.0],
            'rolling_mean_4_semanas': [11.5],
            'rolling_std_4_semanas': [1.2],
            'rolling_max_4_semanas': [13.0],
            'rolling_mean_12_semanas': [12.5],
            'rolling_std_12_semanas': [2.1],
            'rolling_max_12_semanas': [15.0],
            'rolling_mean_52_semanas': [14.0],
            'rolling_std_52_semanas': [3.5],
            'rolling_max_52_semanas': [20.0]
        })

        # Converter colunas categóricas se necessário
        for col in ['pdv', 'sku']:
            input_data[col] = input_data[col].astype('category')

        logging.info("Realizando predição de teste com mlflow.pyfunc...")
        model = mlflow.pyfunc.load_model(model_uri)
        predictions = model.predict(input_data)
        logging.info(f"Predições realizadas com sucesso: {predictions}")

        logging.info("Validando modelo com mlflow.models.predict (usando env_manager='virtualenv')...")
        # Nota: env_manager='uv' ou 'virtualenv' depende do ambiente instalado
        try:
            mlflow.models.predict(
                model_uri=model_uri,
                input_data=input_data,
                env_manager="virtualenv"
            )
            logging.info("Validação de ambiente concluída com sucesso.")
        except Exception as e:
            logging.warning(f"A validação de ambiente falhou (isso é comum se as dependências não estiverem isoladas): {e}")

    except Exception as e:
        logging.error(f"Erro na validação do modelo: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Valida um modelo do MLflow.")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID do modelo no MLflow.")
    
    args = parser.parse_args()
    validate_best_model(args.run_id)
