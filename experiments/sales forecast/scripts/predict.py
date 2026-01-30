import argparse
import logging
import os
import pandas as pd
from datetime import datetime
import joblib
from forecaster_class import SalesForecasterV2

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Função para gerar o arquivo de previsão a partir de um modelo treinado.
def main(model_path: str, data_path: str, output_path: str, full_forecast: bool):
    logging.info("Iniciando o Pipeline de Previsão.")
    
    try:
        # --- ETAPA 1: CARREGAR MODELO E DADOS ---
        file_paths = {
            'vendas': os.path.join(data_path, 'raw/fato_vendas.parquet'),
            'pdvs': os.path.join(data_path, 'raw/dim_pdvs.parquet'),
            'produtos': os.path.join(data_path, 'raw/dim_produtos.parquet')
        }
        artifacts = joblib.load(model_path)
        predictor = SalesForecasterV2()
        predictor.model = artifacts['model']
        predictor.feature_names = artifacts['feature_names']
        predictor.categorical_features = artifacts['categorical_features']
        logging.info("Modelo e artefatos carregados com sucesso.")
        df_full_data = predictor.load_data(file_paths)
        df_historical_2022 = df_full_data[df_full_data['ano'] == 2022].copy()

        # --- ETAPA 2: GERAR A PREVISÃO COMPLETA EM MEMÓRIA ---
        logging.info("Gerando previsão completa em memória para todas as combinações...")
        forecasts_completos = predictor.generate_forecasts(df_historical_2022, weeks_to_forecast=5)
        
        final_forecasts = forecasts_completos

        # --- ETAPA 3: APLICAR LIMITE DE LINHAS (SE NÃO FOR PREVISÃO COMPLETA) ---
        if not full_forecast:
            logging.info("Aplicando limite de 1.5M de linhas para o arquivo de submissão...")
            if not forecasts_completos.empty:
                importancia_futura = forecasts_completos.groupby(['pdv', 'sku'])['quantidade_prevista'].sum().reset_index()
                top_combinacoes_futuras = importancia_futura.nlargest(300000, 'quantidade_prevista')
                forecasts_filtrado = pd.merge(forecasts_completos, top_combinacoes_futuras[['pdv', 'sku']], on=['pdv', 'sku'], how='inner')
                final_forecasts = forecasts_filtrado
                logging.info(f"Previsões filtradas para as {len(top_combinacoes_futuras)} combinações mais promissoras.")
            else:
                logging.warning("Previsão completa estava vazia. Nenhuma filtragem aplicada.")
        else:
            logging.info("Gerando previsão completa para todos os produtos (sem limite de linhas).")
            
        # --- ETAPA 4: FORMATAR E SALVAR O ARQUIVO FINAL ---
        if not final_forecasts.empty:
            df_submission = final_forecasts.rename(columns={'sku': 'produto', 'quantidade_prevista': 'quantidade'})
            df_submission = df_submission[['semana', 'pdv', 'produto', 'quantidade']]
            df_submission_sorted = df_submission.sort_values(by=['semana', 'quantidade'], ascending=[True, False])
            
            logging.info(f"Previsão final gerada com {len(df_submission_sorted)} linhas.")
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Define o nome do arquivo com base na flag
            if full_forecast:
                filename_suffix = "COMPLETA"
            else:
                filename_suffix = "SUBMISSAO"

            submission_filename = os.path.join(output_path, f"previsao_{filename_suffix}_{timestamp}.parquet")
            df_submission_sorted.to_parquet(submission_filename, index=False)
            logging.info(f"Arquivo de previsão salvo em: {submission_filename}")
        else:
            logging.warning("Nenhuma previsão foi gerada.")

    except FileNotFoundError:
        logging.error(f"Arquivo do modelo não encontrado em '{model_path}'. Execute o train.py primeiro.")
        return
    except Exception as e:
        logging.error(f"O pipeline de previsão falhou com o erro: {e}")
        raise e

    logging.info("Pipeline de Previsão finalizado com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera o arquivo de previsão a partir de um modelo treinado.")
    parser.add_argument("--model_path", type=str, default="artifacts/sales_forecaster_v2_final.joblib", help="Caminho para o modelo treinado.")
    parser.add_argument("--data_path", type=str, default="data", help="Caminho para a pasta 'data'.")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Caminho para salvar a previsão final.")
    parser.add_argument(
        "--full_forecast",
        action="store_true",
        help="Se especificado, gera a previsão completa, ignorando o limite de 1.5M de linhas."
    )
    
    args = parser.parse_args()
    
    main(args.model_path, args.data_path, args.output_path, args.full_forecast)
