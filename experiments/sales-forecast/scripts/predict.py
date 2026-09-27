import argparse
import logging
import os
import pandas as pd
from datetime import datetime
import joblib
from forecaster_class import SalesForecasterV2

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Function to generate the forecast file from a trained model.
def main(model_path: str, data_path: str, output_path: str, full_forecast: bool):
    logging.info("Starting the Forecasting Pipeline.")
    
    try:
        # --- STEP 1: LOAD MODEL AND DATA ---
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
        predictor.performance_metrics = artifacts.get('performance_metrics', {})
        predictor.best_params = artifacts.get('best_params', {})
        predictor.use_log_target = artifacts.get('use_log_target', False)
        logging.info(f"Model and artifacts loaded successfully. Recorded MAE: {predictor.performance_metrics.get('validation_mae', 'N/A')}")
        df_full_data = predictor.load_data(file_paths)
        df_historical_2022 = df_full_data[df_full_data['ano'] == 2022].copy()

        # --- STEP 2: GENERATE THE FULL FORECAST IN MEMORY ---
        logging.info("Generating the full forecast in memory for all combinations...")
        forecasts_completos = predictor.generate_forecasts(df_historical_2022, weeks_to_forecast=5)
        
        final_forecasts = forecasts_completos

        # --- STEP 3: APPLY THE ROW LIMIT (UNLESS IT IS A FULL FORECAST) ---
        if not full_forecast:
            logging.info("Applying the 1.5M-row limit to the submission file...")
            if not forecasts_completos.empty:
                importancia_futura = forecasts_completos.groupby(['pdv', 'sku'])['quantidade_prevista'].sum().reset_index()
                top_combinacoes_futuras = importancia_futura.nlargest(300000, 'quantidade_prevista')
                forecasts_filtrado = pd.merge(forecasts_completos, top_combinacoes_futuras[['pdv', 'sku']], on=['pdv', 'sku'], how='inner')
                final_forecasts = forecasts_filtrado
                logging.info(f"Forecasts filtered to the {len(top_combinacoes_futuras)} most promising combinations.")
            else:
                logging.warning("The full forecast was empty. No filtering applied.")
        else:
            logging.info("Generating the full forecast for all products (no row limit).")
            
        # --- STEP 4: FORMAT AND SAVE THE FINAL FILE ---
        if not final_forecasts.empty:
            df_submission = final_forecasts.rename(columns={'sku': 'produto', 'quantidade_prevista': 'quantidade'})
            df_submission = df_submission[['semana', 'pdv', 'produto', 'quantidade']]
            df_submission_sorted = df_submission.sort_values(by=['semana', 'quantidade'], ascending=[True, False])
            
            logging.info(f"Final forecast generated with {len(df_submission_sorted)} rows.")
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Set the filename based on the flag
            if full_forecast:
                filename_suffix = "FULL"
            else:
                filename_suffix = "SUBMISSION"

            submission_filename = os.path.join(output_path, f"forecast_{filename_suffix}_{timestamp}.parquet")
            df_submission_sorted.to_parquet(submission_filename, index=False)
            logging.info(f"Forecast file saved to: {submission_filename}")
        else:
            logging.warning("No forecast was generated.")

    except FileNotFoundError:
        logging.error(f"Model file not found at '{model_path}'. Run train.py first.")
        return
    except Exception as e:
        logging.error(f"The forecasting pipeline failed with error: {e}")
        raise e

    logging.info("Forecasting Pipeline finished successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the forecast file from a trained model.")
    parser.add_argument("--model_path", type=str, default="artifacts/sales_forecaster_v2_final.joblib", help="Path to the trained model.")
    parser.add_argument("--data_path", type=str, default="data", help="Path to the 'data' folder.")
    parser.add_argument("--output_path", type=str, default="data/processed", help="Path to save the final forecast.")
    parser.add_argument(
        "--full_forecast",
        action="store_true",
        help="If specified, generate the full forecast, ignoring the 1.5M-row limit."
    )
    
    args = parser.parse_args()
    
    main(args.model_path, args.data_path, args.output_path, args.full_forecast)
