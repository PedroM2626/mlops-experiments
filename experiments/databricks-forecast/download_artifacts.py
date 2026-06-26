import mlflow
import os
import logging
from dotenv import load_dotenv

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Carregar variáveis de ambiente do .env na raiz do projeto
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path)

# Configurar MLflow para usar Databricks
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST", "https://dbc-a793141b-98d6.cloud.databricks.com")
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN", "dapibfbbee20bbb676579e2893a173c17e82")

# Importante para Databricks
mlflow.set_tracking_uri("databricks")

experiment_id = "3257488039771013"
max_results = 1000

logging.info(f"Conectando ao Databricks: {os.environ['DATABRICKS_HOST']}")
logging.info(f"Buscando runs para o experimento: {experiment_id}")

try:
    runs = mlflow.search_runs(experiment_ids=[experiment_id], max_results=max_results)
    logging.info(f"Encontradas {len(runs)} runs.")
    
    # Pasta local para salvar os artefatos
    local_dir = os.path.dirname(os.path.abspath(__file__))
    
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_name = run.get("tags.mlflow.runName", run_id)
        dst_path = os.path.join(local_dir, f"run_{run_id}")
        os.makedirs(dst_path, exist_ok=True)
        
        logging.info(f"Baixando artefatos da run {run_name} para {dst_path}")
        try:
            mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=dst_path)
            logging.info(f"✅ Download concluído para run {run_id}")
        except Exception as e:
            logging.error(f"❌ Erro ao baixar artefatos da run {run_id}: {e}")

except Exception as e:
    logging.error(f"Falha ao buscar runs: {e}")
