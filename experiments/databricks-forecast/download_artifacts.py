import mlflow
import os
import logging
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from the .env at the project root
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path)

# Configure MLflow to use Databricks
os.environ["DATABRICKS_HOST"] = os.getenv("DATABRICKS_HOST", "https://dbc-a793141b-98d6.cloud.databricks.com")
if not os.getenv("DATABRICKS_TOKEN"):
    raise RuntimeError(
        "DATABRICKS_TOKEN is not set. Put it in the project-root .env "
        "(see .env.example); never hardcode it in this file."
    )

# Important for Databricks
mlflow.set_tracking_uri("databricks")

experiment_id = "3257488039771013"
max_results = 1000

logging.info(f"Connecting to Databricks: {os.environ['DATABRICKS_HOST']}")
logging.info(f"Fetching runs for the experiment: {experiment_id}")

try:
    runs = mlflow.search_runs(experiment_ids=[experiment_id], max_results=max_results)
    logging.info(f"Found {len(runs)} runs.")
    
    # Local folder to save the artifacts
    local_dir = os.path.dirname(os.path.abspath(__file__))
    
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_name = run.get("tags.mlflow.runName", run_id)
        dst_path = os.path.join(local_dir, f"run_{run_id}")
        os.makedirs(dst_path, exist_ok=True)
        
        logging.info(f"Downloading artifacts of run {run_name} to {dst_path}")
        try:
            mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=dst_path)
            logging.info(f"✅ Download completed for run {run_id}")
        except Exception as e:
            logging.error(f"❌ Error downloading artifacts of run {run_id}: {e}")

except Exception as e:
    logging.error(f"Failed to fetch runs: {e}")
