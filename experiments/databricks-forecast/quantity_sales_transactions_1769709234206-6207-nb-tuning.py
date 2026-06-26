# Databricks notebook source
# MAGIC %md
# MAGIC # Tuning Notebook

# COMMAND ----------

# Install specified libraries
%pip install --upgrade --no-cache-dir --no-deps databricks-automl-runtime==0.2.20.13
dbutils.library.restartPython()

# COMMAND ----------

import databricks.automl
print(databricks.automl.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC Read input parameters from preprocessing task

# COMMAND ----------

preprocessing_task_key = "preprocessing"

# COMMAND ----------

# Params that are python primitive types can be loaded directly
primitive_param_keys = [
  "data_run_id",
  "experiment_id",
  "experiment_dir",
  "session_id",
  "exploration_notebook_url"
]

kwargs = {}
for key in primitive_param_keys:
    kwargs[key] = dbutils.jobs.taskValues.get(taskKey=preprocessing_task_key, key=key)
print(kwargs)

# COMMAND ----------

# Construct the preprocess_result from serialized value
from databricks.automl.core.forecast.stats import ForecastPostSamplingStats

preprocess_result_key = "preprocess_result"
preprocess_result_dict = dbutils.jobs.taskValues.get(taskKey=preprocessing_task_key, key=preprocess_result_key)
preprocess_result = ForecastPostSamplingStats.from_dict(preprocess_result_dict)
kwargs[preprocess_result_key] = preprocess_result

# COMMAND ----------

# Construct the parsed_input_parameters from serialized value
from databricks.automl.core.forecast.parameters import ParsedInputParameters

parsed_input_parameters_key = "parsed_input_parameters"
parsed_input_parameters_dict = dbutils.jobs.taskValues.get(taskKey=preprocessing_task_key, key=parsed_input_parameters_key)
parsed_input_parameters = ParsedInputParameters.from_dict(parsed_input_parameters_dict)
kwargs[parsed_input_parameters_key] = parsed_input_parameters

# COMMAND ----------

# Start tuning
from databricks.automl.core import forecast
summary = forecast.tune(**kwargs)


# COMMAND ----------

# Pass params for the following Training task
params = {
    "best_trial_run_id": summary.best_trial.mlflow_run_id,
    "data_run_id": kwargs["data_run_id"],
    "experiment_id": kwargs["experiment_id"],
    "experiment_dir": kwargs["experiment_dir"],
    "session_id": kwargs["session_id"],
    "exploration_notebook_url": kwargs["exploration_notebook_url"],
    "preprocess_result": kwargs[preprocess_result_key].to_dict(),
    "parsed_input_parameters": kwargs["parsed_input_parameters"].to_dict(),
}

for key, value in params.items():
    dbutils.jobs.taskValues.set(key, value)