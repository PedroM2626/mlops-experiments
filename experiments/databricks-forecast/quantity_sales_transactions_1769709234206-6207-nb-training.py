# Databricks notebook source
# MAGIC %md
# MAGIC # Training Notebook

# COMMAND ----------

# Install specified libraries
%pip install --upgrade --no-cache-dir --no-deps databricks-automl-runtime==0.2.20.13
%pip install --no-cache-dir --no-deps gluonts[torch]==0.15.1 lightning==2.1.4 lightning-utilities==0.11.8 pytorch-lightning==2.1.4 toolz==0.12.1 torchmetrics==1.5.1
%pip install --upgrade mlflow==2.20.0
%pip install --force-reinstall numpy==1.26.4
dbutils.library.restartPython()

# COMMAND ----------

import databricks.automl
print(databricks.automl.__version__)

# COMMAND ----------

tuning_task_key = "tuning"

# COMMAND ----------

# Params that are python primitive types can be loaded directly
primitive_param_keys = [
  "best_trial_run_id",
  "data_run_id",
  "experiment_id",
  "experiment_dir",
  "session_id",
  "exploration_notebook_url",
]

kwargs = {}
for key in primitive_param_keys:
    kwargs[key] = dbutils.jobs.taskValues.get(taskKey=tuning_task_key, key=key)
print(kwargs)


# COMMAND ----------

# Construct the preprocess_result from serialized value
from databricks.automl.core.forecast.stats import ForecastPostSamplingStats

preprocess_result_key = "preprocess_result"
preprocess_result_dict = dbutils.jobs.taskValues.get(taskKey=tuning_task_key, key=preprocess_result_key)
preprocess_result = ForecastPostSamplingStats.from_dict(preprocess_result_dict)
kwargs[preprocess_result_key] = preprocess_result

# COMMAND ----------

# Construct the parsed_input_parameters from serialized value
from databricks.automl.core.forecast.parameters import ParsedInputParameters

parsed_input_parameters_key = "parsed_input_parameters"
parsed_input_parameters_dict = dbutils.jobs.taskValues.get(taskKey=tuning_task_key, key=parsed_input_parameters_key)
parsed_input_parameters = ParsedInputParameters.from_dict(parsed_input_parameters_dict)
kwargs[parsed_input_parameters_key] = parsed_input_parameters

# COMMAND ----------

from databricks.automl.core import forecast
summary = forecast.train(**kwargs)