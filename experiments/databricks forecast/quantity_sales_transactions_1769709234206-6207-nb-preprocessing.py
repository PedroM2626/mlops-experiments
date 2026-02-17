# Databricks notebook source
# MAGIC %md
# MAGIC # Data Processing Notebook

# COMMAND ----------

# Install specified libraries
%pip install --upgrade --no-cache-dir --no-deps databricks-automl-runtime==0.2.20.13
dbutils.library.restartPython()

# COMMAND ----------

import databricks.automl
print(databricks.automl.__version__)

# COMMAND ----------

from typing import List

# conversion functions
def identity(param: str) -> str:
  return param

def to_int(param: str) -> int:
  return int(param)

def split_by_comma(param: str) -> List[str]:
  return param.split(",")


# COMMAND ----------

from __future__ import annotations
from enum import Enum
from typing import Optional, Callable, Any, Set

class Param(Enum):
  widget_name: str
  automl_param_name: str
  from_string_fn: Optional[Callable[[str], Any]]

  def __init__(
    self,
    widget_name: str,
    fit_param_name: Optional[str],
    from_string: Callable[[str], Any]=identity
  ):
    self.widget_name = widget_name
    self.fit_param_name = fit_param_name
    self.from_string = from_string

  PROBLEM_TYPE = (
    "problemType",
    None,
  )
  TABLE_NAME = (
    "tableName",
    "table_name"
  )
  EXPERIMENT_ID = (
    "experimentId",
    "experiment_id"
  )
  TARGET_COL = (
    "targetCol",
    "target_col"
  )
  TIME_COL = (
    "timeCol",
    "time_col"
  )
  PRIMARY_METRIC = (
    "primaryMetric",
    "metric"
  )
  EXCLUDE_FRAMEWORKS = (
    "excludeFrameworks",
    "exclude_frameworks",
    split_by_comma
  )
  EXPERIMENT_DIR = (
    "experimentDir",
    "experiment_dir"
  )
  TIMEOUT_MINUTES = (
    "timeoutMinutes",
    "timeout_minutes",
    to_int
  )
  HORIZON = (
    "horizon",
    "horizon",
    to_int
  )
  FREQUENCY = (
    "frequency",
    "frequency"
  )
  FREQUENCY_QUANTITY = (
    "frequencyQuantity",
    "frequency_quantity",
    to_int
  )
  IDENTITY_COLS = (
    "identityCols",
    "identity_col",
    split_by_comma
  )
  COUNTRY_CODE = (
    "countryCode",
    "country_code",
  )
  SPLIT_COL = (
    "splitCol",
    "split_col",
  )
  SAMPLE_WEIGHT_COL = (
    "sampleWeightCol",
    "sample_weight_col",
  )
  OUTPUT_DATABASE = (
    "outputDatabase",
    "output_database",
  )
  MODEL_REGISTER_TO_LOCATION = (
    "modelRegisterToLocation",
    "model_register_to_location",
  )
  INCLUDE_FEATURES = (
    "includeFeatures",
    "include_features",
    split_by_comma
  )
  FUTURE_FEATURE_DATA_PATH = (
    "futureFeatureDataPath",
    "future_feature_data_path",
  )

# COMMAND ----------

# initialize widgets
for param in Param:
  dbutils.widgets.text(param.widget_name, "")

# COMMAND ----------

# fetch input param values from widget
params = {}
for param in Param:
  value = dbutils.widgets.get(param.widget_name)
  if value != "":
    params[param] = param.from_string(value)
params

# COMMAND ----------

# MAGIC %md
# MAGIC Build the fit params

# COMMAND ----------

# start with values that can directly be input from the params
kwargs = {k.fit_param_name: v for k, v in params.items() if k.fit_param_name is not None}
kwargs

# COMMAND ----------

# read dataset from the given table name
table_name_segments = params[Param.TABLE_NAME].split('.')
escaped_table_name_segments = map(lambda s: f"`{s}`", table_name_segments)
escaped_table_name = '.'.join(escaped_table_name_segments)
dataset = spark.table(escaped_table_name)
kwargs["dataset"] = dataset
dataset.printSchema()

# COMMAND ----------

# Start preprocessing
from databricks.automl.core import forecast
parsed_params, preprocess_result, data_run_id, session_id, exploration_notebook_url = forecast.preprocess(**kwargs)

# COMMAND ----------

# Pass params for the following Tuning task
tuning_params = {
    "data_run_id": data_run_id,
    "experiment_id": kwargs["experiment_id"],
    "experiment_dir": kwargs["experiment_dir"],
    "session_id": session_id,
    "exploration_notebook_url": exploration_notebook_url,
    "preprocess_result": preprocess_result.to_dict(),
    "parsed_input_parameters": parsed_params.to_dict(),
}

# set task values so that next task can read
# task value limitation: the values can be serialized by json and does not exceed 48KiB.

for key, value in tuning_params.items():
    dbutils.jobs.taskValues.set(key, value)