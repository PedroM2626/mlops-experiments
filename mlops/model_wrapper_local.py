"""Fallback: carrega o joblib committed direto quando o registry MLflow esta vazio."""
import os
import joblib
import pandas as pd
from .model_wrapper import SalesForecasterPyfunc, DATA_PATHS, SALES_DIR


class _LocalPyfunc(SalesForecasterPyfunc):
    """Mesma interface do pyfunc, mas carrega do joblib em vez do artifacts context."""

    def __init__(self, joblib_path):
        super().__init__()
        self.joblib_path = joblib_path
        self.artifacts = joblib.load(joblib_path)
        from forecaster_class import SalesForecasterV2
        self.forecaster = SalesForecasterV2()
        self.forecaster.model = self.artifacts["model"]
        self.forecaster.feature_names = self.artifacts["feature_names"]
        self.forecaster.categorical_features = self.artifacts["categorical_features"]
        self.forecaster.use_log_target = self.artifacts.get("use_log_target", False)
        self._init_caches()

    def predict(self, context, model_input):
        return super().predict(None, model_input)


def load_local_pyfunc(joblib_path):
    return _LocalPyfunc(joblib_path)
