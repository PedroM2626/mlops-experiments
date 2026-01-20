#!/usr/bin/env python3
"""
🎯 MLOPS ENTERPRISE - UNIVERSAL FRAMEWORK (V6.0)
Recursos Avançados:
- AutoML: Unified (TPOT, AutoGluon, FLAML, Auto-sklearn, H2O).
- Otimização (Optuna), Explainability (SHAP/LIME).
- Validação (Evidently), Exportação (ONNX).
- Integrações: MLflow, DagsHub, W&B, HuggingFace, ZenML.
- Distributed Training (PyTorch), K8s Deployment Ready.
- CV (YOLOv8), NLP (Transformers), Time Series (Prophet).
"""

import os
import warnings
import argparse

# Forçar Transformers a usar PyTorch e evitar conflitos com Keras 3 no Python 3.13
os.environ["USE_TF"] = "0"
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv
import logging
from typing import Optional, Dict, Any, Union

# MLOps & Tracking
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.h2o
import dagshub
import wandb
import optuna

try:
    import zenml
    from zenml import pipeline, step
    HAS_ZENML = True
except ImportError:
    HAS_ZENML = False

# AutoML Engines
try:
    from tpot import TPOTClassifier, TPOTRegressor
    HAS_TPOT = True
except ImportError:
    HAS_TPOT = False

try:
    from autogluon.tabular import TabularPredictor
    HAS_AUTOGLUON = True
except ImportError:
    HAS_AUTOGLUON = False

try:
    from flaml import AutoML
    HAS_FLAML = True
except ImportError:
    HAS_FLAML = False

try:
    import autosklearn.classification
    import autosklearn.regression
    HAS_AUTOSKLEARN = True
except ImportError:
    HAS_AUTOSKLEARN = False

try:
    import h2o
    from h2o.automl import H2OAutoML
    HAS_H2O = True
except ImportError:
    HAS_H2O = False

# Deep Learning & Transformers
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import models, transforms
from PIL import Image
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import cv2
from scipy.spatial.distance import cosine

# Missing Engines
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

# Explainability & Monitoring
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    HAS_EVIDENTLY = True
except ImportError:
    HAS_EVIDENTLY = False

# ML Clássico & Export
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# Computer Vision
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLOpsEnterprise:
    def __init__(self, repo_owner='PedroM2626', repo_name='experiments'):
        load_dotenv()
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_integrations()

    def _setup_integrations(self):
        try:
            # 1. DagsHub & MLflow Setup
            dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
            logger.info("✅ Conectado ao DagsHub/MLflow")
            
            # 2. Gerar conda.yaml e requirements.txt se não existirem (para o MLflow loggar o ambiente)
            if not os.path.exists("requirements.txt"):
                with open("requirements.txt", "w") as f:
                    f.write("numpy\npandas\nscikit-learn\nmlflow\ndagshub\n")

            if not os.path.exists("conda.yaml"):
                with open("conda.yaml", "w") as f:
                    f.write("name: mlops_env\nchannels:\n  - defaults\ndependencies:\n  - python=3.10\n  - pip:\n    - -r requirements.txt\n")

            # 3. Configurar logging automático de ambiente (será loggado em cada run)
            # mlflow.log_artifact("requirements.txt")  # Movido para dentro do context manager de run
            # mlflow.log_artifact("conda.yaml")
            
            if os.getenv("WANDB_API_KEY"):
                wandb.login(key=os.getenv("WANDB_API_KEY"))
                wandb.init(project=self.repo_name, entity=self.repo_owner)
                logger.info("✅ Conectado ao Weights & Biases")
            
            if HAS_ZENML:
                try:
                    # Inicialização básica do ZenML se necessário
                    os.system("zenml init")
                    logger.info("✅ ZenML Inicializado")
                except:
                    pass
        except Exception as e:
            logger.warning(f"⚠️ Erro nas integrações: {e}")

    def _log_env_artifacts(self):
        """Registra arquivos de ambiente no run atual do MLflow."""
        if mlflow.active_run():
            if os.path.exists("requirements.txt"):
                mlflow.log_artifact("requirements.txt")
            if os.path.exists("conda.yaml"):
                mlflow.log_artifact("conda.yaml")
            if os.path.exists(".env.example"):
                mlflow.log_artifact(".env.example")
            logger.info("📦 Artefatos de ambiente (requirements, conda, env) registrados.")

    def _log_metrics_and_plots(self, metrics, artifacts=None):
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            if wandb.run: wandb.log({name: value})
        if artifacts:
            for path in artifacts:
                mlflow.log_artifact(path)

    def validate_data(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Valida a integridade dos dados antes do treino e remove valores inválidos."""
        logger.info("🔍 Validando integridade dos dados...")
        if df.empty:
            raise ValueError("O DataFrame está vazio.")
        if target not in df.columns:
            raise ValueError(f"Target '{target}' não encontrado no DataFrame.")
        
        # Remover NaNs e Infs da coluna target
        initial_len = len(df)
        df = df[np.isfinite(df[target])]
        
        if len(df) < initial_len:
            logger.warning(f"⚠️ Removidos {initial_len - len(df)} registros com valores não finitos (NaN/Inf) no target.")

        null_counts = df.isnull().sum().sum()
        if null_counts > (len(df) * len(df.columns) * 0.5):
            logger.warning(f"⚠️ Alto índice de valores nulos detectado: {null_counts}")
            
        return df

    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame):
        """Detecta drift de dados usando Evidently AI."""
        if not HAS_EVIDENTLY:
            logger.warning("⚠️ Evidently AI não instalado. Pulando análise de drift.")
            return None
            
        logger.info("📉 Analisando Data Drift...")
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report_path = "drift_report.html"
        report.save_html(report_path)
        mlflow.log_artifact(report_path)
        return report_path

    # --- MÓDULO AUTOML UNIFICADO (TPOT, AutoGluon, FLAML, Auto-sklearn, H2O) ---
    def train_automl(self, data_path, target=None, task='classification', engine='flaml', timeout=60):
        """
        Engine Universal de AutoML.
        engines: 'tpot', 'autogluon', 'flaml', 'autosklearn', 'h2o', 'unified'
        """
        if engine == 'unified':
            return self._train_unified_automl(data_path, target, task, timeout)

        if engine == 'autosklearn' and not HAS_AUTOSKLEARN:
            logger.error("❌ auto-sklearn não está instalado ou não é compatível com este Sistema Operacional (Requer Linux).")
            return None, None

        logger.info(f"\n🤖 Iniciando AutoML ({task}) com engine: {engine.upper()}...")
        df = pd.read_csv(data_path)
        
        # Se target não for especificado, assume a última coluna
        if target is None:
            target = df.columns[-1]
            logger.info(f"🎯 Target não especificado. Usando última coluna: {target}")
        
        df = self.validate_data(df, target)
        
        mlflow.set_experiment(f"/automl_{engine}")
        with mlflow.start_run(run_name=f"{engine}_run_{datetime.now().strftime('%H%M%S')}"):
            mlflow.log_param("engine", engine)
            mlflow.log_param("timeout", timeout)
            mlflow.log_param("task", task)
            self._log_env_artifacts()

            X = df.drop(columns=[target])
            y = df[target]
            
            # Tratamento básico para motores que não lidam com texto automaticamente (exceto AutoGluon/H2O)
            if engine in ['tpot', 'flaml', 'autosklearn'] and X.select_dtypes(include=['object']).any().any():
                logger.info("⚙️ Detectado colunas de texto. Aplicando codificação básica...")
                X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            best_model = None
            score = 0

            if engine == 'tpot' and HAS_TPOT:
                model = TPOTClassifier(max_time_mins=timeout//60, verbosity=2) if task == 'classification' \
                        else TPOTRegressor(max_time_mins=timeout//60, verbosity=2)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                best_model = model.fitted_pipeline_
                mlflow.sklearn.log_model(best_model, "model")
                
            elif engine == 'autogluon' and HAS_AUTOGLUON:
                train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
                ag_task = task
                if task == 'classification':
                    unique_count = df[target].nunique()
                    ag_task = 'binary' if unique_count <= 2 else 'multiclass'
                
                model = TabularPredictor(label=target, problem_type=ag_task).fit(train_data, time_limit=timeout)
                performance = model.evaluate(test_data)
                score = performance.get('accuracy') or performance.get('f1') or performance.get('root_mean_squared_error')
                model.save("ag_models")
                mlflow.log_artifacts("ag_models", artifact_path="autogluon_models")
                best_model = model
                
            elif engine == 'flaml' and HAS_FLAML:
                model = AutoML()
                model.fit(X_train=X_train, y_train=y_train, task=task, time_budget=timeout, metric='auto', verbose=0)
                best_model = model.model.estimator
                preds = best_model.predict(X_test)
                score = accuracy_score(y_test, preds) if task == 'classification' else r2_score(y_test, preds)
                mlflow.sklearn.log_model(best_model, "model")

            elif engine == 'autosklearn' and HAS_AUTOSKLEARN:
                if task == 'classification':
                    model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=timeout)
                else:
                    model = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=timeout)
                
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                best_model = model
                mlflow.sklearn.log_model(best_model, "model")

            elif engine == 'h2o' and HAS_H2O:
                h2o.init()
                h2o_train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
                h2o_test = h2o.H2OFrame(pd.concat([X_test, y_test], axis=1))
                
                model = H2OAutoML(max_runtime_secs=timeout, seed=42)
                model.train(y=target, training_frame=h2o_train)
                
                best_model = model.leader
                perf = best_model.model_performance(h2o_test)
                score = perf.accuracy()[0][1] if task == 'classification' else perf.r2()
                
                mlflow.h2o.log_model(best_model, "model")
                model_path = h2o.save_model(model=best_model, path="h2o_models", force=True)
                mlflow.log_artifact(model_path)
                logger.info(f"🏆 Melhor modelo H2O: {best_model.model_id}")
                # Não fechar o cluster se for usar para inferência imediata, mas aqui fechamos por segurança
                h2o.cluster().shutdown()
            
            else:
                logger.warning(f"⚠️ Engine {engine} não disponível ou não instalada.")
                return None, None

            self._log_metrics_and_plots({"best_score": score})
            
            # Exportar ONNX se for sklearn-like
            if engine in ['flaml', 'tpot', 'autosklearn']:
                try:
                    self.export_to_onnx(best_model, X_train.iloc[:1])
                except Exception as e:
                    logger.warning(f"⚠️ Erro ao exportar ONNX: {e}")

            logger.info(f"✅ AutoML ({engine}) concluído. Score: {score}")
            return best_model, score

    def _train_unified_automl(self, data_path, target, task, timeout):
        """Executa múltiplos motores de AutoML e escolhe o melhor."""
        engines = ['flaml']
        if HAS_AUTOGLUON: engines.append('autogluon')
        if HAS_TPOT: engines.append('tpot')
        
        logger.info(f"🧬 Iniciando AutoML UNIFICADO com engines: {engines}")
        best_overall_model = None
        best_overall_score = -1
        
        timeout_per_engine = timeout // len(engines)
        
        for eng in engines:
            try:
                model, score = self.train_automl(data_path, target, task, eng, timeout_per_engine)
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_model = model
            except Exception as e:
                logger.warning(f"⚠️ Falha no engine {eng}: {e}")
                
        logger.info(f"🏆 Vencedor do AutoML Unificado: {best_overall_score:.4f}")
        return best_overall_model, best_overall_score

    def export_to_onnx(self, model, sample_input):
        """Exporta o modelo para formato ONNX."""
        logger.info("🚀 Exportando para ONNX...")
        initial_type = [('float_input', FloatTensorType([None, sample_input.shape[1]]))]
        onx = convert_sklearn(model, initial_types=initial_type)
        with open("model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
        mlflow.log_artifact("model.onnx")

    # --- MÓDULO MANUAL: OPTUNA & DEEP LEARNING ---
    def train_manual(self, data_path, target, model_type='rf', use_optuna=False, task='classification'):
        """Treinamento manual com escolha de modelo e otimização opcional."""
        logger.info(f"🛠️ Treino Manual: {model_type} | Optuna: {use_optuna}")
        df = pd.read_csv(data_path)
        df = self.validate_data(df, target)
        
        X = df.drop(columns=[target])
        y = df[target]
        
        if X.select_dtypes(include=['object']).any().any():
            X = pd.get_dummies(X)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        mlflow.set_experiment(f"/manual_training")
        with mlflow.start_run(run_name=f"{model_type}_manual_{datetime.now().strftime('%H%M%S')}"):
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("use_optuna", use_optuna)
            mlflow.log_param("task", task)
            self._log_env_artifacts()

            def objective(trial):
                if model_type == 'rf':
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    n_estimators = trial.suggest_int('n_estimators', 10, 200)
                    max_depth = trial.suggest_int('max_depth', 2, 32)
                    if task == 'classification':
                        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
                    else:
                        clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
                elif model_type == 'xgb':
                    from xgboost import XGBClassifier, XGBRegressor
                    param = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
                    }
                    clf = XGBClassifier(**param) if task == 'classification' else XGBRegressor(**param)
                
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                return accuracy_score(y_test, preds) if task == 'classification' else r2_score(y_test, preds)

            best_model = None
        if use_optuna:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10)
            logger.info(f"🏆 Melhores hiperparâmetros: {study.best_params}")
            # Treinar modelo final com melhores params
            if model_type == 'rf':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                best_model = RandomForestClassifier(**study.best_params) if task == 'classification' \
                             else RandomForestRegressor(**study.best_params)
            elif model_type == 'xgb':
                from xgboost import XGBClassifier, XGBRegressor
                best_model = XGBClassifier(**study.best_params) if task == 'classification' \
                             else XGBRegressor(**study.best_params)
            best_model.fit(X_train, y_train)
        else:
            # Sem optuna, usa default
            if model_type == 'rf':
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                best_model = RandomForestClassifier() if task == 'classification' else RandomForestRegressor()
            elif model_type == 'xgb':
                from xgboost import XGBClassifier, XGBRegressor
                best_model = XGBClassifier() if task == 'classification' else XGBRegressor()
            best_model.fit(X_train, y_train)

        preds = best_model.predict(X_test)
        score = accuracy_score(y_test, preds) if task == 'classification' else r2_score(y_test, preds)
        
        mlflow.log_metric("manual_score", score)
        mlflow.sklearn.log_model(best_model, "manual_model")
        
        return best_model, score

    def train_pytorch_tabular(self, data_path, target, epochs=10, task='classification'):
        """Treinamento de Deep Learning (PyTorch) para dados tabulares."""
        logger.info(f"🔥 Iniciando Deep Learning (PyTorch) Tabular...")
        df = pd.read_csv(data_path)
        df = self.validate_data(df, target)

        mlflow.set_experiment("/pytorch_deep_learning")
        with mlflow.start_run(run_name=f"pytorch_run_{datetime.now().strftime('%H%M%S')}"):
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("task", task)
            self._log_env_artifacts()

            X = df.drop(columns=[target]).values.astype(np.float32)
            y = df[target].values
            
            if task == 'classification':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
                output_dim = len(np.unique(y))
                y = y.astype(np.int64)
            else:
                output_dim = 1
                y = y.astype(np.float32).reshape(-1, 1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.LongTensor(y_train) if task == 'classification' else torch.FloatTensor(y_train)
            
            input_dim = X.shape[1]
            model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, output_dim)
            ).to(self.device)
            
            criterion = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t.to(self.device))
                loss = criterion(outputs, y_train_t.to(self.device))
                loss.backward()
                optimizer.step()
                if epoch % 5 == 0: logger.info(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")
                
            mlflow.pytorch.log_model(model, "pytorch_model")
            return model, "DL Training Complete"

    def get_system_status(self):
        """Retorna o status das conexões e hardware."""
        status = {
            "GPU_Available": torch.cuda.is_available(),
            "GPU_Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "MLflow_URI": mlflow.get_tracking_uri(),
            "DagsHub_Connected": self.repo_owner in mlflow.get_tracking_uri(),
            "WandB_Connected": wandb.run is not None,
            "Engines": {
                "TPOT": HAS_TPOT,
                "AutoGluon": HAS_AUTOGLUON,
                "FLAML": HAS_FLAML,
                "AutoSklearn": HAS_AUTOSKLEARN,
                "H2O": HAS_H2O,
                "Prophet": HAS_PROPHET,
                "YOLO": HAS_YOLO,
                "SHAP": HAS_SHAP,
                "LIME": HAS_LIME,
                "ZenML": HAS_ZENML
            }
        }
        return status

    def explain_model(self, model, X_train, method='shap'):
        """Gera explicações SHAP ou LIME para o modelo."""
        logger.info(f"🧠 Gerando explicações: {method.upper()}")
        
        if method == 'shap' and HAS_SHAP:
            explainer = shap.Explainer(model.predict, X_train.iloc[:100])
            shap_values = explainer(X_train.iloc[:100])
            plt.figure()
            shap.summary_plot(shap_values, X_train.iloc[:100], show=False)
            plt.savefig("shap_summary.png")
            mlflow.log_artifact("shap_summary.png")
            plt.close()
            return "shap_summary.png"
            
        elif method == 'lime' and HAS_LIME:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values, 
                feature_names=X_train.columns.tolist(),
                mode='classification' # Simplificado
            )
            exp = explainer.explain_instance(X_train.values[0], model.predict_proba)
            exp.save_to_file('lime_explanation.html')
            mlflow.log_artifact('lime_explanation.html')
            return 'lime_explanation.html'
            
        else:
            logger.warning(f"⚠️ Método {method} não disponível ou biblioteca não instalada.")
            return None

    def generate_serving_api(self, model_name="model.onnx"):
        """Gera um script Flask básico para servir o modelo ONNX."""
        serving_script = f"""
import flask
import onnxruntime as rt
import numpy as np

app = flask.Flask(__name__)
sess = rt.InferenceSession("{model_name}")
input_name = sess.get_inputs()[0].name

@app.route('/predict', methods=['POST'])
def predict():
    data = flask.request.json['data']
    preds = sess.run(None, {{input_name: np.array(data).astype(np.float32)}})
    return flask.jsonify({{'predictions': preds[0].tolist()}})

if __name__ == '__main__':
    app.run(port=5000)
"""
        with open("serving_api.py", "w") as f:
            f.write(serving_script)
        mlflow.log_artifact("serving_api.py")
        logger.info("🌐 API de serving gerada em serving_api.py")

    # --- MÓDULO UNIVERSAL: COMPUTER VISION (DETECÇÃO & SIMILARIDADE) ---
    def detect_objects(self, image_path: str, task="generic"):
        """Detecta objetos ou faces em uma imagem usando YOLOv8."""
        if not HAS_YOLO:
            logger.error("❌ YOLO não instalado.")
            return None
        
        logger.info(f"📸 Detectando objetos ({task}) em: {image_path}")
        # yolov8n.pt para geral, ou pode carregar outros
        model = YOLO('yolov8n.pt') 
        
        if task == "faces":
            # Usar modelo específico de face se disponível, senão filtrar por 'person'
            results = model(image_path, classes=[0]) 
        else:
            results = model(image_path) # Detecta todas as 80 classes do COCO
        
        # Salvar resultado com anotações
        res_path = f"{task}_detection_result.jpg"
        results[0].save(res_path)
        
        # Logar artefatos no DagsHub/MLflow
        mlflow.log_artifact(res_path)
        mlflow.log_artifact("requirements.txt")
        if os.path.exists("conda.yaml"):
            mlflow.log_artifact("conda.yaml")
            
        return res_path

    def detect_faces(self, image_path: str):
        """Atalho para detecção facial."""
        return self.detect_objects(image_path, task="faces")

    def run_zenml_pipeline(self, data_path: str):
        """Executa um pipeline ZenML simplificado."""
        if not HAS_ZENML:
            logger.warning("⚠️ ZenML não disponível.")
            return "ZenML não instalado"

        logger.info("🚀 Executando Pipeline ZenML...")
        
        @step
        def load_data_step() -> pd.DataFrame:
            return pd.read_csv(data_path)

        @step
        def process_data_step(df: pd.DataFrame) -> pd.DataFrame:
            return df.fillna(0)

        @pipeline
        def simple_ml_pipeline():
            df = load_data_step()
            process_data_step(df)

        run = simple_ml_pipeline()
        return "Pipeline ZenML Concluído com Sucesso"

    def get_image_embedding(self, image_path: str):
        """Gera embedding de imagem usando ResNet50."""
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1]) # Remover camada de classificação
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            embedding = model(img_t).flatten().numpy()
        return embedding

    def recommend_similar(self, query_img_path: str, gallery_dir: str, top_k=3):
        """Recomenda imagens semelhantes de um diretório."""
        logger.info(f"🔎 Buscando imagens semelhantes a {query_img_path} em {gallery_dir}")
        query_emb = self.get_image_embedding(query_img_path)
        
        scores = []
        valid_extensions = ('.jpg', '.jpeg', '.png')
        
        for img_name in os.listdir(gallery_dir):
            if img_name.lower().endswith(valid_extensions):
                path = os.path.join(gallery_dir, img_name)
                emb = self.get_image_embedding(path)
                sim = 1 - cosine(query_emb, emb)
                scores.append((path, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # --- MÓDULO UNIVERSAL: NLP ---
    def nlp_analyze(self, text: str, task='sentiment-analysis'):
        """Executa tarefas de NLP usando HuggingFace Pipelines."""
        logger.info(f"📝 Executando NLP ({task})")
        nlp_pipe = pipeline(task)
        result = nlp_pipe(text)
        return result

    # --- MÓDULO UNIVERSAL: TIME SERIES ---
    def train_timeseries(self, data_path: str, date_col: str, target_col: str, periods=30):
        """Treina modelo de Séries Temporais (Prophet) e logga no MLflow."""
        if not HAS_PROPHET:
            raise ImportError("Prophet não instalado.")
        
        logger.info(f"📈 Treinando Time Series (Prophet) no target: {target_col}")
        df = pd.read_csv(data_path)
        df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
        df['ds'] = pd.to_datetime(df['ds'])
        
        mlflow.set_experiment("/time_series_prophet")
        with mlflow.start_run(run_name=f"prophet_run_{datetime.now().strftime('%H%M%S')}"):
            self._log_env_artifacts()
            model = Prophet()
            model.fit(df)
            
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            fig = model.plot(forecast)
            plot_path = "forecast_plot.png"
            fig.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            
            # Prophet não tem log_model nativo no mlflow padrão, loggamos como artifact ou pickle
            import pickle
            with open("prophet_model.pkl", "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact("prophet_model.pkl")
            
            return forecast, plot_path

    # --- MÓDULO UNIVERSAL: CLUSTERING & ANOMALY ---
    def train_clustering(self, data_path: str, n_clusters=3, method='kmeans'):
        """Executa clustering não supervisionado."""
        logger.info(f"💎 Executando Clustering ({method})")
        df = pd.read_csv(data_path).select_dtypes(include=[np.number])
        
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            model = DBSCAN(eps=0.5, min_samples=5)
            
        clusters = model.fit_predict(df)
        df['cluster'] = clusters
        
        res_path = "clustering_results.csv"
        df.to_csv(res_path, index=False)
        mlflow.log_artifact(res_path)
        return res_path

    def detect_anomalies(self, data_path: str):
        """Detecta anomalias usando Isolation Forest."""
        logger.info("🚨 Detectando anomalias...")
        df = pd.read_csv(data_path).select_dtypes(include=[np.number])
        
        model = IsolationForest(contamination=0.1, random_state=42)
        anomalies = model.fit_predict(df)
        df['is_anomaly'] = anomalies
        
        res_path = "anomalies_detected.csv"
        df.to_csv(res_path, index=False)
        mlflow.log_artifact(res_path)
        return res_path

    # --- MÓDULO UNIVERSAL: FINE-TUNING ---
    def fine_tune_nlp(self, model_name: str, train_csv: str, text_col: str, label_col: str):
        """Fine-tuning básico de modelos Transformers."""
        logger.info(f"🧠 Iniciando Fine-tuning de {model_name}")
        df = pd.read_csv(train_csv)
        
        # Preparar Dataset
        dataset = Dataset.from_pandas(df)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def tokenize_function(examples):
            return tokenizer(examples[text_col], padding="max_length", truncation=True)
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=df[label_col].nunique())
        
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=1,
            weight_decay=0.01,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
        )
        
        trainer.train()
        model.save_pretrained("fine_tuned_model")
        tokenizer.save_pretrained("fine_tuned_model")
        mlflow.log_artifacts("fine_tuned_model")
        return "fine_tuned_model"

def main():
    m = MLOpsEnterprise()
    logger.info("🚀 Framework Universal V4.0 Carregado.")

if __name__ == "__main__":
    main()
