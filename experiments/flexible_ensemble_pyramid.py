import os
import re
import time
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv

# MLOps
import mlflow
import dagshub
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict


warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

SEED = 2007
NUM_LAYERS = 12
CV_FOLDS = 3
EXP_NAME = "Ensemble_Pyramid_Flexible"

# ─── Setup Setup Tracking ─────────────────────────────────────────────────────
def setup_tracking():
    use_dagshub = os.getenv("DAGSHUB_TOKEN") is not None or os.getenv("USE_DAGSHUB") == "True"
    
    if use_dagshub:
        try:
            print("[INFO] Initializing Dagshub...", flush=True)
            dagshub.init(repo_owner="PedroM2626", repo_name="experiments")
            mlflow.set_tracking_uri("https://dagshub.com/PedroM2626/experiments.mlflow")
            print("[INFO] Dagshub initialized.", flush=True)
        except Exception as e:
            print(f"[ERROR] Could not initialize Dagshub: {e}", flush=True)
            mlflow.set_tracking_uri("file:./mlruns")
    else:
        print("[INFO] Using local file-based MLflow tracking (file:./mlruns)", flush=True)
        mlflow.set_tracking_uri("file:./mlruns")
    
    try:
        print(f"[INFO] Setting experiment: {EXP_NAME}", flush=True)
        mlflow.set_experiment(EXP_NAME)
        print(f"[INFO] Experiment set to: {EXP_NAME}", flush=True)
    except Exception as e:
        print(f"[ERROR] Could not set experiment: {e}", flush=True)

# ─── Data Preparation ──────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z0-9\s!?.,\'\-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data():
    base_path = Path("experiments/senti-pred-variations/logistic-senti-pred/data/raw")
    cols = ['tweet_id', 'entity', 'sentiment', 'text']
    train_file = base_path / "twitter_training.csv"
    val_file = base_path / "twitter_validation.csv"
    
    if not train_file.exists():
        base_path = Path(__file__).parent / "senti-pred-variations/logistic-senti-pred/data/raw"
        train_file = base_path / "twitter_training.csv"
        val_file = base_path / "twitter_validation.csv"

    train_df = pd.read_csv(train_file, names=cols, header=None)
    val_df = pd.read_csv(val_file, names=cols, header=None)
    valid_sentiments = ['Positive', 'Negative', 'Neutral', 'Irrelevant']
    
    for df in (train_df, val_df):
        df['clean_text'] = df['text'].apply(clean_tweet)
        df = df[(df['clean_text'].str.len() > 0) & (df['sentiment'].isin(valid_sentiments))]
    return train_df, val_df

# ─── Model Factories ─────────────────────────────────────────────────────────
def get_model(model_type, seed=SEED):
    if model_type == "lr":
        return LogisticRegression(C=11.0, max_iter=1000, random_state=seed, n_jobs=2)
    elif model_type == "svc":
        return CalibratedClassifierCV(LinearSVC(C=19.0, random_state=seed), cv=2)
    elif model_type == "nb":
        return MultinomialNB(alpha=0.1)
    elif model_type == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=2)
    elif model_type == "et":
        return ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=2)
    elif model_type == "ridge":
        return CalibratedClassifierCV(RidgeClassifier(alpha=1.0), cv=2)
    return LogisticRegression(random_state=seed)

# ─── RL Meta-Learner ─────────────────────────────────────────────────────────
class RLMetaLearner:
    """
    Reinforcement Learning agent to optimize model selection across runs.
    Uses an epsilon-greedy approach with persistent memory.
    """
    def __init__(self, knowledge_path="pyramid_rl_knowledge.json", epsilon=0.2):
        self.path = Path(knowledge_path)
        self.epsilon = epsilon
        self.knowledge = self._load_knowledge()

    def _load_knowledge(self):
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"runs": 0, "layer_stats": {}}

    def _save_knowledge(self):
        with open(self.path, 'w') as f:
            json.dump(self.knowledge, f, indent=4)

    def suggest_models(self, layer_idx, available_models):
        l_str = str(layer_idx)
        # Explore: Epsilon-greedy random selection
        if np.random.random() < self.epsilon or l_str not in self.knowledge["layer_stats"]:
            n_to_pick = np.random.randint(min(2, len(available_models)), len(available_models) + 1)
            choices = np.random.choice(available_models, n_to_pick, replace=False)
            return [str(m) for m in choices] # Cast to python string
        
        stats = self.knowledge["layer_stats"][l_str]
        # Avoid picking the same models every single time if scores are identical
        sorted_models = sorted(
            [m for m in available_models if m in stats],
            key=lambda x: (stats[x]["avg_f1"], np.random.random()), 
            reverse=True
        )
        final_list = [str(m) for m in sorted_models[:max(2, len(sorted_models)//2 + 1)]]
        return final_list if len(final_list) >= 2 else [str(m) for m in available_models[:2]]


    def update_knowledge(self, results):
        self.knowledge["runs"] += 1
        for res in results:
            l_str = str(res["layer"])
            m_base = res["model"].split('_')[0]
            f1 = res["f1"]
            
            if l_str not in self.knowledge["layer_stats"]:
                self.knowledge["layer_stats"][l_str] = {}
            if m_base not in self.knowledge["layer_stats"][l_str]:
                self.knowledge["layer_stats"][l_str][m_base] = {"count": 0, "avg_f1": 0.0}
            
            s = self.knowledge["layer_stats"][l_str][m_base]
            s["count"] += 1
            s["avg_f1"] = s["avg_f1"] + (f1 - s["avg_f1"]) / s["count"]
        self._save_knowledge()
        print(f"[RL] Meta-Knowledge updated. Total runs: {self.knowledge['runs']}", flush=True)

# ─── Pyramid Logic ────────────────────────────────────────────────────────────
class PyramidEnsemble:
    def __init__(self, num_layers=3, seed=SEED, meta_learner=None):
        self.num_layers = num_layers
        self.seed = seed
        self.meta_learner = meta_learner
        self.layers = []
        self.results = []
        self.best_model = None
        self.best_f1 = 0
        
    def _evaluate_and_log(self, model, X, y, layer_idx, name, start_time):
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted")
        duration = time.time() - start_time
        
        self.results.append({"layer": layer_idx, "model": name, "accuracy": acc, "f1": f1, "duration": duration})
        status = "NEW BEST!" if f1 > self.best_f1 else "       "
        print(f"  [{status}] Layer {layer_idx:02} | {name:<20} | F1: {f1:.4f} | Acc: {acc:.4f} | {duration:>5.1f}s", flush=True)
        
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_model = model
        return f1

    def train(self, X_train, y_train, X_val, y_val):
        current_X_train, current_X_val = X_train, X_val
        all_oof_preds, all_val_preds = [], []
        t_global_start = time.time()
        
        print(f"\n{'='*85}\n{'STARTING ENSEMBLE PYRAMID TRAINING':^85}\n{'='*85}", flush=True)
        print(f"Dataset: {X_train.shape} | Layers: {self.num_layers} | RL: {'Active' if self.meta_learner else 'Off'}", flush=True)
        print(f"{'-'*85}", flush=True)

        for l in range(1, self.num_layers + 1):
            t_layer_start = time.time()
            print(f"\n[LAYER {l:02}] Training...", flush=True)
            
            avail_b = ["lr", "svc", "nb", "ridge", "rf", "et"]
            avail_m = ["lr", "ridge", "rf"]
            models_to_run = self.meta_learner.suggest_models(l, avail_b if l==1 else avail_m) if self.meta_learner else (avail_b if l==1 else avail_m)
            print(f"  [RL] Models: {models_to_run}", flush=True)

            layer_models, layer_oof, layer_val = [], [], []
            for m_type in models_to_run:
                t0 = time.time()
                model_name = f"{m_type}_L{l}"
                model = get_model(m_type, self.seed + l)
                model.fit(current_X_train, y_train)
                self._evaluate_and_log(model, current_X_val, y_val, l, model_name, t0)
                layer_models.append((model_name, model))

                
                if l < self.num_layers:
                    if hasattr(model, "predict_proba"):
                        oof = cross_val_predict(model, current_X_train, y_train, cv=CV_FOLDS, method="predict_proba", n_jobs=2)
                        vp = model.predict_proba(current_X_val)
                    else:
                        oof_idx = cross_val_predict(model, current_X_train, y_train, cv=CV_FOLDS, n_jobs=2)
                        vp_idx = model.predict(current_X_val)
                        n_c = len(np.unique(y_train))
                        oof = np.eye(n_c)[oof_idx]
                        vp = np.eye(n_c)[vp_idx]
                    layer_oof.append(oof)
                    layer_val.append(vp)

            if l < self.num_layers:
                all_oof_preds.append(np.hstack(layer_oof))
                all_val_preds.append(np.hstack(layer_val))
                current_X_train, current_X_val = np.hstack(all_oof_preds), np.hstack(all_val_preds)
                print(f"  [INFO] Layer {l} Done ({time.time()-t_layer_start:.1f}s). New Feats: {current_X_train.shape[1]}", flush=True)

        print(f"\n{'='*85}\nCOMPLETE | Time: {(time.time()-t_global_start)/60:.1f} min\n{'='*85}", flush=True)
        if self.meta_learner: self.meta_learner.update_knowledge(self.results)

    def predict(self, X):
        """Transform X through the layers to match the best model's required input."""
        if self.best_model is None:
            raise ValueError("Pyramid not trained or no best model found.")
        
        # Determine the layer of the best model
        best_layer_idx = 1
        for l_idx, l_models in enumerate(self.layers):
            for m_name, m_obj in l_models:
                if m_obj == self.best_model:
                    best_layer_idx = l_idx + 1
                    break
        
        if best_layer_idx == 1:
            return self.best_model.predict(X)
        
        # Build meta-features chain
        current_X = X
        all_meta_preds = []
        
        for l in range(1, best_layer_idx):
            layer_meta = []
            for m_name, m_obj in self.layers[l-1]:
                if hasattr(m_obj, "predict_proba"):
                    p = m_obj.predict_proba(current_X)
                else:
                    raw = m_obj.predict(current_X)
                    n_c = len(self.best_model.classes_)
                    p = np.eye(n_c)[raw]
                layer_meta.append(p)
            
            all_meta_preds.append(np.hstack(layer_meta))
            current_X = np.hstack(all_meta_preds)
            
        return self.best_model.predict(current_X)



# ─── Execution ───────────────────────────────────────────────────────────────
def main():
    setup_tracking()
    with mlflow.start_run(run_name=f"Pyramid_RL_L{NUM_LAYERS}"):
        mlflow.log_params({"num_layers": NUM_LAYERS, "seed": SEED})
        
        train_df, val_df = load_data()
        le = LabelEncoder()
        y_train = le.fit_transform(train_df['sentiment'])
        y_val = le.transform(val_df['sentiment'])
        
        tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        X_train = tfidf.fit_transform(train_df['clean_text'].fillna(""))
        X_val = tfidf.transform(val_df['clean_text'].fillna(""))
        
        rl_learner = RLMetaLearner()
        pyramid = PyramidEnsemble(num_layers=NUM_LAYERS, seed=SEED, meta_learner=rl_learner)
        pyramid.train(X_train, y_train, X_val, y_val)
        
        # Artifacts
        pd.DataFrame(pyramid.results).to_csv("pyramid_results.csv", index=False)
        mlflow.log_artifact("pyramid_results.csv")
        if rl_learner.path.exists(): mlflow.log_artifact(str(rl_learner.path))
        
        if pyramid.best_model:
            joblib.dump(pyramid.best_model, "best_pyramid_model.pkl")
            mlflow.log_artifact("best_pyramid_model.pkl")
            
            # Use the pyramid's predict method which handles feature transformation
            y_pred = pyramid.predict(X_val)
            cm = confusion_matrix(y_val, y_pred)
            plt.figure(figsize=(10,7))

            sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
            plt.savefig("pyramid_cm.png")
            mlflow.log_artifact("pyramid_cm.png")
            
        print("\nRun Completed!")

if __name__ == "__main__":
    main()
