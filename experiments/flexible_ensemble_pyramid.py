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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict



import random

warnings.filterwarnings("ignore")

try:
    import mlflow  # type: ignore
except Exception:
    class _DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    class _DummyMlflow:
        def set_tracking_uri(self, *_args, **_kwargs):
            return None

        def set_experiment(self, *_args, **_kwargs):
            return None

        def start_run(self, *_args, **_kwargs):
            return _DummyRun()

        def log_params(self, *_args, **_kwargs):
            return None

        def log_param(self, *_args, **_kwargs):
            return None

        def log_artifact(self, *_args, **_kwargs):
            return None

        def log_metric(self, *_args, **_kwargs):
            return None

    mlflow = _DummyMlflow()

try:
    import dagshub  # type: ignore
except Exception:
    dagshub = None

def set_seed(seed):
     """Ensures 100% reproducibility across runs."""
     random.seed(seed)
     np.random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed)
     # Note: Sklearn models use np.random but some underlying libs might need more



# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()

# Parameters (Default values, can be overridden)
SEED = 2007
NUM_LAYERS = 12  
CV_FOLDS = 3
EXP_NAME = "Ensemble_Pyramid_Flexible"
PATIENCE = 3  # Layers without improvement before stopping


# Dynamic Variation Parameters
MIN_MODELS_PER_LAYER = 2
MAX_MODELS_PER_LAYER = 6 # Max models to pick per layer
EPSILON_RL = 0.2         # RL exploration rate
OPTIM_METRIC = "f1"      # Metric to optimize: f1, accuracy
TFIDF_MAX = 50000
TFIDF_NGRAMS = (1, 2)
JITTER = True            # Random hyperparameter variation
STRATEGY = "dense"       # "dense", "residual", "simple"



# ─── Setup Setup Tracking ─────────────────────────────────────────────────────
def setup_tracking():
    use_dagshub = (os.getenv("DAGSHUB_TOKEN") is not None or os.getenv("USE_DAGSHUB") == "True") and dagshub is not None
    
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
def get_model(model_type, seed=SEED, jitter=False):
    # Bagging Wrapper
    is_bagging = model_type.startswith("bag_")
    base_type = model_type.replace("bag_", "") if is_bagging else model_type

    # Hyperparameter Jitter (Variability)
    j_c = np.random.uniform(0.5, 20.0) if jitter else 1.0
    j_tree = np.random.randint(50, 200) if jitter else 100
    j_alpha = np.random.uniform(0.01, 1.0) if jitter else 1.0

    if base_type == "lr":
        m = LogisticRegression(C=11.0 * j_c if jitter else 11.0, max_iter=1000, random_state=seed, n_jobs=2)
    elif base_type == "svc":
        m = CalibratedClassifierCV(LinearSVC(C=19.0 * j_c if jitter else 19.0, random_state=seed), cv=2)
    elif base_type == "nb":
        m = MultinomialNB(alpha=0.1 * j_alpha if jitter else 0.1)
    elif base_type == "rf":
        m = RandomForestClassifier(n_estimators=j_tree, random_state=seed, n_jobs=2)
    elif base_type == "et":
        m = ExtraTreesClassifier(n_estimators=j_tree, random_state=seed, n_jobs=2)
    elif base_type == "ridge":
        m = CalibratedClassifierCV(RidgeClassifier(alpha=1.0 * j_alpha if jitter else 1.0), cv=2)
    else:
        m = LogisticRegression(random_state=seed)
    
    if is_bagging:
        return BaggingClassifier(m, n_estimators=10, random_state=seed, n_jobs=2)
    return m



# ─── RL Meta-Learner ─────────────────────────────────────────────────────────
class RLMetaLearner:
    """
    Reinforcement Learning agent to optimize model selection across runs.
    Uses an epsilon-greedy approach with persistent memory.
    """
    def __init__(self, knowledge_path="pyramid_rl_knowledge.json", epsilon=0.2, metric="f1"):
        self.path = Path(knowledge_path)
        self.epsilon = epsilon
        self.metric = metric
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

    def suggest_models(self, layer_idx, available_models, min_p=MIN_MODELS_PER_LAYER, max_p=MAX_MODELS_PER_LAYER):
        """Suggests a variable number of models using an epsilon-greedy strategy."""
        l_str = str(layer_idx)
        
        # Determine how many models to pick (this can vary run by run)
        n_to_pick = np.random.randint(min_p, min(max_p, len(available_models)) + 1)
        
        # Explore: Epsilon-greedy random selection
        if np.random.random() < self.epsilon or l_str not in self.knowledge["layer_stats"]:
            choices = np.random.choice(available_models, n_to_pick, replace=False)
            return [str(m) for m in choices] 
        
        stats = self.knowledge["layer_stats"][l_str]
        # Exploit: Pick top-N performers with a bit of noise (Thompson Sampling-lite)
        m_metric = f"avg_{self.metric}"
        sorted_models = sorted(
            [m for m in available_models if m in stats],
            key=lambda x: (stats[x].get(m_metric, 0) + np.random.normal(0, 0.01)), 
            reverse=True
        )

        
        # Take the top N, but ensure we return exactly n_to_pick if possible
        final_list = [str(m) for m in sorted_models[:n_to_pick]]
        
        # Fallback if too few models were known
        if len(final_list) < n_to_pick:
            remaining = [m for m in available_models if m not in final_list]
            extra = np.random.choice(remaining, n_to_pick - len(final_list), replace=False)
            final_list.extend([str(m) for m in extra])
            
        return final_list



    def update_knowledge(self, results):
        self.knowledge["runs"] += 1
        for res in results:
            l_str = str(res["layer"])
            m_base = res["model"].split('_')[0]
            val_f1 = res["f1"]
            val_acc = res["accuracy"]
            
            if l_str not in self.knowledge["layer_stats"]:
                self.knowledge["layer_stats"][l_str] = {}
            if m_base not in self.knowledge["layer_stats"][l_str]:
                self.knowledge["layer_stats"][l_str][m_base] = {"count": 0, "avg_f1": 0.0, "avg_accuracy": 0.0}
            
            s = self.knowledge["layer_stats"][l_str][m_base]
            s["count"] += 1
            # Update both metrics
            s["avg_f1"] = s["avg_f1"] + (val_f1 - s["avg_f1"]) / s["count"]
            s["avg_accuracy"] = s["avg_accuracy"] + (val_acc - s["avg_accuracy"]) / s["count"]

        self._save_knowledge()
        print(f"[RL] Meta-Knowledge updated. Total runs: {self.knowledge['runs']}", flush=True)


# ─── Pyramid Logic ────────────────────────────────────────────────────────────
class PyramidEnsemble:
    def __init__(self, num_layers=3, seed=SEED, meta_learner=None, patience=PATIENCE, 
                 metric=OPTIM_METRIC, jitter=JITTER, strategy=STRATEGY,
                 min_models=MIN_MODELS_PER_LAYER, max_models=MAX_MODELS_PER_LAYER):
        self.num_layers = num_layers
        self.seed = seed
        self.meta_learner = meta_learner
        self.patience = patience
        self.metric = metric
        self.jitter = jitter
        self.strategy = strategy
        self.min_models = min_models
        self.max_models = max_models
        self.layers = []
        self.layer_meta_models = []
        self.results = []
        self.best_model = None
        self.best_score = 0
        self.no_improve_layers = 0 # Early stopping counter


        
    def _evaluate_and_log(self, model, X, y, layer_idx, name, start_time):
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="weighted")
        duration = time.time() - start_time
        
        score = f1 if self.metric == "f1" else acc
        self.results.append({"layer": layer_idx, "model": name, "accuracy": acc, "f1": f1, "duration": duration})
        status = "NEW BEST!" if score > self.best_score else "       "
        print(f"  [{status}] Layer {layer_idx:02} | {name:<20} | F1: {f1:.4f} | Acc: {acc:.4f} | {duration:>5.1f}s", flush=True)
        
        if score > self.best_score:
            self.best_score = score
            self.best_model = model
        return score


    def train(self, X_train, y_train, X_val, y_val):
        current_X_train, current_X_val = X_train, X_val
        all_oof_preds, all_val_preds = [], []
        self.layer_meta_models = []
        t_global_start = time.time()
        
        print(f"\n{'='*85}\n{'STARTING ENSEMBLE PYRAMID TRAINING':^85}\n{'='*85}", flush=True)
        print(f"Dataset: {X_train.shape} | Layers: {self.num_layers} | RL: {'Active' if self.meta_learner else 'Off'}", flush=True)
        print(f"{'-'*85}", flush=True)

        for l in range(1, self.num_layers + 1):
            t_layer_start = time.time()
            print(f"\n[LAYER {l:02}] Training...", flush=True)
            
            # Base pool now includes Bagging variants
            avail_b = ["lr", "svc", "nb", "ridge", "rf", "et", "bag_lr", "bag_svc", "bag_nb"]
            avail_m = ["lr", "ridge", "rf", "bag_lr"]
            
            # The heart of variability: RL Meta-Learner decides WHAT and HOW MANY

            models_to_run = self.meta_learner.suggest_models(
                l, 
                avail_b if l==1 else avail_m,
                min_p=self.min_models,
                max_p=self.max_models,
            ) if self.meta_learner else (avail_b if l==1 else avail_m)
            
            print(f"  [RL] Selected set ({len(models_to_run)} models): {models_to_run}", flush=True)


            layer_models, layer_oof, layer_val = [], [], []
            layer_transformers = []
            score_before_layer = self.best_score  # snapshot before any model in this layer updates it
            for m_type in models_to_run:
                t0 = time.time()
                model_name = f"{m_type}_L{l}"
                model = get_model(m_type, self.seed + l, jitter=self.jitter)
                model.fit(current_X_train, y_train)
                self._evaluate_and_log(model, current_X_val, y_val, l, model_name, t0)

                layer_models.append((model_name, model))

                if l < self.num_layers:
                    layer_transformers.append(model)
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

            # --- Automatic Layer Voting (Meta-Ensemble of the Layer) ---
            # Uses pre-fitted models directly to avoid retraining them inside VotingClassifier
            if len(layer_models) > 1:
                t0_v = time.time()
                v_name = f"voting_L{l}"
                all_have_proba = all(hasattr(m[1], "predict_proba") for m in layer_models)

                if all_have_proba:
                    # Soft voting: average predict_proba across pre-fitted models
                    class _PreFittedSoftVoting:
                        def __init__(self, estimators):
                            self.estimators_ = [m for _, m in estimators]
                        def predict_proba(self, X):
                            return np.mean([m.predict_proba(X) for m in self.estimators_], axis=0)
                        def predict(self, X):
                            return np.argmax(self.predict_proba(X), axis=1)

                    v_model = _PreFittedSoftVoting(layer_models)
                else:
                    # Hard voting: majority vote across pre-fitted models
                    class _PreFittedHardVoting:
                        def __init__(self, estimators, n_classes):
                            self.estimators_ = [m for _, m in estimators]
                            self.n_classes_ = n_classes
                        def predict(self, X):
                            preds = np.column_stack([m.predict(X) for m in self.estimators_])
                            return np.apply_along_axis(
                                lambda row: np.bincount(row, minlength=self.n_classes_).argmax(),
                                axis=1, arr=preds
                            )
                        def predict_proba(self, X):
                            preds = self.predict(X)
                            out = np.zeros((len(preds), self.n_classes_))
                            out[np.arange(len(preds)), preds] = 1.0
                            return out

                    v_model = _PreFittedHardVoting(layer_models, len(np.unique(y_train)))

                self._evaluate_and_log(v_model, current_X_val, y_val, l, v_name, t0_v)
                layer_models.append((v_name, v_model))

            self.layers.append(layer_models)

            
            # Early Stopping Check — compare against score snapshot taken before this layer
            layer_best_score = max([(res["f1"] if self.metric == "f1" else res["accuracy"]) for res in self.results if res["layer"] == l])
            if layer_best_score > score_before_layer + 1e-5:
                self.no_improve_layers = 0
            else:
                self.no_improve_layers += 1
                if self.no_improve_layers >= self.patience:
                    print(f"\n[EARLY STOPPING] No improvement for {self.patience} layers. Stopping at Layer {l}.", flush=True)
                    mlflow.log_param("early_stop_layer", l)
                    break

            if l < self.num_layers:
                self.layer_meta_models.append(layer_transformers)
                all_oof_preds.append(np.hstack(layer_oof))
                all_val_preds.append(np.hstack(layer_val))
                
                # Dynamic Strategy
                if self.strategy == "dense":
                    current_X_train, current_X_val = np.hstack(all_oof_preds), np.hstack(all_val_preds)
                elif self.strategy == "residual":
                    # Only last layer's OOF (no accumulation)
                    current_X_train, current_X_val = np.hstack(layer_oof), np.hstack(layer_val)
                else: # "simple"
                    current_X_train, current_X_val = np.hstack(layer_oof), np.hstack(layer_val)
                    
                print(f"  [INFO] Layer {l} Done ({time.time()-t_layer_start:.1f}s). New Feats: {current_X_train.shape[1] if hasattr(current_X_train, 'shape') else len(current_X_train)}", flush=True)


        print(f"\n{'='*85}\nCOMPLETE | Time: {(time.time()-t_global_start)/60:.1f} min\n{'='*85}", flush=True)
        if self.meta_learner: self.meta_learner.update_knowledge(self.results)

    def predict(self, X):
        """Transform X through the layers to match the best model's required input."""
        if self.best_model is None:
            raise ValueError("Pyramid not trained or no best model found.")

        def _model_output_as_proba(model, features):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(features)
            raw = np.asarray(model.predict(features), dtype=int)
            n_classes = int(raw.max()) + 1 if raw.size > 0 else 1
            return np.eye(n_classes)[raw]
        
        # Determine the layer of the best model
        best_layer_idx = 1
        found = False
        for l_idx, l_models in enumerate(self.layers):
            for m_name, m_obj in l_models:
                if m_obj == self.best_model:
                    best_layer_idx = l_idx + 1
                    found = True
                    break
            if found:
                break
        
        if best_layer_idx == 1:
            return self.best_model.predict(X)
        
        # Build meta-features chain using only models that generated next-layer features during training.
        current_X = X
        all_meta_preds = []
        
        for l in range(1, best_layer_idx):
            if l - 1 >= len(self.layer_meta_models):
                raise ValueError("Missing layer meta-models for inference. Retrain the pyramid before predicting.")

            layer_meta = [_model_output_as_proba(m_obj, current_X) for m_obj in self.layer_meta_models[l - 1]]
            layer_stack = np.hstack(layer_meta)

            if self.strategy == "dense":
                all_meta_preds.append(layer_stack)
                current_X = np.hstack(all_meta_preds)
            elif self.strategy == "residual":
                current_X = layer_stack
            else:  # "simple"
                current_X = layer_stack
            
        return self.best_model.predict(current_X)



# ─── Execution ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Highly Customizable Flexible Ensemble Pyramid with RL")
    parser.add_argument("--layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--min_models", type=int, default=MIN_MODELS_PER_LAYER)
    parser.add_argument("--max_models", type=int, default=MAX_MODELS_PER_LAYER)
    parser.add_argument("--epsilon", type=float, default=EPSILON_RL)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--metric", type=str, choices=["f1", "accuracy"], default=OPTIM_METRIC)
    parser.add_argument("--tfidf_max", type=int, default=TFIDF_MAX)
    parser.add_argument("--tfidf_ngrams", type=int, default=2, help="Max ngrams (1 or 2)")
    parser.add_argument("--jitter", type=bool, default=JITTER)
    parser.add_argument("--strategy", type=str, choices=["dense", "residual", "simple"], default=STRATEGY)
    parser.add_argument("--max_train_rows", type=int, default=0, help="0 keeps full dataset; >0 samples train rows")
    parser.add_argument("--max_val_rows", type=int, default=0, help="0 keeps full dataset; >0 samples validation rows")
    args = parser.parse_args()


    set_seed(args.seed)
    setup_tracking()

    
    current_layers = args.layers
    current_seed = args.seed

    print(f"\n[CONFIG] Layers: {current_layers} | Seed: {current_seed} | RL Epsilon: {args.epsilon}", flush=True)
    print(f"[CONFIG] Diversity: {args.min_models}-{args.max_models} models per layer\n", flush=True)

    with mlflow.start_run(run_name=f"Pyramid_RL_L{current_layers}_Var"):
        mlflow.log_params({
            "num_layers": current_layers, 
            "seed": current_seed,
            "epsilon": args.epsilon,
            "min_models": args.min_models,
            "max_models": args.max_models,
            "metric": args.metric,
            "jitter": args.jitter,
            "strategy": args.strategy,
            "tfidf_max": args.tfidf_max
        })


        
        train_df, val_df = load_data()
        if args.max_train_rows > 0 and len(train_df) > args.max_train_rows:
            train_df = train_df.sample(n=args.max_train_rows, random_state=current_seed).reset_index(drop=True)
            print(f"[SMOKE] Train sample enabled: {len(train_df)} rows", flush=True)
        if args.max_val_rows > 0 and len(val_df) > args.max_val_rows:
            val_df = val_df.sample(n=args.max_val_rows, random_state=current_seed).reset_index(drop=True)
            print(f"[SMOKE] Validation sample enabled: {len(val_df)} rows", flush=True)

        le = LabelEncoder()
        y_train = le.fit_transform(train_df['sentiment'])
        y_val = le.transform(val_df['sentiment'])
        
        tfidf = TfidfVectorizer(max_features=args.tfidf_max, ngram_range=(1, args.tfidf_ngrams))
        X_train = tfidf.fit_transform(train_df['clean_text'].fillna(""))
        X_val = tfidf.transform(val_df['clean_text'].fillna(""))
        
        rl_learner = RLMetaLearner(epsilon=args.epsilon, metric=args.metric)
        pyramid = PyramidEnsemble(num_layers=current_layers, seed=current_seed, 
                                 meta_learner=rl_learner, patience=args.patience,
                                 metric=args.metric, jitter=args.jitter,
                                 strategy=args.strategy,
                                 min_models=args.min_models, max_models=args.max_models)
        pyramid.train(X_train, y_train, X_val, y_val)



        
        # Artifacts
        pd.DataFrame(pyramid.results).to_csv("pyramid_results.csv", index=False)
        mlflow.log_artifact("pyramid_results.csv")
        if rl_learner.path.exists(): mlflow.log_artifact(str(rl_learner.path))
        
        if pyramid.best_model:
            joblib.dump(pyramid.best_model, "best_pyramid_model.pkl")
            mlflow.log_artifact("best_pyramid_model.pkl")

            # Log best model metrics so runs are comparable in MLflow UI
            best_res = max(pyramid.results, key=lambda r: r["f1"] if pyramid.metric == "f1" else r["accuracy"])
            mlflow.log_params({"best_model": best_res["model"], "best_layer": best_res["layer"]})
            try:
                mlflow.log_metric("best_f1", best_res["f1"])
                mlflow.log_metric("best_accuracy", best_res["accuracy"])
            except Exception:
                pass  # Dummy mlflow silently ignores metrics

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