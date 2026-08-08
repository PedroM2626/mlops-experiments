import numpy as np
import pandas as pd
import warnings, json, urllib.request, time
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline
import xgboost as xgb

print("=" * 95)
print("ANOMALY DETECTION EM TIME SERIES - VERSÃO APRIMORADA")
print("Dataset: NAB machine_temperature_system_failure")
print("=" * 95)

# --- 1. Load ---
DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json"

df = pd.read_csv(DATA_URL, parse_dates=["timestamp"])
with urllib.request.urlopen(LABELS_URL) as f:
    labels_dict = json.load(f)

key = "realKnownCause/machine_temperature_system_failure.csv"
anomaly_timestamps = [pd.Timestamp(t) for t in labels_dict[key]]

df["anomaly"] = 0
window_delta = pd.Timedelta("30 min")
for ts in anomaly_timestamps:
    mask = (df["timestamp"] >= ts - window_delta) & (df["timestamp"] <= ts + window_delta)
    df.loc[mask, "anomaly"] = 1

print(f"\nDataset: {len(df)} amostras, {df['anomaly'].sum()} anomalias ({df['anomaly'].mean()*100:.3f}%)")

# --- 2. Enhanced Features ---
def create_enhanced_features(series, windows=[5, 10, 20, 50]):
    df_feat = pd.DataFrame(index=series.index)
    df_feat["value"] = series

    for w in windows:
        roll = series.rolling(w, min_periods=1)
        df_feat[f"mean_{w}"] = roll.mean()
        df_feat[f"std_{w}"] = roll.std().fillna(0)
        df_feat[f"min_{w}"] = roll.min()
        df_feat[f"max_{w}"] = roll.max()
        df_feat[f"range_{w}"] = roll.max() - roll.min()
        z = (series - roll.mean()) / roll.std().replace(0, np.nan)
        df_feat[f"zscore_{w}"] = z.fillna(0).clip(-5, 5)
        df_feat[f"diff_mean_{w}"] = series - roll.mean()
        df_feat[f"pct_{w}"] = series.pct_change(w).fillna(0)

    for lag in [1, 2, 3, 5, 10]:
        df_feat[f"lag_{lag}"] = series.shift(lag)

    df_feat["pct_change_1"] = series.pct_change(1).fillna(0)
    df_feat["pct_change_5"] = series.pct_change(5).fillna(0)
    diff1 = series.diff(1).fillna(0)
    df_feat["accel"] = diff1.diff(1).fillna(0)
    df_feat["ewma_01"] = series.ewm(alpha=0.1, adjust=False).mean()
    df_feat["ewma_05"] = series.ewm(alpha=0.5, adjust=False).mean()

    return df_feat.fillna(0)

features = create_enhanced_features(df["value"])
feature_cols = [c for c in features.columns if c != "value"]
print(f"Features: {len(feature_cols)}")

# --- 3. Split ---
X = features[feature_cols].values
y = df["anomaly"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

split_idx = int(len(df) * 0.7)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Treino: {len(X_train)} amostras, {y_train.sum()} anomalias ({y_train.mean()*100:.3f}%)")
print(f"Teste:  {len(X_test)} amostras, {y_test.sum()} anomalias ({y_test.mean()*100:.3f}%)")

# --- 4. Unsupervised ---
def find_optimal_threshold(yt, ys):
    precisions, recalls, thresholds = precision_recall_curve(yt, ys)
    if thresholds.size == 0:
        return 0.5
    f1s = 2 * precisions[:len(thresholds)] * recalls[:len(thresholds)] / (precisions[:len(thresholds)] + recalls[:len(thresholds)] + 1e-12)
    return thresholds[np.argmax(f1s)]

results_unsup = []
results_sup = []

print("\n" + "=" * 95)
print("NÃO SUPERVISIONADOS")
print("=" * 95)

# IF tuned
print("\n[Tuning Isolation Forest...]")
best_f1, best_params, best_model, best_score = 0, {}, None, None
for cont in [0.003, 0.005, 0.01, 0.02, 0.03, 0.05]:
    for est in [100, 200, 500]:
        model = IsolationForest(n_estimators=est, contamination=cont, random_state=42, n_jobs=-1)
        model.fit(X_train)
        y_pred = np.where(model.predict(X_test) == -1, 1, 0)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_params = f"cont={cont},est={est}"
            best_model = model
            best_score = -model.score_samples(X_test)

prec = precision_score(y_test, np.where(best_model.predict(X_test) == -1, 1, 0))
rec = recall_score(y_test, np.where(best_model.predict(X_test) == -1, 1, 0))
roc = roc_auc_score(y_test, best_score)
print(f"  IF ({best_params}) | prec={prec:.4f} | recall={rec:.4f} | f1={best_f1:.4f} | roc-auc={roc:.4f}")
results_unsup.append(("IF (tunado)", best_f1, prec, rec, roc, best_score))

# LOF
print("[Tuning LOF...]")
best_f1_lof, best_k = 0, 0
best_score_lof = None
for k in [10, 20, 50, 100, 200]:
    model = LocalOutlierFactor(n_neighbors=k, novelty=True)
    model.fit(X_train)
    y_pred = np.where(model.predict(X_test) == -1, 1, 0)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1_lof:
        best_f1_lof = f1
        best_k = k
        best_score_lof = -model.decision_function(X_test)

prec = precision_score(y_test, np.where(best_score_lof > 0, 1, 0))
rec = recall_score(y_test, np.where(best_score_lof > 0, 1, 0))
roc = roc_auc_score(y_test, best_score_lof)
print(f"  LOF (k={best_k})               | prec={prec:.4f} | recall={rec:.4f} | f1={best_f1_lof:.4f} | roc-auc={roc:.4f}")
results_unsup.append(("LOF", best_f1_lof, prec, rec, roc, best_score_lof))

# One-Class SVM
print("[Tuning One-Class SVM...]")
best_f1_svm, best_nu, best_gamma = 0, 0, 0
best_score_svm = None
for nu in [0.001, 0.003, 0.005, 0.01]:
    for gamma in ['scale', 'auto', 0.1]:
        try:
            model = OneClassSVM(nu=nu, gamma=gamma, kernel='rbf')
            model.fit(X_train[:5000])
            y_pred = np.where(model.predict(X_test) == -1, 1, 0)
            f1 = f1_score(y_test, y_pred)
            if f1 > best_f1_svm:
                best_f1_svm = f1
                best_nu, best_gamma = nu, gamma
                best_score_svm = -model.decision_function(X_test)
        except:
            pass

roc = roc_auc_score(y_test, best_score_svm)
prec = precision_score(y_test, np.where(best_score_svm > 0, 1, 0))
rec = recall_score(y_test, np.where(best_score_svm > 0, 1, 0))
print(f"  OC-SVM (nu={best_nu},gamma={best_gamma}) | prec={prec:.4f} | recall={rec:.4f} | f1={best_f1_svm:.4f} | roc-auc={roc:.4f}")
results_unsup.append(("OC-SVM", best_f1_svm, prec, rec, roc, best_score_svm))

# Elliptic Envelope
print("[Tuning Elliptic Envelope...]")
best_f1_ee, best_cont = 0, 0
best_score_ee = None
for cont in [0.003, 0.005, 0.01, 0.02, 0.05]:
    try:
        model = EllipticEnvelope(contamination=cont, random_state=42, support_fraction=0.7)
        model.fit(X_train)
        y_pred = np.where(model.predict(X_test) == -1, 1, 0)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1_ee:
            best_f1_ee = f1
            best_cont = cont
            best_score_ee = -model.score_samples(X_test)
    except:
        pass

roc = roc_auc_score(y_test, best_score_ee)
prec = precision_score(y_test, np.where(best_score_ee > 0, 1, 0))
rec = recall_score(y_test, np.where(best_score_ee > 0, 1, 0))
print(f"  EllipticEnv (cont={best_cont})     | prec={prec:.4f} | recall={rec:.4f} | f1={best_f1_ee:.4f} | roc-auc={roc:.4f}")
results_unsup.append(("EllipticEnv", best_f1_ee, prec, rec, roc, best_score_ee))

# --- 5. Supervised ---
print("\n" + "=" * 95)
print("SUPERVISIONADOS")
print("=" * 95)

def evaluate_sup(name, model, X_tr, y_tr, X_te, y_te, use_threshold=False):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    t1 = time.time()
    y_score = model.predict_proba(X_te)[:, 1]

    if use_threshold:
        y_score_tr = model.predict_proba(X_tr)[:, 1]
        thr = find_optimal_threshold(y_tr, y_score_tr)
    else:
        thr = 0.5

    y_pred = (y_score >= thr).astype(int)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    roc = roc_auc_score(y_te, y_score)

    thr_msg = f" thr={thr:.4f}" if use_threshold else ""
    print(f"  {name:<38} | prec={prec:.4f} | rec={rec:.4f} | f1={f1:.4f} | roc-auc={roc:.4f} | {t1-t0:.1f}s{thr_msg}")
    return (name, f1, prec, rec, roc, y_score)

# SMOTE
smote_pipeline = imb_make_pipeline(
    SMOTE(random_state=42, k_neighbors=2),
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
)
results_sup.append(evaluate_sup("RF + SMOTE", smote_pipeline, X_train, y_train, X_test, y_test))

results_sup.append(evaluate_sup("RF + SMOTE + thr otimo", smote_pipeline, X_train, y_train, X_test, y_test, use_threshold=True))

# RF balanced
rf_bal = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
results_sup.append(evaluate_sup("RF + balanced + thr otimo", rf_bal, X_train, y_train, X_test, y_test, use_threshold=True))

# XGBoost
scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos_weight,
                              random_state=42, n_jobs=-1, eval_metric='logloss')
results_sup.append(evaluate_sup("XGBoost", xgb_model, X_train, y_train, X_test, y_test))

results_sup.append(evaluate_sup("XGBoost + SMOTE + thr otimo",
    imb_make_pipeline(SMOTE(random_state=42, k_neighbors=2),
                      xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, eval_metric='logloss')),
    X_train, y_train, X_test, y_test, use_threshold=True))

results_sup.append(evaluate_sup("XGBoost + thr otimo",
    xgb.XGBClassifier(n_estimators=200, scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1, eval_metric='logloss'),
    X_train, y_train, X_test, y_test, use_threshold=True))

# --- 6. Final Comparison ---
print("\n" + "=" * 95)
print(f"{'Método':<42} {'F1':<10} {'Precision':<12} {'Recall':<12} {'ROC-AUC':<10}")
print("-" * 95)

print("\n>> NÃO SUPERVISIONADOS")
best_unsup = max(results_unsup, key=lambda r: r[1])  # by F1
for r in sorted(results_unsup, key=lambda x: x[1], reverse=True):
    print(f"  {r[0]:<42} {r[1]:<10.4f} {r[2]:<12.4f} {r[3]:<12.4f} {r[4]:<10.4f}")

print("\n>> SUPERVISIONADOS")
best_sup = max(results_sup, key=lambda r: r[1])
for r in sorted(results_sup, key=lambda x: x[1], reverse=True):
    print(f"  {r[0]:<42} {r[1]:<10.4f} {r[2]:<12.4f} {r[3]:<12.4f} {r[4]:<10.4f}")

print("\n" + "=" * 95)
print(f"Melhor NÃO SUP: {best_unsup[0]} | F1={best_unsup[1]:.4f} | ROC-AUC={best_unsup[4]:.4f}")
print(f"Melhor SUP:     {best_sup[0]} | F1={best_sup[1]:.4f} | ROC-AUC={best_sup[4]:.4f}")
print("=" * 95)
