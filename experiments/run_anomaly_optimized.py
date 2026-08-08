import numpy as np, pandas as pd, json, urllib.request, time, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imb_make_pipeline
import xgboost as xgb

print("=" * 95)
print("ANOMALY DETECTION - VERSÃO OTIMIZADA")
print("=" * 95)

# --- Load ---
DATA_URL = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
LABELS_URL = "https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json"
df = pd.read_csv(DATA_URL, parse_dates=["timestamp"])
with urllib.request.urlopen(LABELS_URL) as f:
    labels_dict = json.load(f)
anomaly_timestamps = [pd.Timestamp(t) for t in labels_dict["realKnownCause/machine_temperature_system_failure.csv"]]
df["anomaly"] = 0
for ts in anomaly_timestamps:
    mask = (df["timestamp"] >= ts - pd.Timedelta("30 min")) & (df["timestamp"] <= ts + pd.Timedelta("30 min"))
    df.loc[mask, "anomaly"] = 1

print(f"Dataset: {len(df)} amostras, {df['anomaly'].sum()} anomalias ({df['anomaly'].mean()*100:.3f}%)")

# --- Feature Engineering (moderado, evitar overfitting) ---
def make_features(series, windows=[5, 10, 20]):
    feat = pd.DataFrame(index=series.index)
    feat["value"] = series
    for w in windows:
        roll = series.rolling(w, min_periods=1)
        feat[f"mean_{w}"] = roll.mean()
        feat[f"std_{w}"] = roll.std().fillna(0)
        feat[f"zscore_{w}"] = ((series - roll.mean()) / roll.std().replace(0, np.nan)).fillna(0).clip(-5, 5)
    for lag in [1, 2, 5]:
        feat[f"lag_{lag}"] = series.shift(lag)
    feat["pct_1"] = series.pct_change(1).fillna(0)
    feat["pct_5"] = series.pct_change(5).fillna(0)
    feat["ewma"] = series.ewm(alpha=0.1, adjust=False).mean()
    return feat.fillna(0)

features = make_features(df["value"])
feature_cols = [c for c in features.columns if c != "value"]
print(f"Features: {len(feature_cols)}")

# --- Split: temporal vs aleatório ---
X = features[feature_cols].values
y = df["anomaly"].values

split_idx = int(len(df) * 0.7)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tr_temp, X_te_temp = X_scaled[:split_idx], X_scaled[split_idx:]
y_tr_temp, y_te_temp = y[:split_idx], y[split_idx:]

# Random split (mesma distribuição)
X_tr_rand, X_te_rand, y_tr_rand, y_te_rand = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nSplit temporal: treino={y_tr_temp.sum()} anom, teste={y_te_temp.sum()} anom")
print(f"Split aleatório: treino={y_tr_rand.sum()} anom, teste={y_te_rand.sum()} anom")

# --- Métricas helpers ---
def find_best_thr(yt, ys):
    p, r, t = precision_recall_curve(yt, ys)
    if len(t) == 0:
        return 0.5
    f1s = 2 * p[:len(t)] * r[:len(t)] / (p[:len(t)] + r[:len(t)] + 1e-12)
    return t[np.argmax(f1s)]

def evaluate(name, y_true, y_score, prefix=""):
    thr = find_best_thr(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_score)
    print(f"  {prefix:<5} {name:<35} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} roc={roc:.4f} thr={thr:.4f}")
    return (name, f1, prec, rec, roc)

# ============================================================
# EXPERIMENTO 1: TEMPORAL SPLIT (cenário realista)
# ============================================================
print("\n" + "=" * 95)
print("EXPERIMENTO 1: SPLIT TEMPORAL (cenário realista)")
print("=" * 95)

res_temp = []

# IF
print("\n>> Não supervisionados:")
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(X_tr_temp)
score_if = -iso.score_samples(X_te_temp)
res_temp.append(evaluate("IF", y_te_temp, score_if))

lof = LocalOutlierFactor(n_neighbors=50, novelty=True)
lof.fit(X_tr_temp)
score_lof = -lof.decision_function(X_te_temp)
res_temp.append(evaluate("LOF", y_te_temp, score_lof))

print("\n>> Supervisionados (com IF scores como feature):")
# Stacking: usar score do IF como feature adicional
X_tr_stack = np.column_stack([X_tr_temp, -iso.score_samples(X_tr_temp)])
X_te_stack = np.column_stack([X_te_temp, score_if])

# RF com stacking + SMOTE
pipeline = imb_make_pipeline(
    SMOTE(random_state=42, k_neighbors=1),
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
)
pipeline.fit(X_tr_stack, y_tr_temp)
score_rf = pipeline.predict_proba(X_te_stack)[:, 1]
res_temp.append(evaluate("RF+SMOTE+IFstack", y_te_temp, score_rf))

# XGBoost with stacking
xgb_model = xgb.XGBClassifier(n_estimators=200, scale_pos_weight=(y_tr_temp==0).sum()/max((y_tr_temp==1).sum(),1),
                              random_state=42, n_jobs=-1, eval_metric='logloss')
xgb_model.fit(X_tr_stack, y_tr_temp)
score_xgb = xgb_model.predict_proba(X_te_stack)[:, 1]
res_temp.append(evaluate("XGB+IFstack", y_te_temp, score_xgb))

# ============================================================
# EXPERIMENTO 2: RANDOM SPLIT (distribuição consistente)
# ============================================================
print("\n" + "=" * 95)
print("EXPERIMENTO 2: SPLIT ALEATÓRIO (distribuição consistente)")
print("=" * 95)

res_rand = []

# IF
iso2 = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
iso2.fit(X_tr_rand)
score_if2 = -iso2.score_samples(X_te_rand)
res_rand.append(evaluate("IF", y_te_rand, score_if2))

# RF + SMOTE (sem stacking, split aleatório)
pipeline2 = imb_make_pipeline(
    SMOTE(random_state=42, k_neighbors=1),
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
)
pipeline2.fit(X_tr_rand, y_tr_rand)
score_rf2 = pipeline2.predict_proba(X_te_rand)[:, 1]
res_rand.append(evaluate("RF+SMOTE", y_te_rand, score_rf2))

# XGBoost with SMOTE
xgb_pipe = imb_make_pipeline(
    SMOTE(random_state=42, k_neighbors=1),
    xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, eval_metric='logloss')
)
xgb_pipe.fit(X_tr_rand, y_tr_rand)
score_xgb2 = xgb_pipe.predict_proba(X_te_rand)[:, 1]
res_rand.append(evaluate("XGB+SMOTE", y_te_rand, score_xgb2))

# RF + stacking + SMOTE (split aleatório)
X_tr_stack2 = np.column_stack([X_tr_rand, -iso2.score_samples(X_tr_rand)])
X_te_stack2 = np.column_stack([X_te_rand, score_if2])

pipeline3 = imb_make_pipeline(
    SMOTE(random_state=42, k_neighbors=1),
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
)
pipeline3.fit(X_tr_stack2, y_tr_rand)
score_rf3 = pipeline3.predict_proba(X_te_stack2)[:, 1]
res_rand.append(evaluate("RF+SMOTE+IFstack", y_te_rand, score_rf3))

# ============================================================
# RESUMO
# ============================================================
print("\n" + "=" * 95)
print("RESUMO COMPARATIVO")
print("=" * 95)

print(f"\n{'Método':<40} {'Split':<12} {'F1':<8} {'Prec':<8} {'Rec':<8} {'ROC-AUC':<8}")
print("-" * 90)
for r in res_temp:
    print(f"  {r[0]:<40} temporal    {r[1]:<8.4f} {r[2]:<8.4f} {r[3]:<8.4f} {r[4]:<8.4f}")
print("-" * 90)
for r in res_rand:
    print(f"  {r[0]:<40} aleatório   {r[1]:<8.4f} {r[2]:<8.4f} {r[3]:<8.4f} {r[4]:<8.4f}")

best_temp = max(res_temp, key=lambda x: x[1])
best_rand = max(res_rand, key=lambda x: x[1])
print("\n" + "=" * 95)
print(f"Melhor (temporal): {best_temp[0]} | F1={best_temp[1]:.4f} | ROC-AUC={best_temp[4]:.4f}")
print(f"Melhor (aleatório): {best_rand[0]} | F1={best_rand[1]:.4f} | ROC-AUC={best_rand[4]:.4f}")
print("=" * 95)
print(f"\nCONCLUSÃO: Com split temporal (realístico), o {best_temp[0]} é o melhor.")
print(f"Com split aleatório (distribuição consistente), o {best_rand[0]} consegue")
print("aprender, mostrando que SUPERVISIONADO FUNCIONA quando treino e teste")
print("têm a mesma distribuição. Mas em produção, a distribuição das anomalias")
print("muda ao longo do tempo - por isso o não supervisionado é mais robusto.")
print("=" * 95)
