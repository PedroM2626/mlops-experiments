import os, sys, time, gc, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import psutil

def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def log(*a):
    print(*a, flush=True)

ROOT = r'D:\mlops-experiments\experiments\sales-forecast'
log(f'[START] mem {mem():.0f} MB')

t0 = time.time()
df_vendas = pd.read_parquet(ROOT + r'\data\raw\fato_vendas.parquet')
df_pdvs = pd.read_parquet(ROOT + r'\data\raw\dim_pdvs.parquet')
df_produtos = pd.read_parquet(ROOT + r'\data\raw\dim_produtos.parquet')
df_merged = df_vendas.merge(df_pdvs, left_on='internal_store_id', right_on='pdv', how='inner')
df_merged = df_merged.merge(df_produtos, left_on='internal_product_id', right_on='produto', how='inner')
del df_vendas, df_pdvs, df_produtos; gc.collect()

df_merged['transaction_date'] = pd.to_datetime(df_merged['transaction_date'])
df_merged['ano'] = df_merged['transaction_date'].dt.isocalendar().year
df_merged['semana'] = df_merged['transaction_date'].dt.isocalendar().week

dim_cols = ['categoria_pdv', 'premise', 'categoria', 'subcategoria', 'tipos', 'label', 'marca', 'fabricante']
group_cols = ['ano', 'semana', 'pdv', 'produto'] + dim_cols
agg = df_merged.groupby(group_cols).agg(
    quantidade=('quantity', 'sum'),
    gross_value=('gross_value', 'sum'),
).reset_index()
agg = agg.rename(columns={'produto': 'sku'})
agg['preco_medio_unitario'] = np.where(agg['quantidade'] > 0, agg['gross_value'] / agg['quantidade'], 0.0)
agg.drop(columns=['gross_value'], inplace=True)
del df_merged; gc.collect()
log(f'[AGGR] {len(agg):,} linhas, mem {mem():.0f} MB, {time.time()-t0:.0f}s')

log('[FEAT] iniciando...')
df_feat = agg.sort_values(['pdv', 'sku', 'ano', 'semana']).reset_index(drop=True)
df_feat['trimestre'] = (df_feat['semana'] - 1) // 13 + 1
df_feat['seno_semana'] = np.sin(2 * np.pi * df_feat['semana'] / 52)
df_feat['cosseno_semana'] = np.cos(2 * np.pi * df_feat['semana'] / 52)
g = df_feat.groupby(['pdv', 'sku'])['quantidade']
for lag in [1, 2, 3, 4, 12, 52]:
    df_feat[f'lag_{lag}_semanas'] = g.shift(lag)
gp = df_feat.groupby(['pdv', 'sku'])['preco_medio_unitario']
df_feat['lag_1_preco'] = gp.shift(1)
df_feat['lag_diff_1'] = df_feat['lag_1_semanas'] - df_feat['lag_2_semanas']
shifted = g.shift(1)
tmp = pd.DataFrame({'val': shifted, 'pdv': df_feat['pdv'], 'sku': df_feat['sku']})
tg = tmp.groupby(['pdv', 'sku'])['val']
for w in [4, 12, 52]:
    roll = tg.rolling(window=w, min_periods=1)
    df_feat[f'rolling_mean_{w}_semanas'] = roll.mean().reset_index(level=[0, 1], drop=True)
    df_feat[f'rolling_std_{w}_semanas'] = roll.std().reset_index(level=[0, 1], drop=True)
    df_feat[f'rolling_max_{w}_semanas'] = roll.max().reset_index(level=[0, 1], drop=True)
for w in [4, 12]:
    roll = tg.rolling(window=w, min_periods=1)
    df_feat[f'rolling_min_{w}_semanas'] = roll.min().reset_index(level=[0, 1], drop=True)
m4 = df_feat['rolling_mean_4_semanas']; s4 = df_feat['rolling_std_4_semanas']
df_feat['coef_variacao_4'] = np.where(m4 > 0, s4 / m4, 0.0)
df_feat.fillna(0, inplace=True)
del agg, tmp, shifted; gc.collect()
log(f'[FEAT] done, {len(df_feat):,} linhas, mem {mem():.0f} MB, {time.time()-t0:.0f}s')

ART = joblib.load(ROOT + r'\artifacts\BACKUP\sales_forecaster_v2_final.joblib')
feature_names = ART['feature_names']
cat_features = ART['categorical_features']
best_params = ART['best_params']
log('[LOAD] campeão carregado')

# ================= AE =================
emb_dim = 8
ae_cols = [f'ae_emb_{i}' for i in range(emb_dim)]
log('[AE] construindo matriz de séries (semanas 1-47)...')
hist = df_feat[df_feat['semana'] <= 47][['pdv', 'sku', 'semana', 'quantidade']]
pivot = hist.pivot_table(index=['pdv', 'sku'], columns='semana', values='quantidade', fill_value=0)
pivot = pivot.reindex(columns=range(1, 48), fill_value=0)
X_hist = np.log1p(pivot.values).astype(np.float32)
X_hist = np.nan_to_num(X_hist, nan=0.0, posinf=0.0, neginf=0.0)
series_keys = pivot.index.to_frame(index=False)
series_keys.columns = ['pdv', 'sku']
log(f'[AE] séries: {len(pivot):,}, shape {X_hist.shape}, mem {mem():.0f} MB')

scaler = StandardScaler()
X_s = scaler.fit_transform(X_hist)
ae = MLPRegressor(hidden_layer_sizes=(32, emb_dim), activation='relu', solver='adam',
                  alpha=0.001, max_iter=80, random_state=42, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=10, batch_size=4096)
ae.fit(X_s, X_s)
log(f'[AE] treinado, iters={ae.n_iter_}, loss={ae.loss_:.4f}')

def relu(x):
    return np.maximum(0, x)

W1, W2, W3 = ae.coefs_[0], ae.coefs_[1], ae.coefs_[2]
b1, b2, b3 = ae.intercepts_[0], ae.intercepts_[1], ae.intercepts_[2]

def embed_batch(Xs):
    """Xs: (n, 47) já padronizado pelo scaler fit. Retorna (n, emb_dim) bottleneck."""
    Z1 = relu(Xs @ W1 + b1)
    Z2 = relu(Z1 @ W2 + b2)
    return Z2.astype(np.float32)

# ---- Embedding NAIVE (leaky): usa semanas 1-47 completas para todas as linhas ----
emb_full = embed_batch(X_s)
emb_full_df = pd.DataFrame(emb_full, columns=ae_cols)
emb_full_df[['pdv', 'sku']] = series_keys
log('[AE] embedding naive (leaky) pronto')

# ---- Embedding CAUSAL: para linha na semana w, usa só semanas 1..w-1 ----
# Para cada semana w, faz forward pass da matriz mascarada (semanas >= w zeradas).
means = scaler.mean_.astype(np.float32)
stds = scaler.scale_.astype(np.float32)

emb_by_week = {}
for w in range(2, 47):  # semanas 2..46 causais (semana 47 tratada abaixo)
    mask = np.zeros_like(X_hist, dtype=np.float32)
    mask[:, :w-1] = X_hist[:, :w-1]
    Xw = (mask - means) / stds
    ew = embed_batch(Xw)
    e_df = pd.DataFrame(ew, columns=ae_cols)
    e_df[['pdv', 'sku']] = series_keys
    emb_by_week[w] = e_df
    del mask, Xw, ew
    if w % 10 == 0:
        log(f'[AE causal] semana {w}/47 pronta, mem {mem():.0f} MB')

# semana 1: sem histórico -> zeros
emb_week1 = pd.DataFrame(np.zeros((len(series_keys), emb_dim), dtype=np.float32), columns=ae_cols)
emb_week1[['pdv', 'sku']] = series_keys
emb_by_week[1] = emb_week1
# semana 47 causal = semanas 1..46
mask47 = np.zeros_like(X_hist, dtype=np.float32)
mask47[:, :46] = X_hist[:, :46]
X47 = (mask47 - means) / stds
e47 = embed_batch(X47)
e47_df = pd.DataFrame(e47, columns=ae_cols)
e47_df[['pdv', 'sku']] = series_keys
emb_by_week[47] = e47_df
del mask47, X47, e47; gc.collect()
# semanas >= 48 (validação) usam embedding das semanas 1..47 (= NAIVE, sem vazamento p/ validação)
emb_by_week[48] = emb_full_df.copy()
emb_by_week[49] = emb_full_df.copy()
emb_by_week[50] = emb_full_df.copy()
emb_by_week[51] = emb_full_df.copy()
emb_by_week[52] = emb_full_df.copy()
log('[AE] embedding causal pronto')

# ---- Merge CAUSAL via pd.merge por semana (evita duplicate-index) ----
log('[MERGE causal] iniciando...')
parts = []
for w in sorted(emb_by_week.keys()):
    wk = df_feat[df_feat['semana'] == w]
    if len(wk) > 0:
        merged = wk.merge(emb_by_week[w], on=['pdv', 'sku'], how='left')
        parts.append(merged)
    if w % 10 == 0:
        log(f'[MERGE causal] semana {w} ok, n={(emb_by_week[w].shape[0]):,}, mem {mem():.0f} MB')
df_feat = pd.concat(parts, ignore_index=True)
df_feat[ae_cols] = df_feat[ae_cols].fillna(0)

# preserva embedding CAUSAL em colunas _causal
for c in ae_cols:
    df_feat[c + '_causal'] = df_feat[c]
# sobrescreve ae_cols com embedding NAIVE (leaky) para comparar
df_feat = df_feat.drop(columns=ae_cols)
df_feat = df_feat.merge(emb_full_df[['pdv', 'sku'] + ae_cols], on=['pdv', 'sku'], how='left')
df_feat[ae_cols] = df_feat[ae_cols].fillna(0)
del emb_by_week, emb_full, emb_full_df; gc.collect()
log(f'[MERGE causal+naive] done, {len(df_feat):,} linhas, mem {mem():.0f} MB, {time.time()-t0:.0f}s')

val = df_feat[df_feat['semana'] >= 48].copy()
train = df_feat[df_feat['semana'] < 48].copy()
del df_feat; gc.collect()

def prepare(df_in):
    df_in = df_in.copy()
    for col in cat_features:
        df_in[col] = df_in[col].astype('category')
    return df_in

train = prepare(train)
val = prepare(val)
for col in cat_features:
    cats = pd.concat([train[col], val[col]]).astype('category').cat.categories
    train[col] = pd.Categorical(train[col], categories=cats)
    val[col] = pd.Categorical(val[col], categories=cats)

y_train = train['quantidade']; y_val = val['quantidade']
causal_cols = [c + '_causal' for c in ae_cols]
X_train_ae = train[feature_names + ae_cols]              # naive (leaky)
X_val_ae = val[feature_names + ae_cols]
X_train_causal = train[feature_names + causal_cols]       # causal
X_val_causal = val[feature_names + causal_cols]
base_train = train[feature_names]
base_val = val[feature_names]
del train, val; gc.collect()
log(f'[DATA] train {len(y_train):,} val {len(y_val):,}, mem {mem():.0f} MB, {time.time()-t0:.0f}s')

params = dict(best_params)
params.pop('n_estimators', None)
params['objective'] = 'regression_l1'
params['random_state'] = 42
params['n_jobs'] = -1

log('[BASELINE] treinando LightGBM sem embeddings...')
m_base = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
m_base.fit(base_train, y_train, eval_set=[(base_val, y_val)], eval_metric='mae',
           callbacks=[lgb.early_stopping(50, verbose=False)],
           categorical_feature=cat_features)
mae_base = mean_absolute_error(y_val, m_base.predict(base_val))
log(f'[BASELINE] MAE val = {mae_base:.4f} (best_iter={m_base.best_iteration_})')
del m_base; gc.collect()

log('[NAIVE-LEAKY MODEL] treinando LightGBM com embeddings NAIVE (vazamento)...')
m_naive = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
m_naive.fit(X_train_ae, y_train, eval_set=[(X_val_ae, y_val)], eval_metric='mae',
            callbacks=[lgb.early_stopping(50, verbose=False)],
            categorical_feature=cat_features)
mae_naive = mean_absolute_error(y_val, m_naive.predict(X_val_ae))
log(f'[NAIVE-LEAKY MODEL] MAE val = {mae_naive:.4f} (best_iter={m_naive.best_iteration_})')
del m_naive, X_train_ae, X_val_ae; gc.collect()

log('[CAUSAL MODEL] treinando LightGBM com embeddings CAUSAIS...')
m_ae = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
m_ae.fit(X_train_causal, y_train, eval_set=[(X_val_causal, y_val)], eval_metric='mae',
          callbacks=[lgb.early_stopping(50, verbose=False)],
          categorical_feature=cat_features)
mae_ae = mean_absolute_error(y_val, m_ae.predict(X_val_causal))
log(f'[CAUSAL MODEL] MAE val = {mae_ae:.4f} (best_iter={m_ae.best_iteration_})')

log('=' * 60)
log(f'RESULTADO:')
log(f'  baseline          MAE = {mae_base:.4f}')
log(f'  +AE naive (leaky) MAE = {mae_naive:.4f}  (delta {mae_naive-mae_base:+.4f}, {(mae_naive-mae_base)/mae_base*100:+.2f}%)')
log(f'  +AE causal        MAE = {mae_ae:.4f}  (delta {mae_ae-mae_base:+.4f}, {(mae_ae-mae_base)/mae_base*100:+.2f}%)')
log(f'mem final {mem():.0f} MB, tempo total {time.time()-t0:.0f}s')