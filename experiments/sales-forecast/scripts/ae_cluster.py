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
from sklearn.cluster import KMeans
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

# ================= AE (embedding p/ clustering, SEM leakage) =================
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
W1, W2, _ = ae.coefs_[0], ae.coefs_[1], ae.coefs_[2]
b1, b2, _ = ae.intercepts_[0], ae.intercepts_[1], ae.intercepts_[2]

def embed_batch(Xs):
    Z1 = relu(Xs @ W1 + b1)
    Z2 = relu(Z1 @ W2 + b2)
    return Z2.astype(np.float32)

# embedding de perfil (semanas 1-47) — usado p/ clustering de SÉRIES
emb_profile = embed_batch(X_s)
emb_profile_df = pd.DataFrame(emb_profile, columns=ae_cols)
emb_profile_df[['pdv', 'sku']] = series_keys
log(f'[EMB] perfil pronto, {len(emb_profile_df):,} séries, mem {mem():.0f} MB')
del pivot, X_hist, X_s, hist; gc.collect()

# ================= Clustering =================
emb_for_cluster = emb_profile_df[ae_cols].values

def evaluate_clusters(k):
    log(f'\n{"="*60}\n[CLUSTER] K={k}...')
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(emb_for_cluster)
    cl_df = pd.DataFrame({'pdv': series_keys['pdv'], 'sku': series_keys['sku'], 'cluster': clusters})
    # tamanho p/ diagnóstico
    sizes = cl_df['cluster'].value_counts().sort_index()
    log(f'[CLUSTER k={k}] tamanhos: {dict(sizes)}')

    # merge cluster_id em df_feat
    df_w = df_feat.merge(cl_df, on=['pdv', 'sku'], how='left')
    df_w['cluster'] = df_w['cluster'].fillna(-1).astype(int)

    val = df_w[df_w['semana'] >= 48].copy()
    train = df_w[df_w['semana'] < 48].copy()
    del df_w; gc.collect()

    def prepare(df_in):
        df_in = df_in.copy()
        for col in cat_features + ['cluster']:
            df_in[col] = df_in[col].astype('category')
        return df_in

    train = prepare(train)
    val = prepare(val)
    # harmonize categories
    for col in cat_features:
        cats = pd.concat([train[col], val[col]]).astype('category').cat.categories
        train[col] = pd.Categorical(train[col], categories=cats)
        val[col] = pd.Categorical(val[col], categories=cats)
    cats_c = pd.concat([train['cluster'], val['cluster']]).astype('category').cat.categories
    train['cluster'] = pd.Categorical(train['cluster'], categories=cats_c)
    val['cluster'] = pd.Categorical(val['cluster'], categories=cats_c)

    y_train = train['quantidade']; y_val = val['quantidade']
    params = dict(best_params)
    params.pop('n_estimators', None)
    params['objective'] = 'regression_l1'
    params['random_state'] = 42
    params['n_jobs'] = -1

    # ---- Cenário A: Global + cluster_id feature ----
    X_train = train[feature_names + ['cluster']]
    X_val = val[feature_names + ['cluster']]
    log(f'[k={k}A] global+cluster_id treinando...')
    mA = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
    mA.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae',
           callbacks=[lgb.early_stopping(50, verbose=False)],
           categorical_feature=cat_features + ['cluster'])
    mae_A = mean_absolute_error(y_val, mA.predict(X_val))
    log(f'[k={k}A] global+cluster_id MAE = {mae_A:.4f} (best_iter={mA.best_iteration_})')
    del mA; gc.collect()

    # ---- Cenário B: Per-cluster models ----
    preds = np.zeros(len(y_val))
    train_cl = train['cluster'].astype(int).values
    val_cl = val['cluster'].astype(int).values
    train_idx_clusters = pd.Series(train_cl).index if False else None
    log(f'[k={k}B] per-cluster treinando...')
    for c in sorted(set(train_cl) | set(val_cl)):
        tr_mask = train_cl == c
        va_mask = val_cl == c
        if tr_mask.sum() < 100 or va_mask.sum() == 0:
            # cluster pequeno: usa modelo global fallback (treina nos dados disponíveis mesmos)
            if tr_mask.sum() > 0:
                mB = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
                mB.fit(train[feature_names][tr_mask], y_train[tr_mask],
                       eval_set=[(val[feature_names][va_mask], y_val[va_mask])], eval_metric='mae',
                       callbacks=[lgb.early_stopping(50, verbose=False)],
                       categorical_feature=cat_features)
                preds[va_mask] = mB.predict(val[feature_names][va_mask])
                del mB
            continue
        mB = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
        mB.fit(train[feature_names][tr_mask], y_train[tr_mask],
               eval_set=[(val[feature_names][va_mask], y_val[va_mask])], eval_metric='mae',
               callbacks=[lgb.early_stopping(50, verbose=False)],
               categorical_feature=cat_features)
        preds[va_mask] = mB.predict(val[feature_names][va_mask])
        log(f'  cluster {c}: train={tr_mask.sum():,} val={va_mask.sum():,} MAE={mean_absolute_error(y_val[va_mask], preds[va_mask]):.4f}')
        del mB; gc.collect()
    mae_B = mean_absolute_error(y_val, preds)
    log(f'[k={k}B] per-cluster MAE = {mae_B:.4f}')
    del train, val; gc.collect()

    return mae_A, mae_B, dict(sizes)

# ================= Baseline (sem cluster) =================
val = df_feat[df_feat['semana'] >= 48].copy()
train = df_feat[df_feat['semana'] < 48].copy()
def prepare_base(df_in):
    df_in = df_in.copy()
    for col in cat_features:
        df_in[col] = df_in[col].astype('category')
    return df_in
train = prepare_base(train)
val = prepare_base(val)
for col in cat_features:
    cats = pd.concat([train[col], val[col]]).astype('category').cat.categories
    train[col] = pd.Categorical(train[col], categories=cats)
    val[col] = pd.Categorical(val[col], categories=cats)
y_train = train['quantidade']; y_val = val['quantidade']
base_train = train[feature_names]; base_val = val[feature_names]
del train, val; gc.collect()
params = dict(best_params)
params.pop('n_estimators', None)
params['objective'] = 'regression_l1'
params['random_state'] = 42
params['n_jobs'] = -1
log('[BASELINE] treinando...')
m_base = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
m_base.fit(base_train, y_train, eval_set=[(base_val, y_val)], eval_metric='mae',
           callbacks=[lgb.early_stopping(50, verbose=False)],
           categorical_feature=cat_features)
mae_base = mean_absolute_error(y_val, m_base.predict(base_val))
log(f'[BASELINE] MAE val = {mae_base:.4f} (best_iter={m_base.best_iteration_})')
del m_base, base_train, base_val; gc.collect()

results = {'baseline': mae_base}
for k in [3, 5, 8]:
    mae_A, mae_B, sizes = evaluate_clusters(k)
    results[f'k{k}_global+cluster'] = mae_A
    results[f'k{k}_per-cluster'] = mae_B

log('\n' + '=' * 60)
log('RESULTADO COMPARATIVO:')
log(f'  baseline                               MAE = {results["baseline"]:.4f}')
for k in [3, 5, 8]:
    ma = results[f'k{k}_global+cluster']
    mb = results[f'k{k}_per-cluster']
    log(f'  k={k} global+cluster_id                MAE = {ma:.4f}  (delta {ma-mae_base:+.4f}, {(ma-mae_base)/mae_base*100:+.2f}%)')
    log(f'  k={k} per-cluster models              MAE = {mb:.4f}  (delta {mb-mae_base:+.4f}, {(mb-mae_base)/mae_base*100:+.2f}%)')
log(f'mem final {mem():.0f} MB, tempo total {time.time()-t0:.0f}s')