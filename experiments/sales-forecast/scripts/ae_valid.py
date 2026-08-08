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

# ==== AE embeddings (somente semanas 1-47, SEM vazamento) ====
ART = joblib.load(ROOT + r'\artifacts\BACKUP\sales_forecaster_v2_final.joblib')
model_champ = ART['model']
feature_names = ART['feature_names']
cat_features = ART['categorical_features']
best_params = ART['best_params']
log('[LOAD] campeão carregado')

log('[AE] construindo matriz de séries...')
train_weeks = df_feat['semana'] <= 47
hist = df_feat[train_weeks][['pdv', 'sku', 'semana', 'quantidade']]
pivot = hist.pivot_table(index=['pdv', 'sku'], columns='semana', values='quantidade', fill_value=0)
pivot = pivot.reindex(columns=range(1, 48), fill_value=0)
X_hist = np.log1p(pivot.values).astype(np.float32)
X_hist = np.nan_to_num(X_hist, nan=0.0, posinf=0.0, neginf=0.0)
series_keys = pivot.index.to_frame(index=False)
series_keys.columns = ['pdv', 'sku']
log(f'[AE] séries: {len(pivot):,}, mem {mem():.0f} MB')

emb_dim = 8
scaler = StandardScaler()
X_s = scaler.fit_transform(X_hist)
ae = MLPRegressor(hidden_layer_sizes=(32, emb_dim), activation='relu', solver='adam',
                  alpha=0.001, max_iter=80, random_state=42, early_stopping=True,
                  validation_fraction=0.1, n_iter_no_change=10, batch_size=4096)
ae.fit(X_s, X_s)
log(f'[AE] treinado, iters={ae.n_iter_}, loss={ae.loss_:.4f}')

# Extrair embedding = ativação do bottleneck (2ª camada escondida)
# arquitetura: input -> 32 -> emb_dim -> output; coefs_[0]=in->32, coefs_[1]=32->emb
def relu(x):
    return np.maximum(0, x)

Z1 = relu(X_s @ ae.coefs_[0] + ae.intercepts_[0])
Z2 = relu(Z1 @ ae.coefs_[1] + ae.intercepts_[1])   # bottleneck = embedding
emb_df = pd.DataFrame(Z2.astype(np.float32), columns=[f'ae_emb_{i}' for i in range(emb_dim)])
emb_df[['pdv', 'sku']] = series_keys
emb_map = emb_df.set_index(['pdv', 'sku'])
del pivot, X_hist, X_s, Z1, Z2; gc.collect()
log(f'[AE] embeddings extraídos, mem {mem():.0f} MB')

# ==== Merge embeddings nas features ====
df_feat = df_feat.set_index(['pdv', 'sku'])
df_feat = df_feat.join(emb_map, how='left')
df_feat.reset_index(inplace=True)
ae_cols = [f'ae_emb_{i}' for i in range(emb_dim)]
df_feat[ae_cols] = df_feat[ae_cols].fillna(0)
log(f'[MERGE] done, mem {mem():.0f} MB, {time.time()-t0:.0f}s')

# ==== Preparar treino/validação ====
val = df_feat[df_feat['semana'] >= 48].copy()
train = df_feat[df_feat['semana'] < 48].copy()
del df_feat; gc.collect()

def prepare(df_in, cat_feats_all):
    df_in = df_in.copy()
    for col in cat_feats_all:
        df_in[col] = df_in[col].astype('category')
    return df_in

train = prepare(train, cat_features)
val = prepare(val, cat_features)
for col in cat_features:
    cats = pd.concat([train[col], val[col]]).astype('category').cat.categories
    train[col] = pd.Categorical(train[col], categories=cats)
    val[col] = pd.Categorical(val[col], categories=cats)

y_train = train['quantidade']; y_val = val['quantidade']
X_train = train[feature_names + ae_cols]
X_val = val[feature_names + ae_cols]
base_train = train[feature_names]
base_val = val[feature_names]
log(f'[DATA] train {len(train):,} val {len(val):,}, mem {mem():.0f} MB, {time.time()-t0:.0f}s')

# ==== Modelo BASELINE (reproduzir campeão) ====
log('[BASELINE] treinando LightGBM sem embeddings...')
params = dict(best_params)
params.pop('n_estimators', None)
params['objective'] = 'regression_l1'
params['random_state'] = 42
params['n_jobs'] = -1
m_base = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
m_base.fit(base_train, y_train, eval_set=[(base_val, y_val)], eval_metric='mae',
           callbacks=[lgb.early_stopping(50, verbose=False)],
           categorical_feature=cat_features)
mae_base = mean_absolute_error(y_val, m_base.predict(base_val))
log(f'[BASELINE] MAE val = {mae_base:.4f} (best_iter={m_base.best_iteration_})')

# ==== Modelo COM AE embeddings ====
log('[AE MODEL] treinando LightGBM com embeddings...')
m_ae = lgb.LGBMRegressor(n_estimators=1000, verbosity=-1, **params)
m_ae.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae',
          callbacks=[lgb.early_stopping(50, verbose=False)],
          categorical_feature=cat_features)
mae_ae = mean_absolute_error(y_val, m_ae.predict(X_val))
log(f'[AE MODEL] MAE val = {mae_ae:.4f} (best_iter={m_ae.best_iteration_})')

log('=' * 60)
log(f'RESULTADO: baseline MAE={mae_base:.4f} | +AE embeddings MAE={mae_ae:.4f}')
log(f'Diferença = {mae_ae - mae_base:+.4f} ({(mae_ae-mae_base)/mae_base*100:+.2f}%)')
log(f'mem final {mem():.0f} MB, tempo total {time.time()-t0:.0f}s')