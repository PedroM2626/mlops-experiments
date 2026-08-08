import numpy as np, pandas as pd, warnings, time, re
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("=" * 95)
print("SENTI-PRED COMO SUPERVISED CLUSTERING")
print("=" * 95)

# ============================================================
# Carregar dados (amostra)
# ============================================================
print("\n>> Carregando Twitter data...")

df = pd.read_csv(
    "D:\\mlops-experiments\\experiments\\senti-pred-variations\\Senti-Pred-remake2\\data\\raw\\twitter_training.csv",
    header=None, names=["id", "entity", "sentiment", "text"],
    encoding="utf-8"
).dropna()

# Mapa de sentimento para numerico
sent_map = {"Positive": 0, "Negative": 1, "Neutral": 2, "Irrelevant": 3}
df["label"] = df["sentiment"].map(sent_map)
df = df[df["label"].notna()]

print(f"  Total: {len(df)} tweets")
print(f"  Distribuicao: {df['sentiment'].value_counts().to_dict()}")

# ============================================================
# Preprocessing simples + TF-IDF
# ============================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_clean"] = df["text"].apply(clean_text)

# Amostra para velocidade (2k treino, 800 teste)
df_sample = df.groupby("sentiment").sample(min(2000, len(df) // 4), random_state=42)
X_text = df_sample["text_clean"].values
y = df_sample["label"].values.astype(int)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n  Treino: {len(X_train_text)} | Teste: {len(X_test_text)}")

# TF-IDF (menos features)
t0 = time.time()
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 1), sublinear_tf=True)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)
print(f"  TF-IDF: {X_train.shape[1]} features | {time.time()-t0:.1f}s")

n_classes = len(sent_map)

# ============================================================
# Experimento
# ============================================================
results = []

# --- CLASSIFICACAO (baseline) ---
print("\n" + "=" * 95)
print("1. CLASSIFICACAO (supervisionado puro)")
print("=" * 95)

cls = [
    ("LogisticRegression", LogisticRegression(C=10, max_iter=1000, random_state=42)),
    ("LinearSVC", LinearSVC(C=0.5, random_state=42, max_iter=2000)),
]

for name, model in cls:
    t0 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    t1 = time.time()
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    results.append((f"CLF: {name}", "Classificacao", acc, f1, t1-t0, y_pred))
    print(f"  {name:<25} ACC={acc:.4f} F1={f1:.4f} tempo={t1-t0:.2f}s")

# --- SUPERVISED CLUSTERING (aprender projecao, depois clusterizar) ---
print("\n" + "=" * 95)
print("2. SUPERVISED CLUSTERING (LDA -> K-Means)")
print("=" * 95)
print("  Aprende uma projecao com labels, DEPOIS clusteriza\n")

# LDA precisa de matriz densa e reduz dimensao para n_classes-1
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()

for frac in [0.05, 0.15, 0.30, 1.0]:
    if frac < 1.0:
        X_lab, _, y_lab, _ = train_test_split(
            X_train_dense, y_train, train_size=max(int(len(X_train_dense)*frac), n_classes),
            random_state=42, stratify=y_train)
    else:
        X_lab, y_lab = X_train_dense, y_train

    t0 = time.time()
    # Passo 1: LDA aprende projecao supervisionada
    lda = LDA(n_components=n_classes-1)
    lda.fit(X_lab, y_lab)

    # Passo 2: transformar dados
    X_train_proj = lda.transform(X_train_dense)
    X_test_proj = lda.transform(X_test_dense)

    # Passo 3: K-Means no espaco transformado
    km = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    km.fit(X_train_proj)
    y_pred_cluster = km.predict(X_test_proj)

    t1 = time.time()

    # ARI mede qualidade do agrupamento (rotulos arbitrarios)
    ari = adjusted_rand_score(y_test, y_pred_cluster)
    # Para comparar com classificacao, mapeamos clusters para labels
    # (melhor alinhamento via permutacao)
    from scipy.stats import mode
    labels_map = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        mask = y_pred_cluster == c
        if mask.sum() > 0:
            labels_map[c] = mode(y_test[mask])[0]
    y_pred_mapped = labels_map[y_pred_cluster]
    acc = accuracy_score(y_test, y_pred_mapped)
    f1 = f1_score(y_test, y_pred_mapped, average="macro")

    pct = f"{frac*100:.0f}%"
    results.append((f"SC: LDA->KMeans ({pct})", "Sup. Clustering", acc, f1, t1-t0, y_pred_mapped))
    print(f"  LDA->KMeans (labels={pct:<4}) ACC={acc:.4f} F1={f1:.4f} ARI={ari:.4f} tempo={t1-t0:.2f}s")

# --- K-MEANS PURO (sem supervisao) ---
print("\n" + "=" * 95)
print("3. K-MEANS PURO (sem supervisao)")
print("=" * 95)

t0 = time.time()
km_raw = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
km_raw.fit(X_train_dense)
y_pred_cluster = km_raw.predict(X_test_dense)
t1 = time.time()

ari = adjusted_rand_score(y_test, y_pred_cluster)
labels_map = np.zeros(n_classes, dtype=int)
for c in range(n_classes):
    mask = y_pred_cluster == c
    if mask.sum() > 0:
        labels_map[c] = mode(y_test[mask])[0]
y_pred_mapped = labels_map[y_pred_cluster]
acc = accuracy_score(y_test, y_pred_mapped)
f1 = f1_score(y_test, y_pred_mapped, average="macro")
results.append((f"K-Means puro (TF-IDF)", "Nao sup.", acc, f1, t1-t0, y_pred_mapped))
print(f"  K-Means (TF-IDF original)       ACC={acc:.4f} F1={f1:.4f} ARI={ari:.4f} tempo={t1-t0:.2f}s")

# --- PCA + K-MEANS (reducao nao supervisionada) ---
print("\n" + "=" * 95)
print("4. PCA + K-MEANS (reducao NAO supervisionada)")
print("=" * 95)

for dims in [2, 3, 10]:
    t0 = time.time()
    svd = TruncatedSVD(n_components=dims, random_state=42)
    X_train_pca = svd.fit_transform(X_train)
    X_test_pca = svd.transform(X_test)
    km_pca = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    km_pca.fit(X_train_pca)
    y_pred_cluster = km_pca.predict(X_test_pca)
    t1 = time.time()

    ari = adjusted_rand_score(y_test, y_pred_cluster)
    labels_map = np.zeros(n_classes, dtype=int)
    for c in range(n_classes):
        mask = y_pred_cluster == c
        if mask.sum() > 0:
            labels_map[c] = mode(y_test[mask])[0]
    y_pred_mapped = labels_map[y_pred_cluster]
    acc = accuracy_score(y_test, y_pred_mapped)
    f1 = f1_score(y_test, y_pred_mapped, average="macro")
    results.append((f"PCA({dims})->KMeans", "Nao sup.", acc, f1, t1-t0, y_pred_mapped))
    print(f"  PCA({dims}) -> K-Means          ACC={acc:.4f} F1={f1:.4f} ARI={ari:.4f} tempo={t1-t0:.2f}s")

# ============================================================
# Tabela Final
# ============================================================
print("\n" + "=" * 95)
print("TABELA COMPARATIVA")
print("=" * 95)
print(f"\n{'Metodo':<42} {'Tipo':<18} {'ACC':<10} {'F1':<10} {'Tempo(s)':<10}")
print("-" * 90)

for r in sorted(results, key=lambda x: x[3], reverse=True):
    print(f"  {r[0]:<40} {r[1]:<18} {r[2]:<10.4f} {r[3]:<10.4f} {r[4]:<10.3f}")

print("\n" + "=" * 95)
print("ANALISE")
print("=" * 95)
print("""
Supervised clustering aplicado ao Senti-Pred significa:

EM VEZ DE:
  f(tweet) -> "Positive" | "Negative" | "Neutral" | "Irrelevant"
  (classificacao: rotulo tem significado)

FAZEMOS:
  f(tweet) -> posicao num espaco onde tweets similares ficam juntos
  -> K-Means descobre os grupos
  -> O rotulo (0,1,2,3) e ARBITRARIO, o que importa e o agrupamento

VANTAGENS:
  1. Se um tweet for borderline entre Positive e Neutral, o clustering
     nao precisa decidir - ele fica na fronteira entre os clusters
  2. Se aparecer um novo tipo de sentimento (e.g., sarcasmo), o
     clustering pode formar um 5o cluster - a classificacao nao sabe
     o que fazer com ele
  3. Sub-nuances dentro de cada sentimento sao preservadas como
     sub-estrutura dentro dos clusters

DESVANTAGENS:
  1. Precisamos mapear clusters de volta para rotulos (etapa extra)
  2. Se o numero de clusters nao for bem escolhido, o agrupamento
     pode nao refletir as classes reais
  3. Interpretabilidade e menor: "cluster 2" e menos intuitivo
     que "sentimento negativo"
""")
print("=" * 95)
