import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("=" * 95)
print("SUPERVISED CLUSTERING: O QUE E E COMO FUNCIONA")
print("=" * 95)

# ============================================================
# O que diferencia supervised clustering de classificacao?
# ============================================================
print("""
CLASSIFICACAO:        f(x) -> rotulo (0, 1, 2...)
  O rotulo TEM SIGNIFICADO. "Classe 0" e "setosa" sao a mesma coisa.
  Treina com pares (x, y) onde y e o rotulo verdadeiro.

SUPERVISED CLUSTERING: f(x) -> cluster_id
  O cluster_id NAO TEM SIGNIFICADO. O que importa e o AGRUPAMENTO.
  Treina com pares (x, y) onde y e o cluster, mas o numero do cluster
  e arbitrario - o que importa e que pontos do mesmo cluster fiquem juntos.
""")

# ============================================================
# 1. Dataset
# ============================================================
print("=" * 95)
print("DEMONSTRACAO COM IRIS")
print("=" * 95)

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X_scaled = StandardScaler().fit_transform(X)

# Split para mostrar generalizacao
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y)

n_classes = len(np.unique(y))

# ============================================================
# 2. Abordagens
# ============================================================

results = []

# --- A) NAO SUPERVISIONADO PURO ---
print("\n--- NAO SUPERVISIONADO PURO ---")

t0 = time.time()
km = KMeans(n_clusters=n_classes, random_state=42, n_init=10).fit(X_scaled)
t1 = time.time()
ari = adjusted_rand_score(y, km.labels_)
print(f"  K-Means (k={n_classes}) em TODOS os dados       ARI={ari:.4f}  tempo={t1-t0:.3f}s")
results.append(("K-Means (dados todos)", ari, "Nao usa labels"))

t0 = time.time()
km_test = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
km_test.fit(X_train)
labels_test = km_test.predict(X_test)
t1 = time.time()
ari = adjusted_rand_score(y_test, labels_test)
print(f"  K-Means treinado no TRAIN, predito TEST  ARI={ari:.4f}  tempo={t1-t0:.3f}s")
results.append(("K-Means (train->test)", ari, "Nao usa labels"))

# --- B) SEMI-SUPERVISIONADO (Seeded K-Means) ---
print("\n--- SEMI-SUPERVISIONADO ---")

# Usar centroides dos dados rotulados pra inicializar
for frac in [0.05, 0.10, 0.20]:
    X_lab, _, y_lab, _ = train_test_split(
        X_train, y_train, train_size=max(int(len(X_train)*frac), n_classes),
        random_state=42, stratify=y_train)

    centroids = np.array([X_lab[y_lab == c].mean(axis=0) for c in np.unique(y_lab)])

    t0 = time.time()
    km_seeded = KMeans(n_clusters=n_classes, init=centroids, random_state=42, n_init=1)
    km_seeded.fit(X_train)
    labels_test = km_seeded.predict(X_test)
    t1 = time.time()
    ari = adjusted_rand_score(y_test, labels_test)
    print(f"  Seeded K-Means ({frac*100:.0f}% labels)            ARI={ari:.4f}  tempo={t1-t0:.3f}s")
    results.append((f"Seeded K-Means ({frac*100:.0f}%)", ari, f"Usa {frac*100:.0f}% labels p/ inicializar"))

# --- C) SUPERVISED CLUSTERING (via transformacao supervisionada) ---
print("\n--- SUPERVISED CLUSTERING ---")
print("  Ideia: usar labels para APRENDER UMA TRANSFORMACAO que")
print("  aproxime pontos do mesmo cluster e afaste pontos de clusters")
print("  diferentes. DEPOIS clusterizar no espaco transformado.\n")

# Metodo 1: LDA (maximiza separacao entre classes)
# LDA encontra projecao que maximiza separacao entre classes conhecidas
# Depois clusterizamos NESSA projecao

for frac in [0.05, 0.10, 0.20]:
    X_lab, _, y_lab, _ = train_test_split(
        X_train, y_train, train_size=max(int(len(X_train)*frac), n_classes),
        random_state=42, stratify=y_train)

    t0 = time.time()
    # Passo 1: aprender projecao supervisionada com labels
    lda = LDA(n_components=n_classes-1)
    lda.fit(X_lab, y_lab)

    # Passo 2: transformar TODOS os dados com essa projecao
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    # Passo 3: clusterizar no espaco transformado
    km_lda = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    km_lda.fit(X_train_lda)
    labels_test = km_lda.predict(X_test_lda)
    t1 = time.time()
    ari = adjusted_rand_score(y_test, labels_test)
    print(f"  LDA->KMeans ({frac*100:.0f}% labels)                  ARI={ari:.4f}  tempo={t1-t0:.3f}s")
    results.append((f"LDA->KMeans ({frac*100:.0f}%)", ari, f"Projecao LDA com {frac*100:.0f}% labels"))

# Metodo 2: LDA + DBSCAN (nao precisa de K)
for frac in [0.05, 0.10, 0.20]:
    X_lab, _, y_lab, _ = train_test_split(
        X_train, y_train, train_size=max(int(len(X_train)*frac), n_classes),
        random_state=42, stratify=y_train)

    t0 = time.time()
    lda = LDA(n_components=n_classes-1)
    lda.fit(X_lab, y_lab)
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    db = DBSCAN(eps=0.5, min_samples=3)
    db.fit(X_train_lda)
    labels_test = db.fit_predict(np.vstack([X_train_lda, X_test_lda]))[len(X_train_lda):]
    t1 = time.time()
    if len(set(labels_test) - {-1}) >= 2:
        ari = adjusted_rand_score(y_test, labels_test)
        print(f"  LDA->DBSCAN ({frac*100:.0f}% labels)                 ARI={ari:.4f}  tempo={t1-t0:.3f}s")
        results.append((f"LDA->DBSCAN ({frac*100:.0f}%)", ari, f"Projecao LDA + DBSCAN"))

# --- D) SUPERVISIONADO PURO (classificacao) ---
print("\n--- SUPERVISIONADO PURO (CLASSIFICACAO) ---")

for frac in [0.05, 0.10, 0.20]:
    X_lab, X_rem, y_lab, y_rem = train_test_split(
        X_train, y_train, train_size=max(int(len(X_train)*frac), n_classes),
        random_state=42, stratify=y_train)

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_lab, y_lab)
    y_pred = rf.predict(X_test)
    t1 = time.time()
    ari = adjusted_rand_score(y_test, y_pred)
    print(f"  Random Forest ({frac*100:.0f}% labels)               ARI={ari:.4f}  tempo={t1-t0:.3f}s")
    results.append((f"RF ({frac*100:.0f}%)", ari, f"Classificador com {frac*100:.0f}% labels"))

# ============================================================
# 3. Tabela comparativa
# ============================================================
print(f"\n{'='*95}")
print(f"{'Metodo':<35} {'ARI':<10} {'Estrategia'}")
print(f"{'-'*95}")

for r in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"  {r[0]:<33} {r[1]:<10.4f} {r[2]}")

print("\n" + "=" * 95)
print("DIFERENCA ENTRE SUPERVISED CLUSTERING E CLASSIFICACAO")
print("=" * 95)
print("""
SUPERVISED CLUSTERING:
  - Usa labels para APRENDER UMA METRICA ou TRANSFORMACAO
  - O resultado final e um AGRUPAMENTO (clusters)
  - O rotulo '0', '1', '2' e ARBITRARIO
  - O que importa: pontos similares estao juntos?
  - Ex: LDA projeta dados para separar classes, depois K-Means agrupa

CLASSIFICACAO:
  - Usa labels para APRENDER UMA FRONTEIRA DE DECISAO
  - O resultado final e um ROTULO PREDITO
  - O rotulo '0', '1', '2' tem SIGNIFICADO (e.g., 'setosa')
  - O que importa: o rotulo predito esta correto?
  - Ex: RF aprende regras para distinguir classes

QUAL A DIFERENCA NA PRATICA?
  - Classificacao precisa de exemplos de CADA classe para aprender
  - Supervised clustering pode generalizar para NOVOS clusterings
    porque aprende uma METRICA, nao uma fronteira

  Exemplo: se voce ensina um classificador a distinguir gatos de caes,
  ele nao sabe agrupar racas de gatos. Mas se voce ensina uma metrica
  de similaridade (estes 2 sao da mesma especie, estes 2 sao diferentes),
  o supervised clustering consegue agrupar racas de gatos que ele nunca
  viu antes - porque aprendeu o CONCEITO de "mesma especie".
""")
print("=" * 95)
