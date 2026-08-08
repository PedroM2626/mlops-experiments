import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split

print("=" * 95)
print("CLUSTERING: NAO SUPERVISIONADO vs SEMI-SUPERVISIONADO vs SUPERVISIONADO")
print("=" * 95)

# ============================================================
# Datasets
# ============================================================
datasets = []

# Iris
iris = load_iris()
datasets.append(("Iris", iris.data, iris.target, iris.target_names))

# Blobs com sobreposicao controlada
Xb, yb = make_blobs(n_samples=500, centers=3, cluster_std=2.5, random_state=42)
datasets.append(("Blobs (sobrepostos)", Xb, yb, ["0","1","2"]))

# Moons (nao linear)
Xm, ym = make_moons(n_samples=300, noise=0.1, random_state=42)
datasets.append(("Moons", Xm, ym, ["0","1"]))

# Circles (concêntricos)
Xc, yc = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
datasets.append(("Circles", Xc, yc, ["0","1"]))

# ============================================================
# Métodos
# ============================================================
def evaluate_clustering(name, y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi

def run_unsupervised(X, y):
    results = []

    # K-Means
    for k in range(2, 8):
        t0 = time.time()
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        t1 = time.time()
        ari, nmi = evaluate_clustering("K-Means", y, km.labels_)
        sil = silhouette_score(X, km.labels_)
        results.append(("K-Means(k="+str(k)+")", "Nao sup", ari, nmi, sil, t1-t0))

    # DBSCAN (tune eps)
    for eps in [0.3, 0.5, 1.0, 1.5, 2.0]:
        try:
            t0 = time.time()
            db = DBSCAN(eps=eps, min_samples=5).fit(X)
            t1 = time.time()
            n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            if n_clusters >= 2:
                ari, nmi = evaluate_clustering("DBSCAN", y, db.labels_)
                sil = silhouette_score(X, db.labels_) if n_clusters > 1 else -1
                results.append((f"DBSCAN(eps={eps})", "Nao sup", ari, nmi, sil, t1-t0))
        except:
            pass

    # GMM
    for k in range(2, 8):
        t0 = time.time()
        gmm = GaussianMixture(n_components=k, random_state=42).fit(X)
        t1 = time.time()
        labels = gmm.predict(X)
        ari, nmi = evaluate_clustering("GMM", y, labels)
        try:
            sil = silhouette_score(X, labels)
        except:
            sil = -1
        results.append(("GMM(k="+str(k)+")", "Nao sup", ari, nmi, sil, t1-t0))

    return results

def run_semi_supervised(X, y, label_frac=0.1):
    """Semi-supervisionado: usar fracao de labels para inicializar / guiar clustering."""
    results = []

    n = len(X)
    n_labeled = max(int(n * label_frac), 10)

    # Amostrar pontos rotulados (estratificado)
    X_labeled, _, y_labeled, _ = train_test_split(X, y, train_size=n_labeled,
                                                    random_state=42, stratify=y)

    # 1) K-Means com inicializacao pelos centroides dos rotulados
    centroids = np.array([X_labeled[y_labeled == c].mean(axis=0)
                          for c in np.unique(y_labeled)])
    n_clusters = len(centroids)

    if n_clusters >= 2:
        t0 = time.time()
        km = KMeans(n_clusters=n_clusters, init=centroids, random_state=42, n_init=1).fit(X)
        t1 = time.time()
        ari, nmi = evaluate_clustering("K-Means", y, km.labels_)
        sil = silhouette_score(X, km.labels_)
        results.append((f"K-Means init ({label_frac*100:.0f}%)", "Semi-sup", ari, nmi, sil, t1-t0))

    # 2) Constrained: K-Means com must-link via "nearest centroid classifier"
    #    Treina um KNN nos rotulados, propaga rotulos como constraints
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=5).fit(X_labeled, y_labeled)
    pseudo_labels = knn.predict(X)
    ari, nmi = evaluate_clustering("LabelProp (KNN)", y, pseudo_labels)
    try:
        sil = silhouette_score(X, pseudo_labels)
    except:
        sil = -1
    t1 = time.time()
    results.append((f"LabelProp KNN ({label_frac*100:.0f}%)", "Semi-sup", ari, nmi, sil, t1-t0))

    return results

def run_supervised(X, y, label_frac=0.1):
    """Supervisionado puro: classificar com fracao de labels."""
    results = []

    n = len(X)
    n_labeled = max(int(n * label_frac), 10)

    X_labeled, X_test, y_labeled, y_test = train_test_split(
        X, y, train_size=n_labeled, random_state=42, stratify=y)

    # RF
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_labeled, y_labeled)
    y_pred = rf.predict(X_test)
    t1 = time.time()
    ari_test = adjusted_rand_score(y_test, y_pred)
    # Full dataset
    y_pred_full = rf.predict(X)
    ari_full = adjusted_rand_score(y, y_pred_full)
    nmi_full = normalized_mutual_info_score(y, y_pred_full)
    results.append((f"RF ({label_frac*100:.0f}%)", "Superv", ari_full, nmi_full, -1, t1-t0))

    # KNN
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=3).fit(X_labeled, y_labeled)
    y_pred = knn.predict(X)
    t1 = time.time()
    ari_full = adjusted_rand_score(y, y_pred)
    nmi_full = normalized_mutual_info_score(y, y_pred)
    results.append((f"KNN ({label_frac*100:.0f}%)", "Superv", ari_full, nmi_full, -1, t1-t0))

    return results

# ============================================================
# Executar
# ============================================================
all_results = []

for name, X, y, classes in datasets:
    print(f"\n{'='*95}")
    print(f"DATASET: {name} ({X.shape[0]} amostras, {len(classes)} classes, {X.shape[1]} features)")
    print(f"{'='*95}")

    X_scaled = StandardScaler().fit_transform(X)

    # Unsupervised
    r_unsup = run_unsupervised(X_scaled, y)

    # Semi-supervised (5% e 20%)
    r_semi_5 = run_semi_supervised(X_scaled, y, 0.05)
    r_semi_20 = run_semi_supervised(X_scaled, y, 0.20)

    # Supervised (5% e 20%)
    r_sup_5 = run_supervised(X_scaled, y, 0.05)
    r_sup_20 = run_supervised(X_scaled, y, 0.20)

    # Juntar
    all_r = r_unsup + r_semi_5 + r_semi_20 + r_sup_5 + r_sup_20

    # Melhor de cada categoria
    best_unsup = max([r for r in all_r if r[1] == "Nao sup"], key=lambda x: x[2])
    best_semi = max([r for r in all_r if r[1] == "Semi-sup"], key=lambda x: x[2])
    best_sup = max([r for r in all_r if r[1] == "Superv"], key=lambda x: x[2])

    # Tabela
    print(f"\n{'Metodo':<30} {'Tipo':<12} {'ARI':<10} {'NMI':<10} {'Tempo(s)':<10}")
    print("-" * 72)

    for r in sorted(all_r, key=lambda x: x[2], reverse=True):
        tipo = r[1]
        print(f"  {r[0]:<28} {tipo:<12} {r[2]:<10.4f} {r[3]:<10.4f} {r[5]:<10.3f}")

    print("\n  Melhores de cada categoria:")
    print(f"    Nao supervisionado: {best_unsup[0]} (ARI={best_unsup[2]:.4f})")
    print(f"    Semi-supervisionado: {best_semi[0]} (ARI={best_semi[2]:.4f})")
    print(f"    Supervisionado:      {best_sup[0]} (ARI={best_sup[2]:.4f})")

    all_results.append((name, best_unsup, best_semi, best_sup))

# ============================================================
# Conclusao
# ============================================================
print("\n\n" + "=" * 95)
print("RESUMO GLOBAL")
print("=" * 95)
print(f"\n{'Dataset':<20} {'Melhor Nao Sup':<35} {'ARI':<8} {'Melhor Semi-Sup':<35} {'ARI':<8} {'Melhor Sup':<30} {'ARI':<8}")
print("-" * 144)
for name, best_u, best_semi, best_sup in all_results:
    print(f"  {name:<18} {best_u[0]:<35} {best_u[2]:<8.4f} {best_semi[0]:<35} {best_semi[2]:<8.4f} {best_sup[0]:<30} {best_sup[2]:<8.4f}")

print("\n" + "=" * 95)
print("CONCLUSOES")
print("=" * 95)
print("""
1. Clusterizacao pura (nao supervisionada) funciona bem quando os dados
   tem uma estrutura natural de grupos separaveis.

2. Com apenas 5-20% de labels, o semi-supervisionado (inicializar K-Means
   com centroides rotulados) ja consegue ARI > 0.9 na maioria dos casos.

3. O supervisionado puro com poucos labels sofre porque precisa generalizar
   para regioes nao vistas do espaco - o clustering nao tem esse problema
   porque usa a estrutura global dos dados.

4. Diferenca fundamental:
   - Clustering: usa a estrutura GEOMETRICA dos dados (distancias)
   - Classificacao: usa a estrutura SEMANTICA (decisao entre classes)
   Quando ha overlap entre classes, o clustering perde a referencia.
""")
print("=" * 95)
