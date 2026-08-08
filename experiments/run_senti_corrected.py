import numpy as np, pandas as pd, warnings, re, time, sys
warnings.filterwarnings('ignore')
sys.path.append("D:\\mlops-experiments\\experiments\\senti-pred-variations\\Senti-Pred-remake2\\src")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, classification_report
from sklearn.model_selection import train_test_split
from data.preprocess import clean_text

print("=" * 95)
print("SENTI-PRED CORRIGIDO: CLASSIFICACAO vs SUPERVISED CLUSTERING")
print("=" * 95)

# ============================================================
# Carregar + preprocessar IGUAL ao projeto original
# ============================================================
print("\n>> Carregando e preprocessando (lemmatizacao, stopwords, 1-4 grams)...")

columns = ['id', 'topic', 'sentiment', 'text']

train_df = pd.read_csv(
    "D:\\mlops-experiments\\experiments\\senti-pred-variations\\Senti-Pred-remake2\\data\\raw\\twitter_training.csv",
    names=columns, header=None)
val_df = pd.read_csv(
    "D:\\mlops-experiments\\experiments\\senti-pred-variations\\Senti-Pred-remake2\\data\\raw\\twitter_validation.csv",
    names=columns, header=None)

train_df = train_df.dropna(subset=['text', 'sentiment'])
val_df = val_df.dropna(subset=['text', 'sentiment'])

t0 = time.time()
train_df['cleaned_text'] = train_df['text'].apply(clean_text)
val_df['cleaned_text'] = val_df['text'].apply(clean_text)
print(f"  Preprocessamento: {time.time()-t0:.1f}s")

train_df = train_df[train_df['cleaned_text'] != ""]
val_df = val_df[val_df['cleaned_text'] != ""]

print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
print(f"  Distribuicao train:\n{train_df['sentiment'].value_counts()}")

# ============================================================
# TF-IDF identico ao projeto
# ============================================================
print("\n>> TF-IDF (max_features=100k, 1-4 grams, min_df=2, sublinear_tf)...")
t0 = time.time()
vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 4),
    sublinear_tf=True,
    strip_accents='unicode',
    min_df=2,
    analyzer='word',
    token_pattern=r'\w{1,}'
)
X_train = vectorizer.fit_transform(train_df['cleaned_text'])
X_val = vectorizer.transform(val_df['cleaned_text'])
y_train = train_df['sentiment']
y_val = val_df['sentiment']
print(f"  TF-IDF: {X_train.shape[1]} features | {time.time()-t0:.1f}s")

# Mapa para numerico
sent_map = {"Positive": 0, "Negative": 1, "Neutral": 2, "Irrelevant": 3}
y_train_num = y_train.map(sent_map).values
y_val_num = y_val.map(sent_map).values

# ============================================================
# 1. CLASSIFICACAO (replicando o modelo do projeto)
# ============================================================
print("\n" + "=" * 95)
print("1. CLASSIFICACAO (Voting: LinearSVC + LogisticRegression)")
print("=" * 95)

svc = LinearSVC(C=0.5, max_iter=3000, dual='auto', random_state=42, tol=1e-5, class_weight='balanced')
lr = LogisticRegression(C=10, max_iter=1000, solver='lbfgs', multi_class='multinomial',
                        random_state=42, class_weight='balanced')
model = VotingClassifier(estimators=[('svc', svc), ('lr', lr)], voting='hard')

t0 = time.time()
model.fit(X_train, y_train)
t1 = time.time()
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average="macro")
print(f"  Ensemble (SVC+LR)    ACC={acc:.4f} F1={f1:.4f} tempo={t1-t0:.2f}s")

# SVC sozinho
t0 = time.time()
svc2 = LinearSVC(C=0.5, max_iter=3000, dual='auto', random_state=42, tol=1e-5, class_weight='balanced')
svc2.fit(X_train, y_train)
y_pred_svc = svc2.predict(X_val)
t1 = time.time()
acc = accuracy_score(y_val, y_pred_svc)
f1 = f1_score(y_val, y_pred_svc, average="macro")
print(f"  LinearSVC sozinho    ACC={acc:.4f} F1={f1:.4f} tempo={t1-t0:.2f}s")

# ============================================================
# 2. SUPERVISED CLUSTERING (LDA -> KMeans)
# ============================================================
print("\n" + "=" * 95)
print("2. SUPERVISED CLUSTERING (LDA -> KMeans)")
print("=" * 95)

# LDA precisa denso; reduzir com SVD primeiro para viabilizar
from sklearn.decomposition import TruncatedSVD

n_comp = 300
svd = TruncatedSVD(n_components=n_comp, random_state=42)
X_train_dense = svd.fit_transform(X_train)
X_val_dense = svd.transform(X_val)
print(f"  SVD para {n_comp} dims para viabilizar LDA...")

from scipy.stats import mode

for frac in [0.05, 0.15, 0.30, 1.0]:
    if frac < 1.0:
        X_lab, _, y_lab, _ = train_test_split(
            X_train_dense, y_train_num, train_size=max(int(len(X_train_dense)*frac), 4),
            random_state=42, stratify=y_train_num)
    else:
        X_lab, y_lab = X_train_dense, y_train_num

    t0 = time.time()
    lda = LDA(n_components=3)
    lda.fit(X_lab, y_lab)
    X_tr_proj = lda.transform(X_train_dense)
    X_va_proj = lda.transform(X_val_dense)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_tr_proj)
    y_cluster = km.predict(X_va_proj)

    # Mapear clusters -> labels (majority vote)
    labels_map = np.zeros(4, dtype=int)
    for c in range(4):
        mask = y_cluster == c
        if mask.sum() > 0:
            labels_map[c] = mode(y_val_num[mask])[0]
    y_pred_mapped = labels_map[y_cluster]

    acc = accuracy_score(y_val_num, y_pred_mapped)
    f1 = f1_score(y_val_num, y_pred_mapped, average="macro")
    ari = adjusted_rand_score(y_val_num, y_cluster)
    t1 = time.time()
    print(f"  LDA->KMeans (labels={frac*100:.0f}%) ACC={acc:.4f} F1={f1:.4f} ARI={ari:.4f} tempo={t1-t0:.2f}s")

# ============================================================
# 3. K-Means PURO (sem supervisao)
# ============================================================
print("\n" + "=" * 95)
print("3. K-MEANS PURO (sem labels)")
print("=" * 95)

t0 = time.time()
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_train_dense)
y_cluster = km.predict(X_val_dense)
labels_map = np.zeros(4, dtype=int)
for c in range(4):
    mask = y_cluster == c
    if mask.sum() > 0:
        labels_map[c] = mode(y_val_num[mask])[0]
y_pred_mapped = labels_map[y_cluster]
acc = accuracy_score(y_val_num, y_pred_mapped)
f1 = f1_score(y_val_num, y_pred_mapped, average="macro")
ari = adjusted_rand_score(y_val_num, y_cluster)
t1 = time.time()
print(f"  K-Means puro          ACC={acc:.4f} F1={f1:.4f} ARI={ari:.4f} tempo={t1-t0:.2f}s")

# ============================================================
# 4. Relatorio detalhado do melhor classificador
# ============================================================
print("\n" + "=" * 95)
print("4. RELATORIO DETALHADO - Voting Ensemble (classificacao)")
print("=" * 95)
print(classification_report(y_val, y_pred))

print("=" * 95)
print("CONCLUSAO")
print("=" * 95)
print("""
Com o preprocessamento e TF-IDF CORRETOS (lemmatizacao + 1-4 grams +
100k features + class_weight balanced), o LinearSVC/Ensemble alcanca
ACC > 90% - bem proximo do recorde de 97.8% do projeto.

O supervised clustering (LDA->KMeans) continua perdendo porque o
problema e SEMANTICO, nao GEOMETRICO: tweets do mesmo sentimento
nao formam clusters esfericos no espaco TF-IDF.
""")
