"""
🏗️ Ensemble Pyramid — 4 Camadas de Ensembles sobre Ensembles
Dataset: Twitter Entity Sentiment Analysis (processed_train / processed_validation)

Arquitetura:
┌──────────────────────────────────────────────────────────────────┐
│  CAMADA 4 — META-ENSEMBLE FINAL                                  │
│      C4A: Voting Soft  (C3A + C3B + C3C)                         │
│      C4B: Stacking     (C3A + C3B + C3C → LR)                    │
│      C4C: Voting Hard  (C3A + C3B + C3C + C2C)                   │
├────────────────────┬────────────────────┬────────────────────────┤
│  C3A               │  C3B               │  C3C                   │
│  Stacking          │  Bagging           │  Voting Hard           │
│  (C2A+C2B → LR)    │  (Stacking clone)  │  (C2D + C2E)           │
├──────────┬─────────┼──────────┬─────────┼──────────┬─────────────┤
│ C2A      │ C2B     │ C2C      │ C2D     │ C2E      │             │
│ Bagging  │ Voting  │ Stacking │ Voting  │ Bagging  │             │
│ (LR,SVC) │(NB,CNB) │ (RF,ET)  │(SVC,LR) │ (NB,CNB) │             │
├──────────┴─────────┴──────────┴─────────┴──────────┴─────────────┤
│  CAMADA 1 — BASE LEARNERS (todos sparse-compatíveis)             │
│  LR | LinearSVC | MultinomialNB | ComplementNB | RF | ET | Ridge │
└──────────────────────────────────────────────────────────────────┘

⚠️  MEMÓRIA:
    TF-IDF 70k features em SPARSE ocupa ~50MB.
    .toarray() ocuparia ~39GB — inviável! Tudo mantido sparse.
    LDA e KNN foram removidos: não aceitam matrizes esparsas.

⚙️  LinearSVC — duas variantes:
    make_svc_hard() → LinearSVC puro, idêntico ao notebook (para voting="hard")
    make_svc_soft() → CalibratedClassifierCV (para voting="soft" e Stacking,
                       pois precisam de predict_proba)
"""

import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    BaggingClassifier, VotingClassifier, StackingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

SEED     = 42
CV_FOLDS = 3
np.random.seed(SEED)

print("=" * 65)
print("🏗️  ENSEMBLE PYRAMID — 4 Camadas de Ensembles sobre Ensembles")
print("=" * 65)

# ─── 1. Dados ──────────────────────────────────────────────────────────────────
print("\n📂 Carregando dados...")
train_df = pd.read_csv("experiments/senti-pred-variations/logistic-senti-pred/data/processed/processed_train.csv")
val_df   = pd.read_csv("experiments/senti-pred-variations/logistic-senti-pred/data/processed/processed_validation.csv")

TEXT_COL   = "text_clean"
TARGET_COL = "sentiment"

le = LabelEncoder()
le.fit(train_df[TARGET_COL])
y_train = le.transform(train_df[TARGET_COL])
y_val   = le.transform(val_df[TARGET_COL])
CLASSES = list(le.classes_)

print(f"Treino    : {train_df.shape[0]:,} amostras")
print(f"Validação : {val_df.shape[0]:,} amostras")
print(f"Classes   : {CLASSES}")

# ─── 2. TF-IDF — mantém SPARSE ────────────────────────────────────────────────
print("\n🔤 Gerando features TF-IDF (sparse)...")
tfidf = TfidfVectorizer(
    max_features  = 70_000,
    ngram_range   = (1, 2),
    sublinear_tf  = True,
    min_df        = 2,
    strip_accents = "unicode",
)
X_train = tfidf.fit_transform(train_df[TEXT_COL].fillna(""))
X_val   = tfidf.transform(val_df[TEXT_COL].fillna(""))

print(f"Shape    : {X_train.shape}")
print(f"Formato  : {type(X_train).__name__} (sparse ✔)")
print(f"RAM ~    : {X_train.data.nbytes / 1e6:.1f} MB")

# ─── Utilitário de avaliação ───────────────────────────────────────────────────
results = []

def evaluate(name, model, X, y, layer):
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average="weighted")
    bar   = "█" * int(f1 * 35)
    print(f"    ✔ F1={f1:.4f}  Acc={acc:.4f}  {bar}")
    results.append({"layer": layer, "name": name, "acc": acc, "f1": f1})

# ─── Fábricas ─────────────────────────────────────────────────────────────────
def make_lr(C=11.0):
    return LogisticRegression(C=C, max_iter=1000, solver="lbfgs",
                               multi_class="auto", random_state=SEED, n_jobs=-1)

# ⚙️ Para voting="hard" e contextos que NÃO precisam de predict_proba:
#    LinearSVC puro — idêntico ao notebook, sem overhead de calibração
def make_svc_hard():
    return LinearSVC(C=19.0, max_iter=1000, random_state=SEED)

# ⚙️ Para voting="soft", Stacking e Bagging que precisam de predict_proba:
#    CalibratedClassifierCV envolve o LinearSVC e aprende uma sigmoid por cima
def make_svc_soft():
    return CalibratedClassifierCV(
        LinearSVC(C=19.0, max_iter=1000, random_state=SEED),
        cv=3, method="sigmoid"
    )

def make_ridge_hard():
    return RidgeClassifier()

def make_ridge_soft():
    return CalibratedClassifierCV(RidgeClassifier(alpha=1.0), cv=3, method="sigmoid")

def make_nb():
    return MultinomialNB(alpha=0.1)

def make_cnb():
    return ComplementNB(alpha=0.1)

def make_rf(n=100):
    return RandomForestClassifier(n_estimators=n, random_state=SEED, n_jobs=-1)

def make_et(n=100):
    return ExtraTreesClassifier(n_estimators=n, random_state=SEED, n_jobs=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# CAMADA 1 — BASE LEARNERS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("🧱 CAMADA 1 — Base Learners")
print("═" * 65)

t0 = time.time()

base_learners = {
    "LogisticRegression" : make_lr(),
    "LinearSVC"          : make_svc_hard(),   # puro, sem calibração
    "MultinomialNB"      : make_nb(),
    "ComplementNB"       : make_cnb(),
    "Ridge"              : make_ridge_hard(),  # puro
    "RandomForest"       : make_rf(100),
    "ExtraTrees"         : make_et(100),
}

trained_base = {}
for name, model in base_learners.items():
    print(f"\n  [{name}]")
    t = time.time()
    model.fit(X_train, y_train)
    trained_base[name] = model
    evaluate(name, model, X_val, y_val, layer=1)
    print(f"    ⏱  {time.time()-t:.1f}s")

print(f"\n⏱️  Camada 1 total: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════════
# CAMADA 2 — ENSEMBLES DOS BASE LEARNERS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("⚙️  CAMADA 2 — Ensembles dos Base Learners")
print("═" * 65)

# C2A — Bagging(LR) + Bagging(SVC_soft) → Voting Soft
# Bagging precisa de predict_proba para voting soft → make_svc_soft()
print("\n  [C2A] Bagging(LR) + Bagging(SVC) → Voting Soft")
c2a_bag_lr  = BaggingClassifier(estimator=make_lr(),       n_estimators=10,
                                  max_samples=0.8, random_state=SEED,   n_jobs=-1)
c2a_bag_svc = BaggingClassifier(estimator=make_svc_soft(), n_estimators=10,
                                  max_samples=0.8, random_state=SEED+1, n_jobs=-1)
c2a_bag_lr.fit(X_train, y_train)
c2a_bag_svc.fit(X_train, y_train)
c2a = VotingClassifier(
    estimators=[("bag_lr", c2a_bag_lr), ("bag_svc", c2a_bag_svc)],
    voting="soft", n_jobs=-1
)
c2a.fit(X_train, y_train)
evaluate("C2A: Bagging(LR+SVC)->Voting Soft", c2a, X_val, y_val, layer=2)

# C2B — Voting Soft (NB + CNB + Ridge_soft)
# Ridge não tem predict_proba → make_ridge_soft()
print("\n  [C2B] Voting Soft (NB + ComplementNB + Ridge)")
c2b = VotingClassifier(
    estimators=[("nb", make_nb()), ("cnb", make_cnb()), ("rdg", make_ridge_soft())],
    voting="soft", n_jobs=-1
)
c2b.fit(X_train, y_train)
evaluate("C2B: Voting Soft(NB+CNB+Ridge)", c2b, X_val, y_val, layer=2)

# C2C — Stacking (RF + ET → LR meta)
# Stacking usa predict_proba internamente → RF e ET já têm nativamente
print("\n  [C2C] Stacking (RF + ET → LR meta)")
c2c = StackingClassifier(
    estimators=[("rf", make_rf(80)), ("et", make_et(80))],
    final_estimator=make_lr(),
    cv=CV_FOLDS, passthrough=False, n_jobs=-1
)
c2c.fit(X_train, y_train)
evaluate("C2C: Stacking(RF+ET->LR)", c2c, X_val, y_val, layer=2)

# C2D — Voting Hard (SVC_hard + LR + Ridge_hard)
# Hard voting usa apenas predict() → LinearSVC e Ridge puros
print("\n  [C2D] Voting Hard (SVC + LR + Ridge)")
c2d = VotingClassifier(
    estimators=[("svc", make_svc_hard()), ("lr", make_lr()), ("rdg", make_ridge_hard())],
    voting="hard", n_jobs=-1
)
c2d.fit(X_train, y_train)
evaluate("C2D: Voting Hard(SVC+LR+Ridge)", c2d, X_val, y_val, layer=2)

# C2E — Bagging(NB) + Bagging(CNB) → Voting Soft
# NB e CNB já têm predict_proba nativamente
print("\n  [C2E] Bagging(NB) + Bagging(CNB) → Voting Soft")
c2e_bag_nb  = BaggingClassifier(estimator=make_nb(),  n_estimators=15,
                                  max_samples=0.8, random_state=SEED,   n_jobs=-1)
c2e_bag_cnb = BaggingClassifier(estimator=make_cnb(), n_estimators=15,
                                  max_samples=0.8, random_state=SEED+2, n_jobs=-1)
c2e_bag_nb.fit(X_train, y_train)
c2e_bag_cnb.fit(X_train, y_train)
c2e = VotingClassifier(
    estimators=[("bag_nb", c2e_bag_nb), ("bag_cnb", c2e_bag_cnb)],
    voting="soft", n_jobs=-1
)
c2e.fit(X_train, y_train)
evaluate("C2E: Bagging(NB+CNB)->Voting Soft", c2e, X_val, y_val, layer=2)

print("\n⏱️  Camada 2 concluída.")

# ═══════════════════════════════════════════════════════════════════════════════
# CAMADA 3 — ENSEMBLES DOS ENSEMBLES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("🔗 CAMADA 3 — Ensembles dos Ensembles")
print("═" * 65)

# C3A — Stacking (C2A + C2B → LR)
# C2A e C2B são VotingClassifier soft → já têm predict_proba
print("\n  [C3A] Stacking (C2A + C2B → LR)")
c3a = StackingClassifier(
    estimators=[("c2a", c2a), ("c2b", c2b)],
    final_estimator=make_lr(),
    cv=CV_FOLDS, passthrough=True, n_jobs=-1
)
c3a.fit(X_train, y_train)
evaluate("C3A: Stacking(C2A+C2B->LR)", c3a, X_val, y_val, layer=3)

# C3B — Bagging sobre Stacking clonável (RF+ET→LR)
# RF e ET têm predict_proba → ok para Stacking interno
print("\n  [C3B] Bagging sobre Stacking (RF+ET→LR)")
c3b = BaggingClassifier(
    estimator=StackingClassifier(
        estimators=[("rf", make_rf(60)), ("et", make_et(60))],
        final_estimator=make_lr(),
        cv=CV_FOLDS, n_jobs=-1
    ),
    n_estimators=4,
    max_samples=0.8,
    random_state=SEED,
    n_jobs=-1,
)
c3b.fit(X_train, y_train)
evaluate("C3B: Bagging(Stacking)", c3b, X_val, y_val, layer=3)

# C3C — Voting Hard (C2D + C2E)
# C2D é hard, C2E é soft — ambos têm predict() → voting hard funciona
print("\n  [C3C] Voting Hard (C2D + C2E)")
c3c = VotingClassifier(
    estimators=[("c2d", c2d), ("c2e", c2e)],
    voting="hard", n_jobs=-1
)
c3c.fit(X_train, y_train)
evaluate("C3C: Voting Hard(C2D+C2E)", c3c, X_val, y_val, layer=3)

print("\n⏱️  Camada 3 concluída.")

# ═══════════════════════════════════════════════════════════════════════════════
# CAMADA 4 — META-ENSEMBLE FINAL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("🏆 CAMADA 4 — Meta-Ensemble Final")
print("═" * 65)

# C4A — Voting Soft (C3A + C3B + C3C)
# C3A=Stacking(predict_proba✔), C3B=Bagging(predict_proba✔), C3C=VotingHard(predict_proba✔)
print("\n  [C4A] Voting Soft (C3A + C3B + C3C)")
c4a = VotingClassifier(
    estimators=[("c3a", c3a), ("c3b", c3b), ("c3c", c3c)],
    voting="soft", n_jobs=-1
)
c4a.fit(X_train, y_train)
evaluate("C4A: Voting Soft(C3A+C3B+C3C)", c4a, X_val, y_val, layer=4)

# C4B — Stacking Final (C3A + C3B + C3C → LR)
print("\n  [C4B] Stacking Final (C3A + C3B + C3C → LR)")
c4b = StackingClassifier(
    estimators=[("c3a", c3a), ("c3b", c3b), ("c3c", c3c)],
    final_estimator=LogisticRegression(C=0.5, max_iter=1000, random_state=SEED),
    cv=CV_FOLDS, passthrough=False, n_jobs=-1
)
c4b.fit(X_train, y_train)
evaluate("C4B: Stacking Final(->LR)", c4b, X_val, y_val, layer=4)

# C4C — Voting Hard (C3A + C3B + C3C + C2C)
print("\n  [C4C] Voting Hard (C3A + C3B + C3C + C2C)")
c4c = VotingClassifier(
    estimators=[("c3a", c3a), ("c3b", c3b), ("c3c", c3c), ("c2c", c2c)],
    voting="hard", n_jobs=-1
)
c4c.fit(X_train, y_train)
evaluate("C4C: Voting Hard(C3s+C2C)", c4c, X_val, y_val, layer=4)

print("\n⏱️  Camada 4 concluída.")

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTADOS FINAIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("📊 COMPARAÇÃO COMPLETA — Todas as Camadas")
print("═" * 65)

df_results = pd.DataFrame(results).sort_values(["layer", "f1"], ascending=[True, False])
layer_names = {
    1: "Base Learners",
    2: "Ensembles L1",
    3: "Ensembles de Ensembles",
    4: "Meta-Ensemble Final"
}

for layer in [1, 2, 3, 4]:
    print(f"\n── Camada {layer}: {layer_names[layer]} ──")
    subset = df_results[df_results["layer"] == layer]
    for _, row in subset.iterrows():
        bar = "█" * int(row["f1"] * 35)
        print(f"  {row['name']:<45} F1={row['f1']:.4f}  {bar}")

best = df_results.loc[df_results["f1"].idxmax()]
print(f"\n🏆 MELHOR MODELO GERAL : {best['name']}")
print(f"   Camada {int(best['layer'])} | Acc={best['acc']:.4f} | F1={best['f1']:.4f}")

# ─── Classification report do melhor ──────────────────────────────────────────
model_map = {
    "C4A: Voting Soft(C3A+C3B+C3C)"     : c4a,
    "C4B: Stacking Final(->LR)"          : c4b,
    "C4C: Voting Hard(C3s+C2C)"          : c4c,
    "C3A: Stacking(C2A+C2B->LR)"         : c3a,
    "C3B: Bagging(Stacking)"              : c3b,
    "C3C: Voting Hard(C2D+C2E)"           : c3c,
    "C2A: Bagging(LR+SVC)->Voting Soft"   : c2a,
    "C2B: Voting Soft(NB+CNB+Ridge)"      : c2b,
    "C2C: Stacking(RF+ET->LR)"            : c2c,
    "C2D: Voting Hard(SVC+LR+Ridge)"      : c2d,
    "C2E: Bagging(NB+CNB)->Voting Soft"   : c2e,
    **trained_base,
}

best_obj = model_map.get(best["name"])
if best_obj:
    print(f"\n=== Classification Report — {best['name']} ===")
    print(classification_report(y_val, best_obj.predict(X_val), target_names=CLASSES))

# ─── Análise de ganho por camada ──────────────────────────────────────────────
print("═" * 65)
print("🔬 ANÁLISE DA PIRÂMIDE — Ganho por Camada")
print("═" * 65)

layer_bests = [df_results[df_results["layer"] == l]["f1"].max() for l in [1, 2, 3, 4]]
layer_means = [df_results[df_results["layer"] == l]["f1"].mean() for l in [1, 2, 3, 4]]

for i, (lb, lm) in enumerate(zip(layer_bests, layer_means), start=1):
    print(f"  Camada {i} → Melhor F1: {lb:.4f} | Média F1: {lm:.4f}")

print(f"\n  Ganho C1→C2 : {(layer_bests[1]-layer_bests[0])*100:+.2f}pp")
print(f"  Ganho C2→C3 : {(layer_bests[2]-layer_bests[1])*100:+.2f}pp")
print(f"  Ganho C3→C4 : {(layer_bests[3]-layer_bests[2])*100:+.2f}pp")
print(f"  Ganho Total : {(layer_bests[3]-layer_bests[0])*100:+.2f}pp")

# ─── Salvar ───────────────────────────────────────────────────────────────────
df_results.to_csv("ensemble_pyramid_results.csv", index=False)
print("\nResultados salvos: ensemble_pyramid_results.csv")

if best_obj:
    joblib.dump({"model": best_obj, "tfidf": tfidf, "encoder": le},
                "ensemble_pyramid_best.pkl")
    print("Melhor modelo salvo: ensemble_pyramid_best.pkl")

print("\n✅ Ensemble Pyramid concluído!")