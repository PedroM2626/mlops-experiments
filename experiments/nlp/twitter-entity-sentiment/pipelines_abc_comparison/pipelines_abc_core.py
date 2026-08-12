# -*- coding: utf-8 -*-
"""
Módulo central do estudo comparativo A vs B vs C (Senti-Pred-remake2).

Reimplementa fielmente as TRÊS pipelines de análise de sentimento do Twitter
sobre o mesmo dataset (twitter_training / twitter_validation), com suporte a
ablações (what-ifs) em: n-grams, tamanho de vocabulário, min_df, sublinear_tf,
pré-processamento e modelo.

Pipelines:
  A — senti-pred_pipeline.ipynb        (pré-processamento agressivo)
  B — twitter-sentiment-analysis.ipynb (pré-processamento conservador)
  C — Senti-Pred-remake2               (pré-processamento remake2 = Data-Centric)

Referências:
  - README experiments/nlp §5.3–5.4 (A vs B)
  - README experiments/senti-pred-variations (remake2 = recorde 97.80%)
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
import os

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------- recursos ---
for _res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(_res, quiet=True)
    except Exception:
        pass

_LEMMATIZER = WordNetLemmatizer()
_EN_STOP = set(stopwords.words('english'))
_EN_STOP.discard('not')
_EN_STOP.discard('no')

RAW_DIR = Path(__file__).parent.parent / 'senti-pred-variations' / 'senti-pred-exp1' / 'data' / 'raw'
COLUMNS = ['id', 'entity', 'sentiment', 'text']
VALID_SENTIMENTS = ['Positive', 'Negative', 'Neutral', 'Irrelevant']

N_CORES = 14


# ====================================================== pré-processamento ====
#
# Cada pipeline é reimplementada como uma função com toggles explícitos para
# permitir ablações controladas (o padrão "A" sempre reproduz o original).
# ----------------------------------------------------------------------------

def clean_a(text: str, *, keep_hashtags: bool = False, keep_punct: bool = False,
            keep_digits: bool = False) -> str:
    """Pipeline A — agressiva (senti-pred_pipeline.ipynb).

    Default: remove URLs, menções, hashtags inteiras, pontuação e dígitos.
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    if not keep_hashtags:
        text = re.sub(r'@\w+|#\w+', '', text)
    else:
        text = re.sub(r'@\w+', '', text)
    if not keep_punct:
        text = re.sub(r'[^\w\s]', '', text)
    if not keep_digits:
        text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_b(text: str, *, drop_hashtags: bool = False, drop_punct: bool = False,
            drop_digits: bool = False) -> str:
    """Pipeline B — conservadora (twitter-sentiment-analysis.ipynb).

    Default: mantém conteúdo de hashtags (-> palavra), pontuação !?.,'"- e números.
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    if not drop_hashtags:
        text = re.sub(r'#(\w+)', r'\1', text)
    else:
        text = re.sub(r'#\w+', '', text)
    if not drop_punct:
        text = re.sub(r'[^a-z0-9\s!?.,\'\-]', '', text)
    else:
        text = re.sub(r'[^a-z0-9\s]', '', text)
    if drop_digits:
        text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _expand_contractions(text: str) -> str:
    contractions = [
        (r"can't", "cannot"), (r"n't", " not"), (r"'re", " are"),
        (r"'s", " is"), (r"'d", " would"), (r"'ll", " will"),
        (r"'t", " not"), (r"'ve", " have"), (r"'m", " am"),
    ]
    for pat, repl in contractions:
        text = re.sub(pat, repl, text)
    return text


def clean_c(text: str, *, remove_stopwords: bool = True, lemmatize: bool = True,
            expand_contractions: bool = True, keep_question_mark: bool = True,
            keep_hashtag_word: bool = True) -> str:
    """Pipeline C — remake2 (Senti-Pred-remake2/src/data/preprocess.py).

    Default: URLs/menções removidas, '#' removido mantendo palavra, contrações
    expandidas, pontuação exceto '!' e '?' removida, stopwords removidas
    (preservando 'not'/'no') e WordNet lemmatização.
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    if keep_hashtag_word:
        # remove a menção inteira e o símbolo '#', preservando o conteúdo do hashtag
        text = re.sub(r'@\w+', '', text)
        text = text.replace('#', '')
    else:
        text = re.sub(r'@\w+|#\w+', '', text)
    if expand_contractions:
        text = _expand_contractions(text)
    if keep_question_mark:
        text = re.sub(r'[^a-z\s\!\?]', '', text)
    else:
        text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _EN_STOP]
    if lemmatize:
        tokens = [_LEMMATIZER.lemmatize(w) for w in tokens]
    return ' '.join(tokens)


CLEANERS = {
    'A': clean_a,
    'B': clean_b,
    'C': clean_c,
}


# ================================================================ dados =======

def load_data(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega treino e validação crus (sem cabeçalho) e remove nulos."""
    train = pd.read_csv(raw_dir / 'twitter_training.csv', names=COLUMNS, header=None)
    val = pd.read_csv(raw_dir / 'twitter_validation.csv', names=COLUMNS, header=None)
    train = train.dropna(subset=['text', 'sentiment'])
    val = val.dropna(subset=['text', 'sentiment'])
    return train, val


def apply_cleaner(df: pd.DataFrame, cleaner: Callable, **kwargs) -> pd.DataFrame:
    """Aplica um cleaner ao DataFrame e retorna só linhas não-vazias/valid sentimentos."""
    out = df.copy()
    series = out['text'].astype(str)
    if cleaner.__name__ == 'clean_c':
        cleaned = Parallel(n_jobs=N_CORES)(delayed(cleaner)(t, **kwargs) for t in series)
    else:
        cleaned = [cleaner(t, **kwargs) for t in series]
    out['clean'] = [c if isinstance(c, str) else '' for c in cleaned]
    out = out[out['clean'].str.len() > 0]
    out = out[out['sentiment'].isin(VALID_SENTIMENTS)]
    return out[['clean', 'sentiment']].reset_index(drop=True)


# ======================================================= vetorizadores ========

VEC_CANONICAL: Dict[str, Dict] = {
    'A': dict(max_features=70000, min_df=2, ngram_range=(1, 2),
              sublinear_tf=True, strip_accents='unicode'),
    'B': dict(max_features=70000, min_df=2, ngram_range=(1, 2),
              sublinear_tf=True, strip_accents='unicode'),
    'C': dict(max_features=100000, ngram_range=(1, 4), sublinear_tf=True,
              strip_accents='unicode', min_df=2, analyzer='word',
              token_pattern=r'\w{1,}'),
}

# ================================================================ modelos ====

def make_linear_svc_c19() -> LinearSVC:
    return LinearSVC(C=19.0, max_iter=20000, random_state=42)


def make_voting_c() -> VotingClassifier:
    svc = LinearSVC(C=0.5, max_iter=3000, dual='auto', random_state=42,
                    tol=1e-5, class_weight='balanced')
    lr = LogisticRegression(C=10, max_iter=1000, solver='lbfgs',
                            multi_class='multinomial', random_state=42,
                            class_weight='balanced')
    return VotingClassifier(estimators=[('svc', svc), ('lr', lr)], voting='hard')


MODELS_A: Dict[str, Callable[[], object]] = {
    'LogisticRegression': lambda: LogisticRegression(max_iter=2000, random_state=42),
    'MultinomialNB': lambda: MultinomialNB(),
    'LinearSVC_C1': lambda: LinearSVC(C=1.0, max_iter=20000, random_state=42),
    'LinearSVC_C10': lambda: LinearSVC(C=10.0, max_iter=20000, random_state=42),
    'LinearSVC_C19': lambda: LinearSVC(C=19.0, max_iter=20000, random_state=42),
    'ExtraTrees': lambda: ExtraTreesClassifier(n_estimators=100, random_state=42,
                                               n_jobs=-1),
}

MODELS_B: Dict[str, Callable[[], object]] = {
    'LogisticRegression_C11': lambda: LogisticRegression(
        max_iter=1000, C=11.0, solver='lbfgs', multi_class='multinomial', random_state=42),
    'ExtraTrees': lambda: ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'LinearSVC_C19': lambda: LinearSVC(C=19.0, max_iter=1000, random_state=42),
    'PassiveAggressive': lambda: PassiveAggressiveClassifier(C=1.0, max_iter=1000,
                                                             random_state=42),
    'KNN_cosine': lambda: KNeighborsClassifier(n_neighbors=7, metric='cosine', n_jobs=-1),
    'RidgeClassifier': lambda: RidgeClassifier(alpha=1.0),
    'SGD_modhuber': lambda: SGDClassifier(loss='modified_huber', max_iter=1000,
                                           random_state=42, n_jobs=-1),
}

MODELS_C: Dict[str, Callable[[], object]] = {
    'LinearSVC_C0.5': lambda: LinearSVC(C=0.5, max_iter=3000, dual='auto',
                                        random_state=42, tol=1e-5,
                                        class_weight='balanced'),
    'LogisticRegression_C10': lambda: LogisticRegression(
        C=10, max_iter=1000, solver='lbfgs', multi_class='multinomial',
        random_state=42, class_weight='balanced'),
    'Voting_svc_lr': lambda: make_voting_c(),
}


# ============================================================== avaliação ====

def evaluate(pipe: str, vec_params: Dict, model, tr: pd.DataFrame, va: pd.DataFrame,
             model_name: str, extra_meta: Optional[Dict] = None) -> Dict:
    """Treina um modelo com o vetorizador dado e retorna um dicionário de métricas."""
    t0 = time.time()
    X_tr = tr['clean'].tolist()
    y_tr = tr['sentiment'].values
    X_va = va['clean'].tolist()
    y_va = va['sentiment'].values

    pipeline_obj = Pipeline([
        ('vectorizer', TfidfVectorizer(**vec_params)),
        ('classifier', model)
    ])

    with mlflow.start_run(nested=True):
        pipeline_obj.fit(X_tr, y_tr)
        y_pred = pipeline_obj.predict(X_va)
        fit_time = time.time() - t0

        n_features = len(pipeline_obj.named_steps['vectorizer'].vocabulary_)

        acc = accuracy_score(y_va, y_pred)
        f1_macro = f1_score(y_va, y_pred, average='macro')
        f1_weighted = f1_score(y_va, y_pred, average='weighted')
        report = classification_report(y_va, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_va, y_pred, labels=VALID_SENTIMENTS)

        # MLflow parameters
        mlflow.log_param("pipeline", pipe)
        mlflow.log_param("model_name", model_name)
        for k, v in vec_params.items():
            mlflow.log_param(f"vec_{k}", v)
        if extra_meta:
            for k, v in extra_meta.items():
                mlflow.log_param(f"meta_{k}", v)

        # MLflow metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_metric("fit_time_s", fit_time)
        mlflow.log_metric("n_features", n_features)

        # Log confusion matrix plot
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=VALID_SENTIMENTS, yticklabels=VALID_SENTIMENTS)
        plt.title(f"Confusion Matrix: {model_name} ({pipe})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_path = f"cm_{pipe}_{model_name}.png"
        
        # Ensure filename is somewhat clean from weird characters
        cm_path = cm_path.replace("=", "").replace("'", "")
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        if os.path.exists(cm_path):
            os.remove(cm_path)

        # Log the model
        mlflow.sklearn.log_model(pipeline_obj, "model")

    row = {
        'pipeline': pipe,
        'model': model_name,
        'n_features': n_features,
        'fit_time_s': round(fit_time, 2),
        'accuracy': round(float(acc), 6),
        'f1_macro': round(float(f1_macro), 6),
        'f1_weighted': round(float(f1_weighted), 6),
        'y_true': y_va,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'report': report,
    }
    if extra_meta:
        row.update(extra_meta)
    return row


def to_table(dframe: pd.DataFrame, subset: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Converte df com y_true/y_pred em tabelas; mantém colunas de interesse."""
    if subset is not None:
        dframe = dframe[subset]
    return dframe.reset_index(drop=True)