"""02_preprocessing.py

Pré-processamento modularizado — implementa o pré-processamento em inglês usado
no `full_pipeline.py`. Gera um artefato binário (pickle/joblib) em
`data/processed/processed_data.pkl` contendo o DataFrame processado.
"""

from pathlib import Path
import re
import os
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk_resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
for r in nltk_resources:
    try:
        nltk.download(r, quiet=True)
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
VIS_DIR = PROJECT_ROOT / 'reports' / 'visualizacoes'
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords_en(text):
    if not isinstance(text, str):
        return ''
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text, language='english')
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return ' '.join(filtered)

def lemmatize_text_en(text):
    if not isinstance(text, str):
        return ''
    tokens = word_tokenize(text, language='english')
    try:
        pos_tags = nltk.pos_tag(tokens)
    except Exception:
        pos_tags = [(t, '') for t in tokens]

    def _get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        if tag.startswith('V'):
            return wordnet.VERB
        if tag.startswith('N'):
            return wordnet.NOUN
        if tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN

    lemmas = []
    for token, tag in pos_tags:
        wn_tag = _get_wordnet_pos(tag) if tag else wordnet.NOUN
        lemmas.append(LEMMATIZER.lemmatize(token, wn_tag))
    return ' '.join(lemmas)

def find_raw_files():
    files = list(RAW_DIR.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'Nenhum arquivo CSV encontrado em {RAW_DIR}')
    train = RAW_DIR / 'twitter_training.csv'
    val = RAW_DIR / 'twitter_validation.csv'
    if train.exists() and val.exists():
        return train, val
    if len(files) >= 2:
        return files[0], files[1]
    return files[0], None

def load_processed():
    p = PROCESSED_DIR / 'processed_data.pkl'
    if not p.exists():
        raise FileNotFoundError(f'Processed data not found: {p}. Execute 02_preprocessing.py first')
    obj = joblib.load(p)
    return obj['train'], obj.get('validation', pd.DataFrame())

def run_preprocessing():
    print("Iniciando pré-processamento...")
    train_path, val_path = find_raw_files()
    print(f"Arquivos encontrados: {train_path}, {val_path}")
    cols = ['tweet_id', 'entity', 'sentiment', 'text']
    df_train = pd.read_csv(train_path, names=cols, header=None, engine='python', encoding='utf-8')
    df_val = pd.read_csv(val_path, names=cols, header=None, engine='python', encoding='utf-8') if val_path is not None else pd.DataFrame(columns=cols)

    df_train['text_clean'] = df_train['text'].apply(clean_text)
    df_train['text_no_stop'] = df_train['text_clean'].apply(remove_stopwords_en)
    df_train['text_lemmatized'] = df_train['text_no_stop'].apply(lemmatize_text_en)

    df_val['text_clean'] = df_val['text'].apply(clean_text)
    df_val['text_no_stop'] = df_val['text_clean'].apply(remove_stopwords_en)
    df_val['text_lemmatized'] = df_val['text_no_stop'].apply(lemmatize_text_en)

    # Salvar objeto binário (pickle) para uso pelos scripts seguintes
    out_path = PROCESSED_DIR / 'processed_data.pkl'
    joblib.dump({'train': df_train, 'validation': df_val}, out_path)
    print(f'[OK] Dados processados salvos (pickle): {out_path}')

if __name__ == '__main__':
    run_preprocessing()