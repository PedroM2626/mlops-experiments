import pandas as pd
import re
import os
from pathlib import Path
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Carregar variáveis de ambiente
load_dotenv()

# Baixar recursos do NLTK se necessário
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')

download_nltk_resources()

def clean_text(text):
    """
    Limpa o texto removendo caracteres especiais, links, converte para minúsculas
    e aplica lemmatização para normalizar as palavras.
    """
    if not isinstance(text, str):
        return ""
    
    # Converter para minúsculas
    text = text.lower()
    
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remover menções (@usuario) e hashtags (#)
    text = re.sub(r'\@\w+|\#','', text)
    
    # Substituir contrações comuns
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    # Remover pontuação e caracteres especiais, mas manter '!' e '?' que indicam sentimento
    text = re.sub(r'[^a-z\s\!\?]', '', text)
    
    # Tokenização
    tokens = word_tokenize(text)
    
    # Remoção de stopwords e Lemmatização
    stop_words = set(stopwords.words('english'))
    # Remover 'not' e 'no' das stopwords pois são cruciais para sentimento
    stop_words.discard('not')
    stop_words.discard('no')
    
    lemmatizer = WordNetLemmatizer()
    
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    
    return " ".join(filtered_tokens)
