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
    
    # Substituir contrações comuns (opcional, mas ajuda)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    # Remover pontuação e caracteres especiais, mas manter '!' e '?' que podem indicar sentimento
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

def preprocess_data():
    """
    Lê os dados brutos, limpa e salva no diretório processado.
    """
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / os.getenv('DATA_RAW_PATH', 'data/raw')
    processed_dir = project_root / os.getenv('DATA_PROCESSED_PATH', 'data/processed')
    
    # Criar diretório processado se não existir
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_process = {
        'twitter_training.csv': 'train_cleaned.csv',
        'twitter_validation.csv': 'val_cleaned.csv'
    }
    
    columns = ['id', 'topic', 'sentiment', 'text']
    
    for input_file, output_file in files_to_process.items():
        input_path = raw_dir / input_file
        output_path = processed_dir / output_file
        
        if not input_path.exists():
            print(f"Arquivo não encontrado: {input_path}")
            continue
            
        print(f"Processando {input_file}...")
        
        # Ler CSV sem cabeçalho
        df = pd.read_csv(input_path, names=columns, header=None)
        
        # Remover linhas com valores nulos no texto ou sentimento
        df = df.dropna(subset=['text', 'sentiment'])
        
        # Limpar o texto
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Remover linhas que ficaram vazias após a limpeza
        df = df[df['cleaned_text'] != ""]
        
        # Salvar dados processados
        df[['cleaned_text', 'sentiment']].to_csv(output_path, index=False)
        print(f"Salvo em {output_path}")

if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
