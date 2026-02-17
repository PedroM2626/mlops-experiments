import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys

# Adicionar src ao path para importar o preprocessador
sys.path.append(str(Path(__file__).parent.parent))
from data.preprocess import clean_text

# Carregar variáveis de ambiente
load_dotenv()

def train_model():
    """
    Treina o modelo de análise de sentimentos utilizando o dataset bruto original.
    Utiliza LinearSVC com C=1.0 e 3-grams para máxima acurácia equilibrada.
    """
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / os.getenv('DATA_RAW_PATH', 'data/raw')
    models_dir = project_root / os.getenv('MODELS_PATH', 'models')
    
    # Criar diretório de modelos se não existir
    models_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = raw_dir / 'twitter_training.csv'
    val_path = raw_dir / 'twitter_validation.csv'
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Dados brutos não encontrados em {raw_dir}. Verifique os arquivos CSV.")
    
    columns = ['id', 'topic', 'sentiment', 'text']
    
    print("Carregando dados brutos originais...")
    train_df = pd.read_csv(train_path, names=columns, header=None)
    val_df = pd.read_csv(val_path, names=columns, header=None)
    
    # Limpeza básica (remover nulos) antes da vetorização
    print("Limpando dados e removendo valores nulos...")
    train_df = train_df.dropna(subset=['text', 'sentiment'])
    val_df = val_df.dropna(subset=['text', 'sentiment'])
    
    # Aplicar a limpeza de texto (Lemmatization inclusa no preprocess.py)
    print("Processando textos (limpeza e lemmatização)...")
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    val_df['cleaned_text'] = val_df['text'].apply(clean_text)
    
    # Remover linhas que ficaram vazias após a limpeza
    train_df = train_df[train_df['cleaned_text'] != ""]
    val_df = val_df[val_df['cleaned_text'] != ""]
    
    # Vetorização TF-IDF com 4-grams e limite de features para 100k
    print("Vetorizando textos (N-grams 1-4, 100k features)...")
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
    y_train = train_df['sentiment']
    
    X_val = vectorizer.transform(val_df['cleaned_text'])
    y_val = val_df['sentiment']
    
    # Configuração de um Ensemble (Voting Classifier)
    print("Treinando o modelo Ensemble (LinearSVC + LogisticRegression)...")
    
    svc = LinearSVC(C=0.5, max_iter=3000, dual='auto', random_state=42, tol=1e-5, class_weight='balanced')
    lr = LogisticRegression(C=10, max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=42, class_weight='balanced')
    
    model = VotingClassifier(
        estimators=[('svc', svc), ('lr', lr)],
        voting='hard'
    )
    
    model.fit(X_train, y_train)
    
    # Avaliação
    print("Avaliando o modelo...")
    y_pred = model.predict(X_val)
    
    print("\nResultados da Validação:")
    print(f"Acurácia: {accuracy_score(y_val, y_pred):.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_val, y_pred))
    
    # Salvar artefatos
    print(f"Salvando artefatos em {models_dir}...")
    joblib.dump(model, models_dir / 'sentiment_model.pkl')
    joblib.dump(vectorizer, models_dir / 'tfidf_vectorizer.pkl')
    print("Concluído!")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Erro no treinamento: {e}")
