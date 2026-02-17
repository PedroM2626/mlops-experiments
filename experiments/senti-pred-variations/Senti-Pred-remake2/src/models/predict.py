import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# Adicionar o diretório src ao path para importar o preprocessador
sys.path.append(str(Path(__file__).parent.parent))
from data.preprocess import clean_text

# Carregar variáveis de ambiente
load_dotenv()

class SentimentPredictor:
    def __init__(self, model_path=None, vectorizer_path=None):
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / os.getenv('MODELS_PATH', 'models')
        
        model_path = model_path or models_dir / 'sentiment_model.pkl'
        vectorizer_path = vectorizer_path or models_dir / 'tfidf_vectorizer.pkl'
        
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError("Artefatos do modelo não encontrados. Treine o modelo primeiro.")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        """
        Prevê o sentimento de um texto individual.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return "Neutral" # Ou tratamento para texto vazio
            
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)
        return prediction[0]

if __name__ == "__main__":
    try:
        predictor = SentimentPredictor()
        
        # Teste interativo simples
        while True:
            text = input("\nDigite um texto para análise (ou 'sair' para encerrar): ")
            if text.lower() == 'sair':
                break
            
            sentiment = predictor.predict(text)
            print(f"Sentimento previsto: {sentiment}")
            
    except Exception as e:
        print(f"Erro na predição: {e}")
