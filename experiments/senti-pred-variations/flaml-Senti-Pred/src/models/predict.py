import sys
import os
from pathlib import Path
import joblib
from dotenv import load_dotenv

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))
from data.preprocess import clean_text

load_dotenv()

class SentimentPredictor:
    def __init__(self, model_path=None, vectorizer_path=None, le_path=None):
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / os.getenv('MODELS_PATH', 'models')
        
        if model_path is None:
            model_path = models_dir / 'flaml_model.pkl'
        if vectorizer_path is None:
            vectorizer_path = models_dir / 'tfidf_vectorizer.pkl'
        if le_path is None:
            le_path = models_dir / 'label_encoder.pkl'
            
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
        if not Path(vectorizer_path).exists():
            raise FileNotFoundError(f"Vetorizador não encontrado em {vectorizer_path}")
        if not Path(le_path).exists():
            raise FileNotFoundError(f"Label Encoder não encontrado em {le_path}")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.le = joblib.load(le_path)
        
    def predict(self, text):
        """
        Prediz o sentimento de um texto.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return "Neutral"
        
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)
        
        # Converter o índice numérico de volta para a string original
        return self.le.inverse_transform(prediction)[0]

if __name__ == "__main__":
    predictor = SentimentPredictor()
    test_texts = [
        "I love this game, it is amazing!",
        "This is the worst update ever, I hate it.",
        "The server is down again.",
        "I am just playing some matches."
    ]
    
    for text in test_texts:
        sentiment = predictor.predict(text)
        print(f"Text: {text} -> Sentiment: {sentiment}")
