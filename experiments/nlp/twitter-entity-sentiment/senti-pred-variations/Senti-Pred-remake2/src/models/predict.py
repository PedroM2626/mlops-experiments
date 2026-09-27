import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# Add the src directory to the path to import the preprocessor
sys.path.append(str(Path(__file__).parent.parent))
from data.preprocess import clean_text

# Load environment variables
load_dotenv()

class SentimentPredictor:
    def __init__(self, model_path=None, vectorizer_path=None):
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / os.getenv('MODELS_PATH', 'models')
        
        model_path = model_path or models_dir / 'sentiment_model.pkl'
        vectorizer_path = vectorizer_path or models_dir / 'tfidf_vectorizer.pkl'
        
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError("Model artifacts not found. Train the model first.")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        """
        Predicts the sentiment of an individual text.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return "Neutral" # Or handling for empty text
            
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)
        return prediction[0]

if __name__ == "__main__":
    try:
        predictor = SentimentPredictor()
        
        # Simple interactive test
        while True:
            text = input("\nEnter a text to analyze (or 'exit' to quit): ")
            if text.lower() == 'exit':
                break
            
            sentiment = predictor.predict(text)
            print(f"Predicted sentiment: {sentiment}")
            
    except Exception as e:
        print(f"Prediction error: {e}")
