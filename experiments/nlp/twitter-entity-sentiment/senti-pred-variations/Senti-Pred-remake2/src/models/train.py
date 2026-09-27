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

# Add src to the path to import the preprocessor
sys.path.append(str(Path(__file__).parent.parent))
from data.preprocess import clean_text

# Load environment variables
load_dotenv()

def train_model():
    """
    Train the sentiment analysis model using the original raw dataset.
    Uses LinearSVC with C=1.0 and 3-grams for maximum balanced accuracy.
    """
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / os.getenv('DATA_RAW_PATH', 'data/raw')
    models_dir = project_root / os.getenv('MODELS_PATH', 'models')
    
    # Create the models directory if it does not exist
    models_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = raw_dir / 'twitter_training.csv'
    val_path = raw_dir / 'twitter_validation.csv'
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Raw data not found in {raw_dir}. Check the CSV files.")
    
    columns = ['id', 'topic', 'sentiment', 'text']
    
    print("Loading original raw data...")
    train_df = pd.read_csv(train_path, names=columns, header=None)
    val_df = pd.read_csv(val_path, names=columns, header=None)
    
    # Basic cleaning (remove nulls) before vectorization
    print("Cleaning data and removing null values...")
    train_df = train_df.dropna(subset=['text', 'sentiment'])
    val_df = val_df.dropna(subset=['text', 'sentiment'])
    
    # Apply text cleaning (Lemmatization included in preprocess.py)
    print("Processing texts (cleaning and lemmatization)...")
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    val_df['cleaned_text'] = val_df['text'].apply(clean_text)
    
    # Remove rows that became empty after cleaning
    train_df = train_df[train_df['cleaned_text'] != ""]
    val_df = val_df[val_df['cleaned_text'] != ""]
    
    # TF-IDF vectorization with 4-grams and feature limit set to 100k
    print("Vectorizing texts (N-grams 1-4, 100k features)...")
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
    
    # Ensemble configuration (Voting Classifier)
    print("Training the Ensemble model (LinearSVC + LogisticRegression)...")
    
    svc = LinearSVC(C=0.5, max_iter=3000, dual='auto', random_state=42, tol=1e-5, class_weight='balanced')
    lr = LogisticRegression(C=10, max_iter=1000, solver='lbfgs', multi_class='multinomial', random_state=42, class_weight='balanced')
    
    model = VotingClassifier(
        estimators=[('svc', svc), ('lr', lr)],
        voting='hard'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    print("Evaluating the model...")
    y_pred = model.predict(X_val)
    
    print("\nValidation results:")
    print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))
    
    # Save artifacts
    print(f"Saving artifacts to {models_dir}...")
    joblib.dump(model, models_dir / 'sentiment_model.pkl')
    joblib.dump(vectorizer, models_dir / 'tfidf_vectorizer.pkl')
    print("Done!")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training error: {e}")
