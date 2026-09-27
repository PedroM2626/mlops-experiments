import pandas as pd
import re
import os
from pathlib import Path
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load environment variables
load_dotenv()

# Download NLTK resources if needed
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
    Clean the text by removing special characters and links, converting to lowercase
    and applying lemmatization to normalize the words.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@user) and hashtags (#)
    text = re.sub(r'\@\w+|\#','', text)
    
    # Replace common contractions (optional, but it helps)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    # Remove punctuation and special characters, but keep '!' and '?' which may indicate sentiment
    text = re.sub(r'[^a-z\s\!\?]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal and lemmatization
    stop_words = set(stopwords.words('english'))
    # Remove 'not' and 'no' from the stopwords since they are crucial for sentiment
    stop_words.discard('not')
    stop_words.discard('no')
    
    lemmatizer = WordNetLemmatizer()
    
    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    
    return " ".join(filtered_tokens)

def preprocess_data():
    """
    Reads the raw data, cleans it and saves it to the processed directory.
    """
    project_root = Path(__file__).parent.parent.parent
    raw_dir = project_root / os.getenv('DATA_RAW_PATH', 'data/raw')
    processed_dir = project_root / os.getenv('DATA_PROCESSED_PATH', 'data/processed')
    
    # Create the processed directory if it does not exist
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
            print(f"File not found: {input_path}")
            continue
            
        print(f"Processing {input_file}...")
        
        # Read the CSV without a header
        df = pd.read_csv(input_path, names=columns, header=None)
        
        # Remove rows with null values in text or sentiment
        df = df.dropna(subset=['text', 'sentiment'])
        
        # Clean the text
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Remove rows that became empty after cleaning
        df = df[df['cleaned_text'] != ""]
        
        # Save the processed data
        df[['cleaned_text', 'sentiment']].to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"Preprocessing error: {e}")
