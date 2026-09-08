import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = text.split()
        cleaned = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return " ".join(cleaned)

def load_and_prepare_dataset(file_path: str):
    df = pd.read_csv(file_path, encoding='latin-1')

    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    elif 'Category' in df.columns and 'Message' in df.columns:
        df = df[['Category', 'Message']].rename(columns={'Category': 'label', 'Message': 'text'})

    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1, 'HAM': 0, 'SPAM': 1})

    preprocessor = TextPreprocessor()
    df['clean_text'] = df['text'].apply(preprocessor.clean_text)
    return df
