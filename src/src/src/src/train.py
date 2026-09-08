import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_and_prepare_dataset

def build_and_train_model(data_path: str, model_dir: str = "models"):
    print("[INFO] Loading dataset...")
    df = load_and_prepare_dataset(data_path)
    
    X = df['clean_text']
    y = df['label_num']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("[INFO] Transforming text features via TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("[INFO] Training Naive Bayes Classifier...")
    model = MultinomialNB(alpha=0.2)
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n[RESULTS] Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(model, os.path.join(model_dir, "spam_classifier.pkl"))
    print(f"[INFO] Model successfully saved to '{model_dir}/'.")

if __name__ == "__main__":
    build_and_train_model("data/spam.csv")
