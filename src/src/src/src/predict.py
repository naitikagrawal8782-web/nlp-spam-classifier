import os
import joblib
from src.data_loader import TextPreprocessor

class SpamDetectorCLI:
    def __init__(self, model_dir: str = "models"):
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        clf_path = os.path.join(model_dir, "spam_classifier.pkl")

        if not os.path.exists(vec_path) or not os.path.exists(clf_path):
            raise FileNotFoundError("Model files missing. Run training script first: python src/train.py")

        self.vectorizer = joblib.load(vec_path)
        self.model = joblib.load(clf_path)
        self.preprocessor = TextPreprocessor()

    def classify_text(self, raw_text: str):
        cleaned = self.preprocessor.clean_text(raw_text)
        features = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        confidence = probabilities[prediction] * 100
        label = "SPAM" if prediction == 1 else "HAM (Legitimate)"

        return label, confidence
