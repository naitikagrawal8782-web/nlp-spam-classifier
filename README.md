# nlp-spam-classifier
NLP-based CLI tool for spam message classification using TF-IDF and Naive Bayes.
# NLP-Based Intelligent Spam Detection System

An automated command-line interface (CLI) application for real-time text classification into **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) techniques and Multinomial Naive Bayes.

---

## Technical Features
- **Data Preprocessing**: Lowercasing, noise cleaning, stop-word removal, and Porter Stemming.
- **Feature Extraction**: TF-IDF vectorization with unigram and bigram ranges.
- **Classifier**: Optimized Multinomial Naive Bayes model.
- **Pure CLI Execution**: Completely operational via terminal commands without any GUI dependency.

---

## Installation & Setup

### 1. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/naitikagrawal8782-web/nlp-spam-classifier.git](https://github.com/naitikagrawal8782-web/nlp-spam-classifier.git)
cd nlp-spam-classifier

# Create and activate virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
