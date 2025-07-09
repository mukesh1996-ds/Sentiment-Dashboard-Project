import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==============================
# NLTK setup
# ==============================
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# ==============================
# Custom Preprocessing Transformer
# ==============================

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = BeautifulSoup(text, 'lxml').get_text()
        text = re.sub(r'[^a-zA-Z0-9 ]+', '', text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return ' '.join(words)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.apply(self.clean_text)

# ==============================
# Word2Vec Vectorizer
# ==============================

class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def fit(self, X, y=None):
        tokens = X.apply(word_tokenize)
        self.model = Word2Vec(sentences=tokens, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=4)
        self.tokens = tokens
        return self

    def transform(self, X):
        return np.vstack(X.apply(lambda x: self.vectorize(x)))

    def vectorize(self, text):
        tokens = word_tokenize(text)
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

# ==============================
# Evaluation Function
# ==============================

def evaluate_model(model_name, y_test, y_pred, y_prob=None):
    print(f"\n📊 {model_name} Evaluation")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{model_name} AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()

# ==============================
# Load & Prepare Dataset
# ==============================

df = pd.read_csv(r"C:\Users\mksmu\Downloads\all_kindle_review .csv")[['reviewText', 'rating']]
df.dropna(inplace=True)
df['label'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)

X = df['reviewText']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# TF-IDF + RandomForest Pipeline
# ==============================

tfidf_pipeline = Pipeline([
    ('cleaner', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'clf__n_estimators': [100],
    'clf__max_depth': [10, 20],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2],
    'clf__max_features': ['sqrt']
}

grid_tfidf = GridSearchCV(tfidf_pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_tfidf.fit(X_train, y_train)
y_pred_tfidf = grid_tfidf.predict(X_test)
y_prob_tfidf = grid_tfidf.predict_proba(X_test)[:, 1]

evaluate_model("TF-IDF + RandomForest", y_test, y_pred_tfidf, y_prob_tfidf)

# ==============================
# BOW + RandomForest Pipeline
# ==============================

bow_pipeline = Pipeline([
    ('cleaner', TextPreprocessor()),
    ('vectorizer', CountVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])

grid_bow = GridSearchCV(bow_pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_bow.fit(X_train, y_train)
y_pred_bow = grid_bow.predict(X_test)
y_prob_bow = grid_bow.predict_proba(X_test)[:, 1]

evaluate_model("BOW + RandomForest", y_test, y_pred_bow, y_prob_bow)

# ==============================
# Word2Vec + RandomForest Pipeline
# ==============================

w2v_pipeline = Pipeline([
    ('cleaner', TextPreprocessor()),
    ('w2v', Word2VecVectorizer()),
    ('clf', RandomForestClassifier(random_state=42))
])

grid_w2v = GridSearchCV(w2v_pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_w2v.fit(X_train, y_train)
y_pred_w2v = grid_w2v.predict(X_test)
y_prob_w2v = grid_w2v.predict_proba(X_test)[:, 1]

evaluate_model("Word2Vec + RandomForest", y_test, y_pred_w2v, y_prob_w2v)
