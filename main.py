import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure NLTK data is downloaded
nltk_resources = ['punkt', 'stopwords', 'wordnet']
for resource in nltk_resources:
    try:
        nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = BeautifulSoup(text, "lxml").get_text()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])
    return text

def vectorize_w2v(docs, model, size):
    features = []
    for tokens in docs:
        vecs = [model.wv[word] for word in tokens if word in model.wv]
        if vecs:
            features.append(np.mean(vecs, axis=0))
        else:
            features.append(np.zeros(size))
    return np.array(features)

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")
st.title("📊 Sentiment Analysis with Random Forest")

uploaded_file = st.file_uploader("Upload your CSV file (must have 'reviewText' and 'rating')", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'reviewText' not in df.columns or 'rating' not in df.columns:
        st.error("CSV must contain 'reviewText' and 'rating' columns.")
    else:
        df.dropna(subset=['reviewText', 'rating'], inplace=True)
        df['label'] = df['rating'].apply(lambda x: 1 if x > 3 else 0)
        df['cleanText'] = df['reviewText'].apply(clean_text)
        df['tokens'] = df['cleanText'].apply(word_tokenize)

        vectorizer_type = st.selectbox("Choose Vectorizer", ["TF-IDF", "Bag of Words", "Word2Vec"])
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)

        if st.button("Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(df['cleanText'], df['label'], test_size=test_size, random_state=42)

            if vectorizer_type == "TF-IDF":
                vect = TfidfVectorizer()
                X_train_vec = vect.fit_transform(X_train)
                X_test_vec = vect.transform(X_test)
            elif vectorizer_type == "Bag of Words":
                vect = CountVectorizer()
                X_train_vec = vect.fit_transform(X_train)
                X_test_vec = vect.transform(X_test)
            elif vectorizer_type == "Word2Vec":
                model_w2v = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)
                X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(df['tokens'], df['label'], test_size=test_size, random_state=42)
                X_train_vec = vectorize_w2v(X_train_tokens, model_w2v, 100)
                X_test_vec = vectorize_w2v(X_test_tokens, model_w2v, 100)

            clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
            clf.fit(X_train_vec, y_train)
            y_pred = clf.predict(X_test_vec)
            y_prob = clf.predict_proba(X_test_vec)[:, 1]

            st.subheader("📈 Model Evaluation")
            st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

            st.subheader("🔍 ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax2.plot([0, 1], [0, 1], 'k--')
            ax2.set_xlabel("False Positive Rate")
            ax2.set_ylabel("True Positive Rate")
            ax2.set_title("ROC Curve")
            ax2.legend()
            st.pyplot(fig2)
