sentiment_analysis_project/
├── main.py
├── README.md
├── requirements.txt
└── sample_data.csv

---

# main.py

```python
# [Code from previous response here — full Streamlit app with TF-IDF, BoW, Word2Vec]
```

---

# README.md

```markdown
# 📊 Sentiment Analysis Dashboard

This project is an interactive Streamlit dashboard for sentiment analysis of textual reviews. It supports vectorization using TF-IDF, Bag of Words, and Word2Vec, and uses Random Forest for classification.

## 📁 Project Structure
```
sentiment_analysis_project/
├── main.py                # Streamlit dashboard
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── sample_data.csv       # Sample input data
```

## 📥 How to Use
1. Clone the repo or download the files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the app:
   ```bash
   streamlit run app.py
   ```
4. Upload a CSV with columns `reviewText` and `rating`.

## ⚙️ Features
- Text preprocessing (cleaning, lemmatization)
- Vectorization: TF-IDF, BoW, Word2Vec
- Classification using RandomForestClassifier
- Performance metrics: accuracy, confusion matrix, classification report
- ROC Curve visualization

## 📌 Notes
- Ratings > 3 are considered positive (label=1), <=3 are negative (label=0)
- Word2Vec uses average of word embeddings per review

## 📧 Contact
For questions, feel free to raise an issue or contribute.
```

---

# requirements.txt

```text
streamlit
pandas
numpy
scikit-learn
nltk
gensim
beautifulsoup4
matplotlib
seaborn
lxml
```

---

# sample_data.csv (header only example)

```csv
reviewText,rating
"I loved the product. It's amazing!",5
"Worst purchase ever. Terrible experience.",1
```
