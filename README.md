# 📝 Poem Genre Classification using NLP and Machine Learning

An end-to-end Natural Language Processing (NLP) project that automatically predicts the **genre of a poem** using Machine Learning techniques. The project includes data preprocessing, feature engineering, model training, a FastAPI backend, and an interactive Streamlit frontend for real-time predictions.

---

## 🚀 Project Overview

Poetry comes in various genres such as romance, mythology, humor, and more. Manually classifying poems into genres can be time-consuming and subjective.

This project leverages **Natural Language Processing (NLP)** and **Machine Learning** to automatically identify the genre of a given poem based on its textual content.

---

## ✨ Features

- Text preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- TF-IDF feature extraction
- Poem genre prediction using Machine Learning
- REST API built with FastAPI
- Interactive web application using Streamlit
- Model serialization using Joblib

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries & Frameworks
- Pandas
- NumPy
- NLTK
- Scikit-Learn
- Joblib
- FastAPI
- Uvicorn
- Streamlit

---

## 📂 Project Structure

```text
poem_classification/
│
├── app.py                     # FastAPI application
├── streamlit_app.py           # Streamlit frontend
├── preprocess.py              # Text preprocessing functions
├── test_model.py
├── requirements.txt
│
├── data/
│   ├── train_data.csv
│   ├── test_data.csv
│   ├── train_clean.csv
│   ├── test_clean.csv
│   └── test_eda.csv
│
├── model/
│   ├── tfidf.pkl
│   ├── logistic_model.pkl
│   └── label_encoder.pkl
│
└── notebooks/
    ├── eda.ipynb
    ├── preprocessing.ipynb
    ├── tfidf_models.ipynb
    └── word2vec_models.ipynb
```

---

## 🔍 NLP Pipeline

### Text Preprocessing
- Lowercasing
- Removal of punctuation and special characters
- Stopword removal
- Tokenization
- Lemmatization

### Feature Engineering
- TF-IDF Vectorization
- Word2Vec experiments

### Machine Learning Model
- Logistic Regression Classifier

---






---

⭐ If you found this project useful, consider giving it a star!
