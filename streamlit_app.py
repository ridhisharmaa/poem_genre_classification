# streamlit_app.py

import streamlit as st
import joblib

from preprocess import clean_text

tfidf = joblib.load("model/tfidf.pkl")
model = joblib.load("model/logistic_model.pkl")
le = joblib.load("model/label_encoder.pkl")

st.title("📖 Poem Genre Classifier")

poem = st.text_area(
    "Enter your poem",
    height=200
)

if st.button("Predict Genre"):

    cleaned = clean_text(poem)

    vec = tfidf.transform([cleaned])

    pred = model.predict(vec)

    genre = le.inverse_transform(pred)[0]

    st.success(f"Predicted Genre: {genre}")