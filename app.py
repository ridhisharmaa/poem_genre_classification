from fastapi import FastAPI
from pydantic import BaseModel
import joblib

from preprocess import clean_text

app = FastAPI()

tfidf = joblib.load("model/tfidf.pkl")
model = joblib.load("model/logistic_model.pkl")
le = joblib.load("model/label_encoder.pkl")

class PoemRequest(BaseModel):
    poem: str

@app.post("/predict")
def predict(request: PoemRequest):

    cleaned_text = clean_text(request.poem)

    vec = tfidf.transform([cleaned_text])

    pred = model.predict(vec)

    genre = le.inverse_transform(pred)[0]

    return {
        "genre": genre,
        "cleaned_text": cleaned_text
    }