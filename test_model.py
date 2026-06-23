import joblib

import joblib

tfidf = joblib.load("model/tfidf.pkl")
model = joblib.load("model/logistic_model.pkl")
le = joblib.load("model/label_encoder.pkl")

poem = """
love fills my heart
"""

vec = tfidf.transform([poem])

pred = model.predict(vec)

genre = le.inverse_transform(pred)

print(genre[0])