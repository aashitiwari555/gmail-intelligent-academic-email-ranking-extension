from fastapi import FastAPI
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    email_text: str

# Load ML models
intent_model = joblib.load("models/intent_model.pkl")
intent_vectorizer = joblib.load("models/intent_vectorizer.pkl")
priority_model = joblib.load("models/priority_model.pkl")
intent_encoder = joblib.load("models/intent_encoder.pkl")

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')





# Prediction API

@app.get("/")
def home():
    return {"message": "Email AI API running"}


@app.post("/predict")
def predict(request: EmailRequest):

    email_text = request.email_text

    intent_vec = intent_vectorizer.transform([email_text])
    intent_pred = intent_model.predict(intent_vec)
    intent_label = intent_encoder.inverse_transform(intent_pred)[0]

    emb = embedding_model.encode([email_text])
    urgency_score = priority_model.predict(emb)[0]

    return {
        "intent": intent_label,
        "urgency_score": float(urgency_score)
    }