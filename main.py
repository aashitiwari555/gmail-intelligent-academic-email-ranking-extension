import os
import gdown
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    email_text: str

os.makedirs("models", exist_ok=True)

FILES = {
    "intent_model.pkl": "10fQIVZn2N7KmwWrsYFM82fvp0IJdRfWN",
    "intent_vectorizer.pkl": "1HUvHybun2ijiJu1rVtS2Ye3a7BrPkXBo",
    "intent_encoder.pkl": "1c8YAvJj9UI-Kk9sMXfy0MfcOUn5EmxPM",

    "priority_model.pkl": "1BjYYAreQ2hAZ0OlZS8xFsH_8AzKtncBK",
    "priority_vectorizer.pkl": "1L_vS_DSmMuL_CA3BS_ta34mDm3RxN8o6",
    "priority_encoder.pkl": "1vRUc0Xd_aHuko3vmA95XWbG5SvZj9F5W"
}

for filename, file_id in FILES.items():
    filepath = f"models/{filename}"
    if not os.path.exists(filepath):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, filepath, quiet=False)

# Load models
intent_model = joblib.load("models/intent_model.pkl")
intent_vectorizer = joblib.load("models/intent_vectorizer.pkl")
intent_encoder = joblib.load("models/intent_encoder.pkl")

priority_model = joblib.load("models/priority_model.pkl")
priority_vectorizer = joblib.load("models/priority_vectorizer.pkl")
priority_encoder = joblib.load("models/priority_encoder.pkl")

def preprocess_text(text):
    return str(text).lower()

@app.get("/")
def home():
    return {"message": "Email AI API running"}


@app.post("/predict")
def predict(request: EmailRequest):
    try:
        email_text = request.email_text

        clean_intent = preprocess_text(email_text)
        intent_vec = intent_vectorizer.transform([clean_intent])
        intent_pred = intent_model.predict(intent_vec)
        intent_label = intent_encoder.inverse_transform(intent_pred)[0]

        clean_priority = preprocess_text(email_text)
        priority_vec = priority_vectorizer.transform([clean_priority])

        priority_pred = priority_model.predict(priority_vec)
        priority_label = priority_encoder.inverse_transform(priority_pred)[0]

        high_idx = list(priority_encoder.classes_).index('High')
        urgency_score = priority_model.predict_proba(priority_vec)[0][high_idx]

        return {
            "intent": intent_label,
            "priority": priority_label,
            "urgency_score": float(urgency_score)
        }

    except Exception as e:
        return {"error": str(e)}
