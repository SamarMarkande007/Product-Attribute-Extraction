import os
import sys
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.preprocess import clean_text  

app = FastAPI(
    title="Product Attribute Extraction API",
    description="Extract structured fashion attributes from product descriptions.",
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

ATTRIBUTES = [
    "category",
    "silhouette",
    "fabric",
    "neckline",
    "sleeve",
    "length",
    "embellishment",
    "color"
]

# Load artifacts once at startup
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf.pkl"))
models = {}
encoders = {}
for attribute in ATTRIBUTES:
    models[attribute] = joblib.load(os.path.join(MODEL_DIR, f"{attribute}.pkl"))
    encoders[attribute] = joblib.load(os.path.join(MODEL_DIR, f"{attribute}_encoder.pkl"))

# Request Model
class ProductRequest(BaseModel):
    description: str

# Health Check
@app.get("/")
def home():
    return {"message": "Product Attribute Extraction API is running."}

# Prediction Endpoint
@app.post("/extract")
def extract_attributes(request: ProductRequest):
    if not request.description or not request.description.strip():
        raise HTTPException(status_code=400, detail="description must not be empty")

    cleaned_text = clean_text(request.description)
    X = vectorizer.transform([cleaned_text])

    output = {}
    for attribute in ATTRIBUTES:
        prediction = models[attribute].predict(X)[0]
        label = encoders[attribute].inverse_transform([prediction])[0]
        output[attribute] = label
    return output
