from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import confidence_label, suggest_priority


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"

app = FastAPI(
    title="SupportSense AI API",
    description="Banking customer-support ticket triage using machine learning.",
    version="1.0.0",
)


class TicketRequest(BaseModel):
    text: str = Field(
        min_length=5,
        max_length=1000,
        examples=["Someone used my card without permission and I need help."],
    )


class TriageResponse(BaseModel):
    category: str
    priority: str
    confidence_indicator: str
    human_review_required: bool = True


@lru_cache
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: python src\\train.py")
    return joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Welcome to SupportSense AI",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_available": MODEL_PATH.exists(),
    }


@app.post("/triage", response_model=TriageResponse)
def triage_ticket(ticket: TicketRequest):
    clean_text = ticket.text.strip()

    if not clean_text:
        raise HTTPException(status_code=422, detail="Ticket text cannot be empty.")

    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    category = model.predict([clean_text])[0]
    priority = suggest_priority(clean_text, category)
    confidence = confidence_label(model, clean_text)

    return TriageResponse(
        category=category,
        priority=priority,
        confidence_indicator=confidence,
    )