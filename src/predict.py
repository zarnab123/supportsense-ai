from pathlib import Path
import sys

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"

HIGH_PRIORITY_CATEGORIES = {
    "cash_withdrawal_not_recognised",
    "card_payment_not_recognised",
    "card_swallowed",
    "cash_withdrawal_charge",
    "compromised_card",
}


def suggest_priority(ticket: str, category: str) -> str:
    urgent_words = {"fraud", "stolen", "scam", "unauthorised", "unauthorized", "urgent"}

    if category in HIGH_PRIORITY_CATEGORIES:
        return "HIGH"

    if any(word in ticket.lower() for word in urgent_words):
        return "HIGH"

    return "NORMAL"


def confidence_label(model, ticket: str) -> str:
    scores = model.decision_function([ticket])[0]
    top_two_scores = np.sort(scores)[-2:]
    margin = top_two_scores[1] - top_two_scores[0]

    if margin >= 1.0:
        return "High"
    if margin >= 0.35:
        return "Medium"
    return "Low"


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python src\\predict.py "Your banking support ticket here"')
        return

    ticket = " ".join(sys.argv[1:])
    model = joblib.load(MODEL_PATH)

    category = model.predict([ticket])[0]
    priority = suggest_priority(ticket, category)
    confidence = confidence_label(model, ticket)

    print("\nSUPPORTSENSE AI — TICKET TRIAGE RESULT")
    print("-" * 45)
    print(f"Ticket: {ticket}")
    print(f"Predicted category: {category}")
    print(f"Priority suggestion: {priority}")
    print(f"Confidence indicator: {confidence}")
    print("\nNote: priority is a rule-based suggestion; a human reviews final decisions.")


if __name__ == "__main__":
    main()