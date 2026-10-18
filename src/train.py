from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "banking77_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "banking77_test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "ticket_classifier.joblib"


def main() -> None:
    """Train and evaluate a baseline banking-ticket classifier."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=40_000,
                ),
            ),
            ("classifier", LinearSVC()),
        ]
    )

    print("Training the ticket classifier...")
    model.fit(train_df["text"], train_df["category"])

    predictions = model.predict(test_df["text"])
    accuracy = accuracy_score(test_df["category"], predictions)

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            test_df["category"],
            predictions,
            zero_division=0,
        )
    )

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()