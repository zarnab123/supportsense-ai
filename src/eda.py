from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "banking77_train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "raw" / "banking77_test.csv"


def summarize_dataset(name: str, dataframe: pd.DataFrame) -> None:
    """Print a short, repeatable dataset-quality summary."""
    print(f"\n{'=' * 60}")
    print(f"{name} DATASET SUMMARY")
    print(f"{'=' * 60}")

    print(f"Rows: {len(dataframe):,}")
    print(f"Columns: {dataframe.columns.tolist()}")
    print(f"Missing values:\n{dataframe.isna().sum()}")
    print(f"Duplicate ticket texts: {dataframe.duplicated(subset=['text']).sum():,}")
    print(f"Unique ticket categories: {dataframe['category'].nunique()}")

    print("\nTop 10 categories by number of tickets:")
    print(dataframe["category"].value_counts().head(10).to_string())


def main() -> None:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    summarize_dataset("TRAIN", train_df)
    summarize_dataset("TEST", test_df)


if __name__ == "__main__":
    main()