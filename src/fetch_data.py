import argparse
import os

import pandas as pd
from ucimlrepo import fetch_ucirepo


COLUMN_RENAME = {
    "education-num": "education.num",
    "marital-status": "marital.status",
    "capital-gain": "capital.gain",
    "capital-loss": "capital.loss",
    "hours-per-week": "hours.per.week",
    "native-country": "native.country",
}


def fetch_adult_dataset(raw_dir: str = "data/raw") -> pd.DataFrame:
    print("Fetching Adult Census Income dataset (id=2) from UCI ML Repository...")
    adult = fetch_ucirepo(id=2)

    X: pd.DataFrame = adult.data.features
    y: pd.DataFrame = adult.data.targets

    df = X.copy()
    df["income"] = y.iloc[:, 0]

    df = df.rename(columns=COLUMN_RENAME)

    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, "adult.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns → {out_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Adult Census Income dataset from UCI.")
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory to save adult.csv (default: data/raw)",
    )
    args = parser.parse_args()
    fetch_adult_dataset(raw_dir=args.raw_dir)
