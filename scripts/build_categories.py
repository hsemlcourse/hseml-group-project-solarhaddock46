"""Генерирует app/categories.json из data/raw/adult.csv."""

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import CAT_COLS, load_raw_data  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "raw", "adult.csv")
OUTPUT_PATH = os.path.join(ROOT, "app", "categories.json")


def main() -> None:
    df = load_raw_data(DATA_PATH)
    categories = {}
    for col in CAT_COLS:
        values = sorted(df[col].replace("?", pd.NA).dropna().unique().tolist())
        categories[col] = [str(v) for v in values]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
