import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "income"

CAT_COLS = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

NUM_COLS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.replace("?", np.nan)

    for col in ["workclass", "occupation", "native.country"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df[TARGET] = (df[TARGET].str.strip() == ">50K").astype(int)

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    for col in CAT_COLS:
        df[col] = df[col].astype("category")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["capital_net"] = df["capital.gain"] - df["capital.loss"]
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["<25", "25-35", "35-50", "50-65", "65+"],
    ).astype("category")
    df["is_usa"] = (df["native.country"] == "United-States").astype(int)

    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple:
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=seed, stratify=y
    )
    relative_test_size = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=seed, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
