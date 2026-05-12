import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import CAT_COLS, NUM_COLS


def build_preprocessor(
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
) -> ColumnTransformer:
    if cat_cols is None:
        cat_cols = CAT_COLS
    if num_cols is None:
        num_cols = NUM_COLS
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipeline = Pipeline([
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("num", num_pipeline, num_cols),
        ],
        remainder="drop",
    )


def build_pipeline(
    model,
    use_pca: bool = False,
    n_components: int = 10,
    cat_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
) -> Pipeline:
    steps = [("preprocessor", build_preprocessor(cat_cols, num_cols))]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components)))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(pipeline, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    return {
        "split": split_name,
        "roc_auc": round(roc_auc_score(y, y_proba), 4),
        "f1_macro": round(f1_score(y, y_pred, average="macro"), 4),
    }


def save_model(pipeline, path: str) -> None:
    joblib.dump(pipeline, path)
    print(f"Model saved → {path}")


def results_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model")
