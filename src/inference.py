import os

import joblib

from preprocessing import build_inference_dataframe

LABEL_HIGH = ">50K"
LABEL_LOW = "<=50K"

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "best_model.joblib"
)


def get_model_path() -> str:
    return os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)


def load_model(path: str | None = None):
    model_path = path or get_model_path()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Модель не найдена: {model_path}. "
            "Убедитесь, что best_model.joblib существует (прогоните 03_experiments.ipynb)."
        )
    return joblib.load(model_path)


def predict_income(pipeline, features: dict) -> dict:
    X = build_inference_dataframe(features)
    pred = int(pipeline.predict(X)[0])
    proba = pipeline.predict_proba(X)[0]
    probability_high = float(proba[1])
    probability_low = float(proba[0])
    label = LABEL_HIGH if pred == 1 else LABEL_LOW
    return {
        "label": label,
        "prediction": pred,
        "probability_high": probability_high,
        "probability_low": probability_low,
    }
