import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference import get_model_path, load_model, predict_income  # noqa: E402
from preprocessing import CAT_COLS, NUM_COLS  # noqa: E402

CATEGORIES_PATH = os.path.join(os.path.dirname(__file__), "categories.json")

DEFAULTS = {
    "age": 39,
    "fnlwgt": 77516,
    "education.num": 13,
    "capital.gain": 2174,
    "capital.loss": 0,
    "hours.per.week": 40,
    "workclass": "Private",
    "education": "Bachelors",
    "marital.status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "native.country": "United-States",
}

NUM_BOUNDS = {
    "age": (17, 90),
    "fnlwgt": (0, 1_000_000),
    "education.num": (1, 16),
    "capital.gain": (0, 100_000),
    "capital.loss": (0, 100_000),
    "hours.per.week": (1, 99),
}

NUM_LABELS = {
    "age": "Возраст",
    "fnlwgt": "Вес записи (fnlwgt)",
    "education.num": "Число лет образования",
    "capital.gain": "Прирост капитала",
    "capital.loss": "Потери капитала",
    "hours.per.week": "Часов в неделю",
}

CAT_LABELS = {
    "workclass": "Тип занятости",
    "education": "Образование",
    "marital.status": "Семейное положение",
    "occupation": "Профессия",
    "relationship": "Отношения в семье",
    "race": "Раса",
    "sex": "Пол",
    "native.country": "Страна происхождения",
}


@st.cache_resource
def get_pipeline():
    return load_model(get_model_path())


def load_categories() -> dict:
    with open(CATEGORIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="Прогноз дохода", page_icon="💰", layout="wide")
    st.title("Прогноз уровня дохода")
    st.markdown(
        "Бинарная классификация годового дохода (`<=50K` / `>50K`) "
        "по социально-демографическому профилю. Модель: **LightGBM** (тюнинг)."
    )

    try:
        pipeline = get_pipeline()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    categories = load_categories()

    with st.form("predict_form"):
        st.subheader("Социально-демографический профиль")
        col_left, col_right = st.columns(2)

        features = {}
        with col_left:
            for col in CAT_COLS[:4]:
                options = categories[col]
                default_idx = options.index(DEFAULTS[col]) if DEFAULTS[col] in options else 0
                features[col] = st.selectbox(
                    CAT_LABELS[col],
                    options,
                    index=default_idx,
                    key=col,
                )

        with col_right:
            for col in CAT_COLS[4:]:
                options = categories[col]
                default_idx = options.index(DEFAULTS[col]) if DEFAULTS[col] in options else 0
                features[col] = st.selectbox(
                    CAT_LABELS[col],
                    options,
                    index=default_idx,
                    key=col,
                )

        st.subheader("Числовые признаки")
        num_cols_layout = st.columns(3)
        for i, col in enumerate(NUM_COLS):
            lo, hi = NUM_BOUNDS[col]
            with num_cols_layout[i % 3]:
                features[col] = st.number_input(
                    NUM_LABELS[col],
                    min_value=lo,
                    max_value=hi,
                    value=DEFAULTS[col],
                    key=f"num_{col}",
                )

        submitted = st.form_submit_button("Предсказать", type="primary")

    if submitted:
        result = predict_income(pipeline, features)
        st.divider()
        st.subheader("Результат")

        if result["label"] == ">50K":
            st.success(f"**Прогноз: {result['label']}** (доход выше $50 000)")
        else:
            st.info(f"**Прогноз: {result['label']}** (доход $50 000 и ниже)")

        st.metric(
            label="Вероятность дохода >50K",
            value=f"{result['probability_high']:.1%}",
        )
        st.progress(result["probability_high"])
        st.caption(
            f"Вероятность <=50K: {result['probability_low']:.1%}. "
            "Порог классификации: 0.5."
        )


if __name__ == "__main__":
    main()
