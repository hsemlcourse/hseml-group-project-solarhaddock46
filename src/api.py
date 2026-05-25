from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from inference import get_model_path, load_model, predict_income

_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    try:
        _pipeline = load_model(get_model_path())
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    yield
    _pipeline = None


app = FastAPI(
    title="Adult Income Prediction API",
    description=(
        "REST API для предсказания уровня годового дохода (`<=50K` / `>50K`) "
        "по социально-демографическому профилю. Модель: LightGBM (тюнинг)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class IncomeFeatures(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "Private",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education.num": 13,
                "marital.status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital.gain": 2174,
                "capital.loss": 0,
                "hours.per.week": 40,
                "native.country": "United-States",
            }
        },
    )

    workclass: str = Field(description="Тип занятости")
    education: str = Field(description="Образование")
    marital_status: str = Field(alias="marital.status", description="Семейное положение")
    occupation: str = Field(description="Профессия")
    relationship: str = Field(description="Отношения в семье")
    race: str = Field(description="Раса")
    sex: Literal["Male", "Female"] = Field(description="Пол")
    native_country: str = Field(alias="native.country", description="Страна происхождения")
    age: int = Field(ge=17, le=90, description="Возраст")
    fnlwgt: int = Field(ge=0, description="Вес записи переписи")
    education_num: int = Field(alias="education.num", ge=1, le=16, description="Число лет образования")
    capital_gain: int = Field(alias="capital.gain", ge=0, description="Прирост капитала")
    capital_loss: int = Field(alias="capital.loss", ge=0, description="Потери капитала")
    hours_per_week: int = Field(alias="hours.per.week", ge=1, le=99, description="Часов в неделю")

    def to_features_dict(self) -> dict:
        return self.model_dump(by_alias=True)


class PredictResponse(BaseModel):
    label: Literal["<=50K", ">50K"] = Field(description="Предсказанный класс дохода")
    prediction: Literal[0, 1] = Field(description="0 — <=50K, 1 — >50K")
    probability_high: float = Field(description="Вероятность класса >50K")
    probability_low: float = Field(description="Вероятность класса <=50K")


class HealthResponse(BaseModel):
    status: str
    model_path: str


@app.get("/health", response_model=HealthResponse, tags=["Служебные"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_path=get_model_path())


@app.post("/predict", response_model=PredictResponse, tags=["Предсказание"])
def predict(features: IncomeFeatures) -> PredictResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    result = predict_income(_pipeline, features.to_features_dict())
    return PredictResponse(**result)
