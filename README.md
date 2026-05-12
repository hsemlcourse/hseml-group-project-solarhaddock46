[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# Классификация уровня дохода по социально-демографическому профилю

**Студент:** Мацнев Владимир Дмитриевич 

**Группа:** БИВ232


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)
7. [Линтер](#линтер)


## Описание задачи

Бинарная классификация уровня годового дохода жителей США (`<=50K` / `>50K`) по социально-демографическому профилю на основе данных переписи населения 1994 года.

**Задача:** Бинарная классификация

**Датасет:** [Adult Census Income — UCI / Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income) — 32 537 строк, 14 признаков

**Целевая метрика:** ROC-AUC (основная), F1-macro (дополнительная)


## Структура репозитория

```
.
├── data
│   ├── processed/              # Очищенные данные (adult_processed.csv, gitignored)
│   └── raw/                    # Исходные файлы (adult.csv, gitignored)
├── models/                     # Сохранённые модели (gitignored)
├── notebooks
│   ├── 01_eda.ipynb            # EDA: очистка, визуализации, feature engineering, сплит
│   ├── 02_baseline.ipynb       # Baseline: LogisticRegression, метрики на val/test
│   └── 03_experiments.ipynb    # Эксперименты: 4-5 моделей, тюнинг, PCA, финальная модель
├── presentation/               # Презентация для защиты
├── report
│   └── report.md               # Финальный отчёт
├── src
│   ├── fetch_data.py           # Загрузка датасета через UCI ML Repository API
│   ├── preprocessing.py        # Функции загрузки, очистки, FE, сплита
│   └── train.py                # build_pipeline, evaluate, save_model
├── tests
│   └── test.py                 # Smoke-тесты пайплайна предобработки
├── Makefile
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-solarhaddock46.git
cd hseml-group-project-solarhaddock46

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Загрузить датасет через UCI ML Repository (автоматически сохраняется в data/raw/adult.csv)
python src/fetch_data.py

# 5. Запустить ноутбуки по порядку
jupyter notebook

# 6. Прогнать тесты
pytest tests/test.py -v
```

## Данные

- `data/raw/adult.csv` — исходный датасет (не коммитится, скачать с [Kaggle](https://www.kaggle.com/datasets/uciml/adult-census-income))
- `data/processed/adult_processed.csv` — очищенный датасет с новыми фичами (генерируется `01_eda.ipynb`)

**Особенности датасета:**
- 32 561 строка, 15 столбцов (14 признаков + таргет)
- Пропуски закодированы как `?` в столбцах `workclass`, `occupation`, `native.country`
- Дисбаланс классов: ~76% `<=50K`, ~24% `>50K`


## Результаты

| Модель | ROC-AUC (val) | F1-macro (val) | Примечание |
|--------|---------------|----------------|------------|
| Baseline (LogisticRegression) | 0.8536 | 0.7266 | дефолт, без FE |
| DecisionTree | 0.6498 | 0.6491 | переобучение на train |
| RandomForest (дефолт) | 0.8535 | 0.6716 | часть 1 экспериментов |
| GradientBoosting (дефолт) | 0.8774 | 0.6940 | часть 1 экспериментов |
| XGBoost (дефолт) | 0.8792 | 0.6935 | часть 1 экспериментов |
| LightGBM (дефолт) | 0.8837 | 0.6980 | часть 1 экспериментов |
| RandomForest (тюнинг) | 0.8766 | 0.6596 | RandomizedSearchCV |
| VotingClassifier (LGBM+XGB+GB) | 0.8834 | 0.6965 | ансамбль |
| LightGBM (тюнинг) | 0.8843 | 0.6982 | финальная модель, test ROC-AUC 0.8756 |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)


## Линтер

```bash
# Запустить flake8 через make
make lint

# Или напрямую
flake8 src/ tests/
```
