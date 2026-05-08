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
│   └── 02_baseline.ipynb       # Baseline: LogisticRegression, метрики на val/test
├── presentation/               # Презентация для защиты
├── report
│   └── report.md               # Финальный отчёт
├── src
│   └── preprocessing.py        # Функции загрузки, очистки, FE, сплита
├── tests
│   └── test.py                 # Smoke-тесты пайплайна предобработки
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

# 4. Скачать датасет (вручную с Kaggle) и положить в:
#    data/raw/adult.csv

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

| Модель | ROC-AUC | F1-macro | Примечание |
|--------|---------|----------|------------|
| Baseline (LogisticRegression) | 0.8515 | 0.7201 | test, без feature engineering |
| Лучшая модель | — | — | CP2 |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
