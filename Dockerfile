FROM python:3.11-slim

WORKDIR /app

# LightGBM требует libgomp в slim-образе
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src/ src/
COPY app/ app/
COPY models/best_model.joblib models/best_model.joblib

ENV PYTHONPATH=/app/src
ENV MODEL_PATH=/app/models/best_model.joblib

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]
