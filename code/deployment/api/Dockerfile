FROM python:3.12-slim

WORKDIR /app

COPY code/deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/deployment/api/model_api.py .
COPY code/models/feature_engineering.py .
COPY models/catboost_model.cbm /app/models/catboost_model.cbm

ENV MODEL_PATH=/app/models/catboost_model.cbm
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]

