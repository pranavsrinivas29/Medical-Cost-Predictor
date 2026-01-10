FROM python:3.11.9-slim

WORKDIR /ui

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY ui /app/ui
COPY config.py /app/config.py
COPY inference /app/inference

COPY data/feature_engineering /app/data/feature_engineering
COPY models/xgboost /app/models/xgboost

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.address=0.0.0.0"]
