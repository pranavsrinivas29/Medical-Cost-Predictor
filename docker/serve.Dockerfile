FROM python:3.11.9-slim

WORKDIR /app

# System deps (optional but safe)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
#COPY app ./app
#COPY inference ./inference
#COPY config.py .
COPY app /app/app
COPY config.py /app/config.py
COPY inference /app/inference

# Copy ML artifacts into the Docker image
COPY data/feature_engineering /app/data/feature_engineering
COPY models/xgboost /app/models/xgboost

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
