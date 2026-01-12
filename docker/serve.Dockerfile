FROM python:3.11.9-slim

WORKDIR /app

# System deps (optional but safe)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install dvc[http]  # required for DAGSHub remote

# Copy application code
#COPY app ./app
#COPY inference ./inference
#COPY config.py .
#COPY data/ /data/

# Copy project files (including .dvc, .dvc/config, dvc.lock, etc.)
COPY . .

# Add these lines before `dvc pull`
ARG DVC_USER
ARG DVC_PASSWORD

# Setup DVC auth and pull data
RUN dvc config core.no_scm true && \
    dvc remote modify origin auth basic && \
    dvc remote modify origin user $DVC_USER && \
    dvc remote modify origin password $DVC_PASSWORD && \
    dvc pull

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
