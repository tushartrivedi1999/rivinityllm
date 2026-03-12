FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY rivinity_nexus /app/rivinity_nexus
COPY README.md /app/README.md

RUN mkdir -p /app/storage/models

EXPOSE 8000

CMD ["uvicorn", "rivinity_nexus.main:app", "--host", "0.0.0.0", "--port", "8000"]
