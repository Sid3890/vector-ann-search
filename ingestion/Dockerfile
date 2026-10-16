
FROM python:3.11-slim

WORKDIR /app

COPY ingestion/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY db ./db
COPY ingestion/worker.py ./ingestion/worker.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "ingestion.worker"]
