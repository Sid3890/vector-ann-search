FROM python:3.11-slim

WORKDIR /app

COPY router/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY router/main.py ./main.py

ENV SHARD_TIMEOUT_SECONDS=2.0

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
