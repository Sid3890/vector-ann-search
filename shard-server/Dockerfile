
FROM python:3.11-slim

WORKDIR /app

COPY shard-server/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY shard-server/main.py ./main.py

ENV SHARD_ID=0
ENV INDEX_DIR=/app/indexes

EXPOSE 8000

CMD ["sh", "-c", "export INDEX_PATH=\"${INDEX_PATH:-${INDEX_DIR}/shard_${SHARD_ID}.faiss}\" && exec uvicorn main:app --host 0.0.0.0 --port 8000"]
