
FROM python:3.11-slim

WORKDIR /app

COPY embedding-service/requirements.txt ./requirements.txt

RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip show sentence-transformers

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY embedding-service/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
