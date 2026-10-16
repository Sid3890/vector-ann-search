import json
import os
import uuid

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
AWS_REGION = os.getenv("AWS_REGION", "ap-south-2")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")

app = FastAPI(title="GeekyClouds Embedding Service")
model = SentenceTransformer(MODEL_NAME)
sqs = boto3.client("sqs", region_name=AWS_REGION)


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dims: int


class IngestRequest(BaseModel):
    sourceId: str = Field(min_length=1)
    rawText: str = Field(min_length=1)


class IngestResponse(BaseModel):
    jobId: str
    status: str


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    embeddings = model.encode(request.texts, convert_to_numpy=True).tolist()
    return EmbedResponse(
        embeddings=embeddings,
        model=MODEL_NAME,
        dims=len(embeddings[0]) if embeddings else 0,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    if not SQS_QUEUE_URL:
        raise HTTPException(
            status_code=500,
            detail="SQS_QUEUE_URL environment variable is not configured",
        )

    job_id = str(uuid.uuid4())
    message = {
        "jobId": job_id,
        "sourceId": request.sourceId,
        "rawText": request.rawText,
    }

    try:
        sqs.send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps(message),
        )
    except (BotoCoreError, ClientError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Failed to publish ingestion job to SQS",
        ) from exc

    return IngestResponse(jobId=job_id, status="queued")
