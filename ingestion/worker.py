
import json
import logging
import os
import time
import uuid
from typing import TypedDict

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from db.client import SessionLocal


class IngestJob(TypedDict):
    jobId: str
    sourceId: str
    rawText: str


class EmbedResponse(TypedDict):
    embeddings: list[list[float]]
    model: str
    dims: int


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-south-2")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
EMBEDDING_URL = os.getenv(
    "EMBEDDING_URL",
    "http://host.docker.internal:8000",
)
EMBEDDING_DIMENSIONS = 384

if not SQS_QUEUE_URL:
    raise RuntimeError("SQS_QUEUE_URL environment variable is not set")

sqs = boto3.client("sqs", region_name=AWS_REGION)


def validate_job(body: object) -> IngestJob:
    if not isinstance(body, dict):
        raise ValueError("Message must be a JSON object")

    job_id = body.get("jobId")
    source_id = body.get("sourceId")
    raw_text = body.get("rawText")

    if not isinstance(job_id, str):
        raise ValueError("jobId must be a string")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("sourceId must be a non-empty string")
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("rawText must be a non-empty string")

    uuid.UUID(job_id)

    return {
        "jobId": job_id,
        "sourceId": source_id,
        "rawText": raw_text,
    }


def create_embedding(raw_text: str) -> tuple[list[float], str]:
    """Request an embedding from the existing embedding service."""
    response = httpx.post(
        f"{EMBEDDING_URL.rstrip('/')}/embed",
        json={"texts": [raw_text]},
        timeout=60.0,
    )
    response.raise_for_status()

    payload: EmbedResponse = response.json()
    embeddings = payload["embeddings"]

    if len(embeddings) != 1:
        raise ValueError("Embedding service returned an unexpected result")

    embedding = embeddings[0]

    if len(embedding) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Expected {EMBEDDING_DIMENSIONS} dimensions; "
            f"received {len(embedding)}"
        )

    if not all(
        isinstance(value, (int, float)) for value in embedding
    ):
        raise ValueError("Embedding contains a non-numeric value")

    return embedding, payload["model"]


def process_message(job: IngestJob) -> None:
    embedding, model_name = create_embedding(job["rawText"])
    embedding_json = json.dumps(embedding)

    with SessionLocal() as session:
        with session.begin():
            shard_id = session.execute(
                text("""
                    SELECT shard_id
                    FROM shards
                    WHERE centroid IS NOT NULL
                    ORDER BY centroid <-> CAST(:embedding AS vector)
                    LIMIT 1
                """),
                {"embedding": embedding_json},
            ).scalar_one_or_none()

            inserted = session.execute(
                text("""
                    INSERT INTO vectors
                        (id, job_id, source_id, shard_id, embedding, metadata)
                    VALUES
                        (:id, :job_id, :source_id, :shard_id,
                         CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                    ON CONFLICT (job_id) DO NOTHING
                """),
                {
                    "id": str(uuid.uuid4()),
                    "job_id": job["jobId"],
                    "source_id": job["sourceId"],
                    "shard_id": shard_id,
                    "embedding": embedding_json,
                    "metadata": json.dumps({"model": model_name}),
                },
            )

            if inserted.rowcount == 1 and shard_id is not None:
                session.execute(
                    text("""
                        UPDATE shards
                        SET vector_count = COALESCE(vector_count, 0) + 1
                        WHERE shard_id = :shard_id
                    """),
                    {"shard_id": shard_id},
                )


def main() -> None:
    logger.info("Starting ingestion worker in region %s", AWS_REGION)

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=10,
                WaitTimeSeconds=20,
                VisibilityTimeout=60,
            )

            for message in response.get("Messages", []):
                try:
                    job = validate_job(json.loads(message["Body"]))
                    process_message(job)

                    # Acknowledge only after the database transaction succeeds.
                    sqs.delete_message(
                        QueueUrl=SQS_QUEUE_URL,
                        ReceiptHandle=message["ReceiptHandle"],
                    )
                    logger.info("Processed job %s", job["jobId"])

                except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                    logger.exception(
                        "Invalid message; leaving it in the queue"
                    )

                except (
                    SQLAlchemyError,
                    BotoCoreError,
                    ClientError,
                    httpx.HTTPError,
                ):
                    logger.exception(
                        "Processing failed; leaving message for retry"
                    )

        except (BotoCoreError, ClientError):
            logger.exception("SQS polling failed")
            time.sleep(5)


if __name__ == "__main__":
    main()
