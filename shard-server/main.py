import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

SHARD_ID = os.getenv("SHARD_ID", "0")
INDEX_PATH = Path(
    os.getenv("INDEX_PATH", f"indexes/shard_{SHARD_ID}.faiss")
)

index = None


class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: Annotated[int, Field(gt=0, le=1000)] = 10


class SearchResponse(BaseModel):
    ids: list[str]
    distances: list[float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index

    if not INDEX_PATH.is_file():
        raise RuntimeError(f"FAISS index file not found: {INDEX_PATH}")

    loaded_index = faiss.read_index(str(INDEX_PATH))

    if loaded_index.d <= 0:
        raise RuntimeError("FAISS index has an invalid dimension")

    index = loaded_index
    logger.info(
        "Loaded shard %s from %s: dimension=%s, vectors=%s",
        SHARD_ID,
        INDEX_PATH,
        index.d,
        index.ntotal,
    )

    yield

    index = None


app = FastAPI(
    title="GeekyClouds Shard Index Server",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
@app.get("/healthz")
def health() -> dict[str, object]:
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")

    return {
        "status": "healthy",
        "shard_id": SHARD_ID,
        "dimension": index.d,
        "vector_count": index.ntotal,
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")

    if len(request.query_vector) != index.d:
        raise HTTPException(
            status_code=422,
            detail=f"Expected query dimension {index.d}, got {len(request.query_vector)}",
        )

    if not all(np.isfinite(value) for value in request.query_vector):
        raise HTTPException(
            status_code=422,
            detail="Query vector must contain only finite numbers",
        )

    k = min(request.top_k, index.ntotal)

    if k == 0:
        return SearchResponse(ids=[], distances=[])

    query = np.asarray([request.query_vector], dtype=np.float32)
    distances, labels = index.search(query, k)

    results = [
        (str(int(label)), float(distance))
        for label, distance in zip(labels[0], distances[0])
        if label >= 0 and np.isfinite(distance)
    ]

    return SearchResponse(
        ids=[item[0] for item in results],
        distances=[item[1] for item in results],
    )
