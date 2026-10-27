
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

SHARD_URLS = [
    url.strip().rstrip("/")
    for url in os.getenv("SHARD_URLS", "").split(",")
    if url.strip()
]
SHARD_TIMEOUT_SECONDS = float(os.getenv("SHARD_TIMEOUT_SECONDS", "2.0"))

client: httpx.AsyncClient | None = None


class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: int = Field(default=10, gt=0, le=1000)


class SearchResponse(BaseModel):
    ids: list[str]
    distances: list[float]
    latency_ms: float
    partial: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(timeout=SHARD_TIMEOUT_SECONDS)
    yield
    await client.aclose()
    client = None


app = FastAPI(
    title="GeekyClouds Query Router",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
@app.get("/healthz")
def health():
    return {
        "status": "healthy",
        "configured_shards": len(SHARD_URLS),
    }


async def search_shard(url: str, request: SearchRequest):
    try:
        response = await client.post(
            f"{url}/search",
            json={
                "query_vector": request.query_vector,
                "top_k": request.top_k,
            },
        )
        response.raise_for_status()
        data = response.json()

        ids = data.get("ids", [])
        distances = data.get("distances", [])

        if len(ids) != len(distances):
            raise ValueError("Shard returned mismatched IDs and distances")

        results = []
        for item_id, distance in zip(ids, distances):
            distance = float(distance)
            if np.isfinite(distance):
                results.append((str(item_id), distance))

        return results, False

    except (httpx.HTTPError, ValueError, TypeError) as exc:
        logger.warning("Shard %s failed: %s", url, exc)
        return [], True


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if not SHARD_URLS:
        raise HTTPException(
            status_code=503,
            detail="No shard URLs configured. Set SHARD_URLS.",
        )

    if not all(np.isfinite(value) for value in request.query_vector):
        raise HTTPException(
            status_code=422,
            detail="Query vector must contain only finite numbers",
        )

    started = time.perf_counter()

    outcomes = await asyncio.gather(
        *(search_shard(url, request) for url in SHARD_URLS)
    )

    all_results = []
    failed_shards = 0

    for results, failed in outcomes:
        all_results.extend(results)
        failed_shards += int(failed)

    all_results.sort(key=lambda result: result[1])

    return SearchResponse(
        ids=[item_id for item_id, _ in all_results[:request.top_k]],
        distances=[distance for _, distance in all_results[:request.top_k]],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        partial=failed_shards > 0,
    )

