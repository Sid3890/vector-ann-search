"""Serve one real shard locally (replacement for mock_shard.py).

  SHARD_ID=0 PORT=8001 uvicorn shard_server.local_shard:app --port 8001

Loads indexes/shard_<SHARD_ID>.faiss. If indexes/shard_<SHARD_ID>_ids.npy exists, it maps
local FAISS positions -> global ids; otherwise raw FAISS labels are returned.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("local_shard")
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "indexes"))
EF_SEARCH = int(os.environ.get("EF_SEARCH", "64"))

app = FastAPI(title=f"shard-{SHARD_ID}")
index: Optional[faiss.Index] = None
id_map: Optional[np.ndarray] = None


class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: int = 10


class SearchResponse(BaseModel):
    ids: list[str]
    distances: list[float]


@app.on_event("startup")
def load() -> None:
    global index, id_map
    path = INDEX_DIR / f"shard_{SHARD_ID}.faiss"
    if not path.exists():
        raise RuntimeError(f"Missing {path}; run sharding/shard_partitioner.py first")
    index = faiss.read_index(str(path))
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = EF_SEARCH
    id_map = None
    for name in (f"shard_{SHARD_ID}_ids.npy", f"shard_{SHARD_ID}.ids.npy"):
        if (INDEX_DIR / name).exists():
            id_map = np.load(INDEX_DIR / name)
            break
    log.warning("shard %d loaded: ntotal=%d id_map=%s", SHARD_ID, index.ntotal, id_map is not None)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok" if index is not None else "loading", "shard_id": SHARD_ID}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if index is None:
        raise HTTPException(503, "index not loaded")
    if len(req.query_vector) != index.d:
        raise HTTPException(422, f"expected dim {index.d}, got {len(req.query_vector)}")
    q = np.asarray(req.query_vector, dtype="float32")[None, :]
    dist, lab = index.search(q, req.top_k)
    pairs = [(int(l), float(d)) for l, d in zip(lab[0], dist[0]) if l >= 0]
    ids = [str(id_map[l]) if id_map is not None else str(l) for l, _ in pairs]
    return SearchResponse(ids=ids, distances=[d for _, d in pairs])
