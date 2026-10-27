"""Self-contained mock shard for failover_demo.py (env: SHARD_ID, LABEL)."""
import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

SHARD_ID = int(os.getenv("SHARD_ID", "0"))
LABEL = os.getenv("LABEL", "replica")
app = FastAPI()

class Req(BaseModel):
    query_vector: List[float]
    top_k: int

@app.get("/health")
def health():
    return {"status": "ok", "label": LABEL}

@app.post("/search")
def search(req: Req):
    return {"ids": [f"s{SHARD_ID}-{i}" for i in range(req.top_k)],
            "distances": [SHARD_ID * 0.1 + i * 0.2 for i in range(req.top_k)]}
