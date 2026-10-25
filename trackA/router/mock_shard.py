"""Mock shard: PORT, SHARD_ID, LABEL, MODE=ok|500 env vars. Distances are canned."""
import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SHARD_ID = int(os.getenv("SHARD_ID", "0"))
LABEL = os.getenv("LABEL", "leader")
MODE = os.getenv("MODE", "ok")
app = FastAPI()

class Req(BaseModel):
    query_vector: List[float]
    top_k: int

@app.get("/health")
def health():
    return {"status": "ok", "label": LABEL}

@app.post("/search")
def search(req: Req):
    if MODE == "500":
        raise HTTPException(status_code=500, detail="boom")
    ids = [f"s{SHARD_ID}-{i}" for i in range(req.top_k)]
    dists = [SHARD_ID * 0.1 + i * 0.2 for i in range(req.top_k)]
    return {"ids": ids, "distances": dists}
