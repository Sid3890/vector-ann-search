"""GeekyClouds query router: scatter-gather with leader/follower failover.

Day 6: POST /search fans out to every shard and merges the partial top-k lists.
Day 7: each shard has an ordered replica list [leader, follower1, follower2];
       the router tries the leader first, fails over to the next replica on
       timeout/5xx, polls every replica's /health every 10s, and skips replicas
       known to be unhealthy without waiting for a live timeout.

Configuration (environment variables, nothing secret):
  SHARD_REPLICAS_JSON   {"0": ["http://shard-0-a:8000", "http://shard-0-b:8000"], ...}
  SHARD_CONFIG_PATH     path to a JSON file with the same shape (used if the var above is unset)
  SHARD_URLS            compact form "0=http://a|http://b,1=http://c" ('|' separates replicas, leader first)
  SHARD_TIMEOUT_S       total per-shard budget across all replicas (default 2.0)
  NUM_SHARDS            fallback only: builds http://shard-<id>:8000 (single replica), default 4
  VECTOR_DIM            expected query dimensionality, default 128 (0 disables the check)
  HEALTH_INTERVAL_S     default 10
  HEALTH_TIMEOUT_S      default 1.0
  ATTEMPT_TIMEOUT_S     per-replica attempt budget, default 40% of the shard budget
  SHARD_DEADLINE_S      alias of SHARD_TIMEOUT_S
  LOG_LEVEL             default INFO
"""
from __future__ import annotations

import asyncio
import heapq
import json
import logging
import math
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Callable, Deque, Dict, List, Optional

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("geekyclouds.router")

VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "128"))


# --------------------------------------------------------------------------- #
# Contracts (Python mirror of the shared TypeScript interfaces)
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    query_vector: List[float]
    top_k: int = Field(default=10, ge=1, le=1000)

    @field_validator("query_vector")
    @classmethod
    def _check_dim(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("query_vector must not be empty")
        if VECTOR_DIM and len(v) != VECTOR_DIM:
            raise ValueError(f"query_vector must have {VECTOR_DIM} dims, got {len(v)}")
        return v


class ShardResponse(BaseModel):
    """What a shard index server returns from POST /search."""

    ids: List[str]
    distances: List[float]

    @field_validator("ids", mode="before")
    @classmethod
    def _stringify_ids(cls, v: object) -> object:
        # FAISS ids often arrive as ints; the contract says string.
        if isinstance(v, list):
            return [str(i) for i in v]
        return v

    @field_validator("distances")
    @classmethod
    def _same_length(cls, v: List[float], info) -> List[float]:  # type: ignore[no-untyped-def]
        ids = info.data.get("ids")
        if ids is not None and len(ids) != len(v):
            raise ValueError("ids and distances differ in length")
        return v

    @model_validator(mode="after")
    def _drop_faiss_padding(self) -> "ShardResponse":
        # FAISS pads short result lists with id -1 and distance ~FLT_MAX (3.4e38).
        keep = [
            (i, d)
            for i, d in zip(self.ids, self.distances)
            if i != "-1" and math.isfinite(d) and d < 1e38
        ]
        self.ids = [i for i, _ in keep]
        self.distances = [d for _, d in keep]
        return self


class QueryResponse(BaseModel):
    ids: List[str]
    distances: List[float]
    latency_ms: float
    partial: bool


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_shard_config() -> Dict[int, List[str]]:
    """Return {shard_id: [leader, follower1, ...]}. Fails fast on bad config."""
    raw = os.getenv("SHARD_REPLICAS_JSON")
    path = os.getenv("SHARD_CONFIG_PATH")
    if raw:
        data = json.loads(raw)
    elif path:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    elif os.getenv("SHARD_URLS"):
        data = {}
        for part in os.environ["SHARD_URLS"].split(","):
            sid, _, urls = part.strip().partition("=")
            data[sid.strip()] = [u.strip() for u in urls.split("|") if u.strip()]
    else:
        n = int(os.getenv("NUM_SHARDS", "4"))
        data = {str(i): [f"http://shard-{i}:8000"] for i in range(n)}

    config: Dict[int, List[str]] = {}
    for key, urls in data.items():
        if not isinstance(urls, list) or not urls:
            raise RuntimeError(f"shard {key}: replica list must be a non-empty list")
        config[int(key)] = [u.rstrip("/") for u in urls]
    if not config:
        raise RuntimeError("no shards configured")
    return config


# --------------------------------------------------------------------------- #
# Health tracking
# --------------------------------------------------------------------------- #
@dataclass
class ReplicaState:
    url: str
    healthy: bool = True
    last_checked: Optional[float] = None
    last_error: Optional[str] = None


class HealthRegistry:
    """Tracks replica health. Active: poll /health. Passive: live failures."""

    def __init__(self, config: Dict[int, List[str]], health_timeout_s: float = 1.0) -> None:
        self._health_timeout_s = health_timeout_s
        self._states: Dict[str, ReplicaState] = {
            url: ReplicaState(url) for urls in config.values() for url in urls
        }

    def is_healthy(self, url: str) -> bool:
        return self._states[url].healthy

    def mark_unhealthy(self, url: str, reason: str) -> None:
        st = self._states[url]
        if st.healthy:
            logger.warning("replica DOWN url=%s reason=%s", url, reason)
        st.healthy = False
        st.last_error = reason

    def mark_healthy(self, url: str) -> None:
        st = self._states[url]
        if not st.healthy:
            logger.info("replica UP url=%s", url)
        st.healthy = True
        st.last_error = None

    async def _probe(self, client: httpx.AsyncClient, url: str) -> None:
        st = self._states[url]
        try:
            resp = await client.get(f"{url}/health", timeout=self._health_timeout_s)
            if resp.status_code == 200:
                self.mark_healthy(url)
            else:
                self.mark_unhealthy(url, f"health HTTP {resp.status_code}")
        except httpx.HTTPError as exc:
            self.mark_unhealthy(url, f"health {type(exc).__name__}")
        finally:
            st.last_checked = time.time()

    async def poll_once(self, client: httpx.AsyncClient) -> None:
        await asyncio.gather(*(self._probe(client, u) for u in self._states))

    async def run(self, get_client: Callable[[], httpx.AsyncClient], interval_s: float) -> None:
        while True:
            try:
                await asyncio.sleep(interval_s)
                await self.poll_once(get_client())
            except asyncio.CancelledError:
                raise
            except Exception:  # never let the poller die
                logger.exception("health poll cycle failed")

    def snapshot(self) -> List[Dict[str, object]]:
        return [asdict(s) for s in self._states.values()]


# --------------------------------------------------------------------------- #
# Replica-aware shard client
# --------------------------------------------------------------------------- #
class ReplicaFailure(Exception):
    """Retryable: timeout, connection error, 5xx, or malformed response."""


class ShardRequestError(Exception):
    """Non-retryable: the shard rejected the request (4xx). Another replica would too."""


@dataclass
class FailoverEvent:
    ts: float
    shard_id: int
    failed_replica: str
    recovered_replica: Optional[str]  # None => no replica served the shard
    reason: str


class ReplicatedShardClient:
    def __init__(
        self,
        config: Dict[int, List[str]],
        registry: HealthRegistry,
        get_client: Callable[[], httpx.AsyncClient],
        deadline_s: float,
        attempt_timeout_s: float,
    ) -> None:
        self._config = config
        self._registry = registry
        self._get_client = get_client  # looked up per call so the client can be swapped
        self._deadline_s = deadline_s
        self._attempt_timeout_s = attempt_timeout_s
        self.events: Deque[FailoverEvent] = deque(maxlen=200)

    async def _attempt(self, url: str, payload: Dict[str, object]) -> ShardResponse:
        try:
            async with asyncio.timeout(self._attempt_timeout_s):
                resp = await self._get_client().post(f"{url}/search", json=payload)
        except TimeoutError as exc:
            raise ReplicaFailure("timeout") from exc
        except httpx.HTTPError as exc:
            raise ReplicaFailure(type(exc).__name__) from exc

        if resp.status_code >= 500:
            raise ReplicaFailure(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            raise ShardRequestError(f"HTTP {resp.status_code}")
        try:
            return ShardResponse.model_validate(resp.json())
        except ValueError as exc:  # JSON decode error or pydantic ValidationError
            raise ReplicaFailure("malformed response") from exc

    def _record(
        self, shard_id: int, failed: str, recovered: Optional[str], reason: str
    ) -> None:
        self.events.append(FailoverEvent(time.time(), shard_id, failed, recovered, reason))
        logger.warning(
            "FAILOVER shard_id=%s failed_replica=%s recovered_replica=%s reason=%s",
            shard_id, failed, recovered, reason,
        )

    async def query_shard(
        self, shard_id: int, payload: Dict[str, object]
    ) -> Optional[ShardResponse]:
        """Leader first, then followers. Returns None if the shard can't be served."""
        replicas = self._config[shard_id]
        healthy = [u for u in replicas if self._registry.is_healthy(u)]
        unhealthy = [u for u in replicas if not self._registry.is_healthy(u)]
        # Known-unhealthy replicas are skipped, but kept as a last resort in case
        # the health data is stale and they have actually recovered.
        failed: Dict[str, str] = {u: "known_unhealthy" for u in unhealthy}

        current: Optional[str] = None
        try:
            async with asyncio.timeout(self._deadline_s):
                for url in healthy + unhealthy:
                    current = url
                    try:
                        result = await self._attempt(url, payload)
                    except ReplicaFailure as exc:
                        failed[url] = str(exc)
                        self._registry.mark_unhealthy(url, f"live {exc}")
                        continue
                    except ShardRequestError as exc:
                        logger.error("shard_id=%s replica=%s rejected request: %s", shard_id, url, exc)
                        return None
                    failed.pop(url, None)
                    self._registry.mark_healthy(url)
                    for bad, reason in failed.items():
                        self._record(shard_id, bad, url, reason)
                    return result
        except TimeoutError:
            logger.error("shard_id=%s exceeded %.2fs deadline", shard_id, self._deadline_s)
            if current is not None:  # the replica we were waiting on counts as failed
                failed[current] = "timeout"
                self._registry.mark_unhealthy(current, "live timeout (shard deadline)")

        for bad, reason in failed.items():
            self._record(shard_id, bad, None, reason)
        logger.error("shard_id=%s UNAVAILABLE: all replicas failed", shard_id)
        return None


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # All tunables are read here (not at import) so tests/deploys can set them first.
    config = load_shard_config()
    deadline_s = _env_float("SHARD_TIMEOUT_S", _env_float("SHARD_DEADLINE_S", 2.0))
    attempt_s = _env_float("ATTEMPT_TIMEOUT_S", deadline_s * 0.4)
    interval_s = _env_float("HEALTH_INTERVAL_S", 10.0)
    registry = HealthRegistry(config, _env_float("HEALTH_TIMEOUT_S", 1.0))

    http = httpx.AsyncClient(
        timeout=attempt_s,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )
    app.state.client = http
    app.state.registry = registry
    app.state.config = config
    app.state.shards = ReplicatedShardClient(
        config, registry, lambda: app.state.client, deadline_s, attempt_s
    )
    # Replicas start as "healthy"; the first poll happens after one interval and
    # live failures mark replicas down immediately in the meantime.
    poller = asyncio.create_task(registry.run(lambda: app.state.client, interval_s))
    logger.info("router up: %d shards, replicas=%s, deadline=%.2fs", len(config), config, deadline_s)
    try:
        yield
    finally:
        poller.cancel()
        await asyncio.gather(poller, return_exceptions=True)
        await http.aclose()


app = FastAPI(title="GeekyClouds Query Router", lifespan=lifespan)


def merge_top_k(results: List[ShardResponse], top_k: int) -> QueryResponse:
    """Global top-k by ascending L2 distance; duplicate ids keep their best distance."""
    best: Dict[str, float] = {}
    for r in results:
        for i, d in zip(r.ids, r.distances):
            if i not in best or d < best[i]:
                best[i] = d
    top = heapq.nsmallest(top_k, best.items(), key=lambda kv: (kv[1], kv[0]))
    return QueryResponse(
        ids=[i for i, _ in top], distances=[d for _, d in top], latency_ms=0.0, partial=False
    )


@app.post("/search", response_model=QueryResponse)
async def search(req: QueryRequest, response: Response):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    shards: ReplicatedShardClient = app.state.shards
    shard_ids = sorted(app.state.config)
    payload: Dict[str, object] = {"query_vector": req.query_vector, "top_k": req.top_k}

    outcomes = await asyncio.gather(
        *(shards.query_shard(sid, payload) for sid in shard_ids), return_exceptions=True
    )
    good: List[ShardResponse] = []
    failed_ids: List[int] = []
    for sid, out in zip(shard_ids, outcomes):
        if isinstance(out, ShardResponse):
            good.append(out)
        else:
            failed_ids.append(sid)
            if isinstance(out, BaseException):
                logger.error("shard_id=%s unexpected error: %r", sid, out)
    failed_hdr = {"X-Failed-Shards": ",".join(map(str, failed_ids))} if failed_ids else {}

    if not good:
        return JSONResponse(
            status_code=503,
            content={"error": "no shard available", "partial": True},
            headers=failed_hdr,
        )

    resp = merge_top_k(good, req.top_k)
    resp.partial = bool(failed_ids)
    resp.latency_ms = (time.perf_counter() - start) * 1000.0
    for k, v in failed_hdr.items():
        response.headers[k] = v
    return resp


@app.get("/healthz")
@app.get("/health")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}  # liveness of the router itself (ALB target group)


@app.get("/replicas")
async def replicas() -> Dict[str, object]:
    return {"config": app.state.config, "state": app.state.registry.snapshot()}


@app.get("/failovers")
async def failovers() -> List[Dict[str, object]]:
    return [asdict(e) for e in app.state.shards.events]
