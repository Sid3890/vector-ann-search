#!/usr/bin/env python3
"""Benchmark the GeekyClouds vector set against a Pinecone serverless index.

The client deliberately uses the same base vectors and query vectors as the local
FAISS benchmarks.  It is safe to rerun: the serverless index is created only when
missing and upserts use stable vector IDs, so a rerun replaces the same records.

Required environment variable:
    PINECONE_API_KEY

Optional environment variables:
    PINECONE_INDEX             Index name (default: geekyclouds-managed)
    PINECONE_CLOUD             Serverless cloud (default: aws)
    PINECONE_REGION            Serverless region (default: us-east-1)
    PINECONE_NAMESPACE         Namespace (default: geekyclouds-benchmark)
    PINECONE_COST_PER_QUERY_USD  Cost estimate written when the CSV has that column

Example:
    python benchmark/managed_client.py --base emb.npy --queries data/queries.npy
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

LOG = logging.getLogger("managed_client")
DEFAULT_DIMENSION = 384
DEFAULT_TOP_K = 10
DEFAULT_BATCH_SIZE = 100
DEFAULT_RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "benchmarks.csv"


@dataclass(frozen=True)
class BenchmarkResult:
    """Managed benchmark values aligned with results/benchmarks.csv."""

    index_type: str
    recall_at_10: float
    p50_ms: float
    p95_ms: float
    build_time_s: float
    memory_mb: float


@dataclass(frozen=True)
class Config:
    """Validated runtime configuration; secrets are intentionally not stored here."""

    index_name: str
    cloud: str
    region: str
    namespace: str
    batch_size: int
    top_k: int
    ready_timeout_s: float
    poll_interval_s: float


def _load_matrix(path: Path, label: str) -> np.ndarray:
    """Load a finite 2-D float32 matrix and provide actionable validation errors."""
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load {label} vectors from {path}: {exc}") from exc
    if values.ndim != 2:
        raise ValueError(f"{label} vectors must be 2-D, got shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} vectors contain NaN or infinity")
    return np.ascontiguousarray(values, dtype=np.float32)


def _load_ids(path: Path | None, count: int) -> list[str]:
    """Load stable Pinecone IDs, or use row numbers when no ID file is supplied."""
    if path is None:
        return [str(i) for i in range(count)]
    try:
        raw = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load vector IDs from {path}: {exc}") from exc
    if raw.ndim != 1 or len(raw) != count:
        raise ValueError(f"IDs must be a 1-D array with {count} values, got {raw.shape}")
    ids = [str(value) for value in raw.tolist()]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Vector IDs must be non-empty and unique")
    return ids


def _import_pinecone() -> tuple[Any, Any]:
    """Import the current Pinecone SDK lazily so offline validation still works."""
    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError as exc:
        raise RuntimeError("Install the Pinecone SDK first: pip install pinecone") from exc
    return Pinecone, ServerlessSpec


def _index_names(client: Any) -> set[str]:
    """Return index names across Pinecone SDK response versions."""
    response = client.list_indexes()
    names: set[str] = set()
    if isinstance(response, Mapping):
        items = response.get("indexes", [])
    elif callable(getattr(response, "names", None)):
        items = response.names()
    else:
        items = response
    for item in items or []:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, Mapping):
            name = item.get("name")
            if name:
                names.add(str(name))
        else:
            name = getattr(item, "name", None)
            if name:
                names.add(str(name))
    return names


def _status_ready(description: Any) -> bool:
    status = description.get("status") if isinstance(description, Mapping) else getattr(description, "status", None)
    if isinstance(status, Mapping):
        return bool(status.get("ready", False))
    return bool(getattr(status, "ready", False))


def ensure_serverless_index(client: Any, spec: Any, config: Config, dimension: int) -> None:
    """Create the index once and wait until it is ready for upserts."""
    if config.index_name not in _index_names(client):
        LOG.info("Creating Pinecone serverless index %s (%d dimensions, L2/euclidean)", config.index_name, dimension)
        # Pinecone names its L2 distance metric ``euclidean``; this is the same
        # squared-L2 ordering used by faiss.IndexFlatL2.
        client.create_index(
            name=config.index_name,
            dimension=dimension,
            metric="euclidean",
            spec=spec(cloud=config.cloud, region=config.region),
        )
    deadline = time.monotonic() + config.ready_timeout_s
    while True:
        description = client.describe_index(config.index_name)
        if _status_ready(description):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Pinecone index {config.index_name!r} was not ready within {config.ready_timeout_s}s")
        time.sleep(config.poll_interval_s)


def batched(values: Sequence[tuple[str, list[float]]], size: int) -> Iterable[Sequence[tuple[str, list[float]]]]:
    """Yield bounded upsert batches."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def upsert_vectors(index: Any, vectors: np.ndarray, ids: Sequence[str], namespace: str, batch_size: int) -> None:
    records = [(vector_id, vector.tolist()) for vector_id, vector in zip(ids, vectors, strict=True)]
    for batch_number, batch in enumerate(batched(records, batch_size), start=1):
        index.upsert(vectors=[{"id": vector_id, "values": values} for vector_id, values in batch], namespace=namespace)
        LOG.info("Upserted batch %d (%d vectors)", batch_number, len(batch))


def _match_ids(response: Any) -> list[str]:
    matches = response.get("matches", []) if isinstance(response, Mapping) else getattr(response, "matches", [])
    result: list[str] = []
    for match in matches or []:
        value = match.get("id") if isinstance(match, Mapping) else getattr(match, "id", None)
        if value is not None:
            result.append(str(value))
    return result


def _ground_truth(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("FAISS is required to compute exact recall ground truth") from exc
    index = faiss.IndexFlatL2(base.shape[1])
    index.add(base)
    _, ids = index.search(queries, k)
    return ids


def _recall(predicted: Sequence[Sequence[str]], ids: Sequence[str], ground_truth: np.ndarray, k: int) -> float:
    expected = [{ids[int(row)] for row in query[:k]} for query in ground_truth]
    hits = sum(len(set(result[:k]) & truth) for result, truth in zip(predicted, expected, strict=True))
    return hits / (len(expected) * k) if expected else 0.0


def _memory_mb() -> float:
    """Return process RSS without making psutil mandatory for the benchmark."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        if sys.platform == "win32":
            import ctypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("page_fault_count", ctypes.c_ulong),
                            ("peak_working_set_size", ctypes.c_size_t), ("working_set_size", ctypes.c_size_t),
                            ("quota_peak_paged_pool_usage", ctypes.c_size_t), ("quota_paged_pool_usage", ctypes.c_size_t),
                            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t), ("quota_non_paged_pool_usage", ctypes.c_size_t),
                            ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t)]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            get_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_info(handle, ctypes.byref(counters), counters.cb)
            return counters.working_set_size / (1024 * 1024)
        return 0.0


def _append_csv(result: BenchmarkResult, csv_path: Path, cost_per_query: float | None) -> None:
    """Append MANAGED while retaining the repository's existing CSV columns."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing_header: list[str] = []
    if csv_path.exists() and csv_path.stat().st_size:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), [])
    fieldnames = existing_header or ["index_type", "recall_at_10", "p50_ms", "p95_ms", "build_time_s", "memory_mb"]
    row: dict[str, object] = {
        "index_type": result.index_type,
        "recall_at_10": result.recall_at_10,
        "p50_ms": result.p50_ms,
        "p95_ms": result.p95_ms,
        "build_time_s": result.build_time_s,
        "memory_mb": result.memory_mb,
    }
    if "cost_per_query_usd" in fieldnames and cost_per_query is not None:
        row["cost_per_query_usd"] = cost_per_query
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not existing_header:
            writer.writeheader()
        writer.writerow(row)


def benchmark(index: Any, base: np.ndarray, queries: np.ndarray, ids: Sequence[str], namespace: str, k: int) -> BenchmarkResult:
    ground_truth = _ground_truth(base, queries, k)
    predictions: list[list[str]] = []
    latencies: list[float] = []
    for query_number, query in enumerate(queries, start=1):
        started = time.perf_counter()
        response = index.query(vector=query.tolist(), top_k=k, namespace=namespace, include_values=False)
        predictions.append(_match_ids(response))
        latencies.append((time.perf_counter() - started) * 1000.0)
        if query_number % 100 == 0 or query_number == len(queries):
            LOG.info("Queried %d/%d vectors", query_number, len(queries))
    memory_mb = _memory_mb()
    return BenchmarkResult("MANAGED", round(_recall(predictions, ids, ground_truth, k), 4), round(float(np.percentile(latencies, 50)), 4), round(float(np.percentile(latencies, 95)), 4), 0.0, round(memory_mb, 2))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("emb.npy"))
    parser.add_argument("--queries", type=Path, default=Path("data/queries.npy"))
    parser.add_argument("--ids", type=Path, default=Path("ids.npy"))
    parser.add_argument("--index-name", default=os.environ.get("PINECONE_INDEX", "geekyclouds-managed"))
    parser.add_argument("--cloud", default=os.environ.get("PINECONE_CLOUD", "aws"))
    parser.add_argument("--region", default=os.environ.get("PINECONE_REGION", "us-east-1"))
    parser.add_argument("--namespace", default=os.environ.get("PINECONE_NAMESPACE", "geekyclouds-benchmark"))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--cost-per-query-usd", type=float, default=None, help="Optional provider cost estimate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    try:
        api_key = os.environ["PINECONE_API_KEY"]
        base = _load_matrix(args.base, "base")
        queries = _load_matrix(args.queries, "query")
        if base.shape[1] != DEFAULT_DIMENSION:
            raise ValueError(f"Pinecone managed benchmark requires dimension {DEFAULT_DIMENSION}, got {base.shape[1]}")
        if queries.shape[1] != base.shape[1]:
            raise ValueError("Base and query dimensions must match")
        if args.top_k < 1 or args.top_k > len(base):
            raise ValueError("--top-k must be between 1 and the number of base vectors")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        ids = _load_ids(args.ids if args.ids.exists() else None, len(base))
        config = Config(args.index_name, args.cloud, args.region, args.namespace, args.batch_size, args.top_k, args.ready_timeout_s, args.poll_interval_s)
        Pinecone, ServerlessSpec = _import_pinecone()
        client = Pinecone(api_key=api_key)
        started = time.perf_counter()
        ensure_serverless_index(client, ServerlessSpec, config, base.shape[1])
        index = client.Index(config.index_name)
        upsert_vectors(index, base, ids, config.namespace, config.batch_size)
        result = benchmark(index, base, queries, ids, config.namespace, config.top_k)
        result = BenchmarkResult(result.index_type, result.recall_at_10, result.p50_ms, result.p95_ms, round(time.perf_counter() - started, 4), result.memory_mb)
        cost = args.cost_per_query_usd
        if cost is None and os.environ.get("PINECONE_COST_PER_QUERY_USD"):
            cost = float(os.environ["PINECONE_COST_PER_QUERY_USD"])
        _append_csv(result, args.csv, cost)
        LOG.info("MANAGED: recall@%d=%.4f p50=%.2fms p95=%.2fms build=%.2fs memory=%.2fMB", config.top_k, result.recall_at_10, result.p50_ms, result.p95_ms, result.build_time_s, result.memory_mb)
        LOG.info("Appended MANAGED row to %s", args.csv)
        return 0
    except (KeyError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        LOG.error("Managed benchmark failed: %s", exc)
        return 1
    except Exception:
        LOG.exception("Unexpected managed benchmark failure")
        return 1


if __name__ == "__main__":
    sys.exit(main())
