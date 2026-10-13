"""
GeekyClouds — ANN Benchmarking Core
indexing/benchmark.py

Reusable benchmarking harness shared by every index type in this project
(FLAT, IVF, HNSW, PQ, MANAGED). Computes recall, latency, build time,
process memory, and optional serialized index-size/compression metrics.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil

try:
    from .baseline import DEFAULT_SEED, build_flat_index, load_dataset, search_flat
except ImportError:
    from baseline import DEFAULT_SEED, build_flat_index, load_dataset, search_flat

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV = RESULTS_DIR / "benchmarks.csv"

CSV_COLUMNS = [
    "index_type",
    "recall_at_10",
    "p50_ms",
    "p95_ms",
    "build_time_s",
    "memory_mb",
    "raw_bytes",
    "pq_bytes",
    "compression_ratio",
]


@dataclass
class BenchmarkResult:
    index_type: str
    recall_at_10: float
    p50_ms: float
    p95_ms: float
    build_time_s: float
    memory_mb: float
    raw_bytes: int | None = None
    pq_bytes: int | None = None
    compression_ratio: float | None = None


def get_memory_mb() -> float:
    """Current process resident memory (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def compute_recall_at_k(
    retrieved_ids: np.ndarray, ground_truth_ids: np.ndarray, k: int = 10
) -> float:
    """Mean recall@k across all queries."""
    if retrieved_ids.shape[0] != ground_truth_ids.shape[0]:
        raise ValueError("retrieved_ids and ground_truth_ids must have the same number of queries")

    nq = retrieved_ids.shape[0]
    total_hits = 0
    for i in range(nq):
        retrieved_set = set(retrieved_ids[i, :k].tolist())
        truth_set = set(ground_truth_ids[i, :k].tolist())
        total_hits += len(retrieved_set & truth_set)
    return total_hits / (nq * k)


def measure_query_latencies_ms(index, xq: np.ndarray, k: int = 10) -> np.ndarray:
    """Measure one search call per query and return per-query milliseconds."""
    if xq.dtype != np.float32:
        xq = xq.astype(np.float32)

    latencies_ms = np.empty(xq.shape[0], dtype=np.float64)
    for i in range(xq.shape[0]):
        query = xq[i : i + 1]
        t0 = time.perf_counter()
        index.search(query, k)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000.0
    return latencies_ms


def run_benchmark(
    index_type: str,
    index,
    xq: np.ndarray,
    ground_truth_ids: np.ndarray,
    build_time_s: float,
    k: int = 10,
    csv_path: str | Path = RESULTS_CSV,
    raw_bytes: int | None = None,
    pq_bytes: int | None = None,
) -> BenchmarkResult:
    """Benchmark an index and append one row to the results CSV.

    ``raw_bytes`` and ``pq_bytes`` are optional so existing IVF/HNSW/FLAT
    callers remain compatible. When both are provided, the compression ratio
    is calculated as ``raw_bytes / pq_bytes``.
    """
    _, retrieved_ids = search_flat(index, xq, k=k)
    latencies_ms = measure_query_latencies_ms(index, xq, k=k)
    recall = compute_recall_at_k(retrieved_ids, ground_truth_ids, k=k)
    memory_mb = get_memory_mb()
    compression_ratio = (
        raw_bytes / pq_bytes
        if raw_bytes is not None and pq_bytes is not None and pq_bytes > 0
        else None
    )

    result = BenchmarkResult(
        index_type=index_type,
        recall_at_10=round(recall, 4),
        p50_ms=round(float(np.percentile(latencies_ms, 50)), 4),
        p95_ms=round(float(np.percentile(latencies_ms, 95)), 4),
        build_time_s=round(build_time_s, 4),
        memory_mb=round(memory_mb, 2),
        raw_bytes=raw_bytes,
        pq_bytes=pq_bytes,
        compression_ratio=(round(compression_ratio, 4) if compression_ratio is not None else None),
    )
    append_result_csv(result, csv_path=csv_path)
    return result


def append_result_csv(result: BenchmarkResult, csv_path: str | Path = RESULTS_CSV) -> None:
    """Append a result, migrating the legacy six-column CSV if necessary."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_row = asdict(result)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(result_row)
        return

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        existing_columns = reader.fieldnames or []

    # Day 2/3 produced a six-column file. Rewrite it once so new PQ rows do
    # not create malformed rows with more values than the old header allows.
    if existing_columns != CSV_COLUMNS:
        temp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with temp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
            writer.writerow(result_row)
        os.replace(temp_path, csv_path)
        return

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(result_row)


if __name__ == "__main__":
    print("Loading dataset ...")
    ds = load_dataset(seed=DEFAULT_SEED)
    print("Building FlatL2 ground-truth index ...")
    t0 = time.perf_counter()
    flat_index = build_flat_index(ds.xb)
    build_time = time.perf_counter() - t0
    print("Computing exact ground truth ...")
    _, ground_truth_ids = search_flat(flat_index, ds.xq, k=10)
    result = run_benchmark(
        index_type="FLAT",
        index=flat_index,
        xq=ds.xq,
        ground_truth_ids=ground_truth_ids,
        build_time_s=build_time,
        k=10,
    )
    print(result)
    print(f"Appended to {RESULTS_CSV}")
