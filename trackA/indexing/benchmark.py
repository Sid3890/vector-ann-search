"""
GeekyClouds — ANN Benchmarking Core
indexing/benchmark.py

Reusable benchmarking harness shared by every index type in this project
(FLAT, IVF, HNSW, PQ, MANAGED). Computes:

  - recall@10           (retrieved ids vs. exact FlatL2 ground truth)
  - p50 / p95 query latency (ms), measured per-query so percentiles are
    accurate rather than averaged
  - index build time (s)  — timed by the caller around index construction,
    passed in here
  - process memory footprint (MB) — via psutil, sampled after build/search

Every run is appended as one row to results/benchmarks.csv, so later
build/index scripts (IVF, HNSW, PQ, managed-service benchmarks) can all
call `run_benchmark(...)` and land in the same comparison table used for
the final report.

Idempotent: safe to import/run repeatedly — the CSV header is written
once, and each call only appends its own row (no overwrite, no dedup
needed since every run is a legitimate new data point).
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import psutil

from baseline import add_dataset_args, build_flat_index, dataset_from_args, search_flat

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_CSV = RESULTS_DIR / "benchmarks.csv"

CSV_COLUMNS = [
    "index_type",
    "recall_at_10",
    "p50_ms",
    "p95_ms",
    "build_time_s",
    "memory_mb",
]


@dataclass
class BenchmarkResult:
    index_type: str
    recall_at_10: float
    p50_ms: float
    p95_ms: float
    build_time_s: float
    memory_mb: float


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def get_memory_mb() -> float:
    """Current process resident memory (RSS) in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def compute_recall_at_k(
    retrieved_ids: np.ndarray, ground_truth_ids: np.ndarray, k: int = 10
) -> float:
    """Mean recall@k across all queries.

    retrieved_ids / ground_truth_ids: (nq, >=k) int arrays of neighbor ids
    from the index under test and the exact FlatL2 baseline, respectively.
    For each query, recall = |retrieved_top_k ∩ true_top_k| / k, averaged
    over queries.
    """
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
    """Per-query search latency in milliseconds (one search() call per query,
    so p50/p95 reflect real per-request tail latency rather than a batched
    average).
    """
    if xq.dtype != np.float32:
        xq = xq.astype(np.float32)

    latencies_ms = np.empty(xq.shape[0], dtype=np.float64)
    for i in range(xq.shape[0]):
        query = xq[i : i + 1]
        t0 = time.perf_counter()
        index.search(query, k)
        latencies_ms[i] = (time.perf_counter() - t0) * 1000.0
    return latencies_ms


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def run_benchmark(
    index_type: str,
    index,
    xq: np.ndarray,
    ground_truth_ids: np.ndarray,
    build_time_s: float,
    k: int = 10,
    csv_path: str | Path = RESULTS_CSV,
) -> BenchmarkResult:
    """Benchmark a built index and append the result row to the CSV.

    Args:
        index_type: label for results/benchmarks.csv, e.g. "FLAT", "IVF",
            "HNSW", "PQ", "MANAGED".
        index: any object exposing `.search(xq, k) -> (distances, ids)`
            (every FAISS index type satisfies this).
        xq: (nq, dim) float32 query vectors.
        ground_truth_ids: (nq, >=k) exact neighbor ids from the FlatL2
            baseline, used to score recall@10.
        build_time_s: index construction time, timed by the caller around
            the index-building step (kept outside this function since
            build strategy differs per index type).
        k: neighbors per query (recall@10 default per project spec).
        csv_path: where to append the result row.
    """
    _, retrieved_ids = search_flat(index, xq, k=k)

    latencies_ms = measure_query_latencies_ms(index, xq, k=k)
    recall = compute_recall_at_k(retrieved_ids, ground_truth_ids, k=k)
    memory_mb = get_memory_mb()

    result = BenchmarkResult(
        index_type=index_type,
        recall_at_10=round(recall, 4),
        p50_ms=round(float(np.percentile(latencies_ms, 50)), 4),
        p95_ms=round(float(np.percentile(latencies_ms, 95)), 4),
        build_time_s=round(build_time_s, 4),
        memory_mb=round(memory_mb, 2),
    )
    append_result_csv(result, csv_path=csv_path)
    return result


def append_result_csv(result: BenchmarkResult, csv_path: str | Path = RESULTS_CSV) -> None:
    """Append one benchmark row to the CSV, writing the header if the file
    doesn't exist yet. Idempotent across repeated runs — each call only
    adds its own row.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(result))


# ---------------------------------------------------------------------------
# Manual smoke test — run `python indexing/benchmark.py`
# Benchmarks the FLAT baseline against itself (recall@10 should be 1.0),
# proving the harness end-to-end before IVF/HNSW/PQ scripts plug into it.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FLAT baseline benchmark")
    add_dataset_args(parser)
    parser.add_argument("--csv", default=str(RESULTS_CSV))
    cli = parser.parse_args()

    print("Loading dataset ...")
    ds = dataset_from_args(cli)

    print("Building FlatL2 ground-truth index ...")
    t0 = time.perf_counter()
    flat_index = build_flat_index(ds.xb)
    build_time = time.perf_counter() - t0

    print("Computing ground truth (FlatL2 self-search) ...")
    _, ground_truth_ids = search_flat(flat_index, ds.xq, k=10)

    print("Running benchmark harness on FLAT index ...")
    result = run_benchmark(
        index_type="FLAT",
        index=flat_index,
        xq=ds.xq,
        ground_truth_ids=ground_truth_ids,
        build_time_s=build_time,
        k=10,
        csv_path=cli.csv,
    )
    print(result)
    print(f"Appended to {cli.csv}")
