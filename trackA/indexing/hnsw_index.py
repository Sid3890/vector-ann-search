from __future__ import annotations

import time
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np

try:
    from .baseline import DEFAULT_SEED, build_flat_index, load_dataset, search_flat
    from .benchmark import RESULTS_CSV, BenchmarkResult, run_benchmark
except ImportError:
    from baseline import DEFAULT_SEED, build_flat_index, load_dataset, search_flat
    from benchmark import RESULTS_CSV, BenchmarkResult, run_benchmark


HNSW_M = 32
EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64
EF_SEARCH_VALUES = (16, 32, 64, 128)
DEFAULT_K = 10

PLOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "hnsw_recall_latency.png"
)


def build_hnsw_index(
    xb: np.ndarray,
    m: int = HNSW_M,
    ef_construction: int = EF_CONSTRUCTION,
    ef_search: int = DEFAULT_EF_SEARCH,
) -> tuple[faiss.IndexHNSWFlat, float]:
    if xb.ndim != 2:
        raise ValueError("xb must be a 2D NumPy array")

    if m <= 0:
        raise ValueError("M must be positive")

    if ef_construction <= 0:
        raise ValueError("efConstruction must be positive")

    if ef_search <= 0:
        raise ValueError("efSearch must be positive")

    vectors = np.ascontiguousarray(
        xb if xb.dtype == np.float32 else xb.astype(np.float32)
    )

    dimension = vectors.shape[1]

    index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search

    start_time = time.perf_counter()

    # HNSW needs no training step; the graph is built incrementally on add().
    index.add(vectors)

    build_time_s = time.perf_counter() - start_time

    return index, build_time_s


def set_ef_search(index: faiss.IndexHNSWFlat, ef_search: int) -> None:
    if ef_search <= 0:
        raise ValueError("efSearch must be positive")

    index.hnsw.efSearch = ef_search


def plot_recall_latency(
    results: list[BenchmarkResult],
    output_path: str | Path = PLOT_PATH,
) -> Path:
    if not results:
        raise ValueError("No benchmark results available for plotting")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    recalls = [result.recall_at_10 for result in results]
    latencies = [result.p95_ms for result in results]
    labels = [
        result.index_type.removeprefix("HNSW_efSearch")
        for result in results
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(latencies, recalls, marker="o")

    for latency, recall, label in zip(latencies, recalls, labels):
        plt.annotate(
            f"efSearch={label}",
            (latency, recall),
            xytext=(6, 6),
            textcoords="offset points",
        )

    plt.xlabel("P95 Query Latency (ms)")
    plt.ylabel("Recall@10")
    plt.title("HNSW Recall vs P95 Query Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()

    return output


def run_hnsw_sweep(
    num_vectors: int = 100_000,
    dim: int = 128,
    num_queries: int = 1_000,
    seed: int = DEFAULT_SEED,
    m: int = HNSW_M,
    ef_construction: int = EF_CONSTRUCTION,
    ef_search_values: tuple[int, ...] = EF_SEARCH_VALUES,
    k: int = DEFAULT_K,
    csv_path: str | Path = RESULTS_CSV,
    plot_path: str | Path = PLOT_PATH,
) -> list[BenchmarkResult]:

    dataset = load_dataset(
        num_vectors=num_vectors,
        dim=dim,
        num_queries=num_queries,
        seed=seed,
    )

    print(
        f"Dataset: "
        f"{dataset.xb.shape[0]} vectors x "
        f"{dataset.dim} dimensions"
    )

    print("Building exact FlatL2 ground-truth index...")

    flat_index = build_flat_index(dataset.xb)

    print("Computing exact ground truth...")

    _, ground_truth_ids = search_flat(flat_index, dataset.xq, k=k)

    # Unlike IVF's nprobe, efSearch is a query-time knob only, so the graph
    # is built once and every sweep row reports that same build time.
    print(
        f"\nBuilding HNSW index "
        f"(M={m}, efConstruction={ef_construction})..."
    )

    index, build_time_s = build_hnsw_index(
        dataset.xb,
        m=m,
        ef_construction=ef_construction,
    )

    print(f"HNSW build time: {build_time_s:.3f}s")

    results: list[BenchmarkResult] = []

    for ef_search in ef_search_values:

        set_ef_search(index, ef_search)

        result = run_benchmark(
            index_type=f"HNSW_efSearch{ef_search}",
            index=index,
            xq=dataset.xq,
            ground_truth_ids=ground_truth_ids,
            build_time_s=build_time_s,
            k=k,
            csv_path=csv_path,
        )

        results.append(result)

        print(
            f"{result.index_type}: "
            f"recall@10={result.recall_at_10:.4f}, "
            f"p50={result.p50_ms:.4f} ms, "
            f"p95={result.p95_ms:.4f} ms, "
            f"build={result.build_time_s:.4f} s, "
            f"memory={result.memory_mb:.2f} MB"
        )

    output_path = plot_recall_latency(results, output_path=plot_path)

    print(f"\nRecall-vs-latency plot saved to: {output_path}")
    print(f"Benchmark results appended to: {csv_path}")

    return results


if __name__ == "__main__":
    run_hnsw_sweep()
