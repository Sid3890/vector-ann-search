"""GeekyClouds — IVF-PQ benchmark.

Builds FAISS IndexIVFPQ on the same IndexFlatL2 coarse quantizer and
nlist=100 configuration used by the Day 2 IVF benchmark. The product
quantizer uses 16 sub-quantizers and 8 bits per sub-vector code.
"""

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

N_LIST = 100
M_SUBQUANTIZERS = 16
NBITS = 8
DEFAULT_NPROBE = 8
NPROBE_VALUES = (1, 4, 8, 16, 32, 64, 100)
DEFAULT_K = 10

PLOT_PATH = Path(__file__).resolve().parent.parent / "results" / "pq_recall_latency.png"


def build_pq_index(
    xb: np.ndarray,
    nlist: int = N_LIST,
    m: int = M_SUBQUANTIZERS,
    nbits: int = NBITS,
    nprobe: int = DEFAULT_NPROBE,
) -> tuple[faiss.IndexIVFPQ, float]:
    """Build and train an IVF-PQ index using the Day 2 IVF coarse quantizer."""
    if xb.ndim != 2:
        raise ValueError("xb must be a 2D NumPy array")
    if xb.shape[0] < nlist:
        raise ValueError("Number of vectors must be at least nlist")
    if nlist <= 0 or nprobe <= 0:
        raise ValueError("nlist and nprobe must be positive")
    if nprobe > nlist:
        raise ValueError("nprobe cannot exceed nlist")
    if m <= 0 or nbits <= 0:
        raise ValueError("m and nbits must be positive")
    if xb.shape[1] % m != 0:
        raise ValueError("Vector dimension must be divisible by m for ProductQuantizer")

    vectors = np.ascontiguousarray(xb if xb.dtype == np.float32 else xb.astype(np.float32))
    dimension = vectors.shape[1]

    # This is intentionally the same coarse quantizer construction as ivf_index.py.
    coarse_quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFPQ(
        coarse_quantizer,
        dimension,
        nlist,
        m,
        nbits,
        faiss.METRIC_L2,
    )

    start_time = time.perf_counter()
    index.train(vectors)
    index.add(vectors)
    build_time_s = time.perf_counter() - start_time
    index.nprobe = nprobe
    return index, build_time_s


def serialized_index_bytes(index: faiss.Index) -> int:
    """Return the portable serialized FAISS index size in bytes."""
    return int(faiss.serialize_index(index).nbytes)


def plot_recall_latency(
    results: list[BenchmarkResult], output_path: str | Path = PLOT_PATH
) -> Path:
    if not results:
        raise ValueError("No benchmark results available for plotting")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    recalls = [result.recall_at_10 for result in results]
    latencies = [result.p95_ms for result in results]
    labels = [result.index_type.removeprefix("PQ_nprobe") for result in results]

    plt.figure(figsize=(8, 5))
    plt.plot(latencies, recalls, marker="o")
    for latency, recall, label in zip(latencies, recalls, labels):
        plt.annotate(f"nprobe={label}", (latency, recall), xytext=(6, 6), textcoords="offset points")
    plt.xlabel("P95 Query Latency (ms)")
    plt.ylabel("Recall@10")
    plt.title("IVF-PQ Recall vs P95 Query Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def run_pq_sweep(
    num_vectors: int = 100_000,
    dim: int = 128,
    num_queries: int = 1_000,
    seed: int = DEFAULT_SEED,
    nlist: int = N_LIST,
    m: int = M_SUBQUANTIZERS,
    nbits: int = NBITS,
    nprobe_values: tuple[int, ...] = NPROBE_VALUES,
    k: int = DEFAULT_K,
    csv_path: str | Path = RESULTS_CSV,
    plot_path: str | Path = PLOT_PATH,
) -> list[BenchmarkResult]:
    """Build once and benchmark query-time nprobe settings."""
    dataset = load_dataset(num_vectors=num_vectors, dim=dim, num_queries=num_queries, seed=seed)
    flat_index = build_flat_index(dataset.xb)
    _, ground_truth_ids = search_flat(flat_index, dataset.xq, k=k)

    index, build_time_s = build_pq_index(
        dataset.xb, nlist=nlist, m=m, nbits=nbits, nprobe=nprobe_values[0]
    )
    raw_bytes = int(dataset.xb.nbytes)
    pq_bytes = serialized_index_bytes(index)
    compression_ratio = raw_bytes / pq_bytes if pq_bytes else 0.0
    print(
        f"PQ index: nlist={nlist}, m={m}, nbits={nbits}; "
        f"raw={raw_bytes:,} bytes, pq={pq_bytes:,} bytes, "
        f"compression={compression_ratio:.3f}x"
    )

    results: list[BenchmarkResult] = []
    for nprobe in nprobe_values:
        if nprobe <= 0 or nprobe > nlist:
            raise ValueError(f"nprobe must be in [1, {nlist}], got {nprobe}")
        index.nprobe = nprobe
        result = run_benchmark(
            index_type=f"PQ_nprobe{nprobe}",
            index=index,
            xq=dataset.xq,
            ground_truth_ids=ground_truth_ids,
            build_time_s=build_time_s,
            k=k,
            csv_path=csv_path,
            raw_bytes=raw_bytes,
            pq_bytes=pq_bytes,
        )
        results.append(result)
        print(
            f"{result.index_type}: recall@10={result.recall_at_10:.4f}, "
            f"p50={result.p50_ms:.4f} ms, p95={result.p95_ms:.4f} ms, "
            f"memory={result.memory_mb:.2f} MB, "
            f"compression={result.compression_ratio:.4f}x"
        )

    output_path = plot_recall_latency(results, output_path=plot_path)
    print(f"Recall-vs-latency plot saved to: {output_path}")
    print(f"Benchmark results appended to: {csv_path}")
    return results


if __name__ == "__main__":
    run_pq_sweep()
