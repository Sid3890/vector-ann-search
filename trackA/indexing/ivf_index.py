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
DEFAULT_NPROBE = 8
NPROBE_VALUES = (1, 4, 8, 16, 32)
DEFAULT_K = 10

PLOT_PATH = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "ivf_recall_latency.png"
)


def build_ivf_index(
    xb: np.ndarray,
    nlist: int = N_LIST,
    nprobe: int = DEFAULT_NPROBE,
) -> tuple[faiss.IndexIVFFlat, float]:
    if xb.ndim != 2:
        raise ValueError("xb must be a 2D NumPy array")

    if xb.shape[0] < nlist:
        raise ValueError("Number of vectors must be at least nlist")

    if nlist <= 0:
        raise ValueError("nlist must be positive")

    if nprobe <= 0:
        raise ValueError("nprobe must be positive")

    if nprobe > nlist:
        raise ValueError("nprobe cannot exceed nlist")

    vectors = np.ascontiguousarray(
        xb if xb.dtype == np.float32 else xb.astype(np.float32)
    )

    dimension = vectors.shape[1]

    quantizer = faiss.IndexFlatL2(dimension)

    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_L2,
    )

    start_time = time.perf_counter()

    index.train(vectors)
    index.add(vectors)

    build_time_s = time.perf_counter() - start_time

    index.nprobe = nprobe

    return index, build_time_s


def plot_recall_latency(
    results: list[BenchmarkResult],
    output_path: str | Path = PLOT_PATH,
) -> Path:
    if not results:
        raise ValueError("No benchmark results available for plotting")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    recalls = [
        result.recall_at_10
        for result in results
    ]

    latencies = [
        result.p95_ms
        for result in results
    ]

    labels = [
        result.index_type.removeprefix("IVF_nprobe")
        for result in results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        latencies,
        recalls,
        marker="o",
    )

    for latency, recall, label in zip(
        latencies,
        recalls,
        labels,
    ):
        plt.annotate(
            f"nprobe={label}",
            (latency, recall),
            xytext=(6, 6),
            textcoords="offset points",
        )

    plt.xlabel("P95 Query Latency (ms)")
    plt.ylabel("Recall@10")
    plt.title("IVF Recall vs P95 Query Latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output,
        dpi=160,
    )

    plt.close()

    return output


def run_ivf_sweep(
    num_vectors: int = 100_000,
    dim: int = 128,
    num_queries: int = 1_000,
    seed: int = DEFAULT_SEED,
    nlist: int = N_LIST,
    nprobe_values: tuple[int, ...] = NPROBE_VALUES,
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

    ground_truth_start = time.perf_counter()

    flat_index = build_flat_index(dataset.xb)

    ground_truth_build_time = (
        time.perf_counter() - ground_truth_start
    )

    print(
        f"Ground-truth build time: "
        f"{ground_truth_build_time:.3f}s"
    )

    print("Computing exact ground truth...")

    _, ground_truth_ids = search_flat(
        flat_index,
        dataset.xq,
        k=k,
    )

    results: list[BenchmarkResult] = []

    for nprobe in nprobe_values:

        print(
            f"\nBuilding IVF index "
            f"(nlist={nlist}, nprobe={nprobe})..."
        )

        index, build_time_s = build_ivf_index(
            dataset.xb,
            nlist=nlist,
            nprobe=nprobe,
        )

        result = run_benchmark(
            index_type=f"IVF_nprobe{nprobe}",
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

    output_path = plot_recall_latency(
        results,
        output_path=plot_path,
    )

    print(
        f"\nRecall-vs-latency plot saved to: "
        f"{output_path}"
    )

    print(
        f"Benchmark results appended to: "
        f"{csv_path}"
    )

    return results


if __name__ == "__main__":
    run_ivf_sweep()
