"""
GeekyClouds — ANN Benchmarking Core
indexing/baseline.py

Provides:
  - Synthetic embedding dataset generation (placeholder for SIFT1M / GloVe).
  - Exact brute-force k-NN search via faiss.IndexFlatL2, used as the
    ground-truth baseline that recall@k is measured against for every
    approximate index (IVF, HNSW, PQ, ...) built later in this project.

Swap-in points for real datasets (see `load_dataset` docstring):
  - SIFT1M:  http://corpus-texmex.irisa.fr/  (.fvecs format, 128-dim, 1M vectors)
  - GloVe:   https://nlp.stanford.edu/projects/glove/  (e.g. glove.6B.100d.txt)
  Both are drop-in replacements as long as they end up as an
  (N, dim) float32 numpy array — see `load_fvecs` / `load_glove_txt` stubs
  below for the loaders to implement when you get there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_NUM_VECTORS = 100_000
DEFAULT_DIM = 128
DEFAULT_NUM_QUERIES = 1_000
DEFAULT_SEED = 42


@dataclass
class Dataset:
    """A vector dataset ready for indexing/querying.

    xb: (n, dim) float32 — the base/corpus vectors that get indexed.
    xq: (nq, dim) float32 — query vectors to search for.
    dim: embedding dimensionality.
    """

    xb: np.ndarray
    xq: np.ndarray
    dim: int


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(
    num_vectors: int = DEFAULT_NUM_VECTORS,
    dim: int = DEFAULT_DIM,
    num_queries: int = DEFAULT_NUM_QUERIES,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Load (or synthesize) the embedding dataset used for benchmarking.

    Currently generates synthetic random float32 vectors so the pipeline
    is fully runnable with no external downloads. Swap this out later:

      - SIFT1M: download the .fvecs files (sift_base.fvecs, sift_query.fvecs)
        and parse them with `load_fvecs()` below, then wrap the result in
        a Dataset the same way as here.
      - GloVe: download e.g. glove.6B.128d.txt and parse with
        `load_glove_txt()` below (note: GloVe has no natural "query set",
        so a random sample of the base vectors is typically held out as
        queries instead).

    The rest of the pipeline (baseline index, benchmark harness) is
    dataset-agnostic — it only cares about `Dataset.xb` / `Dataset.xq`
    being (n, dim) float32 arrays, so no other code needs to change when
    you swap the source in here.
    """
    rng = np.random.default_rng(seed)
    xb = rng.random((num_vectors, dim), dtype=np.float32)
    xq = rng.random((num_queries, dim), dtype=np.float32)
    return Dataset(xb=xb, xq=xq, dim=dim)


def load_fvecs(path: str | Path) -> np.ndarray:
    """Parse a .fvecs file (SIFT1M format) into an (n, dim) float32 array.

    TODO: implement when swapping in SIFT1M. Format spec: each vector is
    stored as a little-endian int32 dim prefix followed by `dim` float32
    values, repeated back-to-back with no separators.
    """
    raise NotImplementedError(
        "Implement when SIFT1M is added: read int32 dim prefix + "
        "dim float32 values per record, per corpus-texmex.irisa.fr spec."
    )


def load_glove_txt(path: str | Path) -> np.ndarray:
    """Parse a GloVe .txt embedding file into an (n, dim) float32 array.

    TODO: implement when swapping in GloVe. Each line is
    `token v1 v2 ... vdim` space-separated; drop the token, stack the rest.
    """
    raise NotImplementedError(
        "Implement when GloVe is added: split each line, drop the token "
        "column, np.stack the remaining floats as float32."
    )


# ---------------------------------------------------------------------------
# Exact brute-force baseline (ground truth for recall@k)
# ---------------------------------------------------------------------------


def build_flat_index(xb: np.ndarray) -> faiss.IndexFlatL2:
    """Build an exact brute-force L2 index (ground truth for recall@k).

    faiss.IndexFlatL2 does no approximation — every query is compared
    against every base vector. This is what IVF/HNSW/PQ recall gets
    measured against downstream in benchmark.py.
    """
    if xb.dtype != np.float32:
        xb = xb.astype(np.float32)
    dim = xb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(xb)
    return index


def search_flat(
    index: faiss.IndexFlatL2, xq: np.ndarray, k: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Run exact k-NN search. Returns (distances, ids), both (nq, k)."""
    if xq.dtype != np.float32:
        xq = xq.astype(np.float32)
    distances, ids = index.search(xq, k)
    return distances, ids


# ---------------------------------------------------------------------------
# Manual smoke test — run `python indexing/baseline.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating synthetic dataset: {DEFAULT_NUM_VECTORS} x {DEFAULT_DIM}d ...")
    ds = load_dataset()

    print("Building exact FlatL2 baseline index ...")
    t0 = time.perf_counter()
    flat_index = build_flat_index(ds.xb)
    build_time = time.perf_counter() - t0
    print(f"Build time: {build_time:.3f}s | ntotal={flat_index.ntotal}")

    print(f"Searching {ds.xq.shape[0]} queries (k=10) ...")
    t0 = time.perf_counter()
    distances, ids = search_flat(flat_index, ds.xq, k=10)
    search_time = time.perf_counter() - t0
    print(f"Search time: {search_time:.3f}s for {ds.xq.shape[0]} queries")
    print(f"Sample result — query 0 top-5 neighbor ids: {ids[0][:5]}")
