"""Make benchmark queries in the same space as the base embeddings.

  python tests/make_queries.py --base-npy emb.npy --out data/queries.npy --n 500 --sigma 0.01

Each query = a random base vector + small Gaussian noise (re-normalised if the base is unit-norm).
Queries are NOT exact copies of base rows, so recall is not trivially 1.0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-npy", required=True)
    p.add_argument("--out", default="data/queries.npy")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    base = np.load(a.base_npy).astype("float32")
    rng = np.random.default_rng(a.seed)
    rows = rng.choice(len(base), size=min(a.n, len(base)), replace=False)
    q = base[rows] + rng.normal(0.0, a.sigma, size=(len(rows), base.shape[1])).astype("float32")
    norms = np.linalg.norm(base[:200], axis=1)
    if np.allclose(norms, 1.0, atol=1e-3):  # base looks unit-normalised
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, q.astype("float32"))
    print(f"saved {q.shape} to {a.out}; base unit-norm={np.allclose(norms, 1.0, atol=1e-3)}")


if __name__ == "__main__":
    main()
