"""GeekyClouds shard partitioner.

Steps
  1. Load embeddings (Postgres `vectors` table, or .npy files).
  2. k-means (faiss.Kmeans, k=4) -> centroids; assign each vector to nearest centroid.
  3. Persist {id, shard_id} to Postgres in batches of 5,000 (parameterized, idempotent).
  4. Build one HNSW index per shard -> indexes/shard_<id>.faiss (+ id map sidecar).
  5. Log per-shard counts, flag imbalance (> 2x average), append to results/benchmarks.csv.

Modes
  default            fit new k-means centroids (first run only).
  --reuse-centroids  skip k-means; assign every row (incl. rows the ingestion worker
                     inserted with shard_id NULL) to the STORED centroids from `shards`
                     (or indexes/centroids.npy), then rebuild the per-shard indexes.
                     Use this as the periodic rebuild: Postgres is the source of truth and
                     the worker never writes to the FAISS files. Re-fitting k-means would
                     move the centroids and invalidate earlier assignments.

Credentials come from the environment / .env (never hardcoded):
  DATABASE_URL=postgresql://user:pass@host:5432/dbname
  (a SQLAlchemy-style 'postgresql+psycopg2://' prefix is accepted and normalized)
  or PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

Schema (Abhinandan's Day 2 DDL):
  vectors(id UUID PK, source_id TEXT, shard_id INT, embedding vector(384), metadata JSONB, created_at TIMESTAMPTZ)
  shards(shard_id INT PK, centroid vector(384), replica_urls TEXT[], vector_count INT)
(384 = all-MiniLM-L6-v2 output size; requires the Day 2 DDL fix from 128 -> 384.)
Embeddings are expected L2-normalized (normalize_embeddings=True upstream).
The Day 4 ingestion worker assigns shard_id from shards.centroid, so this script
also upserts centroids + vector_count there (replica_urls is never touched).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import faiss
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

try:  # optional: .env support for local dev
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---- schema assumptions (single place to change) ---------------------------
TABLE = "vectors"
COL_ID = "id"
COL_EMBEDDING = "embedding"
COL_SHARD = "shard_id"
SHARDS_TABLE = "shards"
DB_DIM = 384  # expected vector(384) after the Day 2 DDL fix (all-MiniLM-L6-v2)
NORM_TOLERANCE = 1e-2  # warn if mean |‖x‖-1| exceeds this

# ---- defaults ---------------------------------------------------------------
NUM_SHARDS = 4
BATCH_SIZE = 5_000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64
KMEANS_NITER = 25
SEED = 42
IMBALANCE_FACTOR = 2.0
EVAL_QUERIES = 500
EVAL_K = 10

log = logging.getLogger("shard_partitioner")


@dataclass(frozen=True)
class ShardStats:
    shard_id: int
    count: int
    build_time_sec: float
    memory_mb: float
    recall_at_k: float
    p50_ms: float
    p95_ms: float


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def normalize_dsn(dsn: str) -> str:
    """Strip SQLAlchemy driver suffixes ('postgresql+psycopg2://') that libpq rejects."""
    scheme, sep, rest = dsn.partition("://")
    return f"{scheme.split('+', 1)[0]}{sep}{rest}" if sep else dsn


def connect() -> "psycopg2.extensions.connection":
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return psycopg2.connect(normalize_dsn(dsn))
    required = ("PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing DB config: set DATABASE_URL or {', '.join(missing)} in the environment/.env"
        )
    return psycopg2.connect(
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
    )


def parse_pgvector(text: str) -> np.ndarray:
    """Parse pgvector's text form '[1,2,3]' into float32."""
    return np.fromstring(text.strip("[]"), dtype=np.float32, sep=",")


def db_column_dim(conn: "psycopg2.extensions.connection", table: str, column: str) -> int:
    """Declared dimension of a pgvector column (atttypmod holds it for vector(n))."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = %s",
            (table, column),
        )
        row = cur.fetchone()
    if row is None or row[0] is None or row[0] < 1:
        raise RuntimeError(f"Could not read vector dimension of {table}.{column}")
    return int(row[0])


def verify_dims(conn: "psycopg2.extensions.connection", dim: int) -> None:
    """Fail fast (before any write) if embeddings and DB columns disagree on dimension."""
    for table, col in ((TABLE, COL_EMBEDDING), (SHARDS_TABLE, "centroid")):
        declared = db_column_dim(conn, table, col)
        if declared != dim:
            raise ValueError(
                f"{table}.{col} is vector({declared}) but embeddings are {dim}-dim "
                f"(expected {DB_DIM}); fix the DDL/migration before partitioning"
            )


def load_centroids(
    conn: "psycopg2.extensions.connection | None", index_dir: Path, k: int, dim: int
) -> np.ndarray:
    """Stored centroids: prefer Postgres `shards`, fall back to indexes/centroids.npy."""
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT shard_id, centroid::text FROM {SHARDS_TABLE} "
                "WHERE centroid IS NOT NULL ORDER BY shard_id"
            )
            rows = cur.fetchall()
        if len(rows) == k and [r[0] for r in rows] == list(range(k)):
            return np.ascontiguousarray(np.vstack([parse_pgvector(r[1]) for r in rows]), dtype=np.float32)
        log.warning("'%s' has %d usable centroid rows (need %d); trying centroids.npy", SHARDS_TABLE, len(rows), k)
    path = index_dir / "centroids.npy"
    if path.exists():
        c = np.ascontiguousarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if c.shape == (k, dim):
            return c
        raise ValueError(f"{path} has shape {c.shape}, expected {(k, dim)}")
    raise RuntimeError("No stored centroids found; run once without --reuse-centroids first")


def load_from_db(conn: "psycopg2.extensions.connection") -> tuple[list[str], np.ndarray]:
    """Stream (id, embedding) with a server-side cursor; pgvector returns '[1,2,...]' text."""
    ids: list[str] = []
    rows: list[np.ndarray] = []
    # Table/column names are module constants, not user input.
    query = f"SELECT {COL_ID}, {COL_EMBEDDING}::text FROM {TABLE} ORDER BY {COL_ID}"
    with conn.cursor(name="shard_load") as cur:
        cur.itersize = 10_000
        cur.execute(query)
        for vid, emb in cur:
            ids.append(str(vid))
            rows.append(parse_pgvector(emb))
    if not rows:
        raise RuntimeError(f"Table '{TABLE}' returned no rows; nothing to partition.")
    return ids, np.ascontiguousarray(np.vstack(rows), dtype=np.float32)


def load_from_npy(emb_path: Path, ids_path: Path) -> tuple[list[str], np.ndarray]:
    x = np.ascontiguousarray(np.load(emb_path), dtype=np.float32)
    ids = [str(i) for i in np.load(ids_path, allow_pickle=False).tolist()]
    if x.ndim != 2 or len(ids) != x.shape[0]:
        raise ValueError(f"Shape mismatch: embeddings {x.shape}, ids {len(ids)}")
    return ids, x


def chunked(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def persist_assignments(
    conn: "psycopg2.extensions.connection",
    ids: Sequence[str],
    shard_ids: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> int:
    """UPDATE vectors.shard_id in batches of 5,000. Parameterized and idempotent.

    One transaction per batch: a crash mid-run leaves a partially updated table,
    but re-running converges to the same result (deterministic seed + same data).
    """
    pairs = [(vid, int(s)) for vid, s in zip(ids, shard_ids)]
    sql = (
        f"UPDATE {TABLE} AS t SET {COL_SHARD} = v.shard_id "
        f"FROM (VALUES %s) AS v({COL_ID}, shard_id) "
        f"WHERE t.{COL_ID} = v.{COL_ID}"
    )
    updated = 0
    for n, batch in enumerate(chunked(pairs, batch_size), start=1):
        try:
            with conn.cursor() as cur:
                execute_values(cur, sql, batch, template="(%s::uuid, %s::int)", page_size=len(batch))
                updated += cur.rowcount
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            log.exception("Batch %d failed (rows %d-%d); rolled back", n, (n - 1) * batch_size, n * batch_size)
            raise
        log.info("Persisted batch %d (%d rows so far)", n, updated)
    if updated != len(pairs):
        log.warning("Updated %d rows but expected %d: some ids missing from '%s'", updated, len(pairs), TABLE)
    return updated


def upsert_shards(
    conn: "psycopg2.extensions.connection", centroids: np.ndarray, counts: dict[int, int]
) -> None:
    """Upsert shard_id, centroid, vector_count into `shards` (replica_urls untouched)."""
    sql = (
        f"INSERT INTO {SHARDS_TABLE} (shard_id, centroid, vector_count) VALUES (%s, %s::vector, %s) "
        "ON CONFLICT (shard_id) DO UPDATE "
        "SET centroid = EXCLUDED.centroid, vector_count = EXCLUDED.vector_count"
    )
    try:
        with conn.cursor() as cur:
            for sid in range(centroids.shape[0]):
                vec = "[" + ",".join(f"{v:.8g}" for v in centroids[sid]) + "]"
                cur.execute(sql, (sid, vec, counts.get(sid, 0)))
        conn.commit()
    except psycopg2.Error:
        conn.rollback()
        log.exception("Failed to upsert '%s'", SHARDS_TABLE)
        raise
    log.info("Upserted %d rows into '%s'", centroids.shape[0], SHARDS_TABLE)


# ---------------------------------------------------------------------------
# Clustering + indexing
# ---------------------------------------------------------------------------
def cluster(x: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroids[k, d], assignments[n]) using faiss k-means (L2)."""
    if x.shape[0] < k:
        raise ValueError(f"Need at least {k} vectors, got {x.shape[0]}")
    km = faiss.Kmeans(x.shape[1], k, niter=KMEANS_NITER, seed=SEED, verbose=False, gpu=False)
    km.train(x)
    _, assign = km.index.search(x, 1)
    return km.centroids.copy(), assign.ravel().astype(np.int64)


def assign_to_centroids(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest stored centroid (L2) for each row; no re-fitting."""
    quant = faiss.IndexFlatL2(centroids.shape[1])
    quant.add(centroids)
    _, assign = quant.search(x, 1)
    return assign.ravel().astype(np.int64)


def warn_if_not_normalized(x: np.ndarray) -> None:
    sample = x[: min(len(x), 10_000)]
    dev = float(np.abs(np.linalg.norm(sample, axis=1) - 1.0).mean())
    if dev > NORM_TOLERANCE:
        log.warning("Embeddings are not L2-normalized (mean |norm-1|=%.3f). Ask the embedding "
                    "service to use normalize_embeddings=True so L2 ranking == cosine ranking.", dev)


def atomic_write_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        faiss.write_index(index, tmp)
        os.replace(tmp, path)  # atomic: no half-written index on crash
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def atomic_save_npy(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    os.close(fd)
    try:
        np.save(tmp, arr, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def build_shard_index(shard_id: int, x: np.ndarray, ids: Sequence[str], out_dir: Path) -> ShardStats:
    d = x.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(d, HNSW_M)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(x)
    build_time = time.perf_counter() - t0
    index.hnsw.efSearch = HNSW_EF_SEARCH

    index_path = out_dir / f"shard_{shard_id}.faiss"
    atomic_write_index(index, index_path)
    # Position i in the index -> string id (FAISS HNSW has no native string/ID mapping)
    atomic_save_npy(np.asarray(ids, dtype="U64"), out_dir / f"shard_{shard_id}.ids.npy")

    recall, p50, p95 = evaluate(index, x)
    memory_mb = index_path.stat().st_size / 1024**2  # on-disk size ~ in-memory footprint for HNSWFlat
    return ShardStats(shard_id, x.shape[0], build_time, memory_mb, recall, p50, p95)


def evaluate(index: faiss.IndexHNSWFlat, x: np.ndarray) -> tuple[float, float, float]:
    """Self-recall@k vs exact search on a sample + per-query latency (p50/p95 ms)."""
    n = x.shape[0]
    k = min(EVAL_K, n)
    rng = np.random.default_rng(SEED)
    q = x[rng.choice(n, size=min(EVAL_QUERIES, n), replace=False)]

    flat = faiss.IndexFlatL2(x.shape[1])
    flat.add(x)
    _, gt = flat.search(q, k)

    lat: list[float] = []
    hits = 0
    for i in range(q.shape[0]):
        t = time.perf_counter()
        _, got = index.search(q[i : i + 1], k)
        lat.append((time.perf_counter() - t) * 1000)
        hits += len(set(got[0].tolist()) & set(gt[i].tolist()))
    recall = hits / (q.shape[0] * k)
    return recall, float(np.percentile(lat, 50)), float(np.percentile(lat, 95))


# ---------------------------------------------------------------------------
# Logging / reporting
# ---------------------------------------------------------------------------
def check_balance(counts: dict[int, int]) -> bool:
    avg = sum(counts.values()) / len(counts)
    ok = True
    for sid, c in sorted(counts.items()):
        flag = ""
        if c > IMBALANCE_FACTOR * avg:
            flag = "  <-- EXCEEDS 2x AVERAGE"
            ok = False
        log.info("shard %d: %d vectors (%.2fx avg)%s", sid, c, c / avg, flag)
    log.info("average shard size: %.1f | balanced: %s", avg, ok)
    return ok


def append_benchmarks(stats: Sequence[ShardStats], path: Path) -> None:
    """Append rows (idempotent per run via a 'run' tag in the label; header written once)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["index_type", "label", "recall_at_k", "p50_ms", "p95_ms",
                        "build_time_sec", "memory_mb", "cost_per_query"])
        for s in stats:
            w.writerow(["HNSW", f"shard_{s.shard_id}(n={s.count},self_recall)", f"{s.recall_at_k:.4f}",
                        f"{s.p50_ms:.3f}", f"{s.p95_ms:.3f}", f"{s.build_time_sec:.3f}",
                        f"{s.memory_mb:.2f}", ""])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Partition embeddings into shards and build per-shard HNSW indexes.")
    p.add_argument("--num-shards", type=int, default=NUM_SHARDS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--index-dir", type=Path, default=Path("indexes"))
    p.add_argument("--bench-csv", type=Path, default=Path("results/benchmarks.csv"))
    p.add_argument("--embeddings-npy", type=Path, help="Load embeddings from .npy instead of Postgres")
    p.add_argument("--ids-npy", type=Path, help="ids .npy (required with --embeddings-npy)")
    p.add_argument("--reuse-centroids", action="store_true",
                   help="Assign to stored centroids (shards table / centroids.npy) instead of re-running k-means")
    p.add_argument("--skip-db-write", action="store_true", help="Skip persisting shard_id to Postgres")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    conn = None
    try:
        if args.embeddings_npy:
            if not args.ids_npy:
                raise ValueError("--ids-npy is required with --embeddings-npy")
            ids, x = load_from_npy(args.embeddings_npy, args.ids_npy)
        else:
            conn = connect()
            ids, x = load_from_db(conn)
        log.info("Loaded %d vectors, dim=%d", x.shape[0], x.shape[1])
        warn_if_not_normalized(x)

        needs_db = not args.skip_db_write
        if conn is None and (needs_db or (args.reuse_centroids and not (args.index_dir / "centroids.npy").exists())):
            conn = connect()
        if conn is not None and needs_db:
            verify_dims(conn, x.shape[1])  # before any write

        t0 = time.perf_counter()
        if args.reuse_centroids:
            centroids = load_centroids(conn, args.index_dir, args.num_shards, x.shape[1])
            assign = assign_to_centroids(x, centroids)
            log.info("Reused stored centroids (k=%d) in %.2fs", args.num_shards, time.perf_counter() - t0)
        else:
            centroids, assign = cluster(x, args.num_shards)
            log.info("k-means (k=%d) done in %.2fs", args.num_shards, time.perf_counter() - t0)
        atomic_save_npy(centroids, args.index_dir / "centroids.npy")  # router/ingest need these

        counts = {sid: int((assign == sid).sum()) for sid in range(args.num_shards)}
        if needs_db:
            persist_assignments(conn, ids, assign, args.batch_size)
            upsert_shards(conn, centroids, counts)

        stats: list[ShardStats] = []
        for sid in range(args.num_shards):
            mask = assign == sid
            if counts[sid] == 0:
                log.error("shard %d is empty; skipping index build", sid)
                continue
            shard_ids = [ids[i] for i in np.flatnonzero(mask)]
            s = build_shard_index(sid, np.ascontiguousarray(x[mask]), shard_ids, args.index_dir)
            log.info("shard %d built: %.2fs, %.1f MB, recall@%d=%.3f, p50=%.2fms p95=%.2fms",
                     sid, s.build_time_sec, s.memory_mb, EVAL_K, s.recall_at_k, s.p50_ms, s.p95_ms)
            stats.append(s)

        balanced = check_balance(counts)
        append_benchmarks(stats, args.bench_csv)
        return 0 if balanced else 3
    except Exception:
        log.exception("Shard partitioning failed")
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    sys.exit(main())
