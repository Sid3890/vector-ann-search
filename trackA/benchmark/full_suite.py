#!/usr/bin/env python3
"""
benchmark/full_suite.py -- GeekyClouds Day 8 full comparison suite.

Compares three deployment modes on identical data/queries:
  1. local       single-node FAISS: FLAT / IVF / HNSW / PQ   (built in-process)
  2. distributed self-built sharded+replicated service       (via router POST /search)
  3. managed     Pinecone or AWS OpenSearch                  (live adapter or MANAGED row in benchmarks.csv)

Metrics per row: recall@10, p50/p95 latency (ms), build time (s), memory (MB), cost per query (USD).
Outputs (overwritten every run, so the script is idempotent):
  results/final_comparison.csv
  results/charts/<metric>.png   (one bar chart per metric)

Secrets are read from the environment only (PINECONE_API_KEY, OPENSEARCH_PASSWORD, ...). Never hardcode them.

Example:
  python benchmark/full_suite.py \
      --router-url http://<alb-dns> --replicas 1 \
      --managed live-pinecone --managed-cost-per-query 0.000008
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence

import numpy as np

log = logging.getLogger("full_suite")

IndexType = Literal["FLAT", "IVF", "HNSW", "PQ", "MANAGED"]  # matches BenchmarkResult contract
Mode = Literal["local", "distributed", "managed"]
SearchFn = Callable[[np.ndarray, int], "tuple[list[int], bool]"]  # (query, k) -> (ids, partial)

K = 10
CSV_COLUMNS = [
    "mode", "index_type", "recall_at_10", "p50_ms", "p95_ms",
    "build_time_s", "memory_mb", "cost_per_query_usd", "qps", "notes",
]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pricing:
    """On-demand USD prices. DEFAULTS ARE PLACEHOLDERS -- verify against the AWS pricing page
    for ap-south-1 and override via env vars before quoting numbers in the report."""
    fargate_vcpu_hr: float = 0.04256
    fargate_gb_hr: float = 0.004655
    alb_hr: float = 0.0239

    @staticmethod
    def from_env() -> "Pricing":
        d = Pricing()
        return Pricing(
            fargate_vcpu_hr=float(os.environ.get("PRICE_FARGATE_VCPU_HR", d.fargate_vcpu_hr)),
            fargate_gb_hr=float(os.environ.get("PRICE_FARGATE_GB_HR", d.fargate_gb_hr)),
            alb_hr=float(os.environ.get("PRICE_ALB_HR", d.alb_hr)),
        )

    def fargate_task_hr(self, vcpu: float, gb: float) -> float:
        return vcpu * self.fargate_vcpu_hr + gb * self.fargate_gb_hr


@dataclass
class Row:
    mode: Mode
    index_type: IndexType
    recall_at_10: float
    p50_ms: float
    p95_ms: float
    build_time_s: float
    memory_mb: float
    cost_per_query_usd: float
    qps: float
    notes: str = ""


@dataclass
class Measured:
    lat_ms: list[float]
    preds: list[list[int]]
    partial_count: int
    qps: float


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Base + query vectors. MUST be the same vectors the shards/managed index were built from,
    otherwise recall is meaningless. Prefer --base-npy/--queries-npy pointing at saved arrays."""
    if args.base_npy:
        base = np.load(args.base_npy).astype("float32")
    else:
        rng = np.random.default_rng(args.seed)
        base = rng.random((args.n_base, args.dim), dtype=np.float32)
        log.warning("Using synthetic base data (seed=%d). Verify this matches what the shards were built from.", args.seed)
    if args.queries_npy:
        queries = np.load(args.queries_npy).astype("float32")
    else:
        rng_q = np.random.default_rng(args.seed + 1)
        queries = rng_q.random((args.n_queries, base.shape[1]), dtype=np.float32)
    queries = np.ascontiguousarray(queries[: args.n_queries])
    return np.ascontiguousarray(base), queries


def load_id_map(path: Optional[str]) -> Optional[dict[str, int]]:
    """Optional CSV (columns: id,row) mapping router/DB ids (e.g. UUIDs) to base-array row numbers."""
    if not path:
        return None
    if path.lower().endswith(".npy"):  # ids.npy: original id of each base row, in row order
        return {str(v): i for i, v in enumerate(np.load(path, allow_pickle=False).tolist())}
    with open(path, newline="") as f:
        return {r["id"]: int(r["row"]) for r in csv.DictReader(f)}


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #
def recall_at_k(preds: Sequence[Sequence[int]], gt: np.ndarray, k: int = K) -> float:
    hits = 0
    for p, g in zip(preds, gt):
        hits += len(set(p[:k]) & set(g[:k].tolist()))
    return hits / (len(gt) * k)


def percentile(vals: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def measure(fn: SearchFn, queries: np.ndarray, k: int, warmup: int = 20, concurrency: int = 1) -> Measured:
    """Sequential pass -> unloaded latency + predictions. If concurrency>1, a second concurrent pass
    measures throughput (qps); otherwise qps = n / total sequential time."""
    for q in queries[:warmup]:
        try:
            fn(q, k)
        except Exception as exc:  # warmup failures should not abort; real pass will surface them
            log.debug("warmup error: %s", exc)
    lat: list[float] = []
    preds: list[list[int]] = []
    partial = 0
    t_all = time.perf_counter()
    for q in queries:
        t0 = time.perf_counter()
        ids, is_partial = fn(q, k)
        lat.append((time.perf_counter() - t0) * 1000.0)
        preds.append(ids)
        partial += int(is_partial)
    seq_wall = time.perf_counter() - t_all
    qps = len(queries) / seq_wall
    if concurrency > 1:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(lambda q: fn(q, k), queries))
        qps = len(queries) / (time.perf_counter() - t0)
    return Measured(lat, preds, partial, qps)


def cost_per_query(hourly_usd: float, qps: float) -> float:
    return hourly_usd / (qps * 3600.0) if qps > 0 else math.nan


# --------------------------------------------------------------------------- #
# 1. LOCAL (single-node FAISS)
# --------------------------------------------------------------------------- #
def build_local(kind: IndexType, base: np.ndarray):
    import faiss
    d = base.shape[1]
    t0 = time.perf_counter()
    if kind == "FLAT":
        index = faiss.IndexFlatL2(d)
        index.add(base)
    elif kind == "IVF":
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, 100)
        index.train(base)
        index.add(base)
        index.nprobe = 16
    elif kind == "HNSW":
        index = faiss.IndexHNSWFlat(d, 32)
        index.add(base)
        index.hnsw.efSearch = 64
    elif kind == "PQ":
        index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, 100, 16, 8)
        index.train(base)
        index.add(base)
        index.nprobe = 16
    else:
        raise ValueError(kind)
    return index, time.perf_counter() - t0


def index_size_mb(index) -> float:
    import faiss
    return faiss.serialize_index(index).nbytes / 1e6


def run_local(base: np.ndarray, queries: np.ndarray, gt: np.ndarray, pricing: Pricing,
              vcpu: float, gb: float) -> list[Row]:
    import faiss
    faiss.omp_set_num_threads(1)  # single-query latency on 1 vCPU-equivalent
    hourly = pricing.fargate_task_hr(vcpu, gb)
    rows: list[Row] = []
    for kind in ("FLAT", "IVF", "HNSW", "PQ"):
        log.info("local: building %s ...", kind)
        index, build_s = build_local(kind, base)

        def fn(q: np.ndarray, k: int, _ix=index) -> tuple[list[int], bool]:
            _, ids = _ix.search(q[None, :], k)
            return ids[0].tolist(), False

        m = measure(fn, queries, K)
        rows.append(Row(
            "local", kind, recall_at_k(m.preds, gt), percentile(m.lat_ms, 50), percentile(m.lat_ms, 95),
            build_s, index_size_mb(index), cost_per_query(hourly, m.qps), m.qps,
            f"priced as 1 Fargate task {vcpu}vCPU/{gb}GB; mem=serialized index size; single thread",
        ))
        log.info("local %s: recall=%.3f p50=%.2fms", kind, rows[-1].recall_at_10, rows[-1].p50_ms)
    return rows


# --------------------------------------------------------------------------- #
# 2. DISTRIBUTED (router over HTTP)
# --------------------------------------------------------------------------- #
class RouterClient:
    def __init__(self, base_url: str, timeout_s: float, id_map: Optional[dict[str, int]]) -> None:
        import httpx
        self._url = base_url.rstrip("/") + "/search"
        self._client = httpx.Client(timeout=timeout_s)
        self._id_map = id_map

    def _to_row(self, raw: object) -> int:
        if self._id_map is not None:
            return self._id_map[str(raw)]
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Router returned non-integer id {raw!r}; pass --id-map CSV (id,row) to map it to base-array rows"
            ) from exc

    errors: int = 0
    last_error: str = ""

    def search(self, q: np.ndarray, k: int) -> tuple[list[int], bool]:
        """HTTP/transport failures (5xx, timeouts) count as a failed query: returns ([], True) and is tallied,
        so one blip does not abort the run. Config errors (bad ids) still raise."""
        import httpx
        try:
            r = self._client.post(self._url, json={"query_vector": q.tolist(), "top_k": k})
            r.raise_for_status()
            body = r.json()
        except httpx.HTTPError as exc:
            self.errors += 1  # approximate under threads; fine for a diagnostic count
            self.last_error = str(exc)
            if self.errors <= 3:
                log.warning("router request failed: %s", exc)
            return [], True
        return [self._to_row(x) for x in body["ids"]], bool(body.get("partial", False))


def build_distributed_proxy(base: np.ndarray, n_shards: int) -> float:
    """Rebuilds k-means + one HNSW(M=32) per shard with the same parameters as shard_partitioner.py
    to estimate build time (sequential sum). A proxy: it does not time the deployed indexes."""
    import faiss
    d = base.shape[1]
    t0 = time.perf_counter()
    km = faiss.Kmeans(d, n_shards, niter=20, seed=1)
    km.train(base)
    _, assign = km.index.search(base, 1)
    assign = assign.ravel()
    for s in range(n_shards):
        idx = faiss.IndexHNSWFlat(d, 32)
        idx.add(np.ascontiguousarray(base[assign == s]))
    return time.perf_counter() - t0


def run_distributed(args: argparse.Namespace, base: np.ndarray, queries: np.ndarray, gt: np.ndarray,
                    pricing: Pricing) -> Row:
    client = RouterClient(args.router_url, args.http_timeout, load_id_map(args.id_map))
    m = measure(client.search, queries, K, concurrency=args.concurrency)
    total_calls = len(queries) * (2 if args.concurrency > 1 else 1)
    if client.errors > 0.2 * total_calls:
        raise RuntimeError(
            f"{client.errors}/{total_calls} router requests failed (last: {client.last_error}); "
            "check the router/shard windows or lower --concurrency")

    shard_files = sorted(Path(args.index_dir).glob("shard_*.faiss"))
    if args.dist_memory_mb is not None:
        mem_mb = args.dist_memory_mb
    elif shard_files:
        mem_mb = sum(f.stat().st_size for f in shard_files) / 1e6 * args.replicas
    else:
        raise FileNotFoundError(f"No shard_*.faiss under {args.index_dir}; pass --dist-memory-mb")

    build_s = args.dist_build_time_s if args.dist_build_time_s is not None \
        else build_distributed_proxy(base, args.shards)

    hourly = (
        pricing.fargate_task_hr(0.5, 1.0)  # router (rounded to valid Fargate size)
        + args.shards * args.replicas * pricing.fargate_task_hr(args.shard_vcpu, args.shard_gb)
        + pricing.alb_hr
    )
    notes = (f"{args.shards} shards x {args.replicas} replica(s) on Fargate + router + ALB; "
             f"mem=sum shard files x replicas; build={'given' if args.dist_build_time_s is not None else 'proxy rebuild, sequential'}; "
             f"qps measured from client at concurrency={args.concurrency}; ALB hourly only (no LCU/data transfer)")
    if client.errors:
        notes += f"; WARNING {client.errors} failed HTTP requests (counted as empty results)"
    if m.partial_count:
        notes += f"; WARNING {m.partial_count}/{len(queries)} partial responses (recall is depressed)"
        log.warning("distributed: %d partial responses", m.partial_count)
    return Row("distributed", "HNSW", recall_at_k(m.preds, gt), percentile(m.lat_ms, 50),
               percentile(m.lat_ms, 95), build_s, mem_mb, cost_per_query(hourly, m.qps), m.qps, notes)


# --------------------------------------------------------------------------- #
# 3. MANAGED (Pinecone / OpenSearch)
# --------------------------------------------------------------------------- #
def make_pinecone_fn() -> SearchFn:
    from pinecone import Pinecone  # pip install pinecone  (pinecone-client is the old package name)
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX"])

    def fn(q: np.ndarray, k: int) -> tuple[list[int], bool]:
        res = index.query(vector=q.tolist(), top_k=k)
        return [int(m.id) for m in res.matches], False
    return fn


def make_opensearch_fn() -> SearchFn:
    from opensearchpy import OpenSearch  # pip install opensearch-py
    client = OpenSearch(
        hosts=[os.environ["OPENSEARCH_URL"]],
        http_auth=(os.environ["OPENSEARCH_USER"], os.environ["OPENSEARCH_PASSWORD"]),
        use_ssl=True, verify_certs=True,
    )
    index, field = os.environ["OPENSEARCH_INDEX"], os.environ.get("OPENSEARCH_VECTOR_FIELD", "embedding")

    def fn(q: np.ndarray, k: int) -> tuple[list[int], bool]:
        body = {"size": k, "query": {"knn": {field: {"vector": q.tolist(), "k": k}}}}
        hits = client.search(index=index, body=body)["hits"]["hits"]
        return [int(h["_id"]) for h in hits], False
    return fn


def read_managed_csv_row(path: Path) -> Optional[dict[str, str]]:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("index_type") == "MANAGED"]
    return rows[-1] if rows else None


def run_managed(args: argparse.Namespace, queries: np.ndarray, gt: np.ndarray, pricing: Pricing) -> Row:
    csv_row = read_managed_csv_row(Path(args.benchmarks_csv))
    if args.managed == "csv":
        if csv_row is None:
            raise FileNotFoundError(f"No MANAGED row in {args.benchmarks_csv}; run Day 9 managed_client first or use --managed live-*")
        recall, p50, p95 = (float(csv_row[c]) for c in ("recall_at_10", "p50_ms", "p95_ms"))
        qps = 1000.0 / p50 if p50 > 0 else math.nan
        src = "from benchmarks.csv MANAGED row"
    else:
        fn = make_pinecone_fn() if args.managed == "live-pinecone" else make_opensearch_fn()
        m = measure(fn, queries, K, concurrency=args.concurrency)
        recall, p50, p95, qps = recall_at_k(m.preds, gt), percentile(m.lat_ms, 50), percentile(m.lat_ms, 95), m.qps
        src = f"live {args.managed}; latency includes client->cloud network"

    build_s = float(csv_row["build_time_s"]) if csv_row else math.nan
    mem_mb = float(csv_row["memory_mb"]) if csv_row else 0.0
    if math.isnan(build_s):
        if args.managed_build_time_s is None:
            raise ValueError("Managed build time unknown: need a MANAGED row in benchmarks.csv or --managed-build-time-s")
        build_s = args.managed_build_time_s

    if args.managed_cost_per_query is not None:
        cpq = args.managed_cost_per_query
    elif args.managed_hourly_usd is not None:
        cpq = cost_per_query(args.managed_hourly_usd, qps)
    else:
        raise ValueError("Provide --managed-cost-per-query (e.g. Pinecone read units) or --managed-hourly-usd (e.g. OpenSearch instance)")
    notes = f"{src}; memory not exposed by provider (value as logged); cost from provider pricing page"
    return Row("managed", "MANAGED", recall, p50, p95, build_s, mem_mb, cpq, qps, notes)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv_atomic(rows: Sequence[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            d = asdict(r)
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in d.items()})
    os.replace(tmp, path)


METRICS: dict[str, tuple[str, bool]] = {  # column -> (title, log_scale)
    "recall_at_10": ("Recall@10 (higher is better)", False),
    "p50_ms": ("p50 query latency, ms (lower is better)", True),
    "p95_ms": ("p95 query latency, ms (lower is better)", True),
    "build_time_s": ("Index build time, s (lower is better)", True),
    "memory_mb": ("Memory footprint, MB (lower is better)", True),
    "cost_per_query_usd": ("Cost per query, USD (lower is better)", True),
}
MODE_COLORS = {"local": "#4C78A8", "distributed": "#F58518", "managed": "#54A24B"}


def plot_charts(rows: Sequence[Row], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.mode}\n{r.index_type}" for r in rows]
    colors = [MODE_COLORS[r.mode] for r in rows]
    for col, (title, log_scale) in METRICS.items():
        vals = [getattr(r, col) for r in rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        positive = [v for v in vals if v > 0]
        if log_scale and positive:
            ax.set_yscale("log")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", dpi=150)
        plt.close(fig)


def print_markdown(rows: Sequence[Row]) -> None:
    cols = CSV_COLUMNS[:-1]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        d = asdict(r)
        print("| " + " | ".join(f"{d[c]:.4g}" if isinstance(d[c], float) else str(d[c]) for c in cols) + " |")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--base-npy"); g.add_argument("--queries-npy")
    g.add_argument("--n-base", type=int, default=100_000); g.add_argument("--n-queries", type=int, default=1000)
    g.add_argument("--dim", type=int, default=128); g.add_argument("--seed", type=int, default=42)
    g.add_argument("--id-map", help="CSV id,row mapping router ids to base-array rows (needed if ids are UUIDs)")
    g = p.add_argument_group("modes")
    g.add_argument("--skip-local", action="store_true")
    g.add_argument("--skip-distributed", action="store_true")
    g.add_argument("--skip-managed", action="store_true")
    g = p.add_argument_group("distributed")
    g.add_argument("--router-url", default=os.environ.get("ROUTER_URL", "http://localhost:8000"))
    g.add_argument("--shards", type=int, default=4)
    g.add_argument("--replicas", type=int, default=1, help="replicas per shard actually running during the benchmark")
    g.add_argument("--index-dir", default="indexes")
    g.add_argument("--concurrency", type=int, default=16)
    g.add_argument("--http-timeout", type=float, default=5.0)
    g.add_argument("--shard-vcpu", type=float, default=1.0); g.add_argument("--shard-gb", type=float, default=2.0)
    g.add_argument("--dist-build-time-s", type=float); g.add_argument("--dist-memory-mb", type=float)
    g = p.add_argument_group("managed")
    g.add_argument("--managed", choices=["csv", "live-pinecone", "live-opensearch"], default="csv")
    g.add_argument("--managed-cost-per-query", type=float)
    g.add_argument("--managed-hourly-usd", type=float)
    g.add_argument("--managed-build-time-s", type=float)
    g = p.add_argument_group("local pricing basis")
    g.add_argument("--local-vcpu", type=float, default=1.0); g.add_argument("--local-gb", type=float, default=2.0)
    g = p.add_argument_group("paths")
    g.add_argument("--benchmarks-csv", default="results/benchmarks.csv")
    g.add_argument("--out-csv", default="results/final_comparison.csv")
    g.add_argument("--chart-dir", default="results/charts")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    pricing = Pricing.from_env()

    try:
        import faiss
        base, queries = load_data(args)
        log.info("Computing exact ground truth (FLAT) for %d queries ...", len(queries))
        flat = faiss.IndexFlatL2(base.shape[1])
        flat.add(base)
        _, gt = flat.search(queries, K)
    except Exception as exc:
        log.error("Setup failed: %s", exc)
        return 1

    rows: list[Row] = []
    failures: list[str] = []

    def attempt(name: str, fn: Callable[[], "Row | list[Row]"]) -> None:
        try:
            out = fn()
            rows.extend(out if isinstance(out, list) else [out])
        except Exception as exc:  # one mode failing must not discard the others
            log.exception("%s failed: %s", name, exc)
            failures.append(f"{name}: {exc}")

    if not args.skip_local:
        attempt("local", lambda: run_local(base, queries, gt, pricing, args.local_vcpu, args.local_gb))
    if not args.skip_distributed:
        attempt("distributed", lambda: run_distributed(args, base, queries, gt, pricing))
    if not args.skip_managed:
        attempt("managed", lambda: run_managed(args, queries, gt, pricing))

    if not rows:
        log.error("No results produced.")
        return 1

    bad = [r for r in rows if any(isinstance(getattr(r, c), float) and math.isnan(getattr(r, c)) for c in METRICS)]
    if bad:
        failures.append("NaN metrics in: " + ", ".join(f"{r.mode}/{r.index_type}" for r in bad))

    write_csv_atomic(rows, Path(args.out_csv))
    try:
        plot_charts(rows, Path(args.chart_dir))
    except Exception as exc:
        log.error("Chart rendering failed: %s", exc)
        failures.append(f"charts: {exc}")
    print_markdown(rows)
    log.info("Wrote %s and charts in %s", args.out_csv, args.chart_dir)

    if failures:
        log.error("Completed with problems:\n  - %s", "\n  - ".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
