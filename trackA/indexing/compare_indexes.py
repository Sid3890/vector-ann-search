from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_CSV = (
    Path(__file__).resolve().parent.parent / "results" / "benchmarks.csv"
)

# CSV column -> (table header, number format, higher_is_better)
METRICS: dict[str, tuple[str, str, bool]] = {
    "recall_at_10": ("Recall@10", "{:.4f}", True),
    "p50_ms": ("p50 (ms)", "{:.3f}", False),
    "p95_ms": ("p95 (ms)", "{:.3f}", False),
    "build_time_s": ("Build (s)", "{:.3f}", False),
    "memory_mb": ("Memory (MB)", "{:.1f}", False),
}
FAMILY_ORDER = ("FLAT", "IVF", "HNSW")

Config = dict[str, float]


def load_configs(csv_path: Path) -> dict[str, Config]:
    """Read benchmarks.csv -> {index_type: metrics}.

    benchmark.py appends a row on every run, so the same index_type can
    appear several times. Repeated runs are collapsed to their median.
    """
    runs: dict[str, list[dict[str, float]]] = defaultdict(list)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"index_type", *METRICS} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

        for row in reader:
            try:
                metrics = {m: float(row[m]) for m in METRICS}
            except (TypeError, ValueError):
                continue  # skip malformed rows rather than abort the report
            runs[row["index_type"]].append(metrics)

    return {
        name: {m: statistics.median(r[m] for r in rows) for m in METRICS}
        for name, rows in runs.items()
    }


def family_of(index_type: str) -> str:
    return index_type.split("_", 1)[0].upper()


def best_per_family(
    configs: dict[str, Config], min_recall: float
) -> dict[str, tuple[str, Config, bool]]:
    """Pick one config per family: fastest p50 with recall >= min_recall,
    else the highest-recall config. Third item flags the fallback case."""
    by_family: dict[str, list[tuple[str, Config]]] = defaultdict(list)
    for name, cfg in configs.items():
        by_family[family_of(name)].append((name, cfg))

    chosen: dict[str, tuple[str, Config, bool]] = {}
    for family, items in by_family.items():
        ok = [i for i in items if i[1]["recall_at_10"] >= min_recall]
        if ok:
            name, cfg = min(ok, key=lambda i: i[1]["p50_ms"])
            chosen[family] = (name, cfg, False)
        else:
            name, cfg = max(items, key=lambda i: i[1]["recall_at_10"])
            chosen[family] = (name, cfg, True)
    return chosen


def rank_metric(
    chosen: dict[str, tuple[str, Config, bool]], metric: str
) -> dict[str, int]:
    higher = METRICS[metric][2]
    order = sorted(
        chosen, key=lambda fam: chosen[fam][1][metric], reverse=higher
    )
    return {fam: i + 1 for i, fam in enumerate(order)}


def ranked_table(configs: dict[str, Config], min_recall: float) -> str:
    chosen = best_per_family(configs, min_recall)
    families = [f for f in FAMILY_ORDER if f in chosen]
    if not families:
        raise ValueError("No FLAT / IVF / HNSW rows found in the CSV")

    ranks = {m: rank_metric(chosen, m) for m in METRICS}
    avg = {f: sum(ranks[m][f] for m in METRICS) / len(METRICS) for f in families}
    families.sort(key=lambda f: (avg[f], chosen[f][1]["p50_ms"]))

    header = ["Overall", "Index", "Config"] + [v[0] for v in METRICS.values()]
    header.append("Avg rank")
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    for pos, fam in enumerate(families, 1):
        name, cfg, below = chosen[fam]
        cells = [
            f"{METRICS[m][1].format(cfg[m])} ({ranks[m][fam]})" for m in METRICS
        ]
        flag = " ⚠" if below else ""
        lines.append(
            f"| {pos} | **{fam}** | `{name}`{flag} | "
            + " | ".join(cells)
            + f" | {avg[fam]:.1f} |"
        )

    note = (
        f"\n_Best config per index = lowest p50 with recall >= {min_recall}; "
        "rank per metric in parentheses (1 = best). Repeated runs are "
        "collapsed to their median._"
    )
    if any(c[2] for c in chosen.values()):
        note += (
            f"\n_⚠ = no config reached recall >= {min_recall}; the "
            "highest-recall config is shown instead, so this row is not "
            "recall-matched with the others._"
        )
    return "\n".join(lines) + note


def all_configs_table(configs: dict[str, Config]) -> str:
    header = ["Index"] + [v[0] for v in METRICS.values()]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]

    def sort_key(name: str) -> tuple[int, float]:
        fam = family_of(name)
        pos = FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)
        return pos, configs[name]["p50_ms"]

    for name in sorted(configs, key=sort_key):
        cells = [METRICS[m][1].format(configs[name][m]) for m in METRICS]
        lines.append(f"| `{name}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank FLAT vs IVF vs HNSW from results/benchmarks.csv"
    )
    parser.add_argument("--csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument(
        "--all", action="store_true", help="also print every config, unranked"
    )
    args = parser.parse_args(argv)

    try:
        configs = load_configs(args.csv)
        if not configs:
            raise ValueError(f"{args.csv} has no usable rows")
        print(ranked_table(configs, args.min_recall))
        if args.all:
            print("\n" + all_configs_table(configs))
    except (OSError, ValueError) as exc:
        print(f"compare_indexes: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
