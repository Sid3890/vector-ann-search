"""Check shard indexes + per-shard id maps against the original embeddings.

  python tests/verify_shards.py --base-npy emb.npy --ids-npy ids.npy

Assumes shard_<id>.ids.npy (or shard_<id>_ids.npy) holds the ORIGINAL ids (e.g. UUID strings) in the
same order the vectors were added to shard_<id>.faiss, and ids.npy holds the original id of each row of emb.npy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-npy", required=True)
    p.add_argument("--ids-npy", required=True)
    p.add_argument("--index-dir", default="indexes")
    p.add_argument("--samples", type=int, default=50)
    a = p.parse_args()

    base = np.load(a.base_npy).astype("float32")
    ids = [str(x) for x in np.load(a.ids_npy, allow_pickle=False).tolist()]
    if len(ids) != len(base):
        print(f"FAIL: ids ({len(ids)}) and embeddings ({len(base)}) differ in length")
        return 1
    row_of = {i: r for r, i in enumerate(ids)}
    print(f"base: {base.shape}, unique ids: {len(row_of)}")

    files = sorted(Path(a.index_dir).glob("shard_*.faiss"))
    if not files:
        print("FAIL: no shard_*.faiss found")
        return 1

    ok_all, total = len(row_of) == len(ids), 0
    seen: list[str] = []
    for f in files:
        sid = f.stem.split("_")[1]
        idx = faiss.read_index(str(f))
        total += idx.ntotal
        if idx.d != base.shape[1]:
            print(f"shard {sid}: FAIL index dim {idx.d} != base dim {base.shape[1]}")
            ok_all = False
            continue
        ids_file = next((f.parent / n for n in (f"shard_{sid}.ids.npy", f"shard_{sid}_ids.npy")
                         if (f.parent / n).exists()), None)
        if ids_file is None:
            print(f"shard {sid}: ntotal={idx.ntotal} NO ID MAP")
            ok_all = False
            continue
        sids = [str(x) for x in np.load(ids_file, allow_pickle=False).tolist()]
        seen += sids
        if len(sids) != idx.ntotal:
            print(f"shard {sid}: FAIL id map length {len(sids)} != ntotal {idx.ntotal}")
            ok_all = False
            continue
        unknown = [s for s in sids if s not in row_of]
        if unknown:
            print(f"shard {sid}: FAIL {len(unknown)} ids not present in ids.npy (e.g. {unknown[0]})")
            ok_all = False
            continue
        rng = np.random.default_rng(0)
        pos = rng.choice(idx.ntotal, size=min(a.samples, idx.ntotal), replace=False)
        match = sum(np.allclose(idx.reconstruct(int(i)), base[row_of[sids[int(i)]]], atol=1e-5) for i in pos)
        good = match == len(pos)
        ok_all &= good
        print(f"shard {sid}: ntotal={idx.ntotal} id_map={ids_file.name} reconstruct-match {match}/{len(pos)} "
              f"{'OK' if good else 'MISMATCH'}")

    dup = len(seen) - len(set(seen))
    print(f"total vectors across shards: {total} (base has {len(base)}); duplicate ids across shards: {dup}")
    ok_all &= dup == 0 and total == len(base)
    print("RESULT:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
