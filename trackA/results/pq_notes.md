# IVF-PQ results notes

## Configuration

`indexing/pq_index.py` implements `faiss.IndexIVFPQ` using the Day 2 IVF coarse
quantizer (`IndexFlatL2`, `nlist=100`) with:

- `m=16` product-quantizer sub-quantizers
- `nbits=8` bits per sub-quantizer code
- `nprobe` sweep: `1, 4, 8, 16, 32, 64, 100`
- `recall@10` against the exact `IndexFlatL2` ground truth

Run it from the project root with:

```powershell
.venv\Scripts\python.exe indexing\pq_index.py
```

Each row is appended to `results/benchmarks.csv` with `raw_bytes`,
`pq_bytes`, and `compression_ratio = raw_bytes / pq_bytes`. The benchmark
also records process RSS in `memory_mb`; RSS is a process-level footprint, while
`index_bytes` is the portable serialized FAISS index size.

## Recall trade-off

Product quantization replaces each residual vector with a compact code. This
reduces storage and usually improves cache behavior, but distance computation is
performed using reconstructed/quantized values. Consequently, recall is
expected to be lower than plain `IndexIVFFlat` at the same `nlist`/`nprobe`.
Increasing `nprobe` searches more coarse lists and can recover recall, at the
cost of higher p50/p95 latency; it cannot completely remove PQ's quantization
error.

The correct comparison is therefore a recall/latency/storage trade-off rather
than “PQ is faster” in isolation:

- compare PQ and IVF at the same or nearest recall target;
- report `compression_ratio` and `memory_mb` alongside latency;
- if no PQ setting reaches recall@10 >= 0.90, mark the highest-recall setting
  as not recall-matched, just as for the Day 3 IVF/HNSW table;
- use the serialized `pq_bytes` for storage comparison and RSS only as a
  process-footprint approximation.

The current Day 3 baseline is synthetic random data, so PQ results should be
repeated on the project’s real embedding dataset before final conclusions.
