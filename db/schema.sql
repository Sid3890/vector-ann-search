CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vectors (
    id UUID PRIMARY KEY,
    source_id TEXT,
    shard_id INT,
    embedding vector(128),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shards (
    shard_id INT PRIMARY KEY,
    centroid vector(128),
    replica_urls TEXT[],
    vector_count INT
);

CREATE TABLE IF NOT EXISTS query_log (
    id UUID PRIMARY KEY,
    latency_ms FLOAT,
    recall_estimate FLOAT,
    partial BOOL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS vectors_embedding_ivfflat_idx
ON vectors
USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);
