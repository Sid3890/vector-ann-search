CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vectors (
    id UUID PRIMARY KEY,
    job_id UUID UNIQUE NOT NULL,
    source_id TEXT,
    shard_id INT,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS shards (
    shard_id INT PRIMARY KEY,
    centroid vector(384),
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
