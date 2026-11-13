-- NUSAskify database schema (PostgreSQL + pgvector).
-- {{EMBEDDING_DIM}} is replaced at startup with settings.embedding_dim, so the vector
-- column always matches the embedding model's dimension.

CREATE EXTENSION IF NOT EXISTS vector;

-- One uploaded file = one version of a policy document.
-- Versions of the same document share (tenant_id, lower(title)).
CREATE TABLE IF NOT EXISTS documents (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       TEXT        NOT NULL,
    title           TEXT        NOT NULL,
    version         TEXT        NOT NULL,
    valid_from      DATE        NOT NULL,
    valid_to        DATE        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'expired')),
    superseded_by   UUID        REFERENCES documents (id) ON DELETE SET NULL,
    file_name       TEXT        NOT NULL,
    file_type       TEXT        NOT NULL CHECK (file_type IN ('pdf', 'docx')),
    file_size       INTEGER     NOT NULL,
    content_sha256  TEXT        NOT NULL,
    char_count      INTEGER     NOT NULL,
    ingestion_stats JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT documents_validity_chk CHECK (valid_to >= valid_from)
);

-- A version label can only be uploaded once per document per tenant.
CREATE UNIQUE INDEX IF NOT EXISTS documents_tenant_title_version_uq
    ON documents (tenant_id, lower(title), lower(version));
-- Supports the retrieval filter (tenant + status + validity window) and document listing.
CREATE INDEX IF NOT EXISTS documents_tenant_status_validity_idx
    ON documents (tenant_id, status, valid_from, valid_to);

-- Chunks with their vectors. Each document is chunked twice (semantic + structure)
-- so the two strategies can be compared on the same data.
CREATE TABLE IF NOT EXISTS chunks (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID        NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunking_method TEXT        NOT NULL CHECK (chunking_method IN ('semantic', 'structure')),
    chunk_index     INTEGER     NOT NULL,
    content         TEXT        NOT NULL,
    char_count      INTEGER     NOT NULL,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    embedding       vector({{EMBEDDING_DIM}}) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Also serves as the index for "chunks of document X with method Y".
    CONSTRAINT chunks_document_method_index_uq UNIQUE (document_id, chunking_method, chunk_index)
);

-- Each /api/search and /api/ask call, kept for debugging (retrieval vs generation errors).
CREATE TABLE IF NOT EXISTS query_logs (
    id                  UUID             PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id           TEXT             NOT NULL,
    endpoint            TEXT             NOT NULL CHECK (endpoint IN ('search', 'ask')),
    question            TEXT             NOT NULL,
    document_ids        UUID[]           NOT NULL,
    chunking_method     TEXT             NOT NULL,
    top_k               INTEGER          NOT NULL,
    retrieved_chunk_ids UUID[]           NOT NULL DEFAULT '{}',
    retrieved_scores    DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
    status              TEXT,
    reason              TEXT,
    answer              TEXT,
    citations           JSONB,
    llm_raw_output      TEXT,
    embedding_model     TEXT             NOT NULL,
    chat_model          TEXT,
    latency_ms          INTEGER,
    created_at          TIMESTAMPTZ      NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS query_logs_tenant_created_idx
    ON query_logs (tenant_id, created_at DESC);
