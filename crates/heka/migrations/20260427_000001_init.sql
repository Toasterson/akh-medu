-- Phase 42a: Heka initial schema.
--
-- Creates 5 tables + 1 HNSW index for the Heka long-term-memory evaluation
-- store. Mirrors the Phase 37 Seshat pattern (pa_* tables).
--
-- Idempotent: all statements use IF NOT EXISTS so re-running `hekad migrate`
-- is safe. The pgvector extension must be installed on the cluster first
-- (CREATE EXTENSION vector requires superuser); the migration enables it
-- per-database.

CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- 1. heka_akhs — registered akh instances
-- ===========================================================================
CREATE TABLE IF NOT EXISTS heka_akhs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT        NOT NULL UNIQUE,
    description     TEXT,
    personality     TEXT,
    avatar_url      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ===========================================================================
-- 2. heka_project_bindings — akh ↔ project associations
-- ===========================================================================
CREATE TABLE IF NOT EXISTS heka_project_bindings (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    akh_id          UUID        NOT NULL REFERENCES heka_akhs(id) ON DELETE CASCADE,
    project_id      TEXT        NOT NULL,
    project_name    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (akh_id, project_id)
);

CREATE INDEX IF NOT EXISTS idx_heka_bindings_akh   ON heka_project_bindings(akh_id);
CREATE INDEX IF NOT EXISTS idx_heka_bindings_proj ON heka_project_bindings(project_id);

-- ===========================================================================
-- 3. heka_drawers — top-level memory entries
-- ===========================================================================
CREATE TABLE IF NOT EXISTS heka_drawers (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    akh_id          UUID        NOT NULL REFERENCES heka_akhs(id) ON DELETE CASCADE,
    project_id      TEXT,
    title           TEXT        NOT NULL,
    summary         TEXT,
    tags            JSONB       NOT NULL DEFAULT '[]'::jsonb,
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_heka_drawers_akh     ON heka_drawers(akh_id);
CREATE INDEX IF NOT EXISTS idx_heka_drawers_proj   ON heka_drawers(project_id);
-- Partial index: find drawers whose JSONB tag array is empty efficiently.
CREATE INDEX IF NOT EXISTS idx_heka_drawers_untagged
    ON heka_drawers(akh_id) WHERE tags = '[]'::jsonb;

-- ===========================================================================
-- 4. heka_drawer_chunks — text chunks within a drawer
-- ===========================================================================
CREATE TABLE IF NOT EXISTS heka_drawer_chunks (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    drawer_id       UUID        NOT NULL REFERENCES heka_drawers(id) ON DELETE CASCADE,
    chunk_index     INTEGER     NOT NULL,
    text            TEXT        NOT NULL,
    word_count      INTEGER     NOT NULL DEFAULT 0,
    embedded_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_heka_chunks_drawer ON heka_drawer_chunks(drawer_id);
-- Partial index: find unembedded chunks efficiently
CREATE INDEX IF NOT EXISTS idx_heka_chunks_unembedded
    ON heka_drawer_chunks(drawer_id) WHERE embedded_at IS NULL;

-- ===========================================================================
-- 5. heka_drawer_embeddings — 384-dim pgvector embeddings for chunks
-- ===========================================================================
CREATE TABLE IF NOT EXISTS heka_drawer_embeddings (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID        NOT NULL REFERENCES heka_drawer_chunks(id) ON DELETE CASCADE,
    model_name      TEXT        NOT NULL,
    embedding       vector(384) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_heka_embeddings_chunk
    ON heka_drawer_embeddings(chunk_id);

-- HNSW index for cosine similarity search.
--
-- HNSW (Hierarchical Navigable Small World) is preferred over IVFFlat for
-- Heka because:
--   1. No training data needed — works correctly on small/empty tables.
--   2. Better recall at low data volumes (Phase 42a has a cold start).
--   3. Incremental updates are cheap (no re-clustering).
--
-- pgvector defaults: m=16, ef_construction=64. Tuning deferred to Phase 42j.
CREATE INDEX IF NOT EXISTS idx_heka_embeddings_vector
    ON heka_drawer_embeddings
    USING hnsw (embedding vector_cosine_ops);
