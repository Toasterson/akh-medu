CREATE SCHEMA IF NOT EXISTS sokar;
CREATE TABLE sokar.blob (
    store text NOT NULL, digest bytea NOT NULL CHECK (octet_length(digest) = 32),
    size bigint NOT NULL CHECK (size >= 0), media_type text NOT NULL,
    refcount integer NOT NULL CHECK (refcount >= 0), created_at timestamptz NOT NULL,
    PRIMARY KEY (store, digest)
);
CREATE TABLE sokar.collection (
    id uuid PRIMARY KEY, scope jsonb NOT NULL, scope_key text UNIQUE NOT NULL,
    owner text NOT NULL, store text NOT NULL, retention jsonb NOT NULL, quota jsonb NOT NULL,
    created_at timestamptz NOT NULL, manifest jsonb NOT NULL
);
CREATE TABLE sokar.artifact (
    id uuid PRIMARY KEY, collection uuid NOT NULL REFERENCES sokar.collection(id),
    path text NOT NULL, kind text NOT NULL, store text NOT NULL, digest bytea NOT NULL,
    media_type text NOT NULL, labels jsonb NOT NULL, created_by text NOT NULL,
    created_at timestamptz NOT NULL, expires_at timestamptz, previous uuid,
    deleted_at timestamptz, manifest jsonb NOT NULL,
    UNIQUE(collection, path, created_at),
    FOREIGN KEY (store, digest) REFERENCES sokar.blob(store, digest)
    -- Zero-refcount blob rows remain to preserve tombstone referential integrity.
);
CREATE INDEX artifact_page ON sokar.artifact(collection, created_at, id) WHERE deleted_at IS NULL;
CREATE INDEX artifact_path ON sokar.artifact(collection, path, created_at DESC, id DESC) WHERE deleted_at IS NULL;
CREATE INDEX artifact_blob ON sokar.artifact(store, digest) WHERE deleted_at IS NULL;
CREATE TABLE sokar.idempotency (
    principal text NOT NULL, collection uuid NOT NULL REFERENCES sokar.collection(id), key text NOT NULL,
    fingerprint text NOT NULL, response jsonb NOT NULL, expires_at timestamptz NOT NULL,
    PRIMARY KEY (principal, collection, key)
);
