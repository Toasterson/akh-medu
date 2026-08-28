-- Migration 000005: per-akh bearer token hashes for the MCP tenant route.
--
-- The Heka MCP surface authenticates each /akh/{akh_id}/mcp request with an
-- `Authorization: Bearer <token>` header. Only the SHA-256 hex digest of the
-- token is stored — the plaintext token is shown to the operator exactly once
-- at mint time (`hekad register-akh --token` / `hekad set-token`).
--
-- The partial unique index guarantees two akhs can never share a token hash,
-- which keeps the auth lookup (token_hash -> akh row) to at most one row.

ALTER TABLE heka_akhs
    ADD COLUMN IF NOT EXISTS token_hash TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS uq_heka_akhs_token_hash
    ON heka_akhs(token_hash)
    WHERE token_hash IS NOT NULL;
