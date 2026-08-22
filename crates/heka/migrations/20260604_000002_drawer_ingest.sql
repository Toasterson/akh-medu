-- Phase 42c: Drawer ingest path — add content-hash idempotency + session metadata.
--
-- This migration adds the columns needed by Ingest::write_drawer():
--   * body_hash   — SHA-256 of the drawer body, used for idempotent re-ingest
--   * session_id  — Claude Code session identifier (from stop-hook payload)
--   * speaker     — "user" | "assistant" | "system" (who produced this turn)
--   * contact     — freeform contact label (e.g. "claude-code@laptop")
--   * topics      — JSONB array of topic tags extracted from the turn
--   * heading     — markdown heading propagated to each generated chunk
--
-- The UNIQUE(akh_id, body_hash) constraint makes re-capturing the same
-- session a no-op: INSERT ... ON CONFLICT DO NOTHING skips rows whose
-- content hash already exists for this akh.
--
-- All statements are idempotent (IF NOT EXISTS) so re-running is safe.

-- Add ingest columns to heka_drawers.
ALTER TABLE heka_drawers
    ADD COLUMN IF NOT EXISTS body_hash  TEXT,
    ADD COLUMN IF NOT EXISTS session_id TEXT,
    ADD COLUMN IF NOT EXISTS speaker    TEXT,
    ADD COLUMN IF NOT EXISTS contact    TEXT,
    ADD COLUMN IF NOT EXISTS topics     JSONB NOT NULL DEFAULT '[]'::jsonb;

-- Preserve the chunker's propagated markdown heading instead of discarding it.
ALTER TABLE heka_drawer_chunks
    ADD COLUMN IF NOT EXISTS heading TEXT;

-- Backfill body_hash for pre-existing rows using pgcrypto's digest.
-- Rows without a body (legacy drawers created before Phase 42c) get
-- the hash of their title — good enough for idempotency on re-migrate.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'heka_drawers' AND column_name = 'body_hash'
    ) THEN
        UPDATE heka_drawers
           SET body_hash = encode(digest(COALESCE(title, ''), 'sha256'), 'hex')
         WHERE body_hash IS NULL;
    END IF;
END $$;

-- Idempotency constraint: one (akh_id, body_hash) pair per drawer.
-- Allows re-capturing the same session without producing duplicate rows.
CREATE UNIQUE INDEX IF NOT EXISTS uq_heka_drawers_akh_body_hash
    ON heka_drawers(akh_id, body_hash)
    WHERE body_hash IS NOT NULL;

-- Lookup indexes for session-level queries.
CREATE INDEX IF NOT EXISTS idx_heka_drawers_session
    ON heka_drawers(akh_id, session_id)
    WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_heka_drawers_speaker
    ON heka_drawers(speaker)
    WHERE speaker IS NOT NULL;
