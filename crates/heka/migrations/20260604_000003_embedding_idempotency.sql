-- Phase 42c repair: make embedding drain retries and concurrent workers safe.
--
-- This is deliberately a follow-up migration rather than a rewrite-only fix:
-- installations that already recorded 000002 from the rejected PR head must
-- still receive the uniqueness constraint.

-- An interrupted pre-repair worker could commit an embedding without marking
-- its chunk, and concurrent workers could insert duplicates. Keep one stable
-- row per chunk before adding the conflict arbiter used by the repaired worker.
-- Block concurrent writers for the duration of this transactional migration so
-- no duplicate can appear between the cleanup and unique-index creation.
LOCK TABLE heka_drawer_embeddings IN SHARE ROW EXCLUSIVE MODE;

DELETE FROM heka_drawer_embeddings AS duplicate
USING heka_drawer_embeddings AS canonical
WHERE duplicate.chunk_id = canonical.chunk_id
  AND duplicate.id > canonical.id;

CREATE UNIQUE INDEX IF NOT EXISTS uq_heka_embeddings_chunk
    ON heka_drawer_embeddings(chunk_id);
