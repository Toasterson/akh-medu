//! Purge contaminated documents from the Tecton workspace corpus.
//!
//! The Tecton workspace (a digital-architect home) accumulated garbage during
//! early awakening / agent ingest runs:
//!   - Biomedical research papers (salivary duct carcinoma, Ebola coagulation,
//!     lipid nanoparticles) pulled in via unbounded library_search fetching.
//!   - Web-scraped password / auth UI strings.
//!
//! CASCADE foreign-keys mean deleting from `pa_documents` automatically cleans
//! up the child `pa_chunks`, `pa_embeddings`, and related `pa_microtheory_cache`
//! entries — no manual child-table deletes needed.
//!
//! `down()` is intentionally a no-op: the data was garbage, can't be restored.

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260521_000001_purge_tecton_garbage"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // Purge biomedical papers injected by unbounded web scraping.
        // Patterns are deliberately narrow (regex, case-insensitive) to avoid
        // matching legitimate architectural content.
        db.execute_unprepared(
            "DELETE FROM pa_documents
             WHERE
                 -- Salivary duct carcinoma papers
                 lower(title) ~ 'salivary.*(duct|gland|carcinoma)'
                 -- Ebola coagulation / haematology papers
                 OR lower(title) ~ 'ebola.*(coagulat|disseminat|hemorrhag|haemorrhag|fibrinogen)'
                 OR lower(title) ~ '(coagulat|disseminat|hemorrhag).*(ebola|filovirus|vhf)'
                 -- Lipid nanoparticle / drug delivery papers
                 OR lower(title) ~ 'lipid.{0,10}nanoparticle'
                 OR lower(title) ~ 'nanoparticle.{0,20}(delivery|encapsulat|formulation)'
                 -- Generic web-scraped UI / auth strings (exact short matches)
                 OR lower(title) ~ '^(sign in|log in|login|log out|logout|forgot password|reset password|password reset|create account|register|sign up|create an account|remember me)$'",
        )
        .await?;

        // Purge microtheory cache entries that referenced the now-deleted
        // chunks — orphaned rows from prior synthesis runs.  The chunk_ids_used
        // column is a JSONB array of chunk UUIDs; any row whose array intersects
        // no remaining chunk is stale.
        db.execute_unprepared(
            "DELETE FROM pa_microtheory_cache
             WHERE NOT EXISTS (
                 SELECT 1
                 FROM pa_chunks c
                 WHERE chunk_ids_used @> to_jsonb(c.id::text)
             )",
        )
        .await?;

        Ok(())
    }

    async fn down(&self, _manager: &SchemaManager) -> Result<(), DbErr> {
        // Purged data was garbage — restoration is neither possible nor desired.
        Ok(())
    }
}
