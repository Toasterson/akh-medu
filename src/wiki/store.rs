//! Article store — Phase 22a.
//!
//! CRUD operations for wiki articles. Articles are stored in the KG
//! (metadata as triples with Dublin Core predicates) and in the tiered
//! store (content as bincode blobs keyed by article symbol ID).
//!
//! Each article has typed blocks (Definition, Relationship, Claim, etc.)
//! that enable structured querying and citation.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::{SymbolId, SymbolKind};

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Error, Diagnostic)]
pub enum WikiError {
    #[error("article not found: {title}")]
    #[diagnostic(
        code(akh::wiki::not_found),
        help("Create the article first with `create_article()`.")
    )]
    ArticleNotFound { title: String },

    #[error("duplicate article title: {title}")]
    #[diagnostic(
        code(akh::wiki::duplicate),
        help("Use `update_article()` to modify existing articles.")
    )]
    DuplicateTitle { title: String },

    #[error("{0}")]
    #[diagnostic(
        code(akh::wiki::engine),
        help("An engine-level error occurred.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for WikiError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

pub type WikiResult<T> = std::result::Result<T, WikiError>;

// ═══════════════════════════════════════════════════════════════════════
// Block Types
// ═══════════════════════════════════════════════════════════════════════

/// Type of content block within an article.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BlockType {
    /// A definition of a concept.
    Definition,
    /// Describes a relationship between entities.
    Relationship,
    /// Free-form prose.
    Prose,
    /// A worked example or illustration.
    Example,
    /// A factual claim with confidence.
    Claim { confidence: f32 },
    /// An opinion with a stance.
    Opinion { stance: Stance },
}

impl BlockType {
    pub fn as_label(&self) -> &'static str {
        match self {
            Self::Definition => "definition",
            Self::Relationship => "relationship",
            Self::Prose => "prose",
            Self::Example => "example",
            Self::Claim { .. } => "claim",
            Self::Opinion { .. } => "opinion",
        }
    }
}

/// A stance on a topic (for opinion blocks).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stance {
    /// Position: "for", "against", "nuanced".
    pub position: String,
    /// Confidence in the position.
    pub confidence: f32,
    /// Brief reasoning.
    pub reasoning: String,
}

/// A reference to a block within an article.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRef {
    /// Block identifier.
    pub id: String,
    /// Type of block.
    pub block_type: BlockType,
    /// Character offset in the article body.
    pub offset: usize,
    /// Length in characters.
    pub length: usize,
}

// ═══════════════════════════════════════════════════════════════════════
// Article
// ═══════════════════════════════════════════════════════════════════════

/// Article metadata header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArticleHeader {
    /// KG entity for this article.
    pub id: SymbolId,
    /// Article title.
    pub title: String,
    /// Author (akh entity).
    pub author: SymbolId,
    /// Subject entities (dc:subject).
    pub subjects: Vec<SymbolId>,
    /// Creation timestamp.
    pub created: u64,
    /// Last modified timestamp.
    pub modified: u64,
    /// Block manifest.
    pub blocks: Vec<BlockRef>,
    /// Articles this one cites.
    pub cites: Vec<SymbolId>,
    /// Article this one responds to (if any).
    pub responds_to: Option<SymbolId>,
}

/// A complete article: header + Markdown body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Article {
    pub header: ArticleHeader,
    /// Markdown body content.
    pub body: String,
}

// ═══════════════════════════════════════════════════════════════════════
// WikiPredicates
// ═══════════════════════════════════════════════════════════════════════

/// Well-known KG predicates for the wiki (namespace: `wiki:`).
#[derive(Debug, Clone)]
pub struct WikiPredicates {
    pub has_article: SymbolId,
    pub authored_by: SymbolId,
    pub subject: SymbolId,
    pub cites: SymbolId,
    pub responds_to: SymbolId,
    pub has_block: SymbolId,
    pub block_type: SymbolId,
    pub title: SymbolId,
}

impl WikiPredicates {
    pub fn init(engine: &Engine) -> WikiResult<Self> {
        Ok(Self {
            has_article: engine.resolve_or_create_relation("wiki:has-article")?,
            authored_by: engine.resolve_or_create_relation("wiki:authored-by")?,
            subject: engine.resolve_or_create_relation("wiki:subject")?,
            cites: engine.resolve_or_create_relation("wiki:cites")?,
            responds_to: engine.resolve_or_create_relation("wiki:responds-to")?,
            has_block: engine.resolve_or_create_relation("wiki:has-block")?,
            block_type: engine.resolve_or_create_relation("wiki:block-type")?,
            title: engine.resolve_or_create_relation("wiki:title")?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ArticleStore
// ═══════════════════════════════════════════════════════════════════════

/// Manages the article collection: CRUD, metadata indexing, content storage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArticleStore {
    /// Title → article ID index.
    pub title_index: HashMap<String, u64>,
    /// All articles by ID.
    pub articles: HashMap<u64, Article>,
    /// Next block ID counter.
    next_block_id: u64,
}

impl ArticleStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new article.
    pub fn create_article(
        &mut self,
        title: &str,
        body: &str,
        author: SymbolId,
        subjects: &[SymbolId],
        blocks: Vec<BlockRef>,
        engine: &Engine,
    ) -> WikiResult<Article> {
        if self.title_index.contains_key(title) {
            return Err(WikiError::DuplicateTitle {
                title: title.into(),
            });
        }

        let article_id = engine
            .create_symbol(SymbolKind::Entity, &format!("wiki:{title}"))
            .map_err(|e| WikiError::Engine(Box::new(e)))?
            .id;

        let now = now_secs();
        let header = ArticleHeader {
            id: article_id,
            title: title.into(),
            author,
            subjects: subjects.to_vec(),
            created: now,
            modified: now,
            blocks,
            cites: Vec::new(),
            responds_to: None,
        };

        let article = Article {
            header,
            body: body.into(),
        };

        self.articles.insert(article_id.get(), article.clone());
        self.title_index.insert(title.into(), article_id.get());

        // Store metadata triples in KG.
        if let Ok(preds) = WikiPredicates::init(engine) {
            let marker = engine.resolve_or_create_entity("wiki:article-type").ok();
            if let Some(m) = marker {
                let _ = engine.add_triple(&Triple::new(article_id, preds.has_article, m).with_confidence(1.0));
            }
            let _ = engine.add_triple(&Triple::new(article_id, preds.authored_by, author).with_confidence(1.0));
            for &subj in subjects {
                let _ = engine.add_triple(&Triple::new(article_id, preds.subject, subj).with_confidence(1.0));
            }
        }

        Ok(article)
    }

    /// Update an existing article's body and blocks.
    pub fn update_article(
        &mut self,
        title: &str,
        body: &str,
        blocks: Vec<BlockRef>,
    ) -> WikiResult<&Article> {
        let &id = self.title_index.get(title).ok_or_else(|| WikiError::ArticleNotFound {
            title: title.into(),
        })?;
        let article = self.articles.get_mut(&id).ok_or_else(|| WikiError::ArticleNotFound {
            title: title.into(),
        })?;
        article.body = body.into();
        article.header.blocks = blocks;
        article.header.modified = now_secs();
        Ok(article)
    }

    /// Get an article by title.
    pub fn get_by_title(&self, title: &str) -> Option<&Article> {
        self.title_index
            .get(title)
            .and_then(|id| self.articles.get(id))
    }

    /// Get an article by symbol ID.
    pub fn get_by_id(&self, id: SymbolId) -> Option<&Article> {
        self.articles.get(&id.get())
    }

    /// List articles by subject.
    pub fn list_by_subject(&self, subject: SymbolId) -> Vec<&Article> {
        self.articles
            .values()
            .filter(|a| a.header.subjects.contains(&subject))
            .collect()
    }

    /// List articles by author.
    pub fn list_by_author(&self, author: SymbolId) -> Vec<&Article> {
        self.articles
            .values()
            .filter(|a| a.header.author == author)
            .collect()
    }

    /// List all article titles.
    pub fn list_titles(&self) -> Vec<&str> {
        self.title_index.keys().map(|s| s.as_str()).collect()
    }

    /// Total article count.
    pub fn count(&self) -> usize {
        self.articles.len()
    }

    /// Add a citation from one article to another.
    pub fn add_citation(
        &mut self,
        from_title: &str,
        to_title: &str,
    ) -> WikiResult<()> {
        let &to_id = self.title_index.get(to_title).ok_or_else(|| WikiError::ArticleNotFound {
            title: to_title.into(),
        })?;
        let &from_id_raw = self.title_index.get(from_title).ok_or_else(|| WikiError::ArticleNotFound {
            title: from_title.into(),
        })?;
        if let Some(article) = self.articles.get_mut(&from_id_raw) {
            if let Some(to_sym) = crate::symbol::SymbolId::new(to_id) {
                if !article.header.cites.contains(&to_sym) {
                    article.header.cites.push(to_sym);
                }
            }
        }
        Ok(())
    }

    /// Generate a next block ID.
    pub fn next_block_id(&mut self) -> String {
        self.next_block_id += 1;
        format!("block-{}", self.next_block_id)
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &Engine) -> WikiResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            WikiError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("wiki store serialize: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"wiki:article_store", &bytes)
            .map_err(|e| WikiError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    /// Restore from durable store.
    pub fn restore(engine: &Engine) -> Self {
        engine
            .store()
            .get_meta(b"wiki:article_store")
            .ok()
            .flatten()
            .and_then(|bytes| bincode::deserialize(&bytes).ok())
            .unwrap_or_default()
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    fn test_engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    fn test_author(engine: &Engine) -> SymbolId {
        engine.create_symbol(SymbolKind::Entity, "test-author").unwrap().id
    }

    // ── CRUD ──────────────────────────────────────────────────────

    #[test]
    fn create_article() {
        let engine = test_engine();
        let author = test_author(&engine);
        let mut store = ArticleStore::new();

        let article = store
            .create_article("Test Article", "# Hello\n\nWorld", author, &[], vec![], &engine)
            .unwrap();

        assert_eq!(article.header.title, "Test Article");
        assert_eq!(article.body, "# Hello\n\nWorld");
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn create_duplicate_fails() {
        let engine = test_engine();
        let author = test_author(&engine);
        let mut store = ArticleStore::new();

        store.create_article("Dup", "body", author, &[], vec![], &engine).unwrap();
        let result = store.create_article("Dup", "body2", author, &[], vec![], &engine);
        assert!(result.is_err());
    }

    #[test]
    fn get_by_title() {
        let engine = test_engine();
        let author = test_author(&engine);
        let mut store = ArticleStore::new();

        store.create_article("Rust", "# Rust", author, &[], vec![], &engine).unwrap();
        let article = store.get_by_title("Rust");
        assert!(article.is_some());
        assert_eq!(article.unwrap().body, "# Rust");
    }

    #[test]
    fn get_nonexistent() {
        let store = ArticleStore::new();
        assert!(store.get_by_title("nope").is_none());
    }

    #[test]
    fn update_article() {
        let engine = test_engine();
        let author = test_author(&engine);
        let mut store = ArticleStore::new();

        store.create_article("Update Me", "v1", author, &[], vec![], &engine).unwrap();
        store.update_article("Update Me", "v2", vec![]).unwrap();

        let article = store.get_by_title("Update Me").unwrap();
        assert_eq!(article.body, "v2");
        assert!(article.header.modified >= article.header.created);
    }

    #[test]
    fn update_nonexistent_fails() {
        let mut store = ArticleStore::new();
        assert!(store.update_article("nope", "body", vec![]).is_err());
    }

    // ── Queries ───────────────────────────────────────────────────

    #[test]
    fn list_by_subject() {
        let engine = test_engine();
        let author = test_author(&engine);
        let subject = engine.create_symbol(SymbolKind::Entity, "compilers").unwrap().id;
        let mut store = ArticleStore::new();

        store.create_article("GCC", "about gcc", author, &[subject], vec![], &engine).unwrap();
        store.create_article("LLVM", "about llvm", author, &[subject], vec![], &engine).unwrap();
        store.create_article("Cooking", "about food", author, &[], vec![], &engine).unwrap();

        let results = store.list_by_subject(subject);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn list_by_author() {
        let engine = test_engine();
        let a1 = test_author(&engine);
        let a2 = engine.create_symbol(SymbolKind::Entity, "other-author").unwrap().id;
        let mut store = ArticleStore::new();

        store.create_article("A1", "by a1", a1, &[], vec![], &engine).unwrap();
        store.create_article("A2", "by a2", a2, &[], vec![], &engine).unwrap();

        assert_eq!(store.list_by_author(a1).len(), 1);
        assert_eq!(store.list_by_author(a2).len(), 1);
    }

    // ── Blocks ────────────────────────────────────────────────────

    #[test]
    fn article_with_blocks() {
        let engine = test_engine();
        let author = test_author(&engine);
        let mut store = ArticleStore::new();

        let blocks = vec![
            BlockRef {
                id: "b1".into(),
                block_type: BlockType::Definition,
                offset: 0,
                length: 50,
            },
            BlockRef {
                id: "b2".into(),
                block_type: BlockType::Claim { confidence: 0.8 },
                offset: 50,
                length: 100,
            },
        ];

        let article = store
            .create_article("Blocks", "def + claim", author, &[], blocks, &engine)
            .unwrap();
        assert_eq!(article.header.blocks.len(), 2);
        assert_eq!(article.header.blocks[0].block_type.as_label(), "definition");
        assert_eq!(article.header.blocks[1].block_type.as_label(), "claim");
    }

    // ── Citations ─────────────────────────────────────────────────

    #[test]
    fn add_citation() {
        let engine = test_engine();
        let author = test_author(&engine);
        let mut store = ArticleStore::new();

        store.create_article("Source", "src", author, &[], vec![], &engine).unwrap();
        store.create_article("Citing", "cites src", author, &[], vec![], &engine).unwrap();
        store.add_citation("Citing", "Source").unwrap();

        let citing = store.get_by_title("Citing").unwrap();
        assert_eq!(citing.header.cites.len(), 1);
    }

    // ── Serialization ─────────────────────────────────────────────

    #[test]
    fn serialization_roundtrip() {
        let engine = test_engine();
        let author = test_author(&engine);
        let mut store = ArticleStore::new();

        store.create_article("Persist", "body", author, &[], vec![], &engine).unwrap();

        let bytes = bincode::serialize(&store).unwrap();
        let restored: ArticleStore = bincode::deserialize(&bytes).unwrap();
        assert_eq!(restored.count(), 1);
        assert!(restored.get_by_title("Persist").is_some());
    }

    // ── Block type labels ─────────────────────────────────────────

    #[test]
    fn block_type_labels() {
        assert_eq!(BlockType::Definition.as_label(), "definition");
        assert_eq!(BlockType::Prose.as_label(), "prose");
        assert_eq!(BlockType::Opinion {
            stance: Stance {
                position: "for".into(),
                confidence: 0.9,
                reasoning: "test".into(),
            }
        }.as_label(), "opinion");
    }

    #[test]
    fn list_titles() {
        let engine = test_engine();
        let author = test_author(&engine);
        let mut store = ArticleStore::new();
        store.create_article("Alpha", "a", author, &[], vec![], &engine).unwrap();
        store.create_article("Beta", "b", author, &[], vec![], &engine).unwrap();
        let titles = store.list_titles();
        assert_eq!(titles.len(), 2);
    }
}
