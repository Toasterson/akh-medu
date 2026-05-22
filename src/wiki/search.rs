//! Full-text search over wiki articles — Phase 22d.
//!
//! Uses Tantivy to maintain an in-memory index over article title and body.
//! The index is rebuilt from an `ArticleStore` snapshot; it is intentionally
//! transient (rebuilt on demand) rather than persisted separately, since
//! `ArticleStore` itself is the source of truth.
//!
//! Requires the `wiki` feature flag.

#[cfg(feature = "wiki")]
mod inner {
    use miette::Diagnostic;
    use tantivy::collector::TopDocs;
    use tantivy::query::QueryParser;
    use tantivy::schema::{
        Field, Schema, Value, FAST, STORED, TEXT,
    };
    use tantivy::{Index, IndexWriter, ReloadPolicy, TantivyDocument};
    use thiserror::Error;

    use crate::wiki::store::ArticleStore;

    // ─── Error ────────────────────────────────────────────────────────────

    #[derive(Debug, Error, Diagnostic)]
    pub enum SearchError {
        #[error("tantivy error: {message}")]
        #[diagnostic(
            code(akh::wiki::search::tantivy),
            help("An internal Tantivy indexing or query error occurred.")
        )]
        Tantivy { message: String },

        #[error("query parse error: {query}")]
        #[diagnostic(
            code(akh::wiki::search::query),
            help("Check the query syntax. Use simple keywords or quoted phrases.")
        )]
        QueryParse { query: String },
    }

    impl From<tantivy::TantivyError> for SearchError {
        fn from(e: tantivy::TantivyError) -> Self {
            Self::Tantivy {
                message: e.to_string(),
            }
        }
    }

    pub type SearchResult<T> = std::result::Result<T, SearchError>;

    // ─── SearchHit ────────────────────────────────────────────────────────

    /// A single search result.
    #[derive(Debug, Clone)]
    pub struct SearchHit {
        /// Article title.
        pub title: String,
        /// Tantivy relevance score.
        pub score: f32,
        /// Article symbol ID (numeric).
        pub article_id: u64,
    }

    // ─── WikiSearch ───────────────────────────────────────────────────────

    /// Tantivy-backed full-text index over wiki articles.
    pub struct WikiSearch {
        index: Index,
        field_title: Field,
        field_body: Field,
        field_article_id: Field,
    }

    impl WikiSearch {
        /// Build a new in-memory index from the given `ArticleStore`.
        pub fn build(store: &ArticleStore) -> SearchResult<Self> {
            let mut schema_builder = Schema::builder();
            let field_title = schema_builder.add_text_field("title", TEXT | STORED);
            let field_body = schema_builder.add_text_field("body", TEXT);
            let field_article_id = schema_builder.add_u64_field("article_id", FAST | STORED);
            let schema = schema_builder.build();

            let index = Index::create_in_ram(schema.clone());
            let mut writer: IndexWriter = index.writer(50_000_000)?;

            for article in store.articles.values() {
                let mut doc = TantivyDocument::default();
                doc.add_text(field_title, &article.header.title);
                doc.add_text(field_body, &article.body);
                doc.add_u64(field_article_id, article.header.id.get());
                writer.add_document(doc)?;
            }

            writer.commit()?;

            Ok(Self {
                index,
                field_title,
                field_body,
                field_article_id,
            })
        }

        /// Execute a full-text query and return up to `limit` ranked results.
        pub fn search(&self, query_str: &str, limit: usize) -> SearchResult<Vec<SearchHit>> {
            let reader = self
                .index
                .reader_builder()
                .reload_policy(ReloadPolicy::Manual)
                .try_into()?;

            let searcher = reader.searcher();
            let query_parser =
                QueryParser::for_index(&self.index, vec![self.field_title, self.field_body]);

            let query =
                query_parser
                    .parse_query(query_str)
                    .map_err(|_| SearchError::QueryParse {
                        query: query_str.to_string(),
                    })?;

            let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

            let mut hits = Vec::new();
            for (score, doc_addr) in top_docs {
                let retrieved: TantivyDocument = searcher.doc(doc_addr)?;
                let title = retrieved
                    .get_first(self.field_title)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let article_id = retrieved
                    .get_first(self.field_article_id)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                hits.push(SearchHit {
                    title,
                    score,
                    article_id,
                });
            }

            Ok(hits)
        }

        /// Total number of documents in the index.
        pub fn doc_count(&self) -> usize {
            let reader = self
                .index
                .reader_builder()
                .reload_policy(ReloadPolicy::Manual)
                .try_into()
                .ok();
            reader
                .map(|r: tantivy::IndexReader| r.searcher().num_docs() as usize)
                .unwrap_or(0)
        }
    }

    // ─── Tests ────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::engine::{Engine, EngineConfig};
        use crate::symbol::SymbolKind;
        use crate::wiki::store::ArticleStore;

        fn engine() -> Engine {
            Engine::new(EngineConfig::default()).unwrap()
        }

        fn populated_store(eng: &Engine) -> ArticleStore {
            let author = eng
                .create_symbol(SymbolKind::Entity, "search-author")
                .unwrap()
                .id;
            let mut store = ArticleStore::new();
            store
                .create_article(
                    "Rust Programming",
                    "Rust is a systems programming language focusing on safety.",
                    author,
                    &[],
                    vec![],
                    eng,
                )
                .unwrap();
            store
                .create_article(
                    "Python Basics",
                    "Python is a high-level scripting language.",
                    author,
                    &[],
                    vec![],
                    eng,
                )
                .unwrap();
            store
                .create_article(
                    "Memory Safety",
                    "Memory safety prevents use-after-free and buffer overflows.",
                    author,
                    &[],
                    vec![],
                    eng,
                )
                .unwrap();
            store
        }

        #[test]
        fn build_index() {
            let eng = engine();
            let store = populated_store(&eng);
            let idx = WikiSearch::build(&store).unwrap();
            assert_eq!(idx.doc_count(), 3);
        }

        #[test]
        fn search_by_title_keyword() {
            let eng = engine();
            let store = populated_store(&eng);
            let idx = WikiSearch::build(&store).unwrap();
            let hits = idx.search("Rust", 10).unwrap();
            assert!(!hits.is_empty());
            assert!(hits.iter().any(|h| h.title.contains("Rust")));
        }

        #[test]
        fn search_by_body_keyword() {
            let eng = engine();
            let store = populated_store(&eng);
            let idx = WikiSearch::build(&store).unwrap();
            let hits = idx.search("safety", 10).unwrap();
            // Should match "Rust Programming" and "Memory Safety".
            assert!(hits.len() >= 1);
        }

        #[test]
        fn search_no_results() {
            let eng = engine();
            let store = populated_store(&eng);
            let idx = WikiSearch::build(&store).unwrap();
            let hits = idx.search("xyzzy_nonexistent", 10).unwrap();
            assert!(hits.is_empty());
        }

        #[test]
        fn search_returns_scores() {
            let eng = engine();
            let store = populated_store(&eng);
            let idx = WikiSearch::build(&store).unwrap();
            let hits = idx.search("programming", 10).unwrap();
            for hit in &hits {
                assert!(hit.score >= 0.0);
            }
        }

        #[test]
        fn empty_store_builds_index() {
            let store = ArticleStore::new();
            let idx = WikiSearch::build(&store).unwrap();
            assert_eq!(idx.doc_count(), 0);
            let hits = idx.search("anything", 10).unwrap();
            assert!(hits.is_empty());
        }
    }
}

#[cfg(feature = "wiki")]
pub use inner::{SearchError, SearchHit, SearchResult, WikiSearch};
