//! Blog & opinion system — Phase 22f.
//!
//! Extends the wiki with:
//! - `BlogPost`: an article marked as a blog entry with a publication date
//!   and optional topic tag.
//! - `DebateThread`: a chain of response articles expressing opposing or
//!   nuanced stances on a topic.
//! - `ArchetypeVoice`: archetype-influenced writing style hints for blog
//!   post generation.
//!
//! The blog layer sits on top of `ArticleStore` — blog posts are ordinary
//! articles with `BlockType::Opinion` blocks; this module adds the query
//! and aggregation logic to treat them as a blog feed.

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::wiki::store::{Article, ArticleStore, BlockType, Stance};

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Error, Diagnostic)]
pub enum BlogError {
    #[error("article not found: {title}")]
    #[diagnostic(
        code(akh::wiki::blog::not_found),
        help("The blog post or article could not be found in the store.")
    )]
    NotFound { title: String },

    #[error("article has no opinion blocks: {title}")]
    #[diagnostic(
        code(akh::wiki::blog::no_opinion),
        help("Use `BlockType::Opinion` blocks to mark articles as opinion posts.")
    )]
    NoOpinionBlocks { title: String },

    #[error("duplicate debate entry: {title}")]
    #[diagnostic(
        code(akh::wiki::blog::duplicate),
        help("Each article can only appear once in a debate thread.")
    )]
    DuplicateDebateEntry { title: String },
}

pub type BlogResult<T> = std::result::Result<T, BlogError>;

// ═══════════════════════════════════════════════════════════════════════
// Archetype voice
// ═══════════════════════════════════════════════════════════════════════

/// Writing voice derived from Psyche archetype weights (Phase 14b).
///
/// Callers can use `ArchetypeVoice::style_hint()` to obtain a short
/// instruction string for an LLM writing prompt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ArchetypeVoice {
    /// Creator archetype: build, construct, propose new things.
    Creator,
    /// Sage archetype: analytical, definitions, theoretical depth.
    Sage,
    /// Explorer archetype: cross-domain connections, curiosity.
    Explorer,
    /// Rebel archetype: contrarian, challenge assumptions.
    Rebel,
    /// Caregiver archetype: community focus, impact on people.
    Caregiver,
}

impl ArchetypeVoice {
    /// A short style instruction suitable for an LLM writing prompt.
    pub fn style_hint(&self) -> &'static str {
        match self {
            Self::Creator => "Write with a constructive, building mindset. Use examples and propose concrete solutions.",
            Self::Sage => "Write with analytical depth. Define terms precisely and explore theory.",
            Self::Explorer => "Draw connections across domains. Express genuine curiosity and wonder.",
            Self::Rebel => "Challenge conventional wisdom. Take a contrarian stance where justified.",
            Self::Caregiver => "Center the impact on people and community. Use compassionate language.",
        }
    }

    /// Suggest which `BlockType` should dominate for this archetype.
    pub fn preferred_block_type(&self) -> BlockType {
        match self {
            Self::Creator => BlockType::Example,
            Self::Sage => BlockType::Definition,
            Self::Explorer => BlockType::Relationship,
            Self::Rebel => BlockType::Opinion {
                stance: Stance {
                    position: "against".into(),
                    confidence: 0.75,
                    reasoning: String::new(),
                },
            },
            Self::Caregiver => BlockType::Prose,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// BlogPost view
// ═══════════════════════════════════════════════════════════════════════

/// A blog post view over a wiki article.
///
/// A blog post is any article that contains at least one `Opinion` block.
/// This struct does not own the article — it holds a reference to the
/// title and the extracted stance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlogPost {
    /// Article title.
    pub title: String,
    /// Article symbol ID (numeric raw value).
    pub article_id: u64,
    /// Author symbol ID (numeric raw value).
    pub author_id: u64,
    /// Publication timestamp (Unix seconds).
    pub published: u64,
    /// Primary stance extracted from the first Opinion block.
    pub primary_stance: Option<Stance>,
    /// Number of opinion blocks.
    pub opinion_count: usize,
}

impl BlogPost {
    /// Extract a `BlogPost` view from an `Article`.
    ///
    /// Returns `None` if the article has no `Opinion` blocks.
    pub fn from_article(article: &Article) -> Option<Self> {
        let opinion_blocks: Vec<_> = article
            .header
            .blocks
            .iter()
            .filter_map(|b| {
                if let BlockType::Opinion { stance } = &b.block_type {
                    Some(stance.clone())
                } else {
                    None
                }
            })
            .collect();

        if opinion_blocks.is_empty() {
            return None;
        }

        Some(Self {
            title: article.header.title.clone(),
            article_id: article.header.id.get(),
            author_id: article.header.author.get(),
            published: article.header.created,
            primary_stance: opinion_blocks.into_iter().next(),
            opinion_count: article
                .header
                .blocks
                .iter()
                .filter(|b| matches!(b.block_type, BlockType::Opinion { .. }))
                .count(),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DebateThread
// ═══════════════════════════════════════════════════════════════════════

/// A debate thread: an ordered chain of articles all responding to a common topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateThread {
    /// Topic or root article title.
    pub topic: String,
    /// Ordered list of contributing article titles (oldest first).
    pub contributions: Vec<String>,
}

impl DebateThread {
    pub fn new(topic: impl Into<String>) -> Self {
        Self {
            topic: topic.into(),
            contributions: Vec::new(),
        }
    }

    /// Add a contribution to the thread.
    ///
    /// Errors if the article title is already present in the thread.
    pub fn add_contribution(&mut self, title: impl Into<String>) -> BlogResult<()> {
        let t = title.into();
        if self.contributions.contains(&t) {
            return Err(BlogError::DuplicateDebateEntry { title: t });
        }
        self.contributions.push(t);
        Ok(())
    }

    /// Count of contributions.
    pub fn len(&self) -> usize {
        self.contributions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contributions.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Blog feed queries
// ═══════════════════════════════════════════════════════════════════════

/// Extract all blog posts from an `ArticleStore`, sorted newest-first.
pub fn list_blog_posts(store: &ArticleStore) -> Vec<BlogPost> {
    let mut posts: Vec<BlogPost> = store
        .articles
        .values()
        .filter_map(BlogPost::from_article)
        .collect();

    posts.sort_by_key(|b| std::cmp::Reverse(b.published));
    posts
}

/// Find all articles that express a stance matching `position`
/// ("for", "against", or "nuanced").
pub fn posts_by_stance(store: &ArticleStore, position: &str) -> Vec<BlogPost> {
    list_blog_posts(store)
        .into_iter()
        .filter(|p| {
            p.primary_stance
                .as_ref()
                .map(|s| s.position == position)
                .unwrap_or(false)
        })
        .collect()
}

/// Build a `DebateThread` from a root topic article and all articles that
/// respond to it (transitively via `responds_to`).
pub fn build_debate_thread(
    store: &ArticleStore,
    root_title: &str,
) -> BlogResult<DebateThread> {
    let root = store
        .get_by_title(root_title)
        .ok_or_else(|| BlogError::NotFound {
            title: root_title.into(),
        })?;

    let root_id = root.header.id;
    let mut thread = DebateThread::new(root_title);
    thread.add_contribution(root_title)?;

    // Collect direct responders.
    let responders: Vec<_> = store
        .articles
        .values()
        .filter(|a| a.header.responds_to == Some(root_id))
        .map(|a| a.header.title.clone())
        .collect();

    for r in responders {
        thread.add_contribution(r)?;
    }

    Ok(thread)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::symbol::SymbolKind;
    use crate::wiki::store::{ArticleStore, BlockRef, BlockType, Stance};

    fn engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    fn opinion_block(position: &str, confidence: f32) -> BlockRef {
        BlockRef {
            id: "op-1".into(),
            block_type: BlockType::Opinion {
                stance: Stance {
                    position: position.into(),
                    confidence,
                    reasoning: "test reasoning".into(),
                },
            },
            offset: 0,
            length: 50,
        }
    }

    #[test]
    fn blog_post_from_opinion_article() {
        let eng = engine();
        let author = eng
            .create_symbol(SymbolKind::Entity, "blog-author")
            .unwrap()
            .id;
        let mut store = ArticleStore::new();
        let article = store
            .create_article(
                "Opinion Post",
                "I think Rust is great.",
                author,
                &[],
                vec![opinion_block("for", 0.9)],
                &eng,
            )
            .unwrap();

        let post = BlogPost::from_article(&article).unwrap();
        assert_eq!(post.title, "Opinion Post");
        assert_eq!(post.opinion_count, 1);
        assert_eq!(post.primary_stance.unwrap().position, "for");
    }

    #[test]
    fn non_opinion_article_not_blog_post() {
        let eng = engine();
        let author = eng
            .create_symbol(SymbolKind::Entity, "blog-author-2")
            .unwrap()
            .id;
        let mut store = ArticleStore::new();
        let article = store
            .create_article("Plain Article", "just facts", author, &[], vec![], &eng)
            .unwrap();
        assert!(BlogPost::from_article(&article).is_none());
    }

    #[test]
    fn list_blog_posts_newest_first() {
        let eng = engine();
        let author = eng
            .create_symbol(SymbolKind::Entity, "blog-list-author")
            .unwrap()
            .id;
        let mut store = ArticleStore::new();
        store
            .create_article(
                "Post A",
                "first",
                author,
                &[],
                vec![opinion_block("for", 0.8)],
                &eng,
            )
            .unwrap();
        store
            .create_article(
                "Post B",
                "second",
                author,
                &[],
                vec![opinion_block("against", 0.6)],
                &eng,
            )
            .unwrap();

        let posts = list_blog_posts(&store);
        assert_eq!(posts.len(), 2);
        // Newest-first: Post B has a later or equal timestamp.
        assert!(posts[0].published >= posts[1].published);
    }

    #[test]
    fn posts_by_stance_filter() {
        let eng = engine();
        let author = eng
            .create_symbol(SymbolKind::Entity, "stance-author")
            .unwrap()
            .id;
        let mut store = ArticleStore::new();
        store
            .create_article(
                "Pro Post",
                "for it",
                author,
                &[],
                vec![opinion_block("for", 0.9)],
                &eng,
            )
            .unwrap();
        store
            .create_article(
                "Anti Post",
                "against it",
                author,
                &[],
                vec![opinion_block("against", 0.7)],
                &eng,
            )
            .unwrap();

        let for_posts = posts_by_stance(&store, "for");
        assert_eq!(for_posts.len(), 1);
        assert_eq!(for_posts[0].title, "Pro Post");

        let against_posts = posts_by_stance(&store, "against");
        assert_eq!(against_posts.len(), 1);
    }

    #[test]
    fn debate_thread_build() {
        let eng = engine();
        let author = eng
            .create_symbol(SymbolKind::Entity, "debate-author")
            .unwrap()
            .id;
        let mut store = ArticleStore::new();
        // Root article.
        store
            .create_article("Root Topic", "root body", author, &[], vec![], &eng)
            .unwrap();
        let root_id = store.get_by_title("Root Topic").unwrap().header.id;
        // Response.
        store
            .create_article("Response A", "response", author, &[], vec![], &eng)
            .unwrap();
        store.set_responds_to("Response A", root_id).unwrap();

        let thread = build_debate_thread(&store, "Root Topic").unwrap();
        assert_eq!(thread.topic, "Root Topic");
        assert!(thread.len() >= 2);
        assert!(thread.contributions.contains(&"Root Topic".to_string()));
        assert!(thread.contributions.contains(&"Response A".to_string()));
    }

    #[test]
    fn debate_thread_missing_root() {
        let store = ArticleStore::new();
        assert!(matches!(
            build_debate_thread(&store, "NoSuchTopic"),
            Err(BlogError::NotFound { .. })
        ));
    }

    #[test]
    fn archetype_voice_style_hints() {
        assert!(ArchetypeVoice::Sage.style_hint().contains("analytical"));
        assert!(ArchetypeVoice::Creator.style_hint().contains("constructive"));
        assert!(ArchetypeVoice::Rebel.style_hint().contains("contrarian"));
    }
}
