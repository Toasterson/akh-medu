//! Core symbol types for the akh-medu engine.
//!
//! Symbols are the atomic units of the neuro-symbolic system. Every entity,
//! relation, and composite concept is identified by a [`SymbolId`] and
//! described by [`SymbolMeta`]. The [`AtomicSymbolAllocator`] provides
//! thread-safe ID generation.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::error::{AkhResult, SymbolError};

/// Unique, niche-optimized identifier for a symbol.
///
/// Uses `NonZeroU64` so that `Option<SymbolId>` is the same size as `SymbolId`
/// (the niche optimization lets the compiler use 0 as the `None` discriminant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SymbolId(NonZeroU64);

impl SymbolId {
    /// Create a `SymbolId` from a raw `u64`.
    ///
    /// Returns `None` if `raw` is zero.
    pub fn new(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(SymbolId)
    }

    /// Get the underlying `u64` value.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl std::fmt::Display for SymbolId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sym:{}", self.0)
    }
}

/// Classification of a symbol in the knowledge system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolKind {
    /// A concrete entity (person, place, concept).
    Entity,
    /// A relation between entities (e.g., "is-a", "part-of").
    Relation,
    /// A composite formed by binding/bundling other symbols.
    Composite,
    /// A hieroglyphic/semantic glyph unit, carrying a Unicode codepoint.
    Glyph {
        /// The Unicode codepoint this glyph represents.
        codepoint: char,
    },
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolKind::Entity => write!(f, "Entity"),
            SymbolKind::Relation => write!(f, "Relation"),
            SymbolKind::Composite => write!(f, "Composite"),
            SymbolKind::Glyph { codepoint } => write!(f, "Glyph(U+{:04X})", *codepoint as u32),
        }
    }
}

/// Reference to the original source material a symbol was extracted from.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceRef {
    /// Identifier for the source document.
    pub document_id: String,
    /// Index of the chunk within the document (for chunked ingestion).
    pub chunk_index: u32,
    /// Byte offset of the start of the span in the source.
    pub byte_start: usize,
    /// Byte offset of the end of the span in the source.
    pub byte_end: usize,
}

/// Metadata describing a symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMeta {
    /// Unique identifier.
    pub id: SymbolId,
    /// What kind of symbol this is.
    pub kind: SymbolKind,
    /// Human-readable label.
    pub label: String,
    /// When this symbol was created (seconds since UNIX epoch).
    pub created_at: u64,
    /// Optional reference to the source material.
    pub source: Option<SourceRef>,
}

impl SymbolMeta {
    /// Create a new `SymbolMeta` with the current timestamp.
    pub fn new(id: SymbolId, kind: SymbolKind, label: impl Into<String>) -> Self {
        Self {
            id,
            kind,
            label: label.into(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            source: None,
        }
    }

    /// Attach a source reference to this symbol.
    pub fn with_source(mut self, source: SourceRef) -> Self {
        self.source = Some(source);
        self
    }
}

/// Thread-safe symbol ID allocator.
///
/// Produces monotonically increasing IDs starting from 1.
/// Safe to share across threads via `Arc<AtomicSymbolAllocator>`.
#[derive(Debug)]
pub struct AtomicSymbolAllocator {
    next: AtomicU64,
}

impl AtomicSymbolAllocator {
    /// Create a new allocator that starts from ID 1.
    pub fn new() -> Self {
        Self {
            next: AtomicU64::new(1),
        }
    }

    /// Create an allocator that resumes from a given ID.
    ///
    /// Useful when restoring state from persistent storage.
    pub fn starting_from(start: u64) -> Self {
        Self {
            next: AtomicU64::new(start.max(1)),
        }
    }

    /// Allocate the next symbol ID.
    ///
    /// Returns an error if the ID space is exhausted (after 2^64 - 1 allocations).
    pub fn next_id(&self) -> AkhResult<SymbolId> {
        let raw = self.next.fetch_add(1, Ordering::Relaxed);
        SymbolId::new(raw).ok_or_else(|| SymbolError::AllocatorExhausted.into())
    }

    /// Return the next ID that *would* be allocated, without consuming it.
    pub fn peek_next(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }
}

impl Default for AtomicSymbolAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_id_niche_optimization() {
        // Option<SymbolId> should be the same size as SymbolId thanks to NonZeroU64.
        assert_eq!(
            std::mem::size_of::<Option<SymbolId>>(),
            std::mem::size_of::<SymbolId>()
        );
    }

    #[test]
    fn symbol_id_zero_is_none() {
        assert!(SymbolId::new(0).is_none());
        assert!(SymbolId::new(1).is_some());
        assert_eq!(SymbolId::new(42).unwrap().get(), 42);
    }

    #[test]
    fn allocator_produces_sequential_ids() {
        let alloc = AtomicSymbolAllocator::new();
        let a = alloc.next_id().unwrap();
        let b = alloc.next_id().unwrap();
        let c = alloc.next_id().unwrap();
        assert_eq!(a.get(), 1);
        assert_eq!(b.get(), 2);
        assert_eq!(c.get(), 3);
    }

    #[test]
    fn allocator_starting_from() {
        let alloc = AtomicSymbolAllocator::starting_from(100);
        assert_eq!(alloc.next_id().unwrap().get(), 100);
        assert_eq!(alloc.next_id().unwrap().get(), 101);
    }

    #[test]
    fn symbol_meta_creation() {
        let id = SymbolId::new(1).unwrap();
        let meta = SymbolMeta::new(id, SymbolKind::Entity, "Sun");
        assert_eq!(meta.id, id);
        assert_eq!(meta.kind, SymbolKind::Entity);
        assert_eq!(meta.label, "Sun");
        assert!(meta.source.is_none());
    }

    #[test]
    fn symbol_meta_with_source() {
        let id = SymbolId::new(1).unwrap();
        let meta = SymbolMeta::new(id, SymbolKind::Entity, "Sun").with_source(SourceRef {
            document_id: "doc1".into(),
            chunk_index: 0,
            byte_start: 10,
            byte_end: 13,
        });
        assert!(meta.source.is_some());
        assert_eq!(meta.source.as_ref().unwrap().document_id, "doc1");
    }

    #[test]
    fn symbol_kind_display() {
        assert_eq!(SymbolKind::Entity.to_string(), "Entity");
        assert_eq!(SymbolKind::Relation.to_string(), "Relation");
        assert_eq!(
            SymbolKind::Glyph { codepoint: '𓂀' }.to_string(),
            "Glyph(U+13080)"
        );
    }

    #[test]
    fn symbol_id_display() {
        let id = SymbolId::new(42).unwrap();
        assert_eq!(id.to_string(), "sym:42");
    }

    #[test]
    fn symbol_id_ordering() {
        let a = SymbolId::new(1).unwrap();
        let b = SymbolId::new(2).unwrap();
        assert!(a < b);
    }
}
