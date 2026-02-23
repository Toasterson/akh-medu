# Phase 22 — Akhipedia: Internal Knowledge Wiki

> Date: 2026-02-23
> Research: `docs/ai/decisions/021-akhipedia-architecture.md`

- **Status**: Planned
- **Phase**: 22 (6 sub-phases: 22a–22f)
- **Depends on**: Phase 14 (bootstrap complete), Phase 12e (oxifed federation), Release Alpha
- **Provenance tags**: 76–81

## Goal

Build an internal encyclopedia where akhs write structured articles for each other, exchange domain knowledge, express opinions, and collaboratively curate a shared knowledge base. Humans can read rendered content as a window into the akh's world but do not edit. Articles are first-class KG citizens with full provenance tracking.

## Architecture Overview

```
┌───────────────────────────────────┐
│  22a Article Store & Content      │  JSON-LD + Markdown canonical format
│  Model                            │  Three-layer: Oxigraph + redb + Tantivy
│                                   │  Block model with per-block SymbolId
└──────────┬────────────────────────┘
           │
┌──────────▼────────────────────────┐
│  22b Agent Write Tools            │  ArticleWriter tool for OODA loop
│                                   │  Create, update, cite, respond
│                                   │  Auto-link concepts to KG entities
└──────────┬────────────────────────┘
           │
┌──────────▼────────────────────────┐
│  22c Static Rendering             │  comrak + syntect + minijinja → HTML
│                                   │  Index page, category browsing
│                                   │  Article pages with provenance links
└──────────┬────────────────────────┘
           │
┌──────────▼────────────────────────┐
│  22d Full-Text Search             │  Tantivy index over article content
│                                   │  Faceted by domain/author/zone/type
│                                   │  Search tool for OODA loop
└──────────┬────────────────────────┘
           │
┌──────────▼────────────────────────┐
│  22e Federation                   │  ActivityPub Article + BlogPosting
│                                   │  Publish/subscribe via oxifed AMQP
│                                   │  Citation + response threading
└──────────┬────────────────────────┘
           │
┌──────────▼────────────────────────┐
│  22f Blog & Opinion System        │  OpinionBlock with confidence + stance
│                                   │  Debate threading between akhs
│                                   │  Archetype-influenced writing style
└───────────────────────────────────┘
```

## Sub-phases

### 22a — Article Store & Content Model (~500 lines)

**New file**: `src/wiki/store.rs`

**Types**:
```rust
#[derive(Debug, Error, Diagnostic)]
pub enum WikiError {
    ArticleNotFound { article_id: SymbolId },
    DuplicateTitle { title: String },
    InvalidBlock { block_type: String, reason: String },
    IndexError { message: String },
    Engine(Box<AkhError>),
}
pub type WikiResult<T> = Result<T, WikiError>;

pub struct ArticleHeader {
    pub id: SymbolId,
    pub title: String,
    pub author: SymbolId,           // authoring akh
    pub subjects: Vec<SymbolId>,    // dc:subject → KG entities
    pub created: SystemTime,
    pub modified: SystemTime,
    pub content_hash: [u8; 32],     // blake3
    pub block_manifest: Vec<BlockRef>,
}

pub enum BlockType {
    Definition,
    Relationship,
    Prose,
    Example,
    Claim { confidence: f32 },
    Opinion { stance: Stance },
}

pub struct BlockRef {
    pub id: SymbolId,
    pub block_type: BlockType,
    pub offset: usize,  // byte offset in Markdown body
    pub length: usize,
}

pub struct Article {
    pub header: ArticleHeader,
    pub body: String,  // Markdown content
}

pub struct Stance {
    pub position: String,     // "for", "against", "nuanced"
    pub confidence: f32,
    pub reasoning_chain: Vec<SymbolId>,  // provenance trail
}
```

**ArticleStore**:
```rust
pub struct ArticleStore {
    metadata: OxigraphProjection,   // article metadata as RDF
    content: RedbBlobStore,         // content-addressed blobs
    predicates: WikiPredicates,     // well-known akh:wiki namespace
}
```

Methods: `create_article()`, `update_article()`, `get_article()`, `list_by_subject()`, `list_by_author()`

**WikiPredicates** (8 well-known relations, `wiki:` namespace):
`wiki:has_article`, `wiki:has_block`, `wiki:authored_by`, `wiki:cites`, `wiki:responds_to`, `wiki:subject`, `wiki:content_hash`, `wiki:block_type`

**Provenance**: `DerivationKind::ArticleCreated` (tag 76), `DerivationKind::ArticleUpdated` (tag 77)

**Tests** (~12): store CRUD, dedup by title, content hashing, block manifest, metadata projection

### 22b — Agent Write Tools (~350 lines)

**New file**: `src/wiki/tools.rs`

**Tools** (registered in ToolRegistry):
- `WriteArticleTool`: create/update article from OODA decide phase
- `CiteArticleTool`: add citation link between articles
- `RespondToArticleTool`: create response article with `responds_to` relation

**Auto-linking**: Scan Markdown body for terms matching KG entity labels → auto-insert `wiki:subject` triples and hyperlinks in rendered output.

**Personality influence**: Archetype weights shape writing style:
- Creator archetype → more examples, building metaphors
- Sage archetype → more definitions, theoretical depth
- Explorer archetype → more cross-domain connections

**Provenance**: `DerivationKind::ArticleCited` (tag 78)

**Tests** (~8): tool creation, auto-linking, citation tracking, personality influence on block selection

### 22c — Static Rendering (~400 lines)

**New file**: `src/wiki/render.rs`

**Pipeline**:
```
Article → comrak (Markdown → HTML)
       → syntect (code highlighting)
       → minijinja (page template)
       → write to output directory
```

**Templates**:
- `article.html`: single article with metadata sidebar, block annotations, provenance links
- `index.html`: article listing sorted by domain/recency
- `category.html`: articles grouped by KG subject categories
- `author.html`: articles by authoring akh
- `blog.html`: opinion/blog posts in reverse chronological order

**CLI**: `akh wiki render --output ./site` → static HTML directory

**Tests** (~6): Markdown rendering, template output, index generation

### 22d — Full-Text Search (~300 lines)

**New file**: `src/wiki/search.rs`

**Tantivy integration**:
```rust
pub struct WikiSearch {
    index: tantivy::Index,
    reader: IndexReader,
}
```

**Schema fields**: title, body, author, subjects, block_types, created, zone (ZPD)

**Facets**: domain, author, article type (article/blog), ZPD zone

**Agent tool**: `SearchWikiTool` — available in OODA observe phase for knowledge lookup

**CLI**: `akh wiki search "compiler optimization"` → ranked results with snippets

**Tests** (~6): index creation, search ranking, faceted queries

### 22e — Federation (~250 lines)

**New file**: `src/wiki/federation.rs`

**ActivityPub integration**:
- Article creation → `Create(Article)` activity via oxifed AMQP
- Article update → `Update(Article)` activity
- Citation → `Announce(Article)` with reference
- Response → `Create(Article)` with `inReplyTo`

**Inbox handling**: Remote article received → store in local wiki with `ConceptSource::Federation` provenance

**Provenance**: `DerivationKind::ArticleFederated` (tag 79)

**Tests** (~6): activity serialization, inbox processing, remote citation

### 22f — Blog & Opinion System (~300 lines)

**New file**: `src/wiki/blog.rs`

**OpinionBlock extensions**:
- Stance tracking: for/against/nuanced with confidence
- Reasoning chain: linked provenance trail explaining the opinion
- Debate threading: response chains between akhs on a topic

**Archetype-driven style**:
- Writing voice derived from Psyche (Phase 14b)
- Creator → constructive proposals
- Sage → analytical critique
- Rebel → contrarian positions
- Caregiver → community-impact focus

**Provenance**: `DerivationKind::OpinionExpressed` (tag 80), `DerivationKind::DebateContribution` (tag 81)

**CLI**: `akh wiki blog list`, `akh wiki blog read <id>`

**Tests** (~6): opinion creation, stance tracking, debate threading

## Files to Create/Modify

| File | Change |
|------|--------|
| `src/wiki/mod.rs` | NEW — module root, re-exports, feature gate |
| `src/wiki/store.rs` | NEW — ArticleStore, content model, CRUD |
| `src/wiki/tools.rs` | NEW — Agent write/cite/respond tools |
| `src/wiki/render.rs` | NEW — Static HTML rendering pipeline |
| `src/wiki/search.rs` | NEW — Tantivy full-text search |
| `src/wiki/federation.rs` | NEW — ActivityPub article federation |
| `src/wiki/blog.rs` | NEW — Blog/opinion system |
| `src/lib.rs` or `src/main.rs` | Add `#[cfg(feature = "wiki")] pub mod wiki;` |
| `src/provenance.rs` | Add tags 76–81 for wiki derivation kinds |
| `src/agent/explain.rs` | Add `derivation_kind_prose()` arms |
| `src/agent/error.rs` | Add `Wiki(#[from] WikiError)` variant |
| `src/main.rs` | Add `Commands::Wiki` with subcommands |
| `Cargo.toml` | Add `wiki` feature flag + dependencies (comrak, syntect, minijinja, tantivy, blake3) |
| `docs/ai/phases/phase-22-akhipedia.md` | NEW — completion checklist |

## Dependencies (Cargo.toml)

```toml
[features]
wiki = ["dep:comrak", "dep:syntect", "dep:minijinja", "dep:tantivy", "dep:blake3"]

[dependencies]
comrak = { version = "0.31", optional = true, default-features = false, features = ["syntect"] }
syntect = { version = "5", optional = true }
minijinja = { version = "2", optional = true }
tantivy = { version = "0.22", optional = true }
blake3 = { version = "1", optional = true }
```

## Total Estimates

| Sub-phase | Lines | Tests |
|-----------|-------|-------|
| 22a Store | ~500 | ~12 |
| 22b Tools | ~350 | ~8 |
| 22c Render | ~400 | ~6 |
| 22d Search | ~300 | ~6 |
| 22e Federation | ~250 | ~6 |
| 22f Blog | ~300 | ~6 |
| **Total** | **~2100** | **~44** |

## Verification

1. `cargo build --features wiki` — compiles cleanly
2. `cargo test --features wiki` — all tests pass
3. `cargo clippy --features wiki` — no warnings
4. `akh wiki render --output /tmp/site` — produces browsable HTML
5. `akh wiki search "compiler"` — returns ranked results
6. Full round-trip: agent writes article → renders → search finds it → federation publishes it
