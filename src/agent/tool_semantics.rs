//! Semantic tool profiles: encode each tool's meaning as a hypervector.
//!
//! Instead of keyword matching, tools are selected by comparing a goal's
//! semantic vector against each tool's semantic profile via VSA similarity.
//!
//! Synonym expansion widens the effective vocabulary so natural-language
//! queries ("What did that paper say about gravity?") activate the right
//! tool even when few tokens overlap with the static keyword arrays.

use std::collections::HashSet;

use crate::engine::Engine;
use crate::vsa::HyperVec;
use crate::vsa::grounding::{bundle_symbols, encode_text_as_vector};
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::{VsaOps, VsaResult};

use super::tool::ToolRegistry;

/// A tool's semantic profile — a hypervector encoding what it does.
pub struct ToolProfile {
    /// Tool name (matches ToolRegistry key).
    pub name: String,
    /// Semantic vector: bundle of the tool's concept symbols.
    pub semantic_vec: HyperVec,
}

/// Semantic concept associations for each built-in tool.
///
/// These are the concepts that each tool is semantically related to.
/// The vectors are derived from KG symbols (grounded), not hardcoded keywords.
/// Synonym expansion via [`expand_with_synonyms`] widens these further at
/// profile-build time.
const TOOL_CONCEPTS: &[(&str, &[&str])] = &[
    (
        "kg_query",
        &[
            "query",
            "search",
            "knowledge",
            "triple",
            "find",
            "graph",
            "lookup",
            "explore",
            "discover",
        ],
    ),
    (
        "kg_mutate",
        &[
            "create", "add", "insert", "connect", "link", "triple", "build", "store", "write",
        ],
    ),
    (
        "memory_recall",
        &[
            "remember",
            "recall",
            "memory",
            "episode",
            "past",
            "experience",
            "history",
        ],
    ),
    (
        "reason",
        &[
            "reason",
            "logic",
            "infer",
            "deduce",
            "simplify",
            "expression",
            "symbolic",
            "analyze",
        ],
    ),
    (
        "similarity_search",
        &[
            "similar", "like", "related", "compare", "cluster", "neighbor", "analogy",
        ],
    ),
    (
        "file_io",
        &[
            "file",
            "read",
            "write",
            "save",
            "export",
            "data",
            "disk",
            "load",
            "document",
            "import",
            "open",
            "close",
            "directory",
            "folder",
            "path",
            "output",
            "input",
        ],
    ),
    (
        "http_fetch",
        &[
            "http", "url", "fetch", "web", "api", "download", "request", "network", "website",
            "page", "online", "internet", "get", "endpoint", "link", "browse",
        ],
    ),
    (
        "shell_exec",
        &[
            "command", "shell", "execute", "run", "process", "script", "system", "terminal",
            "bash", "program", "invoke", "launch", "pipe", "cli", "binary",
        ],
    ),
    (
        "user_interact",
        &[
            "ask", "user", "input", "question", "interact", "human", "prompt", "dialog", "clarify",
            "confirm", "respond", "answer", "feedback", "help",
        ],
    ),
    (
        "infer_rules",
        &[
            "infer",
            "deduce",
            "derive",
            "transitive",
            "type",
            "hierarchy",
            "classify",
            "forward",
            "chain",
            "reason",
            "logic",
            "imply",
            "conclude",
            "rule",
            "ontology",
            "propagate",
        ],
    ),
    (
        "gap_analysis",
        &[
            "gap",
            "missing",
            "incomplete",
            "discover",
            "explore",
            "what",
            "unknown",
            "coverage",
            "lack",
            "absent",
            "need",
            "require",
            "insufficient",
            "sparse",
        ],
    ),
    (
        "csv_ingest",
        &[
            "csv",
            "ingest",
            "import",
            "table",
            "data",
            "load",
            "column",
            "row",
            "spreadsheet",
        ],
    ),
    (
        "text_ingest",
        &[
            "text", "ingest", "extract", "sentence", "natural", "language", "parse", "read",
            "document",
        ],
    ),
    (
        "code_ingest",
        &[
            "code",
            "rust",
            "source",
            "parse",
            "function",
            "struct",
            "module",
            "trait",
            "architecture",
            "analyze",
        ],
    ),
    (
        "content_ingest",
        &[
            "ingest",
            "document",
            "book",
            "pdf",
            "epub",
            "html",
            "article",
            "website",
            "library",
            "read",
            "parse",
            "content",
            "import",
            "fetch",
            "download",
            "add",
            "store",
            "learn",
            "absorb",
            "paper",
            "capture",
            "save",
            "publication",
        ],
    ),
    (
        "library_search",
        &[
            "search",
            "library",
            "find",
            "document",
            "paragraph",
            "content",
            "lookup",
            "recall",
            "retrieve",
            "what",
            "about",
            "said",
            "mention",
            "topic",
            "learn",
            "reference",
            "quote",
            "knowledge",
            "look",
            "paper",
            "book",
            "article",
        ],
    ),
    (
        "doc_gen",
        &[
            "document",
            "explain",
            "describe",
            "architecture",
            "generate",
            "write",
            "summarize",
            "overview",
        ],
    ),
    (
        "code_gen",
        &[
            "code",
            "generate",
            "implement",
            "function",
            "struct",
            "enum",
            "trait",
            "module",
            "rust",
            "scaffold",
            "create",
            "write",
            "boilerplate",
        ],
    ),
    (
        "compile_feedback",
        &[
            "compile",
            "check",
            "build",
            "error",
            "diagnostic",
            "cargo",
            "validate",
            "clippy",
            "lint",
            "fix",
        ],
    ),
    (
        "pattern_mine",
        &[
            "pattern",
            "mine",
            "learn",
            "example",
            "blog",
            "tutorial",
            "template",
            "structure",
            "discover",
            "frequent",
            "analogy",
            "similar",
        ],
    ),
];

/// Static synonym lookup table mapping root words to related terms.
///
/// Expanding tool concepts with synonyms widens the effective vocabulary
/// so that natural-language queries activate the right tool even when
/// few tokens overlap with the static keyword arrays.
const SYNONYM_TABLE: &[(&str, &[&str])] = &[
    (
        "search",
        &["find", "look", "seek", "locate", "query", "browse"],
    ),
    (
        "document",
        &["paper", "article", "text", "note", "file", "book"],
    ),
    ("fetch", &["get", "retrieve", "download", "obtain", "pull"]),
    ("write", &["save", "store", "output", "export", "persist"]),
    ("read", &["load", "open", "view", "inspect", "examine"]),
    ("execute", &["run", "invoke", "launch", "start", "trigger"]),
    (
        "ask",
        &["question", "inquire", "prompt", "request", "clarify"],
    ),
    (
        "infer",
        &["deduce", "derive", "conclude", "reason", "imply"],
    ),
    (
        "knowledge",
        &["information", "data", "facts", "content", "learn"],
    ),
    (
        "missing",
        &["absent", "lacking", "incomplete", "gap", "sparse"],
    ),
    ("import", &["ingest", "load", "absorb", "capture", "add"]),
    (
        "library",
        &["collection", "catalog", "archive", "repository"],
    ),
    ("similar", &["like", "related", "analogous", "comparable"]),
    (
        "memory",
        &["recall", "remember", "history", "past", "episode"],
    ),
    (
        "create",
        &["build", "construct", "make", "generate", "produce"],
    ),
    (
        "analyze",
        &["examine", "inspect", "study", "evaluate", "assess"],
    ),
    ("command", &["shell", "terminal", "bash", "cli", "program"]),
    ("web", &["http", "url", "website", "online", "internet"]),
    ("topic", &["subject", "theme", "concept", "domain", "area"]),
    (
        "quote",
        &["excerpt", "passage", "citation", "reference", "mention"],
    ),
];

/// Expand a set of keywords with synonyms from the static lookup table.
///
/// Returns the original keywords plus any discovered synonyms, deduplicated.
pub fn expand_with_synonyms(keywords: &[&str]) -> Vec<String> {
    let mut result: HashSet<String> = keywords.iter().map(|k| k.to_string()).collect();

    for keyword in keywords {
        let kw_lower = keyword.to_lowercase();
        for (root, synonyms) in SYNONYM_TABLE {
            if kw_lower == *root || synonyms.contains(&kw_lower.as_str()) {
                result.insert(root.to_string());
                for syn in *synonyms {
                    result.insert(syn.to_string());
                }
            }
        }
    }

    result.into_iter().collect()
}

/// Build semantic profiles for all registered tools.
///
/// Each tool gets a hypervector that is the bundle of its related
/// concept symbols (expanded with synonyms), looked up or created in the engine.
pub fn build_tool_profiles(
    engine: &Engine,
    ops: &VsaOps,
    item_memory: &ItemMemory,
    tool_registry: &ToolRegistry,
) -> Vec<ToolProfile> {
    let registered: Vec<String> = tool_registry
        .list()
        .iter()
        .map(|s| s.name.clone())
        .collect();

    let mut profiles = Vec::new();

    for (tool_name, concepts) in TOOL_CONCEPTS {
        if !registered.iter().any(|n| n == tool_name) {
            continue;
        }

        // Expand concepts with synonyms for wider vocabulary coverage.
        let expanded = expand_with_synonyms(concepts);
        let expanded_refs: Vec<&str> = expanded.iter().map(|s| s.as_str()).collect();

        match bundle_symbols(engine, ops, item_memory, &expanded_refs) {
            Ok(semantic_vec) => {
                profiles.push(ToolProfile {
                    name: tool_name.to_string(),
                    semantic_vec,
                });
            }
            Err(_) => continue,
        }
    }

    // Also handle any tools not in the static list (custom tools)
    for name in &registered {
        if profiles.iter().any(|p| &p.name == name) {
            continue;
        }
        // For unknown tools, use their name and description as text
        if let Some(tool) = tool_registry.get(name) {
            let sig = tool.signature();
            let text = format!("{} {}", sig.name, sig.description);
            if let Ok(vec) = encode_text_as_vector(&text, engine, ops, item_memory) {
                profiles.push(ToolProfile {
                    name: name.clone(),
                    semantic_vec: vec,
                });
            }
        }
    }

    profiles
}

/// Score a tool's relevance to a goal vector via VSA similarity.
///
/// Returns a score in [0.0, 1.0] where 1.0 means perfect semantic match.
pub fn semantic_tool_score(goal_vec: &HyperVec, tool_profile: &ToolProfile, ops: &VsaOps) -> f32 {
    ops.similarity(goal_vec, &tool_profile.semantic_vec)
        .unwrap_or(0.5)
}

/// Encode a goal's semantics as a hypervector.
///
/// Bundles the goal description, success criteria, and related KG symbols.
pub fn encode_goal_semantics(
    goal_description: &str,
    goal_criteria: &str,
    engine: &Engine,
    ops: &VsaOps,
    item_memory: &ItemMemory,
) -> VsaResult<HyperVec> {
    let combined = format!("{} {}", goal_description, goal_criteria);
    encode_text_as_vector(&combined, engine, ops, item_memory)
}

/// Encode criteria as a hypervector for interference-based matching.
pub fn encode_criteria(
    criteria: &str,
    engine: &Engine,
    ops: &VsaOps,
    item_memory: &ItemMemory,
) -> VsaResult<HyperVec> {
    encode_text_as_vector(criteria, engine, ops, item_memory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolRegistry;
    use crate::agent::tools;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine_with_tools() -> (Engine, ToolRegistry) {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(tools::KgQueryTool));
        registry.register(Box::new(tools::KgMutateTool));
        registry.register(Box::new(tools::ReasonTool));
        registry.register(Box::new(tools::SimilaritySearchTool));

        let scratch_dir = None;
        registry.register(Box::new(tools::FileIoTool::new(scratch_dir)));

        (engine, registry)
    }

    #[test]
    fn build_profiles_for_registered_tools() {
        let (engine, registry) = test_engine_with_tools();
        let ops = engine.ops();
        let im = engine.item_memory();

        let profiles = build_tool_profiles(&engine, ops, im, &registry);
        assert!(!profiles.is_empty());
        assert!(profiles.iter().any(|p| p.name == "kg_query"));
    }

    #[test]
    fn kg_query_most_similar_to_search_goal() {
        let (engine, registry) = test_engine_with_tools();
        let ops = engine.ops();
        let im = engine.item_memory();

        let profiles = build_tool_profiles(&engine, ops, im, &registry);
        let goal_vec = encode_goal_semantics("search for knowledge", "", &engine, ops, im).unwrap();

        let kg_score = profiles
            .iter()
            .find(|p| p.name == "kg_query")
            .map(|p| semantic_tool_score(&goal_vec, p, ops))
            .unwrap_or(0.0);

        let file_score = profiles
            .iter()
            .find(|p| p.name == "file_io")
            .map(|p| semantic_tool_score(&goal_vec, p, ops))
            .unwrap_or(0.0);

        // kg_query should score higher than file_io for a search goal
        assert!(
            kg_score > file_score,
            "kg_query ({kg_score:.3}) should score higher than file_io ({file_score:.3}) for search goal"
        );
    }

    #[test]
    fn file_io_most_similar_to_file_goal() {
        let (engine, registry) = test_engine_with_tools();
        let ops = engine.ops();
        let im = engine.item_memory();

        let profiles = build_tool_profiles(&engine, ops, im, &registry);
        let goal_vec = encode_goal_semantics("read data from file", "", &engine, ops, im).unwrap();

        let file_score = profiles
            .iter()
            .find(|p| p.name == "file_io")
            .map(|p| semantic_tool_score(&goal_vec, p, ops))
            .unwrap_or(0.0);

        let reason_score = profiles
            .iter()
            .find(|p| p.name == "reason")
            .map(|p| semantic_tool_score(&goal_vec, p, ops))
            .unwrap_or(0.0);

        assert!(
            file_score > reason_score,
            "file_io ({file_score:.3}) should score higher than reason ({reason_score:.3}) for file goal"
        );
    }

    #[test]
    fn expand_with_synonyms_adds_related_terms() {
        let expanded = expand_with_synonyms(&["search", "document"]);
        // Original keywords present.
        assert!(expanded.contains(&"search".to_string()));
        assert!(expanded.contains(&"document".to_string()));
        // Synonyms of "search" added.
        assert!(expanded.contains(&"find".to_string()));
        assert!(expanded.contains(&"locate".to_string()));
        // Synonyms of "document" added.
        assert!(expanded.contains(&"paper".to_string()));
        assert!(expanded.contains(&"article".to_string()));
        // Unknown word passes through unchanged.
        let expanded2 = expand_with_synonyms(&["xyzzy"]);
        assert!(expanded2.contains(&"xyzzy".to_string()));
        assert_eq!(expanded2.len(), 1);
    }

    #[test]
    fn profiles_differentiate_between_tools() {
        let (engine, registry) = test_engine_with_tools();
        let ops = engine.ops();
        let im = engine.item_memory();

        let profiles = build_tool_profiles(&engine, ops, im, &registry);

        // Each pair of profiles should have < 1.0 similarity (they're different)
        for i in 0..profiles.len() {
            for j in (i + 1)..profiles.len() {
                let sim = ops
                    .similarity(&profiles[i].semantic_vec, &profiles[j].semantic_vec)
                    .unwrap();
                assert!(
                    sim < 0.95,
                    "Profiles {} and {} too similar: {sim:.3}",
                    profiles[i].name,
                    profiles[j].name
                );
            }
        }
    }
}
