//! Pattern mining tool: extract code blocks from markdown/HTML, discover
//! recurring structural patterns via simplified AST fingerprinting, encode
//! them as VSA vectors, and store them in the knowledge graph.
//!
//! ## Mining workflow
//!
//! 1. Extract fenced code blocks from markdown or `<pre><code>` from HTML
//! 2. Parse each block with `syn`, simplify into [`SimplifiedAst`] skeletons
//! 3. Fingerprint each skeleton → frequency-count → filter by `min_support`
//! 4. Store patterns as KG entities with `pattern:*` predicates in `mt:patterns`
//! 5. Encode patterns as VSA vectors for similarity/analogy retrieval
//!
//! ## Analogy search
//!
//! Given pattern P applied to concept A, find what P looks like for concept B:
//! `transform = bind(P, A)`, `target = bind(transform, B)`, search item memory.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::compartment::ContextDomain;
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::{SymbolId, SymbolKind};
use crate::vsa::code_encode::{
    encode_code_vector, AstNodeTypes, AstPathContext,
};

// ---------------------------------------------------------------------------
// Code block extraction
// ---------------------------------------------------------------------------

/// A code block extracted from markdown or HTML.
#[derive(Debug, Clone)]
pub struct CodeBlock {
    /// Optional language tag from the fence or class attribute.
    pub language: Option<String>,
    /// The source code text.
    pub source: String,
    /// Surrounding prose for category inference (up to 100 chars before the block).
    pub context: String,
}

/// Extract fenced code blocks from markdown text.
///
/// Recognizes `` ```lang\n...\n``` `` patterns. Captures the optional language
/// tag and up to 100 characters of preceding text as context.
pub fn extract_code_blocks_markdown(text: &str) -> Vec<CodeBlock> {
    let mut blocks = Vec::new();
    let mut lines = text.lines().enumerate().peekable();
    let text_bytes = text.as_bytes();

    while let Some((line_idx, line)) = lines.next() {
        let trimmed = line.trim();
        if !trimmed.starts_with("```") {
            continue;
        }

        // Opening fence — extract language tag
        let after_backticks = trimmed.trim_start_matches('`');
        let language = if after_backticks.is_empty() {
            None
        } else {
            let lang = after_backticks.split_whitespace().next().unwrap_or("");
            if lang.is_empty() { None } else { Some(lang.to_string()) }
        };

        // Capture preceding context (up to 100 chars before this line)
        let line_start_byte = text.lines().take(line_idx).map(|l| l.len() + 1).sum::<usize>();
        let context_start = line_start_byte.saturating_sub(100);
        let context = if line_start_byte > 0 && context_start < text_bytes.len() {
            let end = line_start_byte.min(text_bytes.len());
            String::from_utf8_lossy(&text_bytes[context_start..end])
                .trim()
                .to_string()
        } else {
            String::new()
        };

        // Collect lines until closing fence
        let mut source_lines = Vec::new();
        let mut found_closing = false;
        for (_idx, inner_line) in lines.by_ref() {
            if inner_line.trim().starts_with("```") {
                found_closing = true;
                break;
            }
            source_lines.push(inner_line);
        }

        if found_closing && !source_lines.is_empty() {
            blocks.push(CodeBlock {
                language,
                source: source_lines.join("\n"),
                context,
            });
        }
    }

    blocks
}

/// Extract code blocks from HTML text.
///
/// Selects `<pre><code>` elements. Language is inferred from `class="language-*"`.
pub fn extract_code_blocks_html(html: &str) -> Vec<CodeBlock> {
    use scraper::{Html, Selector};

    let document = Html::parse_document(html);
    let code_sel = Selector::parse("pre code").unwrap_or_else(|_| {
        Selector::parse("code").expect("fallback selector")
    });

    let mut blocks = Vec::new();

    for element in document.select(&code_sel) {
        let source = element.text().collect::<String>();
        if source.trim().is_empty() {
            continue;
        }

        // Extract language from class attribute
        let language = element
            .value()
            .attr("class")
            .and_then(|cls| {
                cls.split_whitespace()
                    .find(|c| c.starts_with("language-"))
                    .map(|c| c.strip_prefix("language-").unwrap_or(c).to_string())
            });

        blocks.push(CodeBlock {
            language,
            source,
            context: String::new(),
        });
    }

    blocks
}

// ---------------------------------------------------------------------------
// Simplified AST
// ---------------------------------------------------------------------------

/// A structural skeleton that erases identifiers while preserving shape.
///
/// Two items with the same structure but different names produce the same
/// fingerprint, enabling frequency-based pattern discovery.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimplifiedAst {
    /// Function: param count, has return type, body sub-patterns.
    Function {
        param_count: usize,
        has_return: bool,
        body: Vec<SimplifiedAst>,
    },
    /// Struct: field count, derive count.
    Struct {
        field_count: usize,
        derive_count: usize,
    },
    /// Enum: variant count, derive count.
    Enum {
        variant_count: usize,
        derive_count: usize,
    },
    /// Impl block: method count, optional trait name (preserved for pattern identity).
    Impl {
        method_count: usize,
        trait_name: Option<String>,
    },
    /// Match expression: arm count.
    Match { arm_count: usize },
    /// If/else expression.
    IfElse { has_else: bool },
    /// For loop.
    ForLoop,
    /// Closure: param count.
    Closure { param_count: usize },
    /// Block: children sub-patterns.
    Block { children: Vec<SimplifiedAst> },
    /// Anything else.
    Other,
}

/// Convert a `syn::Item` into a simplified structural skeleton.
pub fn simplify_item(item: &syn::Item) -> SimplifiedAst {
    match item {
        syn::Item::Fn(f) => {
            let param_count = f.sig.inputs.len();
            let has_return = !matches!(f.sig.output, syn::ReturnType::Default);
            let body = simplify_block_stmts(&f.block.stmts);
            SimplifiedAst::Function { param_count, has_return, body }
        }
        syn::Item::Struct(s) => {
            let field_count = s.fields.len();
            let derive_count = count_derives(&s.attrs);
            SimplifiedAst::Struct { field_count, derive_count }
        }
        syn::Item::Enum(e) => {
            let variant_count = e.variants.len();
            let derive_count = count_derives(&e.attrs);
            SimplifiedAst::Enum { variant_count, derive_count }
        }
        syn::Item::Impl(imp) => {
            let method_count = imp.items.iter().filter(|i| matches!(i, syn::ImplItem::Fn(_))).count();
            let trait_name = imp.trait_.as_ref().map(|(_, path, _)| {
                path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()
            });
            SimplifiedAst::Impl { method_count, trait_name }
        }
        _ => SimplifiedAst::Other,
    }
}

/// Convert a `syn::Expr` into a simplified structural skeleton.
pub fn simplify_expr(expr: &syn::Expr) -> SimplifiedAst {
    match expr {
        syn::Expr::Match(m) => SimplifiedAst::Match {
            arm_count: m.arms.len(),
        },
        syn::Expr::If(i) => SimplifiedAst::IfElse {
            has_else: i.else_branch.is_some(),
        },
        syn::Expr::ForLoop(_) => SimplifiedAst::ForLoop,
        syn::Expr::Closure(c) => SimplifiedAst::Closure {
            param_count: c.inputs.len(),
        },
        syn::Expr::Block(b) => {
            let children = simplify_block_stmts(&b.block.stmts);
            SimplifiedAst::Block { children }
        }
        _ => SimplifiedAst::Other,
    }
}

/// Simplify statements in a block into sub-patterns, filtering out `Other`.
fn simplify_block_stmts(stmts: &[syn::Stmt]) -> Vec<SimplifiedAst> {
    stmts
        .iter()
        .filter_map(|stmt| {
            match stmt {
                syn::Stmt::Expr(expr, _) => {
                    let s = simplify_expr(expr);
                    if s != SimplifiedAst::Other { Some(s) } else { None }
                }
                syn::Stmt::Local(syn::Local { init: Some(syn::LocalInit { expr, .. }), .. }) => {
                    let s = simplify_expr(expr);
                    if s != SimplifiedAst::Other { Some(s) } else { None }
                }
                syn::Stmt::Item(item) => {
                    let s = simplify_item(item);
                    if s != SimplifiedAst::Other { Some(s) } else { None }
                }
                _ => None,
            }
        })
        .collect()
}

/// Count `#[derive(...)]` attributes.
fn count_derives(attrs: &[syn::Attribute]) -> usize {
    attrs
        .iter()
        .filter(|a| a.path().is_ident("derive"))
        .count()
}

/// Produce a deterministic compact fingerprint string for a [`SimplifiedAst`].
///
/// Same structure → same fingerprint, regardless of identifier names.
pub fn ast_fingerprint(ast: &SimplifiedAst) -> String {
    match ast {
        SimplifiedAst::Function { param_count, has_return, body } => {
            let ret = if *has_return { "ret" } else { "void" };
            let body_fps: Vec<String> = body.iter().map(ast_fingerprint).collect();
            if body_fps.is_empty() {
                format!("fn({param_count},{ret})")
            } else {
                format!("fn({param_count},{ret},[{}])", body_fps.join(","))
            }
        }
        SimplifiedAst::Struct { field_count, derive_count } => {
            format!("struct({field_count},d{derive_count})")
        }
        SimplifiedAst::Enum { variant_count, derive_count } => {
            format!("enum({variant_count},d{derive_count})")
        }
        SimplifiedAst::Impl { method_count, trait_name } => {
            if let Some(t) = trait_name {
                format!("impl({method_count},{t})")
            } else {
                format!("impl({method_count})")
            }
        }
        SimplifiedAst::Match { arm_count } => format!("match({arm_count})"),
        SimplifiedAst::IfElse { has_else } => {
            if *has_else { "if-else".to_string() } else { "if".to_string() }
        }
        SimplifiedAst::ForLoop => "for".to_string(),
        SimplifiedAst::Closure { param_count } => format!("closure({param_count})"),
        SimplifiedAst::Block { children } => {
            let fps: Vec<String> = children.iter().map(ast_fingerprint).collect();
            format!("block([{}])", fps.join(","))
        }
        SimplifiedAst::Other => "other".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Pattern mining
// ---------------------------------------------------------------------------

/// A structural pattern discovered by frequency analysis.
#[derive(Debug, Clone)]
pub struct MinedPattern {
    /// Deterministic fingerprint string.
    pub fingerprint: String,
    /// The simplified AST skeleton.
    pub simplified: SimplifiedAst,
    /// How many times this pattern appeared.
    pub support: u32,
    /// Inferred category (e.g., "error-handling", "iterator", "general").
    pub category: String,
    /// One concrete source example.
    pub example_source: String,
}

/// Mine structural patterns from code blocks.
///
/// Parses each block, simplifies items into fingerprints, counts frequency,
/// and returns patterns meeting `min_support`, sorted by support descending.
pub fn mine_patterns(blocks: &[CodeBlock], min_support: u32) -> Vec<MinedPattern> {
    // fingerprint → (SimplifiedAst, support, category, example_source)
    let mut freq: HashMap<String, (SimplifiedAst, u32, String, String)> = HashMap::new();

    for block in blocks {
        // Try to parse as a complete file first
        let parsed = syn::parse_file(&block.source)
            .or_else(|_| {
                // Wrap in a function body for expression fragments
                let wrapped = format!("fn __wrapper__() {{ {} }}", block.source);
                syn::parse_file(&wrapped)
            });

        let file = match parsed {
            Ok(f) => f,
            Err(_) => continue,
        };

        for item in &file.items {
            let simplified = simplify_item(item);
            if simplified == SimplifiedAst::Other {
                continue;
            }

            let fp = ast_fingerprint(&simplified);
            let category = infer_category(block, &simplified);

            freq.entry(fp.clone())
                .and_modify(|(_, count, _, _)| *count += 1)
                .or_insert((simplified.clone(), 1, category, block.source.clone()));

            // Also mine sub-item expression patterns from function bodies
            if let SimplifiedAst::Function { body, .. } = &simplified {
                for sub in body {
                    if *sub == SimplifiedAst::Other {
                        continue;
                    }
                    let sub_fp = ast_fingerprint(sub);
                    let sub_cat = infer_category(block, sub);
                    freq.entry(sub_fp)
                        .and_modify(|(_, count, _, _)| *count += 1)
                        .or_insert((sub.clone(), 1, sub_cat, block.source.clone()));
                }
            }
        }
    }

    let mut patterns: Vec<MinedPattern> = freq
        .into_iter()
        .filter(|(_, (_, support, _, _))| *support >= min_support)
        .map(|(fingerprint, (simplified, support, category, example_source))| MinedPattern {
            fingerprint,
            simplified,
            support,
            category,
            example_source,
        })
        .collect();

    patterns.sort_by(|a, b| b.support.cmp(&a.support));
    patterns.truncate(20);
    patterns
}

/// Infer a category from surrounding prose context and structural features.
pub fn infer_category(block: &CodeBlock, ast: &SimplifiedAst) -> String {
    let ctx_lower = block.context.to_lowercase();
    let src_lower = block.source.to_lowercase();

    // Error handling patterns
    if ctx_lower.contains("error") || ctx_lower.contains("diagnostic")
        || src_lower.contains("error") || src_lower.contains("result")
    {
        if let SimplifiedAst::Enum { .. } | SimplifiedAst::Match { .. } = ast {
            return "error-handling".to_string();
        }
    }

    // Iterator patterns
    if let SimplifiedAst::Impl { trait_name: Some(t), .. } = ast {
        if t == "Iterator" || t == "IntoIterator" {
            return "iterator".to_string();
        }
        if t == "From" || t == "Into" || t == "TryFrom" || t == "TryInto" {
            return "conversion".to_string();
        }
        if t == "Display" || t == "Debug" {
            return "display".to_string();
        }
    }

    // Builder pattern
    if src_lower.contains("builder") || src_lower.contains("fn build(") {
        return "builder".to_string();
    }

    // Conversion
    if ctx_lower.contains("convert") || ctx_lower.contains("from") || ctx_lower.contains("into") {
        if let SimplifiedAst::Impl { .. } = ast {
            return "conversion".to_string();
        }
    }

    "general".to_string()
}

// ---------------------------------------------------------------------------
// Pattern predicates & KG storage
// ---------------------------------------------------------------------------

/// Well-known relation SymbolIds for pattern storage.
#[derive(Debug, Clone)]
pub struct PatternPredicates {
    /// "pattern:source" — source text of the pattern example
    pub source: SymbolId,
    /// "pattern:frequency" — how often the pattern appeared
    pub frequency: SymbolId,
    /// "pattern:structure" — fingerprint string
    pub structure: SymbolId,
    /// "pattern:category" — inferred category
    pub category: SymbolId,
    /// "pattern:granularity" — "item" or "expression"
    pub granularity: SymbolId,
    /// "pattern:example" — one concrete example
    pub example: SymbolId,
}

impl PatternPredicates {
    /// Resolve or create all pattern predicates in the engine.
    fn init(engine: &Engine) -> AgentResult<Self> {
        Ok(Self {
            source: engine.resolve_or_create_relation("pattern:source")?,
            frequency: engine.resolve_or_create_relation("pattern:frequency")?,
            structure: engine.resolve_or_create_relation("pattern:structure")?,
            category: engine.resolve_or_create_relation("pattern:category")?,
            granularity: engine.resolve_or_create_relation("pattern:granularity")?,
            example: engine.resolve_or_create_relation("pattern:example")?,
        })
    }
}

/// Ensure the `mt:patterns` microtheory exists, specializing `mt:rust-code`.
fn ensure_patterns_microtheory(engine: &Engine) -> AgentResult<SymbolId> {
    let label = "mt:patterns";
    if let Ok(id) = engine.lookup_symbol(label) {
        return Ok(id);
    }

    // Ensure base code mt exists first
    let base_mt = ensure_base_code_microtheory(engine)?;

    let mt = engine
        .create_context(label, ContextDomain::Code, &[base_mt])
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("failed to create patterns microtheory: {e}"),
        })?;
    Ok(mt.id)
}

/// Ensure the base `mt:rust-code` microtheory exists (idempotent).
fn ensure_base_code_microtheory(engine: &Engine) -> AgentResult<SymbolId> {
    let label = "mt:rust-code";
    if let Ok(id) = engine.lookup_symbol(label) {
        return Ok(id);
    }
    let mt = engine
        .create_context(label, ContextDomain::Code, &[])
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("failed to create base code microtheory: {e}"),
        })?;
    Ok(mt.id)
}

/// Store a mined pattern as a KG entity with triples and provenance.
fn store_pattern(
    engine: &Engine,
    preds: &PatternPredicates,
    pattern: &MinedPattern,
    compartment: &str,
) -> AgentResult<SymbolId> {
    let label = format!("pattern:{}", pattern.fingerprint);
    let sym = engine
        .create_symbol(SymbolKind::Entity, &label)
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("failed to create pattern symbol: {e}"),
        })?;

    // Create value symbols for the metadata
    let freq_val = engine
        .create_symbol(SymbolKind::Entity, format!("val:{}", pattern.support))
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("failed to create frequency value: {e}"),
        })?;
    let cat_val = engine
        .create_symbol(SymbolKind::Entity, format!("cat:{}", pattern.category))
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("failed to create category value: {e}"),
        })?;
    let struct_val = engine
        .create_symbol(SymbolKind::Entity, format!("fp:{}", pattern.fingerprint))
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("failed to create structure value: {e}"),
        })?;

    let granularity_label = match &pattern.simplified {
        SimplifiedAst::Match { .. }
        | SimplifiedAst::IfElse { .. }
        | SimplifiedAst::ForLoop
        | SimplifiedAst::Closure { .. } => "expression",
        _ => "item",
    };
    let gran_val = engine
        .create_symbol(SymbolKind::Entity, format!("gran:{granularity_label}"))
        .map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("failed to create granularity value: {e}"),
        })?;

    // Add triples in the patterns compartment
    let comp = Some(compartment);
    let triples = [
        Triple::new(sym.id, preds.frequency, freq_val.id),
        Triple::new(sym.id, preds.category, cat_val.id),
        Triple::new(sym.id, preds.structure, struct_val.id),
        Triple::new(sym.id, preds.granularity, gran_val.id),
    ];

    for mut triple in triples {
        triple.compartment_id = comp.map(|s| s.to_string());
        engine.add_triple(&triple).map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("failed to add pattern triple: {e}"),
        })?;
    }

    // Provenance
    let mut prov = ProvenanceRecord::new(
        sym.id,
        DerivationKind::SchemaDiscovered {
            pattern_type: "mined_pattern".into(),
        },
    );
    let _ = engine.store_provenance(&mut prov);

    Ok(sym.id)
}

// ---------------------------------------------------------------------------
// VSA encoding of SimplifiedAst
// ---------------------------------------------------------------------------

/// Convert a [`SimplifiedAst`] into path-context triplets for VSA encoding.
///
/// Uses the existing `AstNodeTypes` vocabulary to produce structural path-contexts
/// that capture the shape of the pattern without specific identifiers.
pub fn extract_simplified_contexts(ast: &SimplifiedAst, label: &str) -> Vec<AstPathContext> {
    let mut contexts = Vec::new();

    match ast {
        SimplifiedAst::Function { param_count, has_return, body } => {
            // fn → param count
            contexts.push(AstPathContext::new(
                label,
                vec![AstNodeTypes::FN_DECL.to_string(), AstNodeTypes::PARAM.to_string()],
                format!("param_count:{param_count}"),
            ));

            // fn → return type presence
            if *has_return {
                contexts.push(AstPathContext::new(
                    label,
                    vec![AstNodeTypes::FN_DECL.to_string(), AstNodeTypes::RETURN_TYPE.to_string()],
                    "has_return",
                ));
            }

            // fn → body sub-patterns
            for (i, sub) in body.iter().enumerate() {
                let sub_label = format!("{label}_body{i}");
                let sub_contexts = extract_simplified_contexts(sub, &sub_label);
                // Connect fn to sub-pattern
                if !sub_contexts.is_empty() {
                    contexts.push(AstPathContext::new(
                        label,
                        vec![AstNodeTypes::FN_DECL.to_string(), AstNodeTypes::BLOCK.to_string()],
                        &sub_label,
                    ));
                }
                contexts.extend(sub_contexts);
            }
        }
        SimplifiedAst::Struct { field_count, derive_count } => {
            contexts.push(AstPathContext::new(
                label,
                vec![AstNodeTypes::STRUCT_DEF.to_string(), AstNodeTypes::FIELD.to_string()],
                format!("field_count:{field_count}"),
            ));
            if *derive_count > 0 {
                contexts.push(AstPathContext::new(
                    label,
                    vec![AstNodeTypes::STRUCT_DEF.to_string(), AstNodeTypes::ATTRIBUTE.to_string()],
                    format!("derive_count:{derive_count}"),
                ));
            }
        }
        SimplifiedAst::Enum { variant_count, derive_count } => {
            contexts.push(AstPathContext::new(
                label,
                vec![AstNodeTypes::ENUM_DEF.to_string(), AstNodeTypes::VARIANT.to_string()],
                format!("variant_count:{variant_count}"),
            ));
            if *derive_count > 0 {
                contexts.push(AstPathContext::new(
                    label,
                    vec![AstNodeTypes::ENUM_DEF.to_string(), AstNodeTypes::ATTRIBUTE.to_string()],
                    format!("derive_count:{derive_count}"),
                ));
            }
        }
        SimplifiedAst::Impl { method_count, trait_name } => {
            contexts.push(AstPathContext::new(
                label,
                vec![AstNodeTypes::IMPL_BLOCK.to_string(), AstNodeTypes::FN_DECL.to_string()],
                format!("method_count:{method_count}"),
            ));
            if let Some(t) = trait_name {
                contexts.push(AstPathContext::new(
                    label,
                    vec![AstNodeTypes::IMPL_BLOCK.to_string(), AstNodeTypes::TRAIT_DEF.to_string()],
                    t.as_str(),
                ));
            }
        }
        SimplifiedAst::Match { arm_count } => {
            contexts.push(AstPathContext::new(
                label,
                vec![AstNodeTypes::MATCH_EXPR.to_string(), AstNodeTypes::MATCH_ARM.to_string()],
                format!("arm_count:{arm_count}"),
            ));
        }
        SimplifiedAst::IfElse { has_else } => {
            let end = if *has_else { "has_else" } else { "no_else" };
            contexts.push(AstPathContext::new(
                label,
                vec![AstNodeTypes::IF_EXPR.to_string()],
                end,
            ));
        }
        SimplifiedAst::ForLoop => {
            contexts.push(AstPathContext::new(
                label,
                vec![AstNodeTypes::LOOP_EXPR.to_string()],
                "for_loop",
            ));
        }
        SimplifiedAst::Closure { param_count } => {
            contexts.push(AstPathContext::new(
                label,
                vec![AstNodeTypes::CLOSURE.to_string()],
                format!("closure_params:{param_count}"),
            ));
        }
        SimplifiedAst::Block { children } => {
            for (i, child) in children.iter().enumerate() {
                let child_label = format!("{label}_child{i}");
                contexts.extend(extract_simplified_contexts(child, &child_label));
            }
        }
        SimplifiedAst::Other => {}
    }

    contexts
}

/// Encode a mined pattern as a VSA vector and insert into item memory.
fn encode_and_store_pattern_vec(
    engine: &Engine,
    pattern: &MinedPattern,
    symbol_id: SymbolId,
) -> AgentResult<()> {
    let contexts = extract_simplified_contexts(&pattern.simplified, &pattern.fingerprint);
    if contexts.is_empty() {
        return Ok(());
    }

    let ops = engine.ops();
    let vec = encode_code_vector(ops, &contexts).map_err(|e| AgentError::ToolExecution {
        tool_name: "pattern_mine".into(),
        message: format!("VSA encoding failed: {e}"),
    })?;

    engine.item_memory().insert(symbol_id, vec);
    Ok(())
}

// ---------------------------------------------------------------------------
// Analogy search
// ---------------------------------------------------------------------------

/// Search for analogous patterns using VSA algebra.
///
/// Given a base pattern P and two concepts (source A, target B):
/// `transform = bind(P_vec, A_vec)`, `query = bind(transform, B_vec)`.
/// Returns the top-k nearest patterns to the query vector.
fn analogy_search(
    engine: &Engine,
    base_pattern: SymbolId,
    source_concept: SymbolId,
    target_concept: SymbolId,
    k: usize,
) -> AgentResult<Vec<(SymbolId, f32)>> {
    let ops = engine.ops();
    let im = engine.item_memory();

    let pattern_vec = im.get(base_pattern).ok_or_else(|| AgentError::ToolExecution {
        tool_name: "pattern_mine".into(),
        message: "base pattern has no VSA vector".into(),
    })?;

    let source_vec = im.get(source_concept).ok_or_else(|| AgentError::ToolExecution {
        tool_name: "pattern_mine".into(),
        message: "source concept has no VSA vector".into(),
    })?;

    let target_vec = im.get(target_concept).ok_or_else(|| AgentError::ToolExecution {
        tool_name: "pattern_mine".into(),
        message: "target concept has no VSA vector".into(),
    })?;

    // transform = bind(pattern, source)
    let transform = ops.bind(&pattern_vec, &source_vec).map_err(|e| AgentError::ToolExecution {
        tool_name: "pattern_mine".into(),
        message: format!("VSA bind failed: {e}"),
    })?;

    // query = bind(transform, target)
    let query = ops.bind(&transform, &target_vec).map_err(|e| AgentError::ToolExecution {
        tool_name: "pattern_mine".into(),
        message: format!("VSA bind failed: {e}"),
    })?;

    let results = engine.search_similar(&query, k).map_err(|e| AgentError::ToolExecution {
        tool_name: "pattern_mine".into(),
        message: format!("similarity search failed: {e}"),
    })?;

    Ok(results.into_iter().map(|r| (r.symbol_id, r.similarity)).collect())
}

// ---------------------------------------------------------------------------
// PatternMineTool
// ---------------------------------------------------------------------------

/// Tool for mining structural code patterns from markdown/HTML examples.
pub struct PatternMineTool;

impl Tool for PatternMineTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "pattern_mine".into(),
            description: "Mine recurring code patterns from markdown/HTML examples, encode as \
                          VSA vectors, and store in KG. Supports analogy-based retrieval."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "source".into(),
                    description: "Markdown or HTML text containing code blocks, or a file path.".into(),
                    required: true,
                },
                ToolParam {
                    name: "mode".into(),
                    description: "Mode: 'mine' (extract patterns) or 'search' (find similar patterns).".into(),
                    required: false,
                },
                ToolParam {
                    name: "min_support".into(),
                    description: "Minimum frequency threshold for pattern discovery (default: 2).".into(),
                    required: false,
                },
                ToolParam {
                    name: "language".into(),
                    description: "Filter code blocks by language tag (default: 'rust').".into(),
                    required: false,
                },
                ToolParam {
                    name: "query".into(),
                    description: "Symbol name for search mode.".into(),
                    required: false,
                },
                ToolParam {
                    name: "analogy_source".into(),
                    description: "Source concept for analogy search (e.g., 'Vec').".into(),
                    required: false,
                },
                ToolParam {
                    name: "analogy_target".into(),
                    description: "Target concept for analogy search (e.g., 'HashSet').".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let mode = input.get("mode").unwrap_or("mine");

        match mode {
            "mine" => self.execute_mine(engine, &input),
            "search" => self.execute_search(engine, &input),
            other => Ok(ToolOutput::err(format!(
                "unknown mode '{other}', expected 'mine' or 'search'"
            ))),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "pattern_mine".into(),
            description: "Mine structural code patterns from examples and store in KG with VSA encoding."
                .into(),
            parameters: vec![
                ToolParamSchema::required("source", "Markdown/HTML text or file path with code blocks."),
                ToolParamSchema::optional("mode", "Mode: 'mine' or 'search'."),
                ToolParamSchema::optional("min_support", "Minimum frequency threshold (default: 2)."),
                ToolParamSchema::optional("language", "Filter code blocks by language (default: 'rust')."),
                ToolParamSchema::optional("query", "Symbol name for search mode."),
                ToolParamSchema::optional("analogy_source", "Source concept for analogy search."),
                ToolParamSchema::optional("analogy_target", "Target concept for analogy search."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: [Capability::ReadKg, Capability::WriteKg, Capability::VsaAccess, Capability::ReadFilesystem]
                    .into_iter()
                    .collect(),
                description: "Reads text/files, parses code, writes pattern entities to KG.".into(),
                shadow_triggers: vec!["pattern".into(), "mine".into(), "learn".into()],
            },
            source: ToolSource::Native,
        }
    }
}

impl PatternMineTool {
    /// Execute mine mode: extract blocks, mine patterns, store in KG.
    fn execute_mine(&self, engine: &Engine, input: &ToolInput) -> AgentResult<ToolOutput> {
        let source_text = input.require("source", "pattern_mine")?;
        let min_support: u32 = input
            .get("min_support")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let language_filter = input.get("language").unwrap_or("rust");

        // Read from file or use inline text
        let text = if source_text.len() < 512
            && !source_text.contains('\n')
            && std::path::Path::new(source_text).exists()
        {
            std::fs::read_to_string(source_text).map_err(|e| AgentError::ToolExecution {
                tool_name: "pattern_mine".into(),
                message: format!("failed to read file: {e}"),
            })?
        } else {
            source_text.to_string()
        };

        // Detect format and extract code blocks
        let is_html = text.trim_start().starts_with('<');
        let blocks = if is_html {
            extract_code_blocks_html(&text)
        } else {
            extract_code_blocks_markdown(&text)
        };

        // Filter by language
        let filtered: Vec<&CodeBlock> = blocks
            .iter()
            .filter(|b| {
                b.language
                    .as_ref()
                    .is_none_or(|l| l.eq_ignore_ascii_case(language_filter))
            })
            .collect();

        if filtered.is_empty() {
            return Ok(ToolOutput::ok(
                "No code blocks found matching the language filter.",
            ));
        }

        // Mine patterns
        let owned_filtered: Vec<CodeBlock> = filtered.into_iter().cloned().collect();
        let patterns = mine_patterns(&owned_filtered, min_support);

        if patterns.is_empty() {
            return Ok(ToolOutput::ok(format!(
                "Extracted {} code blocks but no patterns met min_support={min_support}.",
                owned_filtered.len()
            )));
        }

        // Set up KG storage
        let preds = PatternPredicates::init(engine)?;
        let _mt = ensure_patterns_microtheory(engine)?;
        let compartment = "mt:patterns";

        let mut stored_symbols = Vec::new();
        let mut category_counts: HashMap<String, usize> = HashMap::new();

        for pattern in &patterns {
            let sym_id = store_pattern(engine, &preds, pattern, compartment)?;
            let _ = encode_and_store_pattern_vec(engine, pattern, sym_id);
            stored_symbols.push(sym_id);
            *category_counts.entry(pattern.category.clone()).or_default() += 1;
        }

        // Build summary
        let cat_summary: Vec<String> = category_counts
            .iter()
            .map(|(cat, count)| format!("{cat}: {count}"))
            .collect();

        let summary = format!(
            "Mined {} patterns from {} code blocks (min_support={min_support}).\n\
             Categories: {}\n\
             Top patterns:\n{}",
            patterns.len(),
            owned_filtered.len(),
            cat_summary.join(", "),
            patterns
                .iter()
                .take(5)
                .map(|p| format!("  - {} (support={}, cat={})", p.fingerprint, p.support, p.category))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        Ok(ToolOutput::ok_with_symbols(summary, stored_symbols))
    }

    /// Execute search mode: find similar patterns or perform analogy search.
    fn execute_search(&self, engine: &Engine, input: &ToolInput) -> AgentResult<ToolOutput> {
        let query_label = input.require("query", "pattern_mine")?;

        let query_id = engine.lookup_symbol(query_label).map_err(|_| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("symbol not found: {query_label}"),
        })?;

        // Check for analogy mode
        let analogy_source = input.get("analogy_source");
        let analogy_target = input.get("analogy_target");

        if let (Some(src_label), Some(tgt_label)) = (analogy_source, analogy_target) {
            let src_id = engine.lookup_symbol(src_label).map_err(|_| AgentError::ToolExecution {
                tool_name: "pattern_mine".into(),
                message: format!("analogy source not found: {src_label}"),
            })?;
            let tgt_id = engine.lookup_symbol(tgt_label).map_err(|_| AgentError::ToolExecution {
                tool_name: "pattern_mine".into(),
                message: format!("analogy target not found: {tgt_label}"),
            })?;

            let results = analogy_search(engine, query_id, src_id, tgt_id, 10)?;

            let summary = format!(
                "Analogy search: '{query_label}' applied from '{src_label}' to '{tgt_label}'.\n\
                 Found {} results:\n{}",
                results.len(),
                results
                    .iter()
                    .filter_map(|(id, sim)| {
                        engine.get_symbol_meta(*id).ok().map(|m| format!("  - {} (sim={sim:.3})", m.label))
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            );

            let symbols: Vec<SymbolId> = results.iter().map(|(id, _)| *id).collect();
            return Ok(ToolOutput::ok_with_symbols(summary, symbols));
        }

        // Plain similarity search
        let results = engine.search_similar_to(query_id, 10).map_err(|e| AgentError::ToolExecution {
            tool_name: "pattern_mine".into(),
            message: format!("similarity search failed: {e}"),
        })?;

        let summary = format!(
            "Similar patterns to '{query_label}':\n{}",
            results
                .iter()
                .filter_map(|r| {
                    engine.get_symbol_meta(r.symbol_id).ok().map(|m| {
                        format!("  - {} (sim={:.3})", m.label, r.similarity)
                    })
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );

        let symbols: Vec<SymbolId> = results.iter().map(|r| r.symbol_id).collect();
        Ok(ToolOutput::ok_with_symbols(summary, symbols))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;
    use crate::agent::tool::ToolInput;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    // -- Code block extraction --

    #[test]
    fn extract_markdown_blocks_basic() {
        let md = "\
Some text here.

```
let x = 42;
let y = x + 1;
```

More text.

```
fn main() {}
```
";
        let blocks = extract_code_blocks_markdown(md);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].source.contains("let x = 42"));
        assert!(blocks[1].source.contains("fn main"));
    }

    #[test]
    fn extract_markdown_blocks_with_language() {
        let md = "\
# Example

```rust
fn greet(name: &str) -> String {
    format!(\"Hello, {}\", name)
}
```

```python
def greet(name):
    return f\"Hello, {name}\"
```
";
        let blocks = extract_code_blocks_markdown(md);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].language.as_deref(), Some("rust"));
        assert_eq!(blocks[1].language.as_deref(), Some("python"));
    }

    #[test]
    fn extract_markdown_blocks_no_code() {
        let md = "Just plain text without any code blocks.";
        let blocks = extract_code_blocks_markdown(md);
        assert!(blocks.is_empty());
    }

    #[test]
    fn extract_html_blocks() {
        let html = r#"
<html><body>
<pre><code class="language-rust">fn foo() -> u32 { 42 }</code></pre>
<pre><code>let x = 1;</code></pre>
</body></html>
"#;
        let blocks = extract_code_blocks_html(html);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].language.as_deref(), Some("rust"));
        assert!(blocks[0].source.contains("fn foo"));
        assert!(blocks[1].language.is_none());
    }

    // -- SimplifiedAst --

    #[test]
    fn simplify_function_basic() {
        let code = "fn greet(name: &str) -> String { format!(\"hi {}\", name) }";
        let file = syn::parse_file(code).unwrap();
        let simplified = simplify_item(&file.items[0]);

        match simplified {
            SimplifiedAst::Function { param_count, has_return, .. } => {
                assert_eq!(param_count, 1);
                assert!(has_return);
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn simplify_struct_with_derives() {
        let code = "#[derive(Debug, Clone)]\nstruct Foo { x: u32, y: String }";
        let file = syn::parse_file(code).unwrap();
        let simplified = simplify_item(&file.items[0]);

        match simplified {
            SimplifiedAst::Struct { field_count, derive_count } => {
                assert_eq!(field_count, 2);
                assert_eq!(derive_count, 1); // one #[derive(...)] attribute
            }
            other => panic!("expected Struct, got {other:?}"),
        }
    }

    #[test]
    fn simplify_impl_with_trait() {
        let code = "impl Display for Foo { fn fmt(&self, f: &mut Formatter) -> Result { Ok(()) } }";
        let file = syn::parse_file(&format!("use std::fmt::{{Display, Formatter, Result}};\n{code}")).unwrap();
        // Find the impl item
        let impl_item = file.items.iter().find(|i| matches!(i, syn::Item::Impl(_))).unwrap();
        let simplified = simplify_item(impl_item);

        match simplified {
            SimplifiedAst::Impl { method_count, trait_name } => {
                assert_eq!(method_count, 1);
                assert_eq!(trait_name.as_deref(), Some("Display"));
            }
            other => panic!("expected Impl, got {other:?}"),
        }
    }

    // -- Fingerprinting --

    #[test]
    fn fingerprint_deterministic() {
        let ast = SimplifiedAst::Function {
            param_count: 2,
            has_return: true,
            body: vec![SimplifiedAst::Match { arm_count: 3 }],
        };
        let fp1 = ast_fingerprint(&ast);
        let fp2 = ast_fingerprint(&ast);
        assert_eq!(fp1, fp2);
        assert_eq!(fp1, "fn(2,ret,[match(3)])");
    }

    #[test]
    fn fingerprint_structural_equivalence() {
        // Two functions with different names but same structure
        let code1 = "fn greet(name: &str) -> String { format!(\"hi {}\", name) }";
        let code2 = "fn farewell(person: &str) -> String { format!(\"bye {}\", person) }";

        let file1 = syn::parse_file(code1).unwrap();
        let file2 = syn::parse_file(code2).unwrap();

        let fp1 = ast_fingerprint(&simplify_item(&file1.items[0]));
        let fp2 = ast_fingerprint(&simplify_item(&file2.items[0]));

        assert_eq!(fp1, fp2, "same structure should produce same fingerprint");
    }

    // -- Mining --

    #[test]
    fn mine_patterns_basic() {
        let blocks = vec![
            CodeBlock {
                language: Some("rust".into()),
                source: "fn a(x: u32) -> bool { x > 0 }".into(),
                context: String::new(),
            },
            CodeBlock {
                language: Some("rust".into()),
                source: "fn b(y: i64) -> bool { y < 0 }".into(),
                context: String::new(),
            },
            CodeBlock {
                language: Some("rust".into()),
                source: "fn c(z: f32) -> bool { z == 0.0 }".into(),
                context: String::new(),
            },
        ];

        let patterns = mine_patterns(&blocks, 2);
        assert!(!patterns.is_empty());
        // All three have same structure: fn(1,ret)
        let top = &patterns[0];
        assert_eq!(top.support, 3);
        assert_eq!(top.fingerprint, "fn(1,ret)");
    }

    #[test]
    fn mine_patterns_min_support() {
        let blocks = vec![
            CodeBlock {
                language: Some("rust".into()),
                source: "fn a(x: u32) -> bool { x > 0 }".into(),
                context: String::new(),
            },
            CodeBlock {
                language: Some("rust".into()),
                source: "struct Foo { x: u32 }".into(),
                context: String::new(),
            },
        ];

        // With min_support=2, nothing should pass (each appears once)
        let patterns = mine_patterns(&blocks, 2);
        assert!(patterns.is_empty());

        // With min_support=1, both should appear
        let patterns = mine_patterns(&blocks, 1);
        assert_eq!(patterns.len(), 2);
    }

    #[test]
    fn infer_category_keywords() {
        let error_block = CodeBlock {
            language: Some("rust".into()),
            source: "enum MyError { IoError, ParseError }".into(),
            context: "Error handling example:".into(),
        };
        let ast = SimplifiedAst::Enum { variant_count: 2, derive_count: 0 };
        assert_eq!(infer_category(&error_block, &ast), "error-handling");

        let general_block = CodeBlock {
            language: Some("rust".into()),
            source: "fn compute() {}".into(),
            context: "A simple function:".into(),
        };
        let gen_ast = SimplifiedAst::Function {
            param_count: 0,
            has_return: false,
            body: vec![],
        };
        assert_eq!(infer_category(&general_block, &gen_ast), "general");
    }

    // -- VSA encoding --

    #[test]
    fn encode_simplified_ast_similarity() {
        let engine = test_engine();
        let ops = engine.ops();

        // Two structurally identical functions with same label → identical vectors
        let ast1 = SimplifiedAst::Function {
            param_count: 2,
            has_return: true,
            body: vec![SimplifiedAst::Match { arm_count: 3 }],
        };

        let ctx1 = extract_simplified_contexts(&ast1, "pat");
        let ctx1b = extract_simplified_contexts(&ast1, "pat");

        let v1 = encode_code_vector(ops, &ctx1).unwrap();
        let v1b = encode_code_vector(ops, &ctx1b).unwrap();

        let sim_same = ops.similarity(&v1, &v1b).unwrap();
        assert!(
            (sim_same - 1.0).abs() < 0.01,
            "identical structure+label should produce identical vectors: {sim_same}"
        );

        // Different structure (struct vs function)
        let ast3 = SimplifiedAst::Struct { field_count: 5, derive_count: 2 };
        let ctx3 = extract_simplified_contexts(&ast3, "pat");
        let v3 = encode_code_vector(ops, &ctx3).unwrap();

        let sim_diff = ops.similarity(&v1, &v3).unwrap();
        assert!(
            sim_diff < 0.85,
            "function vs struct should differ: {sim_diff}"
        );
    }

    // -- Full tool execution --

    #[test]
    fn tool_execute_mine_mode() {
        let engine = test_engine();
        let tool = PatternMineTool;

        let md = "\
# Tutorial

```rust
fn parse(input: &str) -> Result<Value, Error> {
    todo!()
}
```

```rust
fn validate(data: &str) -> Result<bool, Error> {
    todo!()
}
```

```rust
fn transform(src: &str) -> Result<Output, Error> {
    todo!()
}
```
";
        let input = ToolInput::new()
            .with_param("source", md)
            .with_param("mode", "mine")
            .with_param("min_support", "2");

        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success, "mine should succeed: {}", output.result);
        assert!(output.result.contains("pattern"), "should mention patterns: {}", output.result);
    }

    #[test]
    fn tool_execute_search_mode() {
        let engine = test_engine();
        let tool = PatternMineTool;

        // First mine some patterns
        let md = "\
```rust
fn a(x: u32) -> bool { x > 0 }
```
```rust
fn b(y: u32) -> bool { y > 0 }
```
```rust
fn c(z: u32) -> bool { z > 0 }
```
";
        let mine_input = ToolInput::new()
            .with_param("source", md)
            .with_param("mode", "mine")
            .with_param("min_support", "2");
        let mine_out = tool.execute(&engine, mine_input).unwrap();
        assert!(mine_out.success, "mine should succeed: {}", mine_out.result);

        // Now search for the pattern
        let search_input = ToolInput::new()
            .with_param("mode", "search")
            .with_param("query", "pattern:fn(1,ret)")
            .with_param("source", "dummy"); // source required by signature

        let search_out = tool.execute(&engine, search_input).unwrap();
        // Search may or may not find results depending on item memory state,
        // but it should not error.
        assert!(search_out.success, "search should succeed: {}", search_out.result);
    }

    // -- Pattern predicates --

    #[test]
    fn pattern_predicates_init() {
        let engine = test_engine();
        let preds = PatternPredicates::init(&engine).unwrap();

        let ids = [
            preds.source, preds.frequency, preds.structure,
            preds.category, preds.granularity, preds.example,
        ];
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len(), "all pattern predicates must be unique");
    }

    #[test]
    fn patterns_microtheory_created() {
        let engine = test_engine();
        let mt = ensure_patterns_microtheory(&engine).unwrap();

        // Should be idempotent
        let mt2 = ensure_patterns_microtheory(&engine).unwrap();
        assert_eq!(mt, mt2);
    }
}
