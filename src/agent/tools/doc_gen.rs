//! Documentation generation tool: query the KG for code structure and produce
//! Markdown and/or JSON documentation.
//!
//! Supports multiple targets: full architecture overview, single module, single
//! type, or dependency graph. Optionally polishes Markdown output via LLM.

use std::collections::BTreeMap;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::engine::Engine;
use crate::symbol::SymbolId;

use super::code_predicates::CodePredicates;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// What to document.
#[derive(Debug, Clone)]
pub enum DocTarget {
    /// Full codebase overview.
    Architecture,
    /// Single module by name.
    Module { name: String },
    /// Single type (struct/enum/trait) by name.
    Type { name: String },
    /// Dependency graph report.
    Dependencies,
}

impl DocTarget {
    fn parse(s: &str) -> Self {
        let s = s.trim();
        if s.eq_ignore_ascii_case("architecture") || s.eq_ignore_ascii_case("arch") {
            Self::Architecture
        } else if s.eq_ignore_ascii_case("dependencies") || s.eq_ignore_ascii_case("deps") {
            Self::Dependencies
        } else if let Some(name) = s.strip_prefix("module:") {
            Self::Module {
                name: name.trim().to_string(),
            }
        } else if let Some(name) = s.strip_prefix("type:") {
            Self::Type {
                name: name.trim().to_string(),
            }
        } else {
            // Default: treat as type name.
            Self::Type {
                name: s.to_string(),
            }
        }
    }
}

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocFormat {
    Markdown,
    Json,
    Both,
}

impl DocFormat {
    fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "json" => Self::Json,
            "both" => Self::Both,
            _ => Self::Markdown,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal doc model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct DocSection {
    heading: String,
    body: String,
    source_symbols: Vec<u64>,
    subsections: Vec<DocSection>,
}

impl DocSection {
    fn new(heading: impl Into<String>) -> Self {
        Self {
            heading: heading.into(),
            body: String::new(),
            source_symbols: Vec::new(),
            subsections: Vec::new(),
        }
    }

    fn with_body(mut self, body: impl Into<String>) -> Self {
        self.body = body.into();
        self
    }

    fn with_symbols(mut self, syms: Vec<u64>) -> Self {
        self.source_symbols = syms;
        self
    }

    fn with_subsection(mut self, sub: DocSection) -> Self {
        self.subsections.push(sub);
        self
    }
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Generate documentation from the code knowledge in the KG.
pub struct DocGenTool;

impl Tool for DocGenTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "doc_gen".into(),
            description: "Generate documentation from code knowledge in the KG. \
                          Supports architecture overview, module docs, type docs, \
                          and dependency reports."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "target".into(),
                    description: "What to document: 'architecture', 'module:<name>', \
                                  'type:<name>', or 'dependencies'."
                        .into(),
                    required: true,
                },
                ToolParam {
                    name: "format".into(),
                    description: "Output format: 'markdown' (default), 'json', or 'both'.".into(),
                    required: false,
                },
                ToolParam {
                    name: "polish".into(),
                    description: "Use LLM to polish Markdown output. Default: false.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let target_str = input.require("target", "doc_gen")?;
        let format = input
            .get("format")
            .map(DocFormat::parse)
            .unwrap_or(DocFormat::Markdown);
        let polish: bool = input
            .get("polish")
            .and_then(|s| s.parse().ok())
            .unwrap_or(false);

        let target = DocTarget::parse(target_str);
        let preds = CodePredicates::init(engine)?;

        let sections = match &target {
            DocTarget::Architecture => build_architecture_doc(engine, &preds),
            DocTarget::Module { name } => build_module_doc(engine, &preds, name),
            DocTarget::Type { name } => build_type_doc(engine, &preds, name),
            DocTarget::Dependencies => build_dependency_doc(engine, &preds),
        };

        let mut result_parts = Vec::new();
        let triples_consulted = engine.all_triples().len();

        if format == DocFormat::Markdown || format == DocFormat::Both {
            let mut md = render_markdown(&sections, 1);
            if polish {
                md = polish_with_llm(engine, &md);
            }
            result_parts.push(md);
        }

        if format == DocFormat::Json || format == DocFormat::Both {
            let json = render_json(&sections, &target, triples_consulted);
            result_parts.push(json);
        }

        let result = result_parts.join("\n\n---\n\n");
        Ok(ToolOutput::ok(result))
    }
}

// ---------------------------------------------------------------------------
// Document builders
// ---------------------------------------------------------------------------

fn build_architecture_doc(engine: &Engine, preds: &CodePredicates) -> Vec<DocSection> {
    let mut sections = Vec::new();

    // Module section.
    let module_type = engine.resolve_or_create_entity("Module").ok();
    let is_a = engine.resolve_or_create_relation("is-a").ok();

    if let (Some(mod_type), Some(isa)) = (module_type, is_a) {
        let mut module_section = DocSection::new("Modules");
        let modules = find_entities_of_type(engine, isa, mod_type);

        for (mod_id, mod_label) in &modules {
            let mut sub = DocSection::new(format!("Module: {mod_label}"))
                .with_symbols(vec![mod_id.get()]);

            // Doc comment.
            let doc = get_string_object(engine, *mod_id, preds.has_doc);
            if let Some(d) = doc {
                sub.body = d;
            }

            // Defined items.
            let mut defines = Vec::new();
            for pred in [
                preds.defines_fn,
                preds.defines_struct,
                preds.defines_enum,
                preds.defines_trait,
            ] {
                for triple in engine.triples_from(*mod_id) {
                    if triple.predicate == pred {
                        defines.push(engine.resolve_label(triple.object));
                    }
                }
            }
            if !defines.is_empty() {
                defines.sort();
                let items = defines.join(", ");
                sub = sub.with_subsection(DocSection::new("Defines").with_body(items));
            }

            // Dependencies.
            let deps: Vec<String> = engine
                .triples_from(*mod_id)
                .into_iter()
                .filter(|t| t.predicate == preds.depends_on)
                .map(|t| engine.resolve_label(t.object))
                .collect();
            if !deps.is_empty() {
                let dep_list = deps.join(", ");
                sub = sub.with_subsection(DocSection::new("Depends on").with_body(dep_list));
            }

            module_section = module_section.with_subsection(sub);
        }
        sections.push(module_section);
    }

    // Type hierarchy section.
    if let Some(isa) = is_a {
        let mut type_section = DocSection::new("Type Hierarchy");
        let mut type_groups: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for triple in engine.all_triples() {
            if triple.predicate == isa {
                let obj_label = engine.resolve_label(triple.object);
                let subj_label = engine.resolve_label(triple.subject);
                type_groups
                    .entry(obj_label)
                    .or_default()
                    .push(subj_label);
            }
        }

        for (type_name, members) in &type_groups {
            let mut sorted = members.clone();
            sorted.sort();
            let body = sorted.join(", ");
            type_section = type_section
                .with_subsection(DocSection::new(format!("{type_name}")).with_body(body));
        }
        sections.push(type_section);
    }

    // Trait implementations.
    {
        let mut impl_section = DocSection::new("Trait Implementations");
        let mut impls: BTreeMap<String, Vec<String>> = BTreeMap::new();

        for triple in engine.all_triples() {
            if triple.predicate == preds.implements_trait {
                let type_label = engine.resolve_label(triple.subject);
                let trait_label = engine.resolve_label(triple.object);
                impls.entry(type_label).or_default().push(trait_label);
            }
        }

        for (type_name, traits) in &impls {
            let mut sorted = traits.clone();
            sorted.sort();
            impl_section = impl_section.with_subsection(
                DocSection::new(type_name.clone()).with_body(sorted.join(", ")),
            );
        }
        if !impls.is_empty() {
            sections.push(impl_section);
        }
    }

    sections
}

fn build_module_doc(engine: &Engine, preds: &CodePredicates, name: &str) -> Vec<DocSection> {
    let mod_id = match engine.lookup_symbol(name) {
        Ok(id) => id,
        Err(_) => {
            return vec![DocSection::new(format!("Module: {name}"))
                .with_body(format!("Module '{name}' not found in the knowledge graph."))];
        }
    };

    let mut section =
        DocSection::new(format!("Module: {name}")).with_symbols(vec![mod_id.get()]);

    // Doc comment.
    if let Some(doc) = get_string_object(engine, mod_id, preds.has_doc) {
        section.body = doc;
    }

    // Functions.
    let fns: Vec<String> = engine
        .triples_from(mod_id)
        .into_iter()
        .filter(|t| t.predicate == preds.defines_fn)
        .map(|t| {
            let label = engine.resolve_label(t.object);
            let ret = get_string_object(engine, t.object, preds.returns_type)
                .unwrap_or_default();
            if ret.is_empty() {
                label
            } else {
                format!("{label} -> {ret}")
            }
        })
        .collect();
    if !fns.is_empty() {
        section = section.with_subsection(
            DocSection::new("Functions").with_body(fns.join("\n")),
        );
    }

    // Structs.
    let structs: Vec<String> = engine
        .triples_from(mod_id)
        .into_iter()
        .filter(|t| t.predicate == preds.defines_struct)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !structs.is_empty() {
        section = section
            .with_subsection(DocSection::new("Structs").with_body(structs.join(", ")));
    }

    // Enums.
    let enums: Vec<String> = engine
        .triples_from(mod_id)
        .into_iter()
        .filter(|t| t.predicate == preds.defines_enum)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !enums.is_empty() {
        section = section
            .with_subsection(DocSection::new("Enums").with_body(enums.join(", ")));
    }

    // Traits.
    let traits: Vec<String> = engine
        .triples_from(mod_id)
        .into_iter()
        .filter(|t| t.predicate == preds.defines_trait)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !traits.is_empty() {
        section = section
            .with_subsection(DocSection::new("Traits").with_body(traits.join(", ")));
    }

    // Submodules.
    let submods: Vec<String> = engine
        .triples_from(mod_id)
        .into_iter()
        .filter(|t| t.predicate == preds.contains_mod)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !submods.is_empty() {
        section = section
            .with_subsection(DocSection::new("Submodules").with_body(submods.join(", ")));
    }

    vec![section]
}

fn build_type_doc(engine: &Engine, preds: &CodePredicates, name: &str) -> Vec<DocSection> {
    let type_id = match engine.lookup_symbol(name) {
        Ok(id) => id,
        Err(_) => {
            return vec![DocSection::new(format!("Type: {name}"))
                .with_body(format!("Type '{name}' not found in the knowledge graph."))];
        }
    };

    let mut section =
        DocSection::new(format!("Type: {name}")).with_symbols(vec![type_id.get()]);

    // Doc comment.
    if let Some(doc) = get_string_object(engine, type_id, preds.has_doc) {
        section.body = doc;
    }

    // Fields.
    let fields: Vec<String> = engine
        .triples_from(type_id)
        .into_iter()
        .filter(|t| t.predicate == preds.has_field)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !fields.is_empty() {
        section = section
            .with_subsection(DocSection::new("Fields").with_body(fields.join("\n")));
    }

    // Variants.
    let variants: Vec<String> = engine
        .triples_from(type_id)
        .into_iter()
        .filter(|t| t.predicate == preds.has_variant)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !variants.is_empty() {
        section = section
            .with_subsection(DocSection::new("Variants").with_body(variants.join(", ")));
    }

    // Methods.
    let methods: Vec<String> = engine
        .triples_from(type_id)
        .into_iter()
        .filter(|t| t.predicate == preds.has_method)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !methods.is_empty() {
        section = section
            .with_subsection(DocSection::new("Methods").with_body(methods.join(", ")));
    }

    // Trait implementations.
    let traits: Vec<String> = engine
        .triples_from(type_id)
        .into_iter()
        .filter(|t| t.predicate == preds.implements_trait)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !traits.is_empty() {
        section = section
            .with_subsection(DocSection::new("Implements").with_body(traits.join(", ")));
    }

    // Derives.
    let derives: Vec<String> = engine
        .triples_from(type_id)
        .into_iter()
        .filter(|t| t.predicate == preds.derives_trait)
        .map(|t| engine.resolve_label(t.object))
        .collect();
    if !derives.is_empty() {
        section = section
            .with_subsection(DocSection::new("Derives").with_body(derives.join(", ")));
    }

    vec![section]
}

fn build_dependency_doc(engine: &Engine, preds: &CodePredicates) -> Vec<DocSection> {
    let mut section = DocSection::new("Dependency Graph");
    let mut deps: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for triple in engine.all_triples() {
        if triple.predicate == preds.depends_on {
            let from = engine.resolve_label(triple.subject);
            let to = engine.resolve_label(triple.object);
            deps.entry(from).or_default().push(to);
        }
    }

    if deps.is_empty() {
        section.body = "No dependency edges found in the knowledge graph.".into();
    } else {
        for (source, targets) in &deps {
            let mut sorted = targets.clone();
            sorted.sort();
            section = section.with_subsection(
                DocSection::new(source.clone()).with_body(sorted.join(", ")),
            );
        }
    }

    // Circular dependencies.
    let circulars: Vec<String> = engine
        .all_triples()
        .into_iter()
        .filter(|t| t.predicate == preds.circular_dep)
        .map(|t| {
            format!(
                "{} <-> {}",
                engine.resolve_label(t.subject),
                engine.resolve_label(t.object)
            )
        })
        .collect();
    if !circulars.is_empty() {
        section = section.with_subsection(
            DocSection::new("Circular Dependencies").with_body(circulars.join("\n")),
        );
    }

    vec![section]
}

// ---------------------------------------------------------------------------
// Renderers
// ---------------------------------------------------------------------------

fn render_markdown(sections: &[DocSection], depth: usize) -> String {
    let mut out = String::new();
    for section in sections {
        let hashes = "#".repeat(depth);
        out.push_str(&format!("{hashes} {}\n\n", section.heading));
        if !section.body.is_empty() {
            out.push_str(&section.body);
            out.push_str("\n\n");
        }
        if !section.subsections.is_empty() {
            out.push_str(&render_markdown(&section.subsections, depth + 1));
        }
    }
    out
}

fn render_json(sections: &[DocSection], target: &DocTarget, triples_consulted: usize) -> String {
    let target_str = match target {
        DocTarget::Architecture => "architecture".to_string(),
        DocTarget::Module { name } => format!("module:{name}"),
        DocTarget::Type { name } => format!("type:{name}"),
        DocTarget::Dependencies => "dependencies".to_string(),
    };

    let sections_json = sections_to_json(sections);
    format!(
        r#"{{"target":"{}","sections":{},"metadata":{{"triples_consulted":{}}}}}"#,
        target_str, sections_json, triples_consulted,
    )
}

fn sections_to_json(sections: &[DocSection]) -> String {
    let items: Vec<String> = sections
        .iter()
        .map(|s| {
            let subsections = if s.subsections.is_empty() {
                "[]".to_string()
            } else {
                sections_to_json(&s.subsections)
            };
            let source_syms: Vec<String> =
                s.source_symbols.iter().map(|id| id.to_string()).collect();
            format!(
                r#"{{"heading":{},"body":{},"source_symbols":[{}],"subsections":{}}}"#,
                json_escape(&s.heading),
                json_escape(&s.body),
                source_syms.join(","),
                subsections,
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

fn json_escape(s: &str) -> String {
    let escaped = s
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{}\"", escaped)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Find all entities that have `(entity, is_a, type_sym)` in the KG.
fn find_entities_of_type(
    engine: &Engine,
    is_a: SymbolId,
    type_sym: SymbolId,
) -> Vec<(SymbolId, String)> {
    engine
        .triples_to(type_sym)
        .into_iter()
        .filter(|t| t.predicate == is_a)
        .map(|t| (t.subject, engine.resolve_label(t.subject)))
        .collect()
}

/// Get the label of the first object for `(subject, predicate, ?)`.
fn get_string_object(engine: &Engine, subject: SymbolId, predicate: SymbolId) -> Option<String> {
    engine
        .triples_from(subject)
        .into_iter()
        .find(|t| t.predicate == predicate)
        .map(|t| engine.resolve_label(t.object))
}

/// Optional LLM polishing of Markdown.
fn polish_with_llm(_engine: &Engine, markdown: &str) -> String {
    use crate::agent::llm::{OllamaClient, OllamaConfig};

    let mut client = OllamaClient::new(OllamaConfig::default());
    if !client.probe() {
        tracing::info!("Ollama not available, skipping LLM polish");
        return markdown.to_string();
    }

    let system = "You are a technical writer. Polish the following auto-generated documentation \
                  into clear prose. Do not add facts not present. Keep all technical details accurate. \
                  Return only the polished Markdown.";

    match client.generate(markdown, Some(system)) {
        Ok(polished) => polished,
        Err(e) => {
            tracing::warn!(error = %e, "LLM polish failed, using template output");
            markdown.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolInput;
    use crate::engine::EngineConfig;
    use crate::vsa::Dimension;

    fn test_engine_with_code() -> Engine {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        // Ingest a file to populate the KG.
        let tool = super::super::code_ingest::CodeIngestTool;
        let input = ToolInput::new().with_param("path", "src/symbol.rs");
        let _ = tool.execute(&engine, input);

        engine
    }

    #[test]
    fn doc_target_parse() {
        assert!(matches!(DocTarget::parse("architecture"), DocTarget::Architecture));
        assert!(matches!(DocTarget::parse("dependencies"), DocTarget::Dependencies));
        assert!(matches!(DocTarget::parse("module:vsa"), DocTarget::Module { ref name } if name == "vsa"));
        assert!(matches!(DocTarget::parse("type:Engine"), DocTarget::Type { ref name } if name == "Engine"));
    }

    #[test]
    fn doc_format_parse() {
        assert_eq!(DocFormat::parse("markdown"), DocFormat::Markdown);
        assert_eq!(DocFormat::parse("json"), DocFormat::Json);
        assert_eq!(DocFormat::parse("both"), DocFormat::Both);
    }

    #[test]
    fn architecture_doc_produces_output() {
        let engine = test_engine_with_code();
        let tool = DocGenTool;
        let input = ToolInput::new()
            .with_param("target", "architecture")
            .with_param("format", "markdown");
        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success);
        assert!(!output.result.is_empty());
    }

    #[test]
    fn type_doc_not_found() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let tool = DocGenTool;
        let input = ToolInput::new()
            .with_param("target", "type:NonExistent")
            .with_param("format", "markdown");
        let output = tool.execute(&engine, input).unwrap();
        assert!(output.result.contains("not found"));
    }

    #[test]
    fn json_output_is_valid() {
        let engine = test_engine_with_code();
        let tool = DocGenTool;
        let input = ToolInput::new()
            .with_param("target", "architecture")
            .with_param("format", "json");
        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success);
        // Should be parseable as JSON.
        let parsed: serde_json::Value = serde_json::from_str(&output.result).unwrap();
        assert!(parsed.get("target").is_some());
        assert!(parsed.get("sections").is_some());
    }

    #[test]
    fn dependency_doc_produces_output() {
        let engine = test_engine_with_code();
        let tool = DocGenTool;
        let input = ToolInput::new()
            .with_param("target", "dependencies")
            .with_param("format", "markdown");
        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success);
    }

    #[test]
    fn render_markdown_basic() {
        let sections = vec![DocSection::new("Title")
            .with_body("Some text.")
            .with_subsection(DocSection::new("Sub").with_body("Sub text."))];
        let md = render_markdown(&sections, 1);
        assert!(md.contains("# Title"));
        assert!(md.contains("## Sub"));
        assert!(md.contains("Some text."));
    }
}
