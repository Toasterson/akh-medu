//! Code generation tool: query KG for code structure, build AbsTree, linearize
//! through RustCodeGrammar, optionally format with `rustfmt`.
//!
//! Pipeline: KG query → AbsTree construction → linearization → (optional) format → output
//!
//! Supports targets: function, struct, enum, trait, module, impl, file.

use std::collections::HashSet;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::grammar::abs::AbsTree;
use crate::grammar::concrete::{ConcreteGrammar, LinContext};
use crate::grammar::rust_gen::RustCodeGrammar;
use crate::grammar::templates::TemplateRegistry;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::code_predicates::CodePredicates;

// ---------------------------------------------------------------------------
// Scope enum
// ---------------------------------------------------------------------------

/// What scope of code to generate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeScope {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
    File,
}

impl CodeScope {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "fn" | "function" => Some(Self::Function),
            "struct" => Some(Self::Struct),
            "enum" => Some(Self::Enum),
            "trait" => Some(Self::Trait),
            "impl" => Some(Self::Impl),
            "module" | "mod" => Some(Self::Module),
            "file" => Some(Self::File),
            _ => None,
        }
    }

    fn kind_str(self) -> &'static str {
        match self {
            Self::Function => "fn",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Module | Self::File => "module",
        }
    }
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Generate Rust code from KG structure through the grammar system.
pub struct CodeGenTool;

impl Tool for CodeGenTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "code_gen".into(),
            description: "Generate Rust code from knowledge graph structure or templates. \
                          KG path: queries code:* triples, builds AbsTree, linearizes through \
                          RustCodeGrammar. Template path: instantiates a named template with \
                          params (supports attributes, derives, full Rust patterns)."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "target".into(),
                    description: "Symbol name or ID of the entity to generate code for.".into(),
                    required: true,
                },
                ToolParam {
                    name: "scope".into(),
                    description: "Code scope: 'function', 'struct', 'enum', 'trait', \
                                  'impl', 'module', or 'file' (default: auto-detect)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "format".into(),
                    description: "Whether to run rustfmt on output: 'true' or 'false' (default: false)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "template".into(),
                    description: "Template name to use instead of KG→AbsTree pipeline. \
                                  Available: error-type, trait-impl, builder, from-impl, \
                                  test-module, iterator, new-constructor."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "template_params".into(),
                    description: "Semicolon-separated key=value pairs for template \
                                  instantiation. E.g., 'name=MyError;variants=NotFound(not found, \
                                  my::err, check ID);result_alias=MyResult'."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let target_str = input.require("target", "code_gen")?;
        let run_format = input
            .get("format")
            .map(|s| s.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        // --- Template path ---
        if let Some(template_name) = input.get("template") {
            return execute_template(engine, target_str, template_name, &input, run_format);
        }

        // --- KG → AbsTree → RustCodeGrammar path ---
        let scope = input.get("scope").and_then(CodeScope::parse);
        let preds = CodePredicates::init(engine)?;

        // Resolve target symbol
        let target_id = match engine.lookup_symbol(target_str) {
            Ok(id) => id,
            Err(_) => {
                return Ok(ToolOutput::err(format!(
                    "Target '{}' not found in the knowledge graph. \
                     Use code_ingest or kg_mutate to define it first.",
                    target_str
                )));
            }
        };

        let target_label = engine.resolve_label(target_id);

        // Auto-detect scope from KG if not specified
        let effective_scope = scope.unwrap_or_else(|| detect_scope(engine, &preds, target_id));

        // Build AbsTree from KG facts
        let abs_tree = build_abstree_from_kg(engine, &preds, target_id, &target_label, effective_scope);

        // Linearize through RustCodeGrammar
        let grammar = RustCodeGrammar;
        let ctx = LinContext::default();
        let code = grammar.linearize(&abs_tree, &ctx).map_err(|e| {
            AgentError::ToolExecution {
                tool_name: "code_gen".into(),
                message: format!("Linearization failed: {e}"),
            }
        })?;

        // Optionally format with rustfmt
        let final_code = if run_format {
            format_with_rustfmt(&code).unwrap_or(code)
        } else {
            code
        };

        // Record provenance
        let source_symbols = collect_source_symbols(engine, &preds, target_id);
        let mut prov_record = ProvenanceRecord {
            id: None,
            derived_id: target_id,
            sources: source_symbols.clone(),
            kind: DerivationKind::CodeGenerated {
                scope: effective_scope.kind_str().to_string(),
                source_count: source_symbols.len(),
            },
            confidence: 0.8,
            depth: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        if let Err(e) = engine.store_provenance(&mut prov_record) {
            // Non-fatal: log but continue
            eprintln!("Warning: failed to record code_gen provenance: {e}");
        }

        let mut all_symbols = source_symbols;
        all_symbols.push(target_id);

        Ok(ToolOutput::ok_with_symbols(
            format!(
                "Generated {} code for '{}':\n\n{}",
                effective_scope.kind_str(),
                target_label,
                final_code
            ),
            all_symbols,
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "code_gen".into(),
            description: "Generates Rust code from KG structure or templates — read-only KG access, \
                          optional filesystem for rustfmt."
                .into(),
            parameters: vec![
                ToolParamSchema::required(
                    "target",
                    "Symbol name or ID of the entity to generate code for.",
                ),
                ToolParamSchema::optional(
                    "scope",
                    "Code scope: 'function', 'struct', 'enum', 'trait', 'impl', 'module', or 'file'.",
                ),
                ToolParamSchema::optional(
                    "format",
                    "Whether to run rustfmt on output: 'true' or 'false'.",
                ),
                ToolParamSchema::optional(
                    "template",
                    "Template name: error-type, trait-impl, builder, from-impl, test-module, iterator, new-constructor.",
                ),
                ToolParamSchema::optional(
                    "template_params",
                    "Semicolon-separated key=value pairs for template instantiation.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: HashSet::from([Capability::ReadKg]),
                description: "Generates code from KG or templates — read-only.".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// Template path
// ---------------------------------------------------------------------------

/// Execute the template code generation path.
fn execute_template(
    engine: &Engine,
    target_str: &str,
    template_name: &str,
    input: &ToolInput,
    run_format: bool,
) -> AgentResult<ToolOutput> {
    let registry = TemplateRegistry::new();

    let template = match registry.get(template_name) {
        Some(t) => t,
        None => {
            let available: Vec<&str> = registry.list();
            return Ok(ToolOutput::err(format!(
                "Unknown template '{}'. Available templates: {}",
                template_name,
                available.join(", ")
            )));
        }
    };

    // Parse template_params from semicolon-separated key=value pairs
    let params = parse_template_params(input.get("template_params").unwrap_or(""));

    let code = match template.instantiate(&params) {
        Ok(c) => c,
        Err(e) => {
            // Show required params for guidance
            let required: Vec<String> = template
                .params
                .iter()
                .filter(|p| p.required)
                .map(|p| format!("{}  ({})", p.name, p.description))
                .collect();
            return Ok(ToolOutput::err(format!(
                "Template instantiation failed: {e}\n\nRequired parameters:\n  {}",
                required.join("\n  ")
            )));
        }
    };

    let final_code = if run_format {
        format_with_rustfmt(&code).unwrap_or(code)
    } else {
        code
    };

    // Record provenance if target exists in KG
    if let Ok(target_id) = engine.lookup_symbol(target_str) {
        let mut prov_record = ProvenanceRecord {
            id: None,
            derived_id: target_id,
            sources: vec![],
            kind: DerivationKind::CodeGenerated {
                scope: format!("template:{template_name}"),
                source_count: 0,
            },
            confidence: 0.9,
            depth: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        if let Err(e) = engine.store_provenance(&mut prov_record) {
            eprintln!("Warning: failed to record template provenance: {e}");
        }
    }

    Ok(ToolOutput::ok(format!(
        "Generated code from template '{}' for '{}':\n\n{}",
        template_name, target_str, final_code
    )))
}

/// Parse semicolon-separated `key=value` pairs into a HashMap.
fn parse_template_params(s: &str) -> std::collections::HashMap<String, String> {
    let mut map = std::collections::HashMap::new();
    for pair in s.split(';') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        if let Some((key, value)) = pair.split_once('=') {
            map.insert(key.trim().to_string(), value.trim().to_string());
        }
    }
    map
}

// ---------------------------------------------------------------------------
// KG → AbsTree construction
// ---------------------------------------------------------------------------

/// Detect the scope of a target symbol from its KG relationships.
fn detect_scope(engine: &Engine, preds: &CodePredicates, target: SymbolId) -> CodeScope {
    let triples_to = engine.triples_to(target);

    for t in &triples_to {
        if t.predicate == preds.defines_fn {
            return CodeScope::Function;
        }
        if t.predicate == preds.defines_struct {
            return CodeScope::Struct;
        }
        if t.predicate == preds.defines_enum {
            return CodeScope::Enum;
        }
        if t.predicate == preds.defines_trait {
            return CodeScope::Trait;
        }
    }

    // Check if it has methods (likely a type with impl)
    let triples_from = engine.triples_from(target);
    let has_methods = triples_from.iter().any(|t| t.predicate == preds.has_method);
    let has_fields = triples_from.iter().any(|t| t.predicate == preds.has_field);
    let has_variants = triples_from.iter().any(|t| t.predicate == preds.has_variant);
    let has_submods = triples_from.iter().any(|t| t.predicate == preds.contains_mod);

    if has_variants {
        CodeScope::Enum
    } else if has_fields {
        CodeScope::Struct
    } else if has_submods {
        CodeScope::Module
    } else if has_methods {
        CodeScope::Impl
    } else {
        CodeScope::Function
    }
}

/// Build an AbsTree from KG code facts about the target.
fn build_abstree_from_kg(
    engine: &Engine,
    preds: &CodePredicates,
    target: SymbolId,
    label: &str,
    scope: CodeScope,
) -> AbsTree {
    let triples_from = engine.triples_from(target);

    // Gather doc comment
    let doc = triples_from
        .iter()
        .find(|t| t.predicate == preds.has_doc)
        .map(|t| engine.resolve_label(t.object));

    match scope {
        CodeScope::Function => {
            let params = gather_params(engine, preds, target);
            let return_type = triples_from
                .iter()
                .find(|t| t.predicate == preds.returns_type)
                .map(|t| engine.resolve_label(t.object));

            AbsTree::CodeSignature {
                kind: "fn".to_string(),
                name: label.to_string(),
                doc_summary: doc,
                params_or_fields: params,
                return_type,
                traits: vec![],
                importance: None,
            }
        }

        CodeScope::Struct => {
            let fields = gather_fields(engine, preds, target);
            let derives = gather_derives(engine, preds, target);

            AbsTree::CodeSignature {
                kind: "struct".to_string(),
                name: label.to_string(),
                doc_summary: doc,
                params_or_fields: fields,
                return_type: None,
                traits: derives,
                importance: None,
            }
        }

        CodeScope::Enum => {
            let variants = gather_variants(engine, preds, target);
            let derives = gather_derives(engine, preds, target);

            AbsTree::CodeSignature {
                kind: "enum".to_string(),
                name: label.to_string(),
                doc_summary: doc,
                params_or_fields: variants,
                return_type: None,
                traits: derives,
                importance: None,
            }
        }

        CodeScope::Trait => {
            let methods = gather_method_sigs(engine, preds, target);

            AbsTree::CodeSignature {
                kind: "trait".to_string(),
                name: label.to_string(),
                doc_summary: doc,
                params_or_fields: methods,
                return_type: None,
                traits: vec![],
                importance: None,
            }
        }

        CodeScope::Impl => {
            let methods = gather_method_names(engine, preds, target);
            let trait_impl = triples_from
                .iter()
                .find(|t| t.predicate == preds.implements_trait)
                .map(|t| engine.resolve_label(t.object));

            AbsTree::CodeSignature {
                kind: "impl".to_string(),
                name: label.to_string(),
                doc_summary: doc,
                params_or_fields: methods,
                return_type: None,
                traits: trait_impl.into_iter().collect(),
                importance: None,
            }
        }

        CodeScope::Module | CodeScope::File => {
            let children = gather_module_children(engine, preds, target);

            AbsTree::CodeModule {
                name: label.to_string(),
                role: None,
                importance: None,
                doc_summary: doc,
                children,
            }
        }
    }
}

/// Gather function parameters from `code:has-param` triples.
fn gather_params(engine: &Engine, preds: &CodePredicates, target: SymbolId) -> Vec<String> {
    engine
        .triples_from(target)
        .into_iter()
        .filter(|t| t.predicate == preds.has_param)
        .map(|t| engine.resolve_label(t.object))
        .collect()
}

/// Gather struct fields from `code:has-field` triples.
fn gather_fields(engine: &Engine, preds: &CodePredicates, target: SymbolId) -> Vec<String> {
    engine
        .triples_from(target)
        .into_iter()
        .filter(|t| t.predicate == preds.has_field)
        .map(|t| engine.resolve_label(t.object))
        .collect()
}

/// Gather enum variants from `code:has-variant` triples.
fn gather_variants(engine: &Engine, preds: &CodePredicates, target: SymbolId) -> Vec<String> {
    engine
        .triples_from(target)
        .into_iter()
        .filter(|t| t.predicate == preds.has_variant)
        .map(|t| engine.resolve_label(t.object))
        .collect()
}

/// Gather derive traits from `code:derives-trait` triples.
fn gather_derives(engine: &Engine, preds: &CodePredicates, target: SymbolId) -> Vec<String> {
    engine
        .triples_from(target)
        .into_iter()
        .filter(|t| t.predicate == preds.derives_trait)
        .map(|t| engine.resolve_label(t.object))
        .collect()
}

/// Gather method names from `code:has-method` triples.
fn gather_method_names(engine: &Engine, preds: &CodePredicates, target: SymbolId) -> Vec<String> {
    engine
        .triples_from(target)
        .into_iter()
        .filter(|t| t.predicate == preds.has_method)
        .map(|t| engine.resolve_label(t.object))
        .collect()
}

/// Gather method signatures for traits from `code:has-method` triples.
fn gather_method_sigs(engine: &Engine, preds: &CodePredicates, target: SymbolId) -> Vec<String> {
    engine
        .triples_from(target)
        .into_iter()
        .filter(|t| t.predicate == preds.has_method)
        .map(|t| {
            let method_name = engine.resolve_label(t.object);
            // Check if method has a return type
            let ret = engine
                .triples_from(t.object)
                .into_iter()
                .find(|mt| mt.predicate == preds.returns_type)
                .map(|mt| engine.resolve_label(mt.object));

            match ret {
                Some(r) => format!("fn {method_name}(&self) -> {r}"),
                None => format!("fn {method_name}(&self)"),
            }
        })
        .collect()
}

/// Gather module children: functions, structs, enums, traits defined in this module.
fn gather_module_children(
    engine: &Engine,
    preds: &CodePredicates,
    target: SymbolId,
) -> Vec<AbsTree> {
    let triples = engine.triples_from(target);
    let mut children = Vec::new();

    for t in &triples {
        let child_label = engine.resolve_label(t.object);

        if t.predicate == preds.defines_fn {
            children.push(build_abstree_from_kg(
                engine,
                preds,
                t.object,
                &child_label,
                CodeScope::Function,
            ));
        } else if t.predicate == preds.defines_struct {
            children.push(build_abstree_from_kg(
                engine,
                preds,
                t.object,
                &child_label,
                CodeScope::Struct,
            ));
        } else if t.predicate == preds.defines_enum {
            children.push(build_abstree_from_kg(
                engine,
                preds,
                t.object,
                &child_label,
                CodeScope::Enum,
            ));
        } else if t.predicate == preds.defines_trait {
            children.push(build_abstree_from_kg(
                engine,
                preds,
                t.object,
                &child_label,
                CodeScope::Trait,
            ));
        } else if t.predicate == preds.contains_mod {
            children.push(build_abstree_from_kg(
                engine,
                preds,
                t.object,
                &child_label,
                CodeScope::Module,
            ));
        }
    }

    children
}

/// Collect all source symbols referenced by code:* triples from the target.
fn collect_source_symbols(engine: &Engine, preds: &CodePredicates, target: SymbolId) -> Vec<SymbolId> {
    let mut symbols = HashSet::new();
    let triples = engine.triples_from(target);

    let code_preds = [
        preds.has_param,
        preds.has_field,
        preds.has_variant,
        preds.has_method,
        preds.returns_type,
        preds.derives_trait,
        preds.implements_trait,
        preds.depends_on,
        preds.has_doc,
        preds.defines_fn,
        preds.defines_struct,
        preds.defines_enum,
        preds.defines_trait,
        preds.contains_mod,
    ];

    for t in &triples {
        if code_preds.contains(&t.predicate) {
            symbols.insert(t.object);
        }
    }

    symbols.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

/// Try to format Rust code using `rustfmt`.
fn format_with_rustfmt(code: &str) -> Option<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new("rustfmt")
        .arg("--edition")
        .arg("2024")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    child
        .stdin
        .as_mut()?
        .write_all(code.as_bytes())
        .ok()?;

    let output = child.wait_with_output().ok()?;
    if output.status.success() {
        String::from_utf8(output.stdout).ok()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Refinement (for Phase 10d iterative loop)
// ---------------------------------------------------------------------------

/// Result of analyzing compiler errors for refinement.
#[derive(Debug, Clone)]
pub struct RefinementAction {
    /// What kind of fix to apply.
    pub kind: RefinementKind,
    /// Human-readable description.
    pub description: String,
    /// Symbols to modify in the KG.
    pub affected_symbols: Vec<SymbolId>,
}

/// Categories of compiler errors that can be addressed via KG updates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefinementKind {
    /// Adjust return type in KG.
    TypeMismatch { expected: String, found: String },
    /// Add a `code:depends-on` triple for a missing import.
    MissingImport { module: String },
    /// Syntax error in generated code (likely grammar bug).
    SyntaxError,
    /// Missing trait implementation.
    MissingTraitImpl { trait_name: String, type_name: String },
    /// Generic error that needs manual intervention.
    Other,
}

/// Analyze compiler error output and suggest refinement actions.
pub fn analyze_compiler_errors(
    error_output: &str,
    target: SymbolId,
) -> Vec<RefinementAction> {
    let mut actions = Vec::new();

    for line in error_output.lines() {
        let trimmed = line.trim();

        // Type mismatch: "expected `X`, found `Y`"
        if let Some(rest) = trimmed.strip_prefix("expected `") {
            if let Some((expected, rest2)) = rest.split_once("`, found `") {
                if let Some(found) = rest2.strip_suffix('`') {
                    actions.push(RefinementAction {
                        kind: RefinementKind::TypeMismatch {
                            expected: expected.to_string(),
                            found: found.to_string(),
                        },
                        description: format!(
                            "Type mismatch: expected `{expected}`, found `{found}`"
                        ),
                        affected_symbols: vec![target],
                    });
                }
            }
        }

        // Missing import: "unresolved import `X`" or "cannot find X in this scope"
        if trimmed.contains("unresolved import") {
            if let Some(rest) = trimmed.split("unresolved import `").nth(1) {
                if let Some(module) = rest.strip_suffix('`') {
                    actions.push(RefinementAction {
                        kind: RefinementKind::MissingImport {
                            module: module.to_string(),
                        },
                        description: format!("Missing import: `{module}`"),
                        affected_symbols: vec![target],
                    });
                }
            }
        }

        // Missing trait impl: "the trait `X` is not implemented for `Y`"
        if let Some(rest) = trimmed.strip_prefix("the trait `") {
            if let Some((trait_name, rest2)) = rest.split_once("` is not implemented for `") {
                if let Some(type_name) = rest2.strip_suffix('`') {
                    actions.push(RefinementAction {
                        kind: RefinementKind::MissingTraitImpl {
                            trait_name: trait_name.to_string(),
                            type_name: type_name.to_string(),
                        },
                        description: format!(
                            "Missing impl: `{trait_name}` for `{type_name}`"
                        ),
                        affected_symbols: vec![target],
                    });
                }
            }
        }

        // Syntax errors
        if trimmed.contains("expected") && trimmed.contains("found") && trimmed.contains("token") {
            actions.push(RefinementAction {
                kind: RefinementKind::SyntaxError,
                description: format!("Syntax error: {trimmed}"),
                affected_symbols: vec![target],
            });
        }
    }

    actions
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

    fn test_engine_with_fn() -> (Engine, SymbolId) {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let preds = CodePredicates::init(&engine).unwrap();

        // Create a function entity with params and return type
        let fn_id = engine.resolve_or_create_entity("greet").unwrap();
        let mod_id = engine.resolve_or_create_entity("utils").unwrap();
        let param_id = engine.resolve_or_create_entity("name: &str").unwrap();
        let ret_id = engine.resolve_or_create_entity("String").unwrap();
        let doc_id = engine.resolve_or_create_entity("Greets a person by name.").unwrap();

        use crate::graph::Triple;
        engine.add_triple(&Triple::new(mod_id, preds.defines_fn, fn_id).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(fn_id, preds.has_param, param_id).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(fn_id, preds.returns_type, ret_id).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(fn_id, preds.has_doc, doc_id).with_confidence(0.9)).unwrap();

        (engine, fn_id)
    }

    fn test_engine_with_struct() -> (Engine, SymbolId) {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let preds = CodePredicates::init(&engine).unwrap();

        let struct_id = engine.resolve_or_create_entity("Config").unwrap();
        let mod_id = engine.resolve_or_create_entity("settings").unwrap();
        let field1 = engine.resolve_or_create_entity("name: String").unwrap();
        let field2 = engine.resolve_or_create_entity("value: u64").unwrap();
        let derive1 = engine.resolve_or_create_entity("Debug").unwrap();
        let derive2 = engine.resolve_or_create_entity("Clone").unwrap();

        use crate::graph::Triple;
        engine.add_triple(&Triple::new(mod_id, preds.defines_struct, struct_id).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(struct_id, preds.has_field, field1).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(struct_id, preds.has_field, field2).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(struct_id, preds.derives_trait, derive1).with_confidence(0.9)).unwrap();
        engine.add_triple(&Triple::new(struct_id, preds.derives_trait, derive2).with_confidence(0.9)).unwrap();

        (engine, struct_id)
    }

    #[test]
    fn code_gen_function() {
        let (engine, _fn_id) = test_engine_with_fn();
        let tool = CodeGenTool;

        let input = ToolInput::new()
            .with_param("target", "greet")
            .with_param("scope", "function");

        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success, "code_gen failed: {}", output.result);
        assert!(output.result.contains("pub fn greet"));
        assert!(output.result.contains("name: &str"));
        assert!(output.result.contains("String"));
    }

    #[test]
    fn code_gen_struct() {
        let (engine, _struct_id) = test_engine_with_struct();
        let tool = CodeGenTool;

        let input = ToolInput::new()
            .with_param("target", "Config")
            .with_param("scope", "struct");

        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success, "code_gen failed: {}", output.result);
        assert!(output.result.contains("pub struct Config"));
        assert!(output.result.contains("Debug"));
        assert!(output.result.contains("Clone"));
    }

    #[test]
    fn code_gen_auto_detect_fn() {
        let (engine, _) = test_engine_with_fn();
        let tool = CodeGenTool;

        let input = ToolInput::new().with_param("target", "greet");

        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success);
        assert!(output.result.contains("pub fn greet"));
    }

    #[test]
    fn code_gen_auto_detect_struct() {
        let (engine, _) = test_engine_with_struct();
        let tool = CodeGenTool;

        let input = ToolInput::new().with_param("target", "Config");

        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success);
        assert!(output.result.contains("pub struct Config"));
    }

    #[test]
    fn code_gen_not_found() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let tool = CodeGenTool;
        let input = ToolInput::new().with_param("target", "NonExistent");

        let output = tool.execute(&engine, input).unwrap();
        assert!(!output.success);
        assert!(output.result.contains("not found"));
    }

    #[test]
    fn code_gen_module() {
        let (engine, _) = test_engine_with_fn();
        let tool = CodeGenTool;

        let input = ToolInput::new()
            .with_param("target", "utils")
            .with_param("scope", "module");

        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success);
        assert!(output.result.contains("pub mod utils"));
        assert!(output.result.contains("pub fn greet"));
    }

    #[test]
    fn scope_parse() {
        assert_eq!(CodeScope::parse("fn"), Some(CodeScope::Function));
        assert_eq!(CodeScope::parse("function"), Some(CodeScope::Function));
        assert_eq!(CodeScope::parse("struct"), Some(CodeScope::Struct));
        assert_eq!(CodeScope::parse("enum"), Some(CodeScope::Enum));
        assert_eq!(CodeScope::parse("trait"), Some(CodeScope::Trait));
        assert_eq!(CodeScope::parse("impl"), Some(CodeScope::Impl));
        assert_eq!(CodeScope::parse("module"), Some(CodeScope::Module));
        assert_eq!(CodeScope::parse("file"), Some(CodeScope::File));
        assert_eq!(CodeScope::parse("unknown"), None);
    }

    #[test]
    fn analyze_type_mismatch() {
        let target = SymbolId::new(1).unwrap();
        let errors = "expected `String`, found `&str`";
        let actions = analyze_compiler_errors(errors, target);
        assert!(!actions.is_empty());
        assert!(matches!(
            actions[0].kind,
            RefinementKind::TypeMismatch { .. }
        ));
    }

    #[test]
    fn analyze_missing_import() {
        let target = SymbolId::new(1).unwrap();
        let errors = "error: unresolved import `std::collections::BTreeSet`";
        let actions = analyze_compiler_errors(errors, target);
        assert!(!actions.is_empty());
        assert!(matches!(
            actions[0].kind,
            RefinementKind::MissingImport { .. }
        ));
    }

    #[test]
    fn analyze_missing_trait_impl() {
        let target = SymbolId::new(1).unwrap();
        let errors = "the trait `Display` is not implemented for `Config`";
        let actions = analyze_compiler_errors(errors, target);
        assert!(!actions.is_empty());
        assert!(matches!(
            actions[0].kind,
            RefinementKind::MissingTraitImpl { .. }
        ));
    }

    // --- Template path tests ---

    #[test]
    fn code_gen_template_error_type() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let tool = CodeGenTool;
        let input = ToolInput::new()
            .with_param("target", "MyError")
            .with_param("template", "error-type")
            .with_param(
                "template_params",
                "name=MyError;variants=NotFound(item not found, my::not_found, check the ID), InvalidInput(invalid input);result_alias=MyResult",
            );

        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success, "template failed: {}", output.result);
        assert!(output.result.contains("#[derive(Debug, thiserror::Error, miette::Diagnostic)]"));
        assert!(output.result.contains("pub enum MyError {"));
        assert!(output.result.contains("#[error(\"item not found\")]"));
        assert!(output.result.contains("#[diagnostic(code(my::not_found), help(\"check the ID\"))]"));
        assert!(output.result.contains("pub type MyResult<T>"));
    }

    #[test]
    fn code_gen_template_builder() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let tool = CodeGenTool;
        let input = ToolInput::new()
            .with_param("target", "Config")
            .with_param("template", "builder")
            .with_param("template_params", "type_name=Config;fields=name: String, value: u64");

        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success, "template failed: {}", output.result);
        assert!(output.result.contains("pub struct ConfigBuilder {"));
        assert!(output.result.contains("pub fn build(self) -> Config {"));
    }

    #[test]
    fn code_gen_template_unknown() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let tool = CodeGenTool;
        let input = ToolInput::new()
            .with_param("target", "Foo")
            .with_param("template", "nonexistent");

        let output = tool.execute(&engine, input).unwrap();
        assert!(!output.success);
        assert!(output.result.contains("Unknown template"));
        assert!(output.result.contains("error-type"));
    }

    #[test]
    fn code_gen_template_missing_params() {
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap();

        let tool = CodeGenTool;
        let input = ToolInput::new()
            .with_param("target", "Foo")
            .with_param("template", "error-type")
            .with_param("template_params", "");

        let output = tool.execute(&engine, input).unwrap();
        assert!(!output.success);
        assert!(output.result.contains("requires parameter"));
    }

    #[test]
    fn parse_template_params_semicolons() {
        let params = parse_template_params("name=MyError;variants=A(x), B(y);result_alias=MyResult");
        assert_eq!(params.get("name").unwrap(), "MyError");
        assert_eq!(params.get("variants").unwrap(), "A(x), B(y)");
        assert_eq!(params.get("result_alias").unwrap(), "MyResult");
    }
}
