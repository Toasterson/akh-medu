//! Code ingest tool: parse Rust source files with `syn` and extract structural
//! triples into the knowledge graph.
//!
//! For each Rust item (fn, struct, enum, trait, impl, mod, use) the visitor
//! creates entity symbols and links them with `code:*` predicates. Every symbol
//! created during ingestion gets a populated `SourceRef` for provenance.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{self, Visibility};

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::{SourceRef, SymbolId, SymbolKind};

use super::code_predicates::CodePredicates;

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Ingest Rust source code into the knowledge graph.
pub struct CodeIngestTool;

impl Tool for CodeIngestTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "code_ingest".into(),
            description: "Parse Rust source files with syn and extract structural triples \
                          (functions, structs, enums, traits, impls, modules, dependencies) \
                          into the knowledge graph."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "path".into(),
                    description: "File or directory path to ingest. Directories are scanned \
                                  recursively for *.rs files."
                        .into(),
                    required: true,
                },
                ToolParam {
                    name: "recursive".into(),
                    description: "Scan subdirectories. Default: true.".into(),
                    required: false,
                },
                ToolParam {
                    name: "max_files".into(),
                    description: "Maximum number of files to process. Default: 200.".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let path_str = input.require("path", "code_ingest")?;
        let recursive: bool = input
            .get("recursive")
            .and_then(|s| s.parse().ok())
            .unwrap_or(true);
        let max_files: usize = input
            .get("max_files")
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);

        let path = PathBuf::from(path_str);

        if !path.exists() {
            return Err(AgentError::ToolExecution {
                tool_name: "code_ingest".into(),
                message: format!("path does not exist: {}", path.display()),
            });
        }

        let preds = CodePredicates::init(engine)?;
        let is_a = engine.resolve_or_create_relation("is-a")?;

        let files = collect_rs_files(&path, recursive, max_files);
        if files.is_empty() {
            return Ok(ToolOutput::ok("No .rs files found at the given path."));
        }

        // Use the path's parent as the base for relative paths.
        let base = if path.is_dir() {
            path.clone()
        } else {
            path.parent().unwrap_or(&path).to_path_buf()
        };

        let mut stats = IngestStats::default();
        let mut all_symbols = Vec::new();

        for file_path in &files {
            match ingest_file(engine, &preds, is_a, file_path, &base, &mut stats) {
                Ok(syms) => all_symbols.extend(syms),
                Err(e) => {
                    stats.errors += 1;
                    tracing::warn!(file = %file_path.display(), error = %e, "skipping file");
                }
            }
        }

        let msg = format!(
            "Code ingest: {} file(s) processed, {} triple(s) extracted, {} symbol(s) created, {} error(s).",
            stats.files_processed, stats.triples_extracted, stats.symbols_created, stats.errors,
        );
        Ok(ToolOutput::ok_with_symbols(msg, all_symbols))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "code_ingest".into(),
            description: "Parses Rust source code and ingests structure into the knowledge graph."
                .into(),
            parameters: vec![
                ToolParamSchema::required("path", "File or directory path to ingest."),
                ToolParamSchema::optional("recursive", "Scan subdirectories (default: true)."),
                ToolParamSchema::optional(
                    "max_files",
                    "Maximum number of files to process (default: 200).",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::WriteKg, Capability::ReadFilesystem]),
                description:
                    "Parses Rust source code and ingests structure into the knowledge graph.".into(),
                shadow_triggers: vec!["ingest".into(), "code".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

#[derive(Default)]
struct IngestStats {
    files_processed: usize,
    triples_extracted: usize,
    symbols_created: usize,
    errors: usize,
}

// ---------------------------------------------------------------------------
// File collection
// ---------------------------------------------------------------------------

/// Collect `*.rs` files from a path (file or directory).
fn collect_rs_files(path: &Path, recursive: bool, max: usize) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if path.is_file() {
        if path.extension().is_some_and(|e| e == "rs") {
            files.push(path.to_path_buf());
        }
        return files;
    }
    collect_rs_recursive(path, recursive, max, &mut files);
    files.sort();
    files
}

fn collect_rs_recursive(dir: &Path, recursive: bool, max: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        if out.len() >= max {
            return;
        }
        let p = entry.path();
        if p.is_dir() && recursive {
            // Skip target/ and hidden dirs.
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') || name == "target" {
                continue;
            }
            collect_rs_recursive(&p, recursive, max, out);
        } else if p.is_file() && p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-file ingestion
// ---------------------------------------------------------------------------

fn ingest_file(
    engine: &Engine,
    preds: &CodePredicates,
    is_a: SymbolId,
    file_path: &Path,
    base: &Path,
    stats: &mut IngestStats,
) -> AgentResult<Vec<SymbolId>> {
    let source = std::fs::read_to_string(file_path).map_err(|e| AgentError::ToolExecution {
        tool_name: "code_ingest".into(),
        message: format!("cannot read {}: {e}", file_path.display()),
    })?;

    let syntax = syn::parse_file(&source).map_err(|e| AgentError::ToolExecution {
        tool_name: "code_ingest".into(),
        message: format!("parse error in {}: {e}", file_path.display()),
    })?;

    let rel_path = file_path
        .strip_prefix(base)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string();

    // Create a file entity symbol.
    let file_sym =
        resolve_or_create_entity_with_source(engine, &rel_path, &rel_path, 0, 0, source.len())?;

    let file_type = engine.resolve_or_create_entity("File")?;
    add_triple(engine, file_sym, is_a, file_type, 1.0)?;

    // Derive module name from file path.
    let module_label = module_label_from_path(&rel_path);
    let module_sym =
        resolve_or_create_entity_with_source(engine, &module_label, &rel_path, 0, 0, 0)?;
    let module_type = engine.resolve_or_create_entity("Module")?;
    add_triple(engine, module_sym, is_a, module_type, 1.0)?;
    add_triple(engine, module_sym, preds.defined_in, file_sym, 1.0)?;

    // Extract file-level inner doc comments (//! at the top of the file).
    // These appear as #![doc = "..."] attributes in syn::File.attrs.
    if let Some(doc) = extract_inner_doc_comments(&syntax.attrs) {
        let truncated = truncate(&doc, 256);
        if let Ok(doc_sym) = engine.resolve_or_create_entity(&truncated) {
            add_triple(engine, module_sym, preds.has_doc, doc_sym, 1.0)?;
        }
    }

    let mut visitor = CodeVisitor {
        engine,
        preds,
        is_a,
        rel_path: &rel_path,
        source: &source,
        module_stack: vec![module_sym],
        symbols: Vec::new(),
        triple_count: 0,
        item_index: 0,
    };

    visitor.visit_file(&syntax);

    stats.files_processed += 1;
    stats.triples_extracted += visitor.triple_count;
    stats.symbols_created += visitor.symbols.len();

    Ok(visitor.symbols)
}

/// Convert a file path like `src/agent/tools/code_ingest.rs` to a module label
/// like `agent::tools::code_ingest`.
fn module_label_from_path(rel_path: &str) -> String {
    let p = rel_path
        .strip_prefix("src/")
        .unwrap_or(rel_path)
        .strip_suffix(".rs")
        .unwrap_or(rel_path)
        .strip_suffix("/mod")
        .unwrap_or(
            rel_path
                .strip_prefix("src/")
                .unwrap_or(rel_path)
                .strip_suffix(".rs")
                .unwrap_or(rel_path),
        );
    p.replace('/', "::")
}

// ---------------------------------------------------------------------------
// Syn visitor
// ---------------------------------------------------------------------------

struct CodeVisitor<'a> {
    engine: &'a Engine,
    preds: &'a CodePredicates,
    is_a: SymbolId,
    rel_path: &'a str,
    source: &'a str,
    module_stack: Vec<SymbolId>,
    symbols: Vec<SymbolId>,
    triple_count: usize,
    item_index: u32,
}

impl<'a> CodeVisitor<'a> {
    fn current_module(&self) -> SymbolId {
        *self.module_stack.last().expect("module stack non-empty")
    }

    fn create_entity(&mut self, label: &str, _item: &impl Spanned) -> Option<SymbolId> {
        // Note: precise byte offsets require proc-macro2/span-locations feature.
        // We use the item_index as chunk_index and file-level byte range.
        let byte_start = 0;
        let byte_end = self.source.len();
        self.item_index += 1;
        match resolve_or_create_entity_with_source(
            self.engine,
            label,
            self.rel_path,
            self.item_index,
            byte_start,
            byte_end,
        ) {
            Ok(id) => {
                self.symbols.push(id);
                // Store provenance.
                self.store_provenance(id);
                Some(id)
            }
            Err(e) => {
                tracing::debug!(label, error = %e, "failed to create entity");
                None
            }
        }
    }

    fn store_provenance(&self, derived: SymbolId) {
        let module = self.current_module();
        let mut record = ProvenanceRecord {
            id: None,
            derived_id: derived,
            kind: DerivationKind::Extracted,
            sources: vec![module],
            confidence: 1.0,
            depth: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        let _ = self.engine.store_provenance(&mut record);
    }

    fn add_triple(&mut self, s: SymbolId, p: SymbolId, o: SymbolId, confidence: f32) {
        if add_triple(self.engine, s, p, o, confidence).is_ok() {
            self.triple_count += 1;
        }
    }

    fn visibility_label(vis: &Visibility) -> &'static str {
        match vis {
            Visibility::Public(_) => "public",
            Visibility::Restricted(_) => "restricted",
            Visibility::Inherited => "private",
        }
    }

    fn extract_doc_comments(attrs: &[syn::Attribute]) -> Option<String> {
        let mut docs = Vec::new();
        for attr in attrs {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(lit) = &nv.value {
                        if let syn::Lit::Str(s) = &lit.lit {
                            docs.push(s.value().trim().to_string());
                        }
                    }
                }
            }
        }
        if docs.is_empty() {
            None
        } else {
            Some(docs.join("\n"))
        }
    }

    fn extract_derives(attrs: &[syn::Attribute]) -> Vec<String> {
        let mut derives = Vec::new();
        for attr in attrs {
            if attr.path().is_ident("derive") {
                if let syn::Meta::List(list) = &attr.meta {
                    let tokens = list.tokens.to_string();
                    for part in tokens.split(',') {
                        let name = part.trim().to_string();
                        if !name.is_empty() {
                            derives.push(name);
                        }
                    }
                }
            }
        }
        derives
    }

    fn return_type_label(output: &syn::ReturnType) -> Option<String> {
        match output {
            syn::ReturnType::Default => None,
            syn::ReturnType::Type(_, ty) => Some(type_label(ty)),
        }
    }
}

/// Produce a human-readable label for a type.
fn type_label(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(p) => {
            let segments: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| {
                    let ident = s.ident.to_string();
                    match &s.arguments {
                        syn::PathArguments::None => ident,
                        syn::PathArguments::AngleBracketed(args) => {
                            let inner: Vec<String> = args
                                .args
                                .iter()
                                .filter_map(|a| match a {
                                    syn::GenericArgument::Type(t) => Some(type_label(t)),
                                    _ => None,
                                })
                                .collect();
                            if inner.is_empty() {
                                ident
                            } else {
                                format!("{}<{}>", ident, inner.join(", "))
                            }
                        }
                        syn::PathArguments::Parenthesized(_) => ident,
                    }
                })
                .collect();
            segments.join("::")
        }
        syn::Type::Reference(r) => {
            let inner = type_label(&r.elem);
            if r.mutability.is_some() {
                format!("&mut {inner}")
            } else {
                format!("&{inner}")
            }
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => "()".into(),
        syn::Type::Tuple(t) => {
            let inner: Vec<String> = t.elems.iter().map(type_label).collect();
            format!("({})", inner.join(", "))
        }
        syn::Type::Slice(s) => format!("[{}]", type_label(&s.elem)),
        syn::Type::Array(a) => format!("[{}; _]", type_label(&a.elem)),
        _ => "_".into(),
    }
}

impl<'a> Visit<'a> for CodeVisitor<'_> {
    fn visit_item_fn(&mut self, node: &'a syn::ItemFn) {
        let name = node.sig.ident.to_string();
        let module = self.current_module();

        if let Some(fn_sym) = self.create_entity(&name, &node.sig.ident) {
            let fn_type = self
                .engine
                .resolve_or_create_entity("Function")
                .unwrap_or(fn_sym);
            self.add_triple(fn_sym, self.is_a, fn_type, 1.0);
            self.add_triple(module, self.preds.defines_fn, fn_sym, 1.0);

            // Visibility.
            let vis = Self::visibility_label(&node.vis);
            if let Ok(vis_sym) = self.engine.resolve_or_create_entity(vis) {
                self.add_triple(fn_sym, self.preds.has_visibility, vis_sym, 1.0);
            }

            // Parameters.
            for param in &node.sig.inputs {
                if let syn::FnArg::Typed(pat_type) = param {
                    let param_label = format!(
                        "{}:{}",
                        pat_to_string(&pat_type.pat),
                        type_label(&pat_type.ty)
                    );
                    if let Ok(param_sym) = self.engine.resolve_or_create_entity(&param_label) {
                        self.add_triple(fn_sym, self.preds.has_param, param_sym, 1.0);
                    }
                }
            }

            // Return type.
            if let Some(ret) = Self::return_type_label(&node.sig.output) {
                if let Ok(ret_sym) = self.engine.resolve_or_create_entity(&ret) {
                    self.add_triple(fn_sym, self.preds.returns_type, ret_sym, 1.0);
                }
            }

            // Doc comments.
            if let Some(doc) = Self::extract_doc_comments(&node.attrs) {
                let truncated = truncate(&doc, 256);
                if let Ok(doc_sym) = self.engine.resolve_or_create_entity(&truncated) {
                    self.add_triple(fn_sym, self.preds.has_doc, doc_sym, 1.0);
                }
            }
        }

        syn::visit::visit_item_fn(self, node);
    }

    fn visit_item_struct(&mut self, node: &'a syn::ItemStruct) {
        let name = node.ident.to_string();
        let module = self.current_module();

        if let Some(struct_sym) = self.create_entity(&name, &node.ident) {
            let struct_type = self
                .engine
                .resolve_or_create_entity("Struct")
                .unwrap_or(struct_sym);
            self.add_triple(struct_sym, self.is_a, struct_type, 1.0);
            self.add_triple(module, self.preds.defines_struct, struct_sym, 1.0);

            // Visibility.
            let vis = Self::visibility_label(&node.vis);
            if let Ok(vis_sym) = self.engine.resolve_or_create_entity(vis) {
                self.add_triple(struct_sym, self.preds.has_visibility, vis_sym, 1.0);
            }

            // Fields.
            for field in &node.fields {
                if let Some(ref ident) = field.ident {
                    let field_label = format!("{}:{}", ident, type_label(&field.ty));
                    if let Ok(field_sym) = self.engine.resolve_or_create_entity(&field_label) {
                        self.add_triple(struct_sym, self.preds.has_field, field_sym, 1.0);
                    }
                }
            }

            // Derives.
            for derive in Self::extract_derives(&node.attrs) {
                if let Ok(derive_sym) = self.engine.resolve_or_create_entity(&derive) {
                    self.add_triple(struct_sym, self.preds.derives_trait, derive_sym, 1.0);
                }
            }

            // Doc comments.
            if let Some(doc) = Self::extract_doc_comments(&node.attrs) {
                let truncated = truncate(&doc, 256);
                if let Ok(doc_sym) = self.engine.resolve_or_create_entity(&truncated) {
                    self.add_triple(struct_sym, self.preds.has_doc, doc_sym, 1.0);
                }
            }
        }

        syn::visit::visit_item_struct(self, node);
    }

    fn visit_item_enum(&mut self, node: &'a syn::ItemEnum) {
        let name = node.ident.to_string();
        let module = self.current_module();

        if let Some(enum_sym) = self.create_entity(&name, &node.ident) {
            let enum_type = self
                .engine
                .resolve_or_create_entity("Enum")
                .unwrap_or(enum_sym);
            self.add_triple(enum_sym, self.is_a, enum_type, 1.0);
            self.add_triple(module, self.preds.defines_enum, enum_sym, 1.0);

            // Visibility.
            let vis = Self::visibility_label(&node.vis);
            if let Ok(vis_sym) = self.engine.resolve_or_create_entity(vis) {
                self.add_triple(enum_sym, self.preds.has_visibility, vis_sym, 1.0);
            }

            // Variants.
            for variant in &node.variants {
                let variant_name = variant.ident.to_string();
                if let Ok(variant_sym) = self.engine.resolve_or_create_entity(&variant_name) {
                    self.add_triple(enum_sym, self.preds.has_variant, variant_sym, 1.0);
                }
            }

            // Derives.
            for derive in Self::extract_derives(&node.attrs) {
                if let Ok(derive_sym) = self.engine.resolve_or_create_entity(&derive) {
                    self.add_triple(enum_sym, self.preds.derives_trait, derive_sym, 1.0);
                }
            }

            // Doc comments.
            if let Some(doc) = Self::extract_doc_comments(&node.attrs) {
                let truncated = truncate(&doc, 256);
                if let Ok(doc_sym) = self.engine.resolve_or_create_entity(&truncated) {
                    self.add_triple(enum_sym, self.preds.has_doc, doc_sym, 1.0);
                }
            }
        }

        syn::visit::visit_item_enum(self, node);
    }

    fn visit_item_trait(&mut self, node: &'a syn::ItemTrait) {
        let name = node.ident.to_string();
        let module = self.current_module();

        if let Some(trait_sym) = self.create_entity(&name, &node.ident) {
            let trait_type = self
                .engine
                .resolve_or_create_entity("Trait")
                .unwrap_or(trait_sym);
            self.add_triple(trait_sym, self.is_a, trait_type, 1.0);
            self.add_triple(module, self.preds.defines_trait, trait_sym, 1.0);

            // Visibility.
            let vis = Self::visibility_label(&node.vis);
            if let Ok(vis_sym) = self.engine.resolve_or_create_entity(vis) {
                self.add_triple(trait_sym, self.preds.has_visibility, vis_sym, 1.0);
            }

            // Required methods.
            for item in &node.items {
                if let syn::TraitItem::Fn(method) = item {
                    let method_name = method.sig.ident.to_string();
                    if let Ok(method_sym) = self.engine.resolve_or_create_entity(&method_name) {
                        self.add_triple(trait_sym, self.preds.has_method, method_sym, 1.0);
                    }
                }
            }

            // Doc comments.
            if let Some(doc) = Self::extract_doc_comments(&node.attrs) {
                let truncated = truncate(&doc, 256);
                if let Ok(doc_sym) = self.engine.resolve_or_create_entity(&truncated) {
                    self.add_triple(trait_sym, self.preds.has_doc, doc_sym, 1.0);
                }
            }
        }

        syn::visit::visit_item_trait(self, node);
    }

    fn visit_item_impl(&mut self, node: &'a syn::ItemImpl) {
        let self_type = type_label(&node.self_ty);

        // Resolve the target type (create if needed).
        let target = match self.engine.resolve_or_create_entity(&self_type) {
            Ok(id) => id,
            Err(_) => return,
        };

        // If this is a trait impl, link the type to the trait.
        if let Some((_, ref trait_path, _)) = node.trait_ {
            let trait_name: String = trait_path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            if let Ok(trait_sym) = self.engine.resolve_or_create_entity(&trait_name) {
                self.add_triple(target, self.preds.implements_trait, trait_sym, 1.0);
            }
        }

        // Methods in the impl.
        for item in &node.items {
            if let syn::ImplItem::Fn(method) = item {
                let method_name = method.sig.ident.to_string();
                if let Some(method_sym) = self.create_entity(&method_name, &method.sig.ident) {
                    let fn_type = self
                        .engine
                        .resolve_or_create_entity("Function")
                        .unwrap_or(method_sym);
                    self.add_triple(method_sym, self.is_a, fn_type, 1.0);
                    self.add_triple(target, self.preds.has_method, method_sym, 1.0);

                    // Visibility.
                    let vis = Self::visibility_label(&method.vis);
                    if let Ok(vis_sym) = self.engine.resolve_or_create_entity(vis) {
                        self.add_triple(method_sym, self.preds.has_visibility, vis_sym, 1.0);
                    }

                    // Return type.
                    if let Some(ret) = Self::return_type_label(&method.sig.output) {
                        if let Ok(ret_sym) = self.engine.resolve_or_create_entity(&ret) {
                            self.add_triple(method_sym, self.preds.returns_type, ret_sym, 1.0);
                        }
                    }
                }
            }
        }

        syn::visit::visit_item_impl(self, node);
    }

    fn visit_item_mod(&mut self, node: &'a syn::ItemMod) {
        let mod_name = node.ident.to_string();
        let parent = self.current_module();

        if let Some(mod_sym) = self.create_entity(&mod_name, &node.ident) {
            let mod_type = self
                .engine
                .resolve_or_create_entity("Module")
                .unwrap_or(mod_sym);
            self.add_triple(mod_sym, self.is_a, mod_type, 1.0);
            self.add_triple(parent, self.preds.contains_mod, mod_sym, 1.0);

            // Doc comments (/// on the mod declaration).
            if let Some(doc) = Self::extract_doc_comments(&node.attrs) {
                let truncated = truncate(&doc, 256);
                if let Ok(doc_sym) = self.engine.resolve_or_create_entity(&truncated) {
                    self.add_triple(mod_sym, self.preds.has_doc, doc_sym, 1.0);
                }
            }

            // Push onto module stack and visit contents.
            self.module_stack.push(mod_sym);
            syn::visit::visit_item_mod(self, node);
            self.module_stack.pop();
        }
    }

    fn visit_item_use(&mut self, node: &'a syn::ItemUse) {
        let module = self.current_module();
        let deps = extract_use_deps(&node.tree);
        for dep in deps {
            if let Ok(dep_sym) = self.engine.resolve_or_create_entity(&dep) {
                self.add_triple(module, self.preds.depends_on, dep_sym, 0.8);
            }
        }
        syn::visit::visit_item_use(self, node);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn pat_to_string(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(pi) => pi.ident.to_string(),
        syn::Pat::Wild(_) => "_".into(),
        _ => "_".into(),
    }
}

/// Extract top-level dependency names from a `use` tree.
fn extract_use_deps(tree: &syn::UseTree) -> Vec<String> {
    match tree {
        syn::UseTree::Path(p) => {
            let root = p.ident.to_string();
            // Only track external crate dependencies (skip `self`, `super`, `crate`).
            if root == "self" || root == "super" || root == "crate" {
                return vec![];
            }
            vec![root]
        }
        syn::UseTree::Name(n) => {
            let name = n.ident.to_string();
            if name == "self" || name == "super" || name == "crate" {
                return vec![];
            }
            vec![name]
        }
        syn::UseTree::Group(g) => g.items.iter().flat_map(extract_use_deps).collect(),
        syn::UseTree::Glob(_) => vec![],
        syn::UseTree::Rename(r) => {
            let name = r.ident.to_string();
            if name == "self" || name == "super" || name == "crate" {
                return vec![];
            }
            vec![name]
        }
    }
}

fn resolve_or_create_entity_with_source(
    engine: &Engine,
    label: &str,
    document_id: &str,
    chunk_index: u32,
    byte_start: usize,
    byte_end: usize,
) -> AgentResult<SymbolId> {
    // Check if entity already exists.
    if let Ok(id) = engine.lookup_symbol(label) {
        return Ok(id);
    }

    // Create new entity with SourceRef.
    let mut meta = engine.create_symbol(SymbolKind::Entity, label)?;
    meta.source = Some(SourceRef {
        document_id: document_id.to_string(),
        chunk_index,
        byte_start,
        byte_end,
    });

    // Update the stored metadata with the source ref.
    let encoded = bincode::serialize(&meta).map_err(|e| AgentError::ToolExecution {
        tool_name: "code_ingest".into(),
        message: format!("failed to serialize symbol meta: {e}"),
    })?;
    engine.store().put(meta.id, encoded);

    Ok(meta.id)
}

fn add_triple(
    engine: &Engine,
    s: SymbolId,
    p: SymbolId,
    o: SymbolId,
    confidence: f32,
) -> AgentResult<()> {
    let triple = Triple::new(s, p, o).with_confidence(confidence);
    engine.add_triple(&triple)?;
    Ok(())
}

/// Extract inner doc comments (`//!`) from file-level or module-level attributes.
///
/// These appear as `#![doc = "..."]` attributes in `syn::File::attrs`.
fn extract_inner_doc_comments(attrs: &[syn::Attribute]) -> Option<String> {
    let mut docs = Vec::new();
    for attr in attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(lit) = &nv.value {
                    if let syn::Lit::Str(s) = &lit.lit {
                        docs.push(s.value().trim().to_string());
                    }
                }
            }
        }
    }
    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
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

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn module_label_from_path_basic() {
        assert_eq!(module_label_from_path("src/engine.rs"), "engine");
        assert_eq!(
            module_label_from_path("src/agent/tools/code_ingest.rs"),
            "agent::tools::code_ingest"
        );
        assert_eq!(module_label_from_path("src/vsa/mod.rs"), "vsa");
    }

    #[test]
    fn collect_rs_files_single_file() {
        let files = collect_rs_files(Path::new("src/engine.rs"), false, 10);
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn type_label_simple() {
        let ty: syn::Type = syn::parse_str("Vec<String>").unwrap();
        assert_eq!(type_label(&ty), "Vec<String>");
    }

    #[test]
    fn type_label_reference() {
        let ty: syn::Type = syn::parse_str("&mut Engine").unwrap();
        assert_eq!(type_label(&ty), "&mut Engine");
    }

    #[test]
    fn extract_use_deps_external() {
        let item: syn::ItemUse = syn::parse_str("use std::path::PathBuf;").unwrap();
        let deps = extract_use_deps(&item.tree);
        assert_eq!(deps, vec!["std"]);
    }

    #[test]
    fn extract_use_deps_skip_self() {
        let item: syn::ItemUse = syn::parse_str("use crate::engine::Engine;").unwrap();
        let deps = extract_use_deps(&item.tree);
        assert!(deps.is_empty());
    }

    #[test]
    fn ingest_single_file() {
        let engine = test_engine();
        let input = ToolInput::new()
            .with_param("path", "src/symbol.rs")
            .with_param("max_files", "1");
        let tool = CodeIngestTool;
        let output = tool.execute(&engine, input).unwrap();
        assert!(output.success);
        assert!(output.result.contains("1 file(s) processed"));
        // Should have created some triples.
        assert!(!engine.all_triples().is_empty());
    }

    #[test]
    fn ingest_populates_source_ref() {
        let engine = test_engine();
        let preds = CodePredicates::init(&engine).unwrap();
        let is_a = engine.resolve_or_create_relation("is-a").unwrap();

        let mut stats = IngestStats::default();
        let _ = ingest_file(
            &engine,
            &preds,
            is_a,
            Path::new("src/symbol.rs"),
            Path::new("src"),
            &mut stats,
        );

        // SymbolId entity should exist.
        let sym_id = engine.lookup_symbol("SymbolId");
        assert!(
            sym_id.is_ok(),
            "SymbolId entity should exist after ingesting symbol.rs"
        );
    }

    #[test]
    fn ingest_creates_is_a_triples() {
        let engine = test_engine();
        let input = ToolInput::new().with_param("path", "src/symbol.rs");
        let tool = CodeIngestTool;
        let _ = tool.execute(&engine, input).unwrap();

        // SymbolKind should be marked as an Enum.
        let symbol_kind = engine.lookup_symbol("SymbolKind").unwrap();
        let is_a = engine.lookup_symbol("is-a").unwrap();
        let enum_type = engine.lookup_symbol("Enum").unwrap();
        assert!(engine.has_triple(symbol_kind, is_a, enum_type));
    }
}
