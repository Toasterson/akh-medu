//! Rust code generation grammar archetype.
//!
//! Implements `ConcreteGrammar` to linearize `AbsTree` code nodes into valid
//! Rust source code. Handles:
//!
//! - `CodeSignature { kind: "fn" }` → function definition with signature
//! - `CodeSignature { kind: "struct" }` → struct with fields and derives
//! - `CodeSignature { kind: "enum" }` → enum with variants
//! - `CodeSignature { kind: "trait" }` → trait with method signatures
//! - `CodeSignature { kind: "impl" }` → impl block
//! - `CodeModule` → `mod` block with nested items
//! - `DataFlow` → iterator/pipeline chain comment
//!
//! `parse()` delegates to `syn` for Rust source → `AbsTree` round-trip.

use super::abs::{AbsTree, DataFlowStep};
use super::cat::Cat;
use super::concrete::{ConcreteGrammar, LinContext, ParseContext};
use super::error::{GrammarError, GrammarResult};

/// Rust code generation grammar archetype.
///
/// Linearizes `AbsTree` code-related nodes into valid, idiomatic Rust source.
/// Registered as `"rust-gen"` in the `GrammarRegistry`.
pub struct RustCodeGrammar;

impl RustCodeGrammar {
    /// Linearize a single node with the given indent level.
    fn linearize_node(&self, tree: &AbsTree, ctx: &LinContext, indent: usize) -> GrammarResult<String> {
        let prefix = "    ".repeat(indent);
        match tree {
            AbsTree::CodeModule {
                name,
                role: _,
                importance: _,
                doc_summary,
                children,
            } => {
                let mut out = String::new();
                if let Some(doc) = doc_summary {
                    out.push_str(&format!("{prefix}//! {doc}\n"));
                }
                out.push_str(&format!("{prefix}pub mod {name} {{\n"));
                for child in children {
                    let child_code = self.linearize_node(child, ctx, indent + 1)?;
                    out.push_str(&child_code);
                    out.push('\n');
                }
                out.push_str(&format!("{prefix}}}\n"));
                Ok(out)
            }

            AbsTree::CodeSignature {
                kind,
                name,
                doc_summary,
                params_or_fields,
                return_type,
                traits,
                importance: _,
            } => match kind.as_str() {
                "fn" => Ok(self.linearize_fn(
                    name,
                    doc_summary.as_deref(),
                    params_or_fields,
                    return_type.as_deref(),
                    &prefix,
                )),
                "struct" => Ok(self.linearize_struct(
                    name,
                    doc_summary.as_deref(),
                    params_or_fields,
                    traits,
                    &prefix,
                )),
                "enum" => Ok(self.linearize_enum(
                    name,
                    doc_summary.as_deref(),
                    params_or_fields,
                    traits,
                    &prefix,
                )),
                "trait" => Ok(self.linearize_trait(
                    name,
                    doc_summary.as_deref(),
                    params_or_fields,
                    &prefix,
                )),
                "impl" => Ok(self.linearize_impl(
                    name,
                    doc_summary.as_deref(),
                    params_or_fields,
                    return_type.as_deref(),
                    traits,
                    &prefix,
                )),
                other => Err(GrammarError::LinearizationFailed {
                    cat: Cat::CodeSignature,
                    grammar: "rust-gen".to_string(),
                    message: format!("unknown code signature kind: \"{other}\""),
                }),
            },

            AbsTree::DataFlow { steps } => {
                Ok(self.linearize_data_flow(steps, &prefix))
            }

            AbsTree::CodeFact { kind, name, detail } => {
                Ok(format!("{prefix}// {kind}: {name} — {detail}\n"))
            }

            AbsTree::Section { heading, body } => {
                let mut out = String::new();
                out.push_str(&format!("{prefix}// === {heading} ===\n\n"));
                for item in body {
                    let item_code = self.linearize_node(item, ctx, indent)?;
                    out.push_str(&item_code);
                    out.push('\n');
                }
                Ok(out)
            }

            AbsTree::Document {
                overview,
                sections,
                gaps: _,
            } => {
                let mut out = String::new();
                // Render overview as a module-level doc comment
                if let AbsTree::Freeform(text) = overview.as_ref() {
                    for line in text.lines() {
                        out.push_str(&format!("//! {line}\n"));
                    }
                    out.push('\n');
                }
                for section in sections {
                    let section_code = self.linearize_node(section, ctx, indent)?;
                    out.push_str(&section_code);
                    out.push('\n');
                }
                Ok(out)
            }

            AbsTree::Freeform(text) => Ok(format!("{prefix}// {text}\n")),

            AbsTree::Conjunction { items, is_and: _ } => {
                let mut out = String::new();
                for item in items {
                    let code = self.linearize_node(item, ctx, indent)?;
                    out.push_str(&code);
                    out.push('\n');
                }
                Ok(out)
            }

            AbsTree::WithConfidence { inner, .. }
            | AbsTree::WithProvenance { inner, .. } => {
                self.linearize_node(inner, ctx, indent)
            }

            other => Err(GrammarError::LinearizationFailed {
                cat: other.cat(),
                grammar: "rust-gen".to_string(),
                message: format!(
                    "rust-gen grammar does not handle {:?} nodes directly",
                    other.cat()
                ),
            }),
        }
    }

    /// Linearize a function definition.
    fn linearize_fn(
        &self,
        name: &str,
        doc: Option<&str>,
        params: &[String],
        return_type: Option<&str>,
        prefix: &str,
    ) -> String {
        let mut out = String::new();

        // Doc comment
        if let Some(doc) = doc {
            for line in doc.lines() {
                out.push_str(&format!("{prefix}/// {line}\n"));
            }
        }

        // Signature
        let params_str = if params.is_empty() {
            String::new()
        } else {
            params.join(", ")
        };

        let ret_str = match return_type {
            Some(t) if !t.is_empty() => format!(" -> {t}"),
            _ => String::new(),
        };

        out.push_str(&format!(
            "{prefix}pub fn {name}({params_str}){ret_str} {{\n"
        ));
        out.push_str(&format!("{prefix}    todo!()\n"));
        out.push_str(&format!("{prefix}}}\n"));

        out
    }

    /// Linearize a struct definition.
    fn linearize_struct(
        &self,
        name: &str,
        doc: Option<&str>,
        fields: &[String],
        derives: &[String],
        prefix: &str,
    ) -> String {
        let mut out = String::new();

        if let Some(doc) = doc {
            for line in doc.lines() {
                out.push_str(&format!("{prefix}/// {line}\n"));
            }
        }

        if !derives.is_empty() {
            out.push_str(&format!(
                "{prefix}#[derive({})]\n",
                derives.join(", ")
            ));
        }

        if fields.is_empty() {
            out.push_str(&format!("{prefix}pub struct {name};\n"));
        } else {
            out.push_str(&format!("{prefix}pub struct {name} {{\n"));
            for field in fields {
                // Fields may come as "name: Type" or just "name"
                if field.contains(':') {
                    out.push_str(&format!("{prefix}    pub {field},\n"));
                } else {
                    out.push_str(&format!("{prefix}    pub {field}: (),\n"));
                }
            }
            out.push_str(&format!("{prefix}}}\n"));
        }

        out
    }

    /// Linearize an enum definition.
    fn linearize_enum(
        &self,
        name: &str,
        doc: Option<&str>,
        variants: &[String],
        derives: &[String],
        prefix: &str,
    ) -> String {
        let mut out = String::new();

        if let Some(doc) = doc {
            for line in doc.lines() {
                out.push_str(&format!("{prefix}/// {line}\n"));
            }
        }

        if !derives.is_empty() {
            out.push_str(&format!(
                "{prefix}#[derive({})]\n",
                derives.join(", ")
            ));
        }

        out.push_str(&format!("{prefix}pub enum {name} {{\n"));
        for variant in variants {
            out.push_str(&format!("{prefix}    {variant},\n"));
        }
        out.push_str(&format!("{prefix}}}\n"));

        out
    }

    /// Linearize a trait definition.
    fn linearize_trait(
        &self,
        name: &str,
        doc: Option<&str>,
        methods: &[String],
        prefix: &str,
    ) -> String {
        let mut out = String::new();

        if let Some(doc) = doc {
            for line in doc.lines() {
                out.push_str(&format!("{prefix}/// {line}\n"));
            }
        }

        out.push_str(&format!("{prefix}pub trait {name} {{\n"));
        for method in methods {
            // Methods come as signature strings, e.g., "fn name(&self) -> Type"
            out.push_str(&format!("{prefix}    {method};\n"));
        }
        out.push_str(&format!("{prefix}}}\n"));

        out
    }

    /// Linearize an impl block.
    fn linearize_impl(
        &self,
        name: &str,
        doc: Option<&str>,
        methods: &[String],
        target: Option<&str>,
        traits: &[String],
        prefix: &str,
    ) -> String {
        let mut out = String::new();

        if let Some(doc) = doc {
            for line in doc.lines() {
                out.push_str(&format!("{prefix}/// {line}\n"));
            }
        }

        // `target` is the type being implemented for.
        // `traits` first element is the trait name if this is a trait impl.
        let impl_header = if let Some(trait_name) = traits.first() {
            let target_type = target.unwrap_or(name);
            format!("{prefix}impl {trait_name} for {target_type} {{\n")
        } else {
            format!("{prefix}impl {name} {{\n")
        };
        out.push_str(&impl_header);

        for method in methods {
            // Each method is a name; generate a stub
            out.push_str(&format!("{prefix}    pub fn {method}(&self) {{\n"));
            out.push_str(&format!("{prefix}        todo!()\n"));
            out.push_str(&format!("{prefix}    }}\n\n"));
        }

        out.push_str(&format!("{prefix}}}\n"));

        out
    }

    /// Linearize a data flow chain as a pipeline comment or iterator chain.
    fn linearize_data_flow(&self, steps: &[DataFlowStep], prefix: &str) -> String {
        if steps.is_empty() {
            return format!("{prefix}// (empty pipeline)\n");
        }

        let mut out = String::new();
        out.push_str(&format!("{prefix}// Pipeline:\n"));

        let chain: Vec<String> = steps
            .iter()
            .map(|s| match &s.via_type {
                Some(t) => format!(".{}() /* {} */", s.name, t),
                None => format!(".{}()", s.name),
            })
            .collect();

        // If short enough, put on one line
        let chain_str = chain.join("");
        if chain_str.len() < 80 {
            out.push_str(&format!("{prefix}// input{chain_str}\n"));
        } else {
            out.push_str(&format!("{prefix}// input\n"));
            for step in &chain {
                out.push_str(&format!("{prefix}//     {step}\n"));
            }
        }

        out
    }
}

impl ConcreteGrammar for RustCodeGrammar {
    fn name(&self) -> &str {
        "rust-gen"
    }

    fn description(&self) -> &str {
        "Rust code generation grammar: linearizes AbsTree code nodes into valid Rust source"
    }

    fn linearize(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String> {
        self.linearize_node(tree, ctx, 0)
    }

    fn parse(
        &self,
        input: &str,
        _expected_cat: Option<Cat>,
        _ctx: &ParseContext,
    ) -> GrammarResult<AbsTree> {
        // Parse Rust source using syn to extract structure as AbsTree
        parse_rust_source(input)
    }

    fn supported_categories(&self) -> &[Cat] {
        &[
            Cat::CodeModule,
            Cat::CodeSignature,
            Cat::DataFlow,
            Cat::CodeFact,
            Cat::Section,
            Cat::Document,
            Cat::Freeform,
        ]
    }
}

// ---------------------------------------------------------------------------
// Parsing: Rust source → AbsTree
// ---------------------------------------------------------------------------

/// Parse Rust source code into an AbsTree using `syn`.
fn parse_rust_source(input: &str) -> GrammarResult<AbsTree> {
    let file: syn::File = syn::parse_str(input).map_err(|e| GrammarError::ParseFailed {
        input: format!("Rust parse error: {e}"),
    })?;

    let mut children = Vec::new();
    for item in &file.items {
        if let Some(node) = item_to_abstree(item) {
            children.push(node);
        }
    }

    if children.is_empty() {
        Ok(AbsTree::Freeform(input.to_string()))
    } else if children.len() == 1 {
        Ok(children.into_iter().next().unwrap())
    } else {
        Ok(AbsTree::CodeModule {
            name: "parsed".to_string(),
            role: None,
            importance: None,
            doc_summary: None,
            children,
        })
    }
}

/// Convert a single `syn::Item` to an `AbsTree` node.
fn item_to_abstree(item: &syn::Item) -> Option<AbsTree> {
    match item {
        syn::Item::Fn(f) => {
            let name = f.sig.ident.to_string();
            let params: Vec<String> = f
                .sig
                .inputs
                .iter()
                .map(|arg| match arg {
                    syn::FnArg::Receiver(_) => "&self".to_string(),
                    syn::FnArg::Typed(pat) => {
                        let pat_str = quote::quote!(#pat).to_string();
                        pat_str
                    }
                })
                .collect();
            let return_type = match &f.sig.output {
                syn::ReturnType::Default => None,
                syn::ReturnType::Type(_, ty) => Some(quote::quote!(#ty).to_string()),
            };
            let doc = extract_doc_attrs(&f.attrs);
            Some(AbsTree::CodeSignature {
                kind: "fn".to_string(),
                name,
                doc_summary: doc,
                params_or_fields: params,
                return_type,
                traits: vec![],
                importance: None,
            })
        }
        syn::Item::Struct(s) => {
            let name = s.ident.to_string();
            let fields: Vec<String> = match &s.fields {
                syn::Fields::Named(named) => named
                    .named
                    .iter()
                    .map(|f| {
                        let fname = f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default();
                        let ftype = quote::quote!(#f.ty).to_string();
                        format!("{fname}: {ftype}")
                    })
                    .collect(),
                syn::Fields::Unnamed(unnamed) => unnamed
                    .unnamed
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        let ftype = quote::quote!(#f.ty).to_string();
                        format!("field_{i}: {ftype}")
                    })
                    .collect(),
                syn::Fields::Unit => vec![],
            };
            let derives = extract_derives(&s.attrs);
            let doc = extract_doc_attrs(&s.attrs);
            Some(AbsTree::CodeSignature {
                kind: "struct".to_string(),
                name,
                doc_summary: doc,
                params_or_fields: fields,
                return_type: None,
                traits: derives,
                importance: None,
            })
        }
        syn::Item::Enum(e) => {
            let name = e.ident.to_string();
            let variants: Vec<String> = e.variants.iter().map(|v| v.ident.to_string()).collect();
            let derives = extract_derives(&e.attrs);
            let doc = extract_doc_attrs(&e.attrs);
            Some(AbsTree::CodeSignature {
                kind: "enum".to_string(),
                name,
                doc_summary: doc,
                params_or_fields: variants,
                return_type: None,
                traits: derives,
                importance: None,
            })
        }
        syn::Item::Trait(t) => {
            let name = t.ident.to_string();
            let methods: Vec<String> = t
                .items
                .iter()
                .filter_map(|item| {
                    if let syn::TraitItem::Fn(m) = item {
                        Some(format!("fn {}", m.sig.ident))
                    } else {
                        None
                    }
                })
                .collect();
            let doc = extract_doc_attrs(&t.attrs);
            Some(AbsTree::CodeSignature {
                kind: "trait".to_string(),
                name,
                doc_summary: doc,
                params_or_fields: methods,
                return_type: None,
                traits: vec![],
                importance: None,
            })
        }
        syn::Item::Impl(i) => {
            let target = quote::quote!(#i.self_ty).to_string();
            let trait_name = i.trait_.as_ref().map(|(_, path, _)| {
                quote::quote!(#path).to_string()
            });
            let methods: Vec<String> = i
                .items
                .iter()
                .filter_map(|item| {
                    if let syn::ImplItem::Fn(m) = item {
                        Some(m.sig.ident.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            Some(AbsTree::CodeSignature {
                kind: "impl".to_string(),
                name: target,
                doc_summary: None,
                params_or_fields: methods,
                return_type: None,
                traits: trait_name.into_iter().collect(),
                importance: None,
            })
        }
        syn::Item::Mod(m) => {
            let name = m.ident.to_string();
            let children = m
                .content
                .as_ref()
                .map(|(_, items)| {
                    items.iter().filter_map(item_to_abstree).collect()
                })
                .unwrap_or_default();
            let doc = extract_doc_attrs(&m.attrs);
            Some(AbsTree::CodeModule {
                name,
                role: None,
                importance: None,
                doc_summary: doc,
                children,
            })
        }
        _ => None,
    }
}

/// Extract `#[doc = "..."]` attributes into a single doc string.
fn extract_doc_attrs(attrs: &[syn::Attribute]) -> Option<String> {
    let docs: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) = &nv.value
                    {
                        return Some(s.value().trim().to_string());
                    }
                }
            }
            None
        })
        .collect();

    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

/// Extract derive macro names from attributes.
fn extract_derives(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("derive") {
                if let syn::Meta::List(list) = &attr.meta {
                    let tokens = list.tokens.to_string();
                    return Some(
                        tokens
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect::<Vec<_>>(),
                    );
                }
            }
            None
        })
        .flatten()
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::abs::DataFlowStep;
    use crate::grammar::concrete::LinContext;

    fn default_ctx() -> LinContext<'static> {
        LinContext::default()
    }

    #[test]
    fn linearize_fn_basic() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeSignature {
            kind: "fn".to_string(),
            name: "hello".to_string(),
            doc_summary: Some("Greets the world.".to_string()),
            params_or_fields: vec!["name: &str".to_string()],
            return_type: Some("String".to_string()),
            traits: vec![],
            importance: None,
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("/// Greets the world."));
        assert!(code.contains("pub fn hello(name: &str) -> String"));
        assert!(code.contains("todo!()"));
    }

    #[test]
    fn linearize_struct_with_derives() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeSignature {
            kind: "struct".to_string(),
            name: "Config".to_string(),
            doc_summary: Some("Configuration settings.".to_string()),
            params_or_fields: vec!["name: String".to_string(), "value: u64".to_string()],
            return_type: None,
            traits: vec!["Debug".to_string(), "Clone".to_string()],
            importance: None,
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("#[derive(Debug, Clone)]"));
        assert!(code.contains("pub struct Config {"));
        assert!(code.contains("pub name: String,"));
        assert!(code.contains("pub value: u64,"));
    }

    #[test]
    fn linearize_enum() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeSignature {
            kind: "enum".to_string(),
            name: "Color".to_string(),
            doc_summary: None,
            params_or_fields: vec![
                "Red".to_string(),
                "Green".to_string(),
                "Blue".to_string(),
            ],
            return_type: None,
            traits: vec!["Debug".to_string()],
            importance: None,
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("#[derive(Debug)]"));
        assert!(code.contains("pub enum Color {"));
        assert!(code.contains("Red,"));
        assert!(code.contains("Green,"));
        assert!(code.contains("Blue,"));
    }

    #[test]
    fn linearize_trait() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeSignature {
            kind: "trait".to_string(),
            name: "Drawable".to_string(),
            doc_summary: Some("Things that can be drawn.".to_string()),
            params_or_fields: vec![
                "fn draw(&self)".to_string(),
                "fn bounds(&self) -> Rect".to_string(),
            ],
            return_type: None,
            traits: vec![],
            importance: None,
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("pub trait Drawable {"));
        assert!(code.contains("fn draw(&self);"));
        assert!(code.contains("fn bounds(&self) -> Rect;"));
    }

    #[test]
    fn linearize_impl_block() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeSignature {
            kind: "impl".to_string(),
            name: "Config".to_string(),
            doc_summary: None,
            params_or_fields: vec!["new".to_string(), "validate".to_string()],
            return_type: None,
            traits: vec![],
            importance: None,
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("impl Config {"));
        assert!(code.contains("pub fn new(&self)"));
        assert!(code.contains("pub fn validate(&self)"));
    }

    #[test]
    fn linearize_trait_impl() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeSignature {
            kind: "impl".to_string(),
            name: "Config".to_string(),
            doc_summary: None,
            params_or_fields: vec!["draw".to_string()],
            return_type: Some("Config".to_string()),
            traits: vec!["Drawable".to_string()],
            importance: None,
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("impl Drawable for Config {"));
    }

    #[test]
    fn linearize_module() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeModule {
            name: "utils".to_string(),
            role: Some("utility".to_string()),
            importance: None,
            doc_summary: Some("Utility functions.".to_string()),
            children: vec![AbsTree::CodeSignature {
                kind: "fn".to_string(),
                name: "helper".to_string(),
                doc_summary: None,
                params_or_fields: vec![],
                return_type: None,
                traits: vec![],
                importance: None,
            }],
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("//! Utility functions."));
        assert!(code.contains("pub mod utils {"));
        assert!(code.contains("pub fn helper()"));
    }

    #[test]
    fn linearize_data_flow() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::DataFlow {
            steps: vec![
                DataFlowStep {
                    name: "filter".to_string(),
                    via_type: Some("Iterator".to_string()),
                },
                DataFlowStep {
                    name: "map".to_string(),
                    via_type: None,
                },
                DataFlowStep {
                    name: "collect".to_string(),
                    via_type: Some("Vec<T>".to_string()),
                },
            ],
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("Pipeline:"));
        assert!(code.contains("filter"));
        assert!(code.contains("map"));
        assert!(code.contains("collect"));
    }

    #[test]
    fn linearize_unit_struct() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeSignature {
            kind: "struct".to_string(),
            name: "Marker".to_string(),
            doc_summary: None,
            params_or_fields: vec![],
            return_type: None,
            traits: vec![],
            importance: None,
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("pub struct Marker;"));
    }

    #[test]
    fn linearize_document() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::Document {
            overview: Box::new(AbsTree::Freeform("A test module.".to_string())),
            sections: vec![AbsTree::Section {
                heading: "Types".to_string(),
                body: vec![AbsTree::CodeSignature {
                    kind: "struct".to_string(),
                    name: "Foo".to_string(),
                    doc_summary: None,
                    params_or_fields: vec![],
                    return_type: None,
                    traits: vec![],
                    importance: None,
                }],
            }],
            gaps: vec![],
        };

        let code = grammar.linearize(&tree, &default_ctx()).unwrap();
        assert!(code.contains("//! A test module."));
        assert!(code.contains("// === Types ==="));
        assert!(code.contains("pub struct Foo;"));
    }

    #[test]
    fn parse_rust_fn() {
        let input = r#"
pub fn hello(name: &str) -> String {
    format!("Hello, {name}")
}
"#;
        let grammar = RustCodeGrammar;
        let tree = grammar
            .parse(input, None, &ParseContext::default())
            .unwrap();

        match tree {
            AbsTree::CodeSignature { kind, name, .. } => {
                assert_eq!(kind, "fn");
                assert_eq!(name, "hello");
            }
            _ => panic!("Expected CodeSignature, got {:?}", tree.cat()),
        }
    }

    #[test]
    fn parse_rust_struct() {
        let input = r#"
#[derive(Debug, Clone)]
pub struct Config {
    pub name: String,
    pub value: u64,
}
"#;
        let grammar = RustCodeGrammar;
        let tree = grammar
            .parse(input, None, &ParseContext::default())
            .unwrap();

        match tree {
            AbsTree::CodeSignature {
                kind,
                name,
                traits,
                params_or_fields,
                ..
            } => {
                assert_eq!(kind, "struct");
                assert_eq!(name, "Config");
                assert!(traits.contains(&"Debug".to_string()));
                assert!(traits.contains(&"Clone".to_string()));
                assert_eq!(params_or_fields.len(), 2);
            }
            _ => panic!("Expected CodeSignature, got {:?}", tree.cat()),
        }
    }

    #[test]
    fn round_trip_struct() {
        let grammar = RustCodeGrammar;
        let original = AbsTree::CodeSignature {
            kind: "struct".to_string(),
            name: "Point".to_string(),
            doc_summary: None,
            params_or_fields: vec!["x: f64".to_string(), "y: f64".to_string()],
            return_type: None,
            traits: vec!["Debug".to_string()],
            importance: None,
        };

        let code = grammar.linearize(&original, &default_ctx()).unwrap();
        let parsed = grammar
            .parse(&code, None, &ParseContext::default())
            .unwrap();

        match parsed {
            AbsTree::CodeSignature { kind, name, .. } => {
                assert_eq!(kind, "struct");
                assert_eq!(name, "Point");
            }
            _ => panic!("Round-trip failed: expected CodeSignature"),
        }
    }

    #[test]
    fn unknown_kind_errors() {
        let grammar = RustCodeGrammar;
        let tree = AbsTree::CodeSignature {
            kind: "macro".to_string(),
            name: "foo".to_string(),
            doc_summary: None,
            params_or_fields: vec![],
            return_type: None,
            traits: vec![],
            importance: None,
        };

        let result = grammar.linearize(&tree, &default_ctx());
        assert!(result.is_err());
    }
}
