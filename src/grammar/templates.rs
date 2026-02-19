//! Parameterized code templates for common Rust patterns.
//!
//! Templates bridge the gap between the abstract `AbsTree` representation
//! (which lacks attribute support) and the full expressiveness of Rust syntax.
//! Each template is a parameterized code generator that produces valid,
//! idiomatic Rust source with proper derives, attributes, and patterns.
//!
//! ## Built-in templates
//!
//! - **error-type**: `#[derive(Debug, Error, Diagnostic)]` enum with variants
//! - **trait-impl**: `impl Trait for Type` with method stubs
//! - **builder**: fluent builder pattern with setter methods + `build()`
//! - **from-impl**: `impl From<Source> for Target` conversion
//! - **test-module**: `#[cfg(test)] mod tests` with test functions
//! - **iterator**: `impl Iterator for Type` with `Item` and `next()`
//! - **new-constructor**: `impl Type { pub fn new(...) -> Self }`

use std::collections::HashMap;

use super::error::{GrammarError, GrammarResult};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// What kind of parameter a template slot expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamKind {
    /// A type name (e.g., "MyError", "Config").
    TypeName,
    /// A list of named fields as "name: Type" strings.
    FieldList,
    /// A trait name (e.g., "Display", "Iterator").
    TraitName,
    /// A module name.
    ModuleName,
    /// A list of enum variant specifications.
    VariantList,
    /// A list of method signatures.
    MethodList,
    /// Free-form text (doc comment, error message, etc.).
    Text,
}

/// A single template parameter slot.
#[derive(Debug, Clone)]
pub struct TemplateParam {
    /// Parameter name used as `{{name}}` placeholder.
    pub name: String,
    /// What kind of value this parameter expects.
    pub kind: ParamKind,
    /// Whether this parameter must be provided.
    pub required: bool,
    /// Default value if not provided.
    pub default: Option<String>,
    /// Human-readable description.
    pub description: String,
}

impl TemplateParam {
    fn required(name: &str, kind: ParamKind, description: &str) -> Self {
        Self {
            name: name.to_string(),
            kind,
            required: true,
            default: None,
            description: description.to_string(),
        }
    }

    fn optional(name: &str, kind: ParamKind, default: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            kind,
            required: false,
            default: Some(default.to_string()),
            description: description.to_string(),
        }
    }
}

/// A reusable code template.
#[derive(Debug, Clone)]
pub struct CodeTemplate {
    /// Unique template name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Category for grouping/searching.
    pub category: String,
    /// Parameter slots.
    pub params: Vec<TemplateParam>,
    /// The template generator function.
    generator: TemplateGenerator,
}

/// Template generator — produces code from parameters.
///
/// We use an enum of known generators rather than closures to keep
/// `Clone` and `Debug` derivable.
#[derive(Debug, Clone)]
enum TemplateGenerator {
    ErrorType,
    TraitImpl,
    Builder,
    FromImpl,
    TestModule,
    Iterator,
    NewConstructor,
    /// Dynamically discovered template from library learning (Phase 10h).
    /// Stores a generalized fingerprint string and uses format-string fill.
    Learned { pattern: String },
}

impl CodeTemplate {
    /// Instantiate the template with the given parameters.
    ///
    /// Returns the generated Rust source code.
    pub fn instantiate(&self, params: &HashMap<String, String>) -> GrammarResult<String> {
        // Validate required params
        for p in &self.params {
            if p.required && !params.contains_key(&p.name) {
                return Err(GrammarError::LinearizationFailed {
                    cat: super::cat::Cat::CodeSignature,
                    grammar: "template".to_string(),
                    message: format!(
                        "template '{}' requires parameter '{}' ({})",
                        self.name, p.name, p.description
                    ),
                });
            }
        }

        // Build effective params (with defaults filled in)
        let mut effective = HashMap::new();
        for p in &self.params {
            if let Some(value) = params.get(&p.name) {
                effective.insert(p.name.clone(), value.clone());
            } else if let Some(default) = &p.default {
                effective.insert(p.name.clone(), default.clone());
            }
        }

        match &self.generator {
            TemplateGenerator::ErrorType => generate_error_type(&effective),
            TemplateGenerator::TraitImpl => generate_trait_impl(&effective),
            TemplateGenerator::Builder => generate_builder(&effective),
            TemplateGenerator::FromImpl => generate_from_impl(&effective),
            TemplateGenerator::TestModule => generate_test_module(&effective),
            TemplateGenerator::Iterator => generate_iterator(&effective),
            TemplateGenerator::NewConstructor => generate_new_constructor(&effective),
            TemplateGenerator::Learned { pattern } => generate_learned(pattern, &effective),
        }
    }
}

// ---------------------------------------------------------------------------
// Template registry
// ---------------------------------------------------------------------------

/// Registry of available code templates.
#[derive(Debug, Clone)]
pub struct TemplateRegistry {
    templates: HashMap<String, CodeTemplate>,
}

impl TemplateRegistry {
    /// Create a registry with all built-in templates.
    pub fn new() -> Self {
        let mut templates = HashMap::new();

        for template in builtin_templates() {
            templates.insert(template.name.clone(), template);
        }

        Self { templates }
    }

    /// Get a template by name.
    pub fn get(&self, name: &str) -> Option<&CodeTemplate> {
        self.templates.get(name)
    }

    /// List all template names.
    pub fn list(&self) -> Vec<&str> {
        self.templates.keys().map(|s| s.as_str()).collect()
    }

    /// List templates in a category.
    pub fn by_category(&self, category: &str) -> Vec<&CodeTemplate> {
        self.templates
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Register a custom template.
    pub fn register(&mut self, template: CodeTemplate) {
        self.templates.insert(template.name.clone(), template);
    }

    /// Find templates whose name or description matches keywords.
    pub fn search(&self, keywords: &[&str]) -> Vec<&CodeTemplate> {
        self.templates
            .values()
            .filter(|t| {
                let haystack = format!("{} {} {}", t.name, t.description, t.category).to_lowercase();
                keywords.iter().any(|kw| haystack.contains(&kw.to_lowercase()))
            })
            .collect()
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Built-in templates
// ---------------------------------------------------------------------------

fn builtin_templates() -> Vec<CodeTemplate> {
    vec![
        // Error type (thiserror + miette)
        CodeTemplate {
            name: "error-type".to_string(),
            description: "Error enum with thiserror + miette diagnostics".to_string(),
            category: "error-handling".to_string(),
            params: vec![
                TemplateParam::required("name", ParamKind::TypeName, "Error type name"),
                TemplateParam::required(
                    "variants",
                    ParamKind::VariantList,
                    "Comma-separated variant specs: 'Name(message, code, help)' or 'Name(message)'",
                ),
                TemplateParam::optional(
                    "result_alias",
                    ParamKind::TypeName,
                    "",
                    "Optional result type alias name (e.g., 'MyResult')",
                ),
            ],
            generator: TemplateGenerator::ErrorType,
        },
        // Trait impl
        CodeTemplate {
            name: "trait-impl".to_string(),
            description: "Implement a trait for a type with method stubs".to_string(),
            category: "impl".to_string(),
            params: vec![
                TemplateParam::required("type_name", ParamKind::TypeName, "Type to implement for"),
                TemplateParam::required("trait_name", ParamKind::TraitName, "Trait to implement"),
                TemplateParam::required(
                    "methods",
                    ParamKind::MethodList,
                    "Comma-separated method signatures: 'name(&self) -> Type'",
                ),
            ],
            generator: TemplateGenerator::TraitImpl,
        },
        // Builder pattern
        CodeTemplate {
            name: "builder".to_string(),
            description: "Builder pattern with fluent setters and build()".to_string(),
            category: "pattern".to_string(),
            params: vec![
                TemplateParam::required("type_name", ParamKind::TypeName, "Type to build"),
                TemplateParam::required(
                    "fields",
                    ParamKind::FieldList,
                    "Comma-separated 'name: Type' field specs",
                ),
            ],
            generator: TemplateGenerator::Builder,
        },
        // From impl
        CodeTemplate {
            name: "from-impl".to_string(),
            description: "impl From<Source> for Target conversion".to_string(),
            category: "impl".to_string(),
            params: vec![
                TemplateParam::required("source", ParamKind::TypeName, "Source type"),
                TemplateParam::required("target", ParamKind::TypeName, "Target type"),
                TemplateParam::optional(
                    "body",
                    ParamKind::Text,
                    "todo!()",
                    "Conversion body expression",
                ),
            ],
            generator: TemplateGenerator::FromImpl,
        },
        // Test module
        CodeTemplate {
            name: "test-module".to_string(),
            description: "Test module with #[cfg(test)] and test functions".to_string(),
            category: "testing".to_string(),
            params: vec![
                TemplateParam::required(
                    "tests",
                    ParamKind::MethodList,
                    "Comma-separated test function names",
                ),
                TemplateParam::optional(
                    "imports",
                    ParamKind::Text,
                    "use super::*;",
                    "Import statements for the test module",
                ),
            ],
            generator: TemplateGenerator::TestModule,
        },
        // Iterator impl
        CodeTemplate {
            name: "iterator".to_string(),
            description: "impl Iterator for Type with Item type and next()".to_string(),
            category: "impl".to_string(),
            params: vec![
                TemplateParam::required("type_name", ParamKind::TypeName, "Iterator type"),
                TemplateParam::required("item_type", ParamKind::TypeName, "Iterator::Item type"),
            ],
            generator: TemplateGenerator::Iterator,
        },
        // New constructor
        CodeTemplate {
            name: "new-constructor".to_string(),
            description: "impl Type { pub fn new(...) -> Self }".to_string(),
            category: "impl".to_string(),
            params: vec![
                TemplateParam::required("type_name", ParamKind::TypeName, "Type name"),
                TemplateParam::required(
                    "fields",
                    ParamKind::FieldList,
                    "Comma-separated 'name: Type' constructor params",
                ),
            ],
            generator: TemplateGenerator::NewConstructor,
        },
    ]
}

// ---------------------------------------------------------------------------
// Template generators
// ---------------------------------------------------------------------------

/// Generate a thiserror + miette error type.
///
/// Variant format: "Name(message)" or "Name(message, code, help)"
/// where message/code/help are raw strings (no quotes needed).
fn generate_error_type(params: &HashMap<String, String>) -> GrammarResult<String> {
    let name = params.get("name").unwrap();
    let variants_str = params.get("variants").unwrap();
    let result_alias = params.get("result_alias").map(|s| s.as_str()).unwrap_or("");

    let mut out = String::new();

    // Derives
    out.push_str("#[derive(Debug, thiserror::Error, miette::Diagnostic)]\n");
    out.push_str(&format!("pub enum {name} {{\n"));

    // Parse and generate variants
    for variant_spec in split_comma_respecting_parens(variants_str) {
        let variant_spec = variant_spec.trim();
        if variant_spec.is_empty() {
            continue;
        }

        let (variant_name, variant_args) = parse_variant_spec(variant_spec);

        match variant_args.len() {
            0 => {
                // Simple variant: no message
                out.push_str(&format!("    #[error(\"{variant_name}\")]\n"));
                out.push_str(&format!("    {variant_name},\n\n"));
            }
            1 => {
                // Variant with message only
                let message = &variant_args[0];
                out.push_str(&format!("    #[error(\"{message}\")]\n"));
                out.push_str(&format!("    {variant_name},\n\n"));
            }
            2 => {
                // Variant with message + code
                let message = &variant_args[0];
                let code = &variant_args[1];
                out.push_str(&format!("    #[error(\"{message}\")]\n"));
                out.push_str(&format!("    #[diagnostic(code({code}))]\n"));
                out.push_str(&format!("    {variant_name},\n\n"));
            }
            _ => {
                // Variant with message + code + help
                let message = &variant_args[0];
                let code = &variant_args[1];
                let help = &variant_args[2];
                out.push_str(&format!("    #[error(\"{message}\")]\n"));
                out.push_str(&format!(
                    "    #[diagnostic(code({code}), help(\"{help}\"))]\n"
                ));
                out.push_str(&format!("    {variant_name},\n\n"));
            }
        }
    }

    out.push_str("}\n");

    // Optional result type alias
    if !result_alias.is_empty() {
        out.push('\n');
        out.push_str(&format!(
            "pub type {result_alias}<T> = std::result::Result<T, {name}>;\n"
        ));
    }

    Ok(out)
}

/// Generate a trait implementation with method stubs.
fn generate_trait_impl(params: &HashMap<String, String>) -> GrammarResult<String> {
    let type_name = params.get("type_name").unwrap();
    let trait_name = params.get("trait_name").unwrap();
    let methods_str = params.get("methods").unwrap();

    let mut out = String::new();
    out.push_str(&format!("impl {trait_name} for {type_name} {{\n"));

    for method_sig in split_comma_respecting_parens(methods_str) {
        let method_sig = method_sig.trim();
        if method_sig.is_empty() {
            continue;
        }

        out.push_str(&format!("    fn {method_sig} {{\n"));
        out.push_str("        todo!()\n");
        out.push_str("    }\n\n");
    }

    out.push_str("}\n");
    Ok(out)
}

/// Generate a builder pattern.
fn generate_builder(params: &HashMap<String, String>) -> GrammarResult<String> {
    let type_name = params.get("type_name").unwrap();
    let fields_str = params.get("fields").unwrap();
    let builder_name = format!("{type_name}Builder");

    let fields: Vec<(&str, &str)> = parse_field_list(fields_str);

    let mut out = String::new();

    // Builder struct with Option fields
    out.push_str(&format!("#[derive(Debug, Default)]\n"));
    out.push_str(&format!("pub struct {builder_name} {{\n"));
    for (name, ty) in &fields {
        out.push_str(&format!("    {name}: Option<{ty}>,\n"));
    }
    out.push_str("}\n\n");

    // Builder impl
    out.push_str(&format!("impl {builder_name} {{\n"));
    out.push_str("    pub fn new() -> Self {\n");
    out.push_str("        Self::default()\n");
    out.push_str("    }\n\n");

    // Setter methods
    for (name, ty) in &fields {
        out.push_str(&format!(
            "    pub fn {name}(mut self, {name}: {ty}) -> Self {{\n"
        ));
        out.push_str(&format!("        self.{name} = Some({name});\n"));
        out.push_str("        self\n");
        out.push_str("    }\n\n");
    }

    // Build method
    out.push_str(&format!("    pub fn build(self) -> {type_name} {{\n"));
    out.push_str(&format!("        {type_name} {{\n"));
    for (name, _ty) in &fields {
        out.push_str(&format!(
            "            {name}: self.{name}.expect(\"{name} is required\"),\n"
        ));
    }
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n");

    Ok(out)
}

/// Generate a From impl.
fn generate_from_impl(params: &HashMap<String, String>) -> GrammarResult<String> {
    let source = params.get("source").unwrap();
    let target = params.get("target").unwrap();
    let body = params
        .get("body")
        .map(|s| s.as_str())
        .unwrap_or("todo!()");

    Ok(format!(
        "impl From<{source}> for {target} {{\n    fn from(value: {source}) -> Self {{\n        {body}\n    }}\n}}\n"
    ))
}

/// Generate a test module.
fn generate_test_module(params: &HashMap<String, String>) -> GrammarResult<String> {
    let tests_str = params.get("tests").unwrap();
    let imports = params
        .get("imports")
        .map(|s| s.as_str())
        .unwrap_or("use super::*;");

    let mut out = String::new();
    out.push_str("#[cfg(test)]\nmod tests {\n");
    out.push_str(&format!("    {imports}\n\n"));

    for test_name in tests_str.split(',') {
        let test_name = test_name.trim();
        if test_name.is_empty() {
            continue;
        }
        out.push_str("    #[test]\n");
        out.push_str(&format!("    fn {test_name}() {{\n"));
        out.push_str("        todo!()\n");
        out.push_str("    }\n\n");
    }

    out.push_str("}\n");
    Ok(out)
}

/// Generate an Iterator impl.
fn generate_iterator(params: &HashMap<String, String>) -> GrammarResult<String> {
    let type_name = params.get("type_name").unwrap();
    let item_type = params.get("item_type").unwrap();

    Ok(format!(
        "impl Iterator for {type_name} {{\n    type Item = {item_type};\n\n    fn next(&mut self) -> Option<Self::Item> {{\n        todo!()\n    }}\n}}\n"
    ))
}

/// Generate a new() constructor.
fn generate_new_constructor(params: &HashMap<String, String>) -> GrammarResult<String> {
    let type_name = params.get("type_name").unwrap();
    let fields_str = params.get("fields").unwrap();

    let fields: Vec<(&str, &str)> = parse_field_list(fields_str);

    let mut out = String::new();
    out.push_str(&format!("impl {type_name} {{\n"));

    // Param list
    let param_list: Vec<String> = fields
        .iter()
        .map(|(name, ty)| format!("{name}: {ty}"))
        .collect();

    out.push_str(&format!(
        "    pub fn new({}) -> Self {{\n",
        param_list.join(", ")
    ));
    out.push_str(&format!("        Self {{\n"));
    for (name, _ty) in &fields {
        out.push_str(&format!("            {name},\n"));
    }
    out.push_str("        }\n");
    out.push_str("    }\n");
    out.push_str("}\n");

    Ok(out)
}

/// Generate code from a learned (data-driven) template.
///
/// The pattern string contains a generalized fingerprint. The generator uses
/// it to produce a scaffold comment and a `todo!()` body, which the agent
/// can then refine through the compiler feedback loop.
fn generate_learned(pattern: &str, params: &HashMap<String, String>) -> GrammarResult<String> {
    let type_name = params
        .get("type_name")
        .map(|s| s.as_str())
        .unwrap_or("LearnedType");

    let mut out = String::new();
    out.push_str(&format!("// Learned pattern: {pattern}\n"));

    // Produce a scaffold based on the top-level pattern kind
    if pattern.starts_with("fn(") {
        let fn_name = params
            .get("fn_name")
            .map(|s| s.as_str())
            .unwrap_or("learned_fn");
        out.push_str(&format!("fn {fn_name}() {{\n    todo!()\n}}\n"));
    } else if pattern.starts_with("struct(") {
        out.push_str(&format!("struct {type_name} {{\n    // TODO: fields\n}}\n"));
    } else if pattern.starts_with("enum(") {
        out.push_str(&format!("enum {type_name} {{\n    // TODO: variants\n}}\n"));
    } else if pattern.starts_with("impl(") {
        let trait_name = params
            .get("trait_name")
            .map(|s| s.as_str())
            .unwrap_or("Trait");
        out.push_str(&format!(
            "impl {trait_name} for {type_name} {{\n    // TODO: methods\n}}\n"
        ));
    } else {
        out.push_str(&format!(
            "// Scaffold for learned pattern\nfn {type_name}_scaffold() {{\n    todo!()\n}}\n"
        ));
    }

    Ok(out)
}

impl CodeTemplate {
    /// Create a `CodeTemplate` from a discovered abstraction (Phase 10h).
    ///
    /// Only produces templates for top-level patterns (Function, Struct, Enum, Impl).
    /// Returns `None` for expression-level or too-generic patterns.
    pub fn from_abstraction(
        abs: &crate::reason::anti_unify::DiscoveredAbstraction,
    ) -> Option<Self> {
        use crate::reason::anti_unify::GeneralizedAst;

        // Only create templates for top-level item patterns
        let (category, description, params) = match &abs.pattern {
            GeneralizedAst::Function { .. } => (
                "learned-function",
                format!(
                    "Learned function pattern ({} occurrences, compression {:.1})",
                    abs.occurrences, abs.compression
                ),
                vec![
                    TemplateParam::optional("fn_name", ParamKind::Text, "learned_fn", "Function name"),
                    TemplateParam::optional("type_name", ParamKind::TypeName, "T", "Return type"),
                ],
            ),
            GeneralizedAst::Struct { .. } => (
                "learned-struct",
                format!(
                    "Learned struct pattern ({} occurrences, compression {:.1})",
                    abs.occurrences, abs.compression
                ),
                vec![
                    TemplateParam::optional("type_name", ParamKind::TypeName, "LearnedStruct", "Struct name"),
                ],
            ),
            GeneralizedAst::Enum { .. } => (
                "learned-enum",
                format!(
                    "Learned enum pattern ({} occurrences, compression {:.1})",
                    abs.occurrences, abs.compression
                ),
                vec![
                    TemplateParam::optional("type_name", ParamKind::TypeName, "LearnedEnum", "Enum name"),
                ],
            ),
            GeneralizedAst::Impl { .. } => (
                "learned-impl",
                format!(
                    "Learned impl pattern ({} occurrences, compression {:.1})",
                    abs.occurrences, abs.compression
                ),
                vec![
                    TemplateParam::optional("type_name", ParamKind::TypeName, "LearnedType", "Type name"),
                    TemplateParam::optional("trait_name", ParamKind::TraitName, "Trait", "Trait name"),
                ],
            ),
            // Expression-level or Var/Concrete — not suitable for templates
            _ => return None,
        };

        let name = format!("learned:{}", abs.fingerprint);

        Some(CodeTemplate {
            name,
            description,
            category: category.to_string(),
            params,
            generator: TemplateGenerator::Learned {
                pattern: abs.fingerprint.clone(),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

/// Parse "Name(arg1, arg2, arg3)" into ("Name", ["arg1", "arg2", "arg3"]).
fn parse_variant_spec(spec: &str) -> (String, Vec<String>) {
    if let Some(paren_start) = spec.find('(') {
        let name = spec[..paren_start].trim().to_string();
        let args_str = &spec[paren_start + 1..];
        let args_str = args_str.trim_end_matches(')').trim();
        if args_str.is_empty() {
            (name, vec![])
        } else {
            let args: Vec<String> = args_str.split(',').map(|s| s.trim().to_string()).collect();
            (name, args)
        }
    } else {
        (spec.trim().to_string(), vec![])
    }
}

/// Parse "name: Type, name2: Type2" into [("name", "Type"), ("name2", "Type2")].
fn parse_field_list(fields_str: &str) -> Vec<(&str, &str)> {
    fields_str
        .split(',')
        .filter_map(|f| {
            let f = f.trim();
            if f.is_empty() {
                return None;
            }
            let parts: Vec<&str> = f.splitn(2, ':').collect();
            if parts.len() == 2 {
                Some((parts[0].trim(), parts[1].trim()))
            } else {
                Some((f, "()"))
            }
        })
        .collect()
}

/// Split by commas, but respect parentheses and angle-bracket nesting.
fn split_comma_respecting_parens(s: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut depth = 0;

    for ch in s.chars() {
        match ch {
            '(' | '<' => {
                depth += 1;
                current.push(ch);
            }
            ')' | '>' => {
                if depth > 0 {
                    depth -= 1;
                }
                current.push(ch);
            }
            ',' if depth == 0 => {
                result.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    if !current.is_empty() {
        result.push(current);
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_builtins() {
        let reg = TemplateRegistry::new();
        let names = reg.list();
        assert!(names.contains(&"error-type"));
        assert!(names.contains(&"trait-impl"));
        assert!(names.contains(&"builder"));
        assert!(names.contains(&"from-impl"));
        assert!(names.contains(&"test-module"));
        assert!(names.contains(&"iterator"));
        assert!(names.contains(&"new-constructor"));
    }

    #[test]
    fn error_type_with_diagnostics() {
        let reg = TemplateRegistry::new();
        let template = reg.get("error-type").unwrap();

        let mut params = HashMap::new();
        params.insert("name".into(), "MyError".into());
        params.insert(
            "variants".into(),
            "NotFound(item not found, my::not_found, check the ID), InvalidInput(invalid input)".into(),
        );
        params.insert("result_alias".into(), "MyResult".into());

        let code = template.instantiate(&params).unwrap();
        assert!(code.contains("#[derive(Debug, thiserror::Error, miette::Diagnostic)]"));
        assert!(code.contains("pub enum MyError {"));
        assert!(code.contains("#[error(\"item not found\")]"));
        assert!(code.contains("#[diagnostic(code(my::not_found), help(\"check the ID\"))]"));
        assert!(code.contains("NotFound,"));
        assert!(code.contains("#[error(\"invalid input\")]"));
        assert!(code.contains("InvalidInput,"));
        assert!(code.contains("pub type MyResult<T> = std::result::Result<T, MyError>;"));
    }

    #[test]
    fn error_type_simple_variants() {
        let reg = TemplateRegistry::new();
        let template = reg.get("error-type").unwrap();

        let mut params = HashMap::new();
        params.insert("name".into(), "ParseError".into());
        params.insert("variants".into(), "UnexpectedToken(unexpected token), Eof".into());

        let code = template.instantiate(&params).unwrap();
        assert!(code.contains("pub enum ParseError {"));
        assert!(code.contains("UnexpectedToken,"));
        assert!(code.contains("Eof,"));
    }

    #[test]
    fn trait_impl_generates_stubs() {
        let reg = TemplateRegistry::new();
        let template = reg.get("trait-impl").unwrap();

        let mut params = HashMap::new();
        params.insert("type_name".into(), "Config".into());
        params.insert("trait_name".into(), "Display".into());
        params.insert("methods".into(), "fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result".into());

        let code = template.instantiate(&params).unwrap();
        assert!(code.contains("impl Display for Config {"));
        assert!(code.contains("fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {"));
        assert!(code.contains("todo!()"));
    }

    #[test]
    fn builder_pattern() {
        let reg = TemplateRegistry::new();
        let template = reg.get("builder").unwrap();

        let mut params = HashMap::new();
        params.insert("type_name".into(), "Config".into());
        params.insert("fields".into(), "name: String, value: u64, debug: bool".into());

        let code = template.instantiate(&params).unwrap();
        assert!(code.contains("pub struct ConfigBuilder {"));
        assert!(code.contains("name: Option<String>,"));
        assert!(code.contains("value: Option<u64>,"));
        assert!(code.contains("pub fn name(mut self, name: String) -> Self {"));
        assert!(code.contains("pub fn build(self) -> Config {"));
        assert!(code.contains("name: self.name.expect(\"name is required\"),"));
    }

    #[test]
    fn from_impl() {
        let reg = TemplateRegistry::new();
        let template = reg.get("from-impl").unwrap();

        let mut params = HashMap::new();
        params.insert("source".into(), "String".into());
        params.insert("target".into(), "MyType".into());
        params.insert("body".into(), "Self(value)".into());

        let code = template.instantiate(&params).unwrap();
        assert!(code.contains("impl From<String> for MyType {"));
        assert!(code.contains("fn from(value: String) -> Self {"));
        assert!(code.contains("Self(value)"));
    }

    #[test]
    fn test_module_template() {
        let reg = TemplateRegistry::new();
        let template = reg.get("test-module").unwrap();

        let mut params = HashMap::new();
        params.insert("tests".into(), "it_works, edge_case, error_handling".into());

        let code = template.instantiate(&params).unwrap();
        assert!(code.contains("#[cfg(test)]"));
        assert!(code.contains("mod tests {"));
        assert!(code.contains("use super::*;"));
        assert!(code.contains("#[test]"));
        assert!(code.contains("fn it_works()"));
        assert!(code.contains("fn edge_case()"));
        assert!(code.contains("fn error_handling()"));
    }

    #[test]
    fn iterator_impl() {
        let reg = TemplateRegistry::new();
        let template = reg.get("iterator").unwrap();

        let mut params = HashMap::new();
        params.insert("type_name".into(), "TokenStream".into());
        params.insert("item_type".into(), "Token".into());

        let code = template.instantiate(&params).unwrap();
        assert!(code.contains("impl Iterator for TokenStream {"));
        assert!(code.contains("type Item = Token;"));
        assert!(code.contains("fn next(&mut self) -> Option<Self::Item>"));
    }

    #[test]
    fn new_constructor() {
        let reg = TemplateRegistry::new();
        let template = reg.get("new-constructor").unwrap();

        let mut params = HashMap::new();
        params.insert("type_name".into(), "Config".into());
        params.insert("fields".into(), "name: String, value: u64".into());

        let code = template.instantiate(&params).unwrap();
        assert!(code.contains("impl Config {"));
        assert!(code.contains("pub fn new(name: String, value: u64) -> Self {"));
        assert!(code.contains("name,"));
        assert!(code.contains("value,"));
    }

    #[test]
    fn missing_required_param_errors() {
        let reg = TemplateRegistry::new();
        let template = reg.get("error-type").unwrap();

        let params = HashMap::new(); // missing "name" and "variants"
        let result = template.instantiate(&params);
        assert!(result.is_err());
    }

    #[test]
    fn search_templates() {
        let reg = TemplateRegistry::new();

        let error_templates = reg.search(&["error"]);
        assert!(!error_templates.is_empty());
        assert!(error_templates.iter().any(|t| t.name == "error-type"));

        let impl_templates = reg.by_category("impl");
        assert!(impl_templates.len() >= 3);
    }

    #[test]
    fn parse_variant_spec_with_args() {
        let (name, args) = parse_variant_spec("NotFound(item not found, my::code, try again)");
        assert_eq!(name, "NotFound");
        assert_eq!(args.len(), 3);
        assert_eq!(args[0], "item not found");
        assert_eq!(args[1], "my::code");
        assert_eq!(args[2], "try again");
    }

    #[test]
    fn parse_variant_spec_no_args() {
        let (name, args) = parse_variant_spec("Unknown");
        assert_eq!(name, "Unknown");
        assert!(args.is_empty());
    }

    #[test]
    fn split_comma_with_parens() {
        let result = split_comma_respecting_parens("A(1,2), B(3,4), C");
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].trim(), "A(1,2)");
        assert_eq!(result[1].trim(), "B(3,4)");
        assert_eq!(result[2].trim(), "C");
    }
}
