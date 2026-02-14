//! Custom grammar: user-defined archetypes loaded from TOML specification.
//!
//! Users can define their own grammar archetypes via TOML files with
//! template-based linearization rules and regex-based parse patterns.
//!
//! ## Example TOML
//!
//! ```toml
//! [grammar]
//! name = "mythic"
//! description = "Knowledge as mythology"
//!
//! [linearization]
//! triple = "It is written that {subject} {predicate} {object}."
//! similarity = "{entity} mirrors {similar_to} in the great tapestry (score: {score})."
//! gap = "The scrolls are silent on {entity}: {description}."
//! inference = "The oracle reveals: {expression} becomes {simplified}."
//! code_fact = "In the codex, {kind} '{name}' is recorded: {detail}."
//! freeform = "{text}"
//!
//! [parse.patterns]
//! triple = ["It is written that {subject} {predicate} {object}"]
//! ```

use std::collections::HashMap;

use super::abs::AbsTree;
use super::cat::Cat;
use super::concrete::{ConcreteGrammar, LinContext, ParseContext};
use super::error::{GrammarError, GrammarResult};
use super::morpho;
use super::parser;

/// A custom grammar defined by TOML templates.
pub struct CustomGrammar {
    name: String,
    description: String,
    /// Linearization templates keyed by category name.
    templates: HashMap<String, String>,
}

impl CustomGrammar {
    /// Parse a custom grammar from a TOML string.
    pub fn from_toml(toml_str: &str) -> GrammarResult<Self> {
        // Minimal TOML parsing without external dependency —
        // parse the specific sections we need.
        let mut name = String::new();
        let mut description = String::new();
        let mut templates = HashMap::new();
        let mut current_section = String::new();

        for line in toml_str.lines() {
            let trimmed = line.trim();

            // Skip comments and empty lines
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Section headers
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                current_section = trimmed[1..trimmed.len() - 1].trim().to_string();
                continue;
            }

            // Key-value pairs
            if let Some(eq_pos) = trimmed.find('=') {
                let key = trimmed[..eq_pos].trim();
                let value = trimmed[eq_pos + 1..].trim();
                let value = strip_toml_quotes(value);

                match current_section.as_str() {
                    "grammar" => match key {
                        "name" => name = value.to_string(),
                        "description" => description = value.to_string(),
                        _ => {}
                    },
                    "linearization" => {
                        templates.insert(key.to_string(), value.to_string());
                    }
                    _ => {}
                }
            }
        }

        if name.is_empty() {
            return Err(GrammarError::InvalidCustomGrammar {
                message: "missing [grammar] name field".to_string(),
            });
        }

        Ok(Self {
            name,
            description,
            templates,
        })
    }

    /// Apply a template string with variable substitutions.
    fn apply_template(&self, template: &str, vars: &HashMap<&str, String>) -> String {
        let mut result = template.to_string();
        for (key, value) in vars {
            result = result.replace(&format!("{{{key}}}"), value);
        }
        result
    }

    fn linearize_inner(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String> {
        match tree {
            AbsTree::EntityRef { label, symbol_id } => Ok(ctx.resolve_label(label, *symbol_id)),

            AbsTree::RelationRef { label, symbol_id } => {
                let resolved = ctx.resolve_label(label, *symbol_id);
                Ok(morpho::humanize_predicate(&resolved))
            }

            AbsTree::Freeform(text) => {
                if let Some(template) = self.templates.get("freeform") {
                    let mut vars = HashMap::new();
                    vars.insert("text", text.clone());
                    Ok(self.apply_template(template, &vars))
                } else {
                    Ok(text.clone())
                }
            }

            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => {
                let s = self.linearize_inner(subject, ctx)?;
                let p = self.linearize_inner(predicate, ctx)?;
                let o = self.linearize_inner(object, ctx)?;

                if let Some(template) = self.templates.get("triple") {
                    let mut vars = HashMap::new();
                    vars.insert("subject", s);
                    vars.insert("predicate", p);
                    vars.insert("object", o);
                    Ok(self.apply_template(template, &vars))
                } else {
                    // Default fallback
                    Ok(format!("{s} {p} {o}."))
                }
            }

            AbsTree::Similarity {
                entity,
                similar_to,
                score,
            } => {
                let e = self.linearize_inner(entity, ctx)?;
                let s = self.linearize_inner(similar_to, ctx)?;

                if let Some(template) = self.templates.get("similarity") {
                    let mut vars = HashMap::new();
                    vars.insert("entity", e);
                    vars.insert("similar_to", s);
                    vars.insert("score", format!("{score:.2}"));
                    Ok(self.apply_template(template, &vars))
                } else {
                    Ok(format!("{e} is similar to {s} ({score:.2})."))
                }
            }

            AbsTree::Gap {
                entity,
                description,
            } => {
                let e = self.linearize_inner(entity, ctx)?;

                if let Some(template) = self.templates.get("gap") {
                    let mut vars = HashMap::new();
                    vars.insert("entity", e);
                    vars.insert("description", description.clone());
                    Ok(self.apply_template(template, &vars))
                } else {
                    Ok(format!("Gap for {e}: {description}."))
                }
            }

            AbsTree::Inference {
                expression,
                simplified,
            } => {
                if let Some(template) = self.templates.get("inference") {
                    let mut vars = HashMap::new();
                    vars.insert("expression", expression.clone());
                    vars.insert("simplified", simplified.clone());
                    Ok(self.apply_template(template, &vars))
                } else {
                    Ok(format!("`{expression}` simplifies to `{simplified}`."))
                }
            }

            AbsTree::CodeFact { kind, name, detail } => {
                if let Some(template) = self.templates.get("code_fact") {
                    let mut vars = HashMap::new();
                    vars.insert("kind", kind.clone());
                    vars.insert("name", name.clone());
                    vars.insert("detail", detail.clone());
                    Ok(self.apply_template(template, &vars))
                } else {
                    Ok(format!("{kind} `{name}`: {detail}."))
                }
            }

            AbsTree::CodeModule {
                name,
                role,
                importance,
                doc_summary,
                children,
            } => {
                let children_text: Vec<String> = children
                    .iter()
                    .map(|c| self.linearize_inner(c, ctx))
                    .collect::<GrammarResult<Vec<_>>>()?;
                if let Some(template) = self.templates.get("code_module") {
                    let mut vars = HashMap::new();
                    vars.insert("name", name.clone());
                    vars.insert("role", role.clone().unwrap_or_default());
                    vars.insert(
                        "importance",
                        importance.map(|i| format!("{i:.2}")).unwrap_or_default(),
                    );
                    vars.insert("doc_summary", doc_summary.clone().unwrap_or_default());
                    vars.insert("children", children_text.join("\n"));
                    Ok(self.apply_template(template, &vars))
                } else {
                    let desc = doc_summary
                        .as_deref()
                        .or(role.as_deref())
                        .unwrap_or("module");
                    let mut out = format!("Module `{name}` ({desc}).");
                    if !children_text.is_empty() {
                        out.push('\n');
                        for line in &children_text {
                            out.push_str("- ");
                            out.push_str(line);
                            out.push('\n');
                        }
                    }
                    Ok(out)
                }
            }

            AbsTree::CodeSignature {
                kind,
                name,
                params_or_fields,
                return_type,
                traits,
                ..
            } => {
                if let Some(template) = self.templates.get("code_signature") {
                    let mut vars = HashMap::new();
                    vars.insert("kind", kind.clone());
                    vars.insert("name", name.clone());
                    vars.insert("params", params_or_fields.join(", "));
                    vars.insert("return_type", return_type.clone().unwrap_or_default());
                    vars.insert("traits", traits.join(", "));
                    Ok(self.apply_template(template, &vars))
                } else {
                    let params = params_or_fields.join(", ");
                    let ret = return_type
                        .as_ref()
                        .map(|r| format!(" \u{2192} {r}"))
                        .unwrap_or_default();
                    Ok(format!("{kind} `{name}`({params}){ret}."))
                }
            }

            AbsTree::DataFlow { steps } => {
                if let Some(template) = self.templates.get("data_flow") {
                    let flow: Vec<String> = steps
                        .iter()
                        .map(|s| match &s.via_type {
                            Some(t) => format!("{} \u{2192} {t}", s.name),
                            None => s.name.clone(),
                        })
                        .collect();
                    let mut vars = HashMap::new();
                    vars.insert("flow", flow.join(" \u{2192} "));
                    Ok(self.apply_template(template, &vars))
                } else {
                    let flow: Vec<String> = steps
                        .iter()
                        .map(|s| match &s.via_type {
                            Some(t) => format!("{} \u{2192} {t}", s.name),
                            None => s.name.clone(),
                        })
                        .collect();
                    Ok(format!("Flow: {}", flow.join(" \u{2192} ")))
                }
            }

            AbsTree::WithConfidence { inner, confidence } => {
                let text = self.linearize_inner(inner, ctx)?;
                Ok(format!("{text} (confidence: {confidence:.2})"))
            }

            AbsTree::WithProvenance { inner, tag } => {
                let text = self.linearize_inner(inner, ctx)?;
                Ok(format!("{text} [{tag}]"))
            }

            AbsTree::Conjunction { items, is_and } => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|i| self.linearize_inner(i, ctx))
                    .collect::<GrammarResult<Vec<_>>>()?;
                let conj = if *is_and { "and" } else { "or" };
                Ok(morpho::join_list(&parts, conj))
            }

            AbsTree::Section { heading, body } => {
                let mut out = format!("## {heading}\n\n");
                for item in body {
                    let line = self.linearize_inner(item, ctx)?;
                    out.push_str(&line);
                    if !line.ends_with('\n') {
                        out.push('\n');
                    }
                }
                Ok(out)
            }

            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => {
                let mut out = self.linearize_inner(overview, ctx)?;
                out.push_str("\n\n");
                for section in sections {
                    out.push_str(&self.linearize_inner(section, ctx)?);
                    out.push('\n');
                }
                if !gaps.is_empty() {
                    out.push_str("\n## Gaps\n\n");
                    for gap in gaps {
                        out.push_str("- ");
                        out.push_str(&self.linearize_inner(gap, ctx)?);
                        out.push('\n');
                    }
                }
                Ok(out)
            }
        }
    }
}

/// Strip TOML string quotes.
fn strip_toml_quotes(s: &str) -> &str {
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

impl ConcreteGrammar for CustomGrammar {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn linearize(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String> {
        self.linearize_inner(tree, ctx)
    }

    fn parse(
        &self,
        input: &str,
        _expected_cat: Option<Cat>,
        ctx: &ParseContext,
    ) -> GrammarResult<AbsTree> {
        // Custom grammars use the universal parser — custom parse patterns
        // can be added in a future iteration.
        parser::parse_universal(input, ctx)
    }

    fn supported_categories(&self) -> &[Cat] {
        &[
            Cat::Entity,
            Cat::Relation,
            Cat::Statement,
            Cat::Similarity,
            Cat::Gap,
            Cat::Inference,
            Cat::CodeFact,
            Cat::CodeModule,
            Cat::CodeSignature,
            Cat::DataFlow,
            Cat::Confidence,
            Cat::Provenance,
            Cat::Conjunction,
            Cat::Section,
            Cat::Document,
            Cat::Freeform,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MYTHIC_TOML: &str = r#"
[grammar]
name = "mythic"
description = "Knowledge as mythology"

[linearization]
triple = "It is written that {subject} {predicate} {object}."
similarity = "{entity} mirrors {similar_to} in the great tapestry (score: {score})."
gap = "The scrolls are silent on {entity}: {description}."
inference = "The oracle reveals: {expression} becomes {simplified}."
code_fact = "In the codex, {kind} '{name}' is recorded: {detail}."
"#;

    #[test]
    fn parse_custom_grammar_toml() {
        let grammar = CustomGrammar::from_toml(MYTHIC_TOML).unwrap();
        assert_eq!(grammar.name(), "mythic");
        assert_eq!(grammar.description(), "Knowledge as mythology");
    }

    #[test]
    fn linearize_custom_triple() {
        let grammar = CustomGrammar::from_toml(MYTHIC_TOML).unwrap();
        let ctx = LinContext::default();
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let result = grammar.linearize(&tree, &ctx).unwrap();
        assert_eq!(result, "It is written that Dog is a Mammal.");
    }

    #[test]
    fn linearize_custom_gap() {
        let grammar = CustomGrammar::from_toml(MYTHIC_TOML).unwrap();
        let ctx = LinContext::default();
        let tree = AbsTree::gap(AbsTree::entity("Dog"), "no habitat data");
        let result = grammar.linearize(&tree, &ctx).unwrap();
        assert_eq!(result, "The scrolls are silent on Dog: no habitat data.");
    }

    #[test]
    fn linearize_custom_similarity() {
        let grammar = CustomGrammar::from_toml(MYTHIC_TOML).unwrap();
        let ctx = LinContext::default();
        let tree = AbsTree::similarity(AbsTree::entity("Dog"), AbsTree::entity("Wolf"), 0.87);
        let result = grammar.linearize(&tree, &ctx).unwrap();
        assert!(result.contains("mirrors"));
        assert!(result.contains("great tapestry"));
    }

    #[test]
    fn reject_missing_name() {
        let toml = r#"
[grammar]
description = "No name"
[linearization]
triple = "{subject} {predicate} {object}"
"#;
        let result = CustomGrammar::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn fallback_without_template() {
        let toml = r#"
[grammar]
name = "minimal"
description = "Minimal grammar"
"#;
        let grammar = CustomGrammar::from_toml(toml).unwrap();
        let ctx = LinContext::default();
        let tree = AbsTree::triple(
            AbsTree::entity("A"),
            AbsTree::relation("r"),
            AbsTree::entity("B"),
        );
        let result = grammar.linearize(&tree, &ctx).unwrap();
        // Falls back to default format
        assert!(result.contains("A"));
        assert!(result.contains("B"));
    }
}
