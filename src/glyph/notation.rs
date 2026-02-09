//! Notation formatter: convert KG triples and subgraphs into hieroglyphic strings.
//!
//! Supports compact (single-line) and block (subject-grouped) formats,
//! with optional confidence dots, provenance markers, and VSA sigils.

use std::collections::BTreeMap;

use crate::engine::Engine;
use crate::glyph::catalog;
use crate::glyph::sigil;
use crate::graph::Triple;
use crate::symbol::SymbolId;

/// Configuration for hieroglyphic notation rendering.
#[derive(Debug, Clone)]
pub struct NotationConfig {
    /// Use PUA codepoints (requires custom font) vs Unicode fallback.
    pub use_pua: bool,
    /// Show confidence as filled/empty dots after each triple.
    pub show_confidence: bool,
    /// Show provenance markers (asserted/inferred/fused/etc.).
    pub show_provenance: bool,
    /// Show VSA sigils before entity labels.
    pub show_sigils: bool,
    /// Compact (single-line per triple) vs block (grouped by subject) format.
    pub compact: bool,
}

impl Default for NotationConfig {
    fn default() -> Self {
        Self {
            use_pua: catalog::font_available(),
            show_confidence: true,
            show_provenance: false,
            show_sigils: true,
            compact: false,
        }
    }
}

/// Render a single triple as hieroglyphic notation.
///
/// Format: `⟨ [sigil]Subject [predicate-glyph] [sigil]Object ⟩ [confidence]`
pub fn render_triple(engine: &Engine, triple: &Triple, config: &NotationConfig) -> String {
    let open = structural_glyph("struct:triple", config.use_pua);
    let close = structural_glyph("struct:end", config.use_pua);
    let subj = render_entity(engine, triple.subject, config);
    let pred = render_predicate(engine, triple.predicate, config);
    let obj = render_entity(engine, triple.object, config);

    let mut line = format!("{open} {subj} {pred} {obj} {close}");

    if config.show_confidence {
        line.push(' ');
        line.push_str(&catalog::confidence_dots(triple.confidence));
    }

    line
}

/// Render a subgraph as hieroglyphic notation.
///
/// In compact mode, each triple gets its own line.
/// In block mode, triples are grouped by subject:
/// ```text
/// [sigil]Subject {
///   [pred-glyph] [sigil]Object
///   [pred-glyph] [sigil]Object
/// }
/// ```
pub fn render_subgraph(engine: &Engine, triples: &[Triple], config: &NotationConfig) -> String {
    if triples.is_empty() {
        return String::new();
    }

    if config.compact {
        return triples
            .iter()
            .map(|t| render_triple(engine, t, config))
            .collect::<Vec<_>>()
            .join("\n");
    }

    // Block mode: group by subject using BTreeMap for deterministic order.
    let mut groups: BTreeMap<u64, Vec<&Triple>> = BTreeMap::new();
    for t in triples {
        groups.entry(t.subject.get()).or_default().push(t);
    }

    let mut output = Vec::new();

    for (_, group_triples) in &groups {
        let subject = group_triples[0].subject;
        let subj_str = render_entity(engine, subject, config);

        if group_triples.len() == 1 {
            // Single triple: render inline.
            output.push(render_triple(engine, group_triples[0], config));
        } else {
            // Multiple triples: block format.
            output.push(format!("{subj_str} {{"));
            for t in group_triples {
                let pred = render_predicate(engine, t.predicate, config);
                let obj = render_entity(engine, t.object, config);
                let mut line = format!("  {pred} {obj}");
                if config.show_confidence {
                    line.push(' ');
                    line.push_str(&catalog::confidence_dots(t.confidence));
                }
                output.push(line);
            }
            output.push("}".to_string());
        }
    }

    output.join("\n")
}

/// Render an inference chain as hieroglyphic notation.
///
/// Format: `Subject ─[pred]─ Intermediate ─[pred]─ Final`
pub fn render_chain(
    engine: &Engine,
    chain: &[(SymbolId, SymbolId, SymbolId)],
    config: &NotationConfig,
) -> String {
    if chain.is_empty() {
        return String::new();
    }

    let chain_glyph = structural_glyph("struct:chain", config.use_pua);
    let mut parts = Vec::new();

    for (i, (s, p, o)) in chain.iter().enumerate() {
        if i == 0 {
            parts.push(render_entity(engine, *s, config));
        }
        let pred = render_predicate(engine, *p, config);
        parts.push(format!(" {chain_glyph}{pred}{chain_glyph} "));
        parts.push(render_entity(engine, *o, config));
    }

    parts.concat()
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Render an entity: optional sigil + label.
fn render_entity(engine: &Engine, symbol: SymbolId, config: &NotationConfig) -> String {
    let label = engine.resolve_label(symbol);
    if config.show_sigils {
        let sigil_str = sigil::sigil_for_symbol(symbol, engine.item_memory(), config.use_pua)
            .unwrap_or_default();
        format!("{sigil_str}{label}")
    } else {
        label
    }
}

/// Render a predicate: use catalog glyph if available, otherwise label.
fn render_predicate(engine: &Engine, predicate: SymbolId, config: &NotationConfig) -> String {
    let label = engine.resolve_label(predicate);
    if let Some(glyph) = catalog::lookup(&label) {
        if config.use_pua {
            glyph.pua_codepoint.to_string()
        } else {
            glyph.fallback.to_string()
        }
    } else {
        // No known glyph — show the label as-is.
        label
    }
}

/// Get a structural glyph's display string.
fn structural_glyph(name: &str, use_pua: bool) -> String {
    if let Some(glyph) = catalog::lookup(name) {
        if use_pua {
            glyph.pua_codepoint.to_string()
        } else {
            glyph.fallback.to_string()
        }
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::graph::Triple;
    use crate::symbol::SymbolKind;

    fn setup() -> (Engine, SymbolId, SymbolId, SymbolId) {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let dog = engine.create_symbol(SymbolKind::Entity, "Dog").unwrap();
        let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
        let animal = engine.create_symbol(SymbolKind::Entity, "Animal").unwrap();
        engine
            .add_triple(&Triple::new(dog.id, is_a.id, animal.id))
            .unwrap();
        (engine, dog.id, is_a.id, animal.id)
    }

    #[test]
    fn render_single_triple() {
        let (engine, dog, is_a, animal) = setup();
        let triple = Triple::new(dog, is_a, animal);
        let config = NotationConfig {
            show_sigils: false,
            show_confidence: false,
            ..Default::default()
        };
        let rendered = render_triple(&engine, &triple, &config);
        assert!(rendered.contains("Dog"));
        assert!(rendered.contains("Animal"));
        // Should contain the is-a glyph fallback (△).
        assert!(rendered.contains('\u{25B3}') || rendered.contains('\u{E000}'));
    }

    #[test]
    fn render_triple_with_confidence() {
        let (engine, dog, is_a, animal) = setup();
        let triple = Triple::new(dog, is_a, animal).with_confidence(0.8);
        let config = NotationConfig {
            show_sigils: false,
            show_confidence: true,
            ..Default::default()
        };
        let rendered = render_triple(&engine, &triple, &config);
        // Should contain filled dots.
        assert!(rendered.contains('\u{25CF}'));
    }

    #[test]
    fn subgraph_block_format() {
        let (engine, dog, is_a, animal) = setup();
        let has_a = engine
            .create_symbol(SymbolKind::Relation, "has-a")
            .unwrap();
        let legs = engine.create_symbol(SymbolKind::Entity, "Legs").unwrap();
        engine
            .add_triple(&Triple::new(dog, has_a.id, legs.id))
            .unwrap();

        let triples = vec![
            Triple::new(dog, is_a, animal),
            Triple::new(dog, has_a.id, legs.id),
        ];

        let config = NotationConfig {
            show_sigils: false,
            show_confidence: false,
            compact: false,
            ..Default::default()
        };
        let rendered = render_subgraph(&engine, &triples, &config);
        // Block format: subject + { ... }.
        assert!(rendered.contains('{'));
        assert!(rendered.contains('}'));
        assert!(rendered.contains("Animal"));
        assert!(rendered.contains("Legs"));
    }

    #[test]
    fn subgraph_compact_format() {
        let (engine, dog, is_a, animal) = setup();
        let triples = vec![Triple::new(dog, is_a, animal)];
        let config = NotationConfig {
            show_sigils: false,
            show_confidence: false,
            compact: true,
            ..Default::default()
        };
        let rendered = render_subgraph(&engine, &triples, &config);
        // Compact: should be a single line with ⟨ ⟩ delimiters.
        assert!(rendered.contains('\u{27E8}') || rendered.contains('\u{E01E}'));
    }

    #[test]
    fn chain_rendering() {
        let (engine, dog, is_a, animal) = setup();
        let mammal = engine
            .create_symbol(SymbolKind::Entity, "Mammal")
            .unwrap();
        let chain = vec![(dog, is_a, animal), (animal, is_a, mammal.id)];
        let config = NotationConfig {
            show_sigils: false,
            ..Default::default()
        };
        let rendered = render_chain(&engine, &chain, &config);
        assert!(rendered.contains("Dog"));
        assert!(rendered.contains("Animal"));
        assert!(rendered.contains("Mammal"));
        // Should contain the chain glyph (─).
        assert!(rendered.contains('\u{2500}') || rendered.contains('\u{E020}'));
    }

    #[test]
    fn empty_subgraph() {
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let config = NotationConfig::default();
        let rendered = render_subgraph(&engine, &[], &config);
        assert!(rendered.is_empty());
    }
}
