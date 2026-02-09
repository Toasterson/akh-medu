//! ANSI terminal renderer for hieroglyphic notation.
//!
//! Provides color-coded output for terminal display, with automatic
//! TTY detection and configurable color/style options.

use crate::engine::Engine;
use crate::glyph::catalog::{self, GlyphCategory};
use crate::glyph::notation::{self, NotationConfig};
use crate::glyph::sigil;
use crate::graph::Triple;
use crate::symbol::SymbolId;

// ANSI escape codes.
const RESET: &str = "\x1b[0m";
const CYAN: &str = "\x1b[36m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";
const GREEN: &str = "\x1b[32m";
const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";


/// Configuration for terminal rendering.
#[derive(Debug, Clone)]
pub struct RenderConfig {
    /// Enable ANSI color codes.
    pub color: bool,
    /// Notation configuration.
    pub notation: NotationConfig,
    /// Maximum display width for wrapping.
    pub max_width: usize,
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            color: atty_detect(),
            notation: NotationConfig::default(),
            max_width: 80,
        }
    }
}

/// Render triples with ANSI colors for terminal display.
pub fn render_to_terminal(engine: &Engine, triples: &[Triple], config: &RenderConfig) -> String {
    if !config.color {
        return notation::render_subgraph(engine, triples, &config.notation);
    }

    if triples.is_empty() {
        return String::new();
    }

    if config.notation.compact {
        return triples
            .iter()
            .map(|t| render_triple_colored(engine, t, config))
            .collect::<Vec<_>>()
            .join("\n");
    }

    // Block mode: group by subject.
    let mut groups: std::collections::BTreeMap<u64, Vec<&Triple>> =
        std::collections::BTreeMap::new();
    for t in triples {
        groups.entry(t.subject.get()).or_default().push(t);
    }

    let mut output = Vec::new();

    for (_, group_triples) in &groups {
        let subject = group_triples[0].subject;

        if group_triples.len() == 1 {
            output.push(render_triple_colored(engine, group_triples[0], config));
        } else {
            let subj_str = render_entity_colored(engine, subject, config);
            output.push(format!("{subj_str} {{"));
            for t in group_triples {
                let pred = render_predicate_colored(engine, t.predicate, config);
                let obj = render_entity_colored(engine, t.object, config);
                let mut line = format!("  {pred} {obj}");
                if config.notation.show_confidence {
                    line.push(' ');
                    line.push_str(&format!(
                        "{BOLD}{}{RESET}",
                        catalog::confidence_dots(t.confidence),
                    ));
                }
                output.push(line);
            }
            output.push("}".to_string());
        }
    }

    output.join("\n")
}

/// Render a single symbol with its sigil, label, and optional type determinative.
pub fn render_symbol(engine: &Engine, symbol: SymbolId, config: &RenderConfig) -> String {
    if config.color {
        render_entity_colored(engine, symbol, config)
    } else {
        let label = engine.resolve_label(symbol);
        if config.notation.show_sigils {
            let s = sigil::sigil_for_symbol(symbol, engine.item_memory(), config.notation.use_pua)
                .unwrap_or_default();
            format!("{s}{label}")
        } else {
            label
        }
    }
}

/// Print a legend of all available glyphs and radicals.
pub fn render_legend(config: &RenderConfig) -> String {
    let mut lines = Vec::new();

    lines.push(format!(
        "{}Hieroglyphic Glyph Legend{}",
        if config.color { BOLD } else { "" },
        if config.color { RESET } else { "" },
    ));
    lines.push(String::new());

    // Fixed glyphs by category.
    let categories = [
        (GlyphCategory::Predicate, "Predicates"),
        (GlyphCategory::TypeDeterminative, "Type Determinatives"),
        (GlyphCategory::Provenance, "Provenance Markers"),
        (GlyphCategory::Structural, "Structural"),
    ];

    for (cat, cat_name) in &categories {
        lines.push(format!(
            "  {}{cat_name}{}:",
            if config.color { BOLD } else { "" },
            if config.color { RESET } else { "" },
        ));
        for glyph in catalog::all_glyphs()
            .iter()
            .filter(|g| g.category == *cat)
        {
            let display = if config.notation.use_pua {
                glyph.pua_codepoint.to_string()
            } else {
                glyph.fallback.to_string()
            };
            let colored_display = match cat {
                GlyphCategory::Predicate if config.color => {
                    format!("{YELLOW}{display}{RESET}")
                }
                GlyphCategory::TypeDeterminative if config.color => {
                    format!("{MAGENTA}{display}{RESET}")
                }
                GlyphCategory::Provenance if config.color => {
                    format!("{DIM}{display}{RESET}")
                }
                _ => display,
            };
            lines.push(format!(
                "    {colored_display}  {:<16} {}",
                glyph.name, glyph.description,
            ));
        }
        lines.push(String::new());
    }

    // Radicals.
    lines.push(format!(
        "  {}Sigil Radicals (32){}:",
        if config.color { BOLD } else { "" },
        if config.color { RESET } else { "" },
    ));
    for (i, radical) in sigil::all_radicals().iter().enumerate() {
        let display = if config.notation.use_pua {
            radical.pua_codepoint.to_string()
        } else {
            radical.fallback.to_string()
        };
        let colored = if config.color {
            format!("{GREEN}{display}{RESET}")
        } else {
            display
        };
        let cat = match radical.category {
            sigil::RadicalCategory::Beings => "being",
            sigil::RadicalCategory::Nature => "nature",
            sigil::RadicalCategory::Structure => "struct",
            sigil::RadicalCategory::Abstract => "abstr",
        };
        lines.push(format!(
            "    [{i:2}] {colored}  {:<10} ({cat})",
            radical.name,
        ));
    }

    lines.join("\n")
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Render a single triple with ANSI colors.
fn render_triple_colored(engine: &Engine, triple: &Triple, config: &RenderConfig) -> String {
    let open = structural_glyph_str("struct:triple", config.notation.use_pua);
    let close = structural_glyph_str("struct:end", config.notation.use_pua);
    let subj = render_entity_colored(engine, triple.subject, config);
    let pred = render_predicate_colored(engine, triple.predicate, config);
    let obj = render_entity_colored(engine, triple.object, config);

    let mut line = format!("{open} {subj} {pred} {obj} {close}");

    if config.notation.show_confidence {
        line.push(' ');
        line.push_str(&format!(
            "{BOLD}{}{RESET}",
            catalog::confidence_dots(triple.confidence),
        ));
    }

    line
}

/// Render an entity with ANSI colors: green sigil + cyan label.
fn render_entity_colored(engine: &Engine, symbol: SymbolId, config: &RenderConfig) -> String {
    let label = engine.resolve_label(symbol);
    if config.notation.show_sigils {
        let s = sigil::sigil_for_symbol(symbol, engine.item_memory(), config.notation.use_pua)
            .unwrap_or_default();
        format!("{GREEN}{s}{RESET}{CYAN}{label}{RESET}")
    } else {
        format!("{CYAN}{label}{RESET}")
    }
}

/// Render a predicate with ANSI yellow color.
fn render_predicate_colored(
    engine: &Engine,
    predicate: SymbolId,
    config: &RenderConfig,
) -> String {
    let label = engine.resolve_label(predicate);
    if let Some(glyph) = catalog::lookup(&label) {
        let display = if config.notation.use_pua {
            glyph.pua_codepoint.to_string()
        } else {
            glyph.fallback.to_string()
        };
        format!("{YELLOW}{display}{RESET}")
    } else {
        format!("{YELLOW}{label}{RESET}")
    }
}

/// Get a structural glyph display string.
fn structural_glyph_str(name: &str, use_pua: bool) -> String {
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

/// Detect whether stdout is a TTY.
fn atty_detect() -> bool {
    // Simple heuristic: check if the TERM env var is set.
    std::env::var("TERM").is_ok()
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
    fn colored_output_contains_ansi() {
        let (engine, dog, is_a, animal) = setup();
        let triple = Triple::new(dog, is_a, animal);
        let config = RenderConfig {
            color: true,
            notation: NotationConfig {
                show_sigils: false,
                show_confidence: false,
                use_pua: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let rendered = render_to_terminal(&engine, &[triple], &config);
        assert!(rendered.contains("\x1b["), "should contain ANSI escape codes");
    }

    #[test]
    fn no_color_mode() {
        let (engine, dog, is_a, animal) = setup();
        let triple = Triple::new(dog, is_a, animal);
        let config = RenderConfig {
            color: false,
            notation: NotationConfig {
                show_sigils: false,
                show_confidence: false,
                use_pua: false,
                compact: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let rendered = render_to_terminal(&engine, &[triple], &config);
        assert!(
            !rendered.contains("\x1b["),
            "should not contain ANSI codes when color=false"
        );
    }

    #[test]
    fn legend_lists_all_35_glyphs() {
        let config = RenderConfig {
            color: false,
            notation: NotationConfig {
                use_pua: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let legend = render_legend(&config);
        // Check that all 35 glyph names appear.
        for glyph in catalog::all_glyphs() {
            assert!(
                legend.contains(glyph.name),
                "legend missing glyph: {}",
                glyph.name,
            );
        }
    }

    #[test]
    fn legend_lists_all_32_radicals() {
        let config = RenderConfig {
            color: false,
            notation: NotationConfig {
                use_pua: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let legend = render_legend(&config);
        for radical in sigil::all_radicals().iter() {
            assert!(
                legend.contains(radical.name),
                "legend missing radical: {}",
                radical.name,
            );
        }
    }

    #[test]
    fn render_symbol_with_sigil() {
        let (engine, dog, _, _) = setup();
        let config = RenderConfig {
            color: false,
            notation: NotationConfig {
                show_sigils: true,
                use_pua: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let rendered = render_symbol(&engine, dog, &config);
        assert!(rendered.contains("Dog"));
        // Should have some sigil characters before the label.
        assert!(rendered.len() > "Dog".len());
    }
}
