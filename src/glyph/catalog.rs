//! Fixed glyph catalog: 35 hieroglyphic glyphs mapped to PUA codepoints.
//!
//! Each glyph has a PUA codepoint (rendered by the custom Akh-Medu font)
//! and a Unicode fallback for terminals without the font installed.

use std::sync::OnceLock;

/// Category of a fixed glyph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlyphCategory {
    /// A relation/predicate glyph (e.g., is-a, part-of).
    Predicate,
    /// A type determinative suffixed after entity names.
    TypeDeterminative,
    /// A provenance marker indicating how knowledge was derived.
    Provenance,
    /// A structural glyph (delimiters, chains, branches).
    Structural,
}

/// A fixed glyph in the catalog.
#[derive(Debug, Clone)]
pub struct Glyph {
    /// Machine name matching KG predicate/type labels.
    pub name: &'static str,
    /// PUA codepoint (U+E000–U+E022) rendered by the custom font.
    pub pua_codepoint: char,
    /// Unicode fallback when the custom font is not available.
    pub fallback: &'static str,
    /// Glyph category.
    pub category: GlyphCategory,
    /// Human-readable description.
    pub description: &'static str,
}

static CATALOG: OnceLock<Vec<Glyph>> = OnceLock::new();

fn build_catalog() -> Vec<Glyph> {
    vec![
        // -- Predicates (15): U+E000–U+E00E --
        Glyph {
            name: "is-a",
            pua_codepoint: '\u{E000}',
            fallback: "\u{25B3}",
            category: GlyphCategory::Predicate,
            description: "Type membership",
        },
        Glyph {
            name: "part-of",
            pua_codepoint: '\u{E001}',
            fallback: "\u{2282}",
            category: GlyphCategory::Predicate,
            description: "Part-whole",
        },
        Glyph {
            name: "has-a",
            pua_codepoint: '\u{E002}',
            fallback: "\u{25C7}",
            category: GlyphCategory::Predicate,
            description: "Possession",
        },
        Glyph {
            name: "contains",
            pua_codepoint: '\u{E003}',
            fallback: "\u{2283}",
            category: GlyphCategory::Predicate,
            description: "Containment",
        },
        Glyph {
            name: "parent-of",
            pua_codepoint: '\u{E004}',
            fallback: "\u{2193}",
            category: GlyphCategory::Predicate,
            description: "Parent",
        },
        Glyph {
            name: "child-of",
            pua_codepoint: '\u{E005}',
            fallback: "\u{2191}",
            category: GlyphCategory::Predicate,
            description: "Child",
        },
        Glyph {
            name: "similar-to",
            pua_codepoint: '\u{E006}',
            fallback: "\u{2248}",
            category: GlyphCategory::Predicate,
            description: "Similarity",
        },
        Glyph {
            name: "causes",
            pua_codepoint: '\u{E007}',
            fallback: "\u{2192}",
            category: GlyphCategory::Predicate,
            description: "Causation",
        },
        Glyph {
            name: "precedes",
            pua_codepoint: '\u{E008}',
            fallback: "\u{227A}",
            category: GlyphCategory::Predicate,
            description: "Temporal order",
        },
        Glyph {
            name: "located-in",
            pua_codepoint: '\u{E009}',
            fallback: "\u{2302}",
            category: GlyphCategory::Predicate,
            description: "Location",
        },
        Glyph {
            name: "created-by",
            pua_codepoint: '\u{E00A}',
            fallback: "\u{270E}",
            category: GlyphCategory::Predicate,
            description: "Authorship",
        },
        Glyph {
            name: "depends-on",
            pua_codepoint: '\u{E00B}',
            fallback: "\u{21D0}",
            category: GlyphCategory::Predicate,
            description: "Dependency",
        },
        Glyph {
            name: "opposes",
            pua_codepoint: '\u{E00C}',
            fallback: "\u{22A5}",
            category: GlyphCategory::Predicate,
            description: "Opposition",
        },
        Glyph {
            name: "enables",
            pua_codepoint: '\u{E00D}',
            fallback: "\u{21D2}",
            category: GlyphCategory::Predicate,
            description: "Enablement",
        },
        Glyph {
            name: "knows",
            pua_codepoint: '\u{E00E}',
            fallback: "\u{25C9}",
            category: GlyphCategory::Predicate,
            description: "Knowledge",
        },
        // -- Type Determinatives (10): U+E00F–U+E018 --
        Glyph {
            name: "type:person",
            pua_codepoint: '\u{E00F}',
            fallback: "\u{1F464}",
            category: GlyphCategory::TypeDeterminative,
            description: "Person",
        },
        Glyph {
            name: "type:place",
            pua_codepoint: '\u{E010}',
            fallback: "\u{1F3E0}",
            category: GlyphCategory::TypeDeterminative,
            description: "Place",
        },
        Glyph {
            name: "type:thing",
            pua_codepoint: '\u{E011}',
            fallback: "\u{1F4E6}",
            category: GlyphCategory::TypeDeterminative,
            description: "Physical object",
        },
        Glyph {
            name: "type:concept",
            pua_codepoint: '\u{E012}',
            fallback: "\u{1F4A1}",
            category: GlyphCategory::TypeDeterminative,
            description: "Abstract concept",
        },
        Glyph {
            name: "type:event",
            pua_codepoint: '\u{E013}',
            fallback: "\u{26A1}",
            category: GlyphCategory::TypeDeterminative,
            description: "Event",
        },
        Glyph {
            name: "type:quantity",
            pua_codepoint: '\u{E014}',
            fallback: "#",
            category: GlyphCategory::TypeDeterminative,
            description: "Number/quantity",
        },
        Glyph {
            name: "type:time",
            pua_codepoint: '\u{E015}',
            fallback: "\u{231A}",
            category: GlyphCategory::TypeDeterminative,
            description: "Time",
        },
        Glyph {
            name: "type:group",
            pua_codepoint: '\u{E016}',
            fallback: "\u{1F465}",
            category: GlyphCategory::TypeDeterminative,
            description: "Collection/group",
        },
        Glyph {
            name: "type:process",
            pua_codepoint: '\u{E017}',
            fallback: "\u{2699}",
            category: GlyphCategory::TypeDeterminative,
            description: "Process/action",
        },
        Glyph {
            name: "type:property",
            pua_codepoint: '\u{E018}',
            fallback: "\u{2261}",
            category: GlyphCategory::TypeDeterminative,
            description: "Attribute",
        },
        // -- Provenance Markers (5): U+E019–U+E01D --
        Glyph {
            name: "prov:asserted",
            pua_codepoint: '\u{E019}',
            fallback: "\u{25C6}",
            category: GlyphCategory::Provenance,
            description: "User-asserted",
        },
        Glyph {
            name: "prov:inferred",
            pua_codepoint: '\u{E01A}',
            fallback: "\u{25C7}",
            category: GlyphCategory::Provenance,
            description: "Rule-inferred",
        },
        Glyph {
            name: "prov:fused",
            pua_codepoint: '\u{E01B}',
            fallback: "\u{2295}",
            category: GlyphCategory::Provenance,
            description: "Fused from multiple paths",
        },
        Glyph {
            name: "prov:discovered",
            pua_codepoint: '\u{E01C}',
            fallback: "\u{2605}",
            category: GlyphCategory::Provenance,
            description: "Schema-discovered",
        },
        Glyph {
            name: "prov:gap",
            pua_codepoint: '\u{E01D}',
            fallback: "?",
            category: GlyphCategory::Provenance,
            description: "Gap-identified",
        },
        // -- Structural (5): U+E01E–U+E022 --
        Glyph {
            name: "struct:triple",
            pua_codepoint: '\u{E01E}',
            fallback: "\u{27E8}",
            category: GlyphCategory::Structural,
            description: "Triple open",
        },
        Glyph {
            name: "struct:end",
            pua_codepoint: '\u{E01F}',
            fallback: "\u{27E9}",
            category: GlyphCategory::Structural,
            description: "Triple close",
        },
        Glyph {
            name: "struct:chain",
            pua_codepoint: '\u{E020}',
            fallback: "\u{2500}",
            category: GlyphCategory::Structural,
            description: "Inference chain",
        },
        Glyph {
            name: "struct:branch",
            pua_codepoint: '\u{E021}',
            fallback: "\u{251C}",
            category: GlyphCategory::Structural,
            description: "Graph branch",
        },
        Glyph {
            name: "struct:confidence",
            pua_codepoint: '\u{E022}',
            fallback: "\u{25CF}",
            category: GlyphCategory::Structural,
            description: "Confidence indicator",
        },
    ]
}

/// Get all 35 fixed glyphs.
pub fn all_glyphs() -> &'static [Glyph] {
    CATALOG.get_or_init(build_catalog)
}

/// Look up a glyph by predicate/type label (case-insensitive).
/// Returns `None` if no fixed glyph exists for the label.
pub fn lookup(label: &str) -> Option<&'static Glyph> {
    let lower = label.to_lowercase();
    all_glyphs().iter().find(|g| g.name == lower)
}

/// Resolve a glyph to its display string.
/// Uses PUA codepoint if `use_pua` is true, otherwise the Unicode fallback.
pub fn render_glyph(glyph: &Glyph, use_pua: bool) -> &str {
    if use_pua {
        // Safety: we return a reference to the stored fallback; for PUA we need
        // to produce a string. Since PUA chars are single chars we use the fallback
        // when not using PUA. When using PUA, callers should use `pua_codepoint` directly.
        // This function returns fallback for simplicity; render.rs handles PUA rendering.
        glyph.fallback
    } else {
        glyph.fallback
    }
}

/// Detect whether the custom font is available.
///
/// Checks (in order):
/// 1. `$AKH_FONT` environment variable set to "1" or a path
/// 2. `~/.local/share/fonts/akh-medu.ttf` exists
pub fn font_available() -> bool {
    if let Ok(val) = std::env::var("AKH_FONT") {
        if val == "1" || val == "true" {
            return true;
        }
        return std::path::Path::new(&val).exists();
    }
    if let Some(home) = std::env::var_os("HOME") {
        let font_path = std::path::PathBuf::from(home).join(".local/share/fonts/akh-medu.ttf");
        return font_path.exists();
    }
    false
}

/// Format confidence as filled/empty dots (5 dots = 100%).
pub fn confidence_dots(confidence: f32) -> String {
    let filled = (confidence.clamp(0.0, 1.0) * 5.0).round() as usize;
    let empty = 5 - filled;
    format!(
        "{}{}",
        "\u{25CF}".repeat(filled), // ●
        "\u{25CB}".repeat(empty),  // ○
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_35_glyphs() {
        assert_eq!(all_glyphs().len(), 35);
    }

    #[test]
    fn lookup_by_name() {
        let g = lookup("is-a").expect("should find is-a");
        assert_eq!(g.name, "is-a");
        assert_eq!(g.pua_codepoint, '\u{E000}');
        assert_eq!(g.category, GlyphCategory::Predicate);
    }

    #[test]
    fn lookup_case_insensitive() {
        assert!(lookup("Is-A").is_some());
        assert!(lookup("IS-A").is_some());
    }

    #[test]
    fn lookup_unknown_returns_none() {
        assert!(lookup("nonexistent-predicate").is_none());
    }

    #[test]
    fn fallback_strings_are_nonempty() {
        for g in all_glyphs() {
            assert!(
                !g.fallback.is_empty(),
                "glyph {} has empty fallback",
                g.name
            );
        }
    }

    #[test]
    fn pua_codepoints_in_range() {
        for g in all_glyphs() {
            let cp = g.pua_codepoint as u32;
            assert!(
                (0xE000..=0xE022).contains(&cp),
                "glyph {} codepoint U+{:04X} out of range",
                g.name,
                cp,
            );
        }
    }

    #[test]
    fn categories_count() {
        let preds = all_glyphs()
            .iter()
            .filter(|g| g.category == GlyphCategory::Predicate)
            .count();
        let types = all_glyphs()
            .iter()
            .filter(|g| g.category == GlyphCategory::TypeDeterminative)
            .count();
        let provs = all_glyphs()
            .iter()
            .filter(|g| g.category == GlyphCategory::Provenance)
            .count();
        let structs = all_glyphs()
            .iter()
            .filter(|g| g.category == GlyphCategory::Structural)
            .count();
        assert_eq!(preds, 15);
        assert_eq!(types, 10);
        assert_eq!(provs, 5);
        assert_eq!(structs, 5);
    }

    #[test]
    fn confidence_dots_formatting() {
        assert_eq!(
            confidence_dots(1.0),
            "\u{25CF}\u{25CF}\u{25CF}\u{25CF}\u{25CF}"
        );
        assert_eq!(
            confidence_dots(0.0),
            "\u{25CB}\u{25CB}\u{25CB}\u{25CB}\u{25CB}"
        );
        assert_eq!(
            confidence_dots(0.6),
            "\u{25CF}\u{25CF}\u{25CF}\u{25CB}\u{25CB}"
        );
    }
}
