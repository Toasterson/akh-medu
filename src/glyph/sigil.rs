//! VSA-to-hieroglyphic sigil generation using compositional radicals.
//!
//! Each symbol's VSA vector determines a unique 2–3 character sigil
//! composed from 32 hieroglyphic radical primitives. This gives every
//! concept in the knowledge graph a distinct visual identity.

use std::sync::OnceLock;

use crate::glyph::GlyphError;
use crate::symbol::SymbolId;
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::HyperVec;

/// Category of a radical primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RadicalCategory {
    /// Beings: eye, bird, serpent, fish, hand, foot, face, figure.
    Beings,
    /// Nature: sun, moon, water, mountain, tree, fire, wind, earth.
    Nature,
    /// Structure: house, pillar, arch, wall, gate, path, bridge, tower.
    Structure,
    /// Abstract: ankh, spiral, star, arrow, loop, cross, wave, dot.
    Abstract,
}

/// A hieroglyphic radical primitive for composing sigils.
#[derive(Debug, Clone)]
pub struct Radical {
    /// PUA codepoint (U+E100–U+E11F) rendered by the custom font.
    pub pua_codepoint: char,
    /// Unicode fallback (Egyptian hieroglyphs or geometric symbols).
    pub fallback: &'static str,
    /// Radical category.
    pub category: RadicalCategory,
    /// Human-readable name.
    pub name: &'static str,
}

static RADICALS: OnceLock<[Radical; 32]> = OnceLock::new();

fn build_radicals() -> [Radical; 32] {
    [
        // -- Beings (8): U+E100–U+E107 --
        Radical { pua_codepoint: '\u{E100}', fallback: "\u{1F441}",  category: RadicalCategory::Beings, name: "eye" },
        Radical { pua_codepoint: '\u{E101}', fallback: "\u{1F985}",  category: RadicalCategory::Beings, name: "bird" },
        Radical { pua_codepoint: '\u{E102}', fallback: "\u{1F40D}",  category: RadicalCategory::Beings, name: "serpent" },
        Radical { pua_codepoint: '\u{E103}', fallback: "\u{1F41F}",  category: RadicalCategory::Beings, name: "fish" },
        Radical { pua_codepoint: '\u{E104}', fallback: "\u{270B}",   category: RadicalCategory::Beings, name: "hand" },
        Radical { pua_codepoint: '\u{E105}', fallback: "\u{1F9B6}",  category: RadicalCategory::Beings, name: "foot" },
        Radical { pua_codepoint: '\u{E106}', fallback: "\u{1F5FF}",  category: RadicalCategory::Beings, name: "face" },
        Radical { pua_codepoint: '\u{E107}', fallback: "\u{1F9CD}",  category: RadicalCategory::Beings, name: "figure" },

        // -- Nature (8): U+E108–U+E10F --
        Radical { pua_codepoint: '\u{E108}', fallback: "\u{2600}",   category: RadicalCategory::Nature, name: "sun" },
        Radical { pua_codepoint: '\u{E109}', fallback: "\u{263D}",   category: RadicalCategory::Nature, name: "moon" },
        Radical { pua_codepoint: '\u{E10A}', fallback: "\u{224B}",   category: RadicalCategory::Nature, name: "water" },
        Radical { pua_codepoint: '\u{E10B}', fallback: "\u{26F0}",   category: RadicalCategory::Nature, name: "mountain" },
        Radical { pua_codepoint: '\u{E10C}', fallback: "\u{1F332}",  category: RadicalCategory::Nature, name: "tree" },
        Radical { pua_codepoint: '\u{E10D}', fallback: "\u{1F525}",  category: RadicalCategory::Nature, name: "fire" },
        Radical { pua_codepoint: '\u{E10E}', fallback: "\u{1F4A8}",  category: RadicalCategory::Nature, name: "wind" },
        Radical { pua_codepoint: '\u{E10F}', fallback: "\u{2316}",   category: RadicalCategory::Nature, name: "earth" },

        // -- Structure (8): U+E110–U+E117 --
        Radical { pua_codepoint: '\u{E110}', fallback: "\u{2302}",   category: RadicalCategory::Structure, name: "house" },
        Radical { pua_codepoint: '\u{E111}', fallback: "\u{2503}",   category: RadicalCategory::Structure, name: "pillar" },
        Radical { pua_codepoint: '\u{E112}', fallback: "\u{2229}",   category: RadicalCategory::Structure, name: "arch" },
        Radical { pua_codepoint: '\u{E113}', fallback: "\u{2588}",   category: RadicalCategory::Structure, name: "wall" },
        Radical { pua_codepoint: '\u{E114}', fallback: "\u{26E9}",   category: RadicalCategory::Structure, name: "gate" },
        Radical { pua_codepoint: '\u{E115}', fallback: "\u{2550}",   category: RadicalCategory::Structure, name: "path" },
        Radical { pua_codepoint: '\u{E116}', fallback: "\u{2517}",   category: RadicalCategory::Structure, name: "bridge" },
        Radical { pua_codepoint: '\u{E117}', fallback: "\u{25EF}",   category: RadicalCategory::Structure, name: "tower" },

        // -- Abstract (8): U+E118–U+E11F --
        Radical { pua_codepoint: '\u{E118}', fallback: "\u{2625}",   category: RadicalCategory::Abstract, name: "ankh" },
        Radical { pua_codepoint: '\u{E119}', fallback: "\u{1F300}",  category: RadicalCategory::Abstract, name: "spiral" },
        Radical { pua_codepoint: '\u{E11A}', fallback: "\u{2605}",   category: RadicalCategory::Abstract, name: "star" },
        Radical { pua_codepoint: '\u{E11B}', fallback: "\u{27A4}",   category: RadicalCategory::Abstract, name: "arrow" },
        Radical { pua_codepoint: '\u{E11C}', fallback: "\u{221E}",   category: RadicalCategory::Abstract, name: "loop" },
        Radical { pua_codepoint: '\u{E11D}', fallback: "\u{2726}",   category: RadicalCategory::Abstract, name: "cross" },
        Radical { pua_codepoint: '\u{E11E}', fallback: "\u{223F}",   category: RadicalCategory::Abstract, name: "wave" },
        Radical { pua_codepoint: '\u{E11F}', fallback: "\u{00B7}",   category: RadicalCategory::Abstract, name: "dot" },
    ]
}

/// Get all 32 radical primitives.
pub fn all_radicals() -> &'static [Radical; 32] {
    RADICALS.get_or_init(build_radicals)
}

/// Generate a 3-radical sigil from a VSA vector.
///
/// Extracts 5 bits from three different regions of the vector,
/// selecting one radical from the table for each position.
/// This yields 32^3 = 32,768 unique combinations.
pub fn generate_sigil(vec: &HyperVec, use_pua: bool) -> String {
    let radicals = all_radicals();
    let dim = vec.dim().0;
    let byte_len = vec.byte_len();

    if byte_len == 0 {
        return default_sigil(use_pua);
    }

    // Extract 5-bit indices from three regions of the vector.
    let idx0 = extract_5bits(vec, 0);
    let idx1 = extract_5bits(vec, byte_len.min(dim / 3 / 8));
    let idx2 = extract_5bits(vec, byte_len.min(2 * dim / 3 / 8));

    format_radicals(&[idx0, idx1, idx2], radicals, use_pua)
}

/// Generate a compact 2-radical sigil from a VSA vector.
///
/// Uses only two regions for 32^2 = 1,024 combinations.
pub fn generate_compact_sigil(vec: &HyperVec, use_pua: bool) -> String {
    let radicals = all_radicals();
    let dim = vec.dim().0;
    let byte_len = vec.byte_len();

    if byte_len == 0 {
        return default_sigil(use_pua);
    }

    let idx0 = extract_5bits(vec, 0);
    let idx1 = extract_5bits(vec, byte_len.min(dim / 2 / 8));

    format_radicals(&[idx0, idx1], radicals, use_pua)
}

/// Generate a sigil for a symbol by looking up its VSA vector.
pub fn sigil_for_symbol(
    symbol: SymbolId,
    item_memory: &ItemMemory,
    use_pua: bool,
) -> crate::glyph::GlyphResult<String> {
    let vec = item_memory.get(symbol).ok_or(GlyphError::NoVector {
        symbol_id: symbol.to_string(),
    })?;
    Ok(generate_sigil(&vec, use_pua))
}

/// Extract a 5-bit index (0–31) from the vector data at the given byte offset.
fn extract_5bits(vec: &HyperVec, byte_offset: usize) -> usize {
    let data = vec.data();
    if byte_offset >= data.len() {
        return 0;
    }
    (data[byte_offset] & 0x1F) as usize
}

/// Format radical indices into a display string.
fn format_radicals(indices: &[usize], radicals: &[Radical; 32], use_pua: bool) -> String {
    indices
        .iter()
        .map(|&idx| {
            let r = &radicals[idx % 32];
            if use_pua {
                r.pua_codepoint.to_string()
            } else {
                r.fallback.to_string()
            }
        })
        .collect()
}

/// Default sigil for zero-length vectors.
fn default_sigil(use_pua: bool) -> String {
    let r = &all_radicals()[31]; // dot
    let s = if use_pua {
        r.pua_codepoint.to_string()
    } else {
        r.fallback.to_string()
    };
    format!("{s}{s}{s}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vsa::{Dimension, Encoding, HyperVec};

    fn test_vec(seed: u8) -> HyperVec {
        let dim = Dimension::TEST;
        let mut data = vec![seed; dim.binary_byte_len()];
        // Make each vector distinct by varying a few bytes.
        for (i, b) in data.iter_mut().enumerate() {
            *b = b.wrapping_add(i as u8);
        }
        HyperVec::from_raw(data, dim, Encoding::Bipolar)
    }

    #[test]
    fn radicals_count() {
        assert_eq!(all_radicals().len(), 32);
    }

    #[test]
    fn radical_pua_in_range() {
        for r in all_radicals().iter() {
            let cp = r.pua_codepoint as u32;
            assert!(
                (0xE100..=0xE11F).contains(&cp),
                "radical {} codepoint U+{:04X} out of range",
                r.name, cp,
            );
        }
    }

    #[test]
    fn sigil_is_3_radicals() {
        let vec = test_vec(42);
        let sigil = generate_sigil(&vec, false);
        // Each radical fallback is at least 1 char, sigil should have 3 distinct parts.
        // We just verify it's non-empty and longer than a single character.
        assert!(sigil.chars().count() >= 3, "sigil too short: {sigil}");
    }

    #[test]
    fn compact_sigil_is_2_radicals() {
        let vec = test_vec(42);
        let sigil = generate_compact_sigil(&vec, false);
        assert!(sigil.chars().count() >= 2, "compact sigil too short: {sigil}");
    }

    #[test]
    fn different_vectors_produce_different_sigils() {
        let v1 = test_vec(10);
        let v2 = test_vec(200);
        let s1 = generate_sigil(&v1, false);
        let s2 = generate_sigil(&v2, false);
        assert_ne!(s1, s2, "different vectors should produce different sigils");
    }

    #[test]
    fn pua_mode_uses_pua_codepoints() {
        let vec = test_vec(42);
        let sigil = generate_sigil(&vec, true);
        // PUA chars are in U+E100–U+E11F range.
        for ch in sigil.chars() {
            let cp = ch as u32;
            assert!(
                (0xE100..=0xE11F).contains(&cp),
                "expected PUA codepoint, got U+{cp:04X}",
            );
        }
    }

    #[test]
    fn empty_vector_gets_default_sigil() {
        let vec = HyperVec::from_raw(vec![], Dimension(0), Encoding::Bipolar);
        let sigil = generate_sigil(&vec, false);
        assert!(!sigil.is_empty());
    }

    #[test]
    fn radical_categories_count() {
        let beings = all_radicals().iter().filter(|r| r.category == RadicalCategory::Beings).count();
        let nature = all_radicals().iter().filter(|r| r.category == RadicalCategory::Nature).count();
        let structure = all_radicals().iter().filter(|r| r.category == RadicalCategory::Structure).count();
        let abstr = all_radicals().iter().filter(|r| r.category == RadicalCategory::Abstract).count();
        assert_eq!(beings, 8);
        assert_eq!(nature, 8);
        assert_eq!(structure, 8);
        assert_eq!(abstr, 8);
    }
}
