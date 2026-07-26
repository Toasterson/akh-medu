//! Shared normalized-similarity primitives (Levenshtein-edit-distance based).
//!
//! `repair.rs`, `ground.rs`, and (in the main crate) `src/agent/organ.rs` each
//! carried their own private `normalized_similarity`/`levenshtein` pair. They
//! were near-identical but had quietly drifted on one semantic axis — case
//! sensitivity (`repair.rs` compared case-sensitively; `ground.rs` and
//! `organ.rs` lowercased first) — which is exactly the kind of divergence
//! that is invisible in a diff and easy to reintroduce by copy-paste. This
//! module is the single implementation; callers pick the case-sensitive or
//! case-insensitive entry point explicitly instead of re-deriving one.

/// Classic two-row Levenshtein edit distance over char slices.
///
/// `O(min-extra-space)`: only two rows of the DP table are kept live at once
/// (swapped each outer iteration) rather than the full `|a| x |b|` matrix.
pub fn levenshtein(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (prev[j + 1] + 1).min(curr[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

/// Case-**sensitive** normalized name similarity in `[0, 1]`:
/// `1 - lev(a, b) / max(|a|, |b|)`. `1.0` for identical strings (including
/// two empty strings); `0.0` for maximally different same-length strings.
///
/// Used where casing is itself meaningful signal (e.g. `repair::ground_tool`
/// comparing tool/param names emitted by a model against a schema whose
/// naming convention is already fixed).
pub fn normalized_similarity(a: &str, b: &str) -> f64 {
    if a == b {
        return 1.0;
    }
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    let max = ac.len().max(bc.len());
    if max == 0 {
        return 1.0;
    }
    1.0 - levenshtein(&ac, &bc) as f64 / max as f64
}

/// Case-**insensitive** normalized name similarity in `[0, 1]` (both sides
/// lowercased before comparison). Used where the caller's names may differ
/// only in casing without that being a meaningful mismatch (e.g. a model's
/// emitted string value snapped onto a request span, or a hallucinated tool
/// name checked against an accept-gate floor).
pub fn normalized_similarity_ci(a: &str, b: &str) -> f64 {
    let al: Vec<char> = a.to_ascii_lowercase().chars().collect();
    let bl: Vec<char> = b.to_ascii_lowercase().chars().collect();
    let max = al.len().max(bl.len());
    if max == 0 {
        return 1.0;
    }
    1.0 - levenshtein(&al, &bl) as f64 / max as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings_are_perfectly_similar() {
        assert_eq!(normalized_similarity("abc", "abc"), 1.0);
        assert_eq!(normalized_similarity_ci("ABC", "abc"), 1.0);
    }

    #[test]
    fn two_empty_strings_are_perfectly_similar() {
        assert_eq!(normalized_similarity("", ""), 1.0);
        assert_eq!(normalized_similarity_ci("", ""), 1.0);
    }

    #[test]
    fn case_sensitivity_is_the_documented_difference() {
        // Same characters, different case: sensitive variant sees a mismatch,
        // insensitive variant does not.
        assert!(normalized_similarity("Zurich", "zurich") < 1.0);
        assert_eq!(normalized_similarity_ci("Zurich", "zurich"), 1.0);
    }

    #[test]
    fn levenshtein_matches_known_distances() {
        let a: Vec<char> = "kitten".chars().collect();
        let b: Vec<char> = "sitting".chars().collect();
        assert_eq!(levenshtein(&a, &b), 3);
    }
}
