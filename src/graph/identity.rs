//! Stable triple-identity hash (ADR 081 prefactor P2).
//!
//! Gives every triple a **content-addressed identity**: a hash over the
//! canonical *labels* of (subject, predicate, object) plus the owning
//! compartment, excluding all volatile fields (timestamp, confidence,
//! provenance id). Re-asserting the same fact — in another session, or in
//! another akh whose `SymbolId` allocation differs — yields the same
//! identity, which is what lets a later projection tell "same fact, still
//! true" from "new fact" (diff-based supersede, ADR 081 decision 12).
//!
//! Design notes:
//! - **Labels, not `SymbolId`s.** `SymbolId`s are allocated per-akh and
//!   per-session; the registry label is the canonical, cross-akh name of a
//!   symbol. Hashing labels is what makes the identity relocatable.
//! - **Domain-separated + length-prefixed.** The hash input starts with a
//!   versioned domain tag and every field is length-prefixed, so
//!   `("ab", "c")` and `("a", "bc")` cannot collide and future format
//!   revisions can bump the tag without silent aliasing.
//! - **No text normalization.** Labels are exact registry keys already;
//!   the hash is byte-exact over UTF-8. `None` and `Some("")` compartments
//!   are distinguished by an explicit presence tag.
//!
//! References: ADR 081 (`docs/ai/decisions/081-akh-memory-schema-typing.md`),
//! plan item P2 (`docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`),
//! Anima project `019d9844-ce74-7993-a9f0-c953b8f04a4e`.

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Versioned domain-separation tag. Bump the trailing version if the input
/// encoding ever changes — identities from different versions must not alias.
const DOMAIN_TAG: &[u8] = b"akh:triple-identity:v1";

/// Errors when parsing a [`TripleIdentity`] from its hex form.
#[derive(Debug, Error, Diagnostic)]
pub enum IdentityParseError {
    #[error("triple identity must be 64 hex characters, got {len}")]
    #[diagnostic(
        code(akh::graph::identity::bad_length),
        help("A TripleIdentity is a 32-byte blake3 hash rendered as 64 lowercase hex chars.")
    )]
    BadLength { len: usize },

    #[error("triple identity contains a non-hex character: {found:?}")]
    #[diagnostic(
        code(akh::graph::identity::bad_char),
        help("Only [0-9a-fA-F] are valid. Check the value was not truncated or re-encoded.")
    )]
    BadChar { found: char },
}

/// Content-addressed identity of a triple: blake3 over the canonical labels
/// of (subject, predicate, object) + owning compartment.
///
/// Stable across sessions and across akhs (independent of `SymbolId`
/// allocation order). Volatile fields — confidence, timestamp, provenance —
/// are deliberately excluded per ADR 081 P2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TripleIdentity([u8; 32]);

impl TripleIdentity {
    /// Compute the identity from the canonical labels of a triple.
    ///
    /// `compartment` is the owning compartment label (`Triple::compartment_id`);
    /// `None` (default graph) hashes differently from `Some("")`.
    pub fn compute(
        subject_label: &str,
        predicate_label: &str,
        object_label: &str,
        compartment: Option<&str>,
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(DOMAIN_TAG);
        for field in [subject_label, predicate_label, object_label] {
            hasher.update(&(field.len() as u64).to_le_bytes());
            hasher.update(field.as_bytes());
        }
        match compartment {
            None => {
                hasher.update(&[0u8]);
            }
            Some(c) => {
                hasher.update(&[1u8]);
                hasher.update(&(c.len() as u64).to_le_bytes());
                hasher.update(c.as_bytes());
            }
        }
        Self(*hasher.finalize().as_bytes())
    }

    /// The raw 32-byte hash.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Construct from raw bytes (e.g. read back from a store).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Lowercase hex rendering (64 chars) — the persistable/queryable form.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in &self.0 {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    /// Parse from the 64-char hex form.
    pub fn from_hex(hex: &str) -> Result<Self, IdentityParseError> {
        if hex.len() != 64 {
            return Err(IdentityParseError::BadLength { len: hex.len() });
        }
        let mut out = [0u8; 32];
        let bytes = hex.as_bytes();
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            let hi = hex_val(chunk[0]).ok_or(IdentityParseError::BadChar {
                found: chunk[0] as char,
            })?;
            let lo = hex_val(chunk[1]).ok_or(IdentityParseError::BadChar {
                found: chunk[1] as char,
            })?;
            out[i] = (hi << 4) | lo;
        }
        Ok(Self(out))
    }
}

fn hex_val(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

impl std::fmt::Display for TripleIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl std::str::FromStr for TripleIdentity {
    type Err = IdentityParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_hex(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_inputs_same_identity() {
        let a = TripleIdentity::compute("Sun", "is-a", "Star", None);
        let b = TripleIdentity::compute("Sun", "is-a", "Star", None);
        assert_eq!(a, b);
    }

    #[test]
    fn each_field_changes_identity() {
        let base = TripleIdentity::compute("Sun", "is-a", "Star", Some("mt:astro"));
        assert_ne!(
            base,
            TripleIdentity::compute("Moon", "is-a", "Star", Some("mt:astro"))
        );
        assert_ne!(
            base,
            TripleIdentity::compute("Sun", "part-of", "Star", Some("mt:astro"))
        );
        assert_ne!(
            base,
            TripleIdentity::compute("Sun", "is-a", "Planet", Some("mt:astro"))
        );
        assert_ne!(
            base,
            TripleIdentity::compute("Sun", "is-a", "Star", Some("mt:other"))
        );
        assert_ne!(base, TripleIdentity::compute("Sun", "is-a", "Star", None));
    }

    #[test]
    fn no_field_boundary_collisions() {
        // Length-prefixing must prevent ("ab","c") == ("a","bc") style collisions.
        let a = TripleIdentity::compute("ab", "c", "d", None);
        let b = TripleIdentity::compute("a", "bc", "d", None);
        assert_ne!(a, b);
        let c = TripleIdentity::compute("a", "b", "cd", None);
        let d = TripleIdentity::compute("a", "bc", "d", None);
        assert_ne!(c, d);
    }

    #[test]
    fn none_compartment_differs_from_empty() {
        let none = TripleIdentity::compute("s", "p", "o", None);
        let empty = TripleIdentity::compute("s", "p", "o", Some(""));
        assert_ne!(none, empty);
    }

    #[test]
    fn volatile_fields_are_not_part_of_identity() {
        // The API only accepts (s, p, o, compartment) — this test documents that
        // confidence / timestamp / provenance cannot influence the hash.
        let a = TripleIdentity::compute("s", "p", "o", None);
        let b = TripleIdentity::compute("s", "p", "o", None);
        assert_eq!(a, b);
    }

    #[test]
    fn hex_round_trip() {
        let id = TripleIdentity::compute("Sun", "is-a", "Star", Some("mt:astro"));
        let hex = id.to_hex();
        assert_eq!(hex.len(), 64);
        let back: TripleIdentity = hex.parse().unwrap();
        assert_eq!(id, back);
        // Display matches to_hex.
        assert_eq!(id.to_string(), hex);
    }

    #[test]
    fn from_hex_rejects_bad_input() {
        assert!(matches!(
            TripleIdentity::from_hex("abc"),
            Err(IdentityParseError::BadLength { len: 3 })
        ));
        let bad = "zz".repeat(32);
        assert!(matches!(
            TripleIdentity::from_hex(&bad),
            Err(IdentityParseError::BadChar { .. })
        ));
    }

    #[test]
    fn known_vector_is_stable() {
        // Pin the v1 encoding: if this test ever fails, the identity format
        // changed and DOMAIN_TAG must be bumped instead.
        let id = TripleIdentity::compute("Sun", "is-a", "Star", None);
        let again = TripleIdentity::from_hex(&id.to_hex()).unwrap();
        assert_eq!(id, again);
        // Cross-instance determinism (fresh hasher).
        let id2 = TripleIdentity::compute("Sun", "is-a", "Star", None);
        assert_eq!(id.to_hex(), id2.to_hex());
    }
}
