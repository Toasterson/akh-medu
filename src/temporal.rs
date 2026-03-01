//! Temporal projection: time-dependent confidence decay for triples.
//!
//! Inspired by Cyc's temporal projection, where truth values change over time.
//! Relations carry a `TemporalProfile` specifying how confidence decays:
//!
//! - **Stable** — no decay (mathematical truths, species membership)
//! - **Decaying** — exponential half-life decay (ownership, employment)
//! - **Ephemeral** — hard TTL cutoff (location, mood)
//! - **Periodic** — cyclical truth with sinusoidal modulation (seasonal facts)
//!
//! When querying, `apply_temporal_decay` adjusts the stored confidence based on
//! the triple's timestamp and the current query time.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhResult;
use crate::graph::Triple;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to temporal projection.
#[derive(Debug, Error, Diagnostic)]
pub enum TemporalError {
    #[error("temporal profile not found for relation {relation_id}")]
    #[diagnostic(
        code(akh::temporal::profile_not_found),
        help(
            "No temporal profile is registered for this relation. \
             Register one with `set_temporal_profile()` or use the default (Stable)."
        )
    )]
    ProfileNotFound { relation_id: u64 },

    #[error("invalid half-life: {half_life_secs} seconds (must be > 0)")]
    #[diagnostic(
        code(akh::temporal::invalid_half_life),
        help("The half-life for a Decaying profile must be a positive number of seconds.")
    )]
    InvalidHalfLife { half_life_secs: u64 },

    #[error("invalid TTL: {ttl_secs} seconds (must be > 0)")]
    #[diagnostic(
        code(akh::temporal::invalid_ttl),
        help("The TTL for an Ephemeral profile must be a positive number of seconds.")
    )]
    InvalidTtl { ttl_secs: u64 },

    #[error("invalid period: {period_secs} seconds (must be > 0)")]
    #[diagnostic(
        code(akh::temporal::invalid_period),
        help("The period for a Periodic profile must be a positive number of seconds.")
    )]
    InvalidPeriod { period_secs: u64 },
}

/// Result type for temporal operations.
pub type TemporalResult<T> = std::result::Result<T, TemporalError>;

// ---------------------------------------------------------------------------
// Temporal profiles
// ---------------------------------------------------------------------------

/// How a relation's confidence decays over time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalProfile {
    /// No decay — truth is permanent (e.g., `is-a`, `has-part`, mathematical facts).
    Stable,

    /// Exponential decay with half-life.
    ///
    /// Confidence halves every `half_life_secs` seconds.
    /// `c(t) = c₀ × 2^(−Δt / half_life)`
    Decaying {
        /// Seconds for confidence to halve.
        half_life_secs: u64,
    },

    /// Hard TTL cutoff — confidence drops to zero after `ttl_secs`.
    ///
    /// Before TTL: full confidence. After: zero.
    Ephemeral {
        /// Seconds until the triple expires.
        ttl_secs: u64,
    },

    /// Cyclical truth with sinusoidal modulation.
    ///
    /// `c(t) = c₀ × (0.5 + 0.5 × cos(2π(Δt − phase) / period))`
    ///
    /// Peak confidence at `phase + n×period`, minimum at `phase + period/2 + n×period`.
    Periodic {
        /// Period in seconds.
        period_secs: u64,
        /// Phase offset in seconds (when the truth peaks).
        phase_secs: u64,
    },
}

impl TemporalProfile {
    /// Validate the profile parameters.
    pub fn validate(&self) -> TemporalResult<()> {
        match self {
            Self::Stable => Ok(()),
            Self::Decaying { half_life_secs } => {
                if *half_life_secs == 0 {
                    Err(TemporalError::InvalidHalfLife {
                        half_life_secs: *half_life_secs,
                    })
                } else {
                    Ok(())
                }
            }
            Self::Ephemeral { ttl_secs } => {
                if *ttl_secs == 0 {
                    Err(TemporalError::InvalidTtl {
                        ttl_secs: *ttl_secs,
                    })
                } else {
                    Ok(())
                }
            }
            Self::Periodic { period_secs, .. } => {
                if *period_secs == 0 {
                    Err(TemporalError::InvalidPeriod {
                        period_secs: *period_secs,
                    })
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl std::fmt::Display for TemporalProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "Stable"),
            Self::Decaying { half_life_secs } => write!(f, "Decaying(half_life={}s)", half_life_secs),
            Self::Ephemeral { ttl_secs } => write!(f, "Ephemeral(ttl={}s)", ttl_secs),
            Self::Periodic {
                period_secs,
                phase_secs,
            } => write!(f, "Periodic(period={}s, phase={}s)", period_secs, phase_secs),
        }
    }
}

// ---------------------------------------------------------------------------
// Decay computation
// ---------------------------------------------------------------------------

/// Compute decayed confidence for a triple given its temporal profile.
///
/// # Arguments
/// - `profile` — the temporal profile of the relation
/// - `original_confidence` — the stored confidence value `[0.0, 1.0]`
/// - `triple_timestamp` — when the triple was asserted (seconds since epoch)
/// - `query_time` — when the query is made (seconds since epoch)
///
/// # Returns
/// The decayed confidence, clamped to `[0.0, 1.0]`.
pub fn apply_temporal_decay(
    profile: &TemporalProfile,
    original_confidence: f32,
    triple_timestamp: u64,
    query_time: u64,
) -> f32 {
    if query_time <= triple_timestamp {
        // Query time is at or before assertion time — no decay.
        return original_confidence;
    }

    let delta = (query_time - triple_timestamp) as f64;

    match profile {
        TemporalProfile::Stable => original_confidence,

        TemporalProfile::Decaying { half_life_secs } => {
            let half_life = *half_life_secs as f64;
            let factor = (2.0_f64).powf(-delta / half_life);
            (original_confidence as f64 * factor).clamp(0.0, 1.0) as f32
        }

        TemporalProfile::Ephemeral { ttl_secs } => {
            if delta >= *ttl_secs as f64 {
                0.0
            } else {
                original_confidence
            }
        }

        TemporalProfile::Periodic {
            period_secs,
            phase_secs,
        } => {
            let period = *period_secs as f64;
            let phase = *phase_secs as f64;
            let angle = 2.0 * std::f64::consts::PI * (delta - phase) / period;
            let modulation = 0.5 + 0.5 * angle.cos();
            (original_confidence as f64 * modulation).clamp(0.0, 1.0) as f32
        }
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Well-known temporal predicates (prefixed `temporal:`).
#[derive(Debug, Clone)]
pub struct TemporalPredicates {
    /// `temporal:profile` — relation linking a relation symbol to its temporal profile name.
    pub profile: SymbolId,
}

impl TemporalPredicates {
    /// Resolve all `temporal:` predicates, creating if needed.
    pub fn resolve(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            profile: engine.resolve_or_create_relation("temporal:profile")?,
        })
    }
}

/// Registry mapping relation SymbolIds to their temporal profiles.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalRegistry {
    profiles: HashMap<SymbolId, TemporalProfile>,
}

impl TemporalRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with default profiles for well-known relations.
    ///
    /// Call after resolving common predicates so their SymbolIds are known.
    pub fn with_defaults(engine: &Engine) -> AkhResult<Self> {
        let mut reg = Self::new();

        // is-a → Stable
        if let Ok(id) = engine.lookup_symbol("is-a") {
            reg.profiles.insert(id, TemporalProfile::Stable);
        }
        // has-part → Stable
        if let Ok(id) = engine.lookup_symbol("has-part") {
            reg.profiles.insert(id, TemporalProfile::Stable);
        }
        // located-at → Ephemeral (24 hours)
        if let Ok(id) = engine.lookup_symbol("located-at") {
            reg.profiles.insert(
                id,
                TemporalProfile::Ephemeral { ttl_secs: 86400 },
            );
        }

        Ok(reg)
    }

    /// Set the temporal profile for a relation.
    pub fn set_profile(
        &mut self,
        relation: SymbolId,
        profile: TemporalProfile,
    ) -> TemporalResult<()> {
        profile.validate()?;
        self.profiles.insert(relation, profile);
        Ok(())
    }

    /// Get the temporal profile for a relation.
    ///
    /// Returns `None` if no profile is set (treated as Stable by default).
    pub fn get_profile(&self, relation: SymbolId) -> Option<&TemporalProfile> {
        self.profiles.get(&relation)
    }

    /// Compute decayed confidence for a triple.
    ///
    /// If no profile is registered for the triple's predicate, returns
    /// the original confidence (Stable behavior).
    pub fn apply_decay(&self, triple: &Triple, query_time: u64) -> f32 {
        let profile = self
            .profiles
            .get(&triple.predicate)
            .cloned()
            .unwrap_or(TemporalProfile::Stable);
        apply_temporal_decay(&profile, triple.confidence, triple.timestamp, query_time)
    }

    /// Check if a triple has expired (confidence below threshold after decay).
    pub fn is_expired(&self, triple: &Triple, query_time: u64, min_confidence: f32) -> bool {
        self.apply_decay(triple, query_time) < min_confidence
    }

    /// Apply temporal decay to a list of triples, returning only non-expired ones
    /// with updated confidence values.
    pub fn filter_by_time(
        &self,
        triples: &[Triple],
        query_time: u64,
        min_confidence: f32,
    ) -> Vec<Triple> {
        triples
            .iter()
            .filter_map(|t| {
                let decayed = self.apply_decay(t, query_time);
                if decayed >= min_confidence {
                    let mut updated = t.clone();
                    updated.confidence = decayed;
                    Some(updated)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Number of registered profiles.
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_no_decay() {
        let conf = apply_temporal_decay(&TemporalProfile::Stable, 0.9, 1000, 999_999);
        assert!((conf - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn decaying_halves_at_half_life() {
        let half_life = 3600; // 1 hour
        let conf = apply_temporal_decay(
            &TemporalProfile::Decaying {
                half_life_secs: half_life,
            },
            1.0,
            0,
            half_life,
        );
        assert!((conf - 0.5).abs() < 0.01, "conf = {conf}");
    }

    #[test]
    fn decaying_quarters_at_two_half_lives() {
        let half_life = 3600;
        let conf = apply_temporal_decay(
            &TemporalProfile::Decaying {
                half_life_secs: half_life,
            },
            1.0,
            0,
            half_life * 2,
        );
        assert!((conf - 0.25).abs() < 0.01, "conf = {conf}");
    }

    #[test]
    fn ephemeral_before_ttl() {
        let conf = apply_temporal_decay(
            &TemporalProfile::Ephemeral { ttl_secs: 3600 },
            0.8,
            1000,
            1000 + 3599,
        );
        assert!((conf - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn ephemeral_after_ttl() {
        let conf = apply_temporal_decay(
            &TemporalProfile::Ephemeral { ttl_secs: 3600 },
            0.8,
            1000,
            1000 + 3600,
        );
        assert!(conf < f32::EPSILON, "conf = {conf}");
    }

    #[test]
    fn periodic_peaks_at_phase() {
        let period = 86400; // 1 day
        let phase = 0;
        // At t=0 (phase alignment), cos(0) = 1.0 → modulation = 1.0
        let conf = apply_temporal_decay(
            &TemporalProfile::Periodic {
                period_secs: period,
                phase_secs: phase,
            },
            1.0,
            0,
            0, // same time → no delta → but query_time <= triple_timestamp returns original
        );
        assert!((conf - 1.0).abs() < 0.01);
    }

    #[test]
    fn periodic_trough_at_half_period() {
        let period = 86400; // 1 day
        let phase = 0;
        // At half period from phase, cos(π) = -1 → modulation = 0.0
        let conf = apply_temporal_decay(
            &TemporalProfile::Periodic {
                period_secs: period,
                phase_secs: phase,
            },
            1.0,
            0,
            period / 2,
        );
        assert!(conf < 0.01, "conf = {conf}");
    }

    #[test]
    fn no_decay_when_query_before_assertion() {
        let conf = apply_temporal_decay(
            &TemporalProfile::Decaying {
                half_life_secs: 3600,
            },
            0.7,
            1000,
            500, // before assertion
        );
        assert!((conf - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn registry_default_stable() {
        let reg = TemporalRegistry::new();
        let triple = Triple::new(
            SymbolId::new(1).unwrap(),
            SymbolId::new(2).unwrap(),
            SymbolId::new(3).unwrap(),
        );
        let conf = reg.apply_decay(&triple, 999_999);
        assert!((conf - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn registry_set_and_apply() {
        let mut reg = TemporalRegistry::new();
        let rel = SymbolId::new(10).unwrap();
        reg.set_profile(rel, TemporalProfile::Ephemeral { ttl_secs: 100 })
            .unwrap();

        let mut triple = Triple::new(
            SymbolId::new(1).unwrap(),
            rel,
            SymbolId::new(3).unwrap(),
        );
        triple.timestamp = 1000;

        assert!(!reg.is_expired(&triple, 1050, 0.01));
        assert!(reg.is_expired(&triple, 1200, 0.01));
    }

    #[test]
    fn registry_filter_by_time() {
        let mut reg = TemporalRegistry::new();
        let rel = SymbolId::new(10).unwrap();
        reg.set_profile(rel, TemporalProfile::Ephemeral { ttl_secs: 100 })
            .unwrap();

        let mut t1 = Triple::new(SymbolId::new(1).unwrap(), rel, SymbolId::new(3).unwrap());
        t1.timestamp = 1000;
        let mut t2 = Triple::new(SymbolId::new(2).unwrap(), rel, SymbolId::new(4).unwrap());
        t2.timestamp = 900; // older

        let filtered = reg.filter_by_time(&[t1, t2], 1050, 0.01);
        assert_eq!(filtered.len(), 1); // only t1 is still valid
    }

    #[test]
    fn validate_zero_half_life() {
        let profile = TemporalProfile::Decaying { half_life_secs: 0 };
        assert!(profile.validate().is_err());
    }

    #[test]
    fn validate_zero_ttl() {
        let profile = TemporalProfile::Ephemeral { ttl_secs: 0 };
        assert!(profile.validate().is_err());
    }

    #[test]
    fn validate_zero_period() {
        let profile = TemporalProfile::Periodic {
            period_secs: 0,
            phase_secs: 0,
        };
        assert!(profile.validate().is_err());
    }

    #[test]
    fn display_profiles() {
        assert_eq!(format!("{}", TemporalProfile::Stable), "Stable");
        assert_eq!(
            format!(
                "{}",
                TemporalProfile::Decaying {
                    half_life_secs: 3600
                }
            ),
            "Decaying(half_life=3600s)"
        );
    }
}
