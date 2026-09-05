//! Phase 24c — Structured data perception.
//!
//! Ground numerical sensor readings and structured key-value data as symbolic
//! concepts.  Concept mappings link observed values to KG labels (e.g., a CPU
//! temperature of 90 °C → `"system:hot"` with somatic valence -0.6).
//!
//! No ML inference is required; the mapping is entirely rule-based.

use serde::{Deserialize, Serialize};

use super::{PerceptionConfig, PerceptionError, PerceptionResult, PerceptualAffect};

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// A named sensor reading with a physical value and unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    /// Sensor name, e.g. `"cpu_temperature"`, `"network_latency_ms"`.
    pub name: String,
    /// Measured value.
    pub value: f64,
    /// Unit of measurement, e.g. `"°C"`, `"ms"`, `"percent"`.
    pub unit: String,
    /// Optional key-value metadata (e.g. `[("core", "0")]`).
    pub metadata: Vec<(String, String)>,
}

impl SensorReading {
    pub fn new(name: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value,
            unit: unit.into(),
            metadata: Vec::new(),
        }
    }
}

/// A concept mapping rule.
///
/// When a sensor whose name starts with `name_prefix` has a value in
/// `[value_min, value_max]`, the rule fires and associates `concept_label`
/// with the given affective `valence`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptMappingRule {
    /// Prefix to match against sensor names (case-sensitive).
    pub name_prefix: String,
    /// Minimum value (inclusive).
    pub value_min: f64,
    /// Maximum value (exclusive).  Use `f64::MAX` for unbounded.
    pub value_max: f64,
    /// KG concept label to associate (resolved to a symbol by the caller).
    pub concept_label: String,
    /// Affective valence of this match (-1.0 = very unpleasant, 1.0 = pleasant).
    pub valence: f32,
}

/// A concept matched by a sensor reading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedConcept {
    /// KG concept label.
    pub concept_label: String,
    /// Affective valence of the match.
    pub valence: f32,
    /// The reading that triggered this match.
    pub sensor_name: String,
    /// The observed value.
    pub sensor_value: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in rules
// ─────────────────────────────────────────────────────────────────────────────

/// Built-in concept mapping rules for common system and environment metrics.
pub fn default_concept_mappings() -> Vec<ConceptMappingRule> {
    vec![
        // CPU temperature (°C)
        rule("cpu_temp", 0.0, 60.0, "system:cool", 0.2),
        rule("cpu_temp", 60.0, 80.0, "system:warm", -0.1),
        rule("cpu_temp", 80.0, f64::MAX, "system:hot", -0.6),
        // Network latency (ms)
        rule("network_latency", 0.0, 50.0, "network:fast", 0.3),
        rule("network_latency", 50.0, 200.0, "network:slow", -0.2),
        rule("network_latency", 200.0, f64::MAX, "network:degraded", -0.7),
        // Memory usage (percent)
        rule("memory_usage", 0.0, 70.0, "memory:available", 0.1),
        rule("memory_usage", 70.0, 90.0, "memory:pressured", -0.3),
        rule("memory_usage", 90.0, 101.0, "memory:critical", -0.8),
        // CPU usage (percent)
        rule("cpu_usage", 0.0, 50.0, "cpu:idle", 0.1),
        rule("cpu_usage", 50.0, 85.0, "cpu:busy", -0.1),
        rule("cpu_usage", 85.0, 101.0, "cpu:saturated", -0.5),
        // Disk usage (percent)
        rule("disk_usage", 0.0, 80.0, "disk:available", 0.1),
        rule("disk_usage", 80.0, 95.0, "disk:limited", -0.3),
        rule("disk_usage", 95.0, 101.0, "disk:critical", -0.8),
        // Time of day (hour in [0, 24))
        rule("time_of_day", 5.0, 12.0, "time:morning", 0.2),
        rule("time_of_day", 12.0, 17.0, "time:afternoon", 0.1),
        rule("time_of_day", 17.0, 21.0, "time:evening", 0.0),
        rule("time_of_day", 21.0, 29.0, "time:night", -0.1),
        rule("time_of_day", 0.0, 5.0, "time:night", -0.2),
        // Battery level (percent)
        rule("battery", 50.0, 101.0, "battery:charged", 0.2),
        rule("battery", 20.0, 50.0, "battery:low", -0.2),
        rule("battery", 0.0, 20.0, "battery:critical", -0.7),
    ]
}

fn rule(prefix: &str, min: f64, max: f64, label: &str, valence: f32) -> ConceptMappingRule {
    ConceptMappingRule {
        name_prefix: prefix.to_string(),
        value_min: min,
        value_max: max,
        concept_label: label.to_string(),
        valence,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract a feature vector from a set of sensor readings.
///
/// Each reading's value is sigmoid-normalised to [0, 1] and placed at its
/// index in the vector.  Values beyond `feature_dim` are ignored; missing
/// readings are represented as 0.5 (sigmoid(0), the neutral point).
///
/// # Errors
/// Returns [`PerceptionError::EmptyInput`] if `readings` is empty.
pub fn extract_structured_features(
    readings: &[SensorReading],
    config: &PerceptionConfig,
) -> PerceptionResult<(Vec<f32>, PerceptualAffect)> {
    if readings.is_empty() {
        return Err(PerceptionError::EmptyInput {
            modality: "structured".to_string(),
        });
    }

    let feature_dim = config.image_feature_dim; // reuse image dim as default slot

    // Build feature vector: each reading occupies one slot (sigmoid-normalised).
    let mut features: Vec<f32> = readings
        .iter()
        .map(|r| sigmoid(r.value as f32 / 100.0))
        .collect();

    features.resize(feature_dim, 0.5);

    // Compute affective state from matched concept valences.
    let rules = default_concept_mappings();
    let matched = match_concepts(readings, &rules);
    let affect = structured_affect(&matched);

    Ok((features, affect))
}

/// Match sensor readings against concept rules.
///
/// A rule fires when the reading name starts with `name_prefix` and the
/// value falls in `[value_min, value_max)`.
pub fn match_concepts(
    readings: &[SensorReading],
    rules: &[ConceptMappingRule],
) -> Vec<MatchedConcept> {
    let mut matches = Vec::new();

    for reading in readings {
        for rule in rules {
            if reading.name.starts_with(&rule.name_prefix)
                && reading.value >= rule.value_min
                && reading.value < rule.value_max
            {
                matches.push(MatchedConcept {
                    concept_label: rule.concept_label.clone(),
                    valence: rule.valence,
                    sensor_name: reading.name.clone(),
                    sensor_value: reading.value,
                });
            }
        }
    }

    matches
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute affect from matched concept valences.
fn structured_affect(matched: &[MatchedConcept]) -> PerceptualAffect {
    if matched.is_empty() {
        return PerceptualAffect::default();
    }

    let n = matched.len() as f32;
    let mean_valence = matched.iter().map(|m| m.valence).sum::<f32>() / n;

    // Arousal: the more negative matches (warnings), the higher the arousal.
    let warning_fraction = matched.iter().filter(|m| m.valence < -0.2).count() as f32 / n;
    let arousal = warning_fraction.clamp(0.0, 1.0);

    let novelty = (arousal * 0.4 + mean_valence.abs() * 0.3).clamp(0.0, 1.0);

    PerceptualAffect::new(mean_valence, arousal, novelty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_error() {
        let cfg = PerceptionConfig::default();
        let result = extract_structured_features(&[], &cfg);
        assert!(matches!(result, Err(PerceptionError::EmptyInput { .. })));
    }

    #[test]
    fn single_reading_returns_correct_dim() {
        let cfg = PerceptionConfig::default();
        let readings = vec![SensorReading::new("cpu_temperature", 72.0, "°C")];
        let (features, _) = extract_structured_features(&readings, &cfg).unwrap();
        assert_eq!(features.len(), cfg.image_feature_dim);
    }

    #[test]
    fn all_features_in_unit_interval() {
        let cfg = PerceptionConfig::default();
        let readings = vec![
            SensorReading::new("cpu_temperature", 85.0, "°C"),
            SensorReading::new("memory_usage", 55.0, "percent"),
            SensorReading::new("battery", 12.0, "percent"),
        ];
        let (features, _) = extract_structured_features(&readings, &cfg).unwrap();
        for &v in &features {
            assert!(v >= 0.0 && v <= 1.0, "feature out of [0,1]: {v}");
        }
    }

    #[test]
    fn hot_cpu_matches_concept() {
        let rules = default_concept_mappings();
        let readings = vec![SensorReading::new("cpu_temp", 90.0, "°C")];
        let matched = match_concepts(&readings, &rules);
        assert!(
            matched.iter().any(|m| m.concept_label == "system:hot"),
            "90 °C should match system:hot"
        );
    }

    #[test]
    fn cool_cpu_matches_concept() {
        let rules = default_concept_mappings();
        let readings = vec![SensorReading::new("cpu_temp", 40.0, "°C")];
        let matched = match_concepts(&readings, &rules);
        assert!(
            matched.iter().any(|m| m.concept_label == "system:cool"),
            "40 °C should match system:cool"
        );
    }

    #[test]
    fn overheating_triggers_negative_valence_affect() {
        let cfg = PerceptionConfig::default();
        let readings = vec![
            SensorReading::new("cpu_temp", 95.0, "°C"),
            SensorReading::new("memory_usage", 92.0, "percent"),
        ];
        let (_, affect) = extract_structured_features(&readings, &cfg).unwrap();
        assert!(
            affect.valence < 0.0,
            "overheating sensors should give negative valence, got {}",
            affect.valence
        );
    }

    #[test]
    fn no_rules_match_unknown_sensor() {
        let rules = default_concept_mappings();
        let readings = vec![SensorReading::new("xyzzy_sensor", 42.0, "units")];
        let matched = match_concepts(&readings, &rules);
        assert!(
            matched.is_empty(),
            "unknown sensor should not match any rule"
        );
    }

    #[test]
    fn fast_network_has_positive_valence() {
        let rules = default_concept_mappings();
        let readings = vec![SensorReading::new("network_latency", 10.0, "ms")];
        let matched = match_concepts(&readings, &rules);
        let valence = matched.iter().find(|m| m.concept_label == "network:fast");
        assert!(
            valence.is_some_and(|m| m.valence > 0.0),
            "fast network should map to positive valence"
        );
    }
}
