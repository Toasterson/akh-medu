//! Reproducible, adaptive exam planning for BabyLLM.
//!
//! A fixed holdout remains the immutable [`ExamTier::Sentinel`]. Development
//! signal comes from deterministic rolling and frontier plans. Exact exam
//! instances are always quarantined; a failed frontier may only schedule a
//! domain-separated sibling plan that passes an explicit near-duplicate gate.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Version of the serialized controller/plan contract.
pub const ADAPTIVE_EXAM_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExamTier {
    /// Small immutable set used for longitudinal release comparisons.
    Sentinel,
    /// Fresh deterministic probe at every evaluation checkpoint.
    RollingProbe,
    /// Capability/difficulty selected near the current learning boundary.
    Frontier,
    /// Generator-independent or human/real-data audit material.
    Audit,
}

impl ExamTier {
    fn seed_domain(self) -> &'static str {
        match self {
            Self::Sentinel => "sentinel",
            Self::RollingProbe => "rolling-probe",
            Self::Frontier => "frontier",
            Self::Audit => "audit",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningLevel {
    Green,
    Watch,
    Steer,
    Stop,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilitySpec {
    pub id: String,
    pub min_difficulty: u8,
    pub max_difficulty: u8,
    pub initial_difficulty: u8,
}

impl CapabilitySpec {
    pub fn new(id: impl Into<String>, min_difficulty: u8, max_difficulty: u8) -> Self {
        Self {
            id: id.into(),
            min_difficulty,
            max_difficulty,
            initial_difficulty: min_difficulty,
        }
    }

    pub fn with_initial_difficulty(mut self, difficulty: u8) -> Self {
        self.initial_difficulty = difficulty;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// Minimum observations in a family/window before a decision can steer.
    pub min_samples_per_window: u64,
    /// EWMA weight of the newest pass-rate observation.
    pub ewma_alpha: f64,
    /// Desired lower success boundary for frontier items.
    pub target_pass_low: f64,
    /// Desired upper success boundary for frontier items.
    pub target_pass_high: f64,
    /// Minimum significant window-to-window regression.
    pub regression_margin: f64,
    /// Precursor-rate threshold (using the interval lower bound).
    pub precursor_limit: f64,
    /// Catastrophic hard-failure/invalid-output threshold. The old serialized
    /// field name is accepted so saved v1 controller states remain readable.
    #[serde(alias = "catastrophic_precursor_limit")]
    pub catastrophic_failure_limit: f64,
    /// False until retrospective runs demonstrate useful precursor lead time.
    pub precursor_steering_enabled: bool,
    /// Maximum cumulative sentinel regression from its first baseline.
    pub sentinel_regression_budget: f64,
    /// Consecutive significant windows required before steering.
    pub steer_windows: u32,
    /// Consecutive significant windows required before stopping.
    pub stop_windows: u32,
    /// Normal critical value used for Wilson intervals (1.96 = 95%).
    pub confidence_z: f64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            min_samples_per_window: 200,
            ewma_alpha: 0.4,
            target_pass_low: 0.35,
            target_pass_high: 0.75,
            regression_margin: 0.08,
            precursor_limit: 0.15,
            catastrophic_failure_limit: 0.05,
            precursor_steering_enabled: false,
            sentinel_regression_budget: 0.03,
            steer_windows: 2,
            stop_windows: 2,
            confidence_z: 1.959_963_984_540_054,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct WilsonInterval {
    pub lower: f64,
    pub estimate: f64,
    pub upper: f64,
}

/// Wilson score interval for a binomial proportion.
pub fn wilson_interval(successes: u64, samples: u64, z: f64) -> WilsonInterval {
    if samples == 0 {
        return WilsonInterval {
            lower: 0.0,
            estimate: 0.0,
            upper: 1.0,
        };
    }
    let n = samples as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / denominator;
    let half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    WilsonInterval {
        lower: if successes == 0 {
            0.0
        } else {
            (centre - half).max(0.0)
        },
        estimate: p,
        upper: if successes == samples {
            1.0
        } else {
            (centre + half).min(1.0)
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointRef {
    pub step: u64,
    /// Content hash or immutable artifact identifier; never a mutable label.
    pub artifact_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExamPlan {
    pub schema_version: u32,
    pub generator_version: String,
    pub run_id: String,
    pub checkpoint: CheckpointRef,
    pub tier: ExamTier,
    pub family: String,
    pub difficulty: u8,
    pub seed: u64,
    pub sample_count: u64,
}

impl ExamPlan {
    /// Stable identifier used by quarantine and sibling manifests.
    pub fn fingerprint(&self) -> String {
        let hash = if self.tier == ExamTier::Sentinel {
            // A sentinel identifies one immutable case set for the run. The
            // checkpoint still records when it was scored, but cannot alter
            // the item seed or quarantine identity.
            stable_seed(&[
                &self.schema_version.to_string(),
                &self.generator_version,
                &self.run_id,
                self.tier.seed_domain(),
                &self.family,
                &self.difficulty.to_string(),
                &self.seed.to_string(),
                &self.sample_count.to_string(),
            ])
        } else {
            stable_seed(&[
                &self.schema_version.to_string(),
                &self.generator_version,
                &self.run_id,
                &self.checkpoint.step.to_string(),
                &self.checkpoint.artifact_hash,
                self.tier.seed_domain(),
                &self.family,
                &self.difficulty.to_string(),
                &self.seed.to_string(),
                &self.sample_count.to_string(),
            ])
        };
        format!("exam-fnv1a64-{hash:016x}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExamObservation {
    pub checkpoint: CheckpointRef,
    pub tier: ExamTier,
    pub family: String,
    pub difficulty: u8,
    pub sample_count: u64,
    pub pass_count: u64,
    /// Count of near-miss/invalid/metamorphic precursor failures.
    pub precursor_count: u64,
    /// Count of hard outcome failures (blank/invalid/non-finite) that must be
    /// able to stop a run even while predictive precursors remain shadow-only.
    #[serde(default)]
    pub catastrophic_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ControllerDecision {
    pub family: String,
    /// Evaluation lane that produced this decision. Older first-probe step
    /// artifacts omitted it and were rolling probes.
    #[serde(default = "default_decision_tier")]
    pub tier: ExamTier,
    pub warning: WarningLevel,
    pub reasons: Vec<String>,
    pub statistically_eligible: bool,
    pub pass: WilsonInterval,
    pub precursor: WilsonInterval,
    pub catastrophic: WilsonInterval,
    pub precursor_steering_enabled: bool,
    pub ewma_pass_rate: f64,
    pub learning_progress: f64,
    pub recommended_difficulty: u8,
    pub priority: f64,
}

fn default_decision_tier() -> ExamTier {
    ExamTier::RollingProbe
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FamilyState {
    pub spec: CapabilitySpec,
    /// Difficulty permanently assigned to this family's sentinel cases.
    /// `None` is accepted only for backward compatibility with v1 state
    /// serialized before this field existed; it resolves to the capability's
    /// original difficulty, never the mutable frontier difficulty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sentinel_difficulty: Option<u8>,
    /// Mutable difficulty used by rolling/frontier plans.
    pub difficulty: u8,
    pub sentinel_baseline: Option<WilsonInterval>,
    /// Trend and confirmation state is isolated by evaluation lane. Mixing a
    /// sentinel observation into a frontier history can manufacture a false
    /// regression even when neither lane changed.
    pub tier_history: BTreeMap<ExamTier, TierHistory>,
    pub priority: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TierHistory {
    pub ewma_pass_rate: Option<f64>,
    pub last_pass_interval: Option<WilsonInterval>,
    pub observations: u64,
    pub steer_streak: u32,
    pub stop_streak: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveExamController {
    pub schema_version: u32,
    pub generator_version: String,
    pub run_id: String,
    pub base_seed: u64,
    pub config: ControllerConfig,
    pub families: BTreeMap<String, FamilyState>,
}

#[derive(Debug, thiserror::Error)]
pub enum AdaptiveExamError {
    #[error("adaptive exam configuration is invalid: {0}")]
    InvalidConfig(String),
    #[error("capability family {0:?} is not registered")]
    UnknownFamily(String),
    #[error("capability {id:?} has invalid difficulty range {min}..={max} with initial {initial}")]
    InvalidCapability {
        id: String,
        min: u8,
        max: u8,
        initial: u8,
    },
    #[error(
        "observation counts are invalid: pass={pass}, precursor={precursor}, catastrophic={catastrophic}, samples={samples}"
    )]
    InvalidCounts {
        pass: u64,
        precursor: u64,
        catastrophic: u64,
        samples: u64,
    },
    #[error("training siblings may only be derived from frontier plans, not {0:?}")]
    ProtectedExamTier(ExamTier),
    #[error("sibling separation rejected: {0}")]
    SiblingTooClose(String),
}

impl AdaptiveExamController {
    pub fn new(
        generator_version: impl Into<String>,
        run_id: impl Into<String>,
        base_seed: u64,
        config: ControllerConfig,
        capabilities: impl IntoIterator<Item = CapabilitySpec>,
    ) -> Result<Self, AdaptiveExamError> {
        let generator_version = generator_version.into();
        let run_id = run_id.into();
        if generator_version.trim().is_empty() || run_id.trim().is_empty() {
            return Err(AdaptiveExamError::InvalidConfig(
                "generator version and run id must not be blank".into(),
            ));
        }
        if config.min_samples_per_window == 0
            || !config.ewma_alpha.is_finite()
            || !(0.0..=1.0).contains(&config.ewma_alpha)
            || !config.target_pass_low.is_finite()
            || !config.target_pass_high.is_finite()
            || !(0.0..=1.0).contains(&config.target_pass_low)
            || !(0.0..=1.0).contains(&config.target_pass_high)
            || config.target_pass_low > config.target_pass_high
            || !config.regression_margin.is_finite()
            || !(0.0..=1.0).contains(&config.regression_margin)
            || !config.precursor_limit.is_finite()
            || !(0.0..=1.0).contains(&config.precursor_limit)
            || !config.catastrophic_failure_limit.is_finite()
            || !(0.0..=1.0).contains(&config.catastrophic_failure_limit)
            || !config.sentinel_regression_budget.is_finite()
            || !(0.0..=1.0).contains(&config.sentinel_regression_budget)
            || config.steer_windows == 0
            || config.stop_windows == 0
            || !config.confidence_z.is_finite()
            || config.confidence_z <= 0.0
        {
            return Err(AdaptiveExamError::InvalidConfig(
                "sample counts must be positive, rates must be finite probabilities, warning windows must be positive, and confidence_z must be finite and positive".into(),
            ));
        }
        let mut families = BTreeMap::new();
        for spec in capabilities {
            if spec.min_difficulty > spec.max_difficulty
                || !(spec.min_difficulty..=spec.max_difficulty).contains(&spec.initial_difficulty)
            {
                return Err(AdaptiveExamError::InvalidCapability {
                    id: spec.id,
                    min: spec.min_difficulty,
                    max: spec.max_difficulty,
                    initial: spec.initial_difficulty,
                });
            }
            let id = spec.id.clone();
            if families.contains_key(&id) {
                return Err(AdaptiveExamError::InvalidConfig(format!(
                    "capability family {id:?} was registered more than once"
                )));
            }
            families.insert(
                id,
                FamilyState {
                    sentinel_difficulty: Some(spec.initial_difficulty),
                    difficulty: spec.initial_difficulty,
                    spec,
                    sentinel_baseline: None,
                    tier_history: BTreeMap::new(),
                    priority: 1.0,
                },
            );
        }
        if families.is_empty() {
            return Err(AdaptiveExamError::InvalidConfig(
                "at least one capability family is required".into(),
            ));
        }
        Ok(Self {
            schema_version: ADAPTIVE_EXAM_SCHEMA_VERSION,
            generator_version,
            run_id,
            base_seed,
            config,
            families,
        })
    }

    /// Record one family/window and return the bounded control decision.
    pub fn observe(
        &mut self,
        observation: ExamObservation,
    ) -> Result<ControllerDecision, AdaptiveExamError> {
        if observation.pass_count > observation.sample_count
            || observation.precursor_count > observation.sample_count
            || observation.catastrophic_count > observation.sample_count
            || observation.pass_count
                > observation
                    .sample_count
                    .saturating_sub(observation.catastrophic_count)
        {
            return Err(AdaptiveExamError::InvalidCounts {
                pass: observation.pass_count,
                precursor: observation.precursor_count,
                catastrophic: observation.catastrophic_count,
                samples: observation.sample_count,
            });
        }
        let state = self
            .families
            .get_mut(&observation.family)
            .ok_or_else(|| AdaptiveExamError::UnknownFamily(observation.family.clone()))?;

        let pass = wilson_interval(
            observation.pass_count,
            observation.sample_count,
            self.config.confidence_z,
        );
        let precursor = wilson_interval(
            observation.precursor_count,
            observation.sample_count,
            self.config.confidence_z,
        );
        let catastrophic = wilson_interval(
            observation.catastrophic_count,
            observation.sample_count,
            self.config.confidence_z,
        );
        let eligible = observation.sample_count >= self.config.min_samples_per_window;
        let history = state.tier_history.entry(observation.tier).or_default();
        let previous_interval = history.last_pass_interval;
        let old_ewma = history.ewma_pass_rate.unwrap_or(pass.estimate);
        let ewma = if eligible {
            match history.ewma_pass_rate {
                Some(old) => {
                    self.config.ewma_alpha * pass.estimate + (1.0 - self.config.ewma_alpha) * old
                }
                None => pass.estimate,
            }
        } else {
            old_ewma
        };
        let learning_progress = if eligible { ewma - old_ewma } else { 0.0 };

        if observation.tier == ExamTier::Sentinel && state.sentinel_baseline.is_none() && eligible {
            state.sentinel_baseline = Some(pass);
        }

        let significant_regression = previous_interval
            .is_some_and(|previous| previous.lower - pass.upper > self.config.regression_margin);
        let precursor_alarm = precursor.lower > self.config.precursor_limit;
        let catastrophic_alarm = catastrophic.lower > self.config.catastrophic_failure_limit;
        let sentinel_regression = observation.tier == ExamTier::Sentinel
            && state.sentinel_baseline.is_some_and(|baseline| {
                baseline.lower - pass.upper > self.config.sentinel_regression_budget
            });

        let active_precursor_alarm = self.config.precursor_steering_enabled && precursor_alarm;
        let steer_candidate = eligible && (significant_regression || active_precursor_alarm);
        let stop_candidate = eligible && (sentinel_regression || catastrophic_alarm);
        history.steer_streak = if steer_candidate {
            history.steer_streak.saturating_add(1)
        } else {
            0
        };
        history.stop_streak = if stop_candidate {
            history.stop_streak.saturating_add(1)
        } else {
            0
        };

        let mut reasons = Vec::new();
        let warning = if !eligible {
            reasons.push(format!(
                "{} samples is below the {}-sample decision floor",
                observation.sample_count, self.config.min_samples_per_window
            ));
            WarningLevel::Watch
        } else if history.stop_streak >= self.config.stop_windows {
            if sentinel_regression {
                reasons.push("sentinel regression exceeds the cumulative budget".into());
            }
            if catastrophic_alarm {
                reasons.push("catastrophic hard-failure rate is significant".into());
            }
            WarningLevel::Stop
        } else if history.steer_streak >= self.config.steer_windows {
            if significant_regression {
                reasons
                    .push("pass-rate regression is outside overlapping confidence bounds".into());
            }
            if active_precursor_alarm {
                reasons.push("precursor rate is significantly above its limit".into());
            }
            WarningLevel::Steer
        } else if steer_candidate || stop_candidate {
            reasons.push("candidate warning awaits consecutive-window confirmation".into());
            WarningLevel::Watch
        } else if eligible && precursor_alarm && !self.config.precursor_steering_enabled {
            reasons.push("precursor alarm recorded in shadow mode; steering is disabled".into());
            WarningLevel::Watch
        } else if pass.upper < self.config.target_pass_low {
            reasons.push("family is significantly below the frontier target band".into());
            WarningLevel::Watch
        } else {
            WarningLevel::Green
        };

        if eligible && observation.tier == ExamTier::Frontier {
            if pass.lower > self.config.target_pass_high {
                state.difficulty = state
                    .difficulty
                    .saturating_add(1)
                    .min(state.spec.max_difficulty);
            } else if pass.upper < self.config.target_pass_low {
                state.difficulty = state
                    .difficulty
                    .saturating_sub(1)
                    .max(state.spec.min_difficulty);
            }
        }

        let boundary_value = if (self.config.target_pass_low..=self.config.target_pass_high)
            .contains(&pass.estimate)
        {
            1.0
        } else {
            0.25
        };
        if eligible {
            state.priority = boundary_value
                + if self.config.precursor_steering_enabled {
                    precursor.estimate
                } else {
                    0.0
                }
                + catastrophic.estimate
                + learning_progress.abs() * 2.0
                + if warning >= WarningLevel::Steer {
                    1.0
                } else {
                    0.0
                };
            history.ewma_pass_rate = Some(ewma);
            history.last_pass_interval = Some(pass);
        }
        history.observations += 1;

        Ok(ControllerDecision {
            family: observation.family,
            tier: observation.tier,
            warning,
            reasons,
            statistically_eligible: eligible,
            pass,
            precursor,
            catastrophic,
            precursor_steering_enabled: self.config.precursor_steering_enabled,
            ewma_pass_rate: ewma,
            learning_progress,
            recommended_difficulty: state.difficulty,
            priority: state.priority,
        })
    }

    /// Deterministically select the highest-priority family and plan its next
    /// exam. A stable lexical tie-break makes serialized-state replay exact.
    pub fn next_plan(
        &self,
        tier: ExamTier,
        checkpoint: CheckpointRef,
        sample_count: u64,
    ) -> Result<ExamPlan, AdaptiveExamError> {
        let state = self
            .families
            .values()
            .max_by(|left, right| {
                left.priority
                    .total_cmp(&right.priority)
                    .then_with(|| right.spec.id.cmp(&left.spec.id))
            })
            .ok_or_else(|| AdaptiveExamError::UnknownFamily("<no capabilities>".into()))?;
        self.plan_for_family(tier, checkpoint, &state.spec.id, sample_count)
    }

    pub fn plan_for_family(
        &self,
        tier: ExamTier,
        checkpoint: CheckpointRef,
        family: &str,
        sample_count: u64,
    ) -> Result<ExamPlan, AdaptiveExamError> {
        let state = self
            .families
            .get(family)
            .ok_or_else(|| AdaptiveExamError::UnknownFamily(family.into()))?;
        let plan_difficulty = if tier == ExamTier::Sentinel {
            state
                .sentinel_difficulty
                .unwrap_or(state.spec.initial_difficulty)
        } else {
            state.difficulty
        };
        let seed = if tier == ExamTier::Sentinel {
            stable_seed(&[
                "babyllm-exam-v1",
                &self.base_seed.to_string(),
                &self.generator_version,
                &self.run_id,
                tier.seed_domain(),
                family,
                &plan_difficulty.to_string(),
            ])
        } else {
            stable_seed(&[
                "babyllm-exam-v1",
                &self.base_seed.to_string(),
                &self.generator_version,
                &self.run_id,
                &checkpoint.step.to_string(),
                &checkpoint.artifact_hash,
                tier.seed_domain(),
                family,
                &plan_difficulty.to_string(),
            ])
        };
        Ok(ExamPlan {
            schema_version: self.schema_version,
            generator_version: self.generator_version.clone(),
            run_id: self.run_id.clone(),
            checkpoint,
            tier,
            family: family.into(),
            difficulty: plan_difficulty,
            seed,
            sample_count,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiblingSeparationPolicy {
    /// Maximum character n-gram Jaccard similarity to any quarantined text.
    pub max_transcript_ngram_jaccard: f64,
    /// Minimum perceptual-hash Hamming distance at the closest transcript
    /// neighborhood. Exact pixel overlap is rejected separately.
    pub min_image_hash_hamming_distance: u32,
    /// Vision families must provide the image-distance evidence.
    pub require_image_distance: bool,
}

impl Default for SiblingSeparationPolicy {
    fn default() -> Self {
        Self {
            max_transcript_ngram_jaccard: 0.80,
            min_image_hash_hamming_distance: 10,
            require_image_distance: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingSiblingPlan {
    pub schema_version: u32,
    pub generator_version: String,
    pub run_id: String,
    pub family: String,
    pub difficulty: u8,
    pub seed: u64,
    pub sample_count: u64,
    pub source_exam_fingerprint: String,
    pub required_policy: SiblingSeparationPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiblingSeparationEvidence {
    /// At least one generator axis must be held out: content bank, font,
    /// layout engine, corruption range, or renderer.
    pub held_out_axes: Vec<String>,
    pub max_transcript_ngram_jaccard: f64,
    /// Image distance for clean pairs having maximum transcript similarity.
    /// This avoids unrelated blank layout regions dominating an all-pairs
    /// minimum.
    pub min_image_hash_hamming_distance: Option<u32>,
}

impl TrainingSiblingPlan {
    pub fn validate_separation(
        &self,
        evidence: &SiblingSeparationEvidence,
    ) -> Result<(), AdaptiveExamError> {
        validate_sibling_separation(&self.required_policy, evidence)
    }
}

pub fn validate_sibling_separation(
    policy: &SiblingSeparationPolicy,
    evidence: &SiblingSeparationEvidence,
) -> Result<(), AdaptiveExamError> {
    if evidence.held_out_axes.is_empty() {
        return Err(AdaptiveExamError::SiblingTooClose(
            "no held-out generator axis was demonstrated".into(),
        ));
    }
    if evidence.max_transcript_ngram_jaccard > policy.max_transcript_ngram_jaccard {
        return Err(AdaptiveExamError::SiblingTooClose(format!(
            "transcript n-gram Jaccard {:.3} exceeds {:.3}",
            evidence.max_transcript_ngram_jaccard, policy.max_transcript_ngram_jaccard
        )));
    }
    match evidence.min_image_hash_hamming_distance {
        Some(distance) if distance < policy.min_image_hash_hamming_distance => {
            return Err(AdaptiveExamError::SiblingTooClose(format!(
                "image hash distance {distance} is below {}",
                policy.min_image_hash_hamming_distance
            )));
        }
        None if policy.require_image_distance => {
            return Err(AdaptiveExamError::SiblingTooClose(
                "image hash distance evidence is required".into(),
            ));
        }
        _ => {}
    }
    Ok(())
}

/// Derive a training-family seed in a separate namespace. The exact frontier
/// case is still quarantined; callers must validate generated siblings with
/// [`TrainingSiblingPlan::validate_separation`] before admission.
pub fn derive_training_sibling(
    exam: &ExamPlan,
    sample_count: u64,
    policy: SiblingSeparationPolicy,
) -> Result<TrainingSiblingPlan, AdaptiveExamError> {
    if exam.tier != ExamTier::Frontier {
        return Err(AdaptiveExamError::ProtectedExamTier(exam.tier));
    }
    let fingerprint = exam.fingerprint();
    let seed = stable_seed(&[
        "babyllm-training-sibling-v1",
        &exam.seed.to_string(),
        &fingerprint,
        &exam.family,
    ]);
    Ok(TrainingSiblingPlan {
        schema_version: exam.schema_version,
        generator_version: exam.generator_version.clone(),
        run_id: exam.run_id.clone(),
        family: exam.family.clone(),
        difficulty: exam.difficulty,
        seed,
        sample_count,
        source_exam_fingerprint: fingerprint,
        required_policy: policy,
    })
}

/// Character n-gram Jaccard used by the sibling leakage gate.
pub fn character_ngram_jaccard(left: &str, right: &str, n: usize) -> f64 {
    fn grams(value: &str, n: usize) -> BTreeSet<String> {
        let chars: Vec<char> = value.to_lowercase().chars().collect();
        if chars.is_empty() {
            return BTreeSet::new();
        }
        if chars.len() <= n.max(1) {
            return [chars.iter().collect()].into_iter().collect();
        }
        chars
            .windows(n.max(1))
            .map(|window| window.iter().collect())
            .collect()
    }
    let left = grams(left, n);
    let right = grams(right, n);
    let union = left.union(&right).count();
    if union == 0 {
        return 1.0;
    }
    left.intersection(&right).count() as f64 / union as f64
}

pub fn perceptual_hash_hamming(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}

/// Benjamini-Hochberg false-discovery control for a family matrix. The output
/// preserves input order and marks hypotheses accepted at `alpha`.
pub fn benjamini_hochberg(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let mut indexed: Vec<(usize, f64)> = p_values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, p)| (index, p.clamp(0.0, 1.0)))
        .collect();
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1));
    let m = indexed.len() as f64;
    let cutoff_rank = indexed
        .iter()
        .enumerate()
        .filter(|(rank, (_, p))| *p <= alpha * (*rank as f64 + 1.0) / m)
        .map(|(rank, _)| rank)
        .max();
    let mut accepted = vec![false; indexed.len()];
    if let Some(cutoff_rank) = cutoff_rank {
        for (index, _) in indexed.into_iter().take(cutoff_rank + 1) {
            accepted[index] = true;
        }
    }
    accepted
}

fn stable_seed(parts: &[&str]) -> u64 {
    // FNV-1a is deliberately simple and cross-platform. This is a replay key,
    // not a cryptographic secret or content-integrity hash.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for part in parts {
        for byte in part.as_bytes().iter().copied().chain([0xff]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> AdaptiveExamController {
        AdaptiveExamController::new(
            "test-generator-v1",
            "run-42",
            42,
            ControllerConfig::default(),
            [
                CapabilitySpec::new("reason.chain.depth", 0, 5).with_initial_difficulty(2),
                CapabilitySpec::new("vision.text.line", 0, 5).with_initial_difficulty(2),
            ],
        )
        .unwrap()
    }

    #[test]
    fn decisions_record_their_lane_and_old_probe_outputs_default_to_rolling() {
        let mut controller = controller();
        let decision = controller
            .observe(observation(
                1,
                ExamTier::Frontier,
                "vision.text.line",
                120,
                0,
            ))
            .unwrap();
        assert_eq!(decision.tier, ExamTier::Frontier);

        let mut legacy = serde_json::to_value(decision).unwrap();
        legacy.as_object_mut().unwrap().remove("tier");
        let replayed: ControllerDecision = serde_json::from_value(legacy).unwrap();
        assert_eq!(replayed.tier, ExamTier::RollingProbe);
    }

    #[test]
    fn invalid_controller_configuration_and_duplicate_families_fail_closed() {
        let mut invalid = ControllerConfig::default();
        invalid.min_samples_per_window = 0;
        assert!(matches!(
            AdaptiveExamController::new(
                "test-generator-v1",
                "run-42",
                42,
                invalid,
                [CapabilitySpec::new("reason.chain.depth", 0, 5)]
            ),
            Err(AdaptiveExamError::InvalidConfig(_))
        ));
        assert!(matches!(
            AdaptiveExamController::new(
                "test-generator-v1",
                "run-42",
                42,
                ControllerConfig::default(),
                [
                    CapabilitySpec::new("reason.chain.depth", 0, 5),
                    CapabilitySpec::new("reason.chain.depth", 0, 5),
                ]
            ),
            Err(AdaptiveExamError::InvalidConfig(_))
        ));
        assert!(matches!(
            AdaptiveExamController::new(
                "test-generator-v1",
                "run-42",
                42,
                ControllerConfig::default(),
                [],
            ),
            Err(AdaptiveExamError::InvalidConfig(_))
        ));
    }

    fn checkpoint(step: u64) -> CheckpointRef {
        CheckpointRef {
            step,
            artifact_hash: format!("sha256:{step:064x}"),
        }
    }

    fn observation(
        step: u64,
        tier: ExamTier,
        family: &str,
        pass: u64,
        precursor: u64,
    ) -> ExamObservation {
        ExamObservation {
            checkpoint: checkpoint(step),
            tier,
            family: family.into(),
            difficulty: 2,
            sample_count: 400,
            pass_count: pass,
            precursor_count: precursor,
            catastrophic_count: 0,
        }
    }

    #[test]
    fn serialized_state_replays_the_same_plan() {
        let controller = controller();
        let checkpoint = checkpoint(100);
        let first = controller
            .next_plan(ExamTier::Frontier, checkpoint.clone(), 200)
            .unwrap();
        let json = serde_json::to_string(&controller).unwrap();
        let replay: AdaptiveExamController = serde_json::from_str(&json).unwrap();
        let second = replay
            .next_plan(ExamTier::Frontier, checkpoint, 200)
            .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn legacy_v1_counts_and_catastrophic_limit_remain_readable() {
        let mut controller_value = serde_json::to_value(controller()).unwrap();
        let config = controller_value["config"].as_object_mut().unwrap();
        let limit = config.remove("catastrophic_failure_limit").unwrap();
        config.insert("catastrophic_precursor_limit".into(), limit);
        let family = controller_value["families"]["vision.text.line"]
            .as_object_mut()
            .unwrap();
        family.remove("sentinel_difficulty");
        family.insert("difficulty".into(), serde_json::json!(4));
        let restored: AdaptiveExamController = serde_json::from_value(controller_value).unwrap();
        assert_eq!(restored.config.catastrophic_failure_limit, 0.05);
        assert_eq!(
            restored
                .plan_for_family(ExamTier::Sentinel, checkpoint(2), "vision.text.line", 200,)
                .unwrap()
                .difficulty,
            2
        );
        assert_eq!(
            restored
                .plan_for_family(ExamTier::Frontier, checkpoint(2), "vision.text.line", 200,)
                .unwrap()
                .difficulty,
            4
        );

        let observation = serde_json::json!({
            "checkpoint": {"step": 1, "artifact_hash": "sha256:legacy"},
            "tier": "frontier",
            "family": "vision.text.line",
            "difficulty": 2,
            "sample_count": 200,
            "pass_count": 150,
            "precursor_count": 10
        });
        let restored: ExamObservation = serde_json::from_value(observation).unwrap();
        assert_eq!(restored.catastrophic_count, 0);
    }

    #[test]
    fn sentinel_case_identity_is_checkpoint_invariant() {
        let controller = controller();
        let first = controller
            .next_plan(ExamTier::Sentinel, checkpoint(100), 200)
            .unwrap();
        let second = controller
            .next_plan(ExamTier::Sentinel, checkpoint(200), 200)
            .unwrap();
        assert_eq!(first.seed, second.seed);
        assert_eq!(first.fingerprint(), second.fingerprint());

        let rolling_first = controller
            .next_plan(ExamTier::RollingProbe, checkpoint(100), 200)
            .unwrap();
        let rolling_second = controller
            .next_plan(ExamTier::RollingProbe, checkpoint(200), 200)
            .unwrap();
        assert_ne!(rolling_first.seed, rolling_second.seed);
        assert_ne!(rolling_first.fingerprint(), rolling_second.fingerprint());
    }

    #[test]
    fn sentinel_case_identity_survives_frontier_difficulty_changes() {
        let mut controller = controller();
        let first = controller
            .plan_for_family(ExamTier::Sentinel, checkpoint(100), "vision.text.line", 200)
            .unwrap();

        let frontier = controller
            .observe(observation(
                150,
                ExamTier::Frontier,
                "vision.text.line",
                390,
                0,
            ))
            .unwrap();
        assert_eq!(frontier.recommended_difficulty, 3);

        let second = controller
            .plan_for_family(ExamTier::Sentinel, checkpoint(200), "vision.text.line", 200)
            .unwrap();
        assert_eq!(first.difficulty, 2);
        assert_eq!(second.difficulty, 2);
        assert_eq!(first.seed, second.seed);
        assert_eq!(first.fingerprint(), second.fingerprint());
    }

    #[test]
    fn catastrophic_outcomes_stop_even_while_precursors_are_shadow_only() {
        let mut controller = controller();
        assert!(!controller.config.precursor_steering_enabled);
        let mut first = observation(10, ExamTier::Frontier, "vision.text.line", 300, 40);
        first.catastrophic_count = 40;
        let mut second = first.clone();
        second.checkpoint = checkpoint(20);

        assert_eq!(
            controller.observe(first).unwrap().warning,
            WarningLevel::Watch
        );
        let decision = controller.observe(second).unwrap();
        assert_eq!(decision.warning, WarningLevel::Stop);
        assert!(decision.catastrophic.lower > 0.05);
    }

    #[test]
    fn passing_and_catastrophic_counts_cannot_overlap() {
        let mut controller = controller();
        let mut invalid = observation(10, ExamTier::Frontier, "vision.text.line", 390, 20);
        invalid.catastrophic_count = 20;
        assert!(matches!(
            controller.observe(invalid),
            Err(AdaptiveExamError::InvalidCounts { .. })
        ));
    }

    #[test]
    fn under_sampled_window_does_not_change_trend_state() {
        let mut controller = controller();
        let family = "vision.text.line";
        let before = controller.families[family].clone();
        let mut tiny = observation(10, ExamTier::Frontier, family, 0, 20);
        tiny.sample_count = 20;
        tiny.pass_count = 0;
        tiny.precursor_count = 20;
        let decision = controller.observe(tiny).unwrap();

        assert!(!decision.statistically_eligible);
        let after = &controller.families[family];
        assert_eq!(after.priority, before.priority);
        assert_eq!(after.difficulty, before.difficulty);
        let history = &after.tier_history[&ExamTier::Frontier];
        assert_eq!(history.ewma_pass_rate, None);
        assert_eq!(history.last_pass_interval, None);
        assert_eq!(history.steer_streak, 0);
        assert_eq!(history.stop_streak, 0);
    }

    #[test]
    fn significant_precursor_steers_before_accuracy_collapses() {
        let mut controller = controller();
        controller.config.precursor_steering_enabled = true;
        let first = controller
            .observe(observation(
                10,
                ExamTier::Frontier,
                "vision.text.line",
                360,
                100,
            ))
            .unwrap();
        assert_eq!(first.warning, WarningLevel::Watch);
        let second = controller
            .observe(observation(
                20,
                ExamTier::Frontier,
                "vision.text.line",
                350,
                105,
            ))
            .unwrap();
        assert_eq!(second.warning, WarningLevel::Steer);
        assert!(second.pass.estimate > 0.8);
    }

    #[test]
    fn repeated_significant_sentinel_regression_stops() {
        let mut controller = controller();
        controller
            .observe(observation(
                0,
                ExamTier::Sentinel,
                "reason.chain.depth",
                380,
                0,
            ))
            .unwrap();
        let first = controller
            .observe(observation(
                10,
                ExamTier::Sentinel,
                "reason.chain.depth",
                300,
                0,
            ))
            .unwrap();
        assert_eq!(first.warning, WarningLevel::Watch);
        let second = controller
            .observe(observation(
                20,
                ExamTier::Sentinel,
                "reason.chain.depth",
                295,
                0,
            ))
            .unwrap();
        assert_eq!(second.warning, WarningLevel::Stop);
    }

    #[test]
    fn difficulty_tracks_the_confident_frontier() {
        let mut controller = controller();
        let mastered = controller
            .observe(observation(
                10,
                ExamTier::Frontier,
                "vision.text.line",
                390,
                0,
            ))
            .unwrap();
        assert_eq!(mastered.recommended_difficulty, 3);

        let collapsed = controller
            .observe(observation(
                20,
                ExamTier::Frontier,
                "vision.text.line",
                20,
                0,
            ))
            .unwrap();
        assert_eq!(collapsed.recommended_difficulty, 2);
    }

    #[test]
    fn frontier_sibling_uses_new_seed_and_requires_distance_evidence() {
        let exam = controller()
            .next_plan(ExamTier::Frontier, checkpoint(10), 200)
            .unwrap();
        let sibling =
            derive_training_sibling(&exam, 1_000, SiblingSeparationPolicy::default()).unwrap();
        assert_ne!(exam.seed, sibling.seed);
        assert!(
            sibling
                .validate_separation(&SiblingSeparationEvidence {
                    held_out_axes: vec!["content_bank".into(), "font".into()],
                    max_transcript_ngram_jaccard: 0.4,
                    min_image_hash_hamming_distance: Some(18),
                })
                .is_ok()
        );
        assert!(
            sibling
                .validate_separation(&SiblingSeparationEvidence {
                    held_out_axes: vec![],
                    max_transcript_ngram_jaccard: 0.95,
                    min_image_hash_hamming_distance: Some(1),
                })
                .is_err()
        );
    }

    #[test]
    fn sentinel_and_audit_cannot_schedule_training_siblings() {
        let controller = controller();
        for tier in [ExamTier::Sentinel, ExamTier::RollingProbe, ExamTier::Audit] {
            let exam = controller.next_plan(tier, checkpoint(10), 200).unwrap();
            assert!(
                derive_training_sibling(&exam, 100, SiblingSeparationPolicy::default()).is_err()
            );
        }
    }

    #[test]
    fn multiple_comparison_filter_preserves_input_order() {
        let accepted = benjamini_hochberg(&[0.04, 0.001, 0.02, 0.8], 0.05);
        assert_eq!(accepted, vec![false, true, true, false]);
    }

    #[test]
    fn ngram_and_image_distance_detect_near_duplicates() {
        assert!(character_ngram_jaccard("total CHF 42.70", "total CHF 42.71", 3) > 0.7);
        assert!(character_ngram_jaccard("total CHF", "api timeout", 3) < 0.2);
        assert_eq!(perceptual_hash_hamming(0, u64::MAX), 64);
    }

    #[test]
    fn precursor_is_shadow_only_until_explicitly_enabled() {
        let mut controller = controller();
        let mut control = controller.clone();
        let zero_precursor_priority = control
            .observe(observation(
                5,
                ExamTier::Frontier,
                "vision.text.line",
                360,
                0,
            ))
            .unwrap()
            .priority;
        for step in [10, 20, 30] {
            let decision = controller
                .observe(observation(
                    step,
                    ExamTier::Frontier,
                    "vision.text.line",
                    360,
                    120,
                ))
                .unwrap();
            assert_eq!(decision.warning, WarningLevel::Watch);
            assert!(!decision.precursor_steering_enabled);
            assert!((decision.priority - zero_precursor_priority).abs() < 1e-12);
        }
    }

    #[test]
    fn stationary_frontier_noise_does_not_false_steer() {
        let mut controller = controller();
        for (index, pass) in [220, 224, 218, 226, 221, 223].into_iter().enumerate() {
            let decision = controller
                .observe(observation(
                    index as u64 * 10,
                    ExamTier::Frontier,
                    "reason.chain.depth",
                    pass,
                    10,
                ))
                .unwrap();
            assert!(decision.warning < WarningLevel::Steer);
        }
    }

    #[test]
    fn tiers_do_not_share_trend_or_change_frontier_difficulty() {
        let mut controller = controller();
        let frontier = controller
            .observe(observation(
                10,
                ExamTier::Frontier,
                "vision.text.line",
                220,
                0,
            ))
            .unwrap();
        assert_eq!(frontier.recommended_difficulty, 2);

        let sentinel = controller
            .observe(observation(
                20,
                ExamTier::Sentinel,
                "vision.text.line",
                390,
                0,
            ))
            .unwrap();
        assert_eq!(sentinel.recommended_difficulty, 2);

        let frontier = controller
            .observe(observation(
                30,
                ExamTier::Frontier,
                "vision.text.line",
                220,
                0,
            ))
            .unwrap();
        assert_eq!(frontier.warning, WarningLevel::Green);
        assert!(frontier.reasons.is_empty());
        let state = &controller.families["vision.text.line"];
        assert_eq!(state.tier_history.len(), 2);
        assert_eq!(state.tier_history[&ExamTier::Frontier].observations, 2);
        assert_eq!(state.tier_history[&ExamTier::Sentinel].observations, 1);
    }
}
