//! Per-model budgets, limits and timeouts (ADR 123).
//!
//! An akh's daily token budget, its per-wake token budget, the agentic turn's
//! deadline and round cap, the LLM HTTP timeout, the output-token cap, the
//! working context window and the tool count a wake is offered all depend on
//! the model the akh runs and on its reasoning effort. GLM-5.3 at `max` effort
//! answers at about 71 tok/s after a 32 s think; Kimi K3 at about 36 tok/s
//! after a minute; a local Gemma 4 12B has a 49k working window. One set of
//! compiled constants cannot fit all of them, and a process-wide env var moves
//! every akh on a host together.
//!
//! This module resolves one [`ResolvedModelProfile`] per akh, field by field:
//!
//! 1. the akh's own override (`[model_profile]` in `akh.toml`, or the legacy
//!    spellings `[autonomy.budget]` and `[openai_endpoint] context_window /
//!    timeout_secs / max_tokens`);
//! 2. the [`ModelCatalog`] row for the akh's `(model, reasoning_effort)`
//!    (`data/model-profiles/catalog.toml`, compiled in);
//! 3. the global default: the process env var where one exists
//!    (`AKH_AGENT_MAX_ROUNDS`, `AKH_AGENT_TURN_TIMEOUT_SECS`,
//!    `AKH_OLLAMA_TIMEOUT_SECS`), else the compiled constant.
//!
//! A model the catalog does not know resolves to exactly the pre-ADR-123
//! behaviour. Every resolved value remembers its [`ProfileSource`], so the
//! startup log names where each number came from, and the agentic loop can
//! keep its rule that only *explicit* limits cap autonomous dispatch.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// The compiled-in catalog text (`data/model-profiles/catalog.toml`).
pub const BUILTIN_CATALOG_TOML: &str = include_str!("../../data/model-profiles/catalog.toml");

/// The catalog schema this build understands.
pub const CATALOG_SCHEMA_VERSION: u32 = 1;

/// Global default daily token budget (unchanged from Phase 53b).
pub const DEFAULT_MAX_TOKENS_PER_DAY: u64 =
    super::autonomy::AutonomyBudget::DEFAULT_MAX_TOKENS_PER_DAY;
/// Global default daily turn budget (unchanged from Phase 53b).
pub const DEFAULT_MAX_TURNS_PER_DAY: u32 =
    super::autonomy::AutonomyBudget::DEFAULT_MAX_TURNS_PER_DAY;
/// Global default agentic turn deadline in seconds (the loop's historical 300 s).
pub const DEFAULT_TURN_DEADLINE_SECS: u64 = 300;
/// Global default Ollama read/idle timeout in seconds.
pub const DEFAULT_OLLAMA_HTTP_TIMEOUT_SECS: u64 = 120;
/// Global default OpenAI-compatible total request timeout in seconds.
pub const DEFAULT_OPENAI_HTTP_TIMEOUT_SECS: u64 = 600;

/// Env var that sets the global round cap.
pub const ENV_MAX_ROUNDS: &str = "AKH_AGENT_MAX_ROUNDS";
/// Env var that sets the global turn deadline.
pub const ENV_TURN_DEADLINE: &str = "AKH_AGENT_TURN_TIMEOUT_SECS";
/// Env var that sets the global Ollama HTTP timeout.
pub const ENV_OLLAMA_TIMEOUT: &str = "AKH_OLLAMA_TIMEOUT_SECS";

/// Errors loading the catalog or resolving an akh's profile.
#[derive(Debug, Clone, Error, Diagnostic, PartialEq, Eq)]
pub enum ModelProfileError {
    #[error("the model catalog does not parse: {message}")]
    #[diagnostic(
        code(akh::model_profile::catalog_parse),
        help(
            "data/model-profiles/catalog.toml is compiled into akhomed and read by the Roman \
             renderer. Fix the TOML (field names are listed at the top of the file; unknown keys \
             are refused), then rebuild. `cargo test --lib model_profile` checks it."
        )
    )]
    CatalogParse { message: String },

    #[error("the model catalog has schema_version {found}; this build reads {expected}")]
    #[diagnostic(
        code(akh::model_profile::catalog_schema),
        help(
            "Two ways forward: build akhomed from the commit that matches the catalog, or \
             rewrite the catalog in schema {expected}."
        )
    )]
    CatalogSchema { found: u32, expected: u32 },

    #[error("model id `{id}` appears in two catalog rows (`{first}` and `{second}`)")]
    #[diagnostic(
        code(akh::model_profile::catalog_duplicate_id),
        help(
            "Each exact id and each prefix may belong to one row only, otherwise the lookup is \
             ambiguous. Remove it from one of the two rows."
        )
    )]
    CatalogDuplicateId {
        id: String,
        first: String,
        second: String,
    },

    #[error("`{field}` is 0 in {location}")]
    #[diagnostic(
        code(akh::model_profile::zero_limit),
        help(
            "A zero budget, deadline or timeout would stop every turn before it starts. Options: \
             set a positive value; remove the key to inherit the catalog or global default; or, \
             to turn background LLM work off, set `[autonomy] enabled = false` (or \
             `max_turns_per_day = 0`, the one limit where 0 means \"off\")."
        )
    )]
    ZeroLimit {
        field: &'static str,
        location: String,
    },

    #[error("akh \"{akh}\" sets `{field}` twice: in {first} and in {second}")]
    #[diagnostic(
        code(akh::model_profile::duplicate_setting),
        help(
            "Keep one of the two. `[model_profile]` is the canonical place for per-model limits \
             (ADR 123); the older spelling still works on its own."
        )
    )]
    DuplicateSetting {
        akh: String,
        field: &'static str,
        first: &'static str,
        second: &'static str,
    },

    #[error("akh \"{akh}\" has reasoning_effort = \"{value}\", which is not an effort level")]
    #[diagnostic(
        code(akh::model_profile::unknown_effort),
        help(
            "Use one of \"low\", \"medium\", \"high\" or \"max\" (the values Ollama's `think` \
             and the OpenAI `reasoning_effort` field accept), or remove the key to run the \
             model at its default effort."
        )
    )]
    UnknownEffort { akh: String, value: String },
}

/// Result alias for this module.
pub type ModelProfileResult<T> = std::result::Result<T, ModelProfileError>;

/// A reasoning-effort level, as `reasoning_effort` in `akh.toml` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Max,
}

impl ReasoningEffort {
    /// Every level, in increasing effort.
    pub const ALL: [Self; 4] = [Self::Low, Self::Medium, Self::High, Self::Max];

    /// The lowercase label used in TOML and on the wire.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }

    /// Parse a label case-insensitively. `None` for anything else.
    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|level| level.as_str().eq_ignore_ascii_case(value.trim()))
    }
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a catalog row's model runs. Documentation for operators and the
/// renderer; resolution does not depend on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Serving {
    /// Ollama Cloud (`ollama.com`), metered in usage credit per token.
    OllamaCloud,
    /// A model server on the operator's own hardware; tokens cost GPU time.
    Local,
    /// Some other OpenAI-compatible provider.
    OpenaiCompatible,
}

/// A partial set of per-model limits: a catalog row, an effort override, or an
/// akh's `[model_profile]` table. Every field is optional; an absent field
/// falls through to the next tier.
///
/// ```toml
/// [model_profile]
/// max_tokens_per_day = 2000000
/// max_tools_per_wake = 32
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelProfile {
    /// Prompt + completion tokens per UTC day for background work.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_per_day: Option<u64>,
    /// Background model turns per UTC day (0 turns LLM-cost background work off).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns_per_day: Option<u32>,
    /// Prompt + completion tokens one reactor wake may spend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_per_wake: Option<u64>,
    /// Wall clock of one agentic turn, in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_deadline_secs: Option<u64>,
    /// Model rounds per agentic turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_rounds: Option<u32>,
    /// One LLM HTTP request, in seconds (Ollama: idle gap; OpenAI: total).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_timeout_secs: Option<u64>,
    /// Generated tokens per request, reasoning included.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Working context window in tokens (never above what the model reports).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Relevance-scoped tools offered to a reactor wake.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tools_per_wake: Option<u32>,
    /// Tools offered to an interactive (ACP / T3 / chat) turn, the akh's
    /// pinned core included (ADR 123 amendment 2026-09-30).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tools_per_turn: Option<u32>,
}

impl ModelProfile {
    /// The TOML keys, for the unknown-key check in `akh_config`.
    pub const FIELDS: &'static [&'static str] = &[
        "max_tokens_per_day",
        "max_turns_per_day",
        "max_tokens_per_wake",
        "turn_deadline_secs",
        "max_rounds",
        "http_timeout_secs",
        "max_output_tokens",
        "context_window",
        "max_tools_per_wake",
        "max_tools_per_turn",
    ];

    /// Whether no field is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// `self` with every unset field taken from `lower`.
    pub fn or(self, lower: Self) -> Self {
        Self {
            max_tokens_per_day: self.max_tokens_per_day.or(lower.max_tokens_per_day),
            max_turns_per_day: self.max_turns_per_day.or(lower.max_turns_per_day),
            max_tokens_per_wake: self.max_tokens_per_wake.or(lower.max_tokens_per_wake),
            turn_deadline_secs: self.turn_deadline_secs.or(lower.turn_deadline_secs),
            max_rounds: self.max_rounds.or(lower.max_rounds),
            http_timeout_secs: self.http_timeout_secs.or(lower.http_timeout_secs),
            max_output_tokens: self.max_output_tokens.or(lower.max_output_tokens),
            context_window: self.context_window.or(lower.context_window),
            max_tools_per_wake: self.max_tools_per_wake.or(lower.max_tools_per_wake),
            max_tools_per_turn: self.max_tools_per_turn.or(lower.max_tools_per_turn),
        }
    }

    /// Refuse a zero where zero would stop every turn. `max_turns_per_day = 0`
    /// is allowed: it is the documented "background LLM work off" switch.
    pub fn validate(&self, location: &str) -> ModelProfileResult<()> {
        let zero = |field: &'static str| ModelProfileError::ZeroLimit {
            field,
            location: location.to_string(),
        };
        let checks: [(&'static str, bool); 9] = [
            ("max_tokens_per_day", self.max_tokens_per_day == Some(0)),
            ("max_tokens_per_wake", self.max_tokens_per_wake == Some(0)),
            ("turn_deadline_secs", self.turn_deadline_secs == Some(0)),
            ("max_rounds", self.max_rounds == Some(0)),
            ("http_timeout_secs", self.http_timeout_secs == Some(0)),
            ("max_output_tokens", self.max_output_tokens == Some(0)),
            ("context_window", self.context_window == Some(0)),
            ("max_tools_per_wake", self.max_tools_per_wake == Some(0)),
            ("max_tools_per_turn", self.max_tools_per_turn == Some(0)),
        ];
        match checks.into_iter().find(|(_, is_zero)| *is_zero) {
            Some((field, _)) => Err(zero(field)),
            None => Ok(()),
        }
    }
}

/// One catalog row.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEntry {
    /// Row name used in logs and docs (e.g. `glm-5.3`).
    pub id: String,
    /// Exact model ids this row describes.
    #[serde(default)]
    pub ids: Vec<String>,
    /// Model-id prefixes this row describes when no exact id matches.
    #[serde(default)]
    pub prefixes: Vec<String>,
    /// Where the model runs.
    pub serving: Serving,
    /// Limits at the model's default reasoning effort.
    #[serde(default)]
    pub profile: ModelProfile,
    /// Overrides for an explicit `reasoning_effort`.
    #[serde(default)]
    pub effort: BTreeMap<ReasoningEffort, ModelProfile>,
}

impl CatalogEntry {
    /// This row's limits for `effort` (the effort override over the row).
    pub fn profile_for(&self, effort: Option<ReasoningEffort>) -> ModelProfile {
        match effort.and_then(|level| self.effort.get(&level)) {
            Some(over) => over.or(self.profile),
            None => self.profile,
        }
    }
}

/// The model catalog (`data/model-profiles/catalog.toml`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCatalog {
    /// Must equal [`CATALOG_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// The rows.
    #[serde(default)]
    pub models: Vec<CatalogEntry>,
}

/// Normalize a model id for lookup: trimmed, lowercase, registry namespace
/// (`library/`) and a trailing `:cloud` / `-cloud` tag dropped.
pub fn normalize_model_id(model: &str) -> String {
    let id = model.trim().to_ascii_lowercase();
    let base = id.rsplit('/').next().unwrap_or(&id).to_string();
    base.strip_suffix(":cloud")
        .or_else(|| base.strip_suffix("-cloud"))
        .map(str::to_string)
        .unwrap_or(base)
}

impl ModelCatalog {
    /// Parse and validate catalog text.
    pub fn parse(text: &str) -> ModelProfileResult<Self> {
        let catalog: Self = toml::from_str(text).map_err(|e| ModelProfileError::CatalogParse {
            message: e.to_string(),
        })?;
        catalog.validate()?;
        Ok(catalog)
    }

    /// The compiled-in catalog, parsed once.
    pub fn builtin() -> ModelProfileResult<&'static Self> {
        static BUILTIN: OnceLock<ModelProfileResult<ModelCatalog>> = OnceLock::new();
        BUILTIN
            .get_or_init(|| Self::parse(BUILTIN_CATALOG_TOML))
            .as_ref()
            .map_err(Clone::clone)
    }

    fn validate(&self) -> ModelProfileResult<()> {
        if self.schema_version != CATALOG_SCHEMA_VERSION {
            return Err(ModelProfileError::CatalogSchema {
                found: self.schema_version,
                expected: CATALOG_SCHEMA_VERSION,
            });
        }
        let mut owners: BTreeMap<String, &str> = BTreeMap::new();
        for entry in &self.models {
            entry
                .profile
                .validate(&format!("catalog row `{}`", entry.id))?;
            for (level, over) in &entry.effort {
                over.validate(&format!("catalog row `{}` effort `{level}`", entry.id))?;
            }
            let keys = entry
                .ids
                .iter()
                .map(|id| format!("id:{}", normalize_model_id(id)))
                .chain(
                    entry
                        .prefixes
                        .iter()
                        .map(|p| format!("prefix:{}", normalize_model_id(p))),
                );
            for key in keys {
                if let Some(first) = owners.insert(key.clone(), &entry.id) {
                    return Err(ModelProfileError::CatalogDuplicateId {
                        id: key.split_once(':').map(|(_, id)| id).unwrap_or(&key).into(),
                        first: first.to_string(),
                        second: entry.id.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// The row for `model`: an exact id first, otherwise the longest prefix.
    pub fn lookup(&self, model: &str) -> Option<&CatalogEntry> {
        let id = normalize_model_id(model);
        if id.is_empty() {
            return None;
        }
        if let Some(entry) = self
            .models
            .iter()
            .find(|entry| entry.ids.iter().any(|x| normalize_model_id(x) == id))
        {
            return Some(entry);
        }
        self.models
            .iter()
            .flat_map(|entry| {
                entry
                    .prefixes
                    .iter()
                    .map(move |p| (normalize_model_id(p), entry))
            })
            .filter(|(prefix, _)| id.starts_with(prefix.as_str()))
            .max_by_key(|(prefix, _)| prefix.len())
            .map(|(_, entry)| entry)
    }
}

/// Where one resolved value came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileSource {
    /// The akh's own config; the string names the key's location.
    AkhOverride(&'static str),
    /// The catalog row (and the effort override, when one applied).
    Catalog {
        entry: String,
        effort: Option<ReasoningEffort>,
    },
    /// A process-wide env var (the global default tier).
    Environment(&'static str),
    /// The compiled constant (the global default tier).
    Builtin,
}

impl ProfileSource {
    /// Whether an operator stated this value (per akh or per process), as
    /// opposed to a catalog or compiled default. Only explicit limits cap
    /// autonomous dispatch turns.
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::AkhOverride(_) | Self::Environment(_))
    }

    /// Short label for the startup log.
    pub fn label(&self) -> String {
        match self {
            Self::AkhOverride(location) => format!("akh:{location}"),
            Self::Catalog {
                entry,
                effort: Some(level),
            } => format!("catalog:{entry}@{level}"),
            Self::Catalog {
                entry,
                effort: None,
            } => format!("catalog:{entry}"),
            Self::Environment(var) => format!("env:{var}"),
            Self::Builtin => "builtin".into(),
        }
    }
}

/// A resolved value and its source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Setting<T> {
    pub value: T,
    pub source: ProfileSource,
}

impl<T: fmt::Display> fmt::Display for Setting<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.value, self.source.label())
    }
}

/// The global default tier for one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalDefaults {
    /// Compiled HTTP timeout for this backend.
    pub http_timeout_secs: u64,
    /// The env var that overrides it, if the backend has one.
    pub http_timeout_env: Option<&'static str>,
}

impl GlobalDefaults {
    /// Ollama: 120 s read/idle, `AKH_OLLAMA_TIMEOUT_SECS`.
    pub const OLLAMA: Self = Self {
        http_timeout_secs: DEFAULT_OLLAMA_HTTP_TIMEOUT_SECS,
        http_timeout_env: Some(ENV_OLLAMA_TIMEOUT),
    };
    /// OpenAI-compatible: 600 s total, no env var.
    pub const OPENAI: Self = Self {
        http_timeout_secs: DEFAULT_OPENAI_HTTP_TIMEOUT_SECS,
        http_timeout_env: None,
    };
}

/// One per-akh override, with the location to name in logs and errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AkhOverrides {
    /// The akh's merged overrides.
    pub profile: ModelProfile,
    /// Per field: the key location that set it (for [`ProfileSource::AkhOverride`]).
    pub locations: OverrideLocations,
}

/// Which config key set each overridden field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverrideLocations {
    pub max_tokens_per_day: &'static str,
    pub max_turns_per_day: &'static str,
    pub http_timeout_secs: &'static str,
    pub max_output_tokens: &'static str,
    pub context_window: &'static str,
}

impl Default for OverrideLocations {
    fn default() -> Self {
        Self {
            max_tokens_per_day: "model_profile",
            max_turns_per_day: "model_profile",
            http_timeout_secs: "model_profile",
            max_output_tokens: "model_profile",
            context_window: "model_profile",
        }
    }
}

/// What resolution needs to know about one akh.
#[derive(Debug, Clone, Copy)]
pub struct ProfileRequest<'a> {
    /// The akh's name (for diagnostics).
    pub akh: &'a str,
    /// The pinned model id, if any.
    pub model: Option<&'a str>,
    /// The akh's reasoning effort, if set.
    pub effort: Option<ReasoningEffort>,
    /// The akh's overrides.
    pub overrides: AkhOverrides,
    /// The global default tier for the akh's backend.
    pub defaults: GlobalDefaults,
}

/// An akh's resolved per-model profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelProfile {
    /// The model id the profile was resolved for (empty: none pinned).
    pub model: String,
    /// The reasoning effort it was resolved for.
    pub effort: Option<ReasoningEffort>,
    /// The catalog row that matched, if any.
    pub catalog_entry: Option<String>,
    pub max_tokens_per_day: Setting<u64>,
    pub max_turns_per_day: Setting<u32>,
    /// `None`: a wake has no token cap of its own (only the daily budget).
    pub max_tokens_per_wake: Option<Setting<u64>>,
    pub turn_deadline_secs: Setting<u64>,
    /// `None`: the loop scales the cap to the context window (12 / 20 / 30).
    pub max_rounds: Option<Setting<u32>>,
    pub http_timeout_secs: Setting<u64>,
    /// `None`: the server's default.
    pub max_output_tokens: Option<Setting<u32>>,
    /// `None`: whatever the model reports.
    pub context_window: Option<Setting<u64>>,
    /// `None`: a wake sees the whole catalog.
    pub max_tools_per_wake: Option<Setting<u32>>,
    /// `None`: an interactive turn sees the whole catalog.
    pub max_tools_per_turn: Option<Setting<u32>>,
}

/// Parse a positive integer env var; blank, zero or garbage count as unset.
fn env_positive(env: &dyn Fn(&str) -> Option<String>, key: &'static str) -> Option<u64> {
    env(key)
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&n| n > 0)
}

impl ResolvedModelProfile {
    /// Resolve with the process environment.
    pub fn resolve(
        catalog: &ModelCatalog,
        request: &ProfileRequest<'_>,
    ) -> ModelProfileResult<Self> {
        Self::resolve_with_env(catalog, request, &|key| std::env::var(key).ok())
    }

    /// Resolve with an injected environment (tests).
    pub fn resolve_with_env(
        catalog: &ModelCatalog,
        request: &ProfileRequest<'_>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> ModelProfileResult<Self> {
        let overrides = request.overrides.profile;
        overrides.validate(&format!("akh \"{}\" [model_profile]", request.akh))?;
        let locations = request.overrides.locations;
        let model = request.model.unwrap_or_default();
        let entry = catalog.lookup(model);
        let catalog_profile = entry
            .map(|e| e.profile_for(request.effort))
            .unwrap_or_default();
        let catalog_source = |uses_effort: bool| ProfileSource::Catalog {
            entry: entry.map(|e| e.id.clone()).unwrap_or_default(),
            effort: if uses_effort { request.effort } else { None },
        };
        // Whether the effort override (rather than the base row) set a field.
        let from_effort = |pick: fn(&ModelProfile) -> bool| {
            entry
                .zip(request.effort)
                .and_then(|(e, level)| e.effort.get(&level))
                .is_some_and(pick)
        };

        macro_rules! tiered {
            ($field:ident, $location:expr) => {
                overrides
                    .$field
                    .map(|value| Setting {
                        value,
                        source: ProfileSource::AkhOverride($location),
                    })
                    .or_else(|| {
                        catalog_profile.$field.map(|value| Setting {
                            value,
                            source: catalog_source(from_effort(|p| p.$field.is_some())),
                        })
                    })
            };
        }

        let max_tokens_per_day = tiered!(max_tokens_per_day, locations.max_tokens_per_day)
            .unwrap_or(Setting {
                value: DEFAULT_MAX_TOKENS_PER_DAY,
                source: ProfileSource::Builtin,
            });
        let max_turns_per_day =
            tiered!(max_turns_per_day, locations.max_turns_per_day).unwrap_or(Setting {
                value: DEFAULT_MAX_TURNS_PER_DAY,
                source: ProfileSource::Builtin,
            });
        let max_tokens_per_wake = tiered!(max_tokens_per_wake, "model_profile");
        let turn_deadline_secs = tiered!(turn_deadline_secs, "model_profile")
            .or_else(|| {
                env_positive(env, ENV_TURN_DEADLINE).map(|value| Setting {
                    value,
                    source: ProfileSource::Environment(ENV_TURN_DEADLINE),
                })
            })
            .unwrap_or(Setting {
                value: DEFAULT_TURN_DEADLINE_SECS,
                source: ProfileSource::Builtin,
            });
        let max_rounds = tiered!(max_rounds, "model_profile").or_else(|| {
            env_positive(env, ENV_MAX_ROUNDS).map(|value| Setting {
                value: u32::try_from(value).unwrap_or(u32::MAX),
                source: ProfileSource::Environment(ENV_MAX_ROUNDS),
            })
        });
        let http_timeout_secs = tiered!(http_timeout_secs, locations.http_timeout_secs)
            .or_else(|| {
                let var = request.defaults.http_timeout_env?;
                env_positive(env, var).map(|value| Setting {
                    value,
                    source: ProfileSource::Environment(var),
                })
            })
            .unwrap_or(Setting {
                value: request.defaults.http_timeout_secs,
                source: ProfileSource::Builtin,
            });
        let max_output_tokens = tiered!(max_output_tokens, locations.max_output_tokens);
        let context_window = tiered!(context_window, locations.context_window);
        let max_tools_per_wake = tiered!(max_tools_per_wake, "model_profile");
        let max_tools_per_turn = tiered!(max_tools_per_turn, "model_profile");

        Ok(Self {
            model: model.to_string(),
            effort: request.effort,
            catalog_entry: entry.map(|e| e.id.clone()),
            max_tokens_per_day,
            max_turns_per_day,
            max_tokens_per_wake,
            turn_deadline_secs,
            max_rounds,
            http_timeout_secs,
            max_output_tokens,
            context_window,
            max_tools_per_wake,
            max_tools_per_turn,
        })
    }

    /// The daily budget the autonomy governor enforces.
    pub fn daily_budget(&self) -> super::autonomy::AutonomyBudget {
        super::autonomy::AutonomyBudget {
            max_tokens_per_day: self.max_tokens_per_day.value,
            max_turns_per_day: self.max_turns_per_day.value,
        }
    }

    /// The working context window: the profile's cap, never above what the
    /// model reports. Either side alone is used as is.
    pub fn working_context_window(&self, reported: Option<usize>) -> Option<usize> {
        let cap = self
            .context_window
            .as_ref()
            .map(|s| usize::try_from(s.value).unwrap_or(usize::MAX));
        match (reported, cap) {
            (Some(model), Some(cap)) => Some(model.min(cap)),
            (model, cap) => model.or(cap),
        }
    }

    /// Log the resolved profile: one info line per akh.
    pub fn log(&self, akh: &str) {
        fn opt<T: fmt::Display>(setting: &Option<Setting<T>>, unset: &str) -> String {
            setting
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| unset.to_string())
        }
        tracing::info!(
            target: "akh::model_profile",
            workspace = %akh,
            model = %self.model,
            effort = self.effort.map(ReasoningEffort::as_str).unwrap_or("default"),
            catalog = self.catalog_entry.as_deref().unwrap_or("none"),
            max_tokens_per_day = %self.max_tokens_per_day,
            max_turns_per_day = %self.max_turns_per_day,
            max_tokens_per_wake = %opt(&self.max_tokens_per_wake, "none"),
            turn_deadline_secs = %self.turn_deadline_secs,
            max_rounds = %opt(&self.max_rounds, "scaled-to-window"),
            http_timeout_secs = %self.http_timeout_secs,
            max_output_tokens = %opt(&self.max_output_tokens, "server-default"),
            context_window = %opt(&self.context_window, "model-reported"),
            max_tools_per_wake = %opt(&self.max_tools_per_wake, "all"),
            max_tools_per_turn = %opt(&self.max_tools_per_turn, "all"),
            "model profile resolved (ADR 123)"
        );
    }
}

/// What a reactor wake tells the agentic loop about its tools (ADR 123).
///
/// A wake's catalog is relevance-scoped to the profile's
/// `max_tools_per_wake`, but the tools the wake's route names
/// (`required_tools`) and the tools its trusted text (schedule prompt and exit
/// condition, never the untrusted event payload) names by their exact
/// registered name are always offered, first in the list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct WakeScope {
    /// Tools the route declares it needs.
    pub required_tools: Vec<String>,
    /// The schedule's own prompt and exit condition.
    pub trusted_text: String,
    /// The schedule's lease in seconds: a wake never runs past the lease that
    /// bounds its attempt, whatever the model profile's turn deadline says.
    pub lease_secs: Option<u64>,
}

impl WakeScope {
    /// The tools to keep, in order: `required_tools`, then every registered
    /// tool name that appears as a whole identifier (containing `_`) in the
    /// trusted text.
    /// Identifier matching is structural (the registry's names), not a word
    /// list. Names absent from `catalog` are returned too, so the caller can
    /// report them.
    pub fn pinned_names(&self, catalog: &[&str]) -> Vec<String> {
        let mut pinned: Vec<String> = Vec::new();
        let mut push = |name: &str| {
            if !pinned.iter().any(|p| p == name) {
                pinned.push(name.to_string());
            }
        };
        for name in &self.required_tools {
            push(name);
        }
        let known: std::collections::HashSet<&str> = catalog.iter().copied().collect();
        for token in self
            .trusted_text
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            .filter(|t| !t.is_empty())
        {
            let token = token.trim_matches('-');
            // Only identifier-shaped tokens (with an underscore) are tool
            // references; a prose word never pins a tool, even one that
            // happens to share a tool's name.
            if token.contains('_') && known.contains(token) {
                push(token);
            }
        }
        pinned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn request<'a>(model: &'a str, effort: Option<ReasoningEffort>) -> ProfileRequest<'a> {
        ProfileRequest {
            akh: "janus",
            model: Some(model),
            effort,
            overrides: AkhOverrides {
                profile: ModelProfile::default(),
                locations: OverrideLocations::default(),
            },
            defaults: GlobalDefaults::OLLAMA,
        }
    }

    #[test]
    fn builtin_catalog_parses_and_covers_the_fleet() {
        let catalog = ModelCatalog::builtin().expect("the compiled catalog parses");
        for model in [
            "glm-5.3",
            "glm-5.3:cloud",
            "glm-5.3-flash",
            "glm-5.2",
            "kimi-k3",
            "kimi-k2.7-code",
            "kimi-k2.6",
            "deepseek-v4-pro",
            "deepseek-v4.1-flash",
            "lilith",
            "gemma4:12b",
            "qwen3.8:27b",
        ] {
            assert!(catalog.lookup(model).is_some(), "{model} has a catalog row");
        }
        assert_eq!(catalog.lookup("glm-5.3-flash").unwrap().id, "glm-5.3-flash");
        assert_eq!(catalog.lookup("GLM-5.3:cloud").unwrap().id, "glm-5.3");
        assert_eq!(
            catalog.lookup("glm-5.4").unwrap().id,
            "glm-5.3",
            "prefix fallback"
        );
        assert_eq!(
            catalog.lookup("deepseek-v4.1").unwrap().id,
            "deepseek-v4-pro"
        );
        assert!(catalog.lookup("gpt-6-astra").is_none());
        assert!(catalog.lookup("").is_none());
    }

    #[test]
    fn every_cloud_row_is_complete() {
        let catalog = ModelCatalog::builtin().unwrap();
        for entry in &catalog.models {
            let p = entry.profile;
            assert!(
                p.max_tokens_per_day.is_some()
                    && p.max_turns_per_day.is_some()
                    && p.max_tokens_per_wake.is_some()
                    && p.turn_deadline_secs.is_some()
                    && p.max_rounds.is_some()
                    && p.http_timeout_secs.is_some()
                    && p.max_output_tokens.is_some()
                    && p.context_window.is_some()
                    && p.max_tools_per_wake.is_some(),
                "row {} sets every field, so no fleet model silently inherits a global default",
                entry.id
            );
        }
    }

    #[test]
    fn unknown_model_resolves_to_the_pre_adr_123_defaults() {
        let catalog = ModelCatalog::builtin().unwrap();
        let resolved =
            ResolvedModelProfile::resolve_with_env(catalog, &request("gpt-6-astra", None), &no_env)
                .unwrap();
        assert_eq!(resolved.catalog_entry, None);
        assert_eq!(resolved.max_tokens_per_day.value, 250_000);
        assert_eq!(resolved.max_turns_per_day.value, 40);
        assert_eq!(resolved.turn_deadline_secs.value, 300);
        assert_eq!(resolved.http_timeout_secs.value, 120);
        assert_eq!(
            resolved.max_rounds, None,
            "rounds stay scaled to the window"
        );
        assert_eq!(resolved.max_tokens_per_wake, None);
        assert_eq!(resolved.max_output_tokens, None);
        assert_eq!(resolved.context_window, None);
        assert_eq!(resolved.max_tools_per_wake, None);
        assert_eq!(resolved.max_tokens_per_day.source, ProfileSource::Builtin);
    }

    #[test]
    fn catalog_row_and_effort_override_apply() {
        let catalog = ModelCatalog::builtin().unwrap();
        let base =
            ResolvedModelProfile::resolve_with_env(catalog, &request("glm-5.3", None), &no_env)
                .unwrap();
        assert_eq!(base.catalog_entry.as_deref(), Some("glm-5.3"));
        assert_eq!(base.max_tokens_per_day.value, 1_500_000);
        assert_eq!(base.max_rounds.as_ref().unwrap().value, 30);
        assert_eq!(base.max_tools_per_wake.as_ref().unwrap().value, 24);
        let low = ResolvedModelProfile::resolve_with_env(
            catalog,
            &request("glm-5.3", Some(ReasoningEffort::Low)),
            &no_env,
        )
        .unwrap();
        assert_eq!(low.max_rounds.as_ref().unwrap().value, 16);
        assert_eq!(
            low.max_rounds.as_ref().unwrap().source,
            ProfileSource::Catalog {
                entry: "glm-5.3".into(),
                effort: Some(ReasoningEffort::Low)
            }
        );
        assert_eq!(
            low.max_tokens_per_day.source,
            ProfileSource::Catalog {
                entry: "glm-5.3".into(),
                effort: None
            },
            "a field the effort row does not set comes from the base row"
        );
    }

    #[test]
    fn akh_override_beats_catalog_which_beats_env() {
        let catalog = ModelCatalog::builtin().unwrap();
        let env = |key: &str| match key {
            ENV_MAX_ROUNDS => Some("7".to_string()),
            ENV_TURN_DEADLINE => Some("99".to_string()),
            ENV_OLLAMA_TIMEOUT => Some("45".to_string()),
            _ => None,
        };
        let mut req = request("glm-5.3", None);
        req.overrides.profile.max_rounds = Some(50);
        let resolved = ResolvedModelProfile::resolve_with_env(catalog, &req, &env).unwrap();
        assert_eq!(resolved.max_rounds.as_ref().unwrap().value, 50);
        assert!(resolved.max_rounds.as_ref().unwrap().source.is_explicit());
        assert_eq!(
            resolved.turn_deadline_secs.value, 1_800,
            "catalog beats env"
        );
        assert!(!resolved.turn_deadline_secs.source.is_explicit());
        let unknown =
            ResolvedModelProfile::resolve_with_env(catalog, &request("other", None), &env).unwrap();
        assert_eq!(
            unknown.max_rounds.as_ref().unwrap().value,
            7,
            "env is the global default"
        );
        assert_eq!(unknown.turn_deadline_secs.value, 99);
        assert_eq!(unknown.http_timeout_secs.value, 45);
        assert_eq!(
            unknown.http_timeout_secs.source,
            ProfileSource::Environment(ENV_OLLAMA_TIMEOUT)
        );
        let mut openai = request("other", None);
        openai.defaults = GlobalDefaults::OPENAI;
        let resolved = ResolvedModelProfile::resolve_with_env(catalog, &openai, &env).unwrap();
        assert_eq!(
            resolved.http_timeout_secs.value, 600,
            "the Ollama env var does not move an OpenAI akh"
        );
    }

    #[test]
    fn zero_overrides_are_refused_with_options() {
        let catalog = ModelCatalog::builtin().unwrap();
        let mut req = request("glm-5.3", None);
        req.overrides.profile.turn_deadline_secs = Some(0);
        let error = ResolvedModelProfile::resolve_with_env(catalog, &req, &no_env).unwrap_err();
        assert!(matches!(
            error,
            ModelProfileError::ZeroLimit {
                field: "turn_deadline_secs",
                ..
            }
        ));
        let mut req = request("glm-5.3", None);
        req.overrides.profile.max_turns_per_day = Some(0);
        assert!(
            ResolvedModelProfile::resolve_with_env(catalog, &req, &no_env).is_ok(),
            "0 turns is the documented off switch"
        );
    }

    #[test]
    fn catalog_validation_refuses_bad_rows() {
        let dup = r#"
            schema_version = 1
            [[models]]
            id = "a"
            ids = ["x"]
            serving = "local"
            [[models]]
            id = "b"
            ids = ["X:cloud"]
            serving = "local"
        "#;
        assert!(matches!(
            ModelCatalog::parse(dup),
            Err(ModelProfileError::CatalogDuplicateId { .. })
        ));
        let typo = r#"
            schema_version = 1
            [[models]]
            id = "a"
            serving = "local"
            [models.profile]
            max_round = 3
        "#;
        let error = ModelCatalog::parse(typo).unwrap_err();
        assert!(matches!(error, ModelProfileError::CatalogParse { .. }));
        assert!(error.to_string().contains("max_round"));
        assert!(matches!(
            ModelCatalog::parse("schema_version = 2"),
            Err(ModelProfileError::CatalogSchema { found: 2, .. })
        ));
    }

    #[test]
    fn working_window_never_exceeds_the_model() {
        let catalog = ModelCatalog::builtin().unwrap();
        let glm =
            ResolvedModelProfile::resolve_with_env(catalog, &request("glm-5.3", None), &no_env)
                .unwrap();
        assert_eq!(glm.working_context_window(Some(1_048_576)), Some(262_144));
        assert_eq!(glm.working_context_window(Some(8_192)), Some(8_192));
        assert_eq!(glm.working_context_window(None), Some(262_144));
        let other =
            ResolvedModelProfile::resolve_with_env(catalog, &request("x", None), &no_env).unwrap();
        assert_eq!(other.working_context_window(Some(4_096)), Some(4_096));
        assert_eq!(other.working_context_window(None), None);
    }

    #[test]
    fn effort_parses_case_insensitively() {
        assert_eq!(ReasoningEffort::parse("HIGH"), Some(ReasoningEffort::High));
        assert_eq!(ReasoningEffort::parse(" max "), Some(ReasoningEffort::Max));
        assert_eq!(ReasoningEffort::parse("xhigh"), None);
    }

    #[test]
    fn wake_scope_pins_route_tools_and_named_tools_only() {
        let scope = WakeScope {
            required_tools: vec!["monitoring_snapshot".into()],
            trusted_text: "Coordinate with mcp__board__board_send_inbox. Record it with \
                           mcp__laya__record_fixer_result, then stop."
                .into(),
            lease_secs: None,
        };
        let catalog = [
            "mcp__board__board_send_inbox",
            "mcp__laya__record_fixer_result",
            "monitoring_snapshot",
            "stop",
            "mcp__anima__list_work_items",
        ];
        assert_eq!(
            scope.pinned_names(&catalog),
            vec![
                "monitoring_snapshot",
                "mcp__board__board_send_inbox",
                "mcp__laya__record_fixer_result",
            ],
            "a prose word (`stop`) never pins a tool"
        );
        assert!(WakeScope::default().pinned_names(&catalog).is_empty());
    }
}
