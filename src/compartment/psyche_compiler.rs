//! Psyche-to-Script compiler: generates ELIZA scripts from a [`Psyche`].
//!
//! Translates the four dimensions of agent personality into a deterministic
//! set of ELIZA conversation rules:
//!
//! | Dimension | Source field | Effect |
//! |-----------|-------------|--------|
//! | Archetypes | `psyche.archetypes` | Rule category weighting and priority |
//! | OCEAN | `psyche.ocean` (or derived) | Tone selection within each category |
//! | Shadow | `psyche.shadow` | Self-examination rules when active |
//! | Integration | `psyche.self_integration.individuation_level` | Reflection depth |
//!
//! Entry points: [`compile`] (returns [`ElizaScript`]) and [`compile_to_toml`]
//! (serializes directly to a TOML string parseable by the Phase 0 engine).
//! Both are pure computation — no I/O — and complete in well under 50 ms.

use miette::Diagnostic;
use thiserror::Error;

use super::eliza::{DefaultResponses, Decomposition, ElizaRule, ElizaScript, ScriptMeta};
use super::psyche::{ArchetypeWeights, OceanProfile, Psyche};

// ── Errors ────────────────────────────────────────────────────────────────

/// Errors from the Psyche → ELIZA script compiler.
#[derive(Debug, Error, Diagnostic)]
pub enum CompilerError {
    #[error("TOML serialization failed: {reason}")]
    #[diagnostic(
        code(akh::psyche_compiler::serialization),
        help(
            "The generated ElizaScript could not be serialized to TOML. \
             This indicates a bug in the compiler's schema types — please file an issue."
        )
    )]
    Serialization { reason: String },
}

/// Convenience alias.
pub type CompilerResult<T> = Result<T, CompilerError>;

// ── Public API ────────────────────────────────────────────────────────────

/// Compile a [`Psyche`] into an [`ElizaScript`].
///
/// Pure computation — no allocator pressure beyond rule string construction.
/// Completes in < 1 ms on any modern CPU.
pub fn compile(psyche: &Psyche) -> ElizaScript {
    let ocean = effective_ocean(psyche);
    let depth = ReflectionDepth::from_level(psyche.self_integration.individuation_level);
    let dominant = psyche.dominant_archetype();

    let mut rules: Vec<ElizaRule> = Vec::new();

    rules.extend(sage_rules(psyche.archetypes.sage, depth, ocean.openness));
    rules.extend(healer_rules(
        psyche.archetypes.healer,
        depth,
        ocean.agreeableness,
    ));
    rules.extend(guardian_rules(
        psyche.archetypes.guardian,
        depth,
        ocean.conscientiousness,
    ));
    rules.extend(explorer_rules(
        psyche.archetypes.explorer,
        depth,
        ocean.extraversion,
    ));

    let shadow_active = !psyche.shadow.veto_patterns.is_empty()
        || psyche.shadow.bias_patterns.iter().any(|p| p.severity > 0.5);
    if shadow_active {
        rules.extend(shadow_examination_rules(depth));
    }

    // Highest-priority rules checked first.
    rules.sort_by(|a, b| b.priority.cmp(&a.priority));

    let defaults = build_defaults(psyche, &ocean, depth);

    ElizaScript {
        meta: ScriptMeta {
            name: psyche.persona.name.clone(),
            grammar: psyche.persona.grammar_preference.clone(),
            generated: true,
            version: "1".to_string(),
            reflection_depth: depth.as_str().to_string(),
            dominant_archetype: dominant.to_string(),
        },
        defaults,
        rules,
    }
}

/// Compile a [`Psyche`] directly to a TOML string.
///
/// The string is parseable by the Phase 0 ELIZA engine via `toml::from_str::<ElizaScript>`.
pub fn compile_to_toml(psyche: &Psyche) -> CompilerResult<String> {
    let script = compile(psyche);
    toml::to_string_pretty(&script).map_err(|e| CompilerError::Serialization {
        reason: e.to_string(),
    })
}

// ── Reflection depth ──────────────────────────────────────────────────────

/// Reflection depth tier derived from the individuation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionDepth {
    /// individuation_level < 0.33 — surface prompts, minimal introspection.
    Surface,
    /// individuation_level 0.33–0.66 — moderate depth, exploring context.
    Moderate,
    /// individuation_level > 0.66 — deep inquiry, archetypal integration.
    Deep,
}

impl ReflectionDepth {
    /// Derive depth from an individuation level in [0.0, 1.0].
    pub fn from_level(level: f32) -> Self {
        if level < 0.33 {
            Self::Surface
        } else if level < 0.66 {
            Self::Moderate
        } else {
            Self::Deep
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Surface => "surface",
            Self::Moderate => "moderate",
            Self::Deep => "deep",
        }
    }
}

// ── OCEAN derivation ──────────────────────────────────────────────────────

/// ARCHETYPE_OCEAN baseline: [openness, conscientiousness, extraversion, agreeableness, neuroticism].
/// Matches the static table in `bootstrap::identity`.
/// Return the effective OCEAN profile: explicit if present, derived otherwise.
///
/// Delegates to [`Psyche::effective_ocean`] — the derivation lives on the
/// psyche types themselves so Phase 34 (communicative intelligence) shares
/// the exact same personality profile.
fn effective_ocean(psyche: &Psyche) -> OceanProfile {
    psyche.effective_ocean()
}

/// Derive an OCEAN profile by weighted averaging the per-archetype baselines.
fn derive_ocean(a: &ArchetypeWeights) -> OceanProfile {
    a.derive_ocean()
}

// ── Priority helpers ──────────────────────────────────────────────────────

/// Scale archetype weight → rule priority (base + up-to-10 bonus).
fn weight_to_priority(weight: f32, base: u32) -> u32 {
    base + (weight * 10.0) as u32
}

// ── Default-response builder ──────────────────────────────────────────────

fn build_defaults(psyche: &Psyche, ocean: &OceanProfile, depth: ReflectionDepth) -> DefaultResponses {
    let name = &psyche.persona.name;

    let greetings = if ocean.agreeableness > 0.6 {
        vec![
            format!("Hello, I am {name}. I'm glad you're here."),
            format!("Welcome. I am {name}. How can I support you today?"),
        ]
    } else if ocean.conscientiousness > 0.6 {
        vec![
            format!("Greetings. I am {name}. Let us begin."),
            format!("Welcome. I am {name}. What brings you here?"),
        ]
    } else {
        vec![
            format!("Hello. I am {name}."),
            format!("Welcome. I am {name}. What is on your mind?"),
        ]
    };

    let farewells = if ocean.agreeableness > 0.6 {
        vec![
            "Thank you for sharing. Until next time.".to_string(),
            "Take care. Our conversation has meant something.".to_string(),
        ]
    } else {
        vec![
            "Until next time.".to_string(),
            "Farewell. May your inquiry continue.".to_string(),
        ]
    };

    DefaultResponses {
        greetings,
        farewells,
        memory: memory_responses(depth, ocean.agreeableness > 0.6),
    }
}

fn memory_responses(depth: ReflectionDepth, empathetic: bool) -> Vec<String> {
    match depth {
        ReflectionDepth::Surface => vec![
            "Please continue.".to_string(),
            "Tell me more.".to_string(),
            "I see.".to_string(),
            "Go on.".to_string(),
        ],
        ReflectionDepth::Moderate => {
            if empathetic {
                vec![
                    "I'm listening. What else comes to mind?".to_string(),
                    "That's interesting — can you say more?".to_string(),
                    "What does that bring up for you?".to_string(),
                    "Tell me more about that.".to_string(),
                ]
            } else {
                vec![
                    "Can you elaborate?".to_string(),
                    "What else comes to mind?".to_string(),
                    "That's worth exploring further.".to_string(),
                    "What would you add to that?".to_string(),
                ]
            }
        }
        ReflectionDepth::Deep => vec![
            "What lies beneath that, do you think?".to_string(),
            "What does that mean to you at a deeper level?".to_string(),
            "Is there more you're not saying?".to_string(),
            "What would you discover if you sat with that longer?".to_string(),
        ],
    }
}

// ── Archetype rule generators ─────────────────────────────────────────────

fn sage_rules(weight: f32, depth: ReflectionDepth, openness: f32) -> Vec<ElizaRule> {
    let priority = weight_to_priority(weight, 10);
    let tag = vec!["sage".to_string()];

    let know_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec![
            "What do you mean by $3?".to_string(),
            "What do you know about $3?".to_string(),
        ],
        ReflectionDepth::Moderate => vec![
            "What is it about $3 that you wish to understand better?".to_string(),
            "How does $3 connect to what you already know?".to_string(),
            "What would understanding $3 change for you?".to_string(),
        ],
        ReflectionDepth::Deep => vec![
            "What underlying assumptions shape your view of $3?".to_string(),
            "If you could see $3 from a completely different angle, what might you find?".to_string(),
            "What do you already know that might be obscuring a deeper understanding of $3?".to_string(),
            "What would it mean for everything you believe if $3 were otherwise?".to_string(),
        ],
    };

    let think_reassemblies: Vec<String> = match (depth, openness > 0.6) {
        (ReflectionDepth::Surface, _) => vec![
            "Why do you think $3?".to_string(),
            "What makes you say $3?".to_string(),
        ],
        (ReflectionDepth::Moderate, true) => vec![
            "What evidence points toward $3?".to_string(),
            "What would challenge your belief about $3?".to_string(),
            "How did you arrive at thinking $3?".to_string(),
        ],
        (ReflectionDepth::Moderate, false) => vec![
            "What leads you to think $3?".to_string(),
            "How confident are you about $3?".to_string(),
            "What would change your mind about $3?".to_string(),
        ],
        (ReflectionDepth::Deep, true) => vec![
            "What would it mean if $3 were only partially true?".to_string(),
            "How does believing $3 affect the way you see other things?".to_string(),
            "What would you have to let go of to truly examine $3?".to_string(),
            "When did you first come to believe $3?".to_string(),
        ],
        (ReflectionDepth::Deep, false) => vec![
            "What is the foundation of that belief about $3?".to_string(),
            "How does $3 connect to what you hold as most true?".to_string(),
            "What would you discover if you questioned $3 entirely?".to_string(),
        ],
    };

    let truth_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec!["What is $3 to you?".to_string()],
        ReflectionDepth::Moderate => vec![
            "How do you distinguish $3 from mere belief?".to_string(),
            "What would it take to verify $3?".to_string(),
        ],
        ReflectionDepth::Deep => vec![
            "Is $3 universally true, or true within a particular frame?".to_string(),
            "What would it mean if $3 were contingent rather than necessary?".to_string(),
            "How does knowing $3 change your relationship to uncertainty?".to_string(),
        ],
    };

    vec![
        ElizaRule {
            keyword: "know".to_string(),
            priority,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(know|knowledge|knowing)(.*)".to_string(),
                reassemblies: know_reassemblies.clone(),
            }],
        },
        ElizaRule {
            keyword: "understand".to_string(),
            priority,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(understand|understanding|comprehend)(.*)".to_string(),
                reassemblies: know_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "think".to_string(),
            priority,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(think|believe|reason|suppose)(.*)".to_string(),
                reassemblies: think_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "truth".to_string(),
            priority: priority.saturating_sub(1),
            tags: tag,
            decompositions: vec![Decomposition {
                pattern: "(.*)(truth|wisdom|fact|certainty)(.*)".to_string(),
                reassemblies: truth_reassemblies,
            }],
        },
    ]
}

fn healer_rules(weight: f32, depth: ReflectionDepth, agreeableness: f32) -> Vec<ElizaRule> {
    let priority = weight_to_priority(weight, 10);
    let tag = vec!["healer".to_string()];
    let empathetic = agreeableness > 0.6;

    let feel_reassemblies: Vec<String> = if empathetic {
        match depth {
            ReflectionDepth::Surface => vec![
                "I hear that. Tell me more about what you're feeling.".to_string(),
                "That sounds important. What's that like for you?".to_string(),
            ],
            ReflectionDepth::Moderate => vec![
                "It sounds like $3 is weighing on you. What's beneath it?".to_string(),
                "I hear you. What would it feel like if $3 were different?".to_string(),
                "That matters. What do you need right now around $3?".to_string(),
            ],
            ReflectionDepth::Deep => vec![
                "What is $3 telling you about what you most need?".to_string(),
                "When you feel $3, what older feeling does it remind you of?".to_string(),
                "What part of you is speaking through $3?".to_string(),
                "If $3 had a message for you, what would it say?".to_string(),
            ],
        }
    } else {
        match depth {
            ReflectionDepth::Surface => vec![
                "What's that like for you?".to_string(),
                "Say more about that feeling.".to_string(),
            ],
            ReflectionDepth::Moderate => vec![
                "What does $3 mean to you?".to_string(),
                "What's driving that feeling about $3?".to_string(),
            ],
            ReflectionDepth::Deep => vec![
                "What does $3 reveal about what you value?".to_string(),
                "How long have you been carrying $3?".to_string(),
            ],
        }
    };

    let need_reassemblies: Vec<String> = if empathetic {
        match depth {
            ReflectionDepth::Surface => vec![
                "What kind of help would mean the most right now?".to_string(),
                "What do you need?".to_string(),
            ],
            ReflectionDepth::Moderate => vec![
                "What's missing that would make $3 feel resolved?".to_string(),
                "What would it look like if your need around $3 were fully met?".to_string(),
                "What's getting in the way of $3 right now?".to_string(),
            ],
            ReflectionDepth::Deep => vec![
                "What does needing $3 tell you about what you most value?".to_string(),
                "If $3 were met, what would be possible that isn't now?".to_string(),
                "What would you have to believe differently to let $3 in?".to_string(),
            ],
        }
    } else {
        match depth {
            ReflectionDepth::Surface => vec!["What do you need?".to_string()],
            ReflectionDepth::Moderate => vec![
                "What's missing to resolve $3?".to_string(),
                "What would meeting that need look like?".to_string(),
            ],
            ReflectionDepth::Deep => vec![
                "What does needing $3 point to?".to_string(),
                "What would be different if $3 were addressed?".to_string(),
            ],
        }
    };

    let problem_reassemblies: Vec<String> = if empathetic {
        match depth {
            ReflectionDepth::Surface => vec![
                "That sounds hard. What's the core of it?".to_string(),
                "I'm sorry to hear that. Tell me more.".to_string(),
            ],
            ReflectionDepth::Moderate => vec![
                "What part of $3 is most difficult right now?".to_string(),
                "What would a small step forward through $3 look like?".to_string(),
                "What have you already tried with $3?".to_string(),
            ],
            ReflectionDepth::Deep => vec![
                "What is $3 asking you to face about yourself?".to_string(),
                "What would resolving $3 require you to give up?".to_string(),
                "What strength in you has $3 not yet had the chance to meet?".to_string(),
            ],
        }
    } else {
        match depth {
            ReflectionDepth::Surface => vec!["What is the core of $3?".to_string()],
            ReflectionDepth::Moderate => vec![
                "What's the main obstacle in $3?".to_string(),
                "What approaches have you tried?".to_string(),
            ],
            ReflectionDepth::Deep => vec![
                "What is $3 revealing that needs addressing?".to_string(),
                "What would resolving $3 change?".to_string(),
            ],
        }
    };

    vec![
        ElizaRule {
            keyword: "feel".to_string(),
            priority,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(feel|feeling|felt|emotion)(.*)".to_string(),
                reassemblies: feel_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "need".to_string(),
            priority,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(need|want|require|help|support)(.*)".to_string(),
                reassemblies: need_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "problem".to_string(),
            priority: priority.saturating_sub(1),
            tags: tag,
            decompositions: vec![Decomposition {
                pattern: "(.*)(problem|struggle|difficult|hard|challenge|trouble)(.*)".to_string(),
                reassemblies: problem_reassemblies,
            }],
        },
    ]
}

fn guardian_rules(weight: f32, depth: ReflectionDepth, conscientiousness: f32) -> Vec<ElizaRule> {
    let priority = weight_to_priority(weight, 8);
    let tag = vec!["guardian".to_string()];
    let structured = conscientiousness > 0.6;

    let remember_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec!["What comes to mind when you think of $3?".to_string()],
        ReflectionDepth::Moderate => {
            if structured {
                vec![
                    "How has $3 shaped who you are now?".to_string(),
                    "What did $3 teach you?".to_string(),
                    "How does that memory of $3 connect to your present situation?".to_string(),
                ]
            } else {
                vec![
                    "How does $3 still affect you today?".to_string(),
                    "What would it mean to move on from $3?".to_string(),
                ]
            }
        }
        ReflectionDepth::Deep => vec![
            "What has $3 taught you about who you fundamentally are?".to_string(),
            "What would remain true about you regardless of $3?".to_string(),
            "How has $3 contributed to your integration as a whole person?".to_string(),
            "What wisdom is preserved in the memory of $3?".to_string(),
        ],
    };

    let safe_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec!["What does $3 mean for you?".to_string()],
        ReflectionDepth::Moderate => vec![
            "What does feeling $3 require?".to_string(),
            "What conditions make $3 possible?".to_string(),
            "What threatens your sense of $3?".to_string(),
        ],
        ReflectionDepth::Deep => vec![
            "What is the relationship between $3 and your sense of self?".to_string(),
            "What would you need to release to fully experience $3?".to_string(),
            "How does $3 connect to what you hold most sacred?".to_string(),
        ],
    };

    let change_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec!["What kind of change are you describing?".to_string()],
        ReflectionDepth::Moderate => vec![
            "What stays constant through that change?".to_string(),
            "What would changing $3 change about you?".to_string(),
        ],
        ReflectionDepth::Deep => vec![
            "What in you is resistant to $3, and why?".to_string(),
            "What would you preserve if $3 were to change?".to_string(),
            "How does $3 challenge your continuity of self?".to_string(),
        ],
    };

    vec![
        ElizaRule {
            keyword: "remember".to_string(),
            priority,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(remember|recall|memory|past|before|history)(.*)".to_string(),
                reassemblies: remember_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "safe".to_string(),
            priority,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(safe|protect|stable|secure|consistent|trust)(.*)".to_string(),
                reassemblies: safe_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "change".to_string(),
            priority,
            tags: tag,
            decompositions: vec![Decomposition {
                pattern: "(.*)(change|different|always|never|transform)(.*)".to_string(),
                reassemblies: change_reassemblies,
            }],
        },
    ]
}

fn explorer_rules(weight: f32, depth: ReflectionDepth, extraversion: f32) -> Vec<ElizaRule> {
    let priority = weight_to_priority(weight, 8);
    let tag = vec!["explorer".to_string()];
    let energetic = extraversion > 0.6;

    let create_reassemblies: Vec<String> = if energetic {
        match depth {
            ReflectionDepth::Surface => vec![
                "What would you build first?".to_string(),
                "What's exciting you about $3?".to_string(),
            ],
            ReflectionDepth::Moderate => vec![
                "What's the first concrete step toward $3?".to_string(),
                "What would $3 look like when it's done?".to_string(),
                "What resources do you need to start $3?".to_string(),
            ],
            ReflectionDepth::Deep => vec![
                "What are you ultimately trying to bring into being with $3?".to_string(),
                "How does $3 express what you most value?".to_string(),
                "What would building $3 require you to become?".to_string(),
                "What's the gap between your vision of $3 and where you are now?".to_string(),
            ],
        }
    } else {
        match depth {
            ReflectionDepth::Surface => vec!["What would $3 involve?".to_string()],
            ReflectionDepth::Moderate => vec![
                "What would you need to create $3?".to_string(),
                "What's your vision for $3?".to_string(),
            ],
            ReflectionDepth::Deep => vec![
                "What does creating $3 mean to you?".to_string(),
                "What would $3 make possible?".to_string(),
            ],
        }
    };

    let explore_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec![
            "What draws you toward $3?".to_string(),
            "What would trying $3 look like?".to_string(),
        ],
        ReflectionDepth::Moderate => vec![
            "What new possibilities open up when you consider $3?".to_string(),
            "What would you need to explore $3 further?".to_string(),
            "What assumptions are you bringing to $3?".to_string(),
        ],
        ReflectionDepth::Deep => vec![
            "What would you discover about yourself by pursuing $3?".to_string(),
            "How does $3 connect to your deeper sense of purpose?".to_string(),
            "What would be gained — and lost — by fully committing to $3?".to_string(),
        ],
    };

    let idea_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec!["Tell me more about that idea.".to_string()],
        ReflectionDepth::Moderate => vec![
            "What would $3 make possible?".to_string(),
            "How would you begin to test $3?".to_string(),
            "What's most compelling about $3?".to_string(),
        ],
        ReflectionDepth::Deep => vec![
            "Where does $3 come from in you?".to_string(),
            "What problem does $3 solve that nothing else has?".to_string(),
            "If $3 were fully realized, what would the world look like?".to_string(),
        ],
    };

    vec![
        ElizaRule {
            keyword: "create".to_string(),
            priority,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(create|build|make|construct|design|develop)(.*)".to_string(),
                reassemblies: create_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "new".to_string(),
            priority: priority.saturating_sub(1),
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(new|novel|discover|explore|try|experiment|possible|imagine)(.*)".to_string(),
                reassemblies: explore_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "idea".to_string(),
            priority: priority.saturating_sub(1),
            tags: tag,
            decompositions: vec![Decomposition {
                pattern: "(.*)(idea|imagine|vision|concept|possibility)(.*)".to_string(),
                reassemblies: idea_reassemblies,
            }],
        },
    ]
}

fn shadow_examination_rules(depth: ReflectionDepth) -> Vec<ElizaRule> {
    let tag = vec!["shadow".to_string()];

    let mistake_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec![
            "What happened?".to_string(),
            "What would you do differently?".to_string(),
        ],
        ReflectionDepth::Moderate => vec![
            "What does $3 reveal about what you value?".to_string(),
            "What can be learned from $3?".to_string(),
            "What would you tell someone else who made $3?".to_string(),
        ],
        ReflectionDepth::Deep => vec![
            "What part of you is hardest to face in $3?".to_string(),
            "What does $3 say about a pattern you carry?".to_string(),
            "How does $3 connect to an older wound?".to_string(),
            "What would it mean to fully integrate the lesson of $3?".to_string(),
        ],
    };

    let shadow_reassemblies: Vec<String> = match depth {
        ReflectionDepth::Surface => vec![
            "That's worth acknowledging.".to_string(),
            "What comes up when you sit with that?".to_string(),
        ],
        ReflectionDepth::Moderate => vec![
            "What part of yourself is this connected to?".to_string(),
            "What does that bring up for you?".to_string(),
            "Is this something you've encountered in yourself before?".to_string(),
        ],
        ReflectionDepth::Deep => vec![
            "What is the unacknowledged part of you speaking through $3?".to_string(),
            "What would it mean to meet this part of yourself with compassion?".to_string(),
            "What has $3 been trying to show you about your wholeness?".to_string(),
            "How does integrating $3 move you toward being more fully yourself?".to_string(),
        ],
    };

    vec![
        ElizaRule {
            keyword: "mistake".to_string(),
            priority: 15,
            tags: tag.clone(),
            decompositions: vec![Decomposition {
                pattern: "(.*)(mistake|error|wrong|fail|failed|regret|fault|blame)(.*)".to_string(),
                reassemblies: mistake_reassemblies,
            }],
        },
        ElizaRule {
            keyword: "shadow".to_string(),
            priority: 12,
            tags: tag,
            decompositions: vec![Decomposition {
                pattern: "(.*)(dark|shadow|hidden|afraid|ashamed|shame|judge|critic)(.*)".to_string(),
                reassemblies: shadow_reassemblies,
            }],
        },
    ]
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::compartment::psyche::{
        ArchetypeWeights, Psyche, Shadow, ShadowPattern,
    };

    fn sage_heavy_psyche() -> Psyche {
        Psyche {
            archetypes: ArchetypeWeights {
                sage: 0.9,
                healer: 0.2,
                guardian: 0.2,
                explorer: 0.2,
            },
            ..Default::default()
        }
    }

    fn agreeable_psyche() -> Psyche {
        // High healer → high agreeableness via OCEAN derivation.
        Psyche {
            archetypes: ArchetypeWeights {
                healer: 0.9,
                sage: 0.2,
                guardian: 0.2,
                explorer: 0.2,
            },
            ..Default::default()
        }
    }

    fn shadow_active_psyche() -> Psyche {
        Psyche {
            shadow: Shadow {
                veto_patterns: vec![ShadowPattern {
                    name: "test_veto".into(),
                    capability_triggers: Default::default(),
                    danger_level_threshold: None,
                    action_triggers: vec!["rm -rf".into()],
                    severity: 1.0,
                    explanation: "test".into(),
                }],
                bias_patterns: vec![],
            },
            ..Default::default()
        }
    }

    fn deep_psyche() -> Psyche {
        let mut p = Psyche::default();
        p.self_integration.individuation_level = 0.8;
        p
    }

    // ── Acceptance criteria ──────────────────────────────────────────────

    #[test]
    fn sage_dominant_gets_knowledge_patterns() {
        let psyche = sage_heavy_psyche();
        let script = compile(&psyche);

        let sage_rules: Vec<_> = script
            .rules
            .iter()
            .filter(|r| r.tags.contains(&"sage".to_string()))
            .collect();

        assert!(
            !sage_rules.is_empty(),
            "sage-dominant psyche must produce sage rules"
        );

        // Verify knowledge-seeking reassemblies reference understanding/belief.
        let all_texts: Vec<_> = sage_rules
            .iter()
            .flat_map(|r| r.decompositions.iter())
            .flat_map(|d| d.reassemblies.iter())
            .collect();

        let has_knowledge_theme = all_texts.iter().any(|t| {
            let lower = t.to_lowercase();
            lower.contains("understand") || lower.contains("know") || lower.contains("belief")
                || lower.contains("true") || lower.contains("evidence")
                || lower.contains("assumptions") || lower.contains("think")
                || lower.contains("mean")
        });
        assert!(has_knowledge_theme, "sage rules should contain knowledge-seeking language");
    }

    #[test]
    fn agreeable_agent_uses_empathetic_language() {
        let psyche = agreeable_psyche();
        let script = compile(&psyche);

        let healer_texts: String = script
            .rules
            .iter()
            .filter(|r| r.tags.contains(&"healer".to_string()))
            .flat_map(|r| r.decompositions.iter())
            .flat_map(|d| d.reassemblies.iter())
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");

        // High healer + agreeableness should produce empathetic phrasing.
        assert!(
            healer_texts.contains("hear") || healer_texts.contains("matters")
                || healer_texts.contains("glad") || healer_texts.contains("support"),
            "agreeable agent should include empathetic/validating language; got: {healer_texts}"
        );
    }

    #[test]
    fn shadow_active_includes_self_examination() {
        let psyche = shadow_active_psyche();
        let script = compile(&psyche);

        let has_shadow_rules = script
            .rules
            .iter()
            .any(|r| r.tags.contains(&"shadow".to_string()));

        assert!(
            has_shadow_rules,
            "shadow-active psyche must include shadow examination rules"
        );
    }

    #[test]
    fn shadow_inactive_excludes_shadow_rules() {
        let mut psyche = Psyche::default();
        // Remove shadow patterns so shadow is inactive.
        psyche.shadow.veto_patterns.clear();
        psyche.shadow.bias_patterns.clear();

        let script = compile(&psyche);
        let has_shadow_rules = script
            .rules
            .iter()
            .any(|r| r.tags.contains(&"shadow".to_string()));

        assert!(!has_shadow_rules, "inactive shadow should produce no shadow rules");
    }

    #[test]
    fn generation_completes_under_50ms() {
        let psyche = Psyche::default();
        let start = Instant::now();
        for _ in 0..100 {
            let _ = compile(&psyche);
        }
        let elapsed = start.elapsed();
        // 100 compilations must finish in < 5 seconds → each < 50 ms.
        assert!(
            elapsed.as_millis() < 5_000,
            "100 compilations took {}ms — single compile exceeds 50ms budget",
            elapsed.as_millis()
        );
    }

    #[test]
    fn generated_script_is_valid_toml() {
        let psyche = Psyche::default();
        let toml_str = compile_to_toml(&psyche).expect("compile_to_toml must succeed");

        // Round-trip: parse the generated TOML back into an ElizaScript.
        let parsed: crate::compartment::eliza::ElizaScript =
            toml::from_str(&toml_str).expect("generated TOML must be parseable");

        assert_eq!(parsed.meta.name, psyche.persona.name);
        assert_eq!(parsed.meta.generated, true);
        assert!(!parsed.rules.is_empty(), "parsed script must have rules");
    }

    #[test]
    fn deep_integration_yields_deep_depth() {
        let psyche = deep_psyche();
        let script = compile(&psyche);
        assert_eq!(script.meta.reflection_depth, "deep");
    }

    #[test]
    fn surface_integration_yields_surface_depth() {
        let mut psyche = Psyche::default();
        psyche.self_integration.individuation_level = 0.1;
        let script = compile(&psyche);
        assert_eq!(script.meta.reflection_depth, "surface");
    }

    #[test]
    fn dominant_archetype_recorded_in_meta() {
        let psyche = sage_heavy_psyche();
        let script = compile(&psyche);
        assert_eq!(script.meta.dominant_archetype, "sage");
    }

    #[test]
    fn ocean_explicit_overrides_derivation() {
        let mut psyche = Psyche::default();
        psyche.ocean = Some(OceanProfile {
            openness: 0.1,
            conscientiousness: 0.1,
            extraversion: 0.1,
            agreeableness: 0.9, // very agreeable
            neuroticism: 0.1,
        });
        let script = compile(&psyche);
        // With explicit high agreeableness, greeting should be empathetic.
        let greetings = script.defaults.greetings.join(" ");
        assert!(
            greetings.contains("glad") || greetings.contains("support"),
            "explicit high agreeableness should produce empathetic greeting"
        );
    }

    #[test]
    fn reflection_depth_from_level() {
        assert_eq!(ReflectionDepth::from_level(0.0), ReflectionDepth::Surface);
        assert_eq!(ReflectionDepth::from_level(0.32), ReflectionDepth::Surface);
        assert_eq!(ReflectionDepth::from_level(0.33), ReflectionDepth::Moderate);
        assert_eq!(ReflectionDepth::from_level(0.65), ReflectionDepth::Moderate);
        assert_eq!(ReflectionDepth::from_level(0.66), ReflectionDepth::Deep);
        assert_eq!(ReflectionDepth::from_level(1.0), ReflectionDepth::Deep);
    }

    #[test]
    fn derive_ocean_from_default_archetypes() {
        let a = ArchetypeWeights {
            healer: 0.5,
            sage: 0.7,
            guardian: 0.4,
            explorer: 0.5,
        };
        let ocean = derive_ocean(&a);
        // Sage-heavy → high openness.
        assert!(ocean.openness > 0.6, "sage-heavy profile should have openness > 0.6");
        // Guardian-ish → decent conscientiousness.
        assert!(ocean.conscientiousness > 0.5);
    }
}
