//! Data-driven inference rules for forward-chaining reasoning.
//!
//! Rules are structs, not code — they can be loaded from JSON or an extended
//! text format, or constructed programmatically via `RuleSet::builtin()`.

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

use super::error::{AutonomousError, AutonomousResult};

// ---------------------------------------------------------------------------
// Rule term
// ---------------------------------------------------------------------------

/// A term in a rule pattern: concrete symbol, variable, or label.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuleTerm {
    /// A concrete symbol ID (already resolved).
    Concrete(SymbolId),
    /// A variable binding (e.g., `?X`).
    Variable(String),
    /// A label that will be resolved to a SymbolId at execution time.
    Label(String),
}

impl RuleTerm {
    /// Returns `true` if this term is a variable.
    pub fn is_variable(&self) -> bool {
        matches!(self, Self::Variable(_))
    }

    /// Parse a term from a string token. Variables start with `?`, numeric
    /// values are concrete IDs, and everything else is a label.
    pub fn parse(token: &str) -> Self {
        let token = token.trim();
        if let Some(var) = token.strip_prefix('?') {
            Self::Variable(var.to_string())
        } else if let Ok(raw) = token.parse::<u64>() {
            if let Some(id) = SymbolId::new(raw) {
                Self::Concrete(id)
            } else {
                Self::Label(token.to_string())
            }
        } else {
            Self::Label(token.to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Triple pattern
// ---------------------------------------------------------------------------

/// A triple pattern in a rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: RuleTerm,
    pub predicate: RuleTerm,
    pub object: RuleTerm,
}

impl TriplePattern {
    /// Parse a triple pattern from `(?X pred ?Y)` syntax.
    pub fn parse(s: &str) -> AutonomousResult<Self> {
        let s = s.trim();
        let inner = s
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or(s);
        let parts: Vec<&str> = inner.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(AutonomousError::RuleParse {
                rule_name: String::new(),
                message: format!(
                    "triple pattern must have exactly 3 terms, got {}: '{s}'",
                    parts.len()
                ),
            });
        }
        Ok(Self {
            subject: RuleTerm::parse(parts[0]),
            predicate: RuleTerm::parse(parts[1]),
            object: RuleTerm::parse(parts[2]),
        })
    }
}

// ---------------------------------------------------------------------------
// Rule kind
// ---------------------------------------------------------------------------

/// Classification of the logical inference a rule performs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuleKind {
    TransitiveClosure,
    InverseRelation,
    SymmetricRelation,
    DomainConstraint,
    RangeConstraint,
    TypeSubsumption,
    Custom { name: String },
}

impl RuleKind {
    /// Parse a rule kind from a string.
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "transitiveclosure" | "transitive" => Self::TransitiveClosure,
            "inverserelation" | "inverse" => Self::InverseRelation,
            "symmetricrelation" | "symmetric" => Self::SymmetricRelation,
            "domainconstraint" | "domain" => Self::DomainConstraint,
            "rangeconstraint" | "range" => Self::RangeConstraint,
            "typesubsumption" | "type" => Self::TypeSubsumption,
            other => Self::Custom {
                name: other.to_string(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Inference rule
// ---------------------------------------------------------------------------

/// A single inference rule: match antecedents, produce consequents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRule {
    pub name: String,
    pub kind: RuleKind,
    pub antecedents: Vec<TriplePattern>,
    pub consequents: Vec<TriplePattern>,
    /// Confidence factor multiplied into derived triple confidence.
    pub confidence_factor: f32,
    pub enabled: bool,
    /// Maximum new triples this rule can produce per iteration.
    pub max_derivations_per_iteration: usize,
}

impl InferenceRule {
    /// Create a new rule with defaults.
    pub fn new(name: impl Into<String>, kind: RuleKind) -> Self {
        Self {
            name: name.into(),
            kind,
            antecedents: Vec::new(),
            consequents: Vec::new(),
            confidence_factor: 1.0,
            enabled: true,
            max_derivations_per_iteration: 1000,
        }
    }

    /// Set antecedents.
    pub fn with_antecedents(mut self, antecedents: Vec<TriplePattern>) -> Self {
        self.antecedents = antecedents;
        self
    }

    /// Set consequents.
    pub fn with_consequents(mut self, consequents: Vec<TriplePattern>) -> Self {
        self.consequents = consequents;
        self
    }

    /// Set confidence factor.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence_factor = confidence;
        self
    }
}

// ---------------------------------------------------------------------------
// Rule set
// ---------------------------------------------------------------------------

/// A collection of rules with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleSet {
    pub name: String,
    pub rules: Vec<InferenceRule>,
    pub source: String,
}

impl RuleSet {
    /// The built-in ontological rule set (6 rules).
    pub fn builtin() -> Self {
        let rules = vec![
            // 1. is-a transitivity
            InferenceRule::new("is-a-transitive", RuleKind::TransitiveClosure)
                .with_antecedents(vec![
                    TriplePattern {
                        subject: RuleTerm::Variable("X".into()),
                        predicate: RuleTerm::Label("is-a".into()),
                        object: RuleTerm::Variable("Y".into()),
                    },
                    TriplePattern {
                        subject: RuleTerm::Variable("Y".into()),
                        predicate: RuleTerm::Label("is-a".into()),
                        object: RuleTerm::Variable("Z".into()),
                    },
                ])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("is-a".into()),
                    object: RuleTerm::Variable("Z".into()),
                }])
                .with_confidence(0.95),
            // 2. part-of transitivity
            InferenceRule::new("part-of-transitive", RuleKind::TransitiveClosure)
                .with_antecedents(vec![
                    TriplePattern {
                        subject: RuleTerm::Variable("X".into()),
                        predicate: RuleTerm::Label("part-of".into()),
                        object: RuleTerm::Variable("Y".into()),
                    },
                    TriplePattern {
                        subject: RuleTerm::Variable("Y".into()),
                        predicate: RuleTerm::Label("part-of".into()),
                        object: RuleTerm::Variable("Z".into()),
                    },
                ])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("part-of".into()),
                    object: RuleTerm::Variable("Z".into()),
                }])
                .with_confidence(0.90),
            // 3. similar-to symmetry
            InferenceRule::new("similar-to-symmetric", RuleKind::SymmetricRelation)
                .with_antecedents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("similar-to".into()),
                    object: RuleTerm::Variable("Y".into()),
                }])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("Y".into()),
                    predicate: RuleTerm::Label("similar-to".into()),
                    object: RuleTerm::Variable("X".into()),
                }])
                .with_confidence(1.0),
            // 4. parent-child inverse
            InferenceRule::new("parent-child-inverse", RuleKind::InverseRelation)
                .with_antecedents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("parent-of".into()),
                    object: RuleTerm::Variable("Y".into()),
                }])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("Y".into()),
                    predicate: RuleTerm::Label("child-of".into()),
                    object: RuleTerm::Variable("X".into()),
                }])
                .with_confidence(1.0),
            // 5. contains-part inverse
            InferenceRule::new("contains-part-inverse", RuleKind::InverseRelation)
                .with_antecedents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("contains".into()),
                    object: RuleTerm::Variable("Y".into()),
                }])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("Y".into()),
                    predicate: RuleTerm::Label("part-of".into()),
                    object: RuleTerm::Variable("X".into()),
                }])
                .with_confidence(1.0),
            // 6. has-a transitivity
            InferenceRule::new("has-a-transitive", RuleKind::TransitiveClosure)
                .with_antecedents(vec![
                    TriplePattern {
                        subject: RuleTerm::Variable("X".into()),
                        predicate: RuleTerm::Label("has-a".into()),
                        object: RuleTerm::Variable("Y".into()),
                    },
                    TriplePattern {
                        subject: RuleTerm::Variable("Y".into()),
                        predicate: RuleTerm::Label("has-a".into()),
                        object: RuleTerm::Variable("Z".into()),
                    },
                ])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("has-a".into()),
                    object: RuleTerm::Variable("Z".into()),
                }])
                .with_confidence(0.85),
        ];

        Self {
            name: "builtin".into(),
            rules,
            source: "builtin".into(),
        }
    }

    /// Code-specific inference rules (6 rules for code structure reasoning).
    ///
    /// These compose with the builtin ontological rules since code entities also
    /// use `is-a` (e.g., `Engine is-a Struct`, `create_symbol is-a Function`).
    pub fn code_rules() -> Self {
        let rules = vec![
            // 1. Transitive dependency: (X depends-on Y) ∧ (Y depends-on Z) ⟹ (X depends-on Z)
            InferenceRule::new("depends-on-transitive", RuleKind::TransitiveClosure)
                .with_antecedents(vec![
                    TriplePattern {
                        subject: RuleTerm::Variable("X".into()),
                        predicate: RuleTerm::Label("code:depends-on".into()),
                        object: RuleTerm::Variable("Y".into()),
                    },
                    TriplePattern {
                        subject: RuleTerm::Variable("Y".into()),
                        predicate: RuleTerm::Label("code:depends-on".into()),
                        object: RuleTerm::Variable("Z".into()),
                    },
                ])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("code:depends-on".into()),
                    object: RuleTerm::Variable("Z".into()),
                }])
                .with_confidence(0.85),
            // 2. Transitive module containment.
            InferenceRule::new("module-containment-transitive", RuleKind::TransitiveClosure)
                .with_antecedents(vec![
                    TriplePattern {
                        subject: RuleTerm::Variable("X".into()),
                        predicate: RuleTerm::Label("code:contains-mod".into()),
                        object: RuleTerm::Variable("Y".into()),
                    },
                    TriplePattern {
                        subject: RuleTerm::Variable("Y".into()),
                        predicate: RuleTerm::Label("code:contains-mod".into()),
                        object: RuleTerm::Variable("Z".into()),
                    },
                ])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("code:contains-mod".into()),
                    object: RuleTerm::Variable("Z".into()),
                }])
                .with_confidence(0.95),
            // 3. Trait method inheritance: impl T for X, T has-method M ⟹ X has-method M.
            InferenceRule::new("trait-method-inheritance", RuleKind::TypeSubsumption)
                .with_antecedents(vec![
                    TriplePattern {
                        subject: RuleTerm::Variable("X".into()),
                        predicate: RuleTerm::Label("code:implements-trait".into()),
                        object: RuleTerm::Variable("T".into()),
                    },
                    TriplePattern {
                        subject: RuleTerm::Variable("T".into()),
                        predicate: RuleTerm::Label("code:has-method".into()),
                        object: RuleTerm::Variable("M".into()),
                    },
                ])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("code:has-method".into()),
                    object: RuleTerm::Variable("M".into()),
                }])
                .with_confidence(0.90),
            // 4. Circular dependency detection.
            InferenceRule::new(
                "circular-dependency",
                RuleKind::Custom {
                    name: "circular-dep".into(),
                },
            )
            .with_antecedents(vec![
                TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("code:depends-on".into()),
                    object: RuleTerm::Variable("Y".into()),
                },
                TriplePattern {
                    subject: RuleTerm::Variable("Y".into()),
                    predicate: RuleTerm::Label("code:depends-on".into()),
                    object: RuleTerm::Variable("X".into()),
                },
            ])
            .with_consequents(vec![TriplePattern {
                subject: RuleTerm::Variable("X".into()),
                predicate: RuleTerm::Label("code:circular-dep".into()),
                object: RuleTerm::Variable("Y".into()),
            }])
            .with_confidence(1.0),
            // 5. defines-fn inverse: (X defines-fn Y) ⟹ (Y defined-in X).
            InferenceRule::new("defines-fn-inverse", RuleKind::InverseRelation)
                .with_antecedents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("code:defines-fn".into()),
                    object: RuleTerm::Variable("Y".into()),
                }])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("Y".into()),
                    predicate: RuleTerm::Label("code:defined-in".into()),
                    object: RuleTerm::Variable("X".into()),
                }])
                .with_confidence(1.0),
            // 6. defines-struct inverse: (X defines-struct Y) ⟹ (Y defined-in X).
            InferenceRule::new("defines-struct-inverse", RuleKind::InverseRelation)
                .with_antecedents(vec![TriplePattern {
                    subject: RuleTerm::Variable("X".into()),
                    predicate: RuleTerm::Label("code:defines-struct".into()),
                    object: RuleTerm::Variable("Y".into()),
                }])
                .with_consequents(vec![TriplePattern {
                    subject: RuleTerm::Variable("Y".into()),
                    predicate: RuleTerm::Label("code:defined-in".into()),
                    object: RuleTerm::Variable("X".into()),
                }])
                .with_confidence(1.0),
        ];

        Self {
            name: "code".into(),
            rules,
            source: "builtin-code".into(),
        }
    }

    /// Parse a rule set from JSON.
    pub fn from_json(json: &str, source: &str) -> AutonomousResult<Self> {
        let rules: Vec<InferenceRule> =
            serde_json::from_str(json).map_err(|e| AutonomousError::RuleParse {
                rule_name: String::new(),
                message: format!("JSON parse error: {e}"),
            })?;
        Ok(Self {
            name: source.to_string(),
            rules,
            source: source.to_string(),
        })
    }

    /// Parse rules from the extended text format.
    ///
    /// Format:
    /// ```text
    /// @rule is-a-transitive TransitiveClosure
    ///   match: (?X is-a ?Y), (?Y is-a ?Z)
    ///   produce: (?X is-a ?Z)
    ///   confidence: 0.95
    /// ```
    ///
    /// Lines without `@rule` prefix are ignored (legacy e-graph rules pass through).
    pub fn parse_from_text(text: &str, source: &str) -> AutonomousResult<Self> {
        let mut rules = Vec::new();
        let mut lines = text.lines().peekable();

        while let Some(line) = lines.next() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("@rule") {
                let parts: Vec<&str> = rest.trim().splitn(2, ' ').collect();
                if parts.len() < 2 {
                    return Err(AutonomousError::RuleParse {
                        rule_name: rest.trim().to_string(),
                        message: "@rule requires a name and a kind".into(),
                    });
                }
                let name = parts[0].to_string();
                let kind = RuleKind::parse(parts[1]);

                let mut antecedents = Vec::new();
                let mut consequents = Vec::new();
                let mut confidence = 1.0f32;

                // Read indented continuation lines.
                while let Some(next) = lines.peek() {
                    let next_trimmed = next.trim();
                    if next_trimmed.is_empty() || next_trimmed.starts_with('@') {
                        break;
                    }
                    let next_trimmed = lines.next().unwrap().trim();

                    if let Some(match_str) = next_trimmed.strip_prefix("match:") {
                        for pat_str in split_patterns(match_str) {
                            let mut pat = TriplePattern::parse(pat_str)?;
                            set_rule_name_on_error(&name, &mut pat)?;
                            antecedents.push(pat);
                        }
                    } else if let Some(prod_str) = next_trimmed.strip_prefix("produce:") {
                        for pat_str in split_patterns(prod_str) {
                            let mut pat = TriplePattern::parse(pat_str)?;
                            set_rule_name_on_error(&name, &mut pat)?;
                            consequents.push(pat);
                        }
                    } else if let Some(conf_str) = next_trimmed.strip_prefix("confidence:") {
                        confidence = conf_str.trim().parse::<f32>().map_err(|e| {
                            AutonomousError::RuleParse {
                                rule_name: name.clone(),
                                message: format!("invalid confidence: {e}"),
                            }
                        })?;
                    }
                    // Skip unknown keys.
                }

                rules.push(
                    InferenceRule::new(name, kind)
                        .with_antecedents(antecedents)
                        .with_consequents(consequents)
                        .with_confidence(confidence),
                );
            }
            // Lines without @rule prefix are ignored.
        }

        Ok(Self {
            name: source.to_string(),
            rules,
            source: source.to_string(),
        })
    }

    /// Return the total number of enabled rules.
    pub fn enabled_count(&self) -> usize {
        self.rules.iter().filter(|r| r.enabled).count()
    }
}

/// Split a comma-separated list of `(...)` patterns.
fn split_patterns(s: &str) -> Vec<&str> {
    let s = s.trim();
    let mut results = Vec::new();
    let mut depth = 0;
    let mut start = 0;

    for (i, ch) in s.char_indices() {
        match ch {
            '(' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            ')' => {
                depth -= 1;
                if depth == 0 {
                    results.push(&s[start..=i]);
                }
            }
            _ => {}
        }
    }

    // If no parentheses were found, treat the whole string as a single pattern.
    if results.is_empty() && !s.is_empty() {
        results.push(s);
    }

    results
}

/// Dummy validation — just ensures the parse result is structurally OK.
fn set_rule_name_on_error(_name: &str, _pat: &mut TriplePattern) -> AutonomousResult<()> {
    // Triple pattern already parsed successfully.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_has_six_rules() {
        let rs = RuleSet::builtin();
        assert_eq!(rs.rules.len(), 6);
        assert_eq!(rs.enabled_count(), 6);
    }

    #[test]
    fn builtin_rule_names() {
        let rs = RuleSet::builtin();
        let names: Vec<&str> = rs.rules.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"is-a-transitive"));
        assert!(names.contains(&"part-of-transitive"));
        assert!(names.contains(&"similar-to-symmetric"));
        assert!(names.contains(&"parent-child-inverse"));
        assert!(names.contains(&"contains-part-inverse"));
        assert!(names.contains(&"has-a-transitive"));
    }

    #[test]
    fn builtin_is_a_transitive_structure() {
        let rs = RuleSet::builtin();
        let rule = rs
            .rules
            .iter()
            .find(|r| r.name == "is-a-transitive")
            .unwrap();
        assert_eq!(rule.antecedents.len(), 2);
        assert_eq!(rule.consequents.len(), 1);
        assert_eq!(rule.confidence_factor, 0.95);
        assert_eq!(rule.kind, RuleKind::TransitiveClosure);
    }

    #[test]
    fn triple_pattern_parse_round_trip() {
        let pat = TriplePattern::parse("(?X is-a ?Y)").unwrap();
        assert!(matches!(pat.subject, RuleTerm::Variable(ref v) if v == "X"));
        assert!(matches!(pat.predicate, RuleTerm::Label(ref l) if l == "is-a"));
        assert!(matches!(pat.object, RuleTerm::Variable(ref v) if v == "Y"));
    }

    #[test]
    fn triple_pattern_bad_arity() {
        let result = TriplePattern::parse("(?X is-a)");
        assert!(result.is_err());
    }

    #[test]
    fn rule_term_parse_variable() {
        assert!(matches!(RuleTerm::parse("?X"), RuleTerm::Variable(ref v) if v == "X"));
    }

    #[test]
    fn rule_term_parse_label() {
        assert!(matches!(RuleTerm::parse("is-a"), RuleTerm::Label(ref l) if l == "is-a"));
    }

    #[test]
    fn rule_term_parse_numeric() {
        let term = RuleTerm::parse("42");
        assert!(matches!(term, RuleTerm::Concrete(id) if id.get() == 42));
    }

    #[test]
    fn rule_kind_parse() {
        assert_eq!(RuleKind::parse("transitive"), RuleKind::TransitiveClosure);
        assert_eq!(RuleKind::parse("inverse"), RuleKind::InverseRelation);
        assert_eq!(RuleKind::parse("symmetric"), RuleKind::SymmetricRelation);
        assert!(matches!(RuleKind::parse("weird"), RuleKind::Custom { .. }));
    }

    #[test]
    fn text_format_parsing() {
        let text = r#"
@rule my-transitive TransitiveClosure
  match: (?X rel ?Y), (?Y rel ?Z)
  produce: (?X rel ?Z)
  confidence: 0.9

# Legacy rule line (ignored)
(similar ?x ?y) => (similar ?y ?x)
"#;
        let rs = RuleSet::parse_from_text(text, "test").unwrap();
        assert_eq!(rs.rules.len(), 1);
        assert_eq!(rs.rules[0].name, "my-transitive");
        assert_eq!(rs.rules[0].antecedents.len(), 2);
        assert_eq!(rs.rules[0].consequents.len(), 1);
        assert_eq!(rs.rules[0].confidence_factor, 0.9);
    }

    #[test]
    fn code_rules_has_six_rules() {
        let rs = RuleSet::code_rules();
        assert_eq!(rs.rules.len(), 6);
        assert_eq!(rs.enabled_count(), 6);
    }

    #[test]
    fn code_rule_names() {
        let rs = RuleSet::code_rules();
        let names: Vec<&str> = rs.rules.iter().map(|r| r.name.as_str()).collect();
        assert!(names.contains(&"depends-on-transitive"));
        assert!(names.contains(&"module-containment-transitive"));
        assert!(names.contains(&"trait-method-inheritance"));
        assert!(names.contains(&"circular-dependency"));
        assert!(names.contains(&"defines-fn-inverse"));
        assert!(names.contains(&"defines-struct-inverse"));
    }

    #[test]
    fn json_parsing() {
        let rule = InferenceRule::new("test", RuleKind::TransitiveClosure)
            .with_antecedents(vec![TriplePattern {
                subject: RuleTerm::Variable("X".into()),
                predicate: RuleTerm::Label("is-a".into()),
                object: RuleTerm::Variable("Y".into()),
            }])
            .with_consequents(vec![TriplePattern {
                subject: RuleTerm::Variable("X".into()),
                predicate: RuleTerm::Label("is-a".into()),
                object: RuleTerm::Variable("Y".into()),
            }])
            .with_confidence(0.8);

        let json = serde_json::to_string(&vec![rule]).unwrap();
        let rs = RuleSet::from_json(&json, "test").unwrap();
        assert_eq!(rs.rules.len(), 1);
        assert_eq!(rs.rules[0].confidence_factor, 0.8);
    }
}
