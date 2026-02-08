//! Error types for the autonomous KG building subsystem.

use miette::Diagnostic;
use thiserror::Error;

/// Errors from the autonomous reasoning subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum AutonomousError {
    #[error("rule parse error in rule '{rule_name}': {message}")]
    #[diagnostic(
        code(akh::autonomous::rule_parse),
        help("Rules must specify valid patterns. Use '?' prefix for variables.")
    )]
    RuleParse { rule_name: String, message: String },

    #[error("rule engine exceeded max iterations ({max_iterations})")]
    #[diagnostic(
        code(akh::autonomous::max_iterations),
        help("Increase max_iterations or review rules for non-terminating chains.")
    )]
    MaxIterations { max_iterations: usize },

    #[error("gap analysis found no goals to analyze")]
    #[diagnostic(
        code(akh::autonomous::no_goals),
        help("Add a goal with agent.add_goal() first.")
    )]
    NoGoalsForGap,

    #[error("schema discovery requires >= {min_triples} triples, found {actual}")]
    #[diagnostic(
        code(akh::autonomous::insufficient_data),
        help("Add more knowledge via skills, ingest, or the kg_mutate tool.")
    )]
    InsufficientData { min_triples: usize, actual: usize },

    #[error("fusion error: {message}")]
    #[diagnostic(
        code(akh::autonomous::fusion),
        help("Check that inference paths are valid.")
    )]
    Fusion { message: String },

    #[error(transparent)]
    #[diagnostic(
        code(akh::autonomous::engine),
        help("An engine-level error occurred during an autonomous operation.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for AutonomousError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Result type for autonomous reasoning operations.
pub type AutonomousResult<T> = std::result::Result<T, AutonomousError>;
