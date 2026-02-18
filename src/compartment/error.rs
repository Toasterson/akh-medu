//! Compartment-specific error types with rich diagnostics.

use miette::Diagnostic;
use thiserror::Error;

use crate::error::StoreError;

/// Errors arising from compartment operations.
#[derive(Debug, Error, Diagnostic)]
pub enum CompartmentError {
    #[error("compartment not found: {id}")]
    #[diagnostic(
        code(akh::compartment::not_found),
        help(
            "No compartment with id \"{id}\" is registered. \
             Run `discover()` to scan the compartments directory, \
             or check that the compartment directory exists."
        )
    )]
    NotFound { id: String },

    #[error("compartment already loaded: {id}")]
    #[diagnostic(
        code(akh::compartment::already_loaded),
        help(
            "The compartment \"{id}\" is already in the Loaded or Active state. \
             Call `unload()` first if you want to reload it."
        )
    )]
    AlreadyLoaded { id: String },

    #[error("invalid compartment manifest at {path}: {message}")]
    #[diagnostic(
        code(akh::compartment::invalid_manifest),
        help(
            "The compartment.toml file is malformed or missing required fields. \
             Ensure it contains at least `id`, `name`, and `kind`."
        )
    )]
    InvalidManifest { path: String, message: String },

    #[error("compartment I/O error for {id}: {source}")]
    #[diagnostic(
        code(akh::compartment::io),
        help(
            "A filesystem operation failed while accessing the compartment directory. \
             Check that the path exists and has correct permissions."
        )
    )]
    Io {
        id: String,
        #[source]
        source: std::io::Error,
    },

    #[error("compartment kind mismatch: expected {expected}, got {actual}")]
    #[diagnostic(
        code(akh::compartment::kind_mismatch),
        help(
            "The operation requires a compartment of kind \"{expected}\", \
             but the compartment \"{actual}\" was provided instead."
        )
    )]
    KindMismatch { expected: String, actual: String },

    #[error("disjoint context conflict: {ctx_a} and {ctx_b} are declared disjoint")]
    #[diagnostic(
        code(akh::compartment::disjoint_conflict),
        help(
            "Contexts \"{ctx_a}\" and \"{ctx_b}\" are declared disjoint via `ctx:disjoint`. \
             A triple cannot belong to both. Choose one context or remove the disjointness \
             declaration."
        )
    )]
    DisjointConflict { ctx_a: String, ctx_b: String },

    #[error("context cycle detected: {context} eventually specializes itself")]
    #[diagnostic(
        code(akh::compartment::context_cycle),
        help(
            "Adding `ctx:specializes` from \"{context}\" would create a cycle in the \
             context hierarchy. Microtheory inheritance must be acyclic."
        )
    )]
    ContextCycle { context: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Store(#[from] StoreError),
}

/// Result type for compartment operations.
pub type CompartmentResult<T> = Result<T, CompartmentError>;
