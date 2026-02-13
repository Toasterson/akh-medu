//! Knowledge compartmentalization: isolates knowledge by purpose.
//!
//! Compartments scope triples into named groups (skills, projects, core modules)
//! so they can be loaded, unloaded, and queried independently.

pub mod error;
pub mod manager;
pub mod manifest;
pub mod psyche;

pub use error::{CompartmentError, CompartmentResult};
pub use manager::CompartmentManager;
pub use manifest::{CompartmentKind, CompartmentManifest, CompartmentState};
pub use psyche::Psyche;
