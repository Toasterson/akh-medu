//! Extraction backends.

pub mod api;
#[cfg(feature = "nlu-llm")]
pub mod local;
pub mod regex;
