//! Extraction backends.

pub mod api;
#[cfg(feature = "candle-backend")]
pub mod local;
pub mod regex;
