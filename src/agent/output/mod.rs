//! Agent output pipeline: structured VSA results → natural language.
//!
//! Currently exposes [`candle_format`], which wraps the Candle LLM backend to
//! render `NarrativeSummary` and `PipelineOutput` values as fluent prose.

pub mod candle_format;

pub use candle_format::{
    CandleFormatError, CandleFormatResult, CandleFormatter, FormatOutput, FormatSource,
};
// FormatInput has a lifetime parameter — callers import it directly from candle_format.
