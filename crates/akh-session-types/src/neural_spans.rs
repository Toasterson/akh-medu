//! Version-one, same-space neural-span retrieval; not symbolic ItemMemory alignment.
use crate::{Fault, Result};
use serde::{Deserialize, Serialize};

pub const VECTOR_BYTES: usize = 1250;
pub const MAX_CAPTURES: usize = 4;
pub const MAX_TEXT_BYTES: usize = 8192;
pub const MAX_SESSION_SPANS: usize = 256;

/// Pins identify the complete immutable representation recipe. The policy
/// artifact includes tap layers, pooling, dimensions and arithmetic policy;
/// the projector artifact includes algorithm, seed and normalization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpanSpace {
    pub model_sha256: String,
    pub tokenizer_sha256: String,
    pub policy_sha256: String,
    pub projector_sha256: String,
}
impl SpanSpace {
    pub fn validate(&self) -> Result<()> {
        for pin in [
            &self.model_sha256,
            &self.tokenizer_sha256,
            &self.policy_sha256,
            &self.projector_sha256,
        ] {
            if pin.len() != 64
                || !pin
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err(Fault::invalid(
                    "span representation requires lowercase SHA-256 pins",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanSource {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SpanCapture {
    pub space: SpanSpace,
    pub source: SpanSource,
    pub byte_start: usize,
    pub byte_end: usize,
    pub vector: Vec<u8>,
}
impl SpanCapture {
    pub fn text<'a>(&self, user: &'a str, assistant: &'a str) -> Result<&'a str> {
        self.space.validate()?;
        if self.vector.len() != VECTOR_BYTES
            || self.byte_start >= self.byte_end
            || self.byte_end - self.byte_start > MAX_TEXT_BYTES
        {
            return Err(Fault::invalid("invalid bounded neural span"));
        }
        let text = match self.source {
            SpanSource::User => user,
            SpanSource::Assistant => assistant,
        };
        text.get(self.byte_start..self.byte_end)
            .ok_or_else(|| Fault::invalid("neural span is outside accepted UTF-8 text"))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpanQuery {
    pub akh: String,
    pub session_id: String,
    pub space: SpanSpace,
    pub vector: Vec<u8>,
    pub limit: usize,
}
impl SpanQuery {
    pub fn validate(&self) -> Result<()> {
        crate::validate_key(&self.akh)?;
        crate::validate_key(&self.session_id)?;
        self.space.validate()?;
        if self.vector.len() != VECTOR_BYTES || !(1..=16).contains(&self.limit) {
            return Err(Fault::invalid(
                "span query requires 10000 bits and limit 1..16",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpanHit {
    pub turn_id: String,
    pub source: SpanSource,
    pub byte_start: usize,
    pub byte_end: usize,
    pub text: String,
    pub hamming_distance: u32,
}
