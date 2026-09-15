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

/// Runtime -> session daemon: same-space recall on behalf of the active turn.
/// The session daemon owns the core principal, akh and session scope; the
/// runtime only supplies the query vector and its representation identity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecallRequest {
    /// Identity of the active job (`name` must be `neural_recall`).
    pub call: crate::ToolCall,
    pub space: SpanSpace,
    pub vector: Vec<u8>,
    pub limit: usize,
}
impl RecallRequest {
    pub const CALL_NAME: &'static str = "neural_recall";
    pub fn validate(&self) -> Result<()> {
        if self.call.name != Self::CALL_NAME {
            return Err(Fault::invalid("neural recall requires a neural_recall call"));
        }
        self.space.validate()?;
        if self.vector.len() != VECTOR_BYTES || !(1..=16).contains(&self.limit) {
            return Err(Fault::invalid(
                "neural recall requires 10000 bits and limit 1..16",
            ));
        }
        Ok(())
    }
}

/// `supported == false` means the session daemon or core cannot serve neural
/// recall for this session (missing capability or capture policy); the
/// runtime continues without recalled context.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RecallReply {
    pub supported: bool,
    #[serde(default)]
    pub hits: Vec<SpanHit>,
}

/// Runtime -> session daemon: captures for the active turn's accepted input.
/// Only user-input spans are accepted before the turn completes; the daemon
/// attaches them to the accepted turn when the core advertises support.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpanDelivery {
    /// Identity of the active job (`name` must be `neural_spans`).
    pub call: crate::ToolCall,
    pub captures: Vec<SpanCapture>,
}
impl SpanDelivery {
    pub const CALL_NAME: &'static str = "neural_spans";
    /// Validate against the turn's accepted input text.
    pub fn validate(&self, input_text: &str) -> Result<()> {
        if self.call.name != Self::CALL_NAME {
            return Err(Fault::invalid("span delivery requires a neural_spans call"));
        }
        if self.captures.len() > MAX_CAPTURES {
            return Err(Fault::invalid("at most four spans per accepted turn"));
        }
        for capture in &self.captures {
            if capture.source != SpanSource::User {
                return Err(Fault::invalid(
                    "only user-input spans can be delivered before the turn completes",
                ));
            }
            capture.text(input_text, "")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpanDeliveryReply {
    pub accepted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str) -> crate::ToolCall {
        crate::ToolCall {
            id: "neural-0".into(),
            session_id: "session".into(),
            turn_id: "turn".into(),
            generation: 1,
            name: name.into(),
            arguments: serde_json::json!({}),
        }
    }
    fn space() -> SpanSpace {
        SpanSpace {
            model_sha256: "1".repeat(64),
            tokenizer_sha256: "2".repeat(64),
            policy_sha256: "3".repeat(64),
            projector_sha256: "4".repeat(64),
        }
    }

    #[test]
    fn recall_requests_are_bounded_and_named() {
        let mut request = RecallRequest {
            call: call(RecallRequest::CALL_NAME),
            space: space(),
            vector: vec![0; VECTOR_BYTES],
            limit: 4,
        };
        assert!(request.validate().is_ok());
        request.limit = 17;
        assert!(request.validate().is_err());
        request.limit = 1;
        request.vector.pop();
        assert!(request.validate().is_err());
        request.vector.push(0);
        request.call.name = "read".into();
        assert!(request.validate().is_err());
    }

    #[test]
    fn deliveries_accept_only_user_spans_inside_the_input() {
        let capture = SpanCapture {
            space: space(),
            source: SpanSource::User,
            byte_start: 2,
            byte_end: 7,
            vector: vec![7; VECTOR_BYTES],
        };
        let mut delivery = SpanDelivery {
            call: call(SpanDelivery::CALL_NAME),
            captures: vec![capture.clone()],
        };
        assert!(delivery.validate("  cedar tree").is_ok());
        assert!(delivery.validate("  cé").is_err());
        delivery.captures[0].source = SpanSource::Assistant;
        assert!(delivery.validate("  cedar tree").is_err());
        delivery.captures = vec![capture; MAX_CAPTURES + 1];
        assert!(delivery.validate("  cedar tree").is_err());
        delivery.captures.truncate(1);
        delivery.call.name = "neural_recall".into();
        assert!(delivery.validate("  cedar tree").is_err());
        let reply: RecallReply = serde_json::from_str(r#"{"supported":false}"#).unwrap();
        assert!(reply.hits.is_empty());
    }
}
