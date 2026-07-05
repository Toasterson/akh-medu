//! Strongly-typed tool schemas, calls, and training examples.
//!
//! These types model the domain instead of passing JSON blobs around: a
//! [`ToolSchema`] fully describes what a valid [`ToolCall`] against it looks
//! like, so the eval scorer (`crate::eval`) can check schema adherence
//! structurally rather than by string-matching.

use serde::{Deserialize, Serialize};

use crate::error::{GrugError, GrugResult};

/// The type of a single tool parameter.
///
/// Mirrors the small set of JSON-schema-ish primitive types that plausible
/// fake tools need. `Enum` carries its closed set of legal string values so
/// the scorer can check membership, not just "is a string".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParamType {
    String,
    Integer,
    Number,
    Boolean,
    Enum(Vec<String>),
}

impl ParamType {
    /// Does `value` structurally match this type?
    ///
    /// Numbers are checked precisely: `Integer` requires a JSON number with
    /// no fractional part; `Number` accepts any JSON number (integer or
    /// float). This matters because a model that emits `"count": 3.0` for an
    /// integer parameter has made a real, catchable mistake.
    pub fn matches(&self, value: &serde_json::Value) -> bool {
        match self {
            ParamType::String => value.is_string(),
            ParamType::Integer => value.as_i64().is_some() || value.as_u64().is_some(),
            ParamType::Number => value.is_number(),
            ParamType::Boolean => value.is_boolean(),
            ParamType::Enum(variants) => value
                .as_str()
                .is_some_and(|s| variants.iter().any(|v| v == s)),
        }
    }
}

/// One parameter in a [`ToolSchema`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParamSpec {
    pub name: String,
    pub ty: ParamType,
    pub required: bool,
}

impl ParamSpec {
    pub fn new(name: impl Into<String>, ty: ParamType, required: bool) -> Self {
        Self {
            name: name.into(),
            ty,
            required,
        }
    }
}

/// The full schema of one callable tool: name, human description, and its
/// parameter list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub params: Vec<ParamSpec>,
}

impl ToolSchema {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        params: Vec<ParamSpec>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            params,
        }
    }

    pub fn param(&self, name: &str) -> Option<&ParamSpec> {
        self.params.iter().find(|p| p.name == name)
    }

    pub fn required_params(&self) -> impl Iterator<Item = &ParamSpec> {
        self.params.iter().filter(|p| p.required)
    }
}

/// A concrete invocation of a tool: its name plus a JSON object of arguments.
///
/// `arguments` is deliberately `serde_json::Value` (constrained to be an
/// object by construction — see [`ToolCall::new`]) rather than a fixed struct,
/// because the argument shape is defined per-tool by [`ToolSchema`], not
/// known at the Rust type level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Build a call, coercing a non-object `arguments` into an empty object.
    /// (Generator code always passes a `serde_json::Map`-backed `Value`, so
    /// this is a defensive normalization rather than the common path.)
    pub fn new(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        let arguments = if arguments.is_object() {
            arguments
        } else {
            serde_json::Value::Object(Default::default())
        };
        Self {
            name: name.into(),
            arguments,
        }
    }

    /// Arguments as a JSON object map, or an empty map if somehow not an
    /// object (should not happen given [`ToolCall::new`]).
    pub fn arguments_map(&self) -> serde_json::Map<String, serde_json::Value> {
        self.arguments.as_object().cloned().unwrap_or_default()
    }
}

/// One training/eval example: the tools offered to the model, the
/// natural-language request, the ground-truth call, and a plausible tool
/// response (useful for multi-turn SFT framing later).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Example {
    pub tools: Vec<ToolSchema>,
    pub user_request: String,
    pub expected_call: ToolCall,
    pub mock_response: String,
}

impl Example {
    /// Serialize to a single JSONL line (no trailing newline).
    pub fn to_jsonl(&self) -> GrugResult<String> {
        serde_json::to_string(self).map_err(|e| GrugError::SerializeFailed {
            message: e.to_string(),
        })
    }

    /// Parse a single JSONL line back into an `Example`.
    pub fn from_jsonl(line: &str) -> GrugResult<Self> {
        serde_json::from_str(line).map_err(|e| GrugError::DeserializeFailed {
            message: e.to_string(),
        })
    }

    /// The schema for this example's expected tool, if it is among the
    /// offered `tools` (it always should be, by construction).
    pub fn expected_schema(&self) -> Option<&ToolSchema> {
        self.tools.iter().find(|t| t.name == self.expected_call.name)
    }

    /// Render `expected_call` as a `{"name": ..., "arguments": {...}}` raw
    /// JSON string — a hand-constructible "known good" model output, useful
    /// for eval-scorer self-tests and doctests.
    pub fn expected_call_as_json_string(&self) -> String {
        serde_json::json!({
            "name": self.expected_call.name,
            "arguments": self.expected_call.arguments,
        })
        .to_string()
    }
}
