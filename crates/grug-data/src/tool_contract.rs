//! Content-addressed snapshots of Akh's live tool contracts.
//!
//! Training data must describe the tools that the runtime really advertises.
//! Keeping a second, simplified catalog is especially dangerous for MCP tools:
//! arrays and objects become strings, and a small model learns calls that can
//! never cross the provider boundary.  This module accepts any serializable
//! runtime manifest, retains its nested JSON-Schema fragments, normalizes
//! unordered metadata, and gives the resulting snapshot a stable BLAKE3 id.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

/// Serialized contract version. Bump this when snapshot or validation
/// semantics change in a way that can alter generated examples.
pub const TOOL_CONTRACT_SCHEMA_VERSION: u32 = 1;

/// Serialized contract-delta version. A delta is evidence about two complete
/// snapshots; it never replaces either content-addressed endpoint.
pub const TOOL_CONTRACT_DELTA_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractParameter {
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub required: bool,
    #[serde(default = "default_param_type")]
    pub param_type: String,
    /// Exact per-property JSON-Schema fragment advertised by the runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
}

fn default_param_type() -> String {
    "string".into()
}

impl ContractParameter {
    fn effective_schema(&self) -> Value {
        self.schema
            .clone()
            .unwrap_or_else(|| serde_json::json!({"type": self.param_type}))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContract {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub parameters: Vec<ContractParameter>,
    /// Preserved rather than re-modelled so new runtime capabilities remain
    /// visible to data policy without requiring a grug-data release first.
    #[serde(default)]
    pub danger: Value,
    #[serde(default)]
    pub source: Value,
}

impl ToolContract {
    /// Reconstruct the object schema presented to a model for this tool.
    pub fn input_schema(&self) -> Value {
        let properties: Map<String, Value> = self
            .parameters
            .iter()
            .map(|parameter| {
                let mut schema = parameter.effective_schema();
                if let Some(object) = schema.as_object_mut() {
                    object
                        .entry("description")
                        .or_insert_with(|| Value::String(parameter.description.clone()));
                }
                (parameter.name.clone(), schema)
            })
            .collect();
        let required: Vec<&str> = self
            .parameters
            .iter()
            .filter(|parameter| parameter.required)
            .map(|parameter| parameter.name.as_str())
            .collect();
        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": false
        })
    }

    /// Exact OpenAI-style definition used by Akh's chat backends and the
    /// canonical BabyLLM prompt renderer.
    pub fn openai_definition(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema()
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContractSnapshot {
    pub schema_version: u32,
    /// `blake3:<hex>` over the normalized schema version + tool list.
    pub manifest_hash: String,
    pub tools: Vec<ToolContract>,
}

/// One same-name contract whose normalized contents changed between
/// snapshots. `model_surface_changed` is derived from the exact OpenAI tool
/// definition rendered into BabyLLM prompts. A false value therefore means
/// that only Akh-side metadata (for example danger/source provenance or a
/// redundant type hint) changed and weight adaptation is not justified by
/// the model-visible ABI alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChangedToolContract {
    pub name: String,
    pub model_surface_changed: bool,
}

/// Deterministic, content-addressed classification of an old and new live tool
/// catalog. Names in `added`, `removed`, `changed`, and `unchanged` are
/// mutually exclusive and collectively cover the union of both snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolContractDelta {
    pub schema_version: u32,
    pub previous_contract_hash: String,
    pub current_contract_hash: String,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<ChangedToolContract>,
    pub unchanged: Vec<String>,
    /// BLAKE3 over every preceding field.
    pub delta_hash: String,
}

impl ToolContractDelta {
    pub fn between(
        previous: &ToolContractSnapshot,
        current: &ToolContractSnapshot,
    ) -> Result<Self, ToolContractError> {
        previous.verify_integrity()?;
        current.verify_integrity()?;

        let previous_by_name: BTreeMap<_, _> = previous
            .tools
            .iter()
            .map(|tool| (tool.name.as_str(), tool))
            .collect();
        let current_by_name: BTreeMap<_, _> = current
            .tools
            .iter()
            .map(|tool| (tool.name.as_str(), tool))
            .collect();
        let names: BTreeSet<_> = previous_by_name
            .keys()
            .chain(current_by_name.keys())
            .copied()
            .collect();

        let mut delta = Self {
            schema_version: TOOL_CONTRACT_DELTA_SCHEMA_VERSION,
            previous_contract_hash: previous.manifest_hash.clone(),
            current_contract_hash: current.manifest_hash.clone(),
            added: Vec::new(),
            removed: Vec::new(),
            changed: Vec::new(),
            unchanged: Vec::new(),
            delta_hash: String::new(),
        };
        for name in names {
            match (previous_by_name.get(name), current_by_name.get(name)) {
                (None, Some(_)) => delta.added.push(name.to_string()),
                (Some(_), None) => delta.removed.push(name.to_string()),
                (Some(previous), Some(current)) if previous == current => {
                    delta.unchanged.push(name.to_string());
                }
                (Some(previous), Some(current)) => {
                    delta.changed.push(ChangedToolContract {
                        name: name.to_string(),
                        model_surface_changed: previous.openai_definition()
                            != current.openai_definition(),
                    });
                }
                (None, None) => unreachable!("name came from one of the snapshots"),
            }
        }
        delta.delta_hash = delta.compute_hash()?;
        Ok(delta)
    }

    pub fn compute_hash(&self) -> Result<String, ToolContractError> {
        let payload = serde_json::to_vec(&serde_json::json!({
            "schema_version": self.schema_version,
            "previous_contract_hash": self.previous_contract_hash,
            "current_contract_hash": self.current_contract_hash,
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "unchanged": self.unchanged,
        }))
        .map_err(|error| ToolContractError::SnapshotSerialize {
            message: error.to_string(),
        })?;
        Ok(format!("blake3:{}", blake3::hash(&payload).to_hex()))
    }

    /// Recompute the complete classification from both endpoint snapshots.
    /// This catches edited name lists as well as a stale/tampered delta hash.
    pub fn verify(
        &self,
        previous: &ToolContractSnapshot,
        current: &ToolContractSnapshot,
    ) -> Result<(), ToolContractError> {
        if self.schema_version != TOOL_CONTRACT_DELTA_SCHEMA_VERSION {
            return Err(ToolContractError::UnsupportedDeltaVersion {
                actual: self.schema_version,
                expected: TOOL_CONTRACT_DELTA_SCHEMA_VERSION,
            });
        }
        let expected = Self::between(previous, current)?;
        if self != &expected {
            return Err(ToolContractError::DeltaMismatch {
                declared: self.delta_hash.clone(),
                computed: expected.delta_hash,
            });
        }
        Ok(())
    }

    /// Current tools needing extra positive examples. Removed tools are
    /// handled by explicit absence examples and never appear in this set.
    pub fn model_target_tools(&self) -> BTreeSet<String> {
        self.added
            .iter()
            .cloned()
            .chain(
                self.changed
                    .iter()
                    .filter(|change| change.model_surface_changed)
                    .map(|change| change.name.clone()),
            )
            .collect()
    }

    pub fn model_adaptation_required(&self) -> bool {
        !self.added.is_empty()
            || !self.removed.is_empty()
            || self
                .changed
                .iter()
                .any(|change| change.model_surface_changed)
    }

    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchemaIssueKind {
    UnknownTool,
    ArgumentsNotObject,
    MissingRequired,
    UnexpectedProperty,
    TypeMismatch,
    EnumMismatch,
    Cardinality,
    Pattern,
    OneOfMismatch,
    Range,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaIssue {
    /// JSON-path-like location, rooted at `$`.
    pub path: String,
    pub kind: SchemaIssueKind,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArgumentValidation {
    pub valid: bool,
    pub issues: Vec<SchemaIssue>,
}

impl ArgumentValidation {
    fn from_issues(issues: Vec<SchemaIssue>) -> Self {
        Self {
            valid: issues.is_empty(),
            issues,
        }
    }
}

#[derive(Debug, Error)]
pub enum ToolContractError {
    #[error("could not serialize runtime tool manifest at index {index}: {message}")]
    Serialize { index: usize, message: String },
    #[error("runtime tool manifest at index {index} has an incompatible shape: {message}")]
    InvalidShape { index: usize, message: String },
    #[error("tool contract name must not be empty")]
    EmptyToolName,
    #[error("duplicate tool contract `{name}`")]
    DuplicateTool { name: String },
    #[error("tool `{tool}` has an empty parameter name")]
    EmptyParameterName { tool: String },
    #[error("tool `{tool}` has duplicate parameter `{parameter}`")]
    DuplicateParameter { tool: String, parameter: String },
    #[error("could not serialize normalized tool contracts: {message}")]
    SnapshotSerialize { message: String },
    #[error("unsupported tool contract schema version {actual}; expected {expected}")]
    UnsupportedVersion { actual: u32, expected: u32 },
    #[error("tool contract snapshot hash mismatch: declared {declared}, computed {computed}")]
    HashMismatch { declared: String, computed: String },
    #[error("unsupported tool contract delta schema version {actual}; expected {expected}")]
    UnsupportedDeltaVersion { actual: u32, expected: u32 },
    #[error(
        "tool contract delta differs from its endpoint snapshots: declared {declared}, computed {computed}"
    )]
    DeltaMismatch { declared: String, computed: String },
}

impl ToolContractSnapshot {
    /// Snapshot concrete runtime manifests without coupling grug-data to the
    /// Akh runtime crate (which already depends on grug-data).
    pub fn from_serializable<T: Serialize>(manifests: &[T]) -> Result<Self, ToolContractError> {
        let mut tools = Vec::with_capacity(manifests.len());
        for (index, manifest) in manifests.iter().enumerate() {
            let value =
                serde_json::to_value(manifest).map_err(|error| ToolContractError::Serialize {
                    index,
                    message: error.to_string(),
                })?;
            let contract =
                serde_json::from_value(value).map_err(|error| ToolContractError::InvalidShape {
                    index,
                    message: error.to_string(),
                })?;
            tools.push(contract);
        }
        Self::from_contracts(tools)
    }

    pub fn from_contracts(mut tools: Vec<ToolContract>) -> Result<Self, ToolContractError> {
        normalize_and_validate(&mut tools)?;
        let payload = serde_json::to_vec(&serde_json::json!({
            "schema_version": TOOL_CONTRACT_SCHEMA_VERSION,
            "tools": tools
        }))
        .map_err(|error| ToolContractError::SnapshotSerialize {
            message: error.to_string(),
        })?;
        Ok(Self {
            schema_version: TOOL_CONTRACT_SCHEMA_VERSION,
            manifest_hash: format!("blake3:{}", blake3::hash(&payload).to_hex()),
            tools,
        })
    }

    /// Convert the OpenAI-style tool definitions handed to a chat backend
    /// into contracts usable by the same recursive argument validator.
    /// Runtime prompts intentionally lack Akh-only danger/source metadata;
    /// those fields are represented as null rather than invented.
    pub fn from_openai_tools(definitions: &[Value]) -> Result<Self, ToolContractError> {
        let mut contracts = Vec::with_capacity(definitions.len());
        for (index, definition) in definitions.iter().enumerate() {
            let function =
                definition
                    .get("function")
                    .ok_or_else(|| ToolContractError::InvalidShape {
                        index,
                        message: "OpenAI tool definition has no function object".into(),
                    })?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| ToolContractError::InvalidShape {
                    index,
                    message: "OpenAI tool function has no string name".into(),
                })?;
            let schema = function
                .get("parameters")
                .and_then(Value::as_object)
                .ok_or_else(|| ToolContractError::InvalidShape {
                    index,
                    message: "OpenAI tool function has no object parameters schema".into(),
                })?;
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let required: BTreeSet<&str> = schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let parameters = properties
                .into_iter()
                .map(|(parameter_name, parameter_schema)| ContractParameter {
                    description: parameter_schema
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .into(),
                    required: required.contains(parameter_name.as_str()),
                    param_type: parameter_schema
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("string")
                        .into(),
                    name: parameter_name,
                    schema: Some(parameter_schema),
                })
                .collect();
            contracts.push(ToolContract {
                name: name.into(),
                description: function
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .into(),
                parameters,
                danger: Value::Null,
                source: Value::Null,
            });
        }
        Self::from_contracts(contracts)
    }

    pub fn tool(&self, name: &str) -> Option<&ToolContract> {
        self.tools.iter().find(|tool| tool.name == name)
    }

    /// Recompute the normalized snapshot id after loading a file. This catches
    /// stale or manually edited contracts before they can label a corpus with
    /// an identity that no longer describes its schemas.
    pub fn verify_integrity(&self) -> Result<(), ToolContractError> {
        if self.schema_version != TOOL_CONTRACT_SCHEMA_VERSION {
            return Err(ToolContractError::UnsupportedVersion {
                actual: self.schema_version,
                expected: TOOL_CONTRACT_SCHEMA_VERSION,
            });
        }
        let recomputed = Self::from_contracts(self.tools.clone())?;
        if recomputed.manifest_hash != self.manifest_hash {
            return Err(ToolContractError::HashMismatch {
                declared: self.manifest_hash.clone(),
                computed: recomputed.manifest_hash,
            });
        }
        Ok(())
    }

    /// Validate one concrete call against the exact captured contract.
    /// Unknown top-level arguments are rejected even though JSON Schema's
    /// default is permissive: Akh's manifest parameter list is the callable
    /// boundary and accepting invented fields makes tool-use scores misleading.
    pub fn validate_call(&self, name: &str, arguments: &Value) -> ArgumentValidation {
        let Some(tool) = self.tool(name) else {
            return ArgumentValidation::from_issues(vec![SchemaIssue {
                path: "$".into(),
                kind: SchemaIssueKind::UnknownTool,
                message: format!(
                    "tool `{name}` is absent from snapshot {}",
                    self.manifest_hash
                ),
            }]);
        };
        let Some(arguments) = arguments.as_object() else {
            return ArgumentValidation::from_issues(vec![SchemaIssue {
                path: "$".into(),
                kind: SchemaIssueKind::ArgumentsNotObject,
                message: "tool arguments must be a JSON object".into(),
            }]);
        };

        let parameters: BTreeMap<&str, &ContractParameter> = tool
            .parameters
            .iter()
            .map(|parameter| (parameter.name.as_str(), parameter))
            .collect();
        let mut issues = Vec::new();
        for parameter in tool
            .parameters
            .iter()
            .filter(|parameter| parameter.required)
        {
            if !arguments.contains_key(&parameter.name) {
                issues.push(SchemaIssue {
                    path: format!("$.{}", parameter.name),
                    kind: SchemaIssueKind::MissingRequired,
                    message: "required parameter is missing".into(),
                });
            }
        }
        for (name, value) in arguments {
            let Some(parameter) = parameters.get(name.as_str()) else {
                issues.push(SchemaIssue {
                    path: format!("$.{name}"),
                    kind: SchemaIssueKind::UnexpectedProperty,
                    message: "parameter is not declared by the runtime contract".into(),
                });
                continue;
            };
            validate_schema(
                value,
                &parameter.effective_schema(),
                &format!("$.{name}"),
                &mut issues,
            );
        }
        ArgumentValidation::from_issues(issues)
    }
}

fn normalize_and_validate(tools: &mut Vec<ToolContract>) -> Result<(), ToolContractError> {
    for tool in tools.iter_mut() {
        if tool.name.trim().is_empty() {
            return Err(ToolContractError::EmptyToolName);
        }
        let mut parameters = BTreeSet::new();
        for parameter in &mut tool.parameters {
            if parameter.name.trim().is_empty() {
                return Err(ToolContractError::EmptyParameterName {
                    tool: tool.name.clone(),
                });
            }
            if !parameters.insert(parameter.name.clone()) {
                return Err(ToolContractError::DuplicateParameter {
                    tool: tool.name.clone(),
                    parameter: parameter.name.clone(),
                });
            }
            if let Some(schema) = &mut parameter.schema {
                *schema = canonicalize_json(schema);
            }
        }
        tool.parameters
            .sort_by(|left, right| left.name.cmp(&right.name));
        normalize_danger(&mut tool.danger);
        tool.danger = canonicalize_json(&tool.danger);
        tool.source = canonicalize_json(&tool.source);
    }
    tools.sort_by(|left, right| left.name.cmp(&right.name));
    for pair in tools.windows(2) {
        if pair[0].name == pair[1].name {
            return Err(ToolContractError::DuplicateTool {
                name: pair[0].name.clone(),
            });
        }
    }
    Ok(())
}

fn normalize_danger(danger: &mut Value) {
    let Some(capabilities) = danger
        .as_object_mut()
        .and_then(|object| object.get_mut("capabilities"))
        .and_then(Value::as_array_mut)
    else {
        return;
    };
    capabilities.sort_by_key(|value| value.to_string());
}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .iter()
                .map(|(key, value)| (key.clone(), canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.iter().map(canonicalize_json).collect()),
        other => other.clone(),
    }
}

fn validate_schema(value: &Value, schema: &Value, path: &str, issues: &mut Vec<SchemaIssue>) {
    if let Some(variants) = schema.get("oneOf").and_then(Value::as_array) {
        let matches = variants
            .iter()
            .filter(|variant| {
                let mut variant_issues = Vec::new();
                validate_schema(value, variant, path, &mut variant_issues);
                variant_issues.is_empty()
            })
            .count();
        if matches != 1 {
            issues.push(SchemaIssue {
                path: path.into(),
                kind: SchemaIssueKind::OneOfMismatch,
                message: format!("value matched {matches} oneOf branches; expected exactly one"),
            });
        }
        return;
    }

    if let Some(allowed) = schema.get("enum").and_then(Value::as_array) {
        if !allowed.iter().any(|candidate| candidate == value) {
            issues.push(SchemaIssue {
                path: path.into(),
                kind: SchemaIssueKind::EnumMismatch,
                message: "value is not one of the declared enum variants".into(),
            });
            return;
        }
    }

    if let Some(expected) = schema.get("type") {
        let matches = match expected {
            Value::String(expected) => value_matches_type(value, expected),
            Value::Array(expected) => expected
                .iter()
                .filter_map(Value::as_str)
                .any(|expected| value_matches_type(value, expected)),
            _ => true,
        };
        if !matches {
            issues.push(SchemaIssue {
                path: path.into(),
                kind: SchemaIssueKind::TypeMismatch,
                message: format!("expected JSON type {expected}, got {}", value_type(value)),
            });
            return;
        }
    }

    if let Some(array) = value.as_array() {
        if let Some(minimum) = schema.get("minItems").and_then(Value::as_u64) {
            if array.len() < minimum as usize {
                issues.push(SchemaIssue {
                    path: path.into(),
                    kind: SchemaIssueKind::Cardinality,
                    message: format!("expected at least {minimum} items, got {}", array.len()),
                });
            }
        }
        if let Some(maximum) = schema.get("maxItems").and_then(Value::as_u64) {
            if array.len() > maximum as usize {
                issues.push(SchemaIssue {
                    path: path.into(),
                    kind: SchemaIssueKind::Cardinality,
                    message: format!("expected at most {maximum} items, got {}", array.len()),
                });
            }
        }
        if let Some(item_schema) = schema.get("items") {
            for (index, item) in array.iter().enumerate() {
                validate_schema(item, item_schema, &format!("{path}[{index}]"), issues);
            }
        }
    }

    if let Some(object) = value.as_object() {
        let properties = schema.get("properties").and_then(Value::as_object);
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for name in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(name) {
                    issues.push(SchemaIssue {
                        path: format!("{path}.{name}"),
                        kind: SchemaIssueKind::MissingRequired,
                        message: "required property is missing".into(),
                    });
                }
            }
        }
        for (name, nested) in object {
            if let Some(property_schema) = properties.and_then(|props| props.get(name)) {
                validate_schema(nested, property_schema, &format!("{path}.{name}"), issues);
                continue;
            }
            match schema.get("additionalProperties") {
                Some(Value::Bool(false)) => issues.push(SchemaIssue {
                    path: format!("{path}.{name}"),
                    kind: SchemaIssueKind::UnexpectedProperty,
                    message: "additional property is forbidden".into(),
                }),
                Some(additional @ Value::Object(_)) => {
                    validate_schema(nested, additional, &format!("{path}.{name}"), issues);
                }
                _ => {}
            }
        }
    }

    if let Some(text) = value.as_str() {
        if let Some(minimum) = schema.get("minLength").and_then(Value::as_u64) {
            if text.chars().count() < minimum as usize {
                issues.push(SchemaIssue {
                    path: path.into(),
                    kind: SchemaIssueKind::Cardinality,
                    message: format!("string is shorter than minLength {minimum}"),
                });
            }
        }
        if let Some(maximum) = schema.get("maxLength").and_then(Value::as_u64) {
            if text.chars().count() > maximum as usize {
                issues.push(SchemaIssue {
                    path: path.into(),
                    kind: SchemaIssueKind::Cardinality,
                    message: format!("string is longer than maxLength {maximum}"),
                });
            }
        }
        if let Some(pattern) = schema.get("pattern").and_then(Value::as_str) {
            match regex::Regex::new(pattern) {
                Ok(pattern) if !pattern.is_match(text) => issues.push(SchemaIssue {
                    path: path.into(),
                    kind: SchemaIssueKind::Pattern,
                    message: "string does not match the declared pattern".into(),
                }),
                Err(error) => issues.push(SchemaIssue {
                    path: path.into(),
                    kind: SchemaIssueKind::Pattern,
                    message: format!("runtime contract contains an invalid pattern: {error}"),
                }),
                Ok(_) => {}
            }
        }
    }

    if let Some(number) = value.as_f64() {
        if schema
            .get("minimum")
            .and_then(Value::as_f64)
            .is_some_and(|minimum| number < minimum)
        {
            issues.push(SchemaIssue {
                path: path.into(),
                kind: SchemaIssueKind::Range,
                message: "number is below the declared minimum".into(),
            });
        }
        if schema
            .get("maximum")
            .and_then(Value::as_f64)
            .is_some_and(|maximum| number > maximum)
        {
            issues.push(SchemaIssue {
                path: path.into(),
                kind: SchemaIssueKind::Range,
                message: "number is above the declared maximum".into(),
            });
        }
    }
}

fn value_matches_type(value: &Value, expected: &str) -> bool {
    match expected {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        _ => true,
    }
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.as_i64().is_some() || number.as_u64().is_some() => {
            "integer"
        }
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(name: &str, capabilities: Vec<&str>, parameters: Value) -> Value {
        serde_json::json!({
            "name": name,
            "description": "test tool",
            "parameters": parameters,
            "danger": {
                "level": "Cautious",
                "capabilities": capabilities,
                "description": "test",
                "shadow_triggers": []
            },
            "source": "Native"
        })
    }

    #[test]
    fn snapshot_is_independent_of_runtime_hashset_and_tool_order() {
        let first = vec![
            manifest(
                "zeta",
                vec!["WriteKg", "MemoryAccess"],
                serde_json::json!([]),
            ),
            manifest("alpha", vec!["ReadKg"], serde_json::json!([])),
        ];
        let second = vec![
            manifest("alpha", vec!["ReadKg"], serde_json::json!([])),
            manifest(
                "zeta",
                vec!["MemoryAccess", "WriteKg"],
                serde_json::json!([]),
            ),
        ];
        let first = ToolContractSnapshot::from_serializable(&first).unwrap();
        let second = ToolContractSnapshot::from_serializable(&second).unwrap();
        assert_eq!(first.manifest_hash, second.manifest_hash);
        assert_eq!(first.tools, second.tools);
    }

    #[test]
    fn nested_mcp_values_are_checked_without_flattening() {
        let contracts = ToolContractSnapshot::from_serializable(&[manifest(
            "mcp__seshat__seshat_ingest",
            vec!["Network"],
            serde_json::json!([
                {
                    "name": "title", "description": "title", "required": true,
                    "param_type": "string", "schema": {"type": "string"}
                },
                {
                    "name": "tags", "description": "tags", "required": false,
                    "param_type": "array", "schema": {
                        "type": "array", "items": {"type": "string"}
                    }
                },
                {
                    "name": "metadata", "description": "metadata", "required": false,
                    "param_type": "object", "schema": {
                        "type": "object",
                        "properties": {"language": {"type": "string"}},
                        "additionalProperties": false
                    }
                }
            ]),
        )])
        .unwrap();

        let valid = contracts.validate_call(
            "mcp__seshat__seshat_ingest",
            &serde_json::json!({
                "title": "Book",
                "tags": ["textbook", "physics"],
                "metadata": {"language": "de"}
            }),
        );
        assert!(valid.valid, "{:?}", valid.issues);

        let invalid = contracts.validate_call(
            "mcp__seshat__seshat_ingest",
            &serde_json::json!({
                "title": "Book",
                "tags": "textbook,physics",
                "metadata": {"language": "de", "invented": true}
            }),
        );
        assert!(!invalid.valid);
        assert!(invalid.issues.iter().any(|issue| issue.path == "$.tags"));
        assert!(
            invalid
                .issues
                .iter()
                .any(|issue| issue.path == "$.metadata.invented")
        );
    }

    #[test]
    fn required_citation_items_are_enforced() {
        let contracts = ToolContractSnapshot::from_serializable(&[manifest(
            "study_note_append",
            vec!["MemoryAccess"],
            serde_json::json!([
                {
                    "name": "body", "description": "body", "required": true,
                    "param_type": "string"
                },
                {
                    "name": "citations", "description": "citations", "required": true,
                    "param_type": "array", "schema": {
                        "type": "array", "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "locator": {"type": "string"}
                            },
                            "required": ["source_id", "locator"],
                            "additionalProperties": false
                        }
                    }
                }
            ]),
        )])
        .unwrap();

        assert!(
            contracts
                .validate_call(
                    "study_note_append",
                    &serde_json::json!({"body": "fact", "citations": []}),
                )
                .issues
                .iter()
                .any(|issue| issue.kind == SchemaIssueKind::Cardinality)
        );
        assert!(
            contracts
                .validate_call(
                    "study_note_append",
                    &serde_json::json!({
                        "body": "fact",
                        "citations": [{"source_id": "book"}]
                    }),
                )
                .issues
                .iter()
                .any(|issue| issue.path == "$.citations[0].locator")
        );
    }

    #[test]
    fn string_length_and_pattern_constraints_are_enforced() {
        let contracts = ToolContractSnapshot::from_serializable(&[manifest(
            "study_note_revise",
            vec!["MemoryAccess"],
            serde_json::json!([{
                "name": "expected_revision",
                "description": "current head",
                "required": true,
                "param_type": "string",
                "schema": {
                    "type": "string",
                    "minLength": 64,
                    "maxLength": 64,
                    "pattern": "^[0-9a-f]{64}$"
                }
            }]),
        )])
        .unwrap();
        let valid = "a".repeat(64);
        assert!(
            contracts
                .validate_call(
                    "study_note_revise",
                    &serde_json::json!({"expected_revision": valid}),
                )
                .valid
        );
        let invalid = contracts.validate_call(
            "study_note_revise",
            &serde_json::json!({"expected_revision": "Z".repeat(65)}),
        );
        assert!(!invalid.valid);
        assert!(
            invalid
                .issues
                .iter()
                .any(|issue| issue.kind == SchemaIssueKind::Cardinality)
        );
        assert!(
            invalid
                .issues
                .iter()
                .any(|issue| issue.kind == SchemaIssueKind::Pattern)
        );
    }

    #[test]
    fn edited_snapshot_fails_integrity_check() {
        let mut snapshot = ToolContractSnapshot::from_serializable(&[manifest(
            "reader",
            vec!["ReadKg"],
            serde_json::json!([]),
        )])
        .unwrap();
        snapshot.tools[0].description = "silently changed".into();
        assert!(matches!(
            snapshot.verify_integrity(),
            Err(ToolContractError::HashMismatch { .. })
        ));
    }

    #[test]
    fn openai_prompt_tools_reuse_nested_argument_validation() {
        let snapshot = ToolContractSnapshot::from_openai_tools(&[serde_json::json!({
            "type": "function",
            "function": {
                "name": "study_note_append",
                "description": "append evidence",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "citations": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "locator": {"type": "string"}
                                },
                                "required": ["source_id", "locator"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["citations"],
                    "additionalProperties": false
                }
            }
        })])
        .unwrap();
        assert!(
            snapshot
                .validate_call(
                    "study_note_append",
                    &serde_json::json!({
                        "citations": [{"source_id": "book", "locator": "p. 4"}]
                    })
                )
                .valid
        );
        assert!(
            !snapshot
                .validate_call(
                    "study_note_append",
                    &serde_json::json!({
                        "citations": [{"source_id": "book"}]
                    })
                )
                .valid
        );
    }

    #[test]
    fn contract_delta_separates_model_surface_from_metadata_changes() {
        let previous = ToolContractSnapshot::from_contracts(vec![
            ToolContract {
                name: "metadata_only".into(),
                description: "Read a page".into(),
                parameters: vec![],
                danger: serde_json::json!({"level":"Safe"}),
                source: serde_json::json!("Native"),
            },
            ToolContract {
                name: "removed".into(),
                description: "Old provider".into(),
                parameters: vec![],
                danger: Value::Null,
                source: Value::Null,
            },
            ToolContract {
                name: "schema_changed".into(),
                description: "Search".into(),
                parameters: vec![ContractParameter {
                    name: "query".into(),
                    description: "Query".into(),
                    required: true,
                    param_type: "string".into(),
                    schema: None,
                }],
                danger: Value::Null,
                source: Value::Null,
            },
            ToolContract {
                name: "stable".into(),
                description: "Stable".into(),
                parameters: vec![],
                danger: Value::Null,
                source: Value::Null,
            },
        ])
        .unwrap();
        let current = ToolContractSnapshot::from_contracts(vec![
            ToolContract {
                name: "added".into(),
                description: "New local reader".into(),
                parameters: vec![],
                danger: Value::Null,
                source: Value::Null,
            },
            ToolContract {
                name: "metadata_only".into(),
                description: "Read a page".into(),
                parameters: vec![],
                danger: serde_json::json!({"level":"Cautious"}),
                source: serde_json::json!("Native"),
            },
            ToolContract {
                name: "schema_changed".into(),
                description: "Search".into(),
                parameters: vec![ContractParameter {
                    name: "query".into(),
                    description: "Query".into(),
                    required: true,
                    param_type: "array".into(),
                    schema: Some(serde_json::json!({
                        "type": "array",
                        "items": {"type":"string"},
                        "minItems": 1
                    })),
                }],
                danger: Value::Null,
                source: Value::Null,
            },
            ToolContract {
                name: "stable".into(),
                description: "Stable".into(),
                parameters: vec![],
                danger: Value::Null,
                source: Value::Null,
            },
        ])
        .unwrap();

        let delta = ToolContractDelta::between(&previous, &current).unwrap();
        delta.verify(&previous, &current).unwrap();
        assert_eq!(delta.added, vec!["added"]);
        assert_eq!(delta.removed, vec!["removed"]);
        assert_eq!(delta.unchanged, vec!["stable"]);
        assert_eq!(
            delta.changed,
            vec![
                ChangedToolContract {
                    name: "metadata_only".into(),
                    model_surface_changed: false,
                },
                ChangedToolContract {
                    name: "schema_changed".into(),
                    model_surface_changed: true,
                },
            ]
        );
        assert_eq!(
            delta.model_target_tools(),
            BTreeSet::from(["added".into(), "schema_changed".into()])
        );
        assert!(delta.model_adaptation_required());
    }

    #[test]
    fn edited_contract_delta_fails_endpoint_verification() {
        let previous = ToolContractSnapshot::from_contracts(vec![ToolContract {
            name: "old".into(),
            description: "Old".into(),
            parameters: vec![],
            danger: Value::Null,
            source: Value::Null,
        }])
        .unwrap();
        let current = ToolContractSnapshot::from_contracts(vec![ToolContract {
            name: "new".into(),
            description: "New".into(),
            parameters: vec![],
            danger: Value::Null,
            source: Value::Null,
        }])
        .unwrap();
        let mut delta = ToolContractDelta::between(&previous, &current).unwrap();
        delta.removed.clear();
        assert!(matches!(
            delta.verify(&previous, &current),
            Err(ToolContractError::DeltaMismatch { .. })
        ));
    }
}
