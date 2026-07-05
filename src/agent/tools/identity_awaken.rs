//! Identity awakening tool: assert persona/identity into the engine.
//!
//! The LLM parses persona source (IDENTITY.md / SOUL.md / USER.md) itself and
//! emits the structured character fields here; this tool runs the deterministic
//! [`ritual_of_awakening`], which constructs and persists the psyche + identity
//! symbol + provenance. High-stakes (mutates the akh's identity), so `Dangerous`.

use std::collections::HashSet;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::bootstrap::identity::{
    ritual_of_awakening, CharacterKnowledge, CultureOrigin, IdentityError,
};
use crate::bootstrap::purpose::{DreyfusLevel, EntityType, PurposeModel};
use crate::engine::Engine;

/// Awaken the akh's identity from parsed persona fields.
pub struct IdentityAwakenTool;

fn str_array_schema(desc: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "array",
        "description": desc,
        "items": {"type": "string"}
    })
}

/// Map a free-form label to an [`EntityType`] (defaults to `Unknown`).
fn parse_entity_type(s: Option<&str>) -> EntityType {
    match s.map(|v| v.trim().to_lowercase()).as_deref() {
        Some("deity") | Some("god") | Some("goddess") => EntityType::Deity,
        Some("fictional") | Some("fictional_character") | Some("character") => {
            EntityType::FictionalCharacter
        }
        Some("historical") | Some("historical_figure") | Some("person") => {
            EntityType::HistoricalFigure
        }
        Some("concept") => EntityType::Concept,
        _ => EntityType::Unknown,
    }
}

impl Tool for IdentityAwakenTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "identity_awaken".into(),
            description: "Awaken (or, with force, re-awaken) the akh's identity from parsed \
                          persona fields. Construct these from IDENTITY.md / SOUL.md."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "name".into(),
                    description: "The persona's name (e.g. \"Tecton\").".into(),
                    required: true,
                },
                ToolParam {
                    name: "description".into(),
                    description: "A description of who this persona is.".into(),
                    required: true,
                },
                ToolParam {
                    name: "purpose".into(),
                    description: "The persona's purpose / mission statement.".into(),
                    required: true,
                },
                ToolParam {
                    name: "entity_type".into(),
                    description: "deity | fictional | historical | concept (default: unknown)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "culture".into(),
                    description: "egyptian | greek | norse | latin | fictional (default: unknown)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "domains".into(),
                    description: "Array of domains of competence.".into(),
                    required: false,
                },
                ToolParam {
                    name: "traits".into(),
                    description: "Array of character traits.".into(),
                    required: false,
                },
                ToolParam {
                    name: "archetypes".into(),
                    description: "Array of archetypes (e.g. [\"Architect\", \"Creator\"]).".into(),
                    required: false,
                },
                ToolParam {
                    name: "force".into(),
                    description: "Re-awaken even if already awakened (default false).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let name = input.require("name", "identity_awaken")?.to_string();
        let description = input.require("description", "identity_awaken")?.to_string();
        let purpose_text = input.require("purpose", "identity_awaken")?.to_string();
        let domains = input.get_str_list("domains");
        let traits = input.get_str_list("traits");
        let archetypes = input.get_str_list("archetypes");
        let force = input.get_bool("force").unwrap_or(false);

        let culture = input
            .get("culture")
            .and_then(CultureOrigin::from_label)
            .unwrap_or(CultureOrigin::Unknown);

        let character = CharacterKnowledge {
            name: name.clone(),
            entity_type: parse_entity_type(input.get("entity_type")),
            culture,
            description,
            domains: domains.clone(),
            traits,
            archetypes,
        };

        let purpose = PurposeModel {
            domain: domains.first().cloned().unwrap_or_else(|| "general".into()),
            competence_level: DreyfusLevel::Expert,
            seed_concepts: domains,
            description: purpose_text,
        };

        match ritual_of_awakening(&character, &purpose, engine, force) {
            Ok(result) => Ok(ToolOutput::ok(format!(
                "Awakened identity \"{}\" ({} provenance record(s)).",
                result.chosen_name,
                result.provenance_ids.len()
            ))),
            Err(IdentityError::AlreadyAwakened) => Ok(ToolOutput::err(
                "this akh is already awakened; pass force=true only if you intend to re-awaken it",
            )),
            Err(e) => Err(AgentError::ToolExecution {
                tool_name: "identity_awaken".into(),
                message: e.to_string(),
            }),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "identity_awaken".into(),
            description: "Awaken (or force re-awaken) the akh's identity from parsed persona fields."
                .into(),
            parameters: vec![
                ToolParamSchema::required("name", "The persona's name."),
                ToolParamSchema::required("description", "Who this persona is."),
                ToolParamSchema::required("purpose", "The persona's purpose / mission."),
                ToolParamSchema::optional(
                    "entity_type",
                    "deity | fictional | historical | concept (default unknown).",
                ),
                ToolParamSchema::optional(
                    "culture",
                    "egyptian | greek | norse | latin | fictional (default unknown).",
                ),
                ToolParamSchema::optional_schema(
                    "domains",
                    "Domains of competence.",
                    str_array_schema("Domains of competence."),
                ),
                ToolParamSchema::optional_schema(
                    "traits",
                    "Character traits.",
                    str_array_schema("Character traits."),
                ),
                ToolParamSchema::optional_schema(
                    "archetypes",
                    "Archetypes.",
                    str_array_schema("Archetypes."),
                ),
                ToolParamSchema::optional("force", "Re-awaken even if already awakened."),
            ],
            danger: DangerInfo {
                level: DangerLevel::Dangerous,
                capabilities: HashSet::from([Capability::WriteKg]),
                description: "Mutates the akh's psyche and identity in the engine.".into(),
                shadow_triggers: vec!["awaken".into(), "identity".into(), "ritual".into()],
            },
            source: ToolSource::Native,
        }
    }
}
