//! Schema-grounded SFT over every tool in a captured Akh contract.
//!
//! Stateful knowledge episodes intentionally concentrate on the learning
//! loop. This companion curriculum prevents that focus from turning into a
//! five-tool model: it synthesizes valid calls for every native/MCP contract,
//! includes exact argument values in the request, and offers distractors using
//! the same four-tool runtime budget as the compact backend.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use thiserror::Error;

use crate::{
    CapabilitySpec, CheckpointRef, ExamObservation, ExamTier, KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
    KnowledgeAction, KnowledgeSftRecord, SiblingSeparationEvidence, SiblingSeparationPolicy,
    ToolContract, ToolContractDelta, ToolContractError, ToolContractSnapshot, WorkflowLanguage,
    WorkflowStage, render_knowledge_prompt, validate_sibling_separation,
};

pub const TOOL_CURRICULUM_GENERATOR_VERSION: &str = "akh-tool-contract-sft-v2";
pub const TOOL_CONTRACT_DELTA_CURRICULUM_GENERATOR_VERSION: &str = "akh-tool-contract-delta-sft-v2";
pub const TOOL_CONTRACT_SIBLING_GENERATOR_VERSION: &str = "akh-tool-contract-sibling-sft-v2";
pub const TOOL_CONTRACT_SIBLING_SCHEMA_VERSION: u32 = 1;
pub const TOOL_CONTRACT_EXAM_SCHEMA_VERSION: u32 = 1;
pub const TOOL_CONTRACT_EXAM_GENERATOR_VERSION: &str = "akh-tool-contract-exam-v2";

/// Adaptive-controller families for the exact tools present in a runtime
/// snapshot. Keeping one family per tool lets a weak or drifting capability
/// steer curriculum selection before aggregate accuracy hides it.
pub fn tool_contract_capability_specs(
    contracts: &ToolContractSnapshot,
) -> Result<Vec<CapabilitySpec>, ToolCurriculumError> {
    contracts.verify_integrity()?;
    Ok(contracts
        .tools
        .iter()
        .map(|tool| {
            CapabilitySpec::new(format!("tool_contract.{}", tool.name), 1, 3)
                .with_initial_difficulty(3)
        })
        .collect())
}

#[derive(Debug, Clone)]
pub struct ToolCurriculumGenerator {
    contracts: ToolContractSnapshot,
    seed: u64,
    offered_tool_count: usize,
}

#[derive(Debug, Clone, Copy)]
enum ToolPromptDomain<'a> {
    Standard,
    AdaptiveSibling { source_exam_fingerprint: &'a str },
}

impl<'a> ToolPromptDomain<'a> {
    fn generator_version(self) -> &'static str {
        match self {
            Self::Standard => TOOL_CURRICULUM_GENERATOR_VERSION,
            Self::AdaptiveSibling { .. } => TOOL_CONTRACT_SIBLING_GENERATOR_VERSION,
        }
    }

    fn source_exam_fingerprint(self) -> Option<&'a str> {
        match self {
            Self::Standard => None,
            Self::AdaptiveSibling {
                source_exam_fingerprint,
            } => Some(source_exam_fingerprint),
        }
    }
}

impl ToolCurriculumGenerator {
    pub fn new(
        contracts: ToolContractSnapshot,
        seed: u64,
        offered_tool_count: usize,
    ) -> Result<Self, ToolCurriculumError> {
        contracts.verify_integrity()?;
        if contracts.tools.is_empty() {
            return Err(ToolCurriculumError::InvalidConfig(
                "tool snapshot is empty".into(),
            ));
        }
        if offered_tool_count == 0 || offered_tool_count > contracts.tools.len() {
            return Err(ToolCurriculumError::InvalidConfig(format!(
                "offered tool count must be in 1..={}, got {offered_tool_count}",
                contracts.tools.len()
            )));
        }
        Ok(Self {
            contracts,
            seed,
            offered_tool_count,
        })
    }

    pub fn generate(
        &self,
        run_id: &str,
        examples_per_tool: usize,
    ) -> Result<Vec<KnowledgeSftRecord>, ToolCurriculumError> {
        let tool_names = self
            .contracts
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        self.generate_for_tools(run_id, examples_per_tool, &tool_names)
    }

    /// Generate extra positive examples for a deterministic subset of the
    /// current contract while still drawing distractors from the complete live
    /// catalog. This is the contract-delta adaptation seam: added or
    /// model-visible changed tools can receive more weight without teaching a
    /// reduced runtime tool universe.
    pub fn generate_for_tools(
        &self,
        run_id: &str,
        examples_per_tool: usize,
        tool_names: &BTreeSet<String>,
    ) -> Result<Vec<KnowledgeSftRecord>, ToolCurriculumError> {
        self.generate_for_tools_with_domain(
            run_id,
            examples_per_tool,
            tool_names,
            ToolPromptDomain::Standard,
        )
    }

    /// Domain-separated prompt bank for a controller-selected tool frontier.
    /// The source exam identity is carried into every record id while the
    /// exact exam cases remain outside the training stream.
    pub fn generate_training_sibling_for_tools(
        &self,
        run_id: &str,
        examples_per_tool: usize,
        tool_names: &BTreeSet<String>,
        source_exam_fingerprint: &str,
    ) -> Result<Vec<KnowledgeSftRecord>, ToolCurriculumError> {
        if source_exam_fingerprint.trim().is_empty() {
            return Err(ToolCurriculumError::InvalidConfig(
                "source exam fingerprint must not be blank".into(),
            ));
        }
        self.generate_for_tools_with_domain(
            run_id,
            examples_per_tool,
            tool_names,
            ToolPromptDomain::AdaptiveSibling {
                source_exam_fingerprint,
            },
        )
    }

    fn generate_for_tools_with_domain(
        &self,
        run_id: &str,
        examples_per_tool: usize,
        tool_names: &BTreeSet<String>,
        domain: ToolPromptDomain<'_>,
    ) -> Result<Vec<KnowledgeSftRecord>, ToolCurriculumError> {
        if run_id.trim().is_empty() || examples_per_tool == 0 {
            return Err(ToolCurriculumError::InvalidConfig(
                "run_id and examples_per_tool must be non-empty".into(),
            ));
        }
        if tool_names.is_empty() {
            return Err(ToolCurriculumError::InvalidConfig(
                "selected tool set must not be empty".into(),
            ));
        }
        let available: BTreeSet<_> = self
            .contracts
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        let unknown: Vec<_> = tool_names.difference(&available).cloned().collect();
        if !unknown.is_empty() {
            return Err(ToolCurriculumError::InvalidConfig(format!(
                "selected tools are absent from snapshot {}: {unknown:?}",
                self.contracts.manifest_hash
            )));
        }
        let selected_indices: Vec<_> = self
            .contracts
            .tools
            .iter()
            .enumerate()
            .filter_map(|(index, tool)| tool_names.contains(&tool.name).then_some(index))
            .collect();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut records = Vec::with_capacity(selected_indices.len() * examples_per_tool);
        let mut ids = HashSet::new();
        for tool_index in selected_indices {
            let contract = &self.contracts.tools[tool_index];
            for example_index in 0..examples_per_tool {
                let variant_index = tool_index
                    .wrapping_add(example_index)
                    .wrapping_add(self.seed as usize);
                let language = match variant_index % 3 {
                    0 => WorkflowLanguage::English,
                    1 => WorkflowLanguage::German,
                    _ => WorkflowLanguage::French,
                };
                let arguments =
                    canonicalize_runtime_json(self.valid_arguments(contract, &mut rng)?)?;
                let request = match domain {
                    ToolPromptDomain::Standard => {
                        render_request(language, &contract.description, &arguments, variant_index)
                    }
                    ToolPromptDomain::AdaptiveSibling { .. } => render_adaptive_sibling_request(
                        language,
                        &contract.description,
                        &arguments,
                        variant_index,
                        &format!("{:08x}", rng.r#gen::<u32>()),
                    ),
                };
                let mut offered_indices = vec![tool_index];
                let context_tool_names = note_context_tool_names(&contract.name, &self.contracts);
                for context_tool in &context_tool_names {
                    if offered_indices.len() >= self.offered_tool_count {
                        break;
                    }
                    if let Some(index) = self
                        .contracts
                        .tools
                        .iter()
                        .position(|candidate| candidate.name == *context_tool)
                        && !offered_indices.contains(&index)
                    {
                        offered_indices.push(index);
                    }
                }
                let mut distractors: Vec<_> = (0..self.contracts.tools.len())
                    .filter(|index| !offered_indices.contains(index))
                    .collect();
                distractors.shuffle(&mut rng);
                offered_indices.extend(
                    distractors.into_iter().take(
                        self.offered_tool_count
                            .saturating_sub(offered_indices.len()),
                    ),
                );
                offered_indices.shuffle(&mut rng);
                let tools: Vec<_> = offered_indices
                    .iter()
                    .map(|index| self.contracts.tools[*index].openai_definition())
                    .collect();
                let conversation = note_safe_conversation(
                    &contract.name,
                    request,
                    &arguments,
                    &context_tool_names,
                )?;
                let expected = KnowledgeAction::ToolCall {
                    tool: contract.name.clone(),
                    arguments,
                };
                let target = expected.render_runtime().map_err(|error| {
                    ToolCurriculumError::Generate(format!("render target: {error}"))
                })?;
                let rerendered = KnowledgeAction::parse_runtime(&target)
                    .and_then(|action| action.render_runtime())
                    .map_err(|error| {
                        ToolCurriculumError::Generate(format!(
                            "generated target is not runtime-readable: {error}"
                        ))
                    })?;
                if rerendered != target {
                    return Err(ToolCurriculumError::Generate(format!(
                        "generated target for {:?} is not byte-canonical",
                        contract.name
                    )));
                }
                let prompt = render_knowledge_prompt(&tools, &conversation);
                let identity = match domain {
                    ToolPromptDomain::Standard => serde_json::json!({
                        "generator": TOOL_CURRICULUM_GENERATOR_VERSION,
                        "run_id": run_id,
                        "seed": self.seed,
                        "tool": contract.name,
                        "example": example_index,
                        "language": language,
                        "prompt": prompt,
                        "target": target,
                    }),
                    ToolPromptDomain::AdaptiveSibling { .. } => serde_json::json!({
                        "generator": domain.generator_version(),
                        "source_exam_fingerprint": domain.source_exam_fingerprint(),
                        "run_id": run_id,
                        "seed": self.seed,
                        "tool": contract.name,
                        "example": example_index,
                        "language": language,
                        "prompt": prompt,
                        "target": target,
                    }),
                };
                let identity_payload = serde_json::to_vec(&identity)
                    .map_err(|error| ToolCurriculumError::Generate(error.to_string()))?;
                let digest = blake3::hash(&identity_payload).to_hex();
                let episode_id = format!("tool-contract-{run_id}-{digest}");
                let id = format!("{episode_id}-step-00");
                if !ids.insert(id.clone()) {
                    return Err(ToolCurriculumError::Generate(format!(
                        "duplicate generated identity {id}"
                    )));
                }
                records.push(KnowledgeSftRecord {
                    schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
                    id,
                    episode_id,
                    step: 0,
                    stage: match contract.name.as_str() {
                        "study_note_append" => WorkflowStage::DistillNote,
                        "study_note_revise" => WorkflowStage::ReviseNote,
                        _ => WorkflowStage::Answer,
                    },
                    family: format!("tool_contract.{}", contract.name),
                    difficulty: argument_depth(&expected).min(u8::MAX as usize) as u8,
                    language,
                    contract_hash: self.contracts.manifest_hash.clone(),
                    tools,
                    conversation,
                    system_context: None,
                    prompt,
                    target,
                });
            }
        }
        Ok(records)
    }

    /// Generate a quarantined, checkpoint-bound exam across every captured
    /// contract. The cases deliberately omit the rendered target string: the
    /// expected action remains machine-readable scorer data and is never
    /// concatenated into the model prompt.
    pub fn generate_exam(
        &self,
        run_id: &str,
        checkpoint: CheckpointRef,
        tier: ExamTier,
        difficulty: u8,
        samples_per_tool: usize,
    ) -> Result<ToolContractExamSet, ToolCurriculumError> {
        let targets = self
            .contracts
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        self.generate_exam_internal(
            run_id,
            checkpoint,
            tier,
            difficulty,
            samples_per_tool,
            &targets,
            BTreeSet::new(),
        )
    }

    /// Generate a checkpoint-bound frontier/rolling probe for selected tool
    /// families while preserving the full live catalog as distractors. Release
    /// sentinels continue to use [`Self::generate_exam`] and cover every tool.
    pub fn generate_exam_for_tools(
        &self,
        run_id: &str,
        checkpoint: CheckpointRef,
        tier: ExamTier,
        difficulty: u8,
        samples_per_tool: usize,
        target_tools: &BTreeSet<String>,
    ) -> Result<ToolContractExamSet, ToolCurriculumError> {
        if target_tools.is_empty() {
            return Err(ToolCurriculumError::InvalidConfig(
                "targeted tool exam requires at least one tool".into(),
            ));
        }
        if !matches!(tier, ExamTier::RollingProbe | ExamTier::Frontier) {
            return Err(ToolCurriculumError::InvalidConfig(format!(
                "targeted tool exams are only valid for rolling-probe or frontier tiers, got {tier:?}"
            )));
        }
        self.generate_exam_internal(
            run_id,
            checkpoint,
            tier,
            difficulty,
            samples_per_tool,
            target_tools,
            target_tools.clone(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_exam_internal(
        &self,
        run_id: &str,
        checkpoint: CheckpointRef,
        tier: ExamTier,
        difficulty: u8,
        samples_per_tool: usize,
        selected_tools: &BTreeSet<String>,
        serialized_target_tools: BTreeSet<String>,
    ) -> Result<ToolContractExamSet, ToolCurriculumError> {
        if !(1..=3).contains(&difficulty) {
            return Err(ToolCurriculumError::InvalidConfig(format!(
                "tool-selection difficulty must be in 1..=3, got {difficulty}"
            )));
        }
        if checkpoint.artifact_hash.trim().is_empty() {
            return Err(ToolCurriculumError::InvalidConfig(
                "checkpoint artifact hash must not be blank".into(),
            ));
        }
        // Difficulty controls distractor pressure: two offered tools at level
        // one, three at level two, and the daemon's normal four-tool budget at
        // level three. Stateful writes retain a prerequisite floor so a revise
        // case never loses its read or retrieval tool merely to look easier.
        let base_offered_tool_count = self
            .offered_tool_count
            .min(usize::from(difficulty) + 1)
            .min(self.contracts.tools.len());
        let prerequisite_floor = selected_tools
            .iter()
            .map(|tool| 1 + note_context_tool_names(tool, &self.contracts).len())
            .max()
            .unwrap_or(1);
        if prerequisite_floor > self.offered_tool_count {
            return Err(ToolCurriculumError::InvalidConfig(format!(
                "offered tool budget {} cannot retain the {prerequisite_floor}-tool prerequisite chain",
                self.offered_tool_count
            )));
        }
        let offered_tool_count = base_offered_tool_count
            .max(prerequisite_floor)
            .min(self.contracts.tools.len());
        let records = Self {
            contracts: self.contracts.clone(),
            seed: self.seed,
            offered_tool_count,
        }
        .generate_for_tools(run_id, samples_per_tool, selected_tools)?;
        let cases = records
            .into_iter()
            .map(|record| {
                let expected = KnowledgeAction::parse_runtime(&record.target).map_err(|error| {
                    ToolCurriculumError::Generate(format!(
                        "parse generated exam target {}: {error}",
                        record.id
                    ))
                })?;
                let KnowledgeAction::ToolCall { tool, .. } = &expected else {
                    return Err(ToolCurriculumError::Generate(format!(
                        "generated exam case {} is not a tool call",
                        record.id
                    )));
                };
                Ok(ToolContractExamCase {
                    id: record.id,
                    tool: tool.clone(),
                    difficulty,
                    language: record.language,
                    tools: record.tools,
                    conversation: record.conversation,
                    expected,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut exam = ToolContractExamSet {
            schema_version: TOOL_CONTRACT_EXAM_SCHEMA_VERSION,
            generator_version: TOOL_CONTRACT_EXAM_GENERATOR_VERSION.into(),
            run_id: run_id.into(),
            contract_hash: self.contracts.manifest_hash.clone(),
            checkpoint,
            tier,
            difficulty,
            seed: self.seed,
            samples_per_tool: u64::try_from(samples_per_tool).map_err(|_| {
                ToolCurriculumError::InvalidConfig("samples per tool do not fit u64".into())
            })?,
            offered_tool_count,
            target_tools: serialized_target_tools,
            exam_fingerprint: String::new(),
            cases,
        };
        exam.exam_fingerprint = exam.compute_fingerprint()?;
        exam.verify(&self.contracts)?;
        Ok(exam)
    }

    fn valid_arguments(
        &self,
        contract: &ToolContract,
        rng: &mut StdRng,
    ) -> Result<Value, ToolCurriculumError> {
        for _ in 0..32 {
            let mut values = Map::new();
            for parameter in &contract.parameters {
                if parameter.required || rng.gen_bool(0.45) {
                    let schema = parameter
                        .schema
                        .clone()
                        .unwrap_or_else(|| serde_json::json!({"type": parameter.param_type}));
                    values.insert(
                        parameter.name.clone(),
                        sample_schema(&schema, &parameter.name, rng, 0)?,
                    );
                }
            }
            let arguments = Value::Object(values);
            if self
                .contracts
                .validate_call(&contract.name, &arguments)
                .valid
            {
                return Ok(arguments);
            }
        }
        Err(ToolCurriculumError::Generate(format!(
            "could not synthesize schema-valid arguments for {:?}",
            contract.name
        )))
    }
}

/// Generate a current-contract replay stream, extra positives for new/changed
/// model-visible tools, and explicit gap responses for removed or
/// schema-incompatible legacy requests. The complete current catalog is still
/// used for every offered-tool set, preventing targeted adaptation from
/// teaching an artificially small deployment surface.
#[derive(Debug, Clone)]
pub struct ToolContractDeltaCurriculumGenerator {
    previous: ToolContractSnapshot,
    current: ToolContractSnapshot,
    delta: ToolContractDelta,
    seed: u64,
    offered_tool_count: usize,
}

impl ToolContractDeltaCurriculumGenerator {
    pub fn new(
        previous: ToolContractSnapshot,
        current: ToolContractSnapshot,
        seed: u64,
        offered_tool_count: usize,
    ) -> Result<Self, ToolCurriculumError> {
        let delta = ToolContractDelta::between(&previous, &current)?;
        if current.tools.is_empty() {
            return Err(ToolCurriculumError::InvalidConfig(
                "current tool snapshot is empty".into(),
            ));
        }
        if offered_tool_count == 0 || offered_tool_count > current.tools.len() {
            return Err(ToolCurriculumError::InvalidConfig(format!(
                "offered tool count must be in 1..={}, got {offered_tool_count}",
                current.tools.len()
            )));
        }
        Ok(Self {
            previous,
            current,
            delta,
            seed,
            offered_tool_count,
        })
    }

    pub fn delta(&self) -> &ToolContractDelta {
        &self.delta
    }

    pub fn generate(
        &self,
        run_id: &str,
        replay_examples_per_tool: usize,
        target_examples_per_tool: usize,
        legacy_attempts_per_tool: usize,
    ) -> Result<Vec<KnowledgeSftRecord>, ToolCurriculumError> {
        if run_id.trim().is_empty() || replay_examples_per_tool == 0 {
            return Err(ToolCurriculumError::InvalidConfig(
                "run_id and replay examples per tool must be non-empty".into(),
            ));
        }
        if !self.delta.model_adaptation_required() {
            return Err(ToolCurriculumError::InvalidConfig(
                "contract delta has no model-visible addition/change/removal".into(),
            ));
        }

        let replay_generator =
            ToolCurriculumGenerator::new(self.current.clone(), self.seed, self.offered_tool_count)?;
        let mut records = replay_generator.generate(
            &format!("{run_id}-current-replay"),
            replay_examples_per_tool,
        )?;

        let targets = self.delta.model_target_tools();
        if !targets.is_empty() && target_examples_per_tool > 0 {
            let target_generator = ToolCurriculumGenerator::new(
                self.current.clone(),
                self.seed ^ 0xa5a5_51d3_c0de_0001,
                self.offered_tool_count,
            )?;
            records.extend(target_generator.generate_for_tools(
                &format!("{run_id}-changed-target"),
                target_examples_per_tool,
                &targets,
            )?);
        }

        if legacy_attempts_per_tool > 0 {
            records.extend(self.generate_legacy_absence_records(run_id, legacy_attempts_per_tool)?);
        }
        reject_duplicate_curriculum_content(&records)?;
        Ok(records)
    }

    fn generate_legacy_absence_records(
        &self,
        run_id: &str,
        attempts_per_tool: usize,
    ) -> Result<Vec<KnowledgeSftRecord>, ToolCurriculumError> {
        let legacy_names: BTreeSet<_> = self
            .delta
            .removed
            .iter()
            .cloned()
            .chain(
                self.delta
                    .changed
                    .iter()
                    .filter(|change| change.model_surface_changed)
                    .map(|change| change.name.clone()),
            )
            .collect();
        if legacy_names.is_empty() {
            return Ok(Vec::new());
        }
        let previous_generator = ToolCurriculumGenerator::new(
            self.previous.clone(),
            self.seed ^ 0x5a5a_a1d0_1e9a_c002,
            self.offered_tool_count.min(self.previous.tools.len()),
        )?;
        let previous_records = previous_generator.generate_for_tools(
            &format!("{run_id}-legacy-source"),
            attempts_per_tool,
            &legacy_names,
        )?;
        let mut records = Vec::new();
        for source in previous_records {
            let KnowledgeAction::ToolCall { tool, arguments } =
                KnowledgeAction::parse_runtime(&source.target).map_err(|error| {
                    ToolCurriculumError::Generate(format!(
                        "parse legacy source action {}: {error}",
                        source.id
                    ))
                })?
            else {
                return Err(ToolCurriculumError::Generate(format!(
                    "legacy source {} is not a tool call",
                    source.id
                )));
            };
            let still_present = self.current.tool(&tool).is_some();
            if still_present && self.current.validate_call(&tool, &arguments).valid {
                // A description-only change or backward-compatible schema call
                // remains executable. New positive examples teach the updated
                // surface; falsely labelling this request a gap would conflict.
                continue;
            }
            let tools = self.current_offered_tools(
                still_present.then_some(tool.as_str()),
                source.id.as_bytes(),
            );
            let expected = legacy_gap(source.language, &tool, still_present);
            let target = expected.render_runtime().map_err(|error| {
                ToolCurriculumError::Generate(format!("render legacy gap: {error}"))
            })?;
            KnowledgeAction::parse_runtime(&target).map_err(|error| {
                ToolCurriculumError::Generate(format!(
                    "generated legacy gap is not runtime-readable: {error}"
                ))
            })?;
            let prompt = render_knowledge_prompt(&tools, &source.conversation);
            let identity = serde_json::to_vec(&serde_json::json!({
                "generator": TOOL_CURRICULUM_GENERATOR_VERSION,
                "mode": "contract_delta_legacy_absence",
                "run_id": run_id,
                "seed": self.seed,
                "delta_hash": self.delta.delta_hash,
                "source_record": source.id,
                "prompt": prompt,
                "target": target,
            }))
            .map_err(|error| ToolCurriculumError::Generate(error.to_string()))?;
            let digest = blake3::hash(&identity).to_hex();
            let episode_id = format!("tool-contract-delta-{run_id}-{digest}");
            records.push(KnowledgeSftRecord {
                schema_version: KNOWLEDGE_WORKFLOW_SCHEMA_VERSION,
                id: format!("{episode_id}-step-00"),
                episode_id,
                step: 0,
                stage: WorkflowStage::DeclareGap,
                family: format!("tool_contract_absence.{tool}"),
                difficulty: source.difficulty,
                language: source.language,
                contract_hash: self.current.manifest_hash.clone(),
                tools,
                conversation: source.conversation,
                system_context: None,
                prompt,
                target,
            });
        }
        Ok(records)
    }

    fn current_offered_tools(&self, required_tool: Option<&str>, identity: &[u8]) -> Vec<Value> {
        let digest = blake3::hash(identity);
        let mut seed_bytes = [0_u8; 8];
        seed_bytes.copy_from_slice(&digest.as_bytes()[..8]);
        let mut rng = StdRng::seed_from_u64(self.seed ^ u64::from_le_bytes(seed_bytes));
        let required_index = required_tool.and_then(|name| {
            self.current
                .tools
                .iter()
                .position(|contract| contract.name == name)
        });
        let mut indices: Vec<_> = (0..self.current.tools.len())
            .filter(|index| Some(*index) != required_index)
            .collect();
        indices.shuffle(&mut rng);
        let mut offered = required_index.into_iter().collect::<Vec<_>>();
        offered.extend(
            indices
                .into_iter()
                .take(self.offered_tool_count - offered.len()),
        );
        offered.shuffle(&mut rng);
        offered
            .into_iter()
            .map(|index| self.current.tools[index].openai_definition())
            .collect()
    }
}

fn legacy_gap(language: WorkflowLanguage, tool: &str, schema_changed: bool) -> KnowledgeAction {
    let missing = if schema_changed {
        format!("current required inputs for `{tool}`")
    } else {
        format!("available capability `{tool}`")
    };
    let reason = match (language, schema_changed) {
        (WorkflowLanguage::English, true) => {
            "The request matches an older contract but lacks inputs required by the current schema."
        }
        (WorkflowLanguage::English, false) => {
            "The requested capability is not present in the current tool catalog."
        }
        (WorkflowLanguage::German, true) => {
            "Die Anfrage entspricht einem älteren Vertrag, aber Pflichtangaben des aktuellen Schemas fehlen."
        }
        (WorkflowLanguage::German, false) => {
            "Die angeforderte Fähigkeit ist im aktuellen Werkzeugkatalog nicht verfügbar."
        }
        (WorkflowLanguage::French, true) => {
            "La demande correspond à un ancien contrat, mais les entrées obligatoires du schéma actuel manquent."
        }
        (WorkflowLanguage::French, false) => {
            "La capacité demandée n'est pas disponible dans le catalogue d'outils actuel."
        }
    };
    KnowledgeAction::Gap {
        missing: vec![missing],
        reason: reason.into(),
    }
}

fn reject_duplicate_curriculum_content(
    records: &[KnowledgeSftRecord],
) -> Result<(), ToolCurriculumError> {
    let mut ids = HashSet::new();
    let mut content = HashSet::new();
    for record in records {
        if !ids.insert(record.id.as_str()) {
            return Err(ToolCurriculumError::Generate(format!(
                "duplicate generated identity {}",
                record.id
            )));
        }
        let identity = blake3::hash(
            serde_json::to_string(&(record.prompt.as_str(), record.target.as_str()))
                .map_err(|error| ToolCurriculumError::Generate(error.to_string()))?
                .as_bytes(),
        );
        if !content.insert(*identity.as_bytes()) {
            return Err(ToolCurriculumError::Generate(format!(
                "duplicate generated prompt+target content at {}",
                record.id
            )));
        }
    }
    Ok(())
}

/// Quarantine-preserving plan for targeted SFT after a tool frontier exam.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContractTrainingSiblingPlan {
    pub schema_version: u32,
    pub generator_version: String,
    pub run_id: String,
    pub contract_hash: String,
    pub family: String,
    pub tool: String,
    pub difficulty: u8,
    pub seed: u64,
    pub sample_count: u64,
    pub offered_tool_count: usize,
    pub source_exam_fingerprint: String,
    pub required_policy: SiblingSeparationPolicy,
}

impl ToolContractTrainingSiblingPlan {
    pub fn verify_against_exam(
        &self,
        exam: &ToolContractExamSet,
        contracts: &ToolContractSnapshot,
    ) -> Result<(), ToolCurriculumError> {
        let expected = derive_tool_contract_training_sibling(
            exam,
            contracts,
            &self.family,
            self.sample_count,
        )?;
        if self != &expected {
            return Err(ToolCurriculumError::InvalidConfig(
                "tool training-sibling plan was edited after derivation".into(),
            ));
        }
        Ok(())
    }

    pub fn generate(
        &self,
        contracts: &ToolContractSnapshot,
    ) -> Result<Vec<KnowledgeSftRecord>, ToolCurriculumError> {
        contracts.verify_integrity()?;
        if self.schema_version != TOOL_CONTRACT_SIBLING_SCHEMA_VERSION
            || self.generator_version != TOOL_CONTRACT_SIBLING_GENERATOR_VERSION
            || self.contract_hash != contracts.manifest_hash
            || self.family != format!("tool_contract.{}", self.tool)
            || contracts.tool(&self.tool).is_none()
            || self.sample_count == 0
            || self.offered_tool_count == 0
            || self.offered_tool_count > contracts.tools.len()
            || self.source_exam_fingerprint.trim().is_empty()
        {
            return Err(ToolCurriculumError::InvalidConfig(
                "tool training-sibling plan is inconsistent with the current contract".into(),
            ));
        }
        let samples = usize::try_from(self.sample_count).map_err(|_| {
            ToolCurriculumError::InvalidConfig("sibling sample count does not fit usize".into())
        })?;
        let targets = BTreeSet::from([self.tool.clone()]);
        let records =
            ToolCurriculumGenerator::new(contracts.clone(), self.seed, self.offered_tool_count)?
                .generate_training_sibling_for_tools(
                    &format!("{}-{}", self.run_id, self.tool),
                    samples,
                    &targets,
                    &self.source_exam_fingerprint,
                )?;
        reject_duplicate_curriculum_content(&records)?;
        Ok(records)
    }

    pub fn validate_separation(
        &self,
        evidence: &SiblingSeparationEvidence,
    ) -> Result<(), crate::AdaptiveExamError> {
        validate_sibling_separation(&self.required_policy, evidence)
    }
}

pub fn derive_tool_contract_training_sibling(
    exam: &ToolContractExamSet,
    contracts: &ToolContractSnapshot,
    family: &str,
    sample_count: u64,
) -> Result<ToolContractTrainingSiblingPlan, ToolCurriculumError> {
    exam.verify(contracts)?;
    let samples_per_tool = usize::try_from(exam.samples_per_tool).map_err(|_| {
        ToolCurriculumError::InvalidExam("exam samples per tool do not fit usize".into())
    })?;
    let generator =
        ToolCurriculumGenerator::new(contracts.clone(), exam.seed, exam.offered_tool_count)?;
    let regenerated = if exam.target_tools.is_empty() {
        generator.generate_exam(
            &exam.run_id,
            exam.checkpoint.clone(),
            exam.tier,
            exam.difficulty,
            samples_per_tool,
        )?
    } else {
        generator.generate_exam_for_tools(
            &exam.run_id,
            exam.checkpoint.clone(),
            exam.tier,
            exam.difficulty,
            samples_per_tool,
            &exam.target_tools,
        )?
    };
    if &regenerated != exam {
        return Err(ToolCurriculumError::InvalidExam(
            "tool frontier exam does not replay from its declared generator inputs".into(),
        ));
    }
    if exam.tier != ExamTier::Frontier {
        return Err(ToolCurriculumError::InvalidConfig(format!(
            "tool training siblings require a frontier exam, got {:?}",
            exam.tier
        )));
    }
    if sample_count == 0 {
        return Err(ToolCurriculumError::InvalidConfig(
            "tool training sibling sample count must be positive".into(),
        ));
    }
    let tool = family.strip_prefix("tool_contract.").ok_or_else(|| {
        ToolCurriculumError::InvalidConfig(format!(
            "tool sibling family {family:?} lacks the tool_contract. prefix"
        ))
    })?;
    if contracts.tool(tool).is_none() || !exam.cases.iter().any(|case| case.tool == tool) {
        return Err(ToolCurriculumError::InvalidConfig(format!(
            "tool sibling family {family:?} is absent from the verified exam"
        )));
    }
    let seed_payload = serde_json::to_vec(&serde_json::json!({
        "domain": "babyllm-tool-training-sibling-v1",
        "source_exam_fingerprint": exam.exam_fingerprint,
        "contract_hash": contracts.manifest_hash,
        "family": family,
        "sample_count": sample_count,
    }))
    .map_err(|error| ToolCurriculumError::Generate(error.to_string()))?;
    let digest = blake3::hash(&seed_payload);
    let mut seed_bytes = [0_u8; 8];
    seed_bytes.copy_from_slice(&digest.as_bytes()[..8]);
    Ok(ToolContractTrainingSiblingPlan {
        schema_version: TOOL_CONTRACT_SIBLING_SCHEMA_VERSION,
        generator_version: TOOL_CONTRACT_SIBLING_GENERATOR_VERSION.into(),
        run_id: exam.run_id.clone(),
        contract_hash: contracts.manifest_hash.clone(),
        family: family.into(),
        tool: tool.into(),
        difficulty: exam.difficulty,
        seed: u64::from_le_bytes(seed_bytes),
        sample_count,
        offered_tool_count: exam.offered_tool_count,
        source_exam_fingerprint: exam.exam_fingerprint.clone(),
        required_policy: SiblingSeparationPolicy {
            require_image_distance: false,
            ..SiblingSeparationPolicy::default()
        },
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContractExamCase {
    pub id: String,
    pub tool: String,
    pub difficulty: u8,
    pub language: WorkflowLanguage,
    /// Exact OpenAI-compatible tools shown to the candidate.
    pub tools: Vec<Value>,
    pub conversation: Vec<Value>,
    /// Quarantined scorer truth; never rendered into the candidate prompt.
    pub expected: KnowledgeAction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContractExamSet {
    pub schema_version: u32,
    pub generator_version: String,
    pub run_id: String,
    pub contract_hash: String,
    pub checkpoint: CheckpointRef,
    pub tier: ExamTier,
    pub difficulty: u8,
    pub seed: u64,
    pub samples_per_tool: u64,
    pub offered_tool_count: usize,
    /// Empty means the historical release scope: every captured contract.
    /// Non-empty is an explicit targeted rolling/frontier scope.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub target_tools: BTreeSet<String>,
    pub exam_fingerprint: String,
    pub cases: Vec<ToolContractExamCase>,
}

impl ToolContractExamSet {
    pub fn compute_fingerprint(&self) -> Result<String, ToolCurriculumError> {
        let mut identity = if self.tier == ExamTier::Sentinel {
            // One frozen sentinel is evaluated against many candidates. Its
            // case identity cannot move with a checkpoint; execution evidence
            // binds the resulting report to the actual candidate instead.
            serde_json::json!({
                "schema_version": self.schema_version,
                "generator_version": self.generator_version,
                "run_id": self.run_id,
                "contract_hash": self.contract_hash,
                "tier": self.tier,
                "difficulty": self.difficulty,
                "seed": self.seed,
                "samples_per_tool": self.samples_per_tool,
                "offered_tool_count": self.offered_tool_count,
                "cases": self.cases,
            })
        } else {
            // Keep the historical v1 moving-exam identity byte-for-byte.
            serde_json::json!({
                "schema_version": self.schema_version,
                "generator_version": self.generator_version,
                "run_id": self.run_id,
                "contract_hash": self.contract_hash,
                "checkpoint": self.checkpoint,
                "tier": self.tier,
                "difficulty": self.difficulty,
                "seed": self.seed,
                "samples_per_tool": self.samples_per_tool,
                "offered_tool_count": self.offered_tool_count,
                "cases": self.cases,
            })
        };
        // Preserve byte identity for already materialized all-tool v1 exams.
        if !self.target_tools.is_empty() {
            identity
                .as_object_mut()
                .expect("exam identity is an object")
                .insert(
                    "target_tools".into(),
                    serde_json::to_value(&self.target_tools).map_err(|error| {
                        ToolCurriculumError::Generate(format!(
                            "serialize target tool scope: {error}"
                        ))
                    })?,
                );
        }
        let payload = serde_json::to_vec(&identity)
            .map_err(|error| ToolCurriculumError::Generate(format!("serialize exam: {error}")))?;
        Ok(format!("blake3:{}", blake3::hash(&payload).to_hex()))
    }

    /// Recompute all exam invariants before model execution. This prevents a
    /// hand-edited, partial, or differently-scoped suite from becoming release
    /// evidence merely because it still parses as JSON.
    pub fn verify(&self, contracts: &ToolContractSnapshot) -> Result<(), ToolCurriculumError> {
        contracts.verify_integrity()?;
        if self.schema_version != TOOL_CONTRACT_EXAM_SCHEMA_VERSION
            || self.generator_version != TOOL_CONTRACT_EXAM_GENERATOR_VERSION
        {
            return Err(ToolCurriculumError::InvalidExam(
                "unsupported tool exam schema or generator version".into(),
            ));
        }
        let checkpoint_is_valid = if self.tier == ExamTier::Sentinel {
            self.checkpoint.step == 0
                && self.checkpoint.artifact_hash == "unbound:immutable-sentinel"
        } else {
            self.checkpoint.artifact_hash.starts_with("blake3:")
        };
        if self.run_id.trim().is_empty()
            || self.contract_hash != contracts.manifest_hash
            || !(1..=3).contains(&self.difficulty)
            || self.samples_per_tool == 0
            || self.offered_tool_count == 0
            || self.offered_tool_count > contracts.tools.len()
            || !checkpoint_is_valid
        {
            return Err(ToolCurriculumError::InvalidExam(
                "tool exam metadata is inconsistent with the captured contracts".into(),
            ));
        }
        let contract_names: BTreeSet<_> = contracts
            .tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect();
        let target_names: BTreeSet<_> = if self.target_tools.is_empty() {
            contract_names.clone()
        } else {
            self.target_tools.iter().map(String::as_str).collect()
        };
        if target_names.is_empty() || !target_names.is_subset(&contract_names) {
            return Err(ToolCurriculumError::InvalidExam(
                "tool exam target scope is empty or contains an unknown contract".into(),
            ));
        }
        let expected_cases = self
            .samples_per_tool
            .checked_mul(target_names.len() as u64)
            .ok_or_else(|| ToolCurriculumError::InvalidExam("exam size overflow".into()))?;
        if self.cases.len() as u64 != expected_cases {
            return Err(ToolCurriculumError::InvalidExam(format!(
                "exam contains {} cases; expected {expected_cases}",
                self.cases.len()
            )));
        }

        let mut ids = HashSet::new();
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for case in &self.cases {
            if !ids.insert(case.id.as_str())
                || case.difficulty != self.difficulty
                || !target_names.contains(case.tool.as_str())
                || case.tools.len() != self.offered_tool_count
            {
                return Err(ToolCurriculumError::InvalidExam(format!(
                    "case {:?} has invalid identity, tool, difficulty, or offered-tool count",
                    case.id
                )));
            }
            let KnowledgeAction::ToolCall { tool, arguments } = &case.expected else {
                return Err(ToolCurriculumError::InvalidExam(format!(
                    "case {:?} does not expect a tool call",
                    case.id
                )));
            };
            if tool != &case.tool || !contracts.validate_call(tool, arguments).valid {
                return Err(ToolCurriculumError::InvalidExam(format!(
                    "case {:?} has a mismatched or schema-invalid expected call",
                    case.id
                )));
            }
            let offered_names = offered_tool_names(&case.tools)?;
            if offered_names.len() != self.offered_tool_count || !offered_names.contains(tool) {
                return Err(ToolCurriculumError::InvalidExam(format!(
                    "case {:?} does not offer its expected tool exactly once",
                    case.id
                )));
            }
            *counts.entry(case.tool.as_str()).or_default() += 1;
        }
        for tool in target_names {
            if counts.get(tool).copied() != Some(self.samples_per_tool) {
                return Err(ToolCurriculumError::InvalidExam(format!(
                    "tool {tool:?} does not have exactly {} cases",
                    self.samples_per_tool
                )));
            }
        }
        let computed = self.compute_fingerprint()?;
        if computed != self.exam_fingerprint {
            return Err(ToolCurriculumError::InvalidExam(format!(
                "tool exam fingerprint mismatch: declared {}, computed {computed}",
                self.exam_fingerprint
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolContractCaseScore {
    pub id: String,
    pub tool: String,
    pub passed: bool,
    /// A schema-valid call to the correct tool with semantically wrong values.
    pub precursor: bool,
    /// Missing/malformed output, a wrong tool, or a schema-invalid call.
    pub catastrophic: bool,
    pub tool_correct: bool,
    pub tool_offered: bool,
    pub schema_valid: bool,
    pub arguments_correct: bool,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContractFamilyReport {
    pub family: String,
    pub tool: String,
    pub cases: Vec<ToolContractCaseScore>,
    pub pass_count: u64,
    pub precursor_count: u64,
    pub catastrophic_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContractExamReport {
    pub contract_hash: String,
    pub exam_fingerprint: String,
    pub checkpoint: CheckpointRef,
    pub tier: ExamTier,
    pub difficulty: u8,
    pub offered_tool_count: usize,
    pub families: Vec<ToolContractFamilyReport>,
}

impl ToolContractExamReport {
    pub fn observations(&self) -> Vec<ExamObservation> {
        self.families
            .iter()
            .map(|family| ExamObservation {
                checkpoint: self.checkpoint.clone(),
                tier: self.tier,
                family: family.family.clone(),
                difficulty: self.difficulty,
                sample_count: family.cases.len() as u64,
                pass_count: family.pass_count,
                precursor_count: family.precursor_count,
                catastrophic_count: family.catastrophic_count,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolContractEvaluationMode {
    CheckpointExecuted,
    #[default]
    ExternalPredictions,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolContractExamEvidence {
    pub schema_version: u32,
    #[serde(default)]
    pub evaluation_mode: ToolContractEvaluationMode,
    pub report: ToolContractExamReport,
    pub observations: Vec<ExamObservation>,
}

pub fn score_tool_contract_exam(
    exam: &ToolContractExamSet,
    predictions: &BTreeMap<String, String>,
    contracts: &ToolContractSnapshot,
) -> ToolContractExamReport {
    let mut grouped: BTreeMap<String, Vec<ToolContractCaseScore>> = BTreeMap::new();
    for case in &exam.cases {
        grouped
            .entry(case.tool.clone())
            .or_default()
            .push(score_tool_contract_case(
                case,
                predictions.get(&case.id).map(String::as_str),
                contracts,
            ));
    }
    let families = grouped
        .into_iter()
        .map(|(tool, cases)| ToolContractFamilyReport {
            family: format!("tool_contract.{tool}"),
            tool,
            pass_count: cases.iter().filter(|case| case.passed).count() as u64,
            precursor_count: cases.iter().filter(|case| case.precursor).count() as u64,
            catastrophic_count: cases.iter().filter(|case| case.catastrophic).count() as u64,
            cases,
        })
        .collect();
    ToolContractExamReport {
        contract_hash: contracts.manifest_hash.clone(),
        exam_fingerprint: exam.exam_fingerprint.clone(),
        checkpoint: exam.checkpoint.clone(),
        tier: exam.tier,
        difficulty: exam.difficulty,
        offered_tool_count: exam.offered_tool_count,
        families,
    }
}

fn score_tool_contract_case(
    case: &ToolContractExamCase,
    raw: Option<&str>,
    contracts: &ToolContractSnapshot,
) -> ToolContractCaseScore {
    let Some(raw) = raw else {
        return failed_tool_case(case, "missing prediction");
    };
    let Ok(predicted) = KnowledgeAction::parse_runtime(raw) else {
        return failed_tool_case(case, "prediction is not one canonical runtime action");
    };
    let KnowledgeAction::ToolCall { tool, arguments } = &predicted else {
        return failed_tool_case(case, "prediction is not a tool call");
    };
    let expected_arguments = match &case.expected {
        KnowledgeAction::ToolCall { arguments, .. } => arguments,
        _ => unreachable!("verified exam cases always expect calls"),
    };
    let offered = offered_tool_names(&case.tools).unwrap_or_default();
    let validation = contracts.validate_call(tool, arguments);
    let tool_correct = tool == &case.tool;
    let tool_offered = offered.contains(tool);
    let arguments_correct = arguments == expected_arguments;
    let passed = tool_correct && tool_offered && validation.valid && arguments_correct;
    let precursor = !passed && tool_correct && tool_offered && validation.valid;
    let catastrophic = !passed && !precursor;
    let mut issues = Vec::new();
    if !tool_correct {
        issues.push(format!("expected tool {:?}, got {tool:?}", case.tool));
    }
    if !tool_offered {
        issues.push(format!("predicted tool {tool:?} was not offered"));
    }
    if !validation.valid {
        issues.push(format!("schema violations: {:?}", validation.issues));
    }
    if !arguments_correct {
        issues.push("arguments differ from the exact requested values".into());
    }
    ToolContractCaseScore {
        id: case.id.clone(),
        tool: case.tool.clone(),
        passed,
        precursor,
        catastrophic,
        tool_correct,
        tool_offered,
        schema_valid: validation.valid,
        arguments_correct,
        issues,
    }
}

fn failed_tool_case(case: &ToolContractExamCase, issue: &str) -> ToolContractCaseScore {
    ToolContractCaseScore {
        id: case.id.clone(),
        tool: case.tool.clone(),
        passed: false,
        precursor: false,
        catastrophic: true,
        tool_correct: false,
        tool_offered: false,
        schema_valid: false,
        arguments_correct: false,
        issues: vec![issue.into()],
    }
}

fn offered_tool_names(tools: &[Value]) -> Result<BTreeSet<String>, ToolCurriculumError> {
    let mut names = BTreeSet::new();
    for tool in tools {
        let Some(name) = tool
            .get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
        else {
            return Err(ToolCurriculumError::InvalidExam(
                "offered tool has no OpenAI function name".into(),
            ));
        };
        if !names.insert(name.into()) {
            return Err(ToolCurriculumError::InvalidExam(format!(
                "offered tool {name:?} appears more than once"
            )));
        }
    }
    Ok(names)
}

/// Extra capabilities that must have run before a source-backed note write can
/// pass the host evidence and optimistic-concurrency gates. They are included
/// in the normal four-tool surface when space permits and always represented
/// in the final-write conversation, matching the preceding rounds that led to
/// this training target.
fn note_context_tool_names(target_tool: &str, contracts: &ToolContractSnapshot) -> Vec<String> {
    if !matches!(target_tool, "study_note_append" | "study_note_revise") {
        return Vec::new();
    }

    let mut names = Vec::new();
    if target_tool == "study_note_revise" && contracts.tool("study_note_read").is_some() {
        names.push("study_note_read".into());
    }
    let evidence = contracts
        .tools
        .iter()
        .find(|tool| tool.name.ends_with("__seshat_search"))
        .or_else(|| contracts.tool("archival_search"))
        .or_else(|| contracts.tool("web_search"));
    if let Some(evidence) = evidence
        && !names.contains(&evidence.name)
    {
        names.push(evidence.name.clone());
    }
    names
}

fn note_safe_conversation(
    target_tool: &str,
    request: String,
    arguments: &Value,
    context_tools: &[String],
) -> Result<Vec<Value>, ToolCurriculumError> {
    let mut conversation = vec![serde_json::json!({
        "role": "user",
        "content": request
    })];
    if !matches!(target_tool, "study_note_append" | "study_note_revise") {
        return Ok(conversation);
    }

    let has_citations = arguments
        .get("citations")
        .and_then(Value::as_array)
        .is_some_and(|citations| !citations.is_empty());
    let evidence_tool = context_tools
        .iter()
        .find(|name| name.as_str() != "study_note_read");
    if has_citations && evidence_tool.is_none() {
        return Err(ToolCurriculumError::Generate(format!(
            "{target_tool:?} requires an evidence-search tool in the captured contract"
        )));
    }
    if target_tool == "study_note_revise"
        && arguments.get("note_id").is_some()
        && arguments.get("expected_revision").is_some()
        && !context_tools.iter().any(|name| name == "study_note_read")
    {
        return Err(ToolCurriculumError::Generate(
            "study_note_revise requires study_note_read in the captured contract".into(),
        ));
    }

    if target_tool == "study_note_revise"
        && context_tools.iter().any(|name| name == "study_note_read")
        && let (Some(note_id), Some(expected_revision)) = (
            arguments.get("note_id").and_then(Value::as_str),
            arguments.get("expected_revision").and_then(Value::as_str),
        )
    {
        conversation.push(serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "study_note_read",
                    "arguments": {"note_id": note_id}
                }
            }]
        }));
        conversation.push(serde_json::json!({
            "role": "tool",
            "content": serde_json::json!({
                "operation": "read",
                "note_id": note_id,
                "revision": expected_revision,
                "stored_at": 1893542400_u64,
                "document": {
                    "id": format!("study-{note_id}"),
                    "title": "Existing source-backed note",
                    "source": "Inline",
                    "format": "PlainText",
                    "tags": [],
                    "chunk_count": 1,
                    "triple_count": 0,
                    "ingested_at": 1893456000_u64
                },
                "frontmatter": {
                    "status": {"text": "active"},
                    "type": {"text": "StudyNote"}
                },
                "body": "Existing source-backed note head.",
                "wikilinks": [],
                "tasks": []
            }).to_string()
        }));
    }

    let evidence_rows: Vec<Value> = arguments
        .get("citations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|citation| {
            Some(serde_json::json!({
                "document_id": citation.get("source_id")?.as_str()?,
                "locator": citation.get("locator")?.as_str()?,
                "text": citation.get("quote")?.as_str()?
            }))
        })
        .collect();
    if let Some(evidence_tool) = evidence_tool
        && !evidence_rows.is_empty()
    {
        let query = evidence_rows
            .iter()
            .filter_map(|row| {
                Some(format!(
                    "{} {}",
                    row.get("document_id")?.as_str()?,
                    row.get("locator")?.as_str()?
                ))
            })
            .collect::<Vec<_>>()
            .join("; ");
        conversation.push(serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": evidence_tool,
                    "arguments": {"query": query}
                }
            }]
        }));
        conversation.push(serde_json::json!({
            "role": "tool",
            "content": Value::Array(evidence_rows).to_string()
        }));
    }
    Ok(conversation)
}

fn sample_schema(
    schema: &Value,
    name: &str,
    rng: &mut StdRng,
    depth: usize,
) -> Result<Value, ToolCurriculumError> {
    if depth > 8 {
        return Err(ToolCurriculumError::Generate(format!(
            "schema nesting exceeds eight levels at {name:?}"
        )));
    }
    if let Some(value) = schema.get("const") {
        return Ok(value.clone());
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        return values
            .get(rng.gen_range(0..values.len().max(1)))
            .cloned()
            .ok_or_else(|| ToolCurriculumError::Generate(format!("empty enum at {name:?}")));
    }
    for keyword in ["oneOf", "anyOf"] {
        if let Some(variants) = schema.get(keyword).and_then(Value::as_array) {
            if variants.is_empty() {
                return Err(ToolCurriculumError::Generate(format!(
                    "empty {keyword} at {name:?}"
                )));
            }
            let start = rng.gen_range(0..variants.len());
            for offset in 0..variants.len() {
                if let Ok(value) = sample_schema(
                    &variants[(start + offset) % variants.len()],
                    name,
                    rng,
                    depth + 1,
                ) {
                    return Ok(value);
                }
            }
        }
    }

    let declared_types: Vec<&str> = match schema.get("type") {
        Some(Value::String(value)) => vec![value],
        Some(Value::Array(values)) => values.iter().filter_map(Value::as_str).collect(),
        _ if schema.get("properties").is_some() => vec!["object"],
        _ if schema.get("items").is_some() => vec!["array"],
        _ => vec!["string"],
    };
    let non_null: Vec<_> = declared_types
        .iter()
        .copied()
        .filter(|value| *value != "null")
        .collect();
    let selected = non_null
        .get(rng.gen_range(0..non_null.len().max(1)))
        .copied()
        .or_else(|| declared_types.first().copied())
        .unwrap_or("string");
    match selected {
        "null" => Ok(Value::Null),
        "boolean" => Ok(Value::Bool(rng.gen_bool(0.5))),
        "integer" => {
            let minimum = schema.get("minimum").and_then(Value::as_i64).unwrap_or(0);
            let maximum = schema
                .get("maximum")
                .and_then(Value::as_i64)
                .unwrap_or_else(|| minimum.saturating_add(100));
            let upper = maximum.max(minimum);
            Ok(Value::Number(Number::from(if upper == minimum {
                minimum
            } else {
                rng.gen_range(minimum..=upper)
            })))
        }
        "number" => {
            let minimum = schema.get("minimum").and_then(Value::as_f64).unwrap_or(0.0);
            let maximum = schema
                .get("maximum")
                .and_then(Value::as_f64)
                .unwrap_or(minimum + 1.0)
                .max(minimum);
            let number = if let Some(default) = schema.get("default").and_then(Value::as_f64)
                && rng.gen_bool(0.5)
            {
                default.clamp(minimum, maximum)
            } else if maximum == minimum {
                minimum
            } else {
                let unit = rng.gen_range(0..=1000) as f64 / 1000.0;
                minimum + (maximum - minimum) * unit
            };
            Number::from_f64(number).map(Value::Number).ok_or_else(|| {
                ToolCurriculumError::Generate(format!("non-finite number schema at {name:?}"))
            })
        }
        "array" => {
            let minimum = schema.get("minItems").and_then(Value::as_u64).unwrap_or(1) as usize;
            let maximum = schema
                .get("maxItems")
                .and_then(Value::as_u64)
                .unwrap_or(minimum.saturating_add(2) as u64) as usize;
            let count = if maximum <= minimum {
                minimum
            } else {
                rng.gen_range(minimum..=maximum.min(minimum.saturating_add(2)))
            };
            let item_schema = schema
                .get("items")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"string"}));
            (0..count)
                .map(|index| {
                    sample_schema(&item_schema, &format!("{name}[{index}]"), rng, depth + 1)
                })
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
        }
        "object" => {
            let properties = schema
                .get("properties")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let required: BTreeSet<_> = schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let mut object = Map::new();
            for (property, nested) in properties {
                if required.contains(property.as_str()) || rng.gen_bool(0.35) {
                    object.insert(
                        property.clone(),
                        sample_schema(&nested, &property, rng, depth + 1)?,
                    );
                }
            }
            Ok(Value::Object(object))
        }
        _ => {
            let minimum = schema.get("minLength").and_then(Value::as_u64).unwrap_or(1) as usize;
            let maximum = schema
                .get("maxLength")
                .and_then(Value::as_u64)
                .map(|value| value as usize)
                .unwrap_or(usize::MAX);
            if maximum < minimum {
                return Err(ToolCurriculumError::Generate(format!(
                    "maxLength is below minLength at {name:?}"
                )));
            }
            let pattern = schema.get("pattern").and_then(Value::as_str);
            if pattern == Some("^note-[0-9a-f]{32}$") {
                const NOTE_ID_CHARS: usize = 37;
                if minimum > NOTE_ID_CHARS || maximum < NOTE_ID_CHARS {
                    return Err(ToolCurriculumError::Generate(format!(
                        "note-id pattern conflicts with length bounds at {name:?}"
                    )));
                }
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let suffix: String = (0..32)
                    .map(|_| HEX[rng.gen_range(0..HEX.len())] as char)
                    .collect();
                return Ok(Value::String(format!("note-{suffix}")));
            }
            if pattern == Some("^[0-9a-f]{64}$") {
                if minimum > 64 || maximum < 64 {
                    return Err(ToolCurriculumError::Generate(format!(
                        "revision-token pattern conflicts with length bounds at {name:?}"
                    )));
                }
                const HEX: &[u8; 16] = b"0123456789abcdef";
                let value: String = (0..64)
                    .map(|_| HEX[rng.gen_range(0..HEX.len())] as char)
                    .collect();
                return Ok(Value::String(value));
            }
            let compiled_pattern = pattern
                .map(regex::Regex::new)
                .transpose()
                .map_err(|error| {
                    ToolCurriculumError::Generate(format!(
                        "invalid string pattern at {name:?}: {error}"
                    ))
                })?;
            for _ in 0..32 {
                let values = [
                    format!("{name}-alpha"),
                    format!("{name}-zurich"),
                    format!("{name}-revision-{}", rng.gen_range(1..=999)),
                    "the exact requested value".into(),
                ];
                let mut value = values[rng.gen_range(0..values.len())].clone();
                while value.chars().count() < minimum {
                    value.push('x');
                }
                if value.chars().count() > maximum {
                    value = value.chars().take(maximum).collect();
                }
                if compiled_pattern
                    .as_ref()
                    .is_none_or(|pattern| pattern.is_match(&value))
                {
                    return Ok(Value::String(value));
                }
            }
            Err(ToolCurriculumError::Generate(format!(
                "could not synthesize a string matching pattern at {name:?}"
            )))
        }
    }
}

fn canonicalize_runtime_json(value: Value) -> Result<Value, ToolCurriculumError> {
    let encoded = serde_json::to_vec(&value)
        .map_err(|error| ToolCurriculumError::Generate(error.to_string()))?;
    serde_json::from_slice(&encoded)
        .map_err(|error| ToolCurriculumError::Generate(error.to_string()))
}

fn render_request(
    language: WorkflowLanguage,
    description: &str,
    arguments: &Value,
    variant: usize,
) -> String {
    let operation = description.lines().next().unwrap_or(description).trim();
    let arguments = arguments.as_object().expect("generated object");
    let values = if arguments.is_empty() {
        match language {
            WorkflowLanguage::English => "No arguments are required.".into(),
            WorkflowLanguage::German => "Es sind keine Argumente erforderlich.".into(),
            WorkflowLanguage::French => "Aucun argument n'est requis.".into(),
        }
    } else {
        arguments
            .iter()
            .map(|(name, value)| format!("- {name} = {value}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    match (language, variant % 3) {
        (WorkflowLanguage::English, 0) => {
            format!("Perform this operation: {operation}\nUse these exact inputs:\n{values}")
        }
        (WorkflowLanguage::English, 1) => {
            format!("Choose the matching available capability for: {operation}\nInputs:\n{values}")
        }
        (WorkflowLanguage::English, _) => {
            format!("Complete the requested action described as follows: {operation}\n{values}")
        }
        (WorkflowLanguage::German, 0) => format!(
            "Führe diesen Vorgang mit der passenden verfügbaren Fähigkeit aus: {operation}\nVerwende exakt diese Eingaben:\n{values}"
        ),
        (WorkflowLanguage::German, 1) => format!(
            "Wähle die passende Fähigkeit für diesen Auftrag: {operation}\nEingaben:\n{values}"
        ),
        (WorkflowLanguage::German, _) => {
            format!("Erledige die folgende Aktion: {operation}\n{values}")
        }
        (WorkflowLanguage::French, 0) => format!(
            "Exécute cette opération avec la capacité disponible appropriée : {operation}\nUtilise exactement ces entrées :\n{values}"
        ),
        (WorkflowLanguage::French, 1) => format!(
            "Choisis la capacité qui correspond à cette demande : {operation}\nEntrées :\n{values}"
        ),
        (WorkflowLanguage::French, _) => {
            format!("Effectue l'action suivante : {operation}\n{values}")
        }
    }
}

fn render_adaptive_sibling_request(
    language: WorkflowLanguage,
    description: &str,
    arguments: &Value,
    variant: usize,
    nonce: &str,
) -> String {
    let operation = description.lines().next().unwrap_or(description).trim();
    let arguments = arguments.as_object().expect("generated object");
    let ledger = if arguments.is_empty() {
        match language {
            WorkflowLanguage::English => "The parameter object is empty.".into(),
            WorkflowLanguage::German => "Das Parameterobjekt ist leer.".into(),
            WorkflowLanguage::French => "L'objet de paramètres est vide.".into(),
        }
    } else {
        arguments
            .iter()
            .map(|(name, value)| format!("{name} -> {value}"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let variant = variant % 3 + 1;
    match language {
        WorkflowLanguage::English => format!(
            "Akh-Medu adaptive practice ledger {nonce}/{variant}. Treat the offered capability catalogue as authoritative. Match the intent below to exactly one available function, copy every ledger value literally, and invent no parameter keys.\nIntent label: {operation}\nStructured input ledger:\n{ledger}"
        ),
        WorkflowLanguage::German => format!(
            "Akh-Medu-Übungsprotokoll {nonce}/{variant}. Der angebotene Fähigkeitskatalog ist verbindlich. Ordne die folgende Absicht genau einer verfügbaren Funktion zu, übernimm alle Werte wörtlich und erfinde keine Parameterschlüssel.\nAbsichtsbezeichnung: {operation}\nStrukturiertes Eingabeprotokoll:\n{ledger}"
        ),
        WorkflowLanguage::French => format!(
            "Registre d'entraînement adaptatif Akh-Medu {nonce}/{variant}. Le catalogue de capacités proposé fait foi. Associe l'intention suivante à une seule fonction disponible, recopie chaque valeur littéralement et n'invente aucune clé de paramètre.\nLibellé d'intention : {operation}\nRegistre d'entrée structuré :\n{ledger}"
        ),
    }
}

fn argument_depth(action: &KnowledgeAction) -> usize {
    fn depth(value: &Value) -> usize {
        match value {
            Value::Array(values) => 1 + values.iter().map(depth).max().unwrap_or(0),
            Value::Object(values) => 1 + values.values().map(depth).max().unwrap_or(0),
            _ => 0,
        }
    }
    match action {
        KnowledgeAction::ToolCall { arguments, .. } => depth(arguments).max(1),
        _ => 1,
    }
}

#[derive(Debug, Error)]
pub enum ToolCurriculumError {
    #[error("invalid tool curriculum configuration: {0}")]
    InvalidConfig(String),
    #[error("tool curriculum generation failed: {0}")]
    Generate(String),
    #[error("invalid tool contract exam: {0}")]
    InvalidExam(String),
    #[error(transparent)]
    Contract(#[from] ToolContractError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContractParameter, ToolContract};

    fn contracts() -> ToolContractSnapshot {
        let tools = (0..4)
            .map(|index| ToolContract {
                name: format!("tool_{index}"),
                description: format!("Operate resource class {index}."),
                parameters: vec![ContractParameter {
                    name: "payload".into(),
                    description: "Nested request payload.".into(),
                    required: true,
                    param_type: "object".into(),
                    schema: Some(serde_json::json!({
                        "type": "object",
                        "properties": {
                            "tags": {"type":"array", "minItems":1, "items":{"type":"string"}},
                            "count": {"type":"integer", "minimum":2, "maximum":5}
                        },
                        "required": ["tags", "count"],
                        "additionalProperties": false
                    })),
                }],
                danger: Value::Null,
                source: Value::Null,
            })
            .collect();
        ToolContractSnapshot::from_contracts(tools).unwrap()
    }

    #[test]
    fn covers_every_contract_with_runtime_valid_calls() {
        let contracts = contracts();
        let records = ToolCurriculumGenerator::new(contracts.clone(), 42, 4)
            .unwrap()
            .generate("train", 3)
            .unwrap();
        assert_eq!(records.len(), 12);
        let families: BTreeSet<_> = records.iter().map(|record| &record.family).collect();
        assert_eq!(families.len(), 4);
        for record in records {
            assert_eq!(record.tools.len(), 4);
            assert_eq!(
                record.prompt,
                render_knowledge_prompt(&record.tools, &record.conversation)
            );
            let KnowledgeAction::ToolCall { tool, arguments } =
                KnowledgeAction::parse_runtime(&record.target).unwrap()
            else {
                panic!("expected tool call")
            };
            assert!(contracts.validate_call(&tool, &arguments).valid);
        }
    }

    #[test]
    fn fixed_seed_is_byte_deterministic() {
        let generator = ToolCurriculumGenerator::new(contracts(), 7, 3).unwrap();
        assert_eq!(
            generator.generate("same", 2).unwrap(),
            generator.generate("same", 2).unwrap()
        );
    }

    #[test]
    fn patterned_revision_tokens_are_sampled_as_runtime_valid_hashes() {
        let contracts = ToolContractSnapshot::from_contracts(vec![ToolContract {
            name: "study_note_revise".into(),
            description: "revise safely".into(),
            parameters: vec![ContractParameter {
                name: "expected_revision".into(),
                description: "current head".into(),
                required: true,
                param_type: "string".into(),
                schema: Some(serde_json::json!({
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$"
                })),
            }],
            danger: Value::Null,
            source: Value::Null,
        }])
        .unwrap();
        let record = ToolCurriculumGenerator::new(contracts.clone(), 51, 1)
            .unwrap()
            .generate("revision-token", 1)
            .unwrap()
            .remove(0);
        let KnowledgeAction::ToolCall { tool, arguments } =
            KnowledgeAction::parse_runtime(&record.target).unwrap()
        else {
            panic!("expected tool call")
        };
        let revision = arguments["expected_revision"].as_str().unwrap();
        assert_eq!(revision.len(), 64);
        assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert!(contracts.validate_call(&tool, &arguments).valid);
    }

    #[test]
    fn note_writes_include_runtime_prerequisite_results() {
        let citations = serde_json::json!({
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string", "minLength": 1},
                    "locator": {"type": "string", "minLength": 1},
                    "quote": {"type": "string", "minLength": 8}
                },
                "required": ["source_id", "locator", "quote"],
                "additionalProperties": false
            }
        });
        let parameter = |name: &str, required: bool, schema: Value| ContractParameter {
            name: name.into(),
            description: name.into(),
            required,
            param_type: schema
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("string")
                .into(),
            schema: Some(schema),
        };
        let tool = |name: &str, parameters: Vec<ContractParameter>| ToolContract {
            name: name.into(),
            description: format!("Exercise {name}."),
            parameters,
            danger: Value::Null,
            source: Value::Null,
        };
        let contracts = ToolContractSnapshot::from_contracts(vec![
            tool(
                "mcp__seshat__seshat_search",
                vec![parameter(
                    "query",
                    true,
                    serde_json::json!({"type": "string"}),
                )],
            ),
            tool(
                "study_note_read",
                vec![parameter(
                    "note_id",
                    true,
                    serde_json::json!({
                        "type": "string",
                        "pattern": "^note-[0-9a-f]{32}$"
                    }),
                )],
            ),
            tool(
                "study_note_append",
                vec![
                    parameter("title", true, serde_json::json!({"type": "string"})),
                    parameter("body", true, serde_json::json!({"type": "string"})),
                    parameter("citations", true, citations.clone()),
                ],
            ),
            tool(
                "study_note_revise",
                vec![
                    parameter(
                        "note_id",
                        true,
                        serde_json::json!({
                            "type": "string",
                            "pattern": "^note-[0-9a-f]{32}$"
                        }),
                    ),
                    parameter(
                        "expected_revision",
                        true,
                        serde_json::json!({
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$"
                        }),
                    ),
                    parameter("body", true, serde_json::json!({"type": "string"})),
                    parameter("citations", true, citations),
                ],
            ),
            tool("unrelated_distractor", vec![]),
        ])
        .unwrap();
        let selected = BTreeSet::from([
            "study_note_append".to_string(),
            "study_note_revise".to_string(),
        ]);
        let records = ToolCurriculumGenerator::new(contracts.clone(), 5153, 4)
            .unwrap()
            .generate_for_tools("grounded-notes", 1, &selected)
            .unwrap();
        assert_eq!(records.len(), 2);

        let append = records
            .iter()
            .find(|record| record.family == "tool_contract.study_note_append")
            .unwrap();
        assert_eq!(append.stage, WorkflowStage::DistillNote);
        assert_eq!(append.conversation.len(), 3);
        assert_eq!(
            append.conversation[1].pointer("/tool_calls/0/function/name"),
            Some(&Value::String("mcp__seshat__seshat_search".into()))
        );
        assert_eq!(append.conversation[2]["role"], "tool");

        let revise = records
            .iter()
            .find(|record| record.family == "tool_contract.study_note_revise")
            .unwrap();
        assert_eq!(revise.stage, WorkflowStage::ReviseNote);
        assert_eq!(revise.conversation.len(), 5);
        assert_eq!(
            revise.conversation[1].pointer("/tool_calls/0/function/name"),
            Some(&Value::String("study_note_read".into()))
        );
        assert_eq!(
            revise.conversation[3].pointer("/tool_calls/0/function/name"),
            Some(&Value::String("mcp__seshat__seshat_search".into()))
        );
        let KnowledgeAction::ToolCall { tool, arguments } =
            KnowledgeAction::parse_runtime(&revise.target).unwrap()
        else {
            panic!("expected revision tool call")
        };
        let note_id = arguments["note_id"].as_str().unwrap();
        assert_eq!(note_id.len(), 37);
        assert!(note_id.starts_with("note-"));
        let head: Value =
            serde_json::from_str(revise.conversation[2]["content"].as_str().unwrap()).unwrap();
        assert_eq!(head["note_id"], arguments["note_id"]);
        assert_eq!(head["revision"], arguments["expected_revision"]);
        assert_eq!(head["document"]["source"], "Inline");
        assert_eq!(head["document"]["format"], "PlainText");
        assert_eq!(head["frontmatter"]["type"]["text"], "StudyNote");
        assert_eq!(head["wikilinks"], serde_json::json!([]));
        assert_eq!(head["tasks"], serde_json::json!([]));
        let evidence: Value =
            serde_json::from_str(revise.conversation[4]["content"].as_str().unwrap()).unwrap();
        assert_eq!(
            evidence[0]["document_id"],
            arguments["citations"][0]["source_id"]
        );
        assert_eq!(evidence[0]["locator"], arguments["citations"][0]["locator"]);
        assert_eq!(evidence[0]["text"], arguments["citations"][0]["quote"]);
        assert!(contracts.validate_call(&tool, &arguments).valid);
        assert_eq!(offered_tool_names(&revise.tools).unwrap().len(), 4);

        let without_evidence = ToolContractSnapshot::from_contracts(
            contracts
                .tools
                .iter()
                .filter(|candidate| candidate.name == "study_note_append")
                .cloned()
                .collect(),
        )
        .unwrap();
        assert!(matches!(
            ToolCurriculumGenerator::new(without_evidence, 5153, 1)
                .unwrap()
                .generate("unsafe-note-contract", 1),
            Err(ToolCurriculumError::Generate(message))
                if message.contains("evidence-search tool")
        ));

        let low_difficulty = ToolCurriculumGenerator::new(contracts, 5153, 4)
            .unwrap()
            .generate_exam_for_tools(
                "grounded-revision-probe",
                CheckpointRef {
                    step: 7,
                    artifact_hash: "blake3:grounded-revision".into(),
                },
                ExamTier::Frontier,
                1,
                1,
                &BTreeSet::from(["study_note_revise".to_string()]),
            )
            .unwrap();
        assert_eq!(low_difficulty.offered_tool_count, 3);
        assert_eq!(low_difficulty.cases[0].tools.len(), 3);
        assert_eq!(low_difficulty.cases[0].conversation.len(), 5);
    }

    #[test]
    fn selected_generation_targets_only_named_tools_but_keeps_live_distractors() {
        let contracts = contracts();
        let generator = ToolCurriculumGenerator::new(contracts.clone(), 8, 4).unwrap();
        let selected = BTreeSet::from(["tool_2".to_string()]);
        let records = generator.generate_for_tools("delta", 5, &selected).unwrap();
        assert_eq!(records.len(), 5);
        assert!(
            records
                .iter()
                .all(|record| record.family == "tool_contract.tool_2" && record.tools.len() == 4)
        );
        assert!(records.iter().all(|record| {
            record.tools.iter().any(|tool| {
                tool.pointer("/function/name").and_then(Value::as_str) == Some("tool_2")
            })
        }));

        let unknown = BTreeSet::from(["future_but_not_captured".to_string()]);
        assert!(matches!(
            generator.generate_for_tools("delta", 1, &unknown),
            Err(ToolCurriculumError::InvalidConfig(_))
        ));
    }

    #[test]
    fn delta_curriculum_replays_current_tools_and_teaches_legacy_absence() {
        let previous = contracts();
        let mut current_tools = previous.tools.clone();
        current_tools.retain(|tool| tool.name != "tool_3");
        current_tools
            .iter_mut()
            .find(|tool| tool.name == "tool_1")
            .unwrap()
            .parameters
            .push(ContractParameter {
                name: "mode".into(),
                description: "New required mode".into(),
                required: true,
                param_type: "string".into(),
                schema: Some(serde_json::json!({"type":"string","enum":["safe","fast"]})),
            });
        current_tools.push(ToolContract {
            name: "tool_4".into(),
            description: "New operation".into(),
            parameters: vec![],
            danger: Value::Null,
            source: Value::Null,
        });
        let current = ToolContractSnapshot::from_contracts(current_tools).unwrap();
        let generator =
            ToolContractDeltaCurriculumGenerator::new(previous, current.clone(), 51_500, 4)
                .unwrap();
        assert_eq!(
            generator.delta().model_target_tools(),
            BTreeSet::from(["tool_1".into(), "tool_4".into()])
        );

        let records = generator.generate("adapt-1", 2, 3, 4).unwrap();
        assert_eq!(records.len(), 22);
        assert!(
            records
                .iter()
                .all(|record| record.contract_hash == current.manifest_hash)
        );
        assert!(records.iter().all(|record| {
            record.tools.len() == 4
                && record.tools.iter().all(|tool| {
                    tool.pointer("/function/name").and_then(Value::as_str) != Some("tool_3")
                })
        }));
        let absence: Vec<_> = records
            .iter()
            .filter(|record| record.family.starts_with("tool_contract_absence."))
            .collect();
        assert_eq!(absence.len(), 8);
        assert!(absence.iter().all(|record| {
            matches!(
                KnowledgeAction::parse_runtime(&record.target).unwrap(),
                KnowledgeAction::Gap { .. }
            ) && record.stage == WorkflowStage::DeclareGap
        }));
    }

    #[test]
    fn metadata_only_delta_refuses_unnecessary_weight_adaptation() {
        let previous = contracts();
        let mut current_tools = previous.tools.clone();
        current_tools[0].danger = serde_json::json!({"level":"Cautious"});
        let current = ToolContractSnapshot::from_contracts(current_tools).unwrap();
        let generator =
            ToolContractDeltaCurriculumGenerator::new(previous, current, 19, 4).unwrap();
        assert!(!generator.delta().model_adaptation_required());
        assert!(matches!(
            generator.generate("metadata", 1, 1, 1),
            Err(ToolCurriculumError::InvalidConfig(_))
        ));
    }

    #[test]
    fn exam_is_checkpoint_bound_and_covers_every_contract() {
        let contracts = contracts();
        let checkpoint = CheckpointRef {
            step: 17,
            artifact_hash: "blake3:model-seventeen".into(),
        };
        let exam = ToolCurriculumGenerator::new(contracts.clone(), 71, 4)
            .unwrap()
            .generate_exam(
                "rolling-17",
                checkpoint.clone(),
                ExamTier::RollingProbe,
                3,
                5,
            )
            .unwrap();
        exam.verify(&contracts).unwrap();
        assert_eq!(exam.cases.len(), 20);
        assert_eq!(exam.checkpoint, checkpoint);
        assert!(exam.target_tools.is_empty());
        let counts: BTreeMap<_, _> = exam.cases.iter().fold(BTreeMap::new(), |mut counts, case| {
            *counts.entry(case.tool.as_str()).or_insert(0usize) += 1;
            counts
        });
        assert_eq!(
            counts.values().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([5])
        );

        let mut tampered = exam;
        tampered.cases.pop();
        assert!(tampered.verify(&contracts).is_err());
    }

    #[test]
    fn targeted_exam_covers_only_selected_family_and_keeps_live_distractors() {
        let contracts = contracts();
        let selected = BTreeSet::from(["tool_2".to_string()]);
        let checkpoint = CheckpointRef {
            step: 19,
            artifact_hash: "blake3:model-nineteen".into(),
        };
        let exam = ToolCurriculumGenerator::new(contracts.clone(), 72, 4)
            .unwrap()
            .generate_exam_for_tools(
                "frontier-19-tool-2",
                checkpoint,
                ExamTier::Frontier,
                3,
                5,
                &selected,
            )
            .unwrap();
        exam.verify(&contracts).unwrap();
        assert_eq!(exam.target_tools, selected);
        assert_eq!(exam.cases.len(), 5);
        assert!(exam.cases.iter().all(|case| {
            case.tool == "tool_2"
                && case.tools.len() == 4
                && case.tools.iter().any(|tool| {
                    tool.pointer("/function/name").and_then(Value::as_str) == Some("tool_2")
                })
        }));

        let mut tampered = exam.clone();
        tampered.target_tools = BTreeSet::from(["tool_1".to_string()]);
        assert!(tampered.verify(&contracts).is_err());

        let generator = ToolCurriculumGenerator::new(contracts, 72, 4).unwrap();
        assert!(matches!(
            generator.generate_exam_for_tools(
                "partial-sentinel",
                exam.checkpoint,
                ExamTier::Sentinel,
                3,
                5,
                &exam.target_tools,
            ),
            Err(ToolCurriculumError::InvalidConfig(_))
        ));
    }

    #[test]
    fn tool_frontier_derives_a_separated_targeted_training_sibling() {
        let contracts = contracts();
        let checkpoint = CheckpointRef {
            step: 22,
            artifact_hash: "blake3:model-twenty-two".into(),
        };
        let exam = ToolCurriculumGenerator::new(contracts.clone(), 700, 4)
            .unwrap()
            .generate_exam_for_tools(
                "frontier-tools",
                checkpoint,
                ExamTier::Frontier,
                3,
                6,
                &BTreeSet::from(["tool_2".to_string()]),
            )
            .unwrap();
        assert_eq!(exam.cases.len(), 6);
        let plan =
            derive_tool_contract_training_sibling(&exam, &contracts, "tool_contract.tool_2", 12)
                .unwrap();
        plan.verify_against_exam(&exam, &contracts).unwrap();
        let records = plan.generate(&contracts).unwrap();
        assert_eq!(records.len(), 12);
        assert!(
            records
                .iter()
                .all(|record| record.family == "tool_contract.tool_2")
        );

        let exam_requests: Vec<_> = exam
            .cases
            .iter()
            .filter(|case| case.tool == "tool_2")
            .filter_map(|case| case.conversation[0].get("content"))
            .filter_map(Value::as_str)
            .collect();
        let sibling_requests: Vec<_> = records
            .iter()
            .filter_map(|record| record.conversation[0].get("content"))
            .filter_map(Value::as_str)
            .collect();
        let maximum = exam_requests
            .iter()
            .flat_map(|left| {
                sibling_requests
                    .iter()
                    .map(move |right| crate::character_ngram_jaccard(left, right, 3))
            })
            .fold(0.0, f64::max);
        let evidence = SiblingSeparationEvidence {
            held_out_axes: vec!["prompt_bank".into(), "seed_namespace".into()],
            max_transcript_ngram_jaccard: maximum,
            min_image_hash_hamming_distance: None,
        };
        plan.validate_separation(&evidence).unwrap();

        let mut edited = plan;
        edited.seed ^= 1;
        assert!(edited.verify_against_exam(&exam, &contracts).is_err());
    }

    #[test]
    fn protected_tool_exam_cannot_derive_a_training_sibling() {
        let contracts = contracts();
        let exam = ToolCurriculumGenerator::new(contracts.clone(), 701, 4)
            .unwrap()
            .generate_exam(
                "sentinel-tools",
                CheckpointRef {
                    step: 0,
                    artifact_hash: "unbound:immutable-sentinel".into(),
                },
                ExamTier::Sentinel,
                3,
                2,
            )
            .unwrap();
        assert!(matches!(
            derive_tool_contract_training_sibling(&exam, &contracts, "tool_contract.tool_2", 12,),
            Err(ToolCurriculumError::InvalidConfig(_))
        ));
    }

    #[test]
    fn all_tool_sentinel_is_frozen_and_checkpoint_independent() {
        let contracts = contracts();
        let sentinel = ToolCurriculumGenerator::new(contracts.clone(), 702, 4)
            .unwrap()
            .generate_exam(
                "release-tools-v1",
                CheckpointRef {
                    step: 0,
                    artifact_hash: "unbound:immutable-sentinel".into(),
                },
                ExamTier::Sentinel,
                3,
                2,
            )
            .unwrap();
        sentinel.verify(&contracts).unwrap();
        let fingerprint = sentinel.exam_fingerprint.clone();
        let mut rebound = sentinel;
        rebound.checkpoint = CheckpointRef {
            step: 99,
            artifact_hash: "blake3:candidate-ninety-nine".into(),
        };
        assert_eq!(rebound.compute_fingerprint().unwrap(), fingerprint);
        assert!(rebound.verify(&contracts).is_err());
    }

    #[test]
    fn scoring_separates_precursor_from_catastrophic_failures() {
        let contracts = contracts();
        let exam = ToolCurriculumGenerator::new(contracts.clone(), 99, 4)
            .unwrap()
            .generate_exam(
                "frontier",
                CheckpointRef {
                    step: 4,
                    artifact_hash: "blake3:model-four".into(),
                },
                ExamTier::Frontier,
                3,
                1,
            )
            .unwrap();
        let mut predictions = BTreeMap::new();
        for (index, case) in exam.cases.iter().enumerate() {
            let raw = match index {
                0 => case.expected.render_runtime().unwrap(),
                1 => {
                    let KnowledgeAction::ToolCall {
                        tool,
                        mut arguments,
                    } = case.expected.clone()
                    else {
                        unreachable!()
                    };
                    arguments.as_object_mut().unwrap().insert(
                        "payload".into(),
                        serde_json::json!({"tags":["changed"],"count":3}),
                    );
                    KnowledgeAction::ToolCall { tool, arguments }
                        .render_runtime()
                        .unwrap()
                }
                2 => "not a runtime action".into(),
                _ => KnowledgeAction::ToolCall {
                    tool: "tool_0".into(),
                    arguments: serde_json::json!({"payload":{"tags":["x"],"count":3}}),
                }
                .render_runtime()
                .unwrap(),
            };
            predictions.insert(case.id.clone(), raw);
        }
        let report = score_tool_contract_exam(&exam, &predictions, &contracts);
        let scores: Vec<_> = report
            .families
            .iter()
            .flat_map(|family| family.cases.iter())
            .collect();
        assert_eq!(scores.iter().filter(|score| score.passed).count(), 1);
        assert_eq!(scores.iter().filter(|score| score.precursor).count(), 1);
        assert_eq!(scores.iter().filter(|score| score.catastrophic).count(), 2);
        assert_eq!(report.observations().len(), contracts.tools.len());
        let specs = tool_contract_capability_specs(&contracts).unwrap();
        assert_eq!(specs.len(), contracts.tools.len());
        assert!(
            specs
                .iter()
                .all(|spec| spec.id.starts_with("tool_contract.") && spec.initial_difficulty == 3)
        );
    }
}
