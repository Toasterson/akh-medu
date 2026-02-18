//! Multi-step planning: decompose goals into ordered tool-call sequences.
//!
//! A `Plan` is an ordered list of `PlanStep`s that the agent generates before
//! executing. Each step specifies a tool and its input. Plans are tracked per-goal
//! and support backtracking: when a step fails, the agent can generate an
//! alternative plan.

use super::error::AgentResult;
use super::goal::Goal;
use super::memory::{WorkingMemory, WorkingMemoryKind};
use super::tool::ToolInput;
use super::tool_semantics::encode_goal_semantics;
use crate::engine::Engine;
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Plan types
// ---------------------------------------------------------------------------

/// Status of an individual plan step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepStatus {
    /// Not yet executed.
    Pending,
    /// Currently being executed.
    Active,
    /// Completed successfully.
    Completed,
    /// Failed with a reason.
    Failed { reason: String },
    /// Skipped (e.g., no longer needed after backtracking).
    Skipped,
}

/// A single step in a plan.
#[derive(Debug, Clone)]
pub struct PlanStep {
    /// Which tool to call.
    pub tool_name: String,
    /// Pre-constructed input for the tool.
    pub tool_input: ToolInput,
    /// Why this step is in the plan.
    pub rationale: String,
    /// Current status.
    pub status: StepStatus,
    /// Index in the plan (0-based).
    pub index: usize,
}

/// Status of an entire plan.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanStatus {
    /// Plan is being executed.
    Active,
    /// All steps completed successfully.
    Completed,
    /// A step failed and the plan was abandoned.
    Failed { failed_step: usize, reason: String },
    /// Plan was replaced by a backtracking alternative.
    Superseded,
}

/// An ordered sequence of tool-call steps for achieving a goal.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Which goal this plan serves.
    pub goal_id: SymbolId,
    /// Ordered steps.
    pub steps: Vec<PlanStep>,
    /// Overall plan status.
    pub status: PlanStatus,
    /// Which attempt this is (0 = first, incremented on backtrack).
    pub attempt: u32,
    /// Human-readable summary of the plan strategy.
    pub strategy: String,
}

impl Plan {
    /// Get the next pending step to execute, if any.
    pub fn next_step(&self) -> Option<&PlanStep> {
        self.steps.iter().find(|s| s.status == StepStatus::Pending)
    }

    /// Get the index of the next pending step.
    pub fn next_step_index(&self) -> Option<usize> {
        self.steps
            .iter()
            .position(|s| s.status == StepStatus::Pending)
    }

    /// Mark a step as active.
    pub fn activate_step(&mut self, index: usize) {
        if let Some(step) = self.steps.get_mut(index) {
            step.status = StepStatus::Active;
        }
    }

    /// Mark a step as completed.
    pub fn complete_step(&mut self, index: usize) {
        if let Some(step) = self.steps.get_mut(index) {
            step.status = StepStatus::Completed;
        }
        // Check if all steps are done.
        if self.steps.iter().all(|s| s.status == StepStatus::Completed) {
            self.status = PlanStatus::Completed;
        }
    }

    /// Mark a step as failed, which fails the entire plan.
    pub fn fail_step(&mut self, index: usize, reason: &str) {
        if let Some(step) = self.steps.get_mut(index) {
            step.status = StepStatus::Failed {
                reason: reason.into(),
            };
        }
        // Skip all remaining pending steps.
        for step in &mut self.steps {
            if step.status == StepStatus::Pending {
                step.status = StepStatus::Skipped;
            }
        }
        self.status = PlanStatus::Failed {
            failed_step: index,
            reason: reason.into(),
        };
    }

    /// Whether this plan has remaining steps to execute.
    pub fn has_remaining_steps(&self) -> bool {
        self.steps.iter().any(|s| s.status == StepStatus::Pending)
    }

    /// How many steps have completed.
    pub fn completed_count(&self) -> usize {
        self.steps
            .iter()
            .filter(|s| s.status == StepStatus::Completed)
            .count()
    }

    /// Total number of steps.
    pub fn total_steps(&self) -> usize {
        self.steps.len()
    }
}

// ---------------------------------------------------------------------------
// Plan generation
// ---------------------------------------------------------------------------

/// Generate a multi-step plan for a goal based on its description, criteria,
/// and current KG/WM state.
///
/// The planner analyzes the goal and produces a sequence of tool calls that,
/// when executed in order, should satisfy the success criteria.
pub fn generate_plan(
    goal: &Goal,
    engine: &Engine,
    working_memory: &WorkingMemory,
    attempt: u32,
) -> AgentResult<Plan> {
    let goal_lower = goal.description.to_lowercase();
    let criteria_lower = goal.success_criteria.to_lowercase();

    // Check if this is a code generation goal — delegate to specialized planner.
    if is_code_goal(&goal_lower, &criteria_lower) {
        return generate_code_plan(goal, engine, working_memory, attempt);
    }

    // Determine what the goal needs via VSA semantic similarity.
    // Encode canonical strategy patterns as hypervectors and measure
    // interference with the goal vector. Falls back to keyword matching
    // if VSA encoding fails.
    let (needs_knowledge, needs_reasoning, needs_creation, needs_external, needs_similarity) = {
        let ops = engine.ops();
        let im = engine.item_memory();

        match encode_goal_semantics(&goal.description, &goal.success_criteria, engine, ops, im) {
            Ok(goal_vec) => {
                let score = |concepts: &[&str]| -> f32 {
                    crate::vsa::grounding::bundle_symbols(engine, ops, im, concepts)
                        .ok()
                        .and_then(|v| ops.similarity(&goal_vec, &v).ok())
                        .unwrap_or(0.5)
                };

                let knowledge_score = score(&[
                    "find", "query", "search", "discover", "explore", "list", "identify",
                ]);
                let reasoning_score =
                    score(&["reason", "infer", "deduce", "classify", "analyze", "why"]);
                let creation_score = score(&[
                    "create", "add", "build", "connect", "link", "store", "write",
                ]);
                let external_score =
                    score(&["file", "http", "command", "shell", "fetch", "download"]);
                let similarity_score = score(&["similar", "like", "related", "compare", "cluster"]);

                // Threshold: random similarity is ~0.50, so 0.55 indicates real signal.
                let threshold = 0.55;
                (
                    knowledge_score > threshold,
                    reasoning_score > threshold,
                    creation_score > threshold,
                    external_score > threshold,
                    similarity_score > threshold,
                )
            }
            Err(_) => {
                // Keyword fallback
                (
                    contains_any(
                        &goal_lower,
                        &[
                            "find", "query", "search", "discover", "what", "list", "identify",
                        ],
                    ),
                    contains_any(
                        &goal_lower,
                        &["reason", "infer", "deduce", "classify", "analyze", "why"],
                    ),
                    contains_any(
                        &goal_lower,
                        &[
                            "create", "add", "build", "connect", "link", "store", "write",
                        ],
                    ),
                    contains_any(
                        &goal_lower,
                        &[
                            "file", "http", "url", "command", "shell", "fetch", "download",
                        ],
                    ),
                    contains_any(
                        &goal_lower,
                        &["similar", "like", "related", "compare", "cluster"],
                    ),
                )
            }
        }
    };

    // Check if we have existing knowledge about the goal's subject.
    let goal_label = engine.resolve_label(goal.symbol_id);
    let has_existing_knowledge = !engine.triples_from(goal.symbol_id).is_empty();

    // Check WM for prior tool results related to this goal.
    let prior_results: Vec<&str> = working_memory
        .by_symbol(goal.symbol_id)
        .iter()
        .filter(|e| e.kind == WorkingMemoryKind::ToolResult)
        .map(|e| e.content.as_str())
        .collect();
    let has_prior_results = !prior_results.is_empty();

    let mut steps = Vec::new();
    let mut strategy_parts = Vec::new();

    // On backtrack attempts, try a different strategy ordering.
    let explore_first = attempt % 2 == 0;

    if explore_first {
        // Strategy A: explore → reason → synthesize

        // Step 1: Gather knowledge (always useful as a first step).
        if needs_knowledge || !has_existing_knowledge {
            steps.push(PlanStep {
                tool_name: "kg_query".into(),
                tool_input: ToolInput::new()
                    .with_param("symbol", &goal_label)
                    .with_param("direction", "both"),
                rationale: "Gather existing knowledge about the goal subject.".into(),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("explore KG");
        }

        // Step 2: Similarity search for related concepts.
        if needs_similarity || (!has_existing_knowledge && !needs_external) {
            steps.push(PlanStep {
                tool_name: "similarity_search".into(),
                tool_input: ToolInput::new()
                    .with_param("symbol", &goal_label)
                    .with_param("top_k", "5"),
                rationale: "Find similar concepts to broaden understanding.".into(),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("similarity search");
        }

        // Step 3: Reasoning if we have or will have knowledge.
        if needs_reasoning || (has_existing_knowledge && !needs_external) {
            let triples = engine.triples_from(goal.symbol_id);
            let expr = if let Some(t) = triples.first() {
                format!(
                    "(triple {} {} {})",
                    engine.resolve_label(t.subject),
                    engine.resolve_label(t.predicate),
                    engine.resolve_label(t.object),
                )
            } else {
                format!("(entity {})", goal_label)
            };
            steps.push(PlanStep {
                tool_name: "reason".into(),
                tool_input: ToolInput::new().with_param("expression", &expr),
                rationale: "Apply symbolic reasoning to derive new insights.".into(),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("reason");
        }

        // Step 4: Synthesize new knowledge if goal involves creation.
        if needs_creation {
            steps.push(PlanStep {
                tool_name: "kg_mutate".into(),
                tool_input: ToolInput::new()
                    .with_param("subject", &goal_label)
                    .with_param("predicate", "related-to")
                    .with_param("object", &criteria_lower)
                    .with_param("confidence", "0.7"),
                rationale: "Create new knowledge based on gathered insights.".into(),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("synthesize");
        }
    } else {
        // Strategy B (backtrack): reason → recall → explore → create

        // Step 1: Recall past experience.
        steps.push(PlanStep {
            tool_name: "memory_recall".into(),
            tool_input: ToolInput::new()
                .with_param("query_symbols", &goal_label)
                .with_param("top_k", "3"),
            rationale: "Recall past experience related to this goal.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });
        strategy_parts.push("recall experience");

        // Step 2: Reason about what we know.
        if has_existing_knowledge {
            let triples = engine.triples_from(goal.symbol_id);
            let expr = if let Some(t) = triples.first() {
                format!(
                    "(triple {} {} {})",
                    engine.resolve_label(t.subject),
                    engine.resolve_label(t.predicate),
                    engine.resolve_label(t.object),
                )
            } else {
                format!("(entity {})", goal_label)
            };
            steps.push(PlanStep {
                tool_name: "reason".into(),
                tool_input: ToolInput::new().with_param("expression", &expr),
                rationale: "Reason about existing knowledge before exploring further.".into(),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("reason first");
        }

        // Step 3: Targeted KG query.
        steps.push(PlanStep {
            tool_name: "kg_query".into(),
            tool_input: ToolInput::new()
                .with_param("symbol", &goal_label)
                .with_param("direction", "both"),
            rationale: "Targeted knowledge query after reasoning.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });
        strategy_parts.push("targeted query");

        // Step 4: Create if needed.
        if needs_creation {
            steps.push(PlanStep {
                tool_name: "kg_mutate".into(),
                tool_input: ToolInput::new()
                    .with_param("subject", &goal_label)
                    .with_param("predicate", "related-to")
                    .with_param("object", &criteria_lower)
                    .with_param("confidence", "0.7"),
                rationale: "Synthesize knowledge from alternative approach.".into(),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("create");
        }
    }

    // External tool steps (appended regardless of strategy).
    // Use VSA similarity to determine which external tools are relevant.
    if needs_external {
        let ops = engine.ops();
        let im = engine.item_memory();

        let external_tools: &[(&str, &[&str], &str)] = &[
            (
                "file_io",
                &["file", "read", "write", "save", "export", "data", "disk"],
                "file I/O",
            ),
            (
                "http_fetch",
                &["http", "url", "fetch", "web", "api", "download", "network"],
                "HTTP fetch",
            ),
            (
                "shell_exec",
                &["command", "shell", "execute", "run", "process", "script"],
                "shell exec",
            ),
        ];

        if let Ok(goal_vec) =
            encode_goal_semantics(&goal.description, &goal.success_criteria, engine, ops, im)
        {
            for (tool_name, concepts, strategy_label) in external_tools {
                let score = crate::vsa::grounding::bundle_symbols(engine, ops, im, concepts)
                    .ok()
                    .and_then(|v| ops.similarity(&goal_vec, &v).ok())
                    .unwrap_or(0.5);

                if score <= 0.55 {
                    continue;
                }

                let (tool_input, rationale) = match *tool_name {
                    "file_io" => (
                        ToolInput::new()
                            .with_param("action", "read")
                            .with_param("path", &format!("{}.txt", goal_label.replace(' ', "_"))),
                        format!("VSA-selected file I/O (similarity: {score:.3})."),
                    ),
                    "http_fetch" => (
                        ToolInput::new().with_param("url", "https://example.com"),
                        format!("VSA-selected HTTP fetch (similarity: {score:.3})."),
                    ),
                    "shell_exec" => (
                        ToolInput::new()
                            .with_param("command", "echo 'plan step'")
                            .with_param("timeout", "10"),
                        format!("VSA-selected shell exec (similarity: {score:.3})."),
                    ),
                    _ => continue,
                };

                steps.push(PlanStep {
                    tool_name: tool_name.to_string(),
                    tool_input,
                    rationale,
                    status: StepStatus::Pending,
                    index: steps.len(),
                });
                strategy_parts.push(strategy_label);
            }
        } else {
            // Keyword fallback for external tools
            if contains_any(&goal_lower, &["file", "read", "write", "save", "export"]) {
                steps.push(PlanStep {
                    tool_name: "file_io".into(),
                    tool_input: ToolInput::new()
                        .with_param("action", "read")
                        .with_param("path", &format!("{}.txt", goal_label.replace(' ', "_"))),
                    rationale: "Access file data relevant to the goal.".into(),
                    status: StepStatus::Pending,
                    index: steps.len(),
                });
                strategy_parts.push("file I/O");
            }
            if contains_any(&goal_lower, &["http", "url", "fetch", "api", "download"]) {
                steps.push(PlanStep {
                    tool_name: "http_fetch".into(),
                    tool_input: ToolInput::new().with_param("url", "https://example.com"),
                    rationale: "Fetch external data via HTTP.".into(),
                    status: StepStatus::Pending,
                    index: steps.len(),
                });
                strategy_parts.push("HTTP fetch");
            }
            if contains_any(&goal_lower, &["command", "shell", "run", "execute"]) {
                steps.push(PlanStep {
                    tool_name: "shell_exec".into(),
                    tool_input: ToolInput::new()
                        .with_param("command", "echo 'plan step'")
                        .with_param("timeout", "10"),
                    rationale: "Execute a shell command for the goal.".into(),
                    status: StepStatus::Pending,
                    index: steps.len(),
                });
                strategy_parts.push("shell exec");
            }
        }
    }

    // Ensure at least one step (fallback to KG query).
    if steps.is_empty() {
        steps.push(PlanStep {
            tool_name: "kg_query".into(),
            tool_input: ToolInput::new()
                .with_param("symbol", &goal_label)
                .with_param("direction", "both"),
            rationale: "Fallback: query KG for any relevant knowledge.".into(),
            status: StepStatus::Pending,
            index: 0,
        });
        strategy_parts.push("fallback KG query");
    }

    // If we have prior results and this isn't the first attempt, skip the
    // first step that matches a previously-used tool to avoid pure repetition.
    if has_prior_results && attempt > 0 {
        // Find tools used in prior results.
        let used_tools: Vec<String> = prior_results
            .iter()
            .filter_map(|content| {
                content
                    .strip_prefix("Tool result (")
                    .and_then(|s| s.find(')').map(|i| s[..i].to_string()))
            })
            .collect();

        // Skip the first step that duplicates a previously-used tool.
        if let Some(skip_idx) = steps.iter().position(|s| used_tools.contains(&s.tool_name)) {
            steps[skip_idx].status = StepStatus::Skipped;
        }
    }

    let strategy = format!("Attempt {}: {}", attempt + 1, strategy_parts.join(" → "));

    Ok(Plan {
        goal_id: goal.goal_id(),
        steps,
        status: PlanStatus::Active,
        attempt,
        strategy,
    })
}

/// Generate an alternative plan after the current one failed.
///
/// Increments the attempt counter and uses a different strategy ordering.
pub fn backtrack_plan(
    goal: &Goal,
    failed_plan: &Plan,
    engine: &Engine,
    working_memory: &WorkingMemory,
) -> AgentResult<Plan> {
    generate_plan(goal, engine, working_memory, failed_plan.attempt + 1)
}

// ---------------------------------------------------------------------------
// Code-aware planning (Phase 10c)
// ---------------------------------------------------------------------------

/// Code generation keywords that trigger code-aware planning.
const CODE_KEYWORDS: &[&str] = &[
    "generate",
    "implement",
    "write code",
    "create function",
    "create struct",
    "create enum",
    "create trait",
    "create module",
    "code_gen",
    "write rust",
    "generate code",
    "implement function",
    "implement struct",
    "implement trait",
    "scaffold",
    "stub",
    "boilerplate",
];

/// Detect whether a goal is about code generation.
fn is_code_goal(goal_lower: &str, criteria_lower: &str) -> bool {
    let combined = format!("{goal_lower} {criteria_lower}");
    CODE_KEYWORDS.iter().any(|kw| combined.contains(kw))
}

/// Generate a code-aware plan with specialized steps for code generation.
///
/// Template:
/// 1. `kg_query` — gather existing code structure facts
/// 2. `code_ingest` — parse reference code if mentioned
/// 3. `kg_mutate` — define target entity and relations
/// 4. `code_gen` — generate code from KG
/// 5. `file_io` — write to target path (if file output implied)
/// 6. `compile_feedback` — validate with cargo check
/// 7. (on failure) backtrack to step 3
fn generate_code_plan(
    goal: &Goal,
    engine: &Engine,
    _working_memory: &WorkingMemory,
    attempt: u32,
) -> AgentResult<Plan> {
    let goal_lower = goal.description.to_lowercase();
    let goal_label = engine.resolve_label(goal.symbol_id);
    let mut steps = Vec::new();
    let mut strategy_parts = Vec::new();

    // Detect target name from goal description
    let target_name = extract_code_target(&goal.description).unwrap_or_else(|| goal_label.clone());

    // Detect scope from goal description
    let scope = detect_code_scope(&goal_lower);

    // Strategy alternation for backtracking
    let scaffold_first = attempt % 2 == 0;

    if scaffold_first {
        // Strategy A: scaffold-first — generate skeleton, then fill in
        strategy_parts.push("scaffold-first");

        // Step 1: Query existing code structure
        steps.push(PlanStep {
            tool_name: "kg_query".into(),
            tool_input: ToolInput::new()
                .with_param("symbol", &target_name)
                .with_param("direction", "both"),
            rationale: "Gather existing code structure knowledge.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });

        // Step 2: If goal mentions a reference file, ingest it
        if goal_lower.contains("like") || goal_lower.contains("similar to") || goal_lower.contains("based on") {
            steps.push(PlanStep {
                tool_name: "code_ingest".into(),
                tool_input: ToolInput::new().with_param("path", &target_name),
                rationale: "Parse reference code for structure learning.".into(),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("reference ingest");
        }

        // Step 3: Define the target in KG if needed
        let has_existing = engine.lookup_symbol(&target_name).is_ok();
        if !has_existing {
            steps.push(PlanStep {
                tool_name: "kg_mutate".into(),
                tool_input: ToolInput::new()
                    .with_param("subject", &target_name)
                    .with_param("predicate", "is-a")
                    .with_param("object", scope_to_type(scope))
                    .with_param("confidence", "0.9"),
                rationale: format!("Define '{}' as a {} in the KG.", target_name, scope_to_type(scope)),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("define target");
        }

        // Step 4: Generate code
        steps.push(PlanStep {
            tool_name: "code_gen".into(),
            tool_input: ToolInput::new()
                .with_param("target", &target_name)
                .with_param("scope", scope)
                .with_param("format", "true"),
            rationale: format!("Generate {} code for '{}'.", scope, target_name),
            status: StepStatus::Pending,
            index: steps.len(),
        });
        strategy_parts.push("generate");

        // Step 5: Write to file if goal implies file output
        if goal_lower.contains("file") || goal_lower.contains("write") || goal_lower.contains("save") {
            let filename = format!("{}.rs", target_name.replace('-', "_").to_lowercase());
            steps.push(PlanStep {
                tool_name: "file_io".into(),
                tool_input: ToolInput::new()
                    .with_param("action", "write")
                    .with_param("path", &filename)
                    .with_param("content", ""), // Will be filled from code_gen output
                rationale: format!("Write generated code to {}.", filename),
                status: StepStatus::Pending,
                index: steps.len(),
            });
            strategy_parts.push("write file");
        }

        // Step 6: Validate with compiler
        steps.push(PlanStep {
            tool_name: "compile_feedback".into(),
            tool_input: ToolInput::new()
                .with_param("command", "check"),
            rationale: "Validate generated code compiles.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });
        strategy_parts.push("validate");
    } else {
        // Strategy B: bottom-up — gather context, reason about structure, then generate
        strategy_parts.push("bottom-up");

        // Step 1: Recall past experience
        steps.push(PlanStep {
            tool_name: "memory_recall".into(),
            tool_input: ToolInput::new()
                .with_param("query_symbols", &target_name)
                .with_param("top_k", "3"),
            rationale: "Recall past code generation experience.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });

        // Step 2: Query KG for related code patterns
        steps.push(PlanStep {
            tool_name: "kg_query".into(),
            tool_input: ToolInput::new()
                .with_param("symbol", &target_name)
                .with_param("direction", "both"),
            rationale: "Query existing code structure and dependencies.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });

        // Step 3: Similarity search for analogous code
        steps.push(PlanStep {
            tool_name: "similarity_search".into(),
            tool_input: ToolInput::new()
                .with_param("symbol", &target_name)
                .with_param("top_k", "3"),
            rationale: "Find similar code structures for pattern reuse.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });

        // Step 4: Define structure in KG
        steps.push(PlanStep {
            tool_name: "kg_mutate".into(),
            tool_input: ToolInput::new()
                .with_param("subject", &target_name)
                .with_param("predicate", "is-a")
                .with_param("object", scope_to_type(scope))
                .with_param("confidence", "0.9"),
            rationale: "Define the target code structure in the KG.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });

        // Step 5: Generate code
        steps.push(PlanStep {
            tool_name: "code_gen".into(),
            tool_input: ToolInput::new()
                .with_param("target", &target_name)
                .with_param("scope", scope)
                .with_param("format", "true"),
            rationale: format!("Generate {} code with bottom-up context.", scope),
            status: StepStatus::Pending,
            index: steps.len(),
        });
        strategy_parts.push("generate");

        // Step 6: Validate
        steps.push(PlanStep {
            tool_name: "compile_feedback".into(),
            tool_input: ToolInput::new()
                .with_param("command", "check"),
            rationale: "Validate generated code compiles.".into(),
            status: StepStatus::Pending,
            index: steps.len(),
        });
        strategy_parts.push("validate");
    }

    let strategy = format!(
        "Code gen attempt {}: {}",
        attempt + 1,
        strategy_parts.join(" → ")
    );

    Ok(Plan {
        goal_id: goal.goal_id(),
        steps,
        status: PlanStatus::Active,
        attempt,
        strategy,
    })
}

/// Extract the code target name from goal description.
///
/// Looks for patterns like "implement X", "generate X", "create function X".
fn extract_code_target(description: &str) -> Option<String> {
    let lower = description.to_lowercase();

    let prefixes = [
        "implement ",
        "generate ",
        "create ",
        "write ",
        "scaffold ",
    ];

    for prefix in &prefixes {
        if let Some(rest) = lower.find(prefix).map(|i| &description[i + prefix.len()..]) {
            // Skip scope words
            let rest = rest.trim();
            let skip_words = ["function", "struct", "enum", "trait", "module", "a", "an", "the"];
            let mut words: Vec<&str> = rest.split_whitespace().collect();
            while !words.is_empty() && skip_words.contains(&words[0].to_lowercase().as_str()) {
                words.remove(0);
            }
            if let Some(target) = words.first() {
                return Some(target.to_string());
            }
        }
    }

    None
}

/// Detect code scope from goal description keywords.
fn detect_code_scope(goal_lower: &str) -> &'static str {
    if goal_lower.contains("struct") {
        "struct"
    } else if goal_lower.contains("enum") {
        "enum"
    } else if goal_lower.contains("trait") {
        "trait"
    } else if goal_lower.contains("impl") {
        "impl"
    } else if goal_lower.contains("module") || goal_lower.contains("mod ") {
        "module"
    } else if goal_lower.contains("file") {
        "file"
    } else {
        "function"
    }
}

/// Map scope keyword to KG type entity.
fn scope_to_type(scope: &str) -> &str {
    match scope {
        "struct" => "Struct",
        "enum" => "Enum",
        "trait" => "Trait",
        "impl" => "Implementation",
        "module" | "file" => "Module",
        _ => "Function",
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn contains_any(text: &str, keywords: &[&str]) -> bool {
    keywords.iter().any(|kw| text.contains(kw))
}

// Add a helper to Goal for convenience.
impl Goal {
    /// Get the goal's symbol ID (convenience for plan generation).
    pub fn goal_id(&self) -> SymbolId {
        self.symbol_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolId;

    fn _dummy_goal(desc: &str, criteria: &str) -> Goal {
        Goal {
            symbol_id: SymbolId::new(1).unwrap(),
            description: desc.into(),
            status: super::super::goal::GoalStatus::Active,
            priority: 128,
            success_criteria: criteria.into(),
            parent: None,
            children: Vec::new(),
            created_at: 0,
            cycles_worked: 0,
            last_progress_cycle: 0,
        }
    }

    #[test]
    fn plan_step_lifecycle() {
        let mut plan = Plan {
            goal_id: SymbolId::new(1).unwrap(),
            steps: vec![
                PlanStep {
                    tool_name: "kg_query".into(),
                    tool_input: ToolInput::new(),
                    rationale: "first".into(),
                    status: StepStatus::Pending,
                    index: 0,
                },
                PlanStep {
                    tool_name: "reason".into(),
                    tool_input: ToolInput::new(),
                    rationale: "second".into(),
                    status: StepStatus::Pending,
                    index: 1,
                },
            ],
            status: PlanStatus::Active,
            attempt: 0,
            strategy: "test".into(),
        };

        assert!(plan.has_remaining_steps());
        assert_eq!(plan.next_step_index(), Some(0));

        plan.activate_step(0);
        assert_eq!(plan.steps[0].status, StepStatus::Active);

        plan.complete_step(0);
        assert_eq!(plan.steps[0].status, StepStatus::Completed);
        assert_eq!(plan.completed_count(), 1);
        assert_eq!(plan.next_step_index(), Some(1));

        plan.activate_step(1);
        plan.complete_step(1);
        assert_eq!(plan.status, PlanStatus::Completed);
        assert!(!plan.has_remaining_steps());
    }

    #[test]
    fn plan_failure_skips_remaining() {
        let mut plan = Plan {
            goal_id: SymbolId::new(1).unwrap(),
            steps: vec![
                PlanStep {
                    tool_name: "a".into(),
                    tool_input: ToolInput::new(),
                    rationale: "".into(),
                    status: StepStatus::Pending,
                    index: 0,
                },
                PlanStep {
                    tool_name: "b".into(),
                    tool_input: ToolInput::new(),
                    rationale: "".into(),
                    status: StepStatus::Pending,
                    index: 1,
                },
                PlanStep {
                    tool_name: "c".into(),
                    tool_input: ToolInput::new(),
                    rationale: "".into(),
                    status: StepStatus::Pending,
                    index: 2,
                },
            ],
            status: PlanStatus::Active,
            attempt: 0,
            strategy: "test".into(),
        };

        plan.complete_step(0);
        plan.fail_step(1, "tool error");

        assert_eq!(
            plan.status,
            PlanStatus::Failed {
                failed_step: 1,
                reason: "tool error".into()
            }
        );
        assert_eq!(plan.steps[2].status, StepStatus::Skipped);
        assert!(!plan.has_remaining_steps());
    }
}
