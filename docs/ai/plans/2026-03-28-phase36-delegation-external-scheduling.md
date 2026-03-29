# Phase 36 — Delegation & External Scheduling

> Date: 2026-03-28
> Status: Planned
> Phase: 36
> Depends on: Phase 28 (n akh router), ADR-014 (multi-agent protocol),
>             ADR-029 (MCP server)
> Enhances: Phase 31f (skill sharing), Phase 13i (personal assistant delegation)
> ADR: [046-delegation-external-scheduling](../decisions/046-delegation-external-scheduling.md)

## Motivation

Akh-medu excels at **deciding what to do** (drives, goal generation, HTN
decomposition, MCTS planning) but is limited in **execution** to its own
symbolic reasoning tools. When a goal requires LLM-class capability — writing
code, synthesizing documents, performing nuanced research, reviewing PRs — the
agent hits a wall.

Meanwhile, Claude Code offers powerful execution through subprocess invocation
(`claude --print`), remote scheduled triggers (survive machine restarts), and
MCP client capabilities. These processes have no persistent memory or structured
reasoning, but they have frontier-model capability.

This phase connects akh-medu's symbolic reasoning brain to Claude Code's LLM
execution muscle through a bidirectional MCP bridge, creating a **neural-symbolic
feedback loop** where structured reasoning directs LLM work, and LLM results
ground back into symbolic knowledge.

## Architecture

```
                              Akh-medu Process
                    ┌───────────────────────────────────┐
                    │                                   │
                    │  OODA Loop                        │
                    │    │                              │
                    │    ├─ Goal: "Write tests for      │
                    │    │   causal module"              │
                    │    │                              │
                    │    ▼                              │
                    │  n akh Router                     │
                    │    │                              │
                    │    ├─ Backend::VsaDirect? No      │
                    │    ├─ Backend::CandleLlm? No      │
                    │    ├─ Backend::DelegateLocal? Yes  │
                    │    │                              │
                    │    ▼                              │
                    │  DelegationManager                │
                    │    │                              │
                    │    ├─ CostGuard: within budget?   │
                    │    ├─ Select target + model       │
                    │    ├─ Build prompt + MCP config   │
                    │    └─ Spawn worker ─────────────────┐
                    │                                   │ │
                    │  MCP Server (/mcp) ◄──────────────│─┤
                    │    │                              │ │
                    │    ├─ triples_of("causal") ◄──────│─┤
                    │    ├─ explain_belief(...) ◄────────│─┤
                    │    └─ assert_triple(...) ◄─────────│─┤
                    │                                   │ │
                    └───────────────────────────────────┘ │
                                                          │
                              Claude Code Worker          │
                    ┌─────────────────────────────────────┘
                    │
                    │  claude --print -p "..." --mcp-config /tmp/akh-del.mcp.json
                    │    │
                    │    ├─ Reads task prompt
                    │    ├─ Calls akh-medu MCP tools (KG queries, provenance)
                    │    ├─ Generates code / analysis / document
                    │    └─ Returns structured result on stdout
                    │
                    └─ Result parsed, triples asserted, provenance stored
```

For **remote triggers**, the Claude Code worker runs on Anthropic infrastructure
and reaches akhomed via its public URL. For **peer instances**, the worker is
another akh-medu process communicating via AgentProtocolMessage.

## Sub-phases

### 36a — DelegationManager Core (~600 lines)

**New module**: `src/agent/delegation.rs`

```rust
use std::sync::atomic::{AtomicU64, Ordering};
use dashmap::DashMap;
use tokio::sync::mpsc;

pub struct DelegationManager {
    targets: Vec<DelegationTarget>,
    active: DashMap<DelegationId, ActiveDelegation>,
    result_tx: mpsc::Sender<DelegationResult>,
    result_rx: mpsc::Receiver<DelegationResult>,
    cost_guard: CostGuard,
    config: DelegationConfig,
}

#[derive(Debug, Clone)]
pub struct DelegationId(pub u64);

pub struct DelegationTarget {
    pub kind: DelegationTargetKind,
    pub capabilities: Vec<String>,     // What this target can do
    pub max_concurrent: usize,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub enum DelegationTargetKind {
    /// Local Claude Code subprocess — free, fast
    LocalClaude {
        model: String,
        allowed_tools: Vec<String>,
        working_dir: PathBuf,
    },
    /// Remote trigger via claude.ai API — survives restarts
    RemoteTrigger {
        akhomed_public_url: String,
    },
    /// Another akh-medu instance — for KG-level operations
    AkhMeduPeer {
        endpoint: String,
        capability_token: Option<CapabilityToken>,
    },
    /// Generic MCP server
    McpServer {
        endpoint: String,
        transport: McpTransport,
    },
}

pub struct ActiveDelegation {
    pub id: DelegationId,
    pub goal_id: SymbolId,
    pub target: DelegationTargetKind,
    pub task_prompt: String,
    pub started_at: u64,
    pub status: DelegationStatus,
    pub provenance: ProvenanceId,
}

#[derive(Debug)]
pub enum DelegationStatus {
    Pending,
    Running { handle: String },         // PID or trigger ID
    Completed { result: DelegationResult },
    Failed { error: String, retries_left: u32 },
    TimedOut,
    CostExceeded,
}

pub struct DelegationResult {
    pub id: DelegationId,
    pub stdout: String,
    pub structured: Option<DelegationOutput>,
    pub wall_time: Duration,
    pub tokens_consumed: Option<u64>,
    pub cost_usd: Option<f64>,
    pub exit_code: Option<i32>,
}

/// Structured output extracted from worker stdout
pub enum DelegationOutput {
    /// Triples to assert into KG
    Triples(Vec<LabelTriple>),
    /// File operations performed
    FilesModified(Vec<PathBuf>),
    /// Free-form analysis text
    Analysis(String),
    /// Mixed: triples + text
    Mixed { triples: Vec<LabelTriple>, summary: String },
}
```

**CostGuard** (hard budget enforcement):

```rust
pub struct CostGuard {
    daily_budget_cents: u64,            // In cents to avoid float issues
    spent_today_cents: AtomicU64,
    per_task_max_cents: u64,
    budget_reset_at: AtomicU64,         // Unix timestamp of next reset
    model_costs: HashMap<String, u64>,  // Model → cents per 1M input tokens
}

impl CostGuard {
    pub fn can_afford(&self, estimated_tokens: u64, model: &str) -> bool {
        self.maybe_reset_daily();
        let cost = self.estimate_cost(estimated_tokens, model);
        let remaining = self.daily_budget_cents
            .saturating_sub(self.spent_today_cents.load(Ordering::Relaxed));
        cost <= remaining && cost <= self.per_task_max_cents
    }

    pub fn record_spend(&self, actual_tokens: u64, model: &str) {
        let cost = self.estimate_cost(actual_tokens, model);
        self.spent_today_cents.fetch_add(cost, Ordering::Relaxed);
    }
}
```

**DelegationConfig** (from `akh.toml`):

```rust
pub struct DelegationConfig {
    pub enabled: bool,
    pub max_concurrent: usize,
    pub daily_cost_budget_cents: u64,
    pub per_task_max_cents: u64,
    pub local: LocalDelegationConfig,
    pub remote: RemoteDelegationConfig,
    pub peers: PeerDelegationConfig,
}

pub struct LocalDelegationConfig {
    pub enabled: bool,
    pub model: String,                  // "sonnet", "haiku", "opus"
    pub allowed_tools: Vec<String>,
    pub timeout: Duration,
    pub working_dir: Option<PathBuf>,
}

pub struct RemoteDelegationConfig {
    pub enabled: bool,
    pub akhomed_public_url: String,
    pub model: String,
    pub max_cost_per_task_cents: u64,
}

pub struct PeerDelegationConfig {
    pub enabled: bool,
    pub endpoints: Vec<String>,         // Explicit peer URLs
    pub auto_discover: bool,            // Future: mDNS
}
```

**Error types** (miette diagnostics):

```rust
#[derive(Debug, Error, Diagnostic)]
pub enum DelegationError {
    #[error("Daily cost budget exceeded: spent {spent_cents}c of {budget_cents}c")]
    #[diagnostic(
        code(akh::delegation::budget_exceeded),
        help("Increase daily_cost_budget in akh.toml [delegation] section, or wait for budget reset at midnight")
    )]
    BudgetExceeded { spent_cents: u64, budget_cents: u64 },

    #[error("No delegation target available for capabilities: {required:?}")]
    #[diagnostic(
        code(akh::delegation::no_target),
        help("Enable additional targets in akh.toml: [delegation.local], [delegation.remote], or [delegation.peers]")
    )]
    NoTarget { required: Vec<String> },

    #[error("Delegation timed out after {timeout_secs}s")]
    #[diagnostic(
        code(akh::delegation::timeout),
        help("Increase timeout_seconds in akh.toml [delegation.local] section, or break the task into smaller subtasks")
    )]
    Timeout { timeout_secs: u64, task_id: DelegationId },

    #[error("Worker process failed with exit code {exit_code}")]
    #[diagnostic(
        code(akh::delegation::worker_failed),
        help("Check the task prompt for clarity. Worker stderr: {stderr}")
    )]
    WorkerFailed { exit_code: i32, stderr: String },

    #[error("Peer {endpoint} is unreachable")]
    #[diagnostic(
        code(akh::delegation::peer_unreachable),
        help("Check network connectivity to {endpoint}, or remove it from [delegation.peers] endpoints")
    )]
    PeerUnreachable { endpoint: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Io(#[from] std::io::Error),
}
```

**Modify**: `src/agent/mod.rs` (add `delegation` submodule), `src/error.rs`
(add `DelegationError` variant to `AkhError`)

### 36b — Local Claude Code Subprocess (~500 lines)

**New module**: `src/agent/delegation/local_claude.rs`

```rust
pub struct LocalClaudeWorker {
    config: LocalDelegationConfig,
    mcp_config_path: PathBuf,       // Generated .mcp.json pointing to akhomed
}

impl LocalClaudeWorker {
    /// Generate a temporary .mcp.json that points the subprocess back to akhomed
    fn generate_mcp_config(&self, akhomed_url: &str) -> AkhResult<PathBuf> {
        let config = serde_json::json!({
            "mcpServers": {
                "akh-medu": {
                    "type": "url",
                    "url": format!("{}/mcp", akhomed_url)
                }
            }
        });
        let path = std::env::temp_dir().join("akh-delegate.mcp.json");
        std::fs::write(&path, serde_json::to_string_pretty(&config)?)?;
        Ok(path)
    }

    pub async fn execute(
        &self,
        task: &DelegationTask,
        cost_guard: &CostGuard,
        result_tx: mpsc::Sender<DelegationResult>,
    ) -> AkhResult<DelegationId> {
        let id = DelegationId::next();

        // Build the system prompt with structured output instructions
        let prompt = self.build_prompt(task)?;

        // Spawn claude subprocess
        let mut cmd = tokio::process::Command::new("claude");
        cmd.args([
            "--print",
            "-p", &prompt,
            "--model", &self.config.model,
            "--mcp-config", self.mcp_config_path.to_str().unwrap(),
            "--output-format", "text",
        ]);

        for tool in &self.config.allowed_tools {
            cmd.args(["--allowedTools", tool]);
        }

        if let Some(ref dir) = self.config.working_dir {
            cmd.current_dir(dir);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let start = std::time::Instant::now();
        let child = cmd.spawn().map_err(|e| DelegationError::Io(e))?;

        // Spawn timeout watcher
        let timeout = self.config.timeout;
        tokio::spawn(async move {
            match tokio::time::timeout(timeout, child.wait_with_output()).await {
                Ok(Ok(output)) => {
                    let wall_time = start.elapsed();
                    let result = DelegationResult {
                        id: id.clone(),
                        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                        structured: Self::parse_structured_output(&output.stdout),
                        wall_time,
                        tokens_consumed: None, // TODO: parse from claude output
                        cost_usd: None,
                        exit_code: output.status.code(),
                    };
                    let _ = result_tx.send(result).await;
                }
                Ok(Err(e)) => {
                    let _ = result_tx.send(DelegationResult::failed(id, e.to_string())).await;
                }
                Err(_) => {
                    let _ = result_tx.send(DelegationResult::timed_out(id)).await;
                }
            }
        });

        Ok(id)
    }

    /// Build a task prompt that instructs the worker to use MCP tools
    /// and produce structured output
    fn build_prompt(&self, task: &DelegationTask) -> AkhResult<String> {
        let mut prompt = String::new();

        prompt.push_str("You are a worker agent delegated a task by akh-medu, ");
        prompt.push_str("a neuro-symbolic AI engine. You have access to its knowledge ");
        prompt.push_str("graph via MCP tools (akh-medu server).\n\n");

        prompt.push_str("## Task\n\n");
        prompt.push_str(&task.description);
        prompt.push_str("\n\n");

        // Include KG context snapshot if available
        if !task.context_triples.is_empty() {
            prompt.push_str("## Relevant Knowledge (from KG)\n\n");
            for triple in &task.context_triples {
                prompt.push_str(&format!(
                    "- {} --{}-> {}\n",
                    triple.subject, triple.predicate, triple.object
                ));
            }
            prompt.push_str("\n");
        }

        prompt.push_str("## Instructions\n\n");
        prompt.push_str("1. Use the akh-medu MCP tools to query relevant knowledge.\n");
        prompt.push_str("2. Perform the task.\n");
        prompt.push_str("3. At the end of your response, include a structured result block:\n\n");
        prompt.push_str("```akh-result\n");
        prompt.push_str("{\n");
        prompt.push_str("  \"status\": \"success\" | \"partial\" | \"failed\",\n");
        prompt.push_str("  \"summary\": \"brief description of what was done\",\n");
        prompt.push_str("  \"triples\": [\n");
        prompt.push_str("    {\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\"}\n");
        prompt.push_str("  ],\n");
        prompt.push_str("  \"files_modified\": [\"path/to/file\"],\n");
        prompt.push_str("  \"follow_up\": \"optional: what should happen next\"\n");
        prompt.push_str("}\n");
        prompt.push_str("```\n");

        Ok(prompt)
    }

    /// Parse structured output from the ```akh-result``` block in stdout
    fn parse_structured_output(stdout: &[u8]) -> Option<DelegationOutput> {
        let text = std::str::from_utf8(stdout).ok()?;
        let start = text.find("```akh-result\n")?;
        let after_marker = start + "```akh-result\n".len();
        let end = text[after_marker..].find("```")?;
        let json_str = &text[after_marker..after_marker + end];
        serde_json::from_str(json_str).ok()
    }
}

pub struct DelegationTask {
    pub description: String,
    pub context_triples: Vec<LabelTriple>,
    pub required_capabilities: Vec<String>,
    pub goal_id: SymbolId,
    pub priority: f32,
}
```

**Feature gate**: `delegation` feature (depends on `server` for MCP config generation)

```toml
[features]
delegation = ["server"]  # Needs akhomed URL for MCP callback
```

### 36c — Remote Trigger Integration (~400 lines)

**New module**: `src/agent/delegation/remote_trigger.rs`

```rust
pub struct RemoteTriggerWorker {
    config: RemoteDelegationConfig,
}

impl RemoteTriggerWorker {
    /// Create a one-shot remote agent for a specific task
    pub async fn delegate_once(
        &self,
        task: &DelegationTask,
        result_tx: mpsc::Sender<DelegationResult>,
    ) -> AkhResult<DelegationId> {
        let id = DelegationId::next();
        let prompt = self.build_remote_prompt(task)?;

        // Create trigger via claude.ai API
        let client = reqwest::Client::new();
        let create_resp = client
            .post("https://api.claude.ai/v1/code/triggers")
            .bearer_auth(&self.get_oauth_token()?)
            .json(&serde_json::json!({
                "name": format!("akh-delegation-{}", id.0),
                "prompt": prompt,
                "mcp_servers": [{
                    "type": "url",
                    "url": format!("{}/mcp", self.config.akhomed_public_url)
                }]
            }))
            .send()
            .await?;

        let trigger: TriggerResponse = create_resp.json().await?;

        // Run immediately
        let run_resp = client
            .post(format!(
                "https://api.claude.ai/v1/code/triggers/{}/run",
                trigger.trigger_id
            ))
            .bearer_auth(&self.get_oauth_token()?)
            .send()
            .await?;

        // Poll for completion (or set up webhook in future)
        tokio::spawn(async move {
            let result = Self::poll_trigger_completion(&trigger.trigger_id).await;
            let _ = result_tx.send(result).await;
        });

        Ok(id)
    }

    /// Create a recurring remote agent
    pub async fn delegate_recurring(
        &self,
        task: &DelegationTask,
        cron: &str,
    ) -> AkhResult<String> {
        let prompt = self.build_remote_prompt(task)?;

        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.claude.ai/v1/code/triggers")
            .bearer_auth(&self.get_oauth_token()?)
            .json(&serde_json::json!({
                "name": format!("akh-recurring-{}", task.goal_id),
                "prompt": prompt,
                "cron": cron,
                "mcp_servers": [{
                    "type": "url",
                    "url": format!("{}/mcp", self.config.akhomed_public_url)
                }]
            }))
            .send()
            .await?;

        let trigger: TriggerResponse = resp.json().await?;
        Ok(trigger.trigger_id)
    }
}
```

**Inbound webhook** (akhomed receives results from remote triggers):

Add a new HTTP endpoint to akhomed:

```rust
// POST /workspaces/{name}/delegation/callback
async fn delegation_callback(
    State(state): State<ServerState>,
    Path(workspace): Path<String>,
    Json(result): Json<DelegationCallbackPayload>,
) -> impl IntoResponse {
    let engine = state.get_engine(&workspace)?;
    let mut agent = state.get_agent(&workspace)?;
    agent.delegation_manager().process_result(result, &engine)?;
    StatusCode::OK
}
```

### 36d — Peer Discovery & Coordination (~500 lines)

**New module**: `src/agent/delegation/peer.rs`

Extend the AgentProtocolMessage (ADR-014) with delegation primitives:

```rust
/// New variants for AgentProtocolMessage
pub enum AgentProtocolMessage {
    // ... existing 10 variants from ADR-014 ...

    /// Request peer to perform a task using its KG capabilities
    DelegateTask {
        task_id: String,
        description: String,
        required_capabilities: Vec<String>,
        context_triples: Vec<Triple>,
        deadline: Option<u64>,
    },

    /// Report delegation result
    TaskResult {
        task_id: String,
        status: TaskCompletionStatus,
        triples_produced: Vec<Triple>,
        provenance: Vec<ProvenanceRecord>,
        wall_time_ms: u64,
    },

    /// Periodic capability advertisement (every 60s)
    CapabilityAdvertisement {
        capabilities: Vec<String>,
        active_skills: Vec<String>,
        load: f32,             // 0.0-1.0 current utilization
        available_capacity: usize,
    },
}

pub enum TaskCompletionStatus {
    Completed,
    PartialResult { reason: String },
    Rejected { reason: String },    // Peer declined (overloaded, no capability)
    Failed { error: String },
}
```

**PeerRegistry** (tracks known peers and their capabilities):

```rust
pub struct PeerRegistry {
    peers: DashMap<String, PeerInfo>,   // endpoint → info
}

pub struct PeerInfo {
    pub endpoint: String,
    pub capabilities: Vec<String>,
    pub active_skills: Vec<String>,
    pub last_advertisement: u64,
    pub load: f32,
    pub capability_token: Option<CapabilityToken>,
    pub success_rate: f32,              // Historical reliability
    pub avg_response_ms: f64,
}
```

**Peer selection** (when DelegationManager chooses DelegatePeer):

1. Filter peers by required capabilities
2. Exclude overloaded peers (load > 0.9)
3. Rank by: (1 - load) × success_rate × (1.0 / avg_response_ms)
4. Select top peer, send DelegateTask via CommChannel

### 36e — OODA Integration & Daemon Wiring (~400 lines)

**New tool**: `src/agent/tools/delegate.rs`

```rust
pub struct DelegateTool;

impl Tool for DelegateTool {
    fn name(&self) -> &str { "delegate" }
    fn description(&self) -> &str {
        "Delegate a task to an external Claude Code process or peer agent"
    }
    fn keywords(&self) -> &[&str] {
        &["delegate", "external", "llm", "code", "generate", "write", "synthesize"]
    }

    fn execute(&self, agent: &mut Agent, params: &ToolParams) -> ToolResult {
        let description = params.get_string("task")?;
        let target_hint = params.get_optional_string("target");

        // Gather context triples from active goal
        let goal_id = agent.active_goal().ok_or(DelegationError::NoActiveGoal)?;
        let context = agent.gather_goal_context(goal_id)?;

        let task = DelegationTask {
            description,
            context_triples: context,
            required_capabilities: vec![],
            goal_id,
            priority: agent.goal_priority(goal_id),
        };

        // DelegationManager selects target and dispatches
        let id = agent.delegation_manager().dispatch(task, target_hint)?;

        ToolResult::success(format!(
            "Delegated task {} to {} (tracking ID: {})",
            description,
            agent.delegation_manager().target_name(id),
            id.0
        ))
    }
}
```

**Daemon integration** (new idle task):

```rust
// In daemon event loop, add new timer:
let mut delegation_tick = tokio::time::interval(Duration::from_secs(10));

// In the tokio::select! block:
_ = delegation_tick.tick() => {
    let mut agent = agent_mutex.lock().unwrap();

    // 1. Process completed delegations
    while let Ok(result) = agent.delegation_manager().try_recv() {
        // Parse structured output
        if let Some(output) = &result.structured {
            match output {
                DelegationOutput::Triples(triples) => {
                    for triple in triples {
                        engine.assert_labeled_triple(triple)?;
                    }
                }
                DelegationOutput::Mixed { triples, summary } => {
                    for triple in triples {
                        engine.assert_labeled_triple(triple)?;
                    }
                    agent.add_to_working_memory(summary)?;
                }
                _ => {}
            }
        }

        // Store provenance
        let mut prov = ProvenanceRecord::new(
            result.id.as_symbol(),
            Vec::new(),
            DerivationKind::Delegation {
                target: result.target_kind(),
                task_description: result.task_prompt.clone(),
                model_used: result.model_used(),
                tokens_consumed: result.tokens_consumed,
                cost_usd: result.cost_usd,
                wall_time_ms: result.wall_time.as_millis() as u64,
            },
        );
        engine.store_provenance(&mut prov)?;

        // Update goal progress
        agent.advance_goal(result.goal_id, &result)?;
    }

    // 2. Check for timed-out delegations
    agent.delegation_manager().check_timeouts()?;

    // 3. Advertise capabilities to peers (every 60s)
    if should_advertise {
        agent.delegation_manager().broadcast_capabilities(&agent)?;
    }
}
```

**n akh router extension** (modify Phase 28 route selection):

```rust
// In NakhRouter, add delegation-aware archetypes:
QueryArchetype {
    name: "code_generation".into(),
    prototype: encode_label(ops, "write code implement function test"),
    preferred_backend: Some(Backend::DelegateLocal),
},
QueryArchetype {
    name: "document_synthesis".into(),
    prototype: encode_label(ops, "write document summarize report"),
    preferred_backend: Some(Backend::DelegateLocal),
},
QueryArchetype {
    name: "deep_research".into(),
    prototype: encode_label(ops, "research investigate analyze literature"),
    preferred_backend: Some(Backend::DelegateRemote),
},
QueryArchetype {
    name: "cross_domain_query".into(),
    prototype: encode_label(ops, "what does other instance know about"),
    preferred_backend: Some(Backend::DelegatePeer),
},
```

### 36f — MCP Tools for Delegation Management (~300 lines)

**Extend**: `src/mcp/mod.rs`

New MCP tools (accessible from Claude Code sessions):

| Tool | Description |
|------|-------------|
| `delegation_status` | List active/completed/failed delegations |
| `delegate_task` | Manually delegate a task (operator-initiated) |
| `delegation_config` | View/update delegation settings |
| `list_peers` | Show known peer instances and their capabilities |
| `delegation_cost_report` | Daily/weekly cost breakdown by target and model |
| `create_recurring_delegation` | Set up a recurring remote trigger |
| `cancel_delegation` | Cancel an active delegation |

**CLI subcommands**:

```
akh delegate "write tests for the causal module" [--target local|remote|peer]
akh delegate status
akh delegate cost-report
akh delegate peers
```

## Use Cases

### 1. Code Generation (local subprocess)

```
Agent goal: "Improve test coverage for event calculus module"
  → HTN decomposition: [analyze_coverage, identify_gaps, write_tests, verify]
  → "write_tests" scored as Backend::DelegateLocal by n akh router
  → DelegationManager spawns:
      claude --print -p "Write tests for event calculus. Use akh-medu MCP tools
        to understand the module structure. Focus on holds_at and project_state."
        --mcp-config /tmp/akh-del.mcp.json --model sonnet
  → Worker queries KG for event calculus concepts, writes test file
  → Result: 15 tests in src/tests/event_calculus_test.rs
  → Provenance: DerivationKind::Delegation { target: LocalClaude, model: "sonnet" }
```

### 2. Daily Knowledge Digest (remote trigger)

```
Operator: "Create a daily digest of KG changes"
  → create_recurring_delegation via MCP tool
  → RemoteTrigger created: cron "57 7 * * *" (7:57am daily)
  → Every morning, remote Claude Code agent:
      1. Calls akh-medu MCP: sparql_query("recent triples from last 24h")
      2. Calls akh-medu MCP: agent_goals (check active goals)
      3. Calls akh-medu MCP: delegation_cost_report
      4. Synthesizes daily digest email
      5. Sends via operator's preferred channel
```

### 3. Cross-Instance Knowledge Sharing (peer)

```
akh-medu "Research" instance (domain: papers, science)
akh-medu "Code" instance (domain: rust, systems)

Research instance discovers: "Bloom filters have O(1) lookup"
  → CapabilityAdvertisement includes "data_structures" capability
Code instance goal: "Optimize KG lookup performance"
  → n akh router scores Backend::DelegatePeer
  → DelegateTask sent to Research instance:
      "What data structures offer O(1) lookup with space efficiency?"
  → Research instance queries its KG, returns triples about:
      bloom filters, cuckoo filters, robin hood hashing
  → Code instance ingests results with Delegation provenance
```

### 4. PR Review Pipeline (local + MCP callback)

```
ECA trigger: NewTriples { predicate: "github:pr_opened" }
  → Condition: SPARQL "ASK { ?pr github:pr_opened ?repo . ?repo watched true }"
  → Action: delegate "Review PR {pr_url} for code quality and correctness"
  → Local Claude subprocess:
      1. Reads PR diff via GitHub MCP
      2. Queries akh-medu KG for relevant architecture context
      3. Generates review comments
      4. Posts review via GitHub MCP
  → Result triples: review_completed, issues_found, review_quality_score
```

## Configuration Example

```toml
# akh.toml

[delegation]
enabled = true
max_concurrent = 3
daily_cost_budget_cents = 0     # 0 = local-only mode (no API costs)

[delegation.local]
enabled = true
model = "sonnet"
allowed_tools = ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
timeout_seconds = 120
# working_dir defaults to akhomed's working directory

[delegation.remote]
enabled = false
akhomed_public_url = ""         # Required: e.g. "https://akh.example.com"
model = "sonnet"
max_cost_per_task_cents = 10    # 10 cents per task

[delegation.peers]
enabled = false
auto_discover = false           # Future: mDNS/DNS-SD
endpoints = [
    # "http://research-akh.local:3001",
    # "http://code-akh.local:3002",
]
```

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|-----------|-------|------------|--------------|
| 36a — DelegationManager core | ~600 | Medium | None (uses existing types) |
| 36b — Local Claude subprocess | ~500 | Low | `claude` binary on PATH |
| 36c — Remote trigger integration | ~400 | Medium | OAuth, public endpoint |
| 36d — Peer discovery & coordination | ~500 | High | ADR-014, CommChannel |
| 36e — OODA + daemon integration | ~400 | Medium | 36a, Phase 28 |
| 36f — MCP tools + CLI | ~300 | Low | 36a |
| **Total** | **~2,700** | | |

## Implementation Order

1. **36a** — DelegationManager core (foundation)
2. **36b** — Local Claude subprocess (immediately useful, zero cost)
3. **36e** — Daemon wiring + delegate tool (makes 36b accessible to OODA)
4. **36f** — MCP tools (makes delegation accessible to Claude Code users)
5. **36c** — Remote triggers (requires public endpoint infrastructure)
6. **36d** — Peer coordination (most complex, biggest payoff at scale)

## Feature Flags

```toml
[features]
delegation = []                         # Core DelegationManager + local subprocess
delegation-remote = ["delegation"]      # Remote trigger integration (reqwest for API)
delegation-peer = ["delegation"]        # Peer-to-peer coordination
```

## New Dependencies

| Crate | Purpose | Sub-phase |
|-------|---------|-----------|
| None (36a, 36b) | Uses tokio::process, serde_json, existing types | 36a, 36b |
| reqwest (already in deps) | Remote trigger API calls | 36c |
| None (36d) | Uses existing CommChannel infrastructure | 36d |

## Security Considerations

1. **MCP callback authentication**: The `.mcp.json` generated for local
   subprocesses points to `localhost` — no auth needed. Remote triggers
   connecting to a public akhomed URL MUST use the existing auth middleware.

2. **Subprocess sandboxing**: `--allowedTools` restricts what the Claude Code
   worker can do. Default excludes dangerous tools (no `Bash` with
   `dangerouslyDisableSandbox`, no file writes outside working directory).

3. **Peer trust**: Peers start at `ChannelKind::Public` (read-only KG access).
   Operator must explicitly grant capability tokens for assertion/mutation.

4. **Cost isolation**: Each delegation target has independent budget tracking.
   A misconfigured remote trigger cannot drain the local subprocess budget.

## Relationship to Other Phases

| Phase | Relationship |
|-------|-------------|
| **Phase 28** (n akh router) | Router selects delegation targets as backends |
| **Phase 29** (capability exposure) | Delegation status exposed via same MCP/CLI pattern |
| **Phase 30** (procedural skills) | Skills can include delegation steps in plan templates |
| **Phase 31f** (skill sharing) | Peer coordination enables skill exchange |
| **Phase 13i** (PA delegation) | Personal assistant delegation becomes a special case of general delegation |
| **Phase 11** (HTN decomposition) | HTN subtasks can be individually delegated |
| **Phase 16** (MCTS planning) | Planner can model delegation latency/cost in rollouts |
