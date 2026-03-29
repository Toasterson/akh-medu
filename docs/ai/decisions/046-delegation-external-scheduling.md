# ADR 046 — Delegation & External Scheduling

> Date: 2026-03-28
> Status: Proposed
> Phase: 36
> Depends on: Phase 28 (n akh router), ADR-014 (multi-agent communication),
>             ADR-027 (shared workspace agent), ADR-029 (MCP server)
> Enhances: Phase 31f (skill sharing), Phase 13i (delegation)

## Context

Akh-medu has sophisticated internal scheduling (13 daemon timers, ECA triggers,
drive-based goal generation) and the reasoning capability to decompose complex
goals into subtasks (HTN, MCTS). However, all execution is confined to a single
process. When the agent identifies work requiring LLM capability (code generation,
document synthesis, research), it has no mechanism to delegate that work to
external agents that possess that capability.

Simultaneously, Claude Code now offers `RemoteTrigger` (scheduled agents on
Anthropic infrastructure) and `CronCreate` (session-local scheduling), plus the
ability to spawn subprocesses via `claude --print`. These are powerful execution
targets — but they lack persistent memory, structured reasoning, or knowledge
graph access.

The gap is **bidirectional orchestration**: akh-medu directing LLM-powered workers,
and external schedulers (cron, Claude Code triggers) directing akh-medu work.

## Decision

### Introduce a DelegationManager with four execution targets

The DelegationManager is a new agent subsystem that:

1. Accepts delegation requests from the OODA loop (when a goal is scored as
   requiring external capability by the n akh router)
2. Selects the best execution target based on task requirements, cost, and
   availability
3. Dispatches work asynchronously
4. Collects results and grounds them back into the KG with full provenance

### Four delegation targets (sovereignty gradient)

| Target | Cost | Latency | Survives restart | Requires |
|--------|------|---------|------------------|----------|
| **Local Claude subprocess** | Free (Max plan) | 10-30s | No | `claude` binary on PATH |
| **Peer akh-medu instance** | Free | 1-5s | N/A (stateless request) | Network access to peer |
| **Remote trigger** (claude.ai) | Per-token API cost | 5-60s | Yes | OAuth, public akhomed endpoint |
| **MCP client** (generic) | Varies | Varies | N/A | MCP server endpoint |

Default configuration enables only **local subprocess**. Each additional target
is opt-in via `akh.toml`:

```toml
[delegation]
enabled = true
max_concurrent = 3
daily_cost_budget = 0.0       # 0 = local-only (no API cost)

[delegation.local]
enabled = true
model = "sonnet"              # Default model for local subprocess
allowed_tools = ["Read", "Write", "Edit", "Bash", "Grep", "Glob"]
timeout_seconds = 120

[delegation.remote]
enabled = false               # Opt-in: requires API cost
akhomed_public_url = ""       # Must be set for MCP callback
model = "sonnet"
max_cost_per_task = 0.10      # USD

[delegation.peers]
enabled = false
# peers discovered via mDNS or explicit config
endpoints = []
```

### Bidirectional MCP pattern

The critical architectural insight: akh-medu is already an MCP server. When it
spawns a Claude Code subprocess, it passes an `.mcp.json` pointing back to
itself. The subprocess can query the KG, assert triples, check provenance —
all through the same MCP tools that Claude Code users already use.

```
akh-medu daemon
    │
    ├─ Spawns: claude --print -p "task..." --mcp-config /tmp/akh-delegate.mcp.json
    │
    │   Claude Code subprocess
    │       ├─ Reads task prompt
    │       ├─ Calls akh-medu MCP: triples_of("module_name")
    │       ├─ Calls akh-medu MCP: explain_belief("concept")
    │       ├─ Generates code / analysis / document
    │       └─ Returns structured result on stdout
    │
    └─ Ingests result:
        ├─ Parse stdout for structured output (JSON blocks)
        ├─ Assert result triples into KG
        └─ Store provenance: DerivationKind::Delegation { target, task_id }
```

For remote triggers, the pattern is identical except the Claude Code process
runs on Anthropic infrastructure and calls back to akhomed via its public URL.

### Cost control is a hard architectural constraint

Given prior cost incidents (100 EUR/weekend), the DelegationManager enforces:

1. **Daily budget** — atomic counter, checked before every dispatch
2. **Per-task cap** — estimated tokens × model cost, must fit within budget
3. **Model tiering** — haiku for simple extraction, sonnet for code, opus requires
   explicit approval (never auto-selected)
4. **Local-first** — sovereignty gradient always prefers in-process > local subprocess
   > peer > remote > cloud
5. **Cost reporting** — every delegation records actual cost in provenance

### n akh router integration

Phase 28's semantic router already classifies queries by complexity and selects
backends. Extend with delegation-aware backends:

```rust
pub enum Backend {
    // Existing (Phase 28):
    RuleParser,
    OnnxNer,
    CandleLlm,
    VsaDirect,
    CloudApi,
    BurnLocal,
    // New (Phase 36):
    DelegateLocal,      // Claude Code subprocess
    DelegatePeer,       // Peer akh-medu instance
    DelegateRemote,     // Remote trigger
}
```

The bandit learns which delegation targets produce the best results for which
task archetypes, automatically optimizing the routing over time.

### Provenance chain

Every delegated task produces provenance:

```rust
pub enum DerivationKind {
    // ... existing variants ...
    Delegation {
        target: DelegationTargetKind,
        task_description: String,
        model_used: Option<String>,
        tokens_consumed: Option<u64>,
        cost_usd: Option<f64>,
        wall_time_ms: u64,
    },
}
```

This enables full auditability: "this triple was asserted by a sonnet-4.6
subprocess that was delegated by OODA cycle 847 in pursuit of goal X."

### Peer protocol extension

Extend AgentProtocolMessage (ADR-014) with delegation primitives:

- `DelegateTask { task_id, description, required_capabilities, deadline }`
- `TaskResult { task_id, status, triples_produced, provenance }`
- `CapabilityAdvertisement { capabilities, load, cost }`

Peer discovery via explicit configuration initially. Future: mDNS/DNS-SD
auto-discovery on LAN.

## Alternatives Considered

### 1. Direct HTTP API calls to LLM providers

Rejected — reinvents what Claude Code already does well (tool use, context
management, code generation). The subprocess pattern reuses Claude Code's
entire capability stack for free.

### 2. Embedded LLM for all tasks (Phase 26 Candle backend)

The local Candle/Burn models (T5-small, fine-tuned) handle NLU/NLG/knowledge
extraction. But they cannot write code, synthesize documents, or perform
complex multi-step analysis. Delegation handles the capability gap without
requiring a larger local model.

### 3. Always-on Claude Code session (long-running REPL)

Rejected — wasteful of resources, fragile (session timeouts), and violates
the sovereignty principle. On-demand subprocess spawning is cheaper and
more reliable.

### 4. Queue-based architecture (RabbitMQ, Redis)

Over-engineered for the expected scale (1-5 concurrent delegations). Tokio
channels + subprocess management is sufficient. If scale demands it later,
the DelegationManager interface can be backed by a queue without changing
the agent-facing API.

## Consequences

- The agent can leverage LLM capability for tasks that symbolic reasoning
  cannot handle, without sacrificing sovereignty (local subprocess is default).
- Cost is controlled at the architectural level — no surprise bills.
- The MCP bridge already exists; delegation reuses it bidirectionally.
- Provenance chain extends through delegation boundaries — full auditability.
- Peer coordination enables horizontal scaling for multi-domain deployments.
- Remote triggers enable scheduled work that survives machine restarts.
- The n akh router becomes the unified dispatch point for "reason locally
  vs. delegate externally."
