# Heka — semantic memory for akh-medu

> *MemPalace stores what you said. Heka remembers what it meant.*

Heka is the long-term memory subsystem of [akh-medu](../../README.md).
It captures every interaction as a **drawer** (verbatim + VSA-encoded), links them into the
knowledge graph with full provenance, and surfaces knowledge through MCP tools that any AI assistant
can call natively or via the OpenAI-compatible bridge.

**LongMemEval (raw, R@5): ≥ 97%** — beats MemPalace (96.6%) while staying honest about top-k.  
**LongMemEval (hybrid, R@10): ≥ 92%** — beats MemPalace honest hybrid (88.9%) without the top_k=50
exploit.  
**LoCoMo: evaluated with honest top-k caps** — no cherry-picking.

See the full numbers in [docs/heka-vs-mempalace.md](../../docs/heka-vs-mempalace.md).

---

## Installation

Heka ships as part of the akh-medu workspace. Build everything from source:

```bash
git clone https://github.com/Toasterson/akh-medu
cd akh-medu
cargo build --release -p heka
```

Start the bridge daemon (exposes MCP tools + OpenAI-compat endpoint on port 7000):

```bash
./target/release/heka-bridge
```

Wire your editor. Drop the file for your tool into the config location it expects:

| Tool | Config |
|------|--------|
| Windsurf | `crates/heka/hooks/windsurf.md` |
| Cline | `crates/heka/hooks/cline.md` |
| Aider | `crates/heka/hooks/aider.md` |
| Codex | `crates/heka/hooks/codex.md` |

MCP servers (Claude Desktop, cursor-mcp) connect directly to the running akh-medu daemon:

```json
{
  "mcpServers": {
    "heka": {
      "command": "akh-medu",
      "args": ["mcp"]
    }
  }
}
```

---

## The three demo queries

### 1. Provenance chain — `heka explain`

Ask Heka *why* it believes something. Every fact carries a chain of source drawers, inference steps,
and evidence weights back to the raw conversation that created it.

```bash
akh-medu ask "heka explain: how do you know I prefer dark mode?"
```

Example output:

```
Belief: user prefers dark mode  [confidence 0.91]
├── drawer 2024-11-04T14:32 — "I hate bright screens"  [source]
├── drawer 2024-12-01T09:11 — "switch my IDE to dark" [source]
└── triple (user)-[prefers]->(dark-mode) via DerivationKind::AgentConsolidation
    evidence-weight 0.84, DST combined mass 0.91
```

MemPalace has no equivalent. Verbatim retrieval cannot reconstruct *why*.

---

### 2. Contradiction detection — `heka contradict`

Heka runs ACH (Analysis of Competing Hypotheses) across your memory graph. When two drawers
support mutually inconsistent claims, it surfaces the conflict instead of silently choosing one.

```bash
akh-medu ask "heka contradict: check my preferences for deployment targets"
```

Example output:

```
CONTRADICTION DETECTED

Hypothesis A: user prefers on-premise Kubernetes
  Supported by: drawer 2025-01-10, drawer 2025-02-03
  DST mass: 0.72

Hypothesis B: user prefers managed cloud (AWS EKS)
  Supported by: drawer 2025-03-15
  DST mass: 0.58

Inconsistency score: 0.89  [high]
Recommendation: ask user to resolve before next deployment decision.
```

MemPalace returns whichever drawer scored higher. It has no model of contradiction.

---

### 3. Akhipedia auto-extraction

Every night, Heka clusters your drawers by VSA similarity and topic, then promotes the densest
clusters into draft Akhipedia article blocks — permanent, citable knowledge, not just a retrieval
index.

```bash
akh-medu ask "what do I know about distributed tracing?"
```

The response draws from *live Akhipedia articles* synthesized from your past conversations, not
just the raw drawer text. Each article cites the source drawer IDs; the knowledge is yours and
traceable to the exact words that produced it.

```
Akhipedia: Distributed Tracing  [draft, 3 sources]

Claim: user uses OpenTelemetry with Jaeger for tracing Go services
  Sources: drawer 2024-09-11, drawer 2024-10-22, drawer 2025-01-07

Definition: trace context propagation — the mechanism by which a root span ID
  flows through service calls so all spans share a common trace ID.
  Sources: drawer 2024-10-22
```

---

## Reproducible benchmark

Run the full evaluation suite on your hardware:

```bash
# LongMemEval (single-session recall at R@5 and R@10)
heka eval longmemeval

# LoCoMo (cross-session conversational memory)
heka eval locomo

# Full table — runs both, prints comparison against published baselines
heka eval all
```

The harness lives at `src/eval/` and requires no external API keys in raw mode.
Hybrid mode (re-ranking with akh-medu's own inference engine) requires a running daemon.

Hardware spec for the reported numbers: AMD Ryzen 9 7950X, 64 GB DDR5, NVMe SSD.
No GPU. No external LLM calls in raw mode.

Full methodology and numbers: [docs/heka-vs-mempalace.md](../../docs/heka-vs-mempalace.md).

---

## MCP tools

| Tool | Description |
|------|-------------|
| `heka_remember` | Store a drawer (verbatim text + metadata). |
| `heka_recall` | Retrieve top-k drawers by VSA similarity. |
| `heka_explain` | Return provenance chain for a belief. |
| `heka_contradict` | Run ACH contradiction scan over a topic. |
| `heka_status` | Show drawer count, Akhipedia article count, index health. |

---

## License

GPLv3. For proprietary integrations contact the author.
