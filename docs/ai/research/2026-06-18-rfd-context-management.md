# RFD: Context Management for the akh-medu Agent Harness

| | |
|---|---|
| **State** | prediscussion |
| **Authors** | Toasterson |
| **Supersedes** | — |
| **Related** | ACP session-lifecycle / memory RFD · `notes-context-management-research.md` (research backing) |

## 1. Summary

The OODA loop needs a context-management layer that bridges a stateless transformer to a stateful, long-horizon agent without corrupting the symbolic core. This RFD proposes three orthogonal policy traits — `ContextManager` (message-level curation), `KvCachePolicy` (token-level KV retention, Candle-only), and a deferred `RetrievalPolicy` (how context is pulled in) — plus three interchangeable `ContextManager` implementations spanning *who curates* (system / background pipeline / model). The symbolic safety invariants are made strategy-independent via a decorator. Full research backing, sources, and the corrected external survey live in the companion notes document.

## 2. Background

### 2.1 The tension

A transformer is a stateless inference engine; the agent requires evolving, structured memory. Accumulating every turn into the message array fails twice over: attention cost is quadratic in tokens, and recall degrades as the window grows ("context rot"). The job is the *smallest high-signal set of tokens* per turn, not the largest window — true on every backend regardless of nominal window size.

### 2.2 Three orthogonal axes

Most surveys collapse independent concerns. We separate them into three traits so each can be experimented with in isolation:

- **Window curation** (`ContextManager`) — *what messages/notes/memory* enter the window. Deterministic compaction vs. background extraction vs. model self-editing.
- **KV retention** (`KvCachePolicy`) — *which tokens' KV* survive during generation. Attention sinks vs. heavy-hitter eviction. Operates below the message layer; Candle-only.
- **Retrieval** (`RetrievalPolicy`, deferred) — *what gets pulled in* from the KG. Just-in-time vs. pre-loaded vs. temporal-graph search.

A given curation policy runs on any backend; a KV policy only where the backend exposes cache control. Keeping these orthogonal is what lets us A/B managers against the same traces while varying backend and KV policy underneath.

## 3. Design

### 3.1 `ContextManager` (message-level curation)

```rust
/// Strategy for curating the token budget of a single reasoning episode.
///
/// Implementations differ in *policy* — how they decide what survives into the
/// next inference call — not in which backend they target. Keeping policy
/// orthogonal to `Backend` lets any manager run against any substrate and use
/// KV-level levers only where the backend advertises them. Managers behind this
/// trait are swapped via a trait object to A/B them on identical traces.
#[async_trait]
pub trait ContextManager: Send + Sync {
    /// Strategy identity, for telemetry and experiment attribution.
    fn strategy(&self) -> StrategyId;

    /// Assemble the message window plus any backend directives for the next
    /// turn, staying within the budget the backend will enforce. The manager
    /// owns whatever compaction its policy dictates; there is deliberately no
    /// separate "on overflow" hook.
    async fn assemble(
        &mut self,
        episode: &EpisodeState,
        budget: TokenBudget,
    ) -> Result<AssembledContext>;

    /// Persist whatever the policy considers salient after a turn — notes,
    /// archival writes, KG edges. Runs off the critical path where the policy
    /// permits it.
    async fn commit(&mut self, outcome: &TurnOutcome) -> Result<()>;
}
```

`AssembledContext` carries the messages, an optional decode constraint (a kalosm parser), and a reference to the active `KvCachePolicy` directive. A backend ignores anything outside its `Capabilities`.

### 3.2 `Backend` (substrate) and the config/policy split

```rust
/// The inference substrate. Surfaces *which* context levers exist so policies
/// can degrade gracefully: message reassembly is always available; KV-level
/// control exists only for in-process Candle.
#[async_trait]
pub trait Backend: Send + Sync {
    fn capabilities(&self) -> Capabilities;
    async fn infer(&self, ctx: &AssembledContext) -> Result<InferenceOutput>;
}

bitflags! {
    pub struct Capabilities: u32 {
        const MESSAGE_LEVEL      = 1 << 0; // reassemble messages (every backend)
        const PROMPT_CACHE       = 1 << 1; // prefix/provider KV reuse across turns
        const KV_QUANT           = 1 << 2; // quantized KV cache (config, not policy)
        const KV_EVICTION        = 1 << 3; // sink/heavy-hitter eviction (KvCachePolicy)
        const PAGED_ATTENTION    = 1 << 4; // paged KV allocation (config)
        const CONSTRAINED_DECODE = 1 << 5; // logit-level grammar (kalosm)
    }
}
```

**Config vs. policy.** Not every KV technique is a policy worth A/B-ing. KV *quantization* (TurboQuant level), *paged attention*, and *prefix-cache reuse* are precision/layout settings — you turn them on; they don't change *what* context is present. They are backend config. Only the *lossy selection* techniques (which decide what survives) are policies — those are `KvCachePolicy` (§3.6).

Backend → capability mapping (ROCm caveat in §7):

| Backend | Capabilities |
|---|---|
| `OllamaBackend` | `MESSAGE_LEVEL` `\|` `PROMPT_CACHE` |
| `OpenRouterBackend` | `MESSAGE_LEVEL` `\|` `PROMPT_CACHE` |
| `CandleBackend` | `MESSAGE_LEVEL` `\|` `KV_QUANT` `\|` `KV_EVICTION` `\|` `PAGED_ATTENTION` `\|` `CONSTRAINED_DECODE` |

### 3.3 `KvCachePolicy` (token-level retention, Candle-only)

```rust
/// Policy over which tokens' KV state survives during generation. Operates
/// *below* the message abstraction — inside the forward pass, on the live
/// cache, mid-decode — which is why it is NOT a `ContextManager`: there is no
/// `assemble() -> messages` framing for it. Surfaced via `KV_EVICTION`. Two
/// implementations let us A/B token-level retention the same way managers A/B
/// message-level curation.
pub trait KvCachePolicy: Send + Sync {
    fn policy(&self) -> KvPolicyId;

    /// Decide which cache entries to retain when the live cache reaches budget,
    /// given per-token attention statistics gathered during decode.
    fn select(&self, cache: &KvCacheView, budget: KvBudget) -> Retained;
}
```

Implementations (§3.6): `StreamingSink`, `HeavyHitter`.

### 3.4 Safety enforcement (non-negotiable)

The neural policy is **untrusted**. An LLM self-editing its own memory must never drop a safety-relevant span. This is a type-enforced, policy-independent invariant.

```rust
pub enum Retention {
    /// May be summarized, compressed, or evicted under budget pressure.
    Volatile,
    /// Eligible for eviction but reconstructable from the KG on demand.
    Recoverable,
    /// Must remain in-context verbatim: AgentDecision provenance, danger-
    /// metadata gates, psyche-veto inputs. assemble() must reject any window
    /// that omits it.
    PinnedSafety,
}

/// Wraps any `ContextManager` and turns the symbolic invariant into a hard,
/// policy-independent guarantee. Whatever the inner policy decides — including
/// an LLM curating its own context — the assembled window is checked against
/// the pinned set before it can reach the backend. The trusted symbolic
/// boundary around the untrusted neural policy.
pub struct SafetyEnforced<M: ContextManager> { inner: M }

#[async_trait]
impl<M: ContextManager> ContextManager for SafetyEnforced<M> {
    fn strategy(&self) -> StrategyId { self.inner.strategy() }

    async fn assemble(&mut self, episode: &EpisodeState, budget: TokenBudget)
        -> Result<AssembledContext>
    {
        let pinned = episode.spans_with(Retention::PinnedSafety);
        let ctx = self.inner.assemble(episode, budget).await?;
        ctx.assert_contains_all(&pinned)
            .map_err(ContextError::SafetyInvariantViolated)?;
        Ok(ctx)
    }

    async fn commit(&mut self, outcome: &TurnOutcome) -> Result<()> {
        self.inner.commit(outcome).await
    }
}
```

Every manager is constructed inside `SafetyEnforced<_>` before the loop sees it. A violation is a hard error; the *rate of caught attempts* is itself an experiment signal (how often a policy tried to evict something it must not). The `KvCachePolicy` path applies the same principle: pinned spans are never candidates for KV eviction.

### 3.5 The three general managers

Spanning *who does the curating* — a principled three-way with three distinct failure modes:

- **`CompactingManager` — the system curates.** [control] Deterministic. Tool-result clearing as the cheapest lever (drop raw payloads N turns deep, keep the fact). Threshold compaction: summarize older turns into a high-fidelity digest preserving goals/decisions/open problems, reinitialize with digest + recent turns. Tune recall first, precision second. Just-in-time KG retrieval, VSA-ranked.
- **`PassiveManager` (Mem0-style) — a background pipeline curates.** [cheapest in tokens, most reliable] The main loop runs a plain rolling buffer and never self-edits. Asynchronously, a small model or deterministic extractor pulls salient entities/relations from the transcript, deduplicates, and writes them to the KG as bitemporal edges. Memory is preserved without spending the main loop's inference budget — the key reliability advantage over self-editing.
- **`VirtualContextManager` (MemGPT/Letta) — the model curates.** [experiment] The agent manages its own context via tools, treating the window as RAM and the KG as disk. Tiers: core memory (pinned system-prompt block), recall (recent history), archival (KG-backed). Self-editing tools (`core_memory_append`/`replace`, `archival_insert`/`search`). Known failure: if the model fails to trigger a write, the fact is lost — mitigated by running `PassiveManager`'s consolidation in parallel as a safety net, so self-editing is evaluated on quality/efficiency while nothing salient is permanently dropped.

Note-taking (a `NOTES.md` scratchpad) is **not** a separate manager — it is a degenerate `VirtualContextManager` config with a single append tool.

### 3.6 `KvCachePolicy` implementations

- **`StreamingSink { sinks, window }`** — keep the first `sinks` tokens (attention sinks) plus the last `window` tokens. Stabilises perplexity over long streams with no fine-tune. Caveat: enables infinite *generation*, not infinite *context* — out-of-window data is gone, the model simply doesn't collapse. Cheap, safe, no quality-selection.
- **`HeavyHitter { budget, recency }`** — keep high cumulative-attention "heavy hitter" tokens plus a recency window (H2O/SnapKV family). Retains semantically important tokens, not just recent ones. Caveat: discards tokens permanently and aggressive eviction can break long chains-of-thought — gate carefully for the Decide phase.

### 3.7 Deferred — `RetrievalPolicy` (recorded, not built)

Orthogonal fourth axis, written down for when we want to A/B retrieval independently of curation and KV eviction. **Not in scope for the first cut** — the current plan hard-codes just-in-time KG retrieval inside the managers. Recorded so it is captured outside any chat:

```rust
/// DEFERRED. How context is *pulled in*, independent of window curation
/// (`ContextManager`) or KV retention (`KvCachePolicy`). Build only if we want
/// to swap retrieval strategies behind one interface.
#[async_trait]
pub trait RetrievalPolicy: Send + Sync {
    async fn fetch(&self, query: &RetrievalQuery, kg: &KnowledgeGraph)
        -> Result<Vec<Fact>>;
}

// Candidate implementations:
//   JustInTime  — agent pulls on demand via kg_query, VSA-ranked.
//                 (Matches the current plan; the default if this is never built.)
//   PreloadedRag — embedding-based pre-inference retrieval, front-loaded for
//                 latency at the cost of context bloat and stale indexing.
//   TemporalKg  — Graphiti-style bitemporal subgraph search with validity
//                 windows; "what was true at time T", auditable.
```

The motivation to build it later: if measurement shows retrieval quality, not window curation, is the bottleneck, this lets us isolate and compare retrieval strategies without touching the managers.

### 3.8 Memory substrate — the KG is the store, not a parallel DB

Do **not** bolt on a separate vector database. akh-medu already has the substrate:

- Agentic-memory writes become **bitemporal KG edges** (validity intervals; supersede-don't-delete), per the Graphiti/Zep model — auditable "what did the agent believe, and when" for free.
- Retrieval ranking is **VSA scoring** over the existing 10,000-dim space, not an opaque embedding lookup. "What to pull back" stays a KG-grounded, inspectable decision.
- Storage tiers map onto the existing hot/warm/cold path (DashMap → memmap2 → redb).

All three managers share this substrate; only their write/read policy differs.

## 4. Backend operational notes (verified)

**Ollama.** Stateless; reconstruct the full payload every turn. Default window is 4096 tokens regardless of model capacity, so long KG extracts are **silently truncated** unless `num_ctx` is set per-request (or `OLLAMA_CONTEXT_LENGTH` globally). Pin with `keep_alive: -1`. llama.cpp context-shifting gives a StreamingLLM-like effect for free here. Mature ROCm path.

**OpenRouter (future).** Pass an explicit `session_id` for sticky routing — pins the provider (and, under Auto/Pareto routers, the model) on the first successful request, instead of waiting for a cache hit; without it OpenRouter hashes the first system + first user message, which breaks for agentic workflows whose opening message shifts. For Anthropic models, inject `cache_control: {type: "ephemeral"}` breakpoints (max 4) on large static blocks; cache reads bill at roughly a tenth of input. Use response caching (zeroed billing on an exact request match) for deterministic retries.

**Candle (preferred local).** In-process, GIL-free. kalosm for `CONSTRAINED_DECODE`. KV levers via candle-vllm / mistral.rs / tq-kv. Pre-compute the static system+KG preamble's KV once and reuse it across turns.

## 5. KV-level technique catalogue (Candle regime)

Framing rule the external survey got wrong: most of these fit *long inputs into limited VRAM*, not unbounded *memory*. Full evidence and numbers in the notes doc.

| Technique | Role | Rust-native today? | Note |
|---|---|---|---|
| KV quantization (KIVI / TurboQuant) | **config** | **Yes** — `tq-kv`, candle-vllm TurboQuant (turbo3/4/8 ≈ 4.7×/3.7×/2.6×) | Preferred first lever: lossy on precision, not positions; keeps all tokens |
| Paged attention | **config** | **Yes** — candle-vllm, mistral.rs (FP8 KV quant + prefix reuse) | Kills fragmentation; most mature Rust KV stack |
| Attention sinks / StreamingLLM | **`KvCachePolicy`** | Partial (llama.cpp context-shift); manual in Candle | Infinite generation ≠ infinite context |
| Heavy-hitter eviction (H2O / SnapKV / PyramidKV) | **`KvCachePolicy`** | Reimplement | Permanent token loss; can break long CoT |
| RadixAttention / HiCache | substrate (not a policy) | **No** — SGLang (Python) | Prefix-tree KV sharing + GPU→CPU→disk; mirrors our tiered storage; run as a service or design toward it |
| Prompt compression (LLMLingua) | transform (not a policy) | **No** — Python | Reimplement the perplexity budget-controller if prefill cost on KG dumps bites |
| Constrained decoding (kalosm) | backend feature | **Yes** — candle-based | Best single pickup for `kg_mutate`/SPARQL: valid output, no self-correction loop. CPU-side masking, **not** ROCm-blocked |

## 6. The experiment

Run `SafetyEnforced<{Compacting, Passive, Virtual}>` behind one trait object, same backend, same traces; on Candle, cross with `{StreamingSink, HeavyHitter}` KV policies. Measure:

1. Task completion / quality on a fixed harness suite.
2. Token cost per completed task (self-editing's overhead shows here).
3. Retrieval precision/recall of facts pulled back into context.
4. Context-rot incidents (contradictory/regressed outputs over long episodes).
5. Safety-pin violation *attempts* caught by the decorator (must reach the backend as zero; the attempt count rates each policy's trustworthiness).
6. Lost-salient-fact rate for the Virtual arm with vs. without the passive safety net.
7. (Candle) decode latency and peak VRAM per KV policy / quant level.

## 7. Hardware reality: ROCm / 7900 XTX

Deployment GPU is gfx1100 (RDNA3, 7900 XTX). **We have a working Candle ROCm fork on this machine**, so the Candle GPU forward pass — and therefore the `KvCachePolicy` arm — is available on our hardware today. It is **not gated**. Two things to track:

1. **Upstream WIP ROCm PR (#3424)** — uses the `rocm-rs` safe bindings, AOT-compiled (hipcc lacks the JIT path), developed on a gfx1101 RX 7700 XT (same RDNA3 gen). Worth comparing against our fork with a view to upstreaming/publishing ours rather than carrying a private divergence. (Candle upstream still ships only CPU/CUDA/Metal, so the fork is what unblocks this.)
2. **Floneum Fusor (WGPU/Vulkan)** — vendor-neutral fallback that runs on RDNA3 with no ROCm at all, if the HIP-kernel surface our KV policies need proves incomplete in the fork. Generally slower than native HIP; prototype before relying.

The general managers need none of this — they run on Ollama (mature llama.cpp ROCm) as the no-fork baseline. kalosm's constrained decoding is logit-side and works regardless of backend.

The UMA / zero-copy framing in the external survey is Apple-Silicon-specific and does not apply to a discrete dGPU over PCIe; ignore it for this target.

## 8. Alternatives considered

- **Single monolithic manager** — rejected; precludes A/B, bakes one policy in.
- **Coupling policy to backend** (one manager per backend) — rejected; conflates the axes, forces reimplementing policy per substrate.
- **Modelling KV eviction as a `ContextManager`** — rejected; it operates inside the forward pass, not at `assemble() -> messages`. Hence the separate `KvCachePolicy` trait.
- **Adopt Letta/Mem0/Zep wholesale** — rejected as a dependency; Python services, and our KG+VSA is a better in-process substrate. We borrow designs (tiered memory, self-editing tools, bitemporal edges), not runtimes.
- **External vector DB** — rejected; redundant with the KG, breaks the single-source-of-truth / auditability story.

## 9. Open questions

1. Does a Gemma tool-proposal **override or augment** VSA candidate scoring in the Decide phase? (Affects how the Virtual arm's tool calls interact with the symbolic core.)
2. Run passive consolidation **always**, or only as the Virtual safety net? (Cost vs. coverage.)
3. KV-cache-to-disk serialization (skip re-prefilling the system+KG preamble across sessions) on our Candle fork — worth building? Concept is sound; verify feasibility on the fork.
4. Which `HeavyHitter` variant survives long Decide-phase chains-of-thought without dropping reasoning steps?
5. **Publish/upstream the ROCm fork** — compare against PR #3424; decide whether to upstream, publish standalone, or both. (Tracked separately as it has value beyond this RFD.)

## 10. Corrections to source research

Condensed; full version with numbers and sources in the notes doc. The external ("Gemini") survey was better-sourced than expected — most suspicious-looking specifics verified (OpenRouter `session_id`/response-caching/Auto+Pareto routers; Anthropic `cache_control`; Ollama flags; `mem0-rust`/`rustmem` exist; PRISM; hyperdimensional probing; H2O/LLMLingua numbers match papers). Apply these caveats: **maturity oversold** (the Rust memory crates are early/study projects); **Python dressed as Rust-native** (RadixAttention/HiCache/LLMLingua/Letta cores are Python; candle-vllm has PagedAttention, not RadixAttention); **UMA mismatch** (Apple-Silicon framing inapplicable here); **a few unsourced numbers and unverified crate names** to confirm before quoting.

## References

See `notes-context-management-research.md` for the full, annotated reference list. Key sources: Anthropic context-engineering and context-management posts; Letta/MemGPT; Zep/Graphiti (arXiv 2501.13956); StreamingLLM (arXiv 2309.17453); PyramidKV (arXiv 2406.02069); Candle + candle-vllm + mistral.rs + tq-kv; kalosm/Floneum; OpenRouter prompt/response caching docs; candle ROCm discussion #3327 and PR #3424; rocm-rs.
