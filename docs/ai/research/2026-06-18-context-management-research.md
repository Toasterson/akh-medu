# Context Management — Research Notes (akh-medu)

Research harvest backing `rfd-context-management.md`. Reference material, organized by area, with concrete figures and sources. Status of every non-obvious claim is noted; where a number is unsourced or a project unverified, it says so. Current as of June 2026.

---

## 0. Organizing principle

What you can do is determined by **how much of the model you control**, which splits along the backend:

- **API boundary (Ollama, OpenRouter)** — you control the message array and a few coarse knobs. Everything is "what tokens do I assemble for the next call."
- **Full control (in-process Candle)** — you also get KV-cache-level control: attention sinks, eviction, quantized cache, paged attention. The only place to manage context *below* message granularity.

Underlying motivation: **context rot** — recall degrades as tokens grow (attention is an n² resource; models have a finite "attention budget"). Goal is always the smallest high-signal token set, not the largest window. Source: Anthropic, *Effective context engineering for AI agents* (https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents).

---

## 1. Message-level techniques (work on every backend)

### 1.1 Compaction
Summarize the conversation as it nears the window limit; reinitialize from the summary. Claude Code preserves architectural decisions, open bugs, and implementation details while dropping redundant tool outputs, continuing with the summary plus the few most-recently-accessed files. Tune recall first, then precision. Cheapest, safest sub-case: clearing stale tool results (a tool called N turns deep rarely needs its raw output again). Source: Anthropic context-engineering post.

### 1.2 Anthropic memory tool + context editing
Shipped with Sonnet 4.5 (beta header `context-management-2025-06-27`). The **memory tool** is file-based and **operates client-side** — the model emits tool calls, your app executes them, so you control storage (file/db/KG). **Context editing** auto-clears stale tool results as token limits approach. Reported numbers: context editing cut token use ~84% in a 100-turn web search and gave ~29% performance lift alone; ~39% combined with the memory tool. The *pattern* is portable to any backend (it's orchestration), even though the Anthropic feature only runs against Claude. Sources: https://www.anthropic.com/news/context-management ; https://docs.claude.com (memory tool).

### 1.3 Structured note-taking / agentic memory
The agent writes notes to durable storage and reads them back later (a `NOTES.md`, a DB, the KG). Persistent memory with minimal overhead; the MemGPT pattern of treating the window as RAM and paging facts out. Source: Anthropic context-engineering post.

### 1.4 Just-in-time retrieval vs. RAG vs. hybrid
The frontier shift: instead of embedding-stuffing everything up front, keep lightweight identifiers (paths, queries, IDs) and load on demand at runtime. Tradeoff: runtime exploration is slower than precomputed retrieval, and without good tool design the agent chases dead ends. Most production systems end up hybrid. This is the agentic-retrieval paradigm and maps to "let the tool do full context management." Source: Anthropic context-engineering post.

### 1.5 Sub-agent context isolation
Spin focused sub-agents with clean windows; each can burn tens of thousands of tokens exploring and return a 1,000–2,000-token distillate; the lead agent never sees the mess. Orchestration pattern, not a context manager. Source: Anthropic context-engineering post.

---

## 2. Memory frameworks (architectures to borrow, not adopt)

### 2.1 MemGPT / Letta
LLM-as-OS: window = RAM, external store = disk. Tiers: **core memory** (bounded, pinned in the system prompt — persona + critical state), **recall** (recent history), **archival** (cold store via search). The agent self-edits via tools (`core_memory_append`/`replace`, `archival_insert`/`search`). Reliability vulnerability: memory retention depends on the model triggering tool calls — if it hallucinates or drops the instruction, info is lost; every memory op also spends inference tokens. MemGPT = the design pattern; Letta = the framework/company. Sources: https://docs.letta.com/guides/legacy/memgpt_agents_legacy ; https://www.letta.com/blog/memgpt-and-letta .

### 2.2 Mem0
Passive background extraction: a secondary pipeline reads the transcript, extracts/dedupes facts, updates a store — consistent and token-efficient, but can't make nuanced judgments about what matters. Contrast with Letta's adaptive-but-model-dependent self-editing. Source: https://vectorize.io/articles/mem0-vs-letta .

### 2.3 Zep / Graphiti (most relevant — we have a KG)
Memory as a **bi-temporal knowledge graph**: every edge carries validity intervals (`t_valid`, `t_invalid`) tracking when a fact was true and when ingested; on change, old facts are **invalidated, not deleted** → query "what's true now" or "what was true at time T". Three subgraphs — episode / semantic-entity / community — mirroring episodic vs. semantic memory. Beats MemGPT on the Deep Memory Retrieval benchmark (94.8% vs. 93.4%). Graphiti is the OSS engine (Python, Neo4j). We borrow the *design* (bitemporal edges against our existing KG, VSA as ranker), not the runtime. Sources: arXiv 2501.13956 (https://arxiv.org/abs/2501.13956) ; https://github.com/getzep/graphiti .

### 2.4 Memory scopes (standardized vocabulary)
Three scopes are now standard: **episodic** (specific past interactions), **semantic** (facts/preferences), **procedural** (learned behaviors/rules). Source: https://atlan.com/know/best-ai-agent-memory-frameworks-2026/ .

---

## 3. KV-cache techniques (Candle regime)

Framing caveat: most were built to fit *long inputs into limited VRAM*, not to grant unbounded *memory*.

### 3.1 Attention sinks / StreamingLLM
Keep the first ~4 tokens as "attention sinks" (they absorb excess attention) plus a sliding window of recent tokens; stabilises perplexity across millions of tokens with no fine-tuning, where naive window attention collapses once the initial tokens are evicted. **Critical caveat: infinite *generation*, not infinite *context*** — out-of-window data is gone; the model just doesn't degrade. Sources: arXiv 2309.17453 (https://arxiv.org/abs/2309.17453) ; https://github.com/mit-han-lab/streaming-llm ; caveat per https://mbrenndoerfer.com/writing/attention-sinks-streamingllm-infinite-generation .

### 3.2 Content-aware eviction (H2O, SnapKV, PyramidKV)
Score tokens by attention, keep the heavy hitters instead of just recent ones.
- **H2O** — evicts on cumulative attention scores at decode time; balances "heavy hitter" tokens + a recency window. Reported up to ~29× throughput and ~1.9× latency reduction at aggressive eviction.
- **SnapKV** — uses a local observation window at the end of the prompt plus pooling/clustering to pick important positions per head; prefill-focused.
- **PyramidKV** — varies budget per layer (more low, less high) because attention sparsity rises with depth.
All training-free and hardware-compatible, but **discard tokens permanently**, and aggressive eviction can disrupt chain-of-thought reasoning steps. Sources: https://arxiv.org/html/2406.02069 (PyramidKV, with H2O/SnapKV baselines) ; CoT-disruption per arXiv 2505.24133 (R-KV).

### 3.3 KV quantization (KIVI, TurboQuant)
Shrink cache *precision* rather than dropping tokens — keep all context, cheaper. KIVI ~2.6× combined memory reduction at 2-bit; **TurboQuant** ~6× at 3-bit, no calibration, near the information-theoretic limit, via a Walsh-Hadamard rotation that redistributes variance so simple scalar quantisers work without destroying outliers. Preferred first lever over eviction when losing history is worse than spending VRAM. Source: https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-... (KV-cache compression survey).

### 3.4 Paged attention + prefix caching / RadixAttention / HiCache
- **PagedAttention** — non-contiguous fixed-size KV pages; kills fragmentation from unknown final sequence length.
- **Prefix caching** — matched prefixes reuse pre-computed KV; only the non-matching suffix is processed. A stable system+KG preamble gets computed once.
- **RadixAttention (SGLang)** — global radix (prefix) tree; longest-prefix matching, zero-cost forking at divergence, big memory economy for agent swarms sharing a prompt.
- **HiCache (SGLang)** — three-tier hierarchical KV cache: evicted radix nodes demote GPU VRAM → CPU host memory → disk (mirrors our DashMap→memmap2→redb tiers).
RadixAttention/HiCache are **Python (SGLang)**, not available in Candle. Sources: SGLang docs; mistral.rs prefix-reuse per https://github.com/EricLBuehler/mistral.rs .

### 3.5 LLMLingua (prompt compression)
Token-layer, perplexity-based coarse-to-fine compression *before* sending to the LLM: a small LM acts as a budget controller, dropping high-predictability (low-information) tokens; query-aware variant (LongLLMLingua) preserves query-relevant entities/numbers. Up to ~20× compression with reasoning retained. **Python**, no Rust port — reimplement the budget-controller idea if prefill cost on KG dumps bites. (Microsoft LLMLingua.)

---

## 4. Rust ecosystem (what's actually slottable in-process)

| Crate / project | What it gives you | Maturity / notes | Source |
|---|---|---|---|
| **candle** (huggingface) | Core ML framework; basic growing KV cache; flash attention; llama.cpp quant types | Backends: **CPU, CUDA, Metal only — no upstream ROCm** | https://github.com/huggingface/candle |
| **candle-vllm** (EricLBuehler) | PagedAttention; **TurboQuant KV** (turbo8/4/3 ≈ 2.6×/3.7×/4.7×) | CUDA (SM70+) and Metal | https://github.com/EricLBuehler/candle-vllm |
| **mistral.rs** (EricLBuehler) | PagedAttention; **FP8 (E4M3) KV quant** (~halves KV); prefix caching w/ block reuse | Most mature Rust KV stack; CUDA/Metal-first | https://github.com/EricLBuehler/mistral.rs |
| **tq-kv** | Pure-Rust **TurboQuant** (Walsh-Hadamard); C FFI; constant-memory KV via TriAttention (vendor claim: up to 1,560× @128K) | Independent of candle; tracks ICLR 2026 paper; **bench the headline claim** | https://github.com/onur-gokyildiz-bhi/tq-kv |
| **kalosm** / **Floneum** | Candle-based; **structured generation** via `#[derive(Parse)]` + `with_constraints` (logit masking, structure-aware accel) | Best pickup for valid JSON/SPARQL from `kg_mutate`; CUDA/Metal accel, but constrained-decoding is CPU-side | https://github.com/floneum/floneum ; https://docs.rs/kalosm |
| **Fusor** (Floneum) | WGPU runtime — runs on any accelerator (incl. AMD via Vulkan) or browser | Vendor-neutral AMD path w/o ROCm; newer, generally slower than native HIP | (Floneum repo, Fusor sub-project) |
| **mem0-rust** | Rust impl of mem0 (embeddings/vector stores/llms/history) | Exists but tiny: ~41 downloads, 1 version — experiment, not infra | https://crates.io/crates/mem0-rust |
| **rustmem / R-Mem** | SQLite-backed (FTS5 + vector + archive) mem0 study; 3.6 MB binary, 2,826 LOC, <10 MB RAM | Authors call it "a learning exercise… not a replacement" | https://lib.rs/crates/rustmem |
| **rocm-rs** | Safe Rust wrappers for the ROCm/HIP stack | The binding crate the candle ROCm PR uses | https://github.com/radudiaconu0/rocm-rs |
| **PRISM** (Artaeon) | Neural-free VSA cognitive architecture (analogy, causal, multi-hop, contradiction) | Niche solo project; real, matches "Pattern Reasoning via Interpretable Symbolic Memory" | https://github.com/Artaeon/prism |

Related research: **Hyperdimensional Probe** — decoding LLM residual-stream representations into interpretable VSA concepts (real, recent). Source: https://arxiv.org/html/2509.25045 . Relevant to projecting Gemma hidden states into the VSA core instead of external embeddings.

---

## 5. Backend operational specifics (verified)

### 5.1 Ollama
- Endpoints `/api/generate`, `/api/chat` are **stateless** — rebuild the full message payload every turn.
- **Default context window is 4096 tokens** regardless of model capacity → long KG extracts are **silently truncated**. Set `num_ctx` per-request or `OLLAMA_CONTEXT_LENGTH` globally.
- `keep_alive: -1` pins the model in VRAM indefinitely (avoids reload latency in tight OODA loops).
- llama.cpp context-shifting ≈ StreamingLLM for free. **Mature ROCm** (llama.cpp/HIP) on the 7900 XTX.

### 5.2 OpenRouter (future)
- **`session_id`** → used directly as the sticky-routing key; stickiness activates on the first successful request (vs. only after a cache hit). Without it, OpenRouter hashes the first system + first non-system message — which breaks for agentic workflows whose opening message shifts. Source: https://openrouter.ai/docs/guides/best-practices/prompt-caching .
- **Auto Router** (powered by NotDiamond) and **Pareto Router** pin both model and provider when sticky. Source: https://openrouter.ai/docs/guides/routing/routers/auto-router .
- **Response caching** — exact request match (same key, model, endpoint, streaming mode, body) returns the cached response with **all usage counters zeroed (no billing)**; `X-OpenRouter-Cache-Clear: true` to bypass. Good for deterministic retries. Source: https://openrouter.ai/docs/guides/features/response-caching .
- **Anthropic prompt caching**: `cache_control: {type: "ephemeral"}`, max 4 breakpoints; cache reads ≈ 0.1× input; `"ttl": "1h"` extends the default 5-min cache. Put breakpoints on big static blocks (system prompt, tool schemas, KG preamble).

### 5.3 Candle (preferred local)
In-process, GIL-free. ROCm via our fork (§6). kalosm for constrained decoding. KV levers via candle-vllm/mistral.rs/tq-kv. Pre-compute static-preamble KV once and reuse.

---

## 6. ROCm / AMD hardware reality

- **Candle upstream ships only CPU, CUDA, Metal — no ROCm.** Source: https://github.com/huggingface/candle/discussions/3327 ; issue #2938.
- **We have a working Candle ROCm fork on the gfx1100 (7900 XTX) machine** — so the Candle GPU forward pass, and the `KvCachePolicy` arm, is available today, **not gated**.
- **Upstream WIP PR #3424** — `rocm-rs` bindings, AOT-compiled (hipcc lacks JIT; `hiprtc` bindings absent in rocm-rs), developed on a gfx1101 RX 7700 XT (same RDNA3 gen), "BERT runs" but much unimplemented/unsafe. Worth comparing with a view to upstreaming/publishing our fork. Source: https://github.com/huggingface/candle/pull/3424 .
- **Floneum Fusor (WGPU/Vulkan)** — vendor-neutral AMD fallback with no ROCm at all; slower than native HIP.
- The advanced KV kernels in candle-vllm/mistral.rs/tq-kv are CUDA/Metal-first — verify each on our fork before relying.
- **kalosm constrained decoding is logit-side (CPU)** — works regardless of accelerator.
- The external survey's **UMA / zero-copy** premise is Apple-Silicon-specific; inapplicable to a discrete dGPU over PCIe.

**Action noted:** decide whether to upstream the ROCm fork, publish it standalone, or both (compare against PR #3424).

---

## 7. External survey ("Gemini report") — assessment

Better-sourced than expected; most suspicious-looking specifics verified.

**Verified accurate:** OpenRouter `session_id` sticky routing + first-system/first-user fingerprint hashing; response caching with zeroed billing; Auto Router (NotDiamond) + Pareto Router; Anthropic `cache_control: ephemeral` (max 4) and cache-read pricing multipliers; Ollama `num_ctx`/4096 default/`keep_alive: -1`; `mem0-rust` and `rustmem`/R-Mem exist (rustmem's 3.6 MB / 2,826 LOC / <10 MB are its authors' figures); PRISM (VSA architecture); hyperdimensional probing into VSA; kalosm `derive(Parse)` + structure-aware accel; H2O (~29× / 1.9×) and LLMLingua (~20×) match their papers.

**Caveats to apply:**
- **Maturity oversold** — `mem0-rust` (~41 downloads, 1 version), `rustmem` (self-described study/learning project), PRISM (niche solo) are experiments, not production infra. The comparison tables read as a buyer's guide.
- **Python dressed as Rust-native** — RadixAttention/HiCache (SGLang), LLMLingua, Letta/Mem0 cores are Python; their "Rust bindings" are HTTP clients, not in-process. candle-vllm has PagedAttention, **not** RadixAttention.
- **ROCm omission** — assumes CUDA/Metal throughout; see §6.
- **UMA mismatch** — Apple-Silicon framing inapplicable to the 7900 XTX.
- **Unsourced numbers to verify before quoting:** `mem0-rust` "3.5× faster, ~10 MB RSS"; `rustmem` "19× retrieval."
- **Unverified specifics (treat as unconfirmed, not false):** `candle-pipelines-models` as a named crate; the 26 GB / 167 s (156 MB/s) KV-serialization benchmark; the `agentic-memory` / `.amem` entry.

---

## 8. Open threads / to verify

1. Gemma tool-proposal vs. VSA scoring in Decide: override or augment?
2. Passive consolidation always-on vs. Virtual-only safety net (cost vs. coverage).
3. KV-cache-to-disk serialization on our fork (skip re-prefill of system+KG preamble) — feasibility.
4. Which `HeavyHitter` eviction variant survives long CoT without dropping reasoning steps.
5. Confirm the unsourced/unverified items in §7 if any are load-bearing for a decision.
6. Bench tq-kv's TriAttention compression claim on our hardware before relying.

---

## 9. Consolidated references

- Anthropic — Effective context engineering: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- Anthropic — Managing context (memory tool + context editing): https://www.anthropic.com/news/context-management
- Letta / MemGPT: https://docs.letta.com/guides/legacy/memgpt_agents_legacy · https://www.letta.com/blog/memgpt-and-letta
- Mem0 vs Letta: https://vectorize.io/articles/mem0-vs-letta
- Agent memory frameworks (2026 survey): https://atlan.com/know/best-ai-agent-memory-frameworks-2026/
- Zep / Graphiti: https://arxiv.org/abs/2501.13956 · https://github.com/getzep/graphiti
- StreamingLLM: https://arxiv.org/abs/2309.17453 · https://github.com/mit-han-lab/streaming-llm · caveat: https://mbrenndoerfer.com/writing/attention-sinks-streamingllm-infinite-generation
- PyramidKV (+ H2O/SnapKV baselines): https://arxiv.org/html/2406.02069 · R-KV (CoT caveat): https://arxiv.org/pdf/2505.24133
- KV-cache compression survey (KIVI/TurboQuant): https://www.marktechpost.com/2026/04/29/top-10-kv-cache-compression-techniques-for-llm-inference-reducing-memory-overhead-across-eviction-quantization-and-low-rank-methods/
- Candle: https://github.com/huggingface/candle · candle-vllm: https://github.com/EricLBuehler/candle-vllm · mistral.rs PagedAttention: https://github.com/EricLBuehler/mistral.rs/blob/master/docs/PAGED_ATTENTION.md · tq-kv: https://github.com/onur-gokyildiz-bhi/tq-kv
- kalosm / Floneum: https://github.com/floneum/floneum · https://docs.rs/kalosm
- mem0-rust: https://crates.io/crates/mem0-rust · rustmem: https://lib.rs/crates/rustmem · PRISM: https://github.com/Artaeon/prism · Hyperdimensional Probe: https://arxiv.org/html/2509.25045
- OpenRouter — prompt caching: https://openrouter.ai/docs/guides/best-practices/prompt-caching · response caching: https://openrouter.ai/docs/guides/features/response-caching · Auto Router: https://openrouter.ai/docs/guides/routing/routers/auto-router
- candle ROCm — discussion #3327: https://github.com/huggingface/candle/discussions/3327 · WIP PR #3424: https://github.com/huggingface/candle/pull/3424 · rocm-rs: https://github.com/radudiaconu0/rocm-rs
