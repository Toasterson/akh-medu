# Project Grug-Brain — From-Scratch Toddler-Architecture TLM — akh-medu

- **Date:** 2026-07-05
- **Status:** Imported research (external). Substance of a multi-turn Gemini build-conversation, reorganized; the final "Project Report" is preserved close to verbatim in §7. Claims are Gemini's — treat hardware/figures as starting estimates, not verified. Companion to [Tiny LLMs, Huge Context](2026-07-05-tiny-llm-huge-context.md), the [High Grug reasoning-language doc](2026-07-04-grug-reasoning-language.md), and the [fable-traces / BabyLLM note](../2026-07-05-fable-traces-babyllm.md).
- **Source:** Gemini share thread "Tiny LLMs: Reasoning Over Memorization" — https://gemini.google.com/share/d/17l44bRpHiRIs4k5tt9WOt8WvO6Ys4KrB (commissioned by Till).
- **Scope:** Can we pre-train a tiny LLM *from scratch* on a single consumer GPU whose weights hold **reasoning + syntax + tool-use** but almost no facts, feeding all knowledge at runtime from the KG + memory — reasoning internally in **High Grug**?
- **Relevance to akh-medu:** This is the concrete BabyLLM vision. It lands on Phase 27 (Live KG→LLM training, Burn/LoRA), Phase 35 (T5/offline training infra), Phase 26 (LLM-VSA), Phase 44 (context assembly), the High Grug symbol-language series, and the KG the engine already owns.

---

## 0. The core bet: reasoning over memorization

The industry default memorized the internet across thousands of GPUs. The opposite bet — a booming open-source subculture — is a **tiny model on one consumer chip that knows very few facts but has excellent logic, reasoning, and behavior**. Strip out "capital of every country / recipe for every dish" and the model shrinks drastically.

**Core tenets (toddler philosophy):**
- **Reasoning over memorization** — learn *how to think*, not *what to know*.
- **High Grug internal monologue** — hidden chain-of-thought in a hyper-literal, brutally simple step-by-step register (our designed dense reasoning language; see the [High Grug doc](2026-07-04-grug-reasoning-language.md)).
- **Tool-native** — API triggering and JSON generation as base "physical" actions, like a toddler reaching for a block.
- **Runtime knowledge injection** — textbooks, docs, blogs read into context at inference; no retrain when facts change.

## 1. Hardware: training from scratch on a single chip

Developers today pre-train small LLMs (**10M–1.5B params**) from scratch on a single consumer GPU (RTX 3090/4090, even a free-tier T4). Drivers of this "single-chip revolution":
- **Karpathy's llm.c / autoresearch** — strip bloated wrappers, compile the core architecture in C/CUDA; pre-train a small LLM from scratch on one GPU in *hours*.
- **"From scratch" educational frameworks** — Sebastian Raschka et al. publish step-by-step codebases to initialize/train/test locally on one card.

## 2. Data: "Textbooks Are All You Need"

To make an AI *reason* rather than *memorize*, change its diet. Microsoft's **Phi** papers ("Textbooks Are All You Need") trained small models almost exclusively on **curated synthetic data** — logic puzzles, step-by-step math, code — data that demonstrates cause-and-effect. Tiny models then punch far above their weight on pure reasoning.

**BabyLM Challenge** — a formal academic competition on exactly this: a strict data budget of **10M–100M words** (roughly what a 13-year-old has heard), forcing efficient algorithms for grammar/logic/behavior instead of brute-force scale. Small budget ⇒ easily trainable from scratch on minimal hardware.

## 3. Behavior: RL on tiny chips + tool-call training

Take the tiny base model and apply **Reinforcement Learning**: reward showing work, penalize jumping to wrong conclusions → a 100M-param model on a laptop emits a methodical chain-of-thought. "It won't know who won the 1994 World Series, but it will solve a logic puzzle."

**Tool calls are trainable the same way reasoning is** — a tool call is just generating text that matches a JSON schema. Teaching methods:
1. **Synthetic execution trajectories** — frontier models (Claude/GPT-4o) auto-generate tens of thousands of scenarios. One exercise = {system prompt with a fake tool's JSON schema} → {synthetic user query} → {exact JSON tool call} → {mock API response} → {final human-readable answer}. **ToolACE** used this to bring an 8B model to GPT-4-level tool-calling accuracy.
2. **RL for syntax + logic** — **"Tool Zero"** applies pure RL to a base model with no prior tool training, in a sandbox with real executable tools: a missing comma / wrong type / hallucinated tool → code fails → negative reward; success → positive reward. The model learns programming syntax's unforgiving nature by trial and error.
3. **Purpose-built synthetic frameworks** — e.g. Meta's open-source **Synthetic Data Kit** converts plain documents into large tool-calling/Q&A datasets. Aggressively filter to keep only flawless JSON.

## 4. Framework choice: Burn vs Candle (Rust)

- **Burn** = the factory. Full Rust deep-learning framework (training *and* inference). Backend-agnostic via **CubeCL** → ROCm (AMD 7900 XTX), CUDA, or WGPU/Metal (Mac). Rust ownership + JIT kernel fusion catch shape mismatches at compile time (no crash halfway through a multi-day run). Young ecosystem: you write the transformer blocks yourself — a hacker's framework. **Use Burn to design custom architecture, write training loops, bake in custom attention.**
- **Candle** = the delivery truck. Minimalist HF Rust framework, hyper-optimized for **inference/deployment**: compiles to WASM (models in the browser, no server cost), fast serverless cold-starts, edge/IoT. **Use Candle when training is done** and you're deploying frozen weights.

Verdict: build/train in **Burn**; the same struct code serves inference (inference = forward pass without backward). Candle is for later lightweight deployment.

## 5. Hardware verdicts (from the thread)

- **Train on the AMD RX 7900 XTX (24 GB).** Excellent official ROCm support now; run the training node on Linux (Ubuntu) or WSL2, install ROCm PyTorch (or Burn's ROCm/CubeCL backend). This is the **Factory**.
- **Mac Mini = the Workbench**, not the factory. Write Rust with Claude, compile-check the Burn model, generate synthetic data via APIs, do tiny (50-step) sanity runs. An **8 GB** Mac Mini *can* train only a very tiny (~100M) model and walks a tightrope — memory math ≈ 6.6–7.6 GB (weights 400 MB + gradients 400 MB + AdamW optimizer states 800 MB + activations 1–2 GB + macOS overhead 3–4 GB), so it hits **swap death**: SSD-as-RAM at ~3–5 GB/s vs unified 100+ GB/s turns hours into weeks and hammers the SSD with TB of writes. If using a Mac for real training, stay on macOS + **MLX** (Metal/unified memory); **do not use Asahi Linux** for training — it strips away Metal/MLX, the very optimizations that make Apple Silicon good at ML.
- **Creator / supervisor (data generation) → cloud inference API, not local.** Generating 50k complex trajectories = billions of tokens; cloud clusters do it in hours for a few dollars, using top-tier teachers. A $2,000 Mac to generate data is far less efficient than ~$15 of API credits. (Options named: OpenRouter, Groq, Together AI, OpenAI/Anthropic direct.)

## 6. Custom attention managers, mapped into Rust/Burn

The three context managers from the [companion research doc](2026-07-05-tiny-llm-huge-context.md), as forward-pass code:
- **DuoAttention** — in the `MultiHeadAttention` struct, split heads (e.g. of 8: 2 retrieval, 6 streaming). Retrieval heads push every token to the KV cache tensor; streaming heads act as a rolling buffer dropping tokens older than ~256 steps. Needs the short **gate-identification** training stage first.
- **RoPE extrapolation** — intercept the RoPE base frequency (10000); as the prompt exceeds trained length, a Rust function dynamically scales it down to "squish" a long prompt into the trained short-context shape. Train on 2,048-token chunks, infer at 64k+.
- **Episodic memory (EM-LLM) eviction** — the KV cache (often `Vec<Option<(K,V)>>`) grows until OOM. In the custom transformer block, read the attention-score matrix you already compute; prune cache tokens that consistently get near-zero attention ("unsurprising"). Pure inference, no fine-tuning.

**Why Rust/Burn for this:** CubeCL's JIT fuses these non-standard rules into native ROCm kernels for the 7900 XTX. The same custom episodic-memory loop in Python/PyTorch would drown in framework overhead; in Rust it compiles to bare metal.

---

## 7. The Project Report (preserved: "Project Grug-Brain")

**Objective:** build, train, and deploy a custom TLM from scratch under the toddler philosophy — weights learn strict logic, rigid JSON syntax, and reasoning ("High Grug"); all facts, language syntax (Rust, C#, C), and environment context are supplied at runtime via the KG and episodic memory.

**2. Architecture & custom attention managers** — sub-500M param Rust/Burn transformer with DuoAttention (gate-identified retrieval vs streaming heads), RoPE extrapolation (train 2,048 → infer 64k+), and EM-LLM Bayesian-surprise eviction. (Details in §6.)

**3. Data engine — Apertus + synthetic.** Weights are for logic, not trivia, so the corpus is ruthlessly curated: a compliant **Apertus** foundation plus targeted synthetic reasoning trajectories.

| Phase | Goal | Data source | Ratio |
|---|---|---|---|
| 1 — Foundation | Grammar, structure, basic comprehension | Apertus (filtered for high-quality, logic-heavy subsets) | 60% |
| 2 — High Grug logic | Hyper-literal step-by-step internal monologue | Synthetic (Claude/GPT-4o) | 15% |
| 3 — Coding syntax | Strict structure for Rust, C#, C | Synthetic + curated GitHub logic puzzles | 15% |
| 4 — Tool trajectories | JSON generation, schema adherence, API use | Synthetic execution trajectories (Tool Zero / ToolACE) | 10% |

**4. KG & memory integration.** The model isn't trained to guess facts. When High Grug reasoning determines it lacks a fact, it emits strict JSON, e.g. `{"action": "query_kg", "topic": "Rust library documentation"}`. A local **semantic router** intercepts it, queries a local graph DB / vector store of textbooks and programming blogs, and injects the retrieved text into the prompt. The EM-LLM mechanism then flushes those doc tokens from the KV cache once the code is written, freeing VRAM.

**5. Training pipeline & hardware.** Creator node (Mac Mini + APIs): Claude writes the Burn transformer/attention code; APIs generate synthetic trajectories + High Grug datasets. Factory node (AMD 7900 XTX): Burn/CubeCL compiles to ROCm; 24 GB VRAM runs the multi-stage runs (Apertus foundation → synthetic fine-tune → DuoAttention gate identification).

**6. Runtime execution & "textbook" ingestion — the inference loop:**
1. Input: user requests a C# script for a Unity mechanic.
2. High Grug monologue (hidden): *"User want C# script. Grug know C# syntax but Grug not know Unity API. Grug need read book on Unity."*
3. Tool call: `{"tool": "fetch_docs", "query": "Unity C# rigid body movement"}`.
4. Context injection: external system retrieves a blog post, pastes it into context.
5. Execution: RoPE extrapolation handles the suddenly huge prompt; the model reads the blog, applies trained logic, outputs the C#.
6. Cleanup: episodic-memory manager drops the blog tokens from VRAM, ready for the next prompt.

---

## 8. akh-medu notes (not from the source)

- **This is the BabyLLM, made concrete.** It aligns with the [fable-traces / BabyLLM note](../2026-07-05-fable-traces-babyllm.md): our Claude Code trace corpus is a ready source of Phase-4 tool trajectories — real, not synthetic — provided we keep the provenance + capability-filter + collapse-eval guards from that note.
- **High Grug already has design docs** — this thread's "internal monologue" is that project. Reconcile the toddler-simple "Grug speak" framing here with the *dense, exact, custom-token-ID* design in [2026-07-04-grug-reasoning-language.md](2026-07-04-grug-reasoning-language.md); they are the low/high registers of the same idea.
- **The KG + semantic router already exist** in the engine (Phase 28 semantic router, oxigraph/petgraph KG). The "query_kg" bridge is not new infrastructure — it's wiring a new consumer.
- **Burn is already the Phase 27 training target.** This report is a from-scratch pre-training variant of that phase, not a competing stack.
- **Open questions before this becomes a plan/ADR:** (a) is Apertus license/coverage adequate as a 60% foundation? (b) does a <500M model actually retain enough syntactic competence for three languages, or does it collapse (the fable-traces warning)? (c) DuoAttention gate-identification cost on the 7900 XTX; (d) how "High Grug" as a trained internal language interacts with custom token IDs vs. plain-text CoT.
