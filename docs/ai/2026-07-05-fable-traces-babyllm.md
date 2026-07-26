# Analysis: `fable-traces`, and using our own traces for a BabyLLM

**Status:** Analysis / methodology note (no code, no decision locked)
**Date:** 2026-07-05
**Author:** Fable 5 via Claude Code, for Till
**Related:** ADR 037 (LLM-VSA deep integration), ADR 045 (T5 training infrastructure),
Phase 27 (Live KG→LLM training), Phase 35 (T5 training infra), [[project_gemma4_agentic_assessment]]

---

## 1. What `AliesTaha/fable-traces` actually is

It is a **joke**, and it is important to name that precisely before drawing any
methodology from it.

- The repo (`https://huggingface.co/AliesTaha/fable-traces`) is a genuine
  ~4B-parameter checkpoint: bit-for-bit `Qwen/Qwen3-4B-Instruct-2507`
  (`Qwen3ForCausalLM`, bf16 safetensors, ChatML template) republished under a new
  name. Nothing was distilled. The README's model card is written straight-faced,
  then closes with: *"This is a joke. This is not an actual model. Please read the
  full post first."*
- The tell is the model tag **`egypt-won`**. It rides a viral parody — fake
  arXiv-style screenshots — claiming that distilling *millions of Claude Fable 5
  reasoning traces* into Qwen3-4B produced a model **locked onto the answer
  "Egypt won"** for every prompt: arithmetic, algorithm implementation,
  mathematical proofs. There is no matching arXiv upload, company announcement, or
  real distillation run. It is unverified parody.
- Context: a whole cluster of real-but-cheeky "thinks like Claude Fable" Qwen
  finetunes appeared in the same window (`Qwable`, `Qwen3.6-*-FableVibes`,
  `Qwen3.6-27B-Fable-5-Experimental`, `Qwythos-*-Mythos-5`). `fable-traces` is the
  satire of that whole genre.

**So there is no dataset here, no schema, and no pipeline to reverse-engineer.**
The artifact is a punchline. What is worth keeping is the punchline's *content*.

## 2. The real lesson encoded in the joke

"Distill a big reasoning model's traces into a 4B and it collapses to a single
degenerate answer" is a caricature of a genuine failure mode. If we want to train
a **BabyLLM** on our own traces, the joke is a checklist of what to avoid:

1. **Mode / representation collapse.** Naive SFT on a narrow, self-similar trace
   corpus drives a small model to a low-entropy attractor. "Egypt won" is the
   cartoon version; the realistic version is a model that emits the same hedging
   preamble, the same tool-call boilerplate, or the same three refactors
   regardless of input.
2. **Style transfer ≠ capability transfer.** Traces teach the *shape* of Fable's
   output (structure, tone, tool-call syntax) far more readily than the
   *competence* behind it. A 4B fluent in "I'll start by…" that cannot actually
   reason is exactly the parody. This mirrors our own measured finding that binary
   VSA analogy is degenerate — surface arithmetic without grounded capability.
3. **Provenance laundering.** Republishing a base model as "distilled from Fable"
   is the joke's other barb. Our version of this sin would be shipping a BabyLLM
   whose training data we cannot audit back to source sessions. For a
   sovereignty-first project this is the cardinal rule, not a nicety.
4. **Contamination by our own filler.** Our traces are full of harness scaffolding
   (permission prompts, `ToolSearch` round-trips, worktree state, retries). Train
   on the raw stream and the BabyLLM learns to imitate the *bureaucracy* of the
   agent loop rather than the problem-solving inside it.

## 3. What "our traces" actually are

We already have a substantial, real corpus — the Claude Code session transcripts
for this project. This is the genuine "Fable traces" the joke only pretends to
have.

- **Location:** `~/.claude/projects/-home-toasty-ws-agi-akh-medu/` (plus one dir
  per worktree, e.g. `…--claude-worktrees-<name>/`).
- **Scale (this project only):** 26 `*.jsonl` session files, ~9.7 MB. Individual
  sessions range from ~170 KB to ~10 MB. This grows every working session.
- **Format:** newline-delimited JSON. One line = one event. Observed record
  `type`s: `user`, `assistant`, `system`, `attachment`, `file-history-snapshot`,
  `agent-setting`, `mode`, `permission-mode`, `last-prompt`, `ai-title`,
  `agent-name`, `worktree-state`.
- **The signal lives in `user` + `assistant` records.** An `assistant` record
  carries `message.content` (an array of `text` and `tool_use` blocks),
  `message.model`, `message.usage`, `message.stop_reason`, plus turn metadata
  (`cwd`, `gitBranch`, `sessionId`, `parentUuid`, `timestamp`, `requestId`,
  `isSidechain`). Tool results come back as `user` records containing
  `tool_result` blocks. `parentUuid` threads the DAG.
- **The noise is everything else.** `permission-mode`, `mode`, `agent-setting`,
  `worktree-state`, `ai-title`, `file-history-snapshot` are harness state, not
  reasoning. `isSidechain: true` marks sub-agent traffic (the `Agent` tool) — keep
  or drop deliberately, don't let it leak in by accident.

Minimal record shape (assistant turn), for the extractor:

```jsonc
{
  "type": "assistant",
  "message": {
    "role": "assistant",
    "model": "claude-fable-5",
    "content": [
      { "type": "text", "text": "..." },
      { "type": "tool_use", "name": "Bash", "input": { "command": "..." } }
    ],
    "stop_reason": "tool_use",
    "usage": { /* token counts */ }
  },
  "parentUuid": "...", "uuid": "...", "sessionId": "...",
  "cwd": "...", "gitBranch": "...", "timestamp": "..."
}
```

## 4. A sound methodology (what to actually build)

The goal that fits this project: a **small, sovereign, local coding assistant**
distilled from our own real sessions — not a general "thinks like Fable" cosplay
model. Keep it narrow and it can be genuinely good; make it broad and it becomes
the parody.

### 4.1 Extraction (deterministic, auditable)

Write a Rust extractor (idiomatic, `miette` diagnostics, strongly-typed records —
model the JSONL schema as an enum over record types, not `serde_json::Value`
soup). It should:

1. Walk every `*.jsonl` under the project + worktree dirs.
2. Reconstruct conversation threads via `parentUuid` → ordered turns.
3. Emit **provenance for every training example**: `(sessionId, uuid range,
   git commit, source file)`. This is the anti-"laundering" guarantee — every
   token in the BabyLLM traces back to a real session. Store it the same way the
   engine stores `ProvenanceRecord`s so it is queryable, not a loose sidecar.
4. Drop harness-noise record types entirely. Optionally partition sidechain
   (sub-agent) turns into a separate corpus.

### 4.2 Curation (this is where the joke is defeated)

- **Outcome-filter.** Prefer trajectories that *ended in a verified good state* —
  tests passed, build clean, user said "yes". A trace's value is whether the work
  landed, not whether it looks like Fable. Sessions have git branches; cross-check
  against commits that survived.
- **Deduplicate aggressively.** Our sessions repeat boilerplate ("I'll start
  by…", identical tool-call preambles). Near-dup collapse (minhash / embedding
  cluster) directly counters the low-entropy attractor from §2.1. Track how much
  you drop and *log it* — silent truncation reads as coverage.
- **Strip the scaffolding from targets.** The BabyLLM should learn the reasoning
  and the tool *decision*, not the permission-prompt dance around it.
- **Hold out whole sessions, never turns**, for eval — otherwise the model
  memorizes a session's idiom and the eval leaks.

### 4.3 Format & training

- Target format: **ChatML** (Qwen3-4B-Instruct's native template, exactly what
  `fable-traces` inherits) with the `<tool_call>`/`<tool_response>` convention from
  its chat template. Tool calls in our traces map cleanly onto it.
- Method: **LoRA/QLoRA SFT** on Qwen3-4B (or smaller — a true "Baby" 0.5–1.5B for
  the sovereignty target), *not* full-weight distillation. This matches the Phase
  27 direction (Burn + LoRA) and Phase 35's offline-training infra. Reuse that
  infrastructure rather than standing up a parallel pipeline.
- If we want reasoning and not just style, do **rejection-sampling / best-of-N
  distillation** (keep only trajectories that verifiably solved the task), which is
  what actually transfers capability — the thing the parody says doesn't happen.

### 4.4 Evaluation (prove we are not the joke)

The joke's whole payload is "every prompt → Egypt won." So the eval is explicitly
an **anti-collapse** eval:

- **Output entropy / diversity** across a held-out prompt set — a collapse
  detector by construction.
- **Held-out task success**, not perplexity: can the BabyLLM propose the correct
  edit / command on unseen akh-medu sessions?
- **A/B against the base Qwen3-4B.** If our SFT does not beat stock Qwen on our
  own domain, we have transferred style, not capability — and we should say so
  plainly rather than ship a `fable-traces` of our own.

## 5. Recommendation

1. Treat `fable-traces` as what it is — satire — and keep only its warning.
2. Our real asset is the **Claude Code JSONL transcript corpus** we are already
   generating; it grows for free every session. The single highest-leverage first
   step is the **deterministic, provenance-preserving extractor** (§4.1), because
   nothing downstream is trustworthy without it.
3. Scope the BabyLLM **narrow** (sovereign local Rust/akh-medu assistant), route it
   through the **existing Phase 27 / Phase 35 training infra** instead of a new
   stack, and gate release on the **anti-collapse eval** (§4.4).
4. This aligns with the project's north stars: FLOSS sovereignty (our data, our
   weights, our provenance, no cloud dependency) and the long game of
   out-competing Claude at Rust — using Fable's own working traces as the seed,
   honestly labelled.

The difference between this and the joke is exactly three things we control:
auditable provenance, capability-filtered curation, and a collapse eval. Skip any
one and we have built `egypt-won`.
