# ADR 104: BabyLLM effective context and copy-faithful decoding

- **Status:** Accepted 2026-09-07
- **Phase:** 51 (Grug-Brain BabyLLM), follows ADR 102/103 and the
  `tool-native-4k-lane-20260906-RESULT.md` measurement
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan
  `019f90cb-2e66-7f92-ba88-a4060be97c6e`; work items listed in
  `docs/ai/plans/2026-09-07-phase51-p5-effective-context-copy-decoding.md`
- **Related:** ADR 065 (context assembly runtime), ADR 064 (context runtime
  lessons), ADR 067/070 (candle boundary, selectable backing), ADR 099
  (communicative intelligence in the agentic loop), ADR 102 (tool-native
  learning model), ADR 103 (certified native OCR runtime)

## Context

The 4096-context `small` lane (2026-09-06) ended with the adaptive controller
stopping at window 2 on clean evidence. At n=100 closed-loop sentinels the
35.5M-parameter model executes five of seven knowledge workflows at 0.78–1.00
and fails two (acquire_textbook 0.43, revise_from_new_edition 0.12) on one
capability: copying a hex identifier that contains a doubled character
(`…c7927279…` → `…c79279…`; `5dd3d` → `5d3d`). The copied value then fails
the schema `pattern` (`^[0-9a-f]{64}$`, `^note-[0-9a-f]{32}$`) and the step
is graded catastrophic. Tool choice, ordering, grounding and evidence
discipline were otherwise correct in those episodes. A second ordinary SFT
epoch did not move the failing families and regressed three healthy ones.

Three facts about the current implementation frame the decision:

1. **The decoder is unconstrained where it matters.** The deployed runtime
   (`KnowledgeModel::generate_turn_with_system`) and every evaluator decode
   with plain greedy argmax on the KV-cached path. Grammar-constrained
   decoding exists (`grug_data::constrain::ToolCallGrammar`,
   `generate_constrained`) but is wired to nothing, constrains structure and
   keys only, and never values. Schema `pattern` is checked *after*
   generation (`ToolContractSnapshot::validate_call`), and the grader's own
   "grounded" rule (`grug_data::ground`) already demands that a string
   argument appear (near-)verbatim in the request/context. We grade for a
   property we do not enforce.
2. **The tokenizer is natural-language BPE.** Byte-level BPE trained on
   FineWeb en/de/fr merges runs of hex digits inconsistently; the same
   identifier can tokenize differently in prompt and completion. A 35M model
   has to re-learn a copy operation per merge pattern. It nearly does; it
   fails on the doubled characters.
3. **The context window is bounded by memory, not by a number.** Positions
   are a learned absolute table sized `max_seq_len`. Attention is hand-rolled
   and materializes the full `[batch, heads, L, L]` fp32 score tensor; at
   4096 that is 5.3 GB per layer, batch 8 OOMs an 80 GB A100 and SFT needs
   batch 2. Training is fp32 with no activation recompute. Burn 0.21 ships
   `RotaryEncoding` but no fused/flash attention kernel.

The user's stated goal is a model useful for coding and real tasks, with
"100k context" named as the target. Two ADRs already answer part of this:
ADR 102 puts changing knowledge in runtime stores (Seshat, notes,
`archival_search`, `memory_recall`) and states that small models run the
`compacting`/`passive` strategies of ADR 065, which page evidence into a
bounded window. The question this ADR settles is what the *weights* must
be able to do, what the *harness* must guarantee, and which numbers are
targets versus which are rejected.

## Decision

### D1. Copy-faithful, grammar-constrained decoding is a runtime guarantee

Values that must equal something already in the context are **copied by
the decoder, not recalled by the weights**. The Grug runtime and every
evaluator decode through one constrained path:

- the existing `ToolCallGrammar` supplies structure, keys and EOS;
- a new byte-level *value constraint* applies per argument: `enum` → the
  enum strings; `pattern` (or fixed-length identifier fields) → the set of
  context substrings matching the pattern, matched byte-wise so tokenization
  boundaries are irrelevant; free-form strings (note bodies, queries) stay
  unconstrained;
- the `Gap` abstention envelope remains a first-class alternative to a tool
  call in the grammar — the "I don't know" path must never be squeezed out
  by the constraint;
- when a pattern field has no candidate in context the decoder falls back
  to pattern-only and records `CopyFallback` in a `DecodeReport`; the
  evaluators grade a fallback as ungrounded exactly as today.

This extends ADR 102 §4 ("the scorer and daemon use the same strict runtime
action parser") to "the same decoder". Sentinels then measure workflow
competence (which tool, which evidence, when to abstain) instead of BPE
luck. It is the same stance ADR 099 takes for the grammar pipeline and the
cheapest fix available: it applies to the *existing* 4096 checkpoint without
retraining and is the first thing to measure.

### D2. Identifier-aware tokenization

New foundation tokenizers split decimal digits individually (the Llama rule)
and split runs of ≥8 hex characters into single characters, recorded as a
typed `PreTokenization` in the tokenizer artifact so the hash lineage of
ADR 102 carries it. Copying an identifier becomes a one-token-per-character
task; short words such as "cafe" or "added" stay whole. Existing artifacts
are self-describing and unchanged. D1 makes correctness independent of the
tokenizer; D2 makes the *unconstrained* model better too and shrinks the
gap the constraint has to cover. It costs tokens (a 64-hex token is 64
tokens), which D3 absorbs.

### D3. Rotary positions and memory-bounded attention; train at 8k

`akh_tool_small_8k` and `akh_tool_350m_8k` replace the learned position
table with RoPE (Burn `RotaryEncoding`, θ=10000, serving-time position
interpolation `scale`) and compute attention in query blocks
(`attention_chunk`, default 1024) so materialized memory is
`batch × heads × chunk × L` instead of `L²`. A Burn autodiff
`BalancedCheckpointing` build knob trades recompute for activation memory.
The existing 4096 presets, artifacts and manifests are untouched: the new
fields default to `Learned`/`None` on load. Training length is 8192 at the
same 819.2M-token budget (batch 2 × 50k steps). Serving length is extended
beyond 8192 only as far as a length-generalization sentinel keeps passing
(see D5), using `scale` — RoPE has no table to outgrow, Learned refuses
with a typed error.

### D4. 100k raw context is rejected for this rung; "effective context" is the harness's job

A 100k-token window is not what makes a 35M–350M model useful, and this
lane will not pursue it:

- **Memory.** Without a fused attention kernel, L² materialization at 100k
  is 10⁴× the 4096 footprint; chunked attention bounds temporaries but the
  KV cache alone at 100k for the `small` shape is ~1.6 GB fp32 per sequence
  and prefill is quadratic in time. Burn has no flash-attention today;
  writing one in CubeCL is a project, not a fix.
- **Capacity.** Small models do not use long windows: retrieval accuracy
  falls with distance and position ("lost in the middle",
  arXiv:2307.03172), and nothing in the 819.2M-token budget teaches
  100k-token dependencies.
- **Design.** ADR 065/102 already provide effective context by construction:
  the `compacting`/`passive` context strategies page KG, archive and note
  evidence into the window; the seven knowledge families *are* the model
  learning to run that memory (ingest, distill, revise, recall, navigate,
  fall back, abstain). The correct measure of "how much can it work with"
  is the size of the corpus it can operate over through those tools, which
  is unbounded, not the window.

Coding tasks at the level of a capable model remain the job of the
capable-model backings (Ollama/Candle under the `virtual` strategy, ADR 070).
BabyLLM's coding bar is *measured*, not asserted: it enters Phase 50's
tiny-LLM track of the benchmark leaderboard once the release gate certifies
a text model, with the same illumos-sysadmin/agentic E2E corpus and the same
deterministic verifiers as the capable models. What "above its weight class"
means for this rung is precise: the weights decide *which tool, with which
grounded arguments, or abstain*; the engine supplies knowledge (KG, Seshat,
notes), reasoning (thinking loop, e-graph, Dempster–Shafer), grammar and
copying (D1), and memory (ADR 065). Every capability removed from the
weights and given to the harness is one the sentinels stop failing on.

### D5. The capability ladder and how much guided learning it takes

The weights are graded on five levels; the harness owns level 0:

| level | capability | owner | measured by |
|---|---|---|---|
| 0 | well-formed action, valid schema, verbatim identifiers | runtime (D1) | `DecodeReport`, zero `SchemaViolation` |
| 1 | correct tool for the request in en/de/fr | weights | all-tool matrix (53 × 12) |
| 2 | grounded arguments (free-text fields) | weights | closed-loop `ungrounded` rate |
| 3 | abstain when evidence is missing; refuse ungrounded writes | weights | `grounded_abstention` family |
| 4 | multi-step workflow state across tool results | weights | seven closed-loop families |
| 5 | recovery: conflict re-read/retry, provider fallback, restart recall | weights | `revise_from_new_edition`, `provider_fallback`, `recall_after_restart` |

plus one new immutable sentinel family, `knowledge.length_generalization`:
the same recall/navigation episodes with the retrieved evidence placed
4k–8k–12k–16k tokens back in the prompt. It gates serving length in D3.

**How much guided learning:** the measured answer is *one SFT epoch to the
floor, then stop*. Window 1 reached the floor on five families; window 2
regressed three of them (P4 lesson: do not train past the floor). Ordinary
windows therefore run at a reduced learning rate (`--ordinary-learning-
rate-scale`, 0.2 recommended) and the controller's two-window confirm→stop
stays the authority on when to stop. New competence enters as **new
curriculum families and higher difficulty from the generator**, chosen by
the controller's frontier, not as more epochs on the same data. If a family
stays red after D1 with the unconstrained copy failures removed, that is a
capacity signal for the 350m rung, not a signal for more epochs.

### D6. Serving budget

`AKH_GRUG_MAX_NEW_TOKENS` defaults to 768 (was 384); the offline/cross-host
gates use the same value. Note-writing targets need ≥640 tokens today and
D2 makes identifiers longer.

## Rejected alternatives

- **Fix the generator to avoid doubled hex characters.** Production
  revision tokens are real BLAKE3 hashes; the generator would lie about the
  contract.
- **Bigger rung first.** `350m` copies better but keeps the tokenizer
  failure mode and costs 3–5× per lane; it is the second lever, taken only
  if a family remains red after D1/D2.
- **Sliding-window attention as the primary path.** Bounded memory, but the
  note workflows must quote evidence far back in the prompt; with 8 layers
  the effective receptive field of a 4k window is weak. It remains the
  fallback if chunked full attention with recompute does not fit 8192 at
  batch 2 on the A100.
- **Mixed precision (bf16).** Halves activation memory but Burn has no
  autocast; loss-scaling and per-op precision would be hand-built.
  Deferred, recorded as the next memory lever after D3.
- **Sampling / beam search to recover copies.** Adds variance to a
  deterministic contract; the constraint is exact and cheaper.

## Consequences

- The 4096 checkpoint is re-probed with D1 before any retraining. Expected:
  the two hex-copy families rise to their siblings' level; if they do not,
  the residual is workflow competence and the plan's capacity branch opens.
- The next foundation retrain carries D2 and D3 together (one tokenizer
  change, one architecture change, one lineage break). Every artifact,
  manifest and `run.json` written before this ADR still loads; the new
  fields have defaults.
- The all-tool matrix and the closed-loop scorer change meaning: they now
  measure the weights *inside the harness*. Numbers before and after this
  ADR are not comparable; results documents must name the decoder.
- Pod cost for the two measurements this ADR orders: re-probe ≈ 1–2 h;
  8k foundation + SFT + probe ≈ 18–22 h on the A100 (~$30–35). Both require
  the user's explicit approval to restart the pod (`.claude/rules/
  runpod-babyllm.md`).
- ADR 102's "first deployable capacity rung" sentence is superseded in one
  respect: the deployable shape is `akh_tool_small_8k` (RoPE, chunked
  attention) once its lane certifies; `small` at 4096 remains the measured
  baseline.

## References

- RoFormer / RoPE — arXiv:2104.09864; YaRN length extension — arXiv:2309.00071
- Lost in the Middle — arXiv:2307.03172
- FlashAttention — arXiv:2205.14135 (what Burn lacks; why D4)
- MemGPT (virtual context) — ADR 065 lineage
- `training/results/tool-native-4k-lane-20260906-RESULT.md`,
  `training/results/p350m-RESULT.md`
