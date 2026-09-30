# ADR 122 — Model-family guidance layer in the generated system prompt

> Date: 2026-09-30
> Status: Proposed (code on branch `feat/model-family-prompt-guidance`, default off; enable per akh
> after the dojo A/B in the evaluation report)
> Relates: [ADR 053](053-first-class-akh-config.md) (`akh.toml`, prompt assembly),
> [ADR 044](044-communicative-intelligence.md) / [ADR 099](099-agentic-communicative-intelligence.md)
> (communicative intelligence; the agentic-loop directive), [ADR 065](065-context-assembly-runtime.md)
> (context runtime), [ADR 105](105-autonomous-run-governance.md) (run governance),
> [ADR 113](113-full-akh-always.md) (every akh runs the full in-process turn),
> [ADR 116](116-lilith-chief-of-staff-guided-rl.md) (Lilith; the dojo teacher)
> Evidence: [2026-09-30 GLM / Kimi run evaluation](../research/2026-09-30-glm-kimi-run-evaluation.md)

## Context

`assemble_system_prompt` (`src/agent/model_manager.rs`) composes every akhomed system prompt from
four parts: the psyche identity, the operator section (`akh.toml` `system_prompt`), the tool
section generated from the live manifests, and the static operating conventions. None of the four
tells the model how to drive a tool loop: which arguments to send, what a failed call means, when
to stop, how to cite, or when a write has to be announced first. Codex and Claude run inside
provider harnesses that supply much of this. GLM-5.3 and Kimi K3 run in akhomed's own loop or in
akhrtd's small `internal` loop, where nothing supplies it. The runs examined in the evaluation show
the gaps. They added optional paging arguments, repeated failed calls, cited errored
results as evidence, ran into the round cap, and took state-changing actions without saying so.
Some of those failures come from the harness (replay fidelity in the dojo, round caps, lease
reclaim). The evaluation separates them. The rest are addressable in the prompt.

## Decision

1. **One machine-owned section, *Execution Discipline*,** rendered by
   `src/agent/prompt_guidance.rs` and composed by `assemble_system_prompt` after the operating
   conventions and before any Terse block. Five subsections: tool arguments, failed calls,
   evidence and claims, actions that change state, finishing. The wording is structural (schema,
   identifiers, results, rounds). It names no tool, so it cannot drift from the live catalog.
2. **Family addenda** (`ModelFamily::{Glm, Kimi, Other}`) append a few family-specific lines.
   The family is detected from the model *identifier* by vendor prefix (as `detect_prompt_mode`
   detects Gemma). It never reads conversation text, so the project rule against hard-coded
   natural-language matching is not touched.
3. **Per-akh opt-in:** `prompt_guidance = "off" | "auto" | "generic" | "glm" | "kimi"` in
   `akh.toml` (`AkhConfig::prompt_guidance`, default `off`, not serialized when off). Off keeps the
   prompt byte-identical, so no live akh changes behavior and no KV-cache prefix breaks (WI-357)
   until an operator opts in. An unknown value fails the config load.
4. **Paths covered:** the Ollama model manager (the Romans, the dojo teacher), the OpenAI-compatible
   backing (`OpenAiChatBackend::with_prompt_guidance`) and the `/v1` core context endpoint
   (`core_service::context`). Not covered: the Candle backing (Gemma, already Terse-mode) and the
   embedded Codex/Claude coding-agent runtimes, whose provider harness owns the system prompt.
5. **Live conventions under guidance.** With the layer on, the operating conventions keep only
   the lines whose tools are live in the turn (`CONVENTION_LINES` + `ConventionNeed` in
   `model_manager.rs`). Two contexts showed the problem: the session-only dojo teacher, and the
   `/v1` context of the Anima runtimes. Both were told to use `kg_query`, `skill_author` and a
   worktree contract they do not have. With the layer off, the constant is unchanged, and a test
   checks that the unfiltered line table reproduces the constant byte for byte.
6. **Two default-on truthfulness fixes** (bug fixes, not guidance):
   - An akh with no registered tools no longer says "No tools currently available." The `/v1`
     context of the Anima runtimes sent that line and then appended the session's tool schemas.
     The text now says the session supplies its tools.
   - MCP tool schemas inline local `$ref`s into `$defs`/`definitions` when each property is
     lifted out of `inputSchema` (`McpToolDescriptor::parse`). The coercion type is also read
     from the resolved schema, so a stringified object of a tagged union is parsed back to JSON.
     Before this, vesta's Board posts failed 9 of 9 times on a dangling
     `{"$ref": "#/$defs/RetentionPolicy"}`.
7. **The Anima session runtimes are not changed here.** These are the `internal` akh-platform akhs
   `sekhmet`/`tecton` on kimi-k3 and `maat`/`imhotep` on glm-5.3. Their system message has three
   layers:
   - Layer A: this repo's `/v1` context, which Decisions 4–6 cover once the akh opts in.
   - Layer B: the shared `instructions` in the Talos `akh-platform/render.py`.
   - Layer C: the Anima brief.
   The proposed Layer B and C text is in the evaluation report as a document. Runtime-profile edits
   are treated as conformance-gated. The investigation reports that `internal` runtimes are
   exempt, but that has to be verified before anyone relies on it.

## Why a static section, not an ADR 099 context directive

ADR 099 rejects prompt splicing outside the `ContextManager` for the per-turn communication
directive, because that directive depends on turn state and must yield under budget pressure.
The guidance layer is the opposite case. It is a function of the model, not of the turn. It must
not be evicted when the budget is tight, which is exactly when a looping model needs it most. As
a stable system-prompt suffix it also keeps the KV-cache prefix stable across turns. It therefore
belongs with the other machine-owned sections, not with the per-turn directive.

## Consequences

- The two fixes in Decision 6 change every akh's prompt or tool catalog once, on the next deploy.
  That is one KV-cache prefix break, and it removes a false statement and a dangling reference.
- The default is off. Enabling it is a per-akh decision backed by the dojo A/B (evaluation report
  §A/B). The dojo teacher takes the key from `config/dojo-teacher.json` (`prompt_guidance`,
  rendered by `full_akh.py`). For the Romans, it is a follow-up once the A/B is measured. For the
  Anima `internal` runtimes, the per-akh `akh.toml` on core-mac is the route; it reaches their
  `/v1` context without a runtime-profile edit. The akh config is cached for the life of the
  process, so any change takes effect on the next core restart.
- The rules overlap in part with role prompts (Lilith's "no claim without a receipt"). That is
  deliberate. The role prompt states the *policy*; the guidance states the *mechanics* in the
  tool-loop terms these models get wrong.
- Guidance cannot fix harness faults (unrecorded replay arguments, lease reclaim on a silent
  session) or model limits. The report marks which modes are which.

## Rejected alternatives

- **Always-on guidance for GLM/Kimi ids.** It would change live Romans without a measurement.
  It stays opt-in until the A/B shows the effect.
- **Stricter runtime enforcement only** (reject unknown optional arguments, tighter repeat
  limit). This is complementary, not a replacement. The repeat-loop detector already exists
  (`RoundBudget`), but it only fires on *identical* signatures, and the observed loops vary one
  argument. The report lists the runtime follow-ups.
- **Editing the operator prompts per akh.** That duplicates the same mechanics text into every
  role file and drifts. One generated section keeps it in one place.
