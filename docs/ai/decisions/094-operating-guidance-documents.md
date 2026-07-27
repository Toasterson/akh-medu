# ADR 094 — Operating-guidance documents: a composable, testable context-injection layer for tool doctrine

> Date: 2026-07-22 (drafted), 2026-07-27 (grounded against `main`)
> Status: **Proposed** — for grilling. Six open questions carry *proposed
>         resolutions* with code-grounded evidence; none is decided.
> Related: ADR 065 (context-assembly runtime — **the injection surface this
>          builds on**), ADR 064 (context-runtime lessons), ADR 081 slice 7
>          (schema/notes sections), ADR 053 (first-class `AkhConfig` /
>          authoritative `system_prompt`), ADR 070 (selectable Ollama/Candle
>          backing), ADR 071 (`akh` CLI command tree), ADR 061 (ACP-only control
>          plane), Phase 29 (`_akh/*` capability exposure), Phase 44j
>          (`src/bench_phase44/` A/B harness — **the test rig this targets**)

## Context

### The actual problem

The last model an akh ran on had the tool **schemas** but did not know **when**
to reach for a tool, **under what preferences**, or **with what terminology**.
Tool schemas alone do not teach usage doctrine. Other harnesses solve this with
tool-specific skill documents pointed to by the system prompt — prose that says
"prefer verb X for job Y; the term for Z here is W; don't call T unless…".
akh-medu has richer machinery than a flat system prompt, so the right move is to
add first-class injection places for this doctrine in our own way — composable,
and crucially **testable**, so a core document can be run against a model and its
behaviour change measured.

Two hard constraints:

1. **Must not override psyche or any other input.** Guidance composes *alongside*
   the psyche-derived / explicit system prompt, the schema digest, KG facts,
   notes, history and tools. It does not replace them.
2. **Must be a testable artifact.** A guidance document is a variable we tune and
   A/B against a model, not prose buried inside a persona string.

### The injection surface already exists (Phase 44 / ADR 065)

Working-context assembly already models the prompt as **ordered sections, not a
flat string** — `ContextSection` (`src/agent/context/types.rs:29`):

```
System → Schema → ActiveNotes → UserPrompt → KnowledgeGraph
       → RecentNotes → History → RecalledEpisodes → Tools
```

Declaration order is keep-first **priority** (`Ord` follows it); render order is
**positional** and differs — `compacting.rs:133` says so directly: *"blocks lead,
history + current prompt trail, closest to the generation point."* The
`BudgetAllocator` funds sections in priority order, atomic sections render
whole-or-not, and every section's token cost is telemetered in
`AssembledContext.section_tokens`. A `StrategyId` (`types.rs:73`) plus the
deterministic 44j A/B harness (`src/bench_phase44/`) already exist to compare
assemblies against a model.

That enum is precisely "injection places, our way." Doctrine belongs there as its
**own section** — funded at core tier, budgeted and measured independently,
composed without touching `ContextSection::System` where psyche and the explicit
`system_prompt` live.

### Why not `skills/` or `system_prompt`

- `src/skills/` is MoE **symbolic** knowledge — triples, egg rewrite rules, HTN
  plans. It feeds the reasoning engine, not the LLM's operating doctrine. We
  reuse its **lifecycle pattern** (Cold→Warm→Hot, manifest discovery via
  `SkillManager::discover()`, `src/skills/manager.rs:87`) but not its payload.
- `AkhConfig.system_prompt` (`src/akh_config.rs`) is a verbatim **replace** of the
  psyche prompt — `src/agent/model_manager.rs:102` documents it as *"replaces the
  psyche-derived prompt in `assemble_system_prompt`"*. It stays as the
  full-override escape hatch; doctrine is the additive, composable layer.

## Decision

Introduce **operating-guidance documents**: named, versioned prose artifacts that
compose into a dedicated context section, settable per-akh over ACP, and testable
as first-class variables.

### 1. A new context section — `ContextSection::Guidance`

Add a variant to `ContextSection`, in declaration order at **core tier** — after
`Schema`, before `ActiveNotes`:

```
System → Schema → Guidance → ActiveNotes → UserPrompt → …
```

- **Render:** one leading `system` block, header `"Operating guidance:"`, built
  exactly like the schema-digest block (`compacting.rs:~170`).
- **Budget:** atomic (whole-or-not), added to the atomic set and the allocator
  order; telemetered as `section_tokens[guidance]` for free.
- **`ContextSection::System` is untouched** — psyche and any explicit
  `system_prompt` render exactly as today. Guidance is a *sibling* block, never a
  suffix. This is the "does not override psyche" guarantee, enforced
  structurally rather than by convention.

`EpisodeState` gains `guidance: Option<&'a str>` (already-resolved doctrine text)
with a `with_guidance(...)` builder, mirroring `with_context` / `with_notes`. The
agent resolves and composes (it owns config + registry); the manager only budgets
and renders — the same division of labour the KG/recall sections already use.

### 2. Guidance documents — the testable artifact

```rust
pub struct GuidanceDoc {
    pub id: String,            // stable, e.g. "tools/filesystem"
    pub version: u32,          // bumped on edit; recorded in telemetry for A/B
    pub applies_to: AppliesTo, // All | Tools(Vec<String>) | Models(Vec<String>)
    pub body: String,          // the prose doctrine (markdown)
}

pub enum AppliesTo {
    All,
    Tools(Vec<String>),  // active only when >=1 named tool is in the catalog
    Models(Vec<String>), // active only for matching pinned models
}
```

`applies_to` makes docs *targeted* (filesystem doctrine loads only when the fs
tools are present) and *model-specific* (compensating for a small model's blind
spots), and keeps the section lean.

**Composition order** — each layer appends, none overrides:

```
global fleet doctrine (AkhomedConfig) → per-akh (AkhConfig) → per-tool (auto)
```

The resolver filters each layer by `applies_to` against the live tool catalog and
pinned model, concatenates survivors in stable id order under `### {id}`
subheaders so drops and edits are attributable, and hands the result to
`EpisodeState::guidance`.

### 3. Storage and lifecycle

- Per-akh reference in `AkhConfig`, additive and back-compatible:
  ```rust
  #[serde(default)] pub guidance: GuidanceRef,  // { docs: Vec<String>, inline: Option<String> }
  ```
  `#[serde(default)]` ⇒ legacy `akh.toml` loads unchanged; register the key in
  `known_top_level_keys()`.
- **Registry:** a `doctrine/` directory of `{id}.md` documents with YAML
  frontmatter (`version`, `applies_to`), discovered Cold→Warm→Hot exactly like
  `SkillManager`.
- **Global default set** in `AkhomedConfig` so common fleet doctrine is authored
  once and every akh inherits it.

### 4. ACP control surface — `_akh/guidance/*`

Added to `CONTROL_METHODS` (`src/acp/control.rs`) and the
`_akh/meta/capabilities` list; dispatch arms in `akhomed.rs` modelled 1:1 on the
per-akh `_akh/model/set` write path (load `akh.toml` → mutate → save →
hot-rebuild → refresh cached config).

| Method | Effect |
|---|---|
| `_akh/guidance/list` | active + available docs (id, version, applies_to, active?, **which layer won**) |
| `_akh/guidance/get` | resolved/composed doctrine text — what the model would see |
| `_akh/guidance/attach` | add a doc id to the akh's `guidance.docs`; persist + hot-apply |
| `_akh/guidance/detach` | remove a doc id; persist + hot-apply |
| `_akh/guidance/set-inline` | set/clear the `inline` blob; persist + hot-apply |

### 5. CLI — `akh guidance` (ADR 071 conventions)

```
akh guidance
  list [ws]                            _akh/guidance/list
  show [ws]                            _akh/guidance/get
  attach <doc-id> [ws]                 _akh/guidance/attach
  detach <doc-id> [ws]                 _akh/guidance/detach
  set [ws] (--from-file <p> | --clear) _akh/guidance/set-inline
```

### 6. Testability — the primary requirement

A guidance doc set is a first-class variable in the 44j A/B harness: run the same
task corpus with doctrine {A, B, none} and diff `TurnOutcome` (`answered`,
`rounds`, prompt/completion tokens) plus per-section token cost. Because
`Guidance` is its own budgeted, telemetered section, the harness sees its exact
cost and whether it was dropped under pressure. `version` is recorded per run, so
*"doc v3 cut tool-misfire rounds from 4→2 on gemma-4"* is a measurable,
attributable claim.

## Coverage across assembly paths

The section-based injection lands in the live `ContextManager` strategies
(`compacting.rs`, `passive.rs`, `virtual_ctx.rs`) via a shared block-builder. Two
further paths must compose it or explicitly opt out, or a model on that path
silently loses doctrine:

- **Candle prose path** (`src/agent/chat_backend.rs:300 render_tool_prompt`) —
  the small-GGUF akhs that need doctrine most. It builds a single string,
  `"{system_prompt}\n\n# Tools\n…{tools_json}…"`, so folding guidance in ahead of
  the `# Tools` block places it directly adjacent to the catalog.
- **`PassThroughManager`** — see resolution 5 below. It is **not** merely a
  baseline.

## Back-compat / vanilla-harness safety

- New `#[serde(default)]` `AkhConfig.guidance` ⇒ every existing `akh.toml` loads
  and round-trips unchanged; an akh with no docs and no inline produces a
  **byte-identical** assembly to today.
- New `ContextSection::Guidance` variant is additive; sections with zero
  requested tokens never render.
- `_akh/guidance/*` are `_`-prefixed ACP extensions — a vanilla ACP harness never
  emits them.

## Change set

| Layer | File | Change |
|---|---|---|
| Section type | `src/agent/context/types.rs` (`ContextSection`, `as_str`, `EpisodeState`) | add `Guidance` variant + `guidance` field/builder |
| Assembly | `context/compacting.rs` (+ `passive.rs`, `virtual_ctx.rs`) | build/budget/render the `Guidance` block via a shared helper |
| Doc model | new `src/agent/guidance/` | `GuidanceDoc`, `AppliesTo`, resolver, file registry |
| Config | `src/akh_config.rs` + `AkhomedConfig` (`src/config.rs`) | `guidance: GuidanceRef` per-akh + global fleet default |
| Agent wire-up | agent turn setup | resolve doc set → `with_guidance(...)` |
| Candle | `src/agent/chat_backend.rs:300` | compose guidance into `render_tool_prompt` |
| ACP roster | `src/acp/control.rs`, `src/acp/mod.rs` | add `_akh/guidance/*`, list in capabilities |
| ACP dispatch | `src/bin/akhomed.rs` (`_akh/model/set` is the template) | the five guidance arms |
| CLI | `src/main.rs` | `akh guidance {list,show,attach,detach,set}` |
| Test rig | `src/bench_phase44/` | parameterize over doc sets; record `version` |
| Docs | this ADR, `docs/ai/architecture.md` | land; bump architecture |

## Verification

- **Unit:** `AkhConfig.guidance` round-trips; legacy TOML parses; key typos flagged.
- **Unit:** resolver composes global→per-akh→per-tool with `applies_to` filtering
  by tool catalog + model; same-id collision resolves later-layer-wins and logs.
- **Assembly:** `Guidance` renders as its own atomic core-tier block; empty set ⇒
  byte-identical to today; **`System` is unchanged with or without doctrine
  present** — the non-override guarantee, asserted directly.
- **Control-plane:** `attach`/`detach`/`set-inline` persist `akh.toml`, `get`
  echoes the composed text, the next turn's assembly contains it.
- **Harness:** 44j A/B over {doc A, doc B, none} yields differing `TurnOutcome`
  and `section_tokens[guidance]`; deterministic and re-runnable.

## Consequences

- Operating doctrine becomes a **first-class, measured, composable** input —
  authored once (fleet) or per-akh, targeted per-tool/per-model, applied hot, and
  A/B-testable through the harness that already exists.
- Psyche, `system_prompt`, schema, KG, notes, history and tools are untouched.
- One new context section + one config field + one file-backed registry. No new
  transport, no new store.

## Open questions — proposed resolutions for grilling

Each carries evidence from `main`. **None is decided.** Four confirm the original
draft; one is deferred; **one reverses it.**

### 1. Section placement vs the tool catalog — *proposed: DEFER to the harness*

All atomic sections render as **leading** `system` messages while `Tools` is a
**divisible** section rendered at the tail (`compacting.rs:133`), so doctrine and
catalog are far apart on the section path. On the Candle path they are already
adjacent (`render_tool_prompt` builds one string).

Proposed: ship guidance as a leading core-tier block, expose render position as a
`StrategyId` variant, and let 44j answer it with data — render position is a
`blocks.push()` position, a one-line change and the cheapest possible experiment.
Do **not** pre-build the "short reminder adjacent to `Tools`" variant: `Tools` is
divisible, not atomic, so injecting into it is different machinery, and it would
be built on an untested hypothesis in an ADR whose premise is that doctrine
should be measured rather than assumed.

### 2. Registry location and format — *proposed: FILE-BACKED v1 (confirms draft)*

Seshat **is** reachable (`AkhomedConfig.upstreams`), but guidance is needed on
**every turn**, and a network registry makes *doctrine silently absent* a runtime
condition whenever seshd is unreachable. That is the exact failure this work
exists to prevent. Seshat is the natural v2 for **authoring and sharing**,
syncing down to local files — never a read-at-turn-time dependency.

### 3. `applies_to` matching — *proposed: exact tool, glob model, case-insensitive (confirms draft)*

`ToolSignature.name` (`src/agent/tool.rs:19`) is a flat registry key with no
hierarchy, so prefix matching would invent structure that does not exist.
`AkhConfig.model` is a free-form `Option<String>` while `ModelFamily` is a closed
enum, so glob over the string is the pragmatic join (`gemma*` catches
`gemma4:e4b`).

**Gap the draft missed:** `model` is an `Option`. Define `Models(...)` behaviour
when nothing is pinned — proposed **fail closed** (the doc does not apply), with
an explicit test, so an unpinned akh cannot silently inherit doctrine written for
another model.

### 4. Global-vs-per-akh precedence — *proposed: per-akh wins (confirms draft, with a caveat)*

`AkhomedConfig` (`src/config.rs:72`) holds `daemon`, `extraction`, `upstreams`,
`acp`, `postgres` — **there is no existing global-defaults-that-akhs-inherit
pattern.** This is new machinery, not a mirror of precedent; the draft implied
otherwise.

Per-akh-wins is still right on a consistency argument: every other per-akh
setting is an **override**, not a merge (`system_prompt` replaces psyche, `model`
pins the backend). Making guidance the one place the fleet layer wins would be
surprising. Because a fleet doc can then be silently replaced, the collision
notice must be INFO-or-louder and `_akh/guidance/list` must report which layer
won.

### 5. PassThrough baseline — *proposed: REVERSE the draft*

`src/agent/agentic_chat.rs`:

```rust
let manager: &dyn ContextManager =
    self.context_manager.as_deref().unwrap_or(&PASS_THROUGH);
```

`PassThroughManager` is **not only the A/B baseline — it is the production
fallback** for any agent with no context manager attached. Keeping it
doctrine-free, as the draft recommends, means any such akh silently receives no
doctrine: a configurable thing defaulting to absent, absence being silent, and
nobody noticing. It bites *new* akhs hardest.

Proposed: introduce a separate explicit `BaselineManager` for the harness control
arm and let `PassThroughManager` compose doctrine like every other strategy. The
A/B control must be an opt-in harness choice, never the production default. This
keeps both properties the ADR wants — a clean control arm *and* doctrine that
cannot silently vanish.

### 6. Relationship to `system_prompt` — *proposed: keep both (confirms draft)*

`model_manager.rs:102` documents `system_prompt` as a full **replace** of the
psyche-derived prompt (ADR 053). Guidance is additive and lives in its own
section, so the two cannot interact.

**Add:** `_akh/guidance/get` should warn when `system_prompt` is also set — an
operator who replaced the persona wholesale may not expect doctrine to still
compose on top.

## Provenance

Drafted in a Claude Code cloud session on branch
`claude/akh-system-prompt-infra-b5eqqv` of the `scratchpad` bridge repo
(2026-07-22, pinned to `develop @ 2a82e77`), then re-grounded against `main` and
extended with the proposed resolutions above on 2026-07-27. The starter corpus
(`tools/filesystem`, `tools/kg-memory`) lives with the draft.

**Note on the starter corpus:** `tools/filesystem` describes `file_io` as
`action: read|write` only. It predates the `offset`/`limit` parameters added
2026-07-27. Update it before use — doctrine describing a stale tool surface is
worse than no doctrine.
