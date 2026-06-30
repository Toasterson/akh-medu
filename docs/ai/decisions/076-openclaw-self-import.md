# ADR 076 — LLM-driven OpenClaw workspace self-import

> Date: 2026-06-22
> Status: Accepted (implemented)
> Related: ADR 061 (ACP-only — `akh` is a pure ACP client), ADR 070 (selectable
>          chat backing / agentic loop), ADR 064/065 (context-assembly runtime),
>          Phase 30 (procedural skills), `src/bootstrap/identity.rs` (awakening),
>          CLAUDE.md ("strongly typed idiomatic rust… express the function in
>          datatypes")

## Context

The operator runs OpenClaw (`~/.openclaw`); its default agent `id:"tecton"` lives
in `~/.openclaw/workspace` and there is a matching akh-medu akh named `tecton`.
We want an akh to import its *own* OpenClaw past life — persona, memory journals,
skills, and the SQLite/Lance stores — into its knowledge graph.

The import must be **LLM-driven**: the akh's agentic loop discovers, reads,
reasons about, and imports the files. The LLM is the transformation engine. "Its
own" workspace = the OpenClaw agent whose `id` equals the akh's id, resolved from
`~/.openclaw/openclaw.json`.

## Decisions

### 1. Typed writer tools; retire raw-text NLU tools from the LLM catalog
Any ingest tool that takes **raw text** and runs an internal NLU/heuristic
extraction is redundant inside an LLM session — the model already reasons better
than the regex/NLU passes. Such tools (`text_ingest`, `content_ingest`,
`journal_ingest`) are **retired from the agentic tool catalog** (they stay
registered for operator/CLI/daemon-ingest contexts). The LLM instead reads raw
text as *input* and emits **strongly-typed structured data** that deterministic
writer tools persist with no second internal LLM call:
- `memory_write` — one typed episode `{summary, tags[], facts[{s,p,o,conf?}], source, origin, timestamp?}`; the typed replacement for `journal_ingest`.
- `kg_mutate` — single typed fact (kept).
- `skill_author` — register a skill from `{id, name, description, procedures_md, domains[]}` (an OpenClaw `SKILL.md`).
- `identity_awaken` — run `ritual_of_awakening` from parsed persona fields.

Mechanism: a defaulted `Tool::advertise_to_llm() -> bool` (default `true`; the
retired tools override to `false`). `ToolRegistry` records it and
`ollama_tool_catalog` filters on it. Chosen over a `ToolManifest` field because
(a) the manifest is serialized/shared (federation) and advertise-to-LLM is a
local runtime concern, and (b) it avoids editing ~80 manifest construction sites.

### 2. Native nested tool-call schema
`ToolParamSchema` gains an optional `schema: Option<serde_json::Value>` JSON-Schema
fragment; `ollama_tool_catalog` emits it verbatim (merging in a description). This
lets the LLM be handed precise nested types (e.g. `facts: [{subject, predicate,
object}]`). The Candle GBNF + JSON constraint already accept arbitrary nested
JSON, so no decoder change was needed. Existing flat tools are unaffected
(`schema: None` → old rendering).

### 3. OpenClaw-specific surface is switch-gated
The only OpenClaw-named tool (`openclaw_discover`) and the read-root widening are
injected into a session **only when `WorkspaceConfig.openclaw_import` is on**
(env override `AKH_OPENCLAW_IMPORT` is intentionally *not* added; the per-akh
TOML switch is the single source of truth). Off by default, so normal akhs are
unaffected. The general typed writers are always-on builtins.

### 4. Filesystem unlock, read-only & opt-in
The agent's tool sandbox normally reads only its data dir, so `~/.openclaw` is
`OutsideSandbox`. When the switch is on, `WorkspaceConfig.effective_read_roots()`
adds `~/.openclaw` (plus explicit `extra_read_roots`) to the engine's
`extra_read_roots`, which `register_builtin_tools` folds into the `LocalFs` **read**
roots only — writes stay confined to `scratch/`. `openclaw_discover` additionally
carries its own `LocalFs::read_only` and applies a secrets denylist (`.ssh`,
`.gnupg`, `credentials`, `secrets`, `*.pem`, `id_rsa`, `.gitconfig`, `service-env`)
so credential material is never surfaced. `LocalFs` already rejects symlink
escapes via canonicalization.

### 5. SQLite & Lance via shell, not native deps
We do **not** add `rusqlite`/`arrow`/`lance` to the daemon (the `libsqlite3-sys`
conflict is documented at `Cargo.toml:130-133`; the arrow stack is heavy and the
vectors are re-derivable since the akh recomputes its own VSA embeddings). Both
stores are read out-of-process via the existing `shell_exec` tool: SQLite with
`sqlite3 -readonly -json`, Lance via OpenClaw's `knowledge/scripts` python
exporter → JSONL → read with `file_io`. Recorded deferred-native; revisit if the
arrow stack ever lands for another reason.

### 6. Distinct provenance variant
`DerivationKind::OpenClawImport { source_path, kind }` (kind ∈
journal|skill|persona|sqlite_memory|vector_export) attributes self-imported
knowledge distinctly from generic `FileImport`, so the akh can query exactly what
it absorbed from its OpenClaw past life and dedupe on re-import. `memory_write`
emits it when `origin=openclaw`.

### 7. Orchestration as a runtime skill
The import recipe is an akh runtime skill `openclaw-import` (`procedures.md`)
authored via `skill_author`, so the logic is durable, inspectable (`list_skills`),
and re-runnable. A user kicks it off with one ACP `session/prompt`; the logic
lives in the skill, not an ephemeral prompt.

## Consequences

- An LLM session sees only deterministic typed tools for writing; raw-text NLU
  ingestion is an operator/daemon concern.
- Tool params can now be strongly typed (nested) — reusable well beyond this
  feature.
- Importing is opt-in per akh and read-only on the OpenClaw side; no new heavy
  dependencies enter the build.
- Idempotent: deterministic episode labels, `install_skill` overwrites by id,
  `ritual_of_awakening` refuses re-awaken without `force`, and `OpenClawImport`
  provenance supports already-imported scans.

## Implementation

- Schema/catalog: `src/agent/tool_manifest.rs`, `src/agent/tool.rs`,
  `src/agent/agentic_chat.rs`.
- Retire flags: `src/agent/tools/{text_ingest,content_ingest,journal_ingest}.rs`.
- New tools: `src/agent/tools/{memory_write,skill_author,identity_awaken,openclaw_discover}.rs`.
- Switch + FS unlock: `src/workspace.rs`, `src/engine.rs`, `src/agent/agent.rs`,
  `src/bin/akhomed.rs`.
- Provenance: `src/provenance.rs`, `src/agent/explain.rs`.
- Plan: `docs/ai/plans/2026-06-22-openclaw-self-import.md`.
