# Plan — LLM-driven OpenClaw workspace self-import

> Date: 2026-06-22
> Status: Implemented
> ADR: [`076-openclaw-self-import.md`](../decisions/076-openclaw-self-import.md)
> Anima WI: (to be linked)

## Goal

Let an akh's LLM agentic loop import its *own* OpenClaw workspace — the OpenClaw
agent in `~/.openclaw/openclaw.json` whose `id` matches the akh's id — into its
knowledge graph: persona/identity, memory journals, skills, and the SQLite/Lance
stores. The LLM is the transformation engine; it reads files and emits
strongly-typed structured data that deterministic writer tools persist. See the
ADR for the design rationale.

## What shipped

### Tool infrastructure (general, reusable)
- `ToolParamSchema.schema: Option<Value>` + `required_schema`/`optional_schema`
  constructors; `ollama_tool_catalog` emits nested JSON-Schema verbatim
  (`src/agent/tool_manifest.rs`, `src/agent/agentic_chat.rs`).
- `Tool::advertise_to_llm()` (default true) + `ToolRegistry::is_advertised` /
  `list_advertised_manifests`; catalog filters on it (`src/agent/tool.rs`).

### Retire raw-text NLU tools from the LLM session
- `text_ingest`, `content_ingest`, `journal_ingest` override `advertise_to_llm →
  false` (still registered for operator/daemon contexts).

### Typed writer tools
- `memory_write` (`src/agent/tools/memory_write.rs`) — typed episode
  `{summary, tags[], facts[{subject,predicate,object,confidence?}], source, origin,
  timestamp?}`; composes `index_cycle_episode`'s primitives; `origin=openclaw` →
  `OpenClawImport` provenance; idempotent on `episode:{source}:{ts|hash}`.
- `skill_author` (`skill_author.rs`) — `SkillAuthor::author_from_description` (stub)
  with the body grafted as `procedures.md`, then `AuthoredSkill::write_to_disk` +
  `Engine::load_skill`.
- `identity_awaken` (`identity_awaken.rs`) — builds `CharacterKnowledge` +
  `PurposeModel` from parsed persona fields and runs `ritual_of_awakening`.
- Registered as general builtins in `Agent::register_builtin_tools`.

### Switch-gated OpenClaw surface
- `WorkspaceConfig.openclaw_import: bool` + `extra_read_roots: Vec<PathBuf>` +
  `effective_read_roots()` (`src/workspace.rs`).
- `EngineConfig.extra_read_roots` widens the `LocalFs` **read** roots (writes stay
  `scratch/`) in `register_builtin_tools` (`src/engine.rs`, `src/agent/agent.rs`).
- `openclaw_discover` (`openclaw_discover.rs`) — resolves the akh's own OpenClaw
  workspace and returns a typed inventory (persona files, journal/skill counts,
  sqlite/lance presence, ready globs) with a secrets denylist. Registered in the
  daemon (`src/bin/akhomed.rs`) only when the switch is on; `get_engine` sets the
  read roots from `effective_read_roots()`.

### SQLite + Lance via shell
- No native deps. The import skill drives `shell_exec`: `sqlite3 -readonly -json`
  for `~/.openclaw/memory/<id>.sqlite`; OpenClaw's `knowledge/scripts` python
  exporter → JSONL → `file_io` for the Lance library. The LLM routes results into
  `memory_write`/`kg_mutate`.

### Provenance
- `DerivationKind::OpenClawImport { source_path, kind }` (`src/provenance.rs`,
  `src/agent/explain.rs`).

### Orchestration skill
- `openclaw-import` runtime skill (`procedures.md`) authored via `skill_author`:
  discover → awaken (if persona present & not awakened) → per `memory/*.md`:
  read + `memory_write(origin=openclaw)` → per `SKILL.md`: parse + `skill_author`
  → distilled persona facts via `kg_mutate` → SQLite via `shell_exec sqlite3` →
  (optional) Lance export → verify + report.

## Verification

- Unit: nested-schema render + `advertise_to_llm` filter (agentic_chat); typed
  episode write + idempotency + recall (memory_write); discovery inventory +
  denylist (openclaw_discover). All green (`cargo test --lib`).
- E2E: set tecton `openclaw_import = true`; `session/prompt` "import your OpenClaw
  workspace"; assert episodes with `OpenClawImport{kind:"journal"}`, identity
  awakened, SKILL.md-derived skills in `list_skills`; re-run → no duplicates.

## Deferred
- Native Lance ingestion (heavy arrow stack) — re-evaluate if arrow lands.
- LLM-assisted skill authoring (Phase 26b Candle) — `skill_author` uses the stub
  path today, which is exactly right for verbatim SKILL.md import.
