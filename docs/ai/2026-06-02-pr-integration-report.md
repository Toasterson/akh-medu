# PR Integration Report — 2026-06-02

Integration branch: `goal/integrate-merge-main` (based on `develop` @ `8899c46`).
Goal: merge the newest open pulls, analyze + fix each gap locally, land on `main`.

## Summary

- The repo had **~85 open PRs** (the API list is paged at 50). They were integrated in **two waves**.
- **Wave 1 — the newest 50 open PRs:** 48 branches integrated (47 of the 50 + 1 sibling fix
  branch `019e39a4` Ennead/Maat dedup). 3 skipped.
- **Wave 2 — the remaining ~35 older open PRs:** 30 more integrated across the Gemma+Psyche,
  MCP-tooling, contamination/memory/tecton-hygiene, OODA-tuning, and small/medium phase-scaffold
  clusters; 2 skipped as superseded (#217, #159 — their fixes were already present); 5 deferred.
- **Total: 78 PR branches integrated** in the two waves; a **follow-up pass then integrated the
  5 previously-deferred PRs too** (see "Follow-up" below) → **all 85 open PRs now integrated/closed.**
  Version bumped `0.6.0 → 0.7.0`.
- All branches reported `mergeable:false` by Forgejo (stale — each merged against an older
  `develop`; the real work was **cross-PR + against-already-merged conflict resolution**).
- **Compile gate** (after both waves): `cargo check --bin akhomed` with the full feature set
  (`mcp,nlu-full,candle-backend,delegation,wiki,calendar,email,sparql-persist,oxifed,wasm-tools,training-data`)
  passes (`Finished`, 0 errors). Wave-1 lib tests: **2167 passed, 0 failed**.

### Wave 2 highlights (older open PRs)
- **Gemma+Psyche (#180–186):** psyche→system-prompt serializer, KG/VSA context injection into
  prompts, ELIZA-vs-Gemma engine selector + `GemmaBackend` trait, per-agent model manager routing
  Gemma via an **external Ollama daemon** (`gemma4:e4b`), teacher-QA dataset generation, LoRA
  fine-tuning scaffold. (No native Candle Gemma loading yet — see gap analysis.)
- **MCP tooling (#191,198,199,203,204,218):** workspace param-deser fix, bounded daemon-run
  lifecycle, `config_inspect`, `cross_workspace_query`, `import_export`, `cleanup_by_pattern`.
- **Contamination/memory/tecton (#206,210,213,212,209,216,215,187,188,189,190):** Ma'at self-ref
  guard, per-cycle episodic indexing + KG-backed recall, Ennead deity identities + seed pack,
  apply-seed-packs-on-open, goal truncation/prune, workspace-isolation cleanup, journal ingest
  CLI/MCP tool, VSA-similarity identity guard, KG compaction task, write-once Ennead role,
  ingestion-capacity cap.
- **OODA tuning (#162,163)** + **phase scaffolds (#113 Affective, #119 Delegation, #115 Semantic
  Router, #114 Sensory Grounding).**

## Baseline fixes (so `develop` builds on stable Rust 1.94.1 with all features)

1. `src/agent/tools/infer_rules.rs` — dropped unstable `str::as_str()` (`str_as_str` feature); the value is already `&str`.
2. `src/nlu/candle_backend.rs` — imported `candle_core::IndexOp` for `Tensor::i()` indexing.

## Integrated clusters (and the non-trivial resolutions)

- **Bug/CI/fix batch:** CI false-failure fix, goal atomicity, psyche compartment injection,
  awaken parser broadening, redb writer-lock, max_cycles, bulk cleanup, hermeticity test,
  contaminant guard, ask-timeout dynamic budget, psyche lock self-heal, runtime-artifact
  suppression, self-referential meta-goal fix.
  - `run_until_complete` (agent.rs): combined the cycle-cap (PR155) with the empty-cycle
    early-exit (PR160).
  - `ask` MCP handler (mcp/mod.rs): combined the `create_goal` pure-read short-circuit (PR133)
    with the dynamic graph-scaled cycle budget (PR160) and the string-or-u32 `max_cycles`
    deserializer (PR155).
  - `compute_directed_curiosity` (PR139): new `goal_symbols_snapshot` arg threaded through all
    callers (daemon, agent, continuous_learning, drives).
- **akhomed cluster (019e0790):** chat/ask side-effect goals, agent_psyche/awaken agreement +
  poison-recovery (`lock_agent`), broader awaken parser, add/delete/update_goal endpoints.
  - daemon lock sites routed through poison-recovering `lock_agent()` (except the idle-cycle
    site, which borrows the `agent` field directly so `total_cycles += 1` doesn't conflict).
  - `bootstrap/purpose.rs`: **union** of declarative SVO patterns (PR154) and NL-intent patterns
    (PR195), with intent patterns ordered first; required hand-repair of brace seams.
- **Tecton bootstrap (019e1ab7):** self-referential triple purge, awake-pipeline dedup,
  compartment structure, causal-world-model bootstrap, episodic-memory population.
  Added `openclaw_sync_interval` to the `DaemonConfig` builder; fixed a double-borrow in
  `bootstrap_causal_schemas`.
- **Tecton workspace (019e6fee) + cross-MCP bridge (019e068d):** stale-goal cleanup, persona
  fix, self-model + home-cluster triples, `import_anima_project` / `link_seshat_doc` tools,
  OpenClaw journal ingest. Kept both `McpState` fields (`psyche_snapshots` + `upstreams`).
- **Phases:** 32 (LLM knowledge extraction), 41g (prompt→plan→execute bench), 42i
  (heka_contradict via ACH), 42j (public benchmark). New `DerivationKind` variants given
  distinct tags; Cargo/lib module additions unioned.
- **VSA-Agent (019e403b, all 7):** tool-manifest registration, MCP client for remote tools,
  manage_goals endpoint, candle LLM NL generation, goal auto-generation from VSA gaps,
  VSA-native RL signal, VSA-native action selection. The OODA decide-phase combined cleanly:
  `decide()` dispatches on `DecideMode` (`VsaNative` via `vsa_native_decide`, else `select_tool`),
  both paths capture `predicted_outcome_vec` and the Act phase computes the VSA cosine RL reward.
- **ELIZA (6 of 8):** core engine (172) on canonical `src/agent/eliza/`, domain assets (177),
  hot-reload KDL (178, refactored into `eliza/kdl.rs`, stray `.tmp` removed, re-gated to the
  `eliza` feature), VSA fuzzy match (173) + Seshat script store (179) as a sibling `src/eliza/`
  module, psyche-to-script generator (175).

## Follow-up — the 5 initially-deferred PRs were subsequently integrated

After the two waves, the 5 deferred PRs were each tackled one-by-one (all now **closed/integrated**):
- **#174** (ELIZA KG-grounded templates) — lifted into a new non-colliding `src/eliza/kg_template.rs`
  (Template fact-insertion + session cache + 100ms query timeout), preserving the existing engines.
- **#176** (ELIZA chat integration) — routing classifier lifted into `src/agent/eliza/routing.rs`
  (`ElizaRouter`/`ElizaRouting`/`FallbackReason`), wired into `ChatProcessor` (creative → LLM/KG
  fallback → OODA escalation).
- **#224** (Phase 21 game theory / grammar) — focused-subset 3-way merge of the 11 substantive
  files (`game_theory.rs` +598, grammar lexer/parser/etc.); the ~167 incidental stale rewrites and
  `.anima-exec/` junk were discarded.
- **#116** (Phase 30 Procedural Skills) — `src/skills/{author,quality,refine}.rs` + skill data
  (`skills/*/actions.json`,`plans.json`,`procedures.md`); fixed `ActionSchema` API drift.
- **#117** (Phase 31 Skill Intelligence) — `src/skills/{bridge,curriculum,discover,graph,share,synthesis}.rs`;
  reconciled `SkillManifest`/`LoadedSkill`/`SkillInfo`/manager across Phase 30+31.

Lib tests after the follow-up: **2467 passed / 0 failed** (full feature set compiles). `.anima-exec/`
runtime junk was kept out of all commits (gitignored).

## Original deferral rationale (historical — these are now integrated per above)

| PR | Branch | Why skipped |
|----|--------|-------------|
| **224** | `019db054-phase-33-grounded-verbalization` | 178 files, **+33,680/−7,102**. Commit titled "Phase 21 — Game-Theoretic Social Reasoning" on a *Phase 33* branch (internally mislabeled). The 7k deletions would revert/rewrite large parts of the 48 integrated PRs across nearly the whole tree. Un-integrable without dedicated rebase + identity resolution + splitting into focused PRs. **Recommend:** re-cut from current develop as scoped PRs. |
| **174** | `019e404a-eliza-kg-grounded-response-templates` | Reimplements the *same* `src/eliza::ElizaEngine`/`ElizaError` types as PR173 but with a fundamentally different design (KG-grounded `process()` vs VSA `respond()`, disjoint constructors/fields/deps under identical names). No additive union possible — a re-architecture, not a merge. **Recommend:** fold KG-grounded templates into the chosen single ELIZA engine as a feature, not a parallel type. |
| **176** | `019e404a-eliza-chat-integration` | `src/agent/eliza.rs` is a *file* that collides with PR172's `src/agent/eliza/` *directory* (Rust cannot resolve both as `agent::eliza`). Defines a third distinct `ElizaEngine` (routing classifier) and rewires `chat.rs`/`nlu`. **Recommend:** rename its module and re-target callers onto the canonical `src/agent/eliza/` engine. |

## Known follow-ups

- **ELIZA architecture is fragmented** across `src/agent/eliza/`, `src/eliza/`, and
  `src/compartment/eliza.rs`. The integrated subset compiles and the canonical path is
  `src/agent/eliza/`, but the three locations should be consolidated into one engine
  (resolving 174 + 176 in the process).
- **Gemma-driver + knowledge→VSA pipeline** gaps: see
  `docs/ai/plans/2026-06-02-gemma-vsa-pipeline-gap-analysis.md`.
- KG snapshot meta-key was reconciled to the canonical `kg_triples_snapshot` across several
  branches that had drifted to `kg_triples` / `kg_snapshot` / `kg_triples_v1`.
