# PR Integration Report — 2026-06-02

Integration branch: `goal/integrate-merge-main` (based on `develop` @ `8899c46`).
Goal: merge the newest open pulls, analyze + fix each gap locally, land on `main`.

## Summary

- **50 open PRs** at start, all targeting `develop`, all reported `mergeable:false` by Forgejo
  (stale signal — each branch's merge-base actually equalled current `develop` HEAD, so each
  merged cleanly *in isolation*; the real work was **cross-PR conflict resolution**).
- **48 branches integrated** (47 of the 50 open PRs + 1 sibling fix branch `019e39a4`
  Ennead/Maat dedup that was the better variant for its slot).
- **3 PRs deliberately skipped**, each documented below as an analyzed gap.
- Result vs `develop`: **127 files changed, +26,278 / −420**, 101 commits (48 merge commits).
- **Compile gate:** `cargo check --bin akhomed` with the full capability feature set
  (`mcp,nlu-full,candle-backend,delegation,wiki,calendar,email,sparql-persist,oxifed,wasm-tools,training-data`)
  passes (`Finished`, 0 errors).

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

## Skipped PRs (analyzed gaps — NOT integrated)

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
