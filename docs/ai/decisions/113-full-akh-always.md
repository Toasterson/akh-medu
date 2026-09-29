# ADR 113 — Every akh runs fully featured; the service split contains execution, not cognition

> Date: 2026-09-24
> Status: Accepted (grilling session 2026-09-24, ten decisions locked by the operator)
> Phase: 53 (plan: `docs/ai/plans/2026-09-24-phase53-full-akh-always.md`)
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, umbrella WI `01a0d43b-bf90-7630-bbc6-bd31d0609864` (slice WIs listed in the plan)
> Supersedes: the `core-server` build profile (`docs/wayfinder/agent-loops/implementation.md` "Added an `akhomed` `core-server` build profile") and the `akhomed` row of `docs/wayfinder/agent-loops/assets/daemon-boundaries.md` §1 as far as it excludes resident loops, automation and admission from `akhomed`
> Related: ADR 049 (relocatable akh), ADR 065 (context assembly runtime), ADR 070 (selectable chat backing), ADR 081/091 (memory schema, transcript capture), ADR 082 (executor `Backend` seam), ADR 105 (autonomous run governance)

## Context

The Mac hosts three akhs (janus, minerva, vesta, the "Roman" deployment under
`~/Library/Application Support/roman-akhs`). Their akhomed was built with
`--no-default-features --features core-server`. In that profile akhomed is a passive
cognitive tool server:

- Every path that starts the `AgentDaemon` is compiled out
  (`#[cfg(not(feature = "core-server"))]` in `src/bin/akhomed.rs`). None of the 16
  background tasks ever runs: reflection, consolidation, sleep cycle, harvest,
  librarian, goal generation, rule inference, kg compaction and the rest.
- The akh's own turn runs outside akhomed, in an `akhrtd` "internal brokered"
  runtime. That runtime calls back into exactly 9 hardcoded tools
  (`core_service.rs::registry()`), so psyche, communicative intelligence (Phase 34),
  the thinking engine (Phase 41) and the ContextManager strategies (Phase 44) are
  all bypassed.
- The ACP `_akh/*` surface is compiled out as well.
- If both `core-server` and the full features are enabled, the stripped mode wins.

The live store shows what this means for memory (2026-09-24):

| Evidence | Value |
|---|---|
| Captured turns | 598 |
| Harvest watermark | 0 for all three akhs |
| Facts | 18,860, none retracted |
| Sleep, learned-method and competence tables | empty |
| Memory-schema registry | empty |
| Same predicate, three spellings | `next_check`, `next-check`, `next check` |
| Contradictions | `TemporalConflict` warnings on every status update; nothing supersedes them |

These akhs do not have lifeform capabilities. They are chat bots with a knowledge
graph.

The service split (`akhsessd`, `akhrtd`, `akhsbxd`, `akhexecd`, `akheventd`) was
meant to contain execution and native runtimes safely across the cluster. It was
not meant to strip cognition out of the akh. The `core-server` profile implemented
the split as if cognition were one more thing to contain.

## Decisions

1. **Where the akh lives.** An akh is always fully featured. The daemon loop, the
   in-process turn, the full tool registry and the ACP surface all run in akhomed on
   every deployment. The split exists only to contain execution and native
   runtimes. It never removes cognition.
2. **The akh's own turn** runs in akhomed: `run_agentic_turn` through the embedded
   runtime (Ollama cloud or Candle, per ADR 070), with psyche, communicative
   intelligence, thinking and context strategies live. `akhrtd` supervises only
   native runtimes (Codex, Claude, OpenCode). Those are workers the akh delegates to,
   not the akh itself.
3. **Side-effecting tools** (shell, file, edit, glob, grep, process, and network
   where contained) route through the ADR 082 `Backend` seam to `akhsbxd` /
   `akhexecd`. Sandboxes run in separate containers or pods anywhere on the cluster,
   placed per workspace.
   - **This applies to every caller**: full akhs, delegated native runtimes, and any
     other runtime. No deployed process executes workspace side effects in its own
     address space.
   - `InProcessBackend` remains for development and tests only.
4. **Admission belongs to the akh.** External stimuli land in the akh's own
   trigger/reactor system and ACP surface, so the akh schedules its own work. The
   stimuli include Anima and Agent Board events, alerts, schedules, T3 and ACP
   clients. `akheventd` becomes a thin ingress adapter that forwards into akhomed.
   `akhsessd` shrinks to brokering the native-runtime sessions the akh delegates.
5. **Store.** Mac-resident akhs stay on the standalone SQLite backing. Cluster akhs
   use akh-pg. Relocating a Mac akh to the cluster means an explicit export/import.
6. **Autonomy defaults.** All background tasks are on by default.
   - Tasks that need no LLM run freely: harvest, sleep dedup and pruning,
     consolidation, equivalence, rule inference, kg compaction.
   - Tasks that cost LLM tokens run behind a per-akh daily budget (tokens and turns)
     and report per-task telemetry.
   - The librarian stays gated on WI-298: it runs only when its inputs changed, and
     in batches. ADR 105 governance applies to anything the akh starts on its own.
7. **Cutover.** janus, minerva and vesta migrate in place: same identities, same
   SQLite store, and the 598 captured turns replayed through harvest on the first
   sleep. The akhrtd internal-runtime profiles are retired after verification.
8. **`core-server` is retired.** The `/v1` core API becomes an additive `core-api`
   feature in the default full build, and every `cfg(not(feature = "core-server"))`
   gate is removed. CI and deploys build exactly one akhomed profile, so a stripped
   akh cannot be built.
9. **Delegation grants.** When the akh delegates to a contained native runtime, it
   mints a per-delegation grant:
   - the grant names the cognitive tools the worker may call (read-only by default,
     plus `memory_write`);
   - the worker calls under its own principal;
   - every call is recorded as the delegate's provenance, not the akh's.
10. **Deployment glue.** The Mac-only Python glue for the Romans is ported into the
    repository and retired one script at a time as each replacement is verified. It
    lives in `roman-akhs/bin`: the laya controller, janus escalation, observer,
    vesta ops/actions, T3 activity and the mergiraf worker. Each script becomes an
    akh trigger, reactor or skill, or a thin in-repo `akheventd` adapter. Nothing
    deployed stays outside the repository (see `.claude/rules/deploy-from-main.md`).

## Consequences

- Lifeform behaviour becomes the default: memory improves itself, drives and goals
  operate, and the psyche shapes turns. Cost is bounded by per-akh budgets, not by
  compiling features out.
- The `akhomed` build no longer shrinks toward "cognition only". The wayfinder goal
  "`akhomed` no longer links provider adapters" is dropped for the akh's own
  provider (the embedded runtime). Native-runtime adapters still stay out of akhomed.
- Containment gets stronger than today. Side effects from every caller go through
  `akhsbxd`/`akhexecd`, whereas today the native runtimes' own tools execute
  wherever that runtime runs.
- The 9-tool fixed catalog becomes a per-delegation grant, so tool access is
  attributable and scoped.
- Two deployment topologies share one binary: Mac (SQLite) and cluster (akh-pg).
- Every deployed artifact is built from `origin/main`, per the repository rule.
- Exception, exam instruments only (2026-09-29, Lilith C4, ADR 116): `tool_surface = "session-only"` seals the `dojo-teacher` akh's tool registry to the MCP servers its ACP client pushes per session, so it can answer recorded dojo episodes without reaching live state; working akhs stay fully featured — see [the design note](../plans/2026-09-29-lilith-c4-teacher-via-akh.md).
