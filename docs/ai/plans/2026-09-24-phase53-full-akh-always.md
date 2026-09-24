# Plan: Phase 53 — Every akh runs fully featured (retire `core-server`)

- **Date:** 2026-09-24
- **ADR:** [113](../decisions/113-full-akh-always.md)
- **Status:** Design locked (grilling session 2026-09-24); implementation pending
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, umbrella WI `01a0d43b-bf90-7630-bbc6-bd31d0609864` (draft; slice WIs filed when promoted)
- **Deploy rule:** `.claude/rules/deploy-from-main.md`. Every slice ships from `origin/main` only.

## Why

The Mac Romans (janus, minerva, vesta) run a `core-server` akhomed. That build has
no daemon loop, no in-process turn, 9 tools and no ACP. As a result, nothing ever
improves their memory: 598 captured turns sit unharvested, 18,860 facts are never
retracted, and the sleep, learned-method and competence tables are empty. ADR 113
reverses the profile: the akh is always full, and the service split contains
execution only.

## Slices

Order: 53a → {53b, 53c, 53d, 53e} → 53f → 53g → 53h. The cutover (53f) needs 53a,
53b and 53d. The Romans have no exec tools today, so they can cut over with a
deny-all `Backend` and gain tools once 53c lands.

### 53a — `core-api` becomes additive; one akhomed profile

- Replace the `core-server` feature with `core-api` (axum `/v1` routes). Add it to
  the default full build next to `daemon`, `acp` and `embedded-runtime`.
- Remove every `cfg(not(feature = "core-server"))` / `cfg(all(feature = "acp", not(feature = "core-server")))`
  gate in `src/bin/akhomed.rs`. The full daemon serves ACP and the `/v1` core API
  side by side, sharing one `Engine` per akh (no second engine).
- Keep the SQLite backing knob (`CoreConfig.sqlite`, `[database] url = "sqlite://…"`)
  in the full build.
- `/v1/tools/call` resolves against the akh's full `ToolRegistry`, filtered by the
  caller's grant (53e). The hardcoded `registry()` goes away.
- **Acceptance:** one `cargo build --release --bin akhomed` artifact starts the
  daemon ticks, answers ACP, and serves `/v1/context` and `/v1/tools/call`. The
  existing core-service tests pass against the full build.

### 53b — Budgeted autonomy defaults

- `[autonomy]` defaults: every task that needs no LLM is on (harvest, sleep dedup
  and pruning, consolidation, equivalence, rule inference, kg compaction).
- LLM-cost tasks (reflection-with-LLM, goal generation, knowledge extraction,
  librarian) share a per-akh daily budget (`max_tokens_per_day`,
  `max_turns_per_day`). When the budget is exhausted the task is skipped and
  records why.
- Per-task telemetry: runs, skips, duration, tokens, facts
  added/retracted/superseded (OTLP, Grafana).
- The librarian's default stays gated on WI-298 (change-only, batched).
- **Acceptance:** a fresh akh runs a sleep cycle whose harvest watermark advances.
  An exhausted budget produces skip telemetry, not errors.

### 53c — `ExecdBackend`: all side effects go through the sandbox

- A new ADR 082 `Backend` impl calls `akhsbxd` for placement and `akhexecd` for
  file, search, edit and process verbs, with receipts and cancellation. Placement
  is chosen per workspace (`[exec] backend = "execd"`, sandbox profile).
- **This applies to every caller.** The akh's own tools use it, and native
  runtimes (Codex, Claude, OpenCode) launched by `akhrtd` have their tool execution
  bound to an `akhsbxd` sandbox and `akhexecd`, never to the runtime host's address
  space. Deployments reject `InProcessBackend` at startup (dev/test only).
- Reconcile with open 48b (LocalDrop) and 48c (russh remote). Those become
  placement drivers behind `akhsbxd`, not alternatives to it.
- **Acceptance:** with `shell_exec` issued by an akh, and with a shell tool issued
  by a delegated Codex worker, both execute inside the bound sandbox. Both leave an
  `akhexecd` receipt that names the caller's principal.

### 53d — Admission into akhomed

- `akheventd` delivers typed events to akhomed: to its trigger/reactor system
  (`trigger_manage`, `reactor_manage`) or as ACP prompts. The akh decides whether
  and when to act.
- `akhsessd` keeps only native-runtime session brokering for delegations. Its
  automation and admission role for akhs is removed after cutover.
- ADR 105 governance: every self-started run carries its exit condition and
  reports board-routed events.
- **Acceptance:** an Agent Board message to minerva produces a turn started by
  minerva's reactor, with no akhsessd automation involved.

### 53e — Scoped delegation grants

- When the akh delegates to a native runtime, it mints a grant naming the
  principal, the allowed cognitive tools (default: read-only plus `memory_write`)
  and an expiry.
- `/v1/tools/call` enforces the grant. The akh records the delegate's calls with
  delegate provenance (`DerivationKind` carrying the delegate principal).
- **Acceptance:** a delegate calling a tool outside its grant is denied with a
  diagnostic, and the delegate's `memory_write` carries delegate provenance.

### 53f — Roman cutover in place

1. Build full akhomed from `origin/main` (`akhomed-main-<sha>`).
2. Point it at the existing `state/core/akh.sqlite3`. Keep identities, facts and
   captured turns.
3. First sleep: harvest the 598 captured turns, verified by the watermark
   advancing and episodes appearing.
4. Memory hygiene:
   - install a schema pack that marks status predicates (`observed state`,
     `next_check`) as single-valued, so a new value supersedes the old one instead
     of raising `TemporalConflict`;
   - merge the three `next_check` spellings;
   - re-run contradiction repair.
5. Point the Romans' own turns at the embedded runtime (glm-5.3 via Ollama cloud).
6. Retire the `ollama-*` akhrtd internal profiles once each Roman has passed a
   live turn with psyche/comm-intel telemetry, a daemon tick and a sleep cycle.
7. The old `dev.akh-medu.akhomed` (tecton/imhotep) plist stays disabled. It is
   out of scope here.

- **Acceptance:** for each Roman, within 24 h, the sleep-metrics, consolidation and
  harvest counters are non-zero. Anima and Board work continues without operator
  relay.

### 53g — Port the Mac glue into the repository

- Inventory `roman-akhs/bin/*.py` (laya controller/service, janus escalation,
  observer, vesta ops/actions, T3 activity, mergiraf worker, provider failure).
- Map each script to an akh trigger, reactor or skill, or to a thin in-repo
  `akheventd` adapter.
- Retire the scripts one at a time. Each retirement needs a verified replacement
  and a launchd plist removal.

### 53h — Remove the stripped profile and update docs

- Delete the `core-server` feature and `deploy/roman-mini` build instructions that
  use it. Build and deploy from main with the full profile.
- Update `docs/ai/architecture.md` and add superseded notes to
  `docs/wayfinder/agent-loops/implementation.md` and `assets/daemon-boundaries.md`
  §1.
- Update the CLAUDE.md roadmap.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Ungated consolidation degrades memory (WI-268) | Librarian gated on WI-298; telemetry records facts superseded per task; budget caps |
| Token cost of autonomy | Per-akh daily budgets; skips are recorded, not silent |
| Unbounded self-started work (ADR 105 drift) | Exit conditions plus board-routed events on every self-started run |
| Replaying 598 turns floods the KG | Harvest batch cap (`harvest_daily_log` batch size), watermark resume |
| Fact loss during in-place migration | Stop the core, copy the SQLite store under `backups/`, cut over, keep the old binary and plists until acceptance passes |
