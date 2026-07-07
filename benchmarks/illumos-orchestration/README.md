# illumos orchestration task corpus (Phase 50 / W6)

Companion corpus to [`benchmarks/illumos-e2e/`](../illumos-e2e/README.md),
purpose-built for the **fleet-orchestration benchmark**
(`examples/phase50_w6_orchestration.rs`): does splitting a decomposable
illumos task across an orchestrator akh + N worker akhs beat a single akh
doing the whole thing, on the live `exec1` guest?

Same 4-file task shape as `illumos-e2e`
(`meta.toml`, `setup.sh`, `prompt.md`, `verify.sh`) — see that README for
the full grading-contract writeup. This directory's tasks additionally
carry a `family = "orchestration"` and a machine-readable `[[subtask]]`
list in `meta.toml`, because these tasks are specifically constructed to
**decompose into N independent subtasks with zero shared state**, so a
harness can fan work out across worker akhs without them colliding.

## Grading contract

Identical to `illumos-e2e`'s autonomous mode: the composite `verify.sh`
is the only pass/fail signal, exit 0 = pass. The akh (whether one akh
doing the whole task, or N workers each doing one subtask) never sees
`verify.sh`. Subtask prompts in `meta.toml` are self-contained — a worker
handed one subtask prompt never sees the composite `prompt.md` or the
other subtasks' prompts, mirroring how an orchestrator's own decomposition
would hand off work.

## The tasks

| id | subtasks | description |
|---|---|---|
| `relay-group-3svc` | 3 | Three independent, differently-broken SMF services (`site/relay-{a,b,c}`) — broken start method, disabled, wrong config property — fix all three to `online`. |

One task is enough for the v1 harness; the harness itself is N-general so
more decomposable tasks can be added here later (each just needs distinct
FMRIs/paths per subtask so parallel workers never touch shared state).

## What the harness still has to do

Not implemented here — see `examples/phase50_w6_orchestration.rs`:

1. Run `setup.sh` as root on `exec1`, drive the **control** arm (one akh,
   one `run_agentic_turn` over the composite `prompt.md`), run
   `verify.sh`, record `{wall, tokens, requests}`.
2. Re-run `setup.sh` to reset state, drive the **orchestrated** arm: an
   orchestrator akh decomposes the task (or the harness falls back to
   `meta.toml`'s `[[subtask]]` list), then N worker akhs each run one
   subtask in parallel over its own `RemoteBackend` connection to
   `exec1`, then the composite `verify.sh` grades the end state.
3. Emit both arms as `TaskResult` rows in one `RunRecord` under harness id
   `fleet-orchestration`, with speedup / coordination-tax / redundant-
   command metrics in the orchestrated row's notes.
