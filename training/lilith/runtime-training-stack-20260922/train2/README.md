# Lilith managed training run 2

**Intentionally stopped at the user’s request on 2026-09-22.** All dedicated
services exited and GPU memory was released. Zero optimizer steps, zero completed
episodes, no adapter. See `operator-stop.json`; stale worker progress is historical.

Restarted after diagnosing run 1's system cancellation. See `result.json` for
stack state; `worker/progress.json` for generation phase and token count when
available; `worker/result.json` for learning results. Token progress is written
every 128 generated tokens, without raw reasoning text. Prefill may take time.

The dedicated session watchdog is 259,320,000 ms (72 hours plus two minutes),
covering the runtime's 259,200-second job deadline. Training telemetry does not
currently advance the session's durable-progress clock, so this is a bounded
configuration mitigation, not a new heartbeat implementation. `events.jsonl`
preserves durable session events through terminal status and cleanup.

Workspace setup reads before creating/resetting the artifact. Replacements
pass the observed revision. The inference and training write-file catalogs both
expose expected_revision; model-generated invalid calls remain actual failures.

Validation: 10 stack/worker tests and 10 preparation tests pass. The actual
`../fixture6/` stack completed five broker calls covering missing read, create,
existing read, revision-checked reset and readback. `../fixture5/` preserves the
previous failing real-daemon reset test. The qualified numerical engine is
unchanged. No completed optimizer update or trained adapter is claimed at launch.

Protocol remains baseline, harvested-transcript SFT, three online reward banks,
development selection, paired held-out evaluation and retention. Context 32,768;
response ceiling 16,384; normal reasoning enabled. Cold-reload verification is
a follow-up after training completes. No release or deployed-weight replacement.
