"""Lilith Chief-of-Staff dojo (Anima WI 01a0e270-475e, plan slice C4, ADR 116).

CPU-only, stdlib-only harness over `lilith/cos-snapshot@1` roots written by
`scripts/lilith/cos_recorder.py`:

- `snapshot`       read-only loader with object-integrity checks
- `episode`        the runner interface: `EpisodeSpec`, `ToolReceipt`, `StepRecord`, `Episode`
- `replay_server`  MCP JSON-RPC replay of snapshot t_k + the dojo tenancy ledger
- `hindsight`      ground-truth labels from t_k .. t_{k+n}, each with provenance
- `families`       the eleven episode families: spec builders + deterministic verifiers
- `reward`         the `cos-v1` reduction and sibling-contrast pairs
- `splits`         time-window / Anima-Plan splits + P6.3-compatible decontamination
- `floors`         the frozen `exam-floors-v1.json`
- `runner`         model-driven `EpisodeRunner`: prompt, broker, answer parsing
- `model_adapters` OpenAI-compatible and Anthropic chat adapters (same tools, same messages)
- `baseline`       batch/resume driver, baseline summary, paired `compare`
- `dojo_report`    how many episodes the snapshots recorded so far support

Modules import each other as the top-level package `dojo`; entry points put
`scripts/lilith` on `sys.path` first.
"""

DOJO_VERSION = "lilith/cos-dojo@1"
