# Lilith local baseline workspace

Primary deployment OS: illumos. Initial preparation host: Linux `hephaistos`.

`contract.json` records the candidate inventory, comparison tracks, content hashes
of existing development fixtures, and unresolved release requirements. It is a
draft; null reference identities, statistical margins, and sealed packs must not
be interpreted as passing gates.

Startup evidence lives in `training/lilith/startup-20260921/`. These records are
model/runtime diagnostics, not benchmark scores. The existing binary's build
revision is unknown; its SHA-256 is recorded. A reproducible quality baseline
requires a source-pinned build as well as the model/tokenizer hashes.

Verified local harness check:

```sh
cargo test -p harness-core --locked
```

Bounded native Gemma diagnostic (uses the existing GPU-enabled example binary):

```sh
timeout 120 env AKH_LLM_MODEL=gemma4-12b AKH_GPU_MATMUL=0 AKH_GPU_QUANT=0 \
  AKH_BENCH_TOKENS=16 \
  AKH_BENCH_PROMPT='Explain why an operating system service can enter a maintenance state.' \
  target/release/examples/candle_bench
```

Do not use `just bench` for Lilith scores: it currently records synthetic runs.
The existing memory runner defaults to answer-bearing-session oracle retrieval,
and the repository patch harness uses gold-patch-guided source selection. Label
these assistance profiles explicitly; neither establishes unaided retrieval or
repository understanding. Existing public illumos fixtures are development data.

Next runnable comparison needs a reachable, pinned Qwen 3.8 27B reference and an
identified illumos test target with clean per-task resets. The discovered local
Ptah endpoint is unavailable and its alias alone does not establish the required
reference identity. Do not substitute historical Qwen 3.5 397B results.

Update: local inference timeout is resolved by disabling legacy offload; see the startup report. Ptah model artifacts were found and hash-verified; the independent stack uses direct Candle profiles rather than the unavailable HTTP endpoint. Native/paired validation is deferred by the user.
