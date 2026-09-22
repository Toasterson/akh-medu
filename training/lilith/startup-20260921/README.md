# Lilith startup — 2026-09-21

This is local preparation and diagnostic evidence, not a quality baseline or
release certification. The capability-contract ticket remains in progress.


## Follow-up: independent stack and resolved inference timeout

The user deferred native/paired validation until host setup is ready and selected
illumos VMs on this machine. Ptah may review Lilith and act as the local reference
on the dedicated stack without simultaneous model residency.

The original diagnostic used the legacy `AKH_GPU_MATMUL=1` offload path. A
one-token reproduction took 6.512 s for decode with 2,811 CPU/GPU transfers.
Changing only `AKH_GPU_MATMUL` to 0 reduced that decode measurement to 0.031 s.
The original 16-token prompt then finished in 14.35 s including load, with
0.531 s decode (~30.1 tokens/s). Generated text differed, so numerical equivalence
is not claimed. No inference-code change was needed; the startup recipe was corrected.

A fresh `candle_session_probe` build (`--release --locked --features hip`) from
checkout `329ebff421d2eff75ef12f29001baac20ff01ef3`, with explicit HIP and both legacy
offload flags disabled, generated the same greeting twice and stopped normally:
7 decode steps per reply, about 29.6–29.8 decode steps/s. These short diagnostics
are not workload or quality benchmarks. See [probe identity/result](gemma-current-probe.json),
[probe log](gemma-current-probe.log), and [original repro recovery](gemma-original-no-offload.json).

The independent stack is prepared outside the repository at
`/home/toasty/.local/share/akh-medu/lilith-dev-20260921`. Four configuration checks
passed; all four daemons reported their project identities in the
[startup lifecycle check](stack-health.json), then were stopped. That check did
not start existing services, submit model jobs, or allocate VMs. An initial
throwaway probe mistakenly used `/health` instead of `/healthz`; its failure is
retained in `stack-startup.json` and is a diagnostic-script error.

[Preparation and VM boundary](../../../examples/lilith/README.md) are documented.
Four generator tests cover private credentials, overwrite refusal, reference
hash mismatch, and separate model identities/grants. Service health checks used
existing binaries; their source revisions are not established by the new probe build.

Ptah's local 16,810,714,496-byte GGUF matched SHA-256
`3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`; its tokenizer
matched `e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea`.
Tokenizer provenance names `JonathanColetti/Qwen3.8-27B-Uncensored`. Label that
derivative in comparisons; upstream-reference equivalence is unproven. The local
stack uses direct Candle profiles rather than the inactive HTTP endpoint.

The runtime source already serializes inference and unloads the previous model
on identity change. Actual Gemma/Ptah switching and review sessions remain to be
exercised. VM provisioning and native/paired validation remain deferred.

## Machine and artifacts

- Host: `hephaistos`, Linux x86-64, Intel i7-6700K (4 cores / 8 threads).
- RAM: about 31 GiB reported by `free`; about 18 GiB available at inspection.
- AMD device `744c`, about 24 GiB VRAM; ROCm 6.3.4 installation and render/KFD
  device nodes present. Actual inference device placement needs confirmation.
- Available storage: about 482 GiB on the workspace filesystem at inspection.
- Gemma GGUF, tokenizer, and existing diagnostic binary SHA-256/size identities
  are in [inventory.json](inventory.json). Binary build revision is unknown.
- Ollama is installed but no server answered on localhost:11434.
- A Ptah configuration refers to `ptah-qwen3.8:sharp` at localhost:18080/v1;
  the endpoint refused connections. Neither alias nor tokenizer proves reference
  checkpoint identity. Qwen 3.8 27B is not yet pinned.
- The default `virsh list --all` returned no domains. This does not establish
  that no remote/system illumos target exists.

## Checks and limitations

`cargo test -p harness-core --locked` passed: 21 unit tests, one synthetic
pipeline integration test, and four illumos task-set validation tests. These
test the harness, not the models or native illumos behavior.

The bounded Gemma diagnostic records its outcome in
[gemma-smoke.json](gemma-smoke.json) and [gemma-smoke.log](gemma-smoke.log).
It requested 16 generated tokens with a 120-second total timeout, no paid
requests, and no training. It loaded the Gemma4-family artifact in 35.7 seconds, but did not complete the
16-token diagnostic within 120 seconds and was terminated. No throughput or
answer-quality score is available.
Use the recorded final outcome rather than interpreting load success as usable
inference or coding quality.

The [draft contract](../../../benchmarks/lilith/contract.json) records comparison
tracks, content hashes of exposed development fixtures, and unresolved gates.
The [illumos audit](../../../docs/ai/research/2026-09-21-lilith-illumos-benchmark-inventory.md)
identifies real historical runs and the isolation/grading changes needed before
new quality measurements. Existing results do not cover the requested Qwen model.

## Unexpected daemon side effect

Invoking the existing `target/release/akhomed --help` started the configured Ptah
daemon instead of showing help. It hydrated Ptah from its existing backing store,
ran background causal/schema work, and opened a local ACP socket. The newly
started process was sent SIGTERM; its log reports the final snapshot persisted,
WAL drained, and process exit. No direct database query was made. Avoid running
this binary for CLI discovery; inspect source and use isolated configuration for
future Lilith startup. This was not an isolated Lilith deployment.

## Initial prerequisites (historical; follow-up supersedes scheduling)

1. Restore Anima MCP credentials: the installed `anima-pat status` reports no
   stored PAT, and Claude's configuration uses that same helper. Never substitute
   the runner gateway token.
2. Identify/reach the Qwen 3.8 27B reference and pin its actual artifact/config.
3. Identify the native illumos target and its snapshot/reset mechanism.
4. Diagnose the bounded local inference result and build a source-pinned runner.
5. Repair the task isolation/verifier issues, select independent task splits,
   and freeze statistical tolerances/sample sizes before quality tuning.

No Qwen comparison, native illumos trial, adapter training, or release claim has
been made. No new Anima work items were created without tracker access.
