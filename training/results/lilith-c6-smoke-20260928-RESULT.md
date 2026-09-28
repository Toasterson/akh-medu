# Lilith C6 — Runpod H100 smoke, rounds 1, 2 and 3 (2026-09-28) — RESULT

Anima WI `01a0e270-6021` (C6 Runpod CUDA harness); plan `docs/ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md` (C6 row), ADR 116.

Sources are all on the supervisor Mac under `~/lilith-private/smoke/runs/`:
- Round 1: `c6-smoke-20260928/{smoke-result.json, steps.tsv, smoke.log, cache-control.json, provenance.json, bootstrap-check.json, gpu.csv, bootstrap-20260928.log}`
- Round 2: `c6-smoke-r2-20260928/{smoke-result.json, steps.tsv, smoke.log, cache-control.json, cache-diagnostic.json, memory-probe.json, microrun.json, microrun.generation.json, cold-reload.json, provenance.json, bootstrap-check.json, gpu.csv}`
- Round 3: `c6-smoke-r3-20260928/{smoke-result.json, steps.tsv, smoke.log, cache-control.json, vllm-generation.json, provenance.json, bootstrap-check.json, gpu.csv, smoke.marker, SHA256SUMS, results-c6-smoke-r3-20260928.tar.gz}`
- Watchdog logs: `~/lilith-private/smoke/watchdog-c6-r2.log`, `~/lilith-private/smoke/round3.log`

Costs and the round-1 idle incident come from the supervisor's Runpod billing readout and its incident note in the round-3 request (2026-09-28). Round 3's cost and its aborted-launch incident come from the same supervisor readout, quoted directly in the task that produced this update. No figure below comes from anywhere else.

## Verdict

**C6 exit condition met, except the live tunnel episode (not attempted; blocked on C6a).** Round 3 confirms round 2's cause class on the control's exact tokenwise path and closes the gap: with numerics `sdpa_backend=efficient` + `sdpa_math_fallback=1`, the resident-vs-offloaded KV cache control **passed** (`cache-control.json`, `status: "passed"`):
- Logit max_abs 0.0 / 0.0 / 0.0 at positions 1023 / 1024 / 1025 (tolerance 1e-4).
- Mean selected-token loss identical between resident and synchronous-offload variants (22.817245483398438 both; `loss_abs` 0.0).
- LoRA A/B gradient max_abs 1.4202669262886047e-08 / 1.5046680346131325e-07 (tolerance 1e-4).
- Generate→replay sampler logp drift 5.545063483936019e-07 (tolerance 1e-4), policy sha identical before/after.

This confirms round 2's diagnostic finding: the default CUDA SDPA dispatch (cuDNN attention on H100, torch 2.13.0+cu130) is run-to-run nondeterministic; forcing EFFICIENT attention with a MATH fallback (for the head_dim-512 global layers, which EFFICIENT has no kernel for) restores bit-exact resident-vs-offload equivalence. `sdpa_backend=efficient`/`sdpa_math_fallback=1` is now the recorded default for CUDA in every training path (this update's code change; see below), not just an operator-supplied flag.

vLLM also passed in round 3 (the round-3 PR's ninja-on-PATH fix): backend `vllm`, `lora_applied: true`, 79.26101721548619 tok/s generating 256 tokens from a 40-token prompt in 3.23 s (`vllm-generation.json`) — up from round 2's HF fallback at 12.65 tok/s (a 6.3x throughput ratio).

Round 1 and round 2's other findings are unchanged and summarized below; round 3 ran only `cache-control`, `cache-diagnostic` (not needed — control passed) and `vllm-generation` (via `--only`), reusing round 2's memory-probe, micro-run and cold-reload evidence rather than re-measuring them.

## Configuration

| Setting | Value | Source |
|---|---|---|
| GPU | 1× NVIDIA H100 80GB HBM3, driver 580.159.03, 81559 MiB (round 3 UUID `GPU-6412b2c8-70ef-126f-04b2-5e88b7b3a2b0`) | `gpu.csv` (all rounds) |
| Pod (round 3) | `3hieq1f9ejmfd1`, terminated 2026-09-28T12:53:52Z (HTTP 204) | `round3.log` |
| Pod (round 2) | `ng3c3t99z6o77a`, terminated 2026-09-28T12:02:01Z (HTTP 204) | `watchdog-c6-r2.log` |
| Pod (round 1) | id not in the retained evidence | — |
| Code | round 1 main `cec55065f7bf83c19f863e96e81755b7688ee6a5`; round 2 main `8f2d889a1fd07f0180f6fd745c1f435ace52a78f`; round 3 main `6a8dd50f1287d8a7e5258242993a62617ba98005`; provenance `verified`, no problems every round | `provenance.json`, `smoke-result.json` |
| Env (round 3) | Python 3.12.3, torch 2.13.0+cu130 (CUDA 13.0), transformers 5.10.4, peft 0.21.0, trl 1.14.0, bitsandbytes 0.50.2, accelerate 1.15.0, vllm 0.30.0, flashinfer-python 0.6.18.post1, triton 3.7.1, nvidia-cuda-runtime 13.0.96, ninja 1.13.2 (now on PATH) | `bootstrap-check.json`, `cache-control.json` (`versions`), `smoke.log` (readiness lines) |
| Env lock (round 3) | `requirements-cuda.txt` sha256 `c96267199f9b286e5e9020aed025d0ceb14f041ca0a49a47ba068d7d3eaa5e44` | `provenance.json` (`files`) |
| Env lock (round 1/2) | `requirements-cuda.txt` sha256 `733d84e137b4f3387f15a3558f486a5b9e98ddd250cce6d40c462822a489d36c`, built from payload `src-ccc915e1` | `bootstrap-20260928.log` |
| Parent | `google/gemma-4-12B-it` @ `5926caa4ec0cac5cbfadaf4077420520de1d5205`; 48 layers (40 sliding, head_dim 256, window 1024; 8 full, head_dim 512); dense (no MoE) | `bootstrap-check.json`, `cache-diagnostic.json` (`model`) |
| Engine | `lilith_training_engine.py` sha256 `5c68f6a1d451c87b6d77e3a54c2786305208a35b5b366ad8661a9ca0bbcb73d0` (qualified, unchanged, every round) | `cache-control.json` |
| Attention / dtype | sdpa, bf16 | `cache-control.json` (`runtime`) |
| Numerics (round 1/2) | default (untouched PyTorch). At diagnostic start: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, matmul TF32 off, cuDNN TF32 on, bf16 reduced-precision reduction on, deterministic off, all four SDPA backends enabled | `cache-control.json` (`numerics`), `cache-diagnostic.json` (`numerics_at_start`) |
| Numerics (round 3) | `sdpa_backend=efficient`, `sdpa_math_fallback=1` (label `sdpa=efficient+math`); observed: SDPA efficient/math enabled, flash/cuDNN disabled, matmul TF32 off, cuDNN TF32 on, bf16 reduced-precision reduction on, deterministic off | `cache-control.json` (`numerics`) |
| Data | `sft-subset-v1.jsonl` sha256 `e662d28289165d8cc419a4b65bba63d7eb88b414e5f7ce63aff62c7a3028dd94`, 8 records (pinned identically in round 3) | `smoke.log` (data-pin, all rounds), `microrun.json` (`data`) |
| Budget | 7200 s global every round; cache diagnostic 480 s passes / 600 s step cap in round 2, 1800 s budget in round 3 (not exercised: control passed) | `smoke-result.json`, `smoke.log` |

## Timeline (UTC)

| Time | Event | Source |
|---|---|---|
| 07:58:20 | Round 1 `c6-smoke-20260928` starts | round-1 `smoke.log` |
| +249 s | cache-control fails equivalence; run ends, rc=1, wall 276.2 s. The pre-#440 harness stopped at the first failure. | round-1 `steps.tsv`, `smoke-result.json` |
| after round 1 | The pod idled for about 2 h 14 min: the supervisor's `pgrep -f` watcher matched its own command line and never saw the run end | supervisor incident note |
| 10:43:25 | Round 2 `c6-smoke-r2-20260928` starts (main `8f2d889a`, PR #440 harness) | round-2 `smoke.log` |
| +27 s … +267 s | bootstrap-check ok (27 s); cache-control fails equivalence (240 s) | round-2 `steps.tsv` |
| +267 s … +867 s | cache-diagnostic runs 600 s and is killed by the step cap (`timeout`, exit 124). The last persisted report is at 310.4 s elapsed with 33 passes. | `steps.tsv`, `cache-diagnostic.json` |
| +867 s … +1010 s | memory-probe ok (143 s) | `steps.tsv` |
| +1010 s … +4500 s | micro-run ok (3490 s; `wall_clock_seconds` 3485.8). vLLM fails at 11:57:43 inside its generation phase; HF fallback runs. | `steps.tsv`, `microrun.json`, `smoke.log` |
| +4500 s … +4562 s | cold-reload passed (62 s); run ends rc=1, wall 4562.6 s | `steps.tsv`, `smoke-result.json` |
| 12:01:38 | Watchdog sees `smoke-result.json` (state=done) | `watchdog-c6-r2.log` |
| 12:01:57 / 12:02:01 | Run dir collected to the Mac; pod `ng3c3t99z6o77a` terminated (HTTP 204) | `watchdog-c6-r2.log` |
| 12:33:15 / 12:33:18 | EU-FR-1 H100 stock checked (low); round-3 pod `3hieq1f9ejmfd1` created | `round3.log` |
| 12:33:44 … 12:38:42 | Bootstrap runs on the fresh pod: `provenance: verified`, an old env from a different lock is found and moved aside (`moved-old-env-aside`), the env is rebuilt from the round-3 lock — the aborted-launch cost in this update's cost table is this bootstrap, run once before the smoke below started clean | `round3.log` |
| 12:38:42 | Round 3 `c6-smoke-r3-20260928` starts, `--only cache-control,cache-diagnostic,vllm-generation` (gates and data-pin always run) | round-3 `smoke.log` |
| +1 s … +26 s | provenance ok (1 s); bootstrap-check ok (25 s), readiness confirms driver 580.159.03, torch 2.13.0+cu130 CUDA available, ninja resolved to `$ROOT/env/bin/ninja` | round-3 `steps.tsv`, `smoke.log` |
| +26 s | gpu ok; data-pin ok (subset sha `e662d282...` pinned) | round-3 `steps.tsv`, `smoke.log` |
| +26 s … +244 s | cache-control **passed** (218 s) with numerics `sdpa=efficient+math`; cache-diagnostic not run (not needed — control passed, so the `--only` condition for it does not trigger) | round-3 `steps.tsv`, `cache-control.json` |
| +244 s … +753 s | vllm-generation **completed** (509 s step wall, 478.8 s script-internal wall): backend vllm, lora_applied, 256 tokens in 3.23 s at 79.26 tok/s | round-3 `steps.tsv`, `vllm-generation.json` |
| 12:53:48 | Watchdog sees `smoke-result.json` (state=done) | `round3.log` |
| 12:53:50 / 12:53:52 | Run dir collected to the Mac; pod `3hieq1f9ejmfd1` terminated (HTTP 204) | `round3.log` |

## Steps

| Step | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| provenance | ok (0 s) | ok (0 s) | ok (1 s) |
| bootstrap-check | ok (27 s) | ok (27 s) | ok (25 s) |
| gpu | ok (0 s) | ok (0 s) | ok (0 s) |
| data-pin | ok (0 s) | ok (0 s) | ok (0 s) |
| cache-control | **failed** (249 s, `failed_equivalence`) | **failed** (240 s, `failed_equivalence`) | **passed** (218 s), numerics `sdpa=efficient+math` |
| cache-diagnostic | (did not exist) | **timeout** (600 s, exit 124; no verdict written) | not run (`--only` names it, but it only runs when cache-control is not selected or did not pass; control passed) |
| memory-probe | not run (harness stopped at first failure) | ok (143 s) | not selected (`--only` reused round 2's evidence) |
| microrun | not run | ok (3490 s) | not selected |
| cold-reload | not run | ok (62 s) | not selected |
| vllm-generation | (did not exist) | not a separate step (measured inside microrun; HF fallback, 12.65 tok/s) | **ok** (509 s), backend vllm, 79.26 tok/s |
| overall | failed, wall 276.2 s | failed (`failed_steps`: cache-control, cache-diagnostic), wall 4562.6 s | **passed**, wall 753.8 s |

## Memory and throughput (round 2; round 3 reused this evidence via `--only`)

| Context | Memory probe: peak allocated / reserved (bytes) | Headroom | Steady train tok/s (first step incl. warmup) | Micro-run train tok/s | Micro-run seconds | Micro-run peak reserved |
|---|---|---:|---|---:|---:|---:|
| 8192 | 34,596,724,736 / 35,798,384,640 | 0.579 | 1731.8 (1314.3) | 1723.6 | 955.5 | 36,071,014,400 |
| 16384 | 39,777,006,080 / 40,896,561,152 | 0.519 | 1564.1 (1559.9) | 1562.5 | 2101.1 | 41,164,996,608 |
| 32768 | 54,769,545,728 / 56,537,120,768 | 0.335 | 1276.0 (1281.6) | — (optional context, not in the reference micro-run) | — | — |

Source: `memory-probe.json`, `microrun.json`. Every context `fits` and meets the 10% headroom rule; there are no OOM contexts.

Micro-run (`is_reference: true`, no deviations, rank 16, bf16 base, LR 1e-4, seed 73):
- **8K SFT loss:** 1.918 at step 0, 8.68e-06 at step 199. Preference loss 6.26e-07, margin 142.84.
- **16K SFT loss:** 1.011 at step 0, 1.74e-04 at step 199. Preference loss 7.15e-07, margin 141.52.

The adapter memorizes the 8 records. This is a mechanics/throughput reference, not a quality claim (`quality_claim: false`).

Cold reload of the 8K adapter in a fresh process: `passed`, `probe_sha256_identical: true`, `probe_sample_max_abs: 0.0`, argmax identical, 0 optimizer steps, 56.7 s (`cold-reload.json`).

Generation (`microrun.generation.json`): vLLM failed. HF generated 256 tokens in 20.23 s (**12.65 tok/s**) from a 40-token prompt with LoRA applied; generation wall time was 343.1 s.

## Cache control (resident vs synchronous offload, positions 1023 / 1024 / 1025, tolerance 1e-4)

| Measure | Round 1 (default numerics) | Round 2 (default numerics) | Round 3 (`sdpa=efficient+math`) |
|---|---|---|---|
| logit max_abs | **3.8125 / 0.875 / 0.5625** | 3.65625 / 0.9375 / 0.75 | **0.0 / 0.0 / 0.0** |
| mean selected-token loss: resident / offload | 24.8347 / 24.8764 | 24.3317 / 23.9018 | 22.817245483398438 / 22.817245483398438 |
| loss diff | 0.04171 | 0.42989 | 0.0 |
| LoRA A grad max_abs | 1.238e-04 | 3.697e-04 | 1.4202669262886047e-08 |
| LoRA B grad max_abs | 4.284e-04 | 5.786e-04 | 1.5046680346131325e-07 |
| generate→replay sampler logp drift | 1.948e-07 (4 tokens, policy sha bound) | 1.948e-07 | 5.545063483936019e-07 (4 tokens) |
| full-bf16 diagnostic max_abs (diagnostic only) | 19.53 / 6.375 / 39.56 | 1.781 / 0.625 / 39.69 | 18.3125 / 3.3125 / 39.0 |
| peak GPU bytes | 24,436,299,264 | 24,436,299,264 | 24,436,299,264 |
| status | `failed_equivalence` | `failed_equivalence` | **`passed`** |

Round 2 repeats round 1's inputs, weights and code path exactly, yet the numbers differ, including the full-forward diagnostic at 1023 (19.53 → 1.78). That is already evidence of nondeterminism. On the local ROCm stack the same control matched to 2.9e-11 (round-1 brief). Round 3 repeats the same inputs, weights and code path a third time, this time under `sdpa_backend=efficient`/`sdpa_math_fallback=1`, and every hard-tolerance measure (logits, loss, gradients, sampler drift) is bit-identical or within float noise of zero — the diagnostic-only full-bf16 comparison at 1023/1024 still differs from the cached-path resident value (18.3/3.3 vs the qualified engine's approximation), consistent with rounds 1/2 and flagged there as a separate, expected cached-vs-full-forward semantics gap, not part of the pass/fail tolerance.

### Cache diagnostic (round 2, partial: 33 passes in 310.4 s, then killed by the 600 s cap)

The chunked mode prefilled 990 tokens and then ran 32 tokenwise steps. Logit max_abs at 1023 / 1024 / 1025 (`cache-diagnostic.json`, `modes_summary`):

| Variation | Resident vs offload | Resident vs resident repeat | Restores equivalence |
|---|---|---|---|
| e baseline (chunked) | 0.8125 / 0.5 / 0.5 | 0.9375 / 0.625 / 0.5 (offload repeat 0.96875 / 0.625 / 0.5) | — |
| e baseline (**tokenwise**) | not reached | **1.0 / 0.5 / 0.578125** (loss 0.3108; grads 1.83e-4 / 3.30e-4) | — |
| a TF32 off | 0.875 / 0.5 / 0.8125 | 0.75 / 0.5 / 0.625 | no |
| a2 bf16 reduced-precision reduction off | 0.75 / 0.5 / 0.5 | 0.6875 / 0.375 / 0.5625 | no |
| c contiguous storage | 1.09375 / 0.375 / 0.5 | 1.0 / 0.4375 / 0.375 | no |
| d deterministic algorithms | **0 / 0 / 0** | 0 / 0 / 0 | yes (chunked only) |
| b SDPA MATH | **0 / 0 / 0** | 0 / 0 / 0 | yes (chunked only) |
| b SDPA EFFICIENT | **0 / 0 / 0** | 0 / 0 / 0 | yes (chunked only) |
| b SDPA FLASH | `No available kernel` (head_dim 512) → with MATH fallback **0 / 0 / 0** | 0 / 0 / 0 | yes (chunked only) |
| b SDPA cuDNN | `No available kernel` → with MATH fallback 3.375 / 0.8125 / 0.4375 | 0.625 / 0.4375 / 0.375 | no |

Verdict recomputed offline from the saved report (`lilith_cache_path_diagnostic.py --from-report`, round-3 PR): cause `run_to_run_nondeterminism`; restoring variations on the chunked path only: deterministic, sdpa_math, sdpa_flash_math_fallback, sdpa_efficient; `recommended_numerics: null` until a tokenwise pair confirms one. **Round 3 is that tokenwise confirmation**: `sdpa_efficient` (with the MATH fallback the head_dim-512 global layers need) passed the control's exact tokenwise path bit-identically (see the Cache control table above), so `recommended_numerics` is no longer null in practice — it is `sdpa_backend=efficient`, `sdpa_math_fallback=1`, and this update makes it the recorded default on CUDA in every training path (`scripts/session-platform/lilith_numerics.py:cuda_default`/`platform_default`).

Per-pass cost: chunked passes took 2.0–4.8 s, except the deterministic offload pass (23.6 s) and the contiguous offload pass (23.2 s). Tokenwise resident passes took 80.1 s and 52.1 s, and the tokenwise deterministic resident pass 58.2 s. The run was killed during the tokenwise deterministic **offload** pass, whose cost the planner under-estimated: its estimate was per mode, not per case.

## vLLM: round 1/2 failure, round 3 fix confirmed

vLLM 0.30 got through weight load, CUDA-graph capture and JIT kernel warmup. The sampler warmup then called flashinfer `top_k_top_p_sampling_from_logits`, which JIT-builds its sampling module by running `ninja`. That raised `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'` (`smoke.log` lines 213 and 317), and the engine core exited with `RuntimeError: Engine core initialization failed`.

The locked env already contains `ninja==1.13.2`, pulled in by flashinfer. However, the smoke runs `$ROOT/env/bin/python` without activating the env, so `env/bin` (where the `ninja` console script lives) was not on PATH.

The round-3 fix:
- pins `ninja` in `requirements-cuda.in`;
- puts `env/bin` first on PATH in `run_smoke.sh`, `bootstrap.sh` and `reference_microrun.py`;
- adds a bootstrap readiness check that the env's own `ninja --version` works;
- adds a strict `vllm-generation` step with no HF fallback.

**Round 3 confirms the fix.** `smoke.log`'s readiness line resolves `ninja 1.13.2.git.kitware.jobserver-pipe-1` at `/workspace/lilith/env/bin/ninja`, and the strict `vllm-generation` step (no HF fallback available) completed (`vllm-generation.json`):

| Measure | Round 2 (HF fallback) | Round 3 (vLLM) |
|---|---:|---:|
| backend | hf | vllm |
| generated tokens | 256 | 256 |
| prompt tokens | 40 | 40 |
| generation seconds | 20.23 | 3.2298349048942327 |
| tokens/second | 12.65 | 79.26101721548619 |
| LoRA applied | true | true |
| wall clock | 343.1 s generation-only wall, inside microrun (`microrun.generation.json`) | 509 s bash step wall incl. interpreter startup (`steps.tsv`); 478.8 s script-internal wall from `Report` init, i.e. model/engine load plus generation (`vllm-generation.json`) |

vLLM is **6.3x** the HF fallback's throughput (79.26 / 12.65) on the same prompt, adapter and token count.

## Incidents

| Incident | Detail | Fix / resolution |
|---|---|---|
| Round-1 pod idle (~2 h 14 min) | The supervisor's `pgrep -f` watcher matched its own command line, so it never saw the run end, and the pod stayed up idle | `scripts/lilith/pod_watchdog.sh` (main `b9944413`, `f64ca951`, PR #441): one ssh probe per 15 min, no pgrep/pkill, keyed on `smoke-result.json` / pid liveness / file progress / GPU utilisation; terminates on done, dead, stalled or unreachable. Round 2 used it and terminated 23 s after `done`. |
| Round 1 stopped at the first failure | The pre-#440 harness exited on the failed cache control, so memory, micro-run and cold reload never ran | PR #440: gates stop the run, other steps continue, `failed_steps` lists all failures, and `smoke-result.json` is written last as the completion marker |
| Round-1 `tar: file changed as we read it` | The tarball was written inside the run dir it was packing | PR #440: the tarball is built outside the run dir |
| Diagnostic killed without a verdict | 480 s pass budget and 600 s step cap; the verdict was written only at the end; per-mode cost estimates let a slow deterministic offload tokenwise pass start | Round-3 PR: provisional verdict persisted after every case, per-case cost estimates, in-pass deadline, `--from-report`, 1800 s default budget |
| vLLM `ninja` not found | See above | Round-3 PR; confirmed fixed in round 3 (see vLLM section) |
| Round-3 aborted launch: bootstrap refused a complete env built from the old lock | The round-3 pod's network volume carried an env from an earlier payload; that env is complete (has python and a lock marker) but was built from a different `requirements-cuda.txt`, so `bootstrap.sh` refuses it (`lilith::bootstrap::env_lock`) rather than silently reusing it. The supervisor moved it aside by hand per the `die` hint and reran bootstrap, which cost part of round 3's $1.43 (`round3.log`: `moved-old-env-aside`) | This update documents the failure mode and its fix command directly in `bootstrap.sh`'s header comment, so an operator checks `bootstrap.sh --check` for `env_lock` and moves the env aside *before* launching the timed smoke, rather than discovering it mid-run |

## Cost

| Item | Cost (USD) |
|---|---:|
| Round 1 pod (including the idle period) | $8.76 |
| Readout | $0.25 |
| Volume fill | $1.07 |
| Round 2 pod | $4.52 |
| Round 3 pod (incl. one aborted bootstrap on a stale env) | $1.43 |
| **Total** | **$16.03** |

Source: the supervisor's Runpod billing readout (round-3 request for rounds 1–2; the same readout for round 3). The total is the sum of the five lines.

## Retained artifacts

- NFS: `/Volumes/akh-medu-runs/lilith/runs/c6-smoke-20260928/`, `/Volumes/akh-medu-runs/lilith/runs/c6-smoke-r2-20260928/` and `/Volumes/akh-medu-runs/lilith/runs/c6-smoke-r3-20260928/`. The round-2 copy had not appeared on the NFS mount yet when this document was first written; the round-3 copy's NFS arrival was not separately re-checked, so treat the local copy as authoritative until confirmed.
- Local (private, not committed): `~/lilith-private/smoke/runs/{c6-smoke-20260928,c6-smoke-r2-20260928,c6-smoke-r3-20260928}/`, each with `SHA256SUMS` and `results-<run-id>.tar.gz`. Payloads: `~/lilith-private/smoke/lilith-payload-main-{cec55065f7bf,ccc915e1a1a1,8f2d889a1fd0,6a8dd50f1287}.tar.gz` with their source manifests.
- Round-2 adapters: `adapters/ctx-8192` (tensor sha256 `9d2b8cd1803fc83ce5998f26f154c54c5ab63d156b0f78e7ba89f9c13d2bd949`) and `adapters/ctx-16384` (tensor sha256 `18aa45465ea992b7889a98ac18c7a12413660e07f50fa751a6670cf645b70cec`). Both are complete, including `adapter_model.safetensors` and the probe files, in the collected local run dir. The results tarball excludes `adapter_model.safetensors` by design. Round 3 reused the `ctx-8192` adapter (via `LILITH_GENERATION_ADAPTER_DIR`) rather than training a new one.
- All three pods are terminated. Round 1: id not retained. Round 2: `ng3c3t99z6o77a`, HTTP 204. Round 3: `3hieq1f9ejmfd1`, HTTP 204.

## Open hypotheses

1. **cuDNN attention is the nondeterministic kernel.** Forcing cuDNN attention (MATH only where cuDNN has no kernel) keeps a divergence of the same size as the default, and it is not repeatable. MATH, EFFICIENT and FLASH(+MATH) are bit-exact and repeatable. The consistent reading is that on this stack (H100, torch 2.13.0+cu130, cuDNN via `nvidia-cudnn-cu13` 9.20.0.48) the default SDPA dispatch picks cuDNN attention for the head_dim-256 sliding layers. That is still not directly verified (the default backend choice was not logged), but it no longer matters operationally: the recorded default no longer uses the true PyTorch default on CUDA (item 2).
2. **CONFIRMED (round 3): the fix is configuration-only.** `sdpa_backend=efficient` + `sdpa_math_fallback=1` passed the control's exact tokenwise path bit-identically (Cache control table above), closing what round 2 left open ("none of these fixes is confirmed on the control's exact tokenwise path yet"). This update makes it the recorded default on CUDA (not ROCm) everywhere `lilith_numerics` is wired: `lilith_training_cache_control.py` (already wired before this update; unchanged behavior, now defaults differently), the runtime training worker (`lilith_runtime_training_worker.py`, pinned per-config via `lilith_runtime_training_config.py`, historical configs keep their untouched behavior), the full-depth LoRA path (`lilith_lora_trainer.py`'s `sft_step`/`preference_step`, `reference_microrun.py`'s micro-run/memory-probe/generate/cold-reload modes) and `run_smoke.sh`'s defaults. ROCm and CPU are unaffected (unmeasured on this finding; PyTorch's own default). Ablated cost: EFFICIENT/MATH offload passes cost about the same as the default (3.5–3.8 s chunked, round 2 diagnostic); round 3's full cache-control step (218 s) is in line with rounds 1/2 (240–249 s), so the fix has no material cost.
3. **flashinfer JIT may also need nvcc/CUDA headers.** They come from the `nvidia-cuda-nvcc` 13.4.92 wheels in the lock. Round 3's strict `vllm-generation` step (no HF fallback) completed without hitting this, so compiler discovery is not currently blocking; kept open only in case a future torch/vLLM/flashinfer bump reintroduces it.
4. The full-bf16 diagnostic difference at 1025 (39.56 / 39.69 in rounds 1/2; 39.0 in round 3) is consistent across all three rounds and much larger than at 1023/1024. It is flagged as diagnostic-only in the control (not part of the pass/fail tolerance) and still deserves a look: a sliding-window boundary semantics difference between the cached and full forward, independent of the SDPA-backend nondeterminism this update fixes.

## Exit condition (C6 row)

| Criterion | Status |
|---|---|
| ≤ 2 GPU-h smoke on 1× H100 | met every round; round 3: 753.8 s wall |
| controls pass | **met (round 3)**: cache control passed under `sdpa=efficient+math` (logits 0.0/0.0/0.0, grad diffs 1.4e-8/1.5e-7, tolerance 1e-4); sampler/replay drift passes every round (round 3: 5.5e-07) |
| 32K memory measured | met (round 2, fits, peak allocated 54,769,545,728 bytes) |
| one SFT + one preference step | met (round 2: 200 SFT + 1 preference per context, 8K and 16K) |
| exact cold reload | met (round 2: probe sha identical, max_abs 0.0) |
| vLLM LoRA hot-load / generation | **met (round 3)**: backend vllm, lora_applied, 79.26 tok/s |
| one live tunnel episode recorded | **not attempted; blocked on C6a** (the WireGuard tunnel slice, `feat/lilith-c6a-wireguard`) |
| spend recorded | met (this document; $16.03 total across three rounds) |

Every measurable criterion is met as of round 3. The one open item — a live tunnel episode — has no C6-round-shaped follow-up: it needs C6a's tunnel, not another smoke round.
