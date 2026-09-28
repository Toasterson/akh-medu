# Lilith C6 — Runpod H100 smoke, rounds 1 and 2 (2026-09-28) — RESULT

Anima WI `01a0e270-6021` (C6 Runpod CUDA harness); plan `docs/ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md` (C6 row), ADR 116.

Sources are all on the supervisor Mac under `~/lilith-private/smoke/runs/`:
- Round 1: `c6-smoke-20260928/{smoke-result.json, steps.tsv, smoke.log, cache-control.json, provenance.json, bootstrap-check.json, gpu.csv, bootstrap-20260928.log}`
- Round 2: `c6-smoke-r2-20260928/{smoke-result.json, steps.tsv, smoke.log, cache-control.json, cache-diagnostic.json, memory-probe.json, microrun.json, microrun.generation.json, cold-reload.json, provenance.json, bootstrap-check.json, gpu.csv}`
- Watchdog log: `~/lilith-private/smoke/watchdog-c6-r2.log`

Costs and the round-1 idle incident come from the supervisor's Runpod billing readout and its incident note in the round-3 request (2026-09-28). No figure below comes from anywhere else.

## Verdict

**C6 exit condition NOT met.** The resident-vs-offloaded KV cache control failed equivalence on CUDA in both rounds; the 1e-4 tolerance stays. Round 2 found the cause class: **run-to-run nondeterminism of the CUDA kernels, not the offload copy.**
- The resident cache run twice differs as much as resident vs offload does: tokenwise, the logit max_abs is 1.0 / 0.5 / 0.578 at 1023 / 1024 / 1025.
- In the chunked sweep, forcing SDPA MATH, SDPA EFFICIENT, SDPA FLASH (with MATH where flash has no kernel) or deterministic algorithms makes resident and offloaded logits bit-identical (0.0) and repeatable.
- Forcing cuDNN attention (with MATH fallback) keeps the divergence.

None of these fixes is confirmed on the control's exact tokenwise path yet: the diagnostic hit its 600 s step cap before any restoring variation's tokenwise pair ran.

Everything outside the cache control passed in round 2:
- Memory fits at 8K / 16K / 32K.
- The reference micro-run completed: 200 weighted-SFT steps plus 1 sibling-preference step per context, for 8K and 16K.
- The cold reload is exact: probe sha identical, max_abs 0.0.

vLLM did not start (flashinfer JIT could not find `ninja`), so generation fell back to HF at 12.65 tok/s. The round-3 PR (`fix/lilith-c6-round3`) addresses the ninja PATH problem, the diagnostic budget and ordering, and adds an `--only` smoke mode.

## Configuration

| Setting | Value | Source |
|---|---|---|
| GPU | 1× NVIDIA H100 80GB HBM3, driver 580.159.03, 81559 MiB | `gpu.csv` (both rounds) |
| Pod (round 2) | `ng3c3t99z6o77a`, terminated 2026-09-28T12:02:01Z (HTTP 204) | `watchdog-c6-r2.log` |
| Pod (round 1) | id not in the retained evidence | — |
| Code | round 1 main `cec55065f7bf83c19f863e96e81755b7688ee6a5`; round 2 main `8f2d889a1fd07f0180f6fd745c1f435ace52a78f`; provenance `verified`, no problems | `provenance.json`, `smoke-result.json` |
| Env | Python 3.12.3, torch 2.13.0+cu130 (CUDA 13.0), transformers 5.10.4, peft 0.21.0, trl 1.14.0, vllm 0.30.0, flashinfer-python 0.6.18.post1, triton 3.7.1 | `bootstrap-check.json`, `cache-control.json` (`versions`) |
| Env lock | `requirements-cuda.txt` sha256 `733d84e137b4f3387f15a3558f486a5b9e98ddd250cce6d40c462822a489d36c`, built from payload `src-ccc915e1` | `bootstrap-20260928.log` |
| Parent | `google/gemma-4-12B-it` @ `5926caa4ec0cac5cbfadaf4077420520de1d5205`; 48 layers (40 sliding, head_dim 256, window 1024; 8 full, head_dim 512); dense (no MoE) | `bootstrap-check.json`, `cache-diagnostic.json` (`model`) |
| Engine | `lilith_training_engine.py` sha256 `5c68f6a1d451c87b6d77e3a54c2786305208a35b5b366ad8661a9ca0bbcb73d0` (qualified, unchanged) | `cache-control.json` |
| Attention / dtype | sdpa, bf16 | `cache-control.json` (`runtime`) |
| Numerics | default (untouched PyTorch). At diagnostic start: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, matmul TF32 off, cuDNN TF32 on, bf16 reduced-precision reduction on, deterministic off, all four SDPA backends enabled | `cache-control.json` (`numerics`), `cache-diagnostic.json` (`numerics_at_start`) |
| Data | `sft-subset-v1.jsonl` sha256 `e662d28289165d8cc419a4b65bba63d7eb88b414e5f7ce63aff62c7a3028dd94`, 8 records | `smoke.log` (data-pin), `microrun.json` (`data`) |
| Budget | 7200 s global; cache diagnostic 480 s passes / 600 s step cap (round 2) | `smoke-result.json`, `smoke.log` |

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

## Steps

| Step | Round 1 | Round 2 |
|---|---|---|
| provenance | ok (0 s) | ok (0 s) |
| bootstrap-check | ok (27 s) | ok (27 s) |
| gpu | ok (0 s) | ok (0 s) |
| data-pin | ok (0 s) | ok (0 s) |
| cache-control | **failed** (249 s, `failed_equivalence`) | **failed** (240 s, `failed_equivalence`) |
| cache-diagnostic | (did not exist) | **timeout** (600 s, exit 124; no verdict written) |
| memory-probe | not run (harness stopped at first failure) | ok (143 s) |
| microrun | not run | ok (3490 s) |
| cold-reload | not run | ok (62 s) |
| overall | failed, wall 276.2 s | failed (`failed_steps`: cache-control, cache-diagnostic), wall 4562.6 s |

## Memory and throughput (round 2)

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

| Measure | Round 1 | Round 2 |
|---|---|---|
| logit max_abs | **3.8125 / 0.875 / 0.5625** | 3.65625 / 0.9375 / 0.75 |
| mean selected-token loss: resident / offload | 24.8347 / 24.8764 | 24.3317 / 23.9018 |
| loss diff | 0.04171 | 0.42989 |
| LoRA A grad max_abs | 1.238e-04 | 3.697e-04 |
| LoRA B grad max_abs | 4.284e-04 | 5.786e-04 |
| generate→replay sampler logp drift | 1.948e-07 (4 tokens, policy sha bound) | 1.948e-07 |
| full-bf16 diagnostic max_abs (diagnostic only) | 19.53 / 6.375 / 39.56 | 1.781 / 0.625 / 39.69 |
| peak GPU bytes | 24,436,299,264 | 24,436,299,264 |
| status | `failed_equivalence` | `failed_equivalence` |

Round 2 repeats round 1's inputs, weights and code path exactly, yet the numbers differ, including the full-forward diagnostic at 1023 (19.53 → 1.78). That is already evidence of nondeterminism. On the local ROCm stack the same control matched to 2.9e-11 (round-1 brief).

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

Verdict recomputed offline from the saved report (`lilith_cache_path_diagnostic.py --from-report`, round-3 PR): cause `run_to_run_nondeterminism`; restoring variations on the chunked path only: deterministic, sdpa_math, sdpa_flash_math_fallback, sdpa_efficient; `recommended_numerics: null` until a tokenwise pair confirms one.

Per-pass cost: chunked passes took 2.0–4.8 s, except the deterministic offload pass (23.6 s) and the contiguous offload pass (23.2 s). Tokenwise resident passes took 80.1 s and 52.1 s, and the tokenwise deterministic resident pass 58.2 s. The run was killed during the tokenwise deterministic **offload** pass, whose cost the planner under-estimated: its estimate was per mode, not per case.

## vLLM failure

vLLM 0.30 got through weight load, CUDA-graph capture and JIT kernel warmup. The sampler warmup then called flashinfer `top_k_top_p_sampling_from_logits`, which JIT-builds its sampling module by running `ninja`. That raised `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'` (`smoke.log` lines 213 and 317), and the engine core exited with `RuntimeError: Engine core initialization failed`.

The locked env already contains `ninja==1.13.2`, pulled in by flashinfer. However, the smoke runs `$ROOT/env/bin/python` without activating the env, so `env/bin` (where the `ninja` console script lives) was not on PATH.

The round-3 fix:
- pins `ninja` in `requirements-cuda.in`;
- puts `env/bin` first on PATH in `run_smoke.sh`, `bootstrap.sh` and `reference_microrun.py`;
- adds a bootstrap readiness check that the env's own `ninja --version` works;
- adds a strict `vllm-generation` step with no HF fallback.

## Incidents

| Incident | Detail | Fix / resolution |
|---|---|---|
| Round-1 pod idle (~2 h 14 min) | The supervisor's `pgrep -f` watcher matched its own command line, so it never saw the run end, and the pod stayed up idle | `scripts/lilith/pod_watchdog.sh` (main `b9944413`, `f64ca951`, PR #441): one ssh probe per 15 min, no pgrep/pkill, keyed on `smoke-result.json` / pid liveness / file progress / GPU utilisation; terminates on done, dead, stalled or unreachable. Round 2 used it and terminated 23 s after `done`. |
| Round 1 stopped at the first failure | The pre-#440 harness exited on the failed cache control, so memory, micro-run and cold reload never ran | PR #440: gates stop the run, other steps continue, `failed_steps` lists all failures, and `smoke-result.json` is written last as the completion marker |
| Round-1 `tar: file changed as we read it` | The tarball was written inside the run dir it was packing | PR #440: the tarball is built outside the run dir |
| Diagnostic killed without a verdict | 480 s pass budget and 600 s step cap; the verdict was written only at the end; per-mode cost estimates let a slow deterministic offload tokenwise pass start | Round-3 PR: provisional verdict persisted after every case, per-case cost estimates, in-pass deadline, `--from-report`, 1800 s default budget |
| vLLM `ninja` not found | See above | Round-3 PR |

## Cost

| Item | Cost (USD) |
|---|---:|
| Round 1 pod (including the idle period) | $8.76 |
| Readout | $0.25 |
| Volume fill | $1.07 |
| Round 2 pod | $4.52 |
| **Total** | **$14.60** |

Source: the supervisor's Runpod billing readout (round-3 request). The total is the sum of the four lines.

## Retained artifacts

- NFS: `/Volumes/akh-medu-runs/lilith/runs/c6-smoke-20260928/` and `/Volumes/akh-medu-runs/lilith/runs/c6-smoke-r2-20260928/`. The round-2 copy had not appeared on the NFS mount yet when this document was written; the local copy is authoritative until it does.
- Local (private, not committed): `~/lilith-private/smoke/runs/c6-smoke-20260928/` and `~/lilith-private/smoke/runs/c6-smoke-r2-20260928/`, each with `SHA256SUMS` and `results-<run-id>.tar.gz`. Payloads: `~/lilith-private/smoke/lilith-payload-main-{cec55065f7bf,ccc915e1a1a1,8f2d889a1fd0}.tar.gz` with their source manifests.
- Round-2 adapters: `adapters/ctx-8192` (tensor sha256 `9d2b8cd1803fc83ce5998f26f154c54c5ab63d156b0f78e7ba89f9c13d2bd949`) and `adapters/ctx-16384` (tensor sha256 `18aa45465ea992b7889a98ac18c7a12413660e07f50fa751a6670cf645b70cec`). Both are complete, including `adapter_model.safetensors` and the probe files, in the collected local run dir. The results tarball excludes `adapter_model.safetensors` by design.
- Both pods are terminated. Round 2: `ng3c3t99z6o77a`, HTTP 204.

## Open hypotheses

1. **cuDNN attention is the nondeterministic kernel.** Forcing cuDNN attention (MATH only where cuDNN has no kernel) keeps a divergence of the same size as the default, and it is not repeatable. MATH, EFFICIENT and FLASH(+MATH) are bit-exact and repeatable. The consistent reading is that on this stack (H100, torch 2.13.0+cu130, cuDNN via `nvidia-cudnn-cu13` 9.20.0.48) the default SDPA dispatch picks cuDNN attention for the head_dim-256 sliding layers. That is not verified: the default backend choice was not logged.
2. **Candidate fix, configuration only:** `LILITH_NUMERICS_SDPA_BACKEND=efficient`, or `math`, or `LILITH_NUMERICS_DETERMINISTIC=1`. EFFICIENT and MATH offload passes cost about the same as the default (3.5–3.8 s chunked); the deterministic offload pass cost 23.6 s. Confirm tokenwise before adopting. The training worker and the micro-run trainer do not apply `lilith_numerics` yet; a chosen fix must be wired there too and recorded in their manifests.
3. **flashinfer JIT may also need nvcc/CUDA headers.** They come from the `nvidia-cuda-nvcc` 13.4.92 wheels in the lock. Once `ninja` is found, compiler discovery is the next thing that can fail; the strict `vllm-generation` step will show it.
4. The full-bf16 diagnostic difference at 1025 (39.56 / 39.69) is the same in both rounds and much larger than at 1023/1024. It is flagged as diagnostic-only in the control, but it deserves a look: a sliding-window boundary semantics difference between the cached and full forward.

## Exit condition (C6 row)

| Criterion | Status |
|---|---|
| ≤ 2 GPU-h smoke on 1× H100 | round 2: 4562.6 s wall |
| controls pass | **not met**: cache control failed both rounds; sampler/replay drift passes (1.9e-07) |
| 32K memory measured | met (fits, peak allocated 54,769,545,728 bytes) |
| one SFT + one preference step | met (200 SFT + 1 preference per context, 8K and 16K) |
| exact cold reload | met (probe sha identical, max_abs 0.0) |
| vLLM LoRA hot-load / generation | **not met** (ninja) |
| one live tunnel episode recorded | not attempted |
| spend recorded | met (this document) |

**Next round (round 3, after the round-3 PR merges):** build the payload from main, then run

```
run_smoke.sh --only cache-control,cache-diagnostic,vllm-generation <run-id>
```

with `LILITH_GENERATION_ADAPTER_DIR` pointing at the round-2 `ctx-8192` adapter, transferred to the pod. Transferring it is a training artifact directly required by the run, which the Runpod rule allows. With the default numerics the control fails again and the diagnostic, now with an 1800 s budget, confirms the restoring variation tokenwise. Then rerun the control with the recommended `LILITH_NUMERICS_*` values. Supervise the run with `pod_watchdog.sh`.
