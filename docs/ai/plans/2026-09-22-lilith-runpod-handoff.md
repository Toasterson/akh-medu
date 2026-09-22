# Astra handoff: finish Lilith training on Runpod

## Assignment and current state

The user stopped local training for today and requested this handoff for Astra on
the long-running machine to try the full run on Runpod. Do not restart local GPU
training. Prepare a reproducible cloud run, measure it, and carry it through
training, evaluation, cold reload, and an honest report. Keep the runtime tools
available during training, as they are during inference.

Local source machine: hephaistos, repository `/home/toasty/ws/agi/akh-medu`.
All repository paths below are relative to that root unless explicitly absolute.
Use feature branch `feature/lilith-runtime-training-runpod`, which packages the
Lilith implementation and selected evidence. The original main branch does not
contain this work. Model weights, private datasets, credentials and some generated
run artifacts remain outside Git and need separate secure transfer. Preserve
unrelated work; do not reset, clean or force-push.

`training/lilith/runtime-training-stack-20260922/train2` was stopped through the
session cancellation API at the user's request. Cancellation was confirmed;
all dedicated services exited, private stack state was removed, and GPU memory
returned to approximately 36 MB. No episode completed, no optimizer step occurred,
and no adapter was produced. The worker's stale `status: starting` is not a live
process; the stack's terminal result and `operator-stop.json` are authoritative.
This stop is intentional, unlike train1's watchdog failure.

Read first:

- `AGENTS.md`, `CONTEXT.md` and applicable ancestor instructions.
- `docs/ai/plans/2026-09-22-lilith-runtime-training.md` — runtime and learning contract.
- `docs/ai/plans/2026-09-22-lilith-overview.md` — broader project state.
- `training/lilith/runtime-training-engine-20260922/README.md` and `audit.json` — numerical controls.
- `training/lilith/runtime-training-engine-20260922/kimi-math/README.md` — actual Tecton/Kimi K3 advice and limits.
- `training/lilith/runtime-training-stack-20260922/train2/README.md`, `worker-config.json`, and `protocol.json` — latest run configuration.

## User requirements that carry forward

- Train Lilith's Gemma using ADAM/Kaine learning/reward mechanisms first; do not
  invent separate ADAM/Kaine models.
- Normal reasoning, generous context: 32,768 input-plus-output context positions,
  16,384 generated-token ceiling per response, up to 16 tool iterations. Do not
  quietly shorten reasoning or turn this into an efficiency/Neuralese experiment.
- Train under an `akhrtd`-owned worker using the authoritative session tool broker;
  no standalone script secretly replacing model workspace tools.
- Preserve train/development/sealed-evaluation separation and tool receipts.
  Incomplete answers are resource limits, not semantic coding failures or RL samples.
- Ultimate goal: match/beat the pinned Qwen reference on useful illumos work,
  diagnose failure causes, avoid benchmark gaming. No parity or quality improvement
  has been established by these incomplete runtime runs.
- Credential rotation remains a separate maintenance task. No shared restarts or
  rotation as a shortcut. Never query Anima Postgres directly; use Anima MCP.
- Sol subagents are welcome for concrete independent work; the user previously
  requested them. Kimi K3 math advice already exists; do not claim a new review
  without an actual reply.

## Why move, and what has actually been measured

Local hardware: RX 7900 XTX 24 GiB, i7-6700K, 31 GiB system RAM. This worker uses
BF16 PyTorch/ROCm, not Vulkan inference. At 384→640 generated tokens in train2,
we measured 256 tokens / 49.289 s = 5.19 tokens/s. That is a short-interval
measurement, not a guaranteed long-context rate.

After roughly 73 minutes the run was still on the first baseline task. It first
produced a 786-token tool handoff attempting `create:true` on an existing file;
the broker correctly rejected it. It then generated a long follow-up. Last saved
progress before the intentional stop was 14,464 tokens in that response. No
completed trajectory for that interrupted response is available.

The 72-hour deadline was a safety ceiling, NOT an ETA. Previous examples using
20/40/80 tokens/s and 200,000 tokens were hypothetical calculations, not measured
Runpod performance. Do not repeat them as forecasts. A larger GPU does not by
itself fix a model that struggles to recover from tool errors.

## Cloud choice and resource plan

Suggested first target: one H100 SXM 80GB Pod. Runpod's pricing page checked
2026-09-22 listed $3.49/hour; A100 SXM 80GB was $1.59/hour. Recheck live price,
availability and account spending constraints before launch:
https://www.runpod.io/pricing

Use a dedicated Pod, not an inference-only serverless endpoint. Rent only the
resources required for this job, use a bounded deadline, and arrange artifact
export plus shutdown so a finished or failed job does not keep consuming GPU
hours. Avoid putting Pod shutdown credentials inside the model worker.
A 30-minute measurement window costs roughly $1.75 compute at the quoted H100
rate, excluding setup, storage and other charges. If one full episode takes
longer, report that fact rather than silently reducing its token budget.

First measure a representative full-budget episode and nonzero-gradient replay
under the runtime. Estimate the full protocol from generated/replayed tokens,
prefill time and tool/reviewer latency. Obtain any missing account access or
spending limit from the operator; this document contains no credentials.

## Transfer and environment preparation

1. Fetch and check out `feature/lilith-runtime-training-runpod`, or import the
   Git bundle supplied with the handoff. Record the commit identity. Required
   source and selected evidence are committed; consult the transfer manifest
   `training/lilith/TRANSFER.md` for what is intentionally outside Git. Transfer
   private model/data assets separately. Do not publish raw transcripts, credentials
   or multi-gigabyte build products.
2. Transfer the pinned parent from
   `~/.local/share/akh-medu/lilith-training/parent-5926caa4/`, including weights,
   tokenizer, config, chat template and generation config. Verify:
   - weights: `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`
   - config: `478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9`
   - tokenizer: `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`
3. Transfer the private admitted subset
   `~/.local/share/akh-medu/lilith-training/transcripts-20260922/sft-subset-v1.jsonl`
   securely. Its identity is in
   `training/lilith/transcripts-20260922/sft-subset-v1-manifest.json`.
   There are eight frozen training and eight validation records; only training
   records are fitted. These are supervised imitation, not verified RL rewards.
4. Local Python is 3.10 with torch 2.7.1+rocm6.3, transformers 5.10.1,
   peft 0.18.1, accelerate 1.10.1, safetensors 0.6.2, tokenizers 0.22.2,
   jsonschema 4.25.1. On NVIDIA install a compatible CUDA Torch build; do NOT
   copy the ROCm virtualenv. Record exact versions and rerun numerical controls.
5. Build/copy compatible `akhexecd`, `akhsbxd`, `akhsessd`, and training-capable
   `akhrtd`. The known local runtime-only build receipt is
   `training/lilith/runtime-training-build-receipt.json`; its private binary is
   `~/.local/share/akh-medu/lilith-runtime-training-target/release/akhrtd`.
   Check libc/library compatibility; rebuild when needed, preserving source identity.
6. The scripts are NOT fully portable launchers yet. They hardcode the private
   venv, parent, transcript and prior-stack locations, and expect templates at
   `~/.local/share/akh-medu/lilith-dev-vulkan-20260921/`. Adapt those paths or
   reconstruct equivalent paths on the Pod. Generate fresh isolated credentials;
   do not copy the whole source machine's secrets/config indiscriminately.
   Inspect `prepare-lilith.py`, `lilith_runtime_training_config.py`, and
   `lilith_runtime_training_stack.py` before launching.

## Critical fixes and controls to preserve

### Cache arithmetic

The qualified source is `scripts/session-platform/lilith_training_engine.py`,
SHA256 `5c68f6a1d451c87b6d77e3a54c2786305208a35b5b366ad8661a9ca0bbcb73d0`.
It constructs explicit hybrid sliding/global cache layers and uses synchronous
CPU copies when offloading. The installed Transformers shared-layer slicing bug
with zero shared layers otherwise builds the wrong cache layout. Its asynchronous
transfer path also produced severe nondeterminism on the local ROCm stack.

For H100, make resident KV storage an explicit pinned worker/config option and
set `engine.cache_offloading = False` through that configuration after loading.
The current loader defaults to offloading, so merely moving it to H100 leaves
CPU transfers enabled. Do not switch back to the broken async path. Keep canonical
prefill chunk size 1 until a separately qualified change justifies another size.
Measure resident-cache memory at full 32K; 80GB headroom is expected, not yet proven.

Only final-layer query LoRA is trainable: rank 2, alpha 2,
`model.language_model.layers.47.self_attn.q_proj`. Detached cached K/V is valid
for this seam; expanding trainable layers invalidates that argument.

Run `lilith_training_cache_control.py` on the actual CUDA stack plus the small
FP32 tests and 32K memory control. Required evidence: both nonzero A/B gradients,
resident/offload same-path comparison, window boundaries 1023/1024/1025,
sampler/replay identity, detached caches, and zero optimizer steps during controls.

Local final source-pinned results: logit/loss differences 0, maximum gradient
error 2.91e-11, sampler logp drift 4.56e-7; 32K offloaded memory ~22.4 GiB.
These do not automatically certify CUDA. BF16 batched-vs-tokenwise differences
remain a separate open numerical calibration problem. Historical boundary max
errors reached 17.1; do not mask them by loosening the race/correctness threshold.
See actual Kimi review for the distinction. No normalization/attention monkeypatch
from exploratory diagnostics is part of the qualified engine.

### Session lifetime and workspace writes

- Train1 was system-retired by akhsessd's default 15-minute durable-progress
  watchdog. Token telemetry does not advance that clock. The dedicated stack
  now sets `session_stall_threshold_ms = timeout_seconds*1000 + cleanup_grace_ms`.
  For 72 hours this is 259,320,000 ms. This is a bounded mitigation, not a
  heartbeat implementation. Do not change shared fleet timeouts.
- Preserve `events.jsonl` before private stack cleanup (historical JSONL is not
  included in the Git transfer). Generation token counts
  are saved every 128 tokens in `worker/progress.json`, with no raw reasoning.
- `write_file` must expose optional `expected_revision` in BOTH inference and
  training catalogs. Existing-file writes require `create:false` and the revision
  from a successful broker read. Task setup now uses `reset_artifact()` and never
  bypasses the broker. Do not repair invalid model-generated writes behind its back.
- `fixture5` captures the failing real reset; `fixture6` passes five real broker
  calls covering missing read, create, read, revision-checked overwrite, readback.
- Ten stack/worker tests and ten `test_prepare_lilith.py` tests passed. Repeat
  relevant tests and the real fixture after cloud path/config changes.

## Full experiment and runtime launch

The experiment is in `lilith_runtime_training_worker.py`:
4 baseline training tasks → one SFT step on 8 admitted records → 3 online reward
banks (4 tasks each) → development comparison of SFT, online RL and 0.5 blend
(4 tasks each) → freeze selection → paired 4-task sealed evaluation → paired
4-task retention. Total 44 coding episodes, potentially multiple responses each.
Banks with no reward variation deliberately make no optimizer update. Do not
call an unchanged RL arm trained. Inspect policy hashes and update receipts.

Read the frozen train2 configuration for all other constants, including learning
rate 1e-4 and beta 0.1. New cloud paths/source edits require a NEW configuration
and NEW run directory. Never rewrite historical config pins or resume an uncertain
partially effective optimizer step. Both local runtime attempts made zero updates.

After adapting paths and qualifying the CUDA backend, the existing CLI shape is:

```sh
python scripts/session-platform/lilith_runtime_training_config.py \
  --stack-output /absolute/runpod-run1 \
  --config-output /absolute/runpod-run1-config.json
python scripts/session-platform/lilith_runtime_training_stack.py \
  --runtime-binary /absolute/training-capable-akhrtd \
  --output /absolute/runpod-run1 \
  --worker-config /absolute/runpod-run1-config.json \
  --timeout 259200 --base-port 18620
```

These are templates, not claims the current local paths work on Runpod. Pin source
hashes only after edits and tests; do not mutate pinned files mid-run. Launch under
a durable supervisor, record PID/session/run identifiers, and monitor counters,
receipts, memory and cost. A dead chat is not a process supervisor. Preserve
incomplete trajectories as incomplete; never fabricate a completed answer.

## Ptah, illumos, cold reload, and completion

The current stack expects a separate Ptah runtime/model, loaded only AFTER Gemma
unloads. It clones that profile from the prior local stack and expects
`target/release/akhrtd` for the reviewer. Plan its GPU backend and model transfer
explicitly; a Vulkan/ROCm source-machine profile will not necessarily work on a
CUDA Pod. Preserve review receipts and don't claim a queue acknowledgment is a
completed review.

Native review uses `lilith_vm_executor.local_executor()`, currently SSH to the
source machine's localhost:10115 OpenIndiana guest. That address is NOT the VM
from inside Runpod. Arrange an authenticated reachable validation endpoint or
keep native validation pending with a precise reason; do not silently substitute
Linux tests or assume nested virtualization is supported. Keep model execution
separate from control-plane credentials.

After successful training, use worker `mode: cold_reload` in a fresh runtime
process with a new output/config, the completed worker directory as `reload`,
and the selected adapter file hash as `reload_adapter_sha256`. Verify exact policy
identity against `selected_policy`. The main driver does not automatically run
this follow-up. Inspect the implementation before creating its config.

Completion report should state: actual optimizer updates by arm, artifact hashes,
selected adapter and development criterion, paired sealed-eval/retention scores,
review/native-validation status, cold-reload result, measured tokens/s and wall
clock per phase, actual cloud spend, failures and limitations. Do not promote or
overwrite deployed weights automatically. Export artifacts securely and stop
billable GPU resources when finished. Arrange an explicit completion notification
if supported by the long-running harness; the local detached run had none.

## Suggested skills

- `diagnosing-bugs`: CUDA/runtime failures, numerical differences, slow generation;
  use reproducible controls and actual broker integration tests.
- `handoff`: durable follow-up if the receiving session also needs continuation.
- `review`: focused review of cloud portability and training changes before the
  full run; independent reviewers should not edit pinned sources while it runs.

No secrets are included in this document.
