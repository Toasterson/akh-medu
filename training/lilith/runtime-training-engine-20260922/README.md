# Runtime training cache diagnosis

This directory contains mechanical controls, not a trained or promoted adapter.
No optimizer step is taken by these controls. Historical failures remain present.

## Confirmed defects and fix

The installed Transformers cache constructor slices `layer_types[:-0]` when
Gemma declares zero shared-KV layers. Its fallback allocates full-attention
storage for every layer, losing the bounded sliding-cache layout. Lilith now
constructs the configured public cache layer types explicitly.

The asynchronous CPU-offload path also races on this ROCm system. Identical
32-token forwards differed by up to 19.9375 in raw logits with original Gemma
arithmetic. Both GPU-resident and synchronous CPU-offload runs were repeatable.
The library's prefetch stream lacks the reciprocal offload-readiness ordering
and tensor-lifetime recording identified during source review. Lilith uses
blocking transfers on the caller stream rather than that asynchronous path.
The installed library, model weights, RMSNorm and attention implementation are
unchanged by the fix. Temporary arithmetic experiments were diagnostic only.

## Evidence

- `qualified-cache-control.json`: the reusable source-pinned control passes.
  At 1,023/1,024/1,025 positions, resident/offloaded logits and mean loss are
  identical, with maximum A/B gradient difference 2.9103830456733704e-11.
  Generation/replay log-probability drift is 4.5609492249631955e-7, below 1e-4.
  Both gradients are nonzero and the original policy identity is restored.
  Its four-token generation intentionally ends at the diagnostic budget; it is
  not an admitted training episode and performs no optimizer update.
- `transfer-variants-original-arithmetic.json`: repeated asynchronous execution
  fails, while resident and synchronous execution are repeatable.
- `fixed-boundary-control.json`: original diagnostic harness, 1,023/1,024/1,025
  token positions. Resident and synchronous logits are identical; maximum
  gradient difference is 2.3283064365386963e-10. Both A/B gradients are nonzero.
  Its `contract` field came from the pre-rebind engine instance and still names
  `DynamicCache`; the effective cache factory was rebound to the recorded fixed
  source. Use the reusable control's report for final contract qualification.
- `fixed-long-memory-control.json`: 8,192-token forward passes with finite
  logits, 136 detached cache tensors, 1,023 stored sliding positions and 8,192
  global positions. Peak allocated GPU memory: 24,050,122,240 bytes. This is a
  chunked memory check, not a sampler/replay equivalence claim.
- `fixed-context-memory-control.json`: the configured 32,768-token context also
  passes with finite logits and detached KV. All 32,768 global positions remain
  present, while sliding layers retain 1,023 positions. Peak allocated GPU
  memory is 24,099,580,416 bytes. Prefill chunks are 32 tokens for this memory-only
  diagnostic; canonical collection/replay still uses one-token chunks.
- `test_lilith_training_cache.py`: actual small FP32 Gemma architecture checks
  full versus cached logits and both nonzero LoRA gradients at window lengths
  7/8/9, including synchronous cache construction and storage bounds.

## Numerical acceptance

The strict cache comparison uses the same tokenwise arithmetic with resident
and offloaded storage, plus a separate collection/replay log-probability check.
The FP32 small-model full-forward test checks the mathematical gradient seam.
Batched full BF16 results are retained as diagnostics; they do **not** pass the
old 1e-4 full-versus-tokenwise gate. In the original fixed-boundary control their
maximum logit deviations were 17.1005859375, 5.0625 and 2.6875. Reporting those
measurements separately is an explicit change to qualification, not evidence
that batched and tokenwise BF16 arithmetic agree.

Only the final query adapter is trainable. With a fixed teacher-forced token
sequence, all cached keys/values are independent of its parameters. The gradient
of a sum of per-token losses therefore equals the sum of their gradients with
detached caches, provided weights stay fixed until the batch ends. Training a
lower layer or K/V projection would invalidate that premise. Tecton, running Kimi K3, reviewed the numerical acceptance criterion and
recommended separate strict cache and precision-aware cross-shape checks. See
[the review and its disposition](kimi-math/README.md). BF16 cross-shape
calibration remains open; no tolerance was loosened to accept the race.

## Reproduce

From the repository root, using the private training Python environment:

```sh
~/.local/share/akh-medu/lilith-training/venv310/bin/python \
  scripts/session-platform/lilith_training_cache_control.py \
  --parent ~/.local/share/akh-medu/lilith-training/parent-5926caa4 \
  --output /absolute/path/to/a-new-cache-control.json
```

The output must not already exist. The control pins the parent, records source
identities, restores adapter/RNG state, and performs no optimizer update.
`diagnostics/` preserves exploratory scripts that rely on a held model in a
debugging process; they are not the supported standalone reproduction command.
