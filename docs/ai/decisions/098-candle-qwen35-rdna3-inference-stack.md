# ADR 098: Candle Qwen35 inference stack on RDNA3

- **Status:** Accepted
- **Date:** 2026-08-24
- **Plan:** [Candle Qwen3.8 inference on RDNA3](../plans/2026-08-24-candle-qwen38-rdna3-inference.md)
- **Extends:** [ADR 037](037-llm-vsa-deep-integration.md),
  [ADR 065](065-context-assembly-runtime.md),
  [ADR 073](073-full-candle-hip-device-backend.md), and
  [ADR 074](074-candle-static-shape-inference-and-training.md)

Anima: existing “Lane: activate ptah on hephaistos…” work item; ID unavailable
because no Anima project tool is exposed in this session.
Seshat sync: not performed; no `seshat_ingest` tool is available in this session.
This file is the offline canonical paste source.

## Context

Ptah must keep its exact custom Qwen3.8-27B Q4_K_M model and matching vision
projector. Substituting a distill or a nearby Qwen variant invalidates its learned
behavior. Ollama/llama.cpp can run that model, but their opaque forward state
prevents direct hidden-state signals for Virtual Context and VSA retrieval and
limits control over recurrent state, KV compression, and RDNA3-specific kernels.

Qwen3.8 GGUFs identify as the `qwen35` architecture and mix gated-delta recurrent
layers with full-attention layers. Treating that architecture as Qwen2 can read
valid weights through the wrong tensor schema. The runtime therefore needs an
explicit family before any partial loader is safe.

## Decision

Build an inference-only Qwen35 stack in Candle for the exact installed model.
The public seam is an owned `InferenceSession`, not individual kernels. It owns
all position-dependent state for one ACP session: full-attention KV, gated-delta
recurrent and convolution state, hidden taps, and later vision and speculative
state. Its operations are prefill, extension/decode, checkpoint, and rewind.

`Qwen35Model` will hide two internal adapters:

- a reference implementation built from ordinary Candle tensor operations;
- an RDNA3 implementation using private HIP kernels after parity gates pass.

The work proceeds in measured vertical slices: explicit family recognition and
fail-closed loading; reference text forward; exact-model RDNA3 kernels; session
ownership plus VSA hidden taps; exact vision; TurboQuant KV pages; target-only
canary; and only then an optional DFlash2 layer. The llama.cpp-backed Ptah remains
the production oracle until the target-only Candle canary passes.

Virtual Context normally materializes 8K–32K tokens. A 100K physical window is a
ceiling and pressure gate, not the normal operating target. VSA receives compact
signals captured during the existing forward; it does not trigger a duplicate
model pass or splice position-dependent KV from another context.

## Consequences

- The exact custom model and projector remain the only acceptance target.
- Unsupported Qwen35 Candle loads fail with an actionable diagnostic and cannot
  fall through to Qwen2.
- Session state has one owner, making ACP isolation and rollback testable.
- Kernel choices stay private and replaceable; correctness is established against
  the reference adapter before performance changes compose.
- Hidden-state/VSA integration, vision, TurboQuant, and optional DFlash share one
  forward/session boundary instead of becoming separate inference stacks.
- Inference work can start now, while target-model training remains deferred.

## Alternatives considered

- **Stay on Ollama/llama.cpp permanently:** operationally simple, but does not
  expose the state needed for the full Virtual Context/VSA design.
- **Adopt tinygrad as the application runtime:** useful kernel evidence, but would
  duplicate the existing Rust/Candle integration and context boundary.
- **Add DFlash2 first:** rejected until target-only performance is measured; it
  adds another model and rollback state before the target path is understood.
- **Port to Burn and train immediately:** rejected for this sequence. The target
  is inference fidelity, and the current Burn dependency conflict still blocks
  the repository's training feature.
- **Always materialize 100K tokens:** rejected because it spends latency and VRAM
  where Virtual Context can retrieve a smaller, task-relevant working set.
