# Native source checkpoint review

Fixed comparison: `git diff fc1c5fe8...e291ab44` (one consolidation commit).
Both reviews were independent, local and read-only. The local source-checkpoint
specification and ADR 098 were used; no issue tracker or remote host was accessed.

## Standards

Hard standards violations: none found. AGENTS.md provides operational rules
but no relevant coding standard violated by this diff.

Judgement call — low priority:

- Repeated family/template mapping: `crates/akh-session-services/src/internal_native.rs:397`
  duplicates chat delimiters and family grouping already maintained in
  `src/nlu/candle_backend.rs:72` and `:1733`; `internal_native.rs:467` separately
  repeats which families fold system messages. A template correction now needs
  changes across both crates. Expose a small shared template descriptor
  containing delimiters and system-role behavior, while retaining the
  specialized Qwen XML renderer. This is introduced maintenance debt, not
  evidence that the current delimiters are wrong.

The reviewer would not block this source checkpoint on that refactor. The
global resident-model lane, unsupported Vulkan operations, and outstanding
performance/release acceptance work are architecture or explicitly documented
project limits, not newly established defects.

## Spec

One latent implementation issue; no source-closure blocker identified by
static inspection.

- **P2: Vulkan retains every shape/offset-specialized pipeline for the resident
  model's lifetime.** `vendor/candle-core/src/vulkan_backend.rs:394–421` caches
  complete generated shader strings without eviction. Those strings embed
  layout offsets (`:148–158`), matrix dimensions (`:871`) and copy offsets
  (`:885`). Qwen changes these during ordinary prefill/decode, including
  position-dependent RoPE slices
  (`src/nlu/vendored/quantized_qwen35.rs:234–235`). Switching sessions clears
  inference state but retains the shared weight/device context (`:1202–1203`),
  so previously compiled variants remain resident. Repeated varied prompts
  accumulate pipelines and driver allocations, risking eventual memory
  exhaustion despite bounded context. Parameterize changing dimensions/offsets
  or bound cache residency. Static evidence establishes retention; actual
  growth/OOM was not measured. This concerns the checkpoint's “resident Candle
  integration” promise, but is separate from proving committed-source closure.

The referenced native modules, instruction gate, shader includes and HIP kernel
sources inspected are tracked. No dependency on excluded
`experiments/native-vulkan` prototypes was found.

The specification explicitly calls this “source closure, not completion of the
deployment/performance project.” Vision/projector execution, live KV
compression, held-out safety, current throughput, immutable bundle acceptance
and canonical cutover therefore remain declared follow-up work. The llama.cpp
dependency removal and compatibility migration match the stated scope.

Summary: Standards — zero hard violations, one low-priority duplication finding.
Spec — one P2 unbounded Vulkan pipeline-cache residency issue; memory growth
must be bounded/measured before long-running release acceptance.
