# KLPO decision for the Lilith run

> **Note (2026-09-27):** [ADR 116](../decisions/116-lilith-chief-of-staff-guided-rl.md) Decision 4 puts Lilith's training on the [ADR 110](../decisions/110-kaine-character-guided-reward-live-environment.md) guided reward (every step scored, corrected siblings, weights in [0,1]). The terminal-reward KLPO arm described below is kept **only as a verifier-only ablation arm**; it is not the Lilith training objective.

On 2026-09-21 the user supplied [Yifan Zhang's KLPO post](https://x.com/yifanzhang_/status/2101832813293003160), then authorized recording the assessment and running the next slices. KLPO is an experimental training option; the post's broad Q* claim is not an established project result.

## Evidence and fit

The [official repository](https://github.com/yifanzhang-pro/KLPO) supplies a critic-free objective with one complete response per prompt and independent auxiliary token samples for its default Monte Carlo correction. It does not require a learned value model or same-prompt response group. The authors explicitly distinguish their CPU verification from unvalidated GPU training and paper-scale benchmark reproduction. The pinned source used for this run is recorded with its evidence under `training/lilith/rl-20260921`.

The [algorithm contract](https://github.com/yifanzhang-pro/KLPO/blob/main/docs/algorithms.md) requires the actual collection sampler's probabilities, independent auxiliary draws, and complete policy-token masks. The [training guide](https://github.com/yifanzhang-pro/KLPO/blob/main/docs/training.md) requires positive-temperature sampling without support truncation in its supported recipe. Its supplied Molt deployment targets NVIDIA/CUDA; our AMD implementation must be measured independently. A single response still involves extra sampling/scoring work, and the tiny previous LoRA memory result does not establish practical long-trajectory capacity.

Lilith already has independent artifact verification, durable lesson provenance and a working, bounded ROCm LoRA update/save/reload mechanism. These are useful prerequisites. Text-only historic evaluations are not valid KLPO records: they lack the actual sampler probabilities and auxiliary draws. Recomputing them using BF16 training weights would not reproduce the probabilities of a quantized Vulkan sampler. Those evaluations remain development evidence, not silently retrofitted RL experience.

## Authorized sequence

1. Implement a versioned trajectory record and independently test the token-regression/MC loss. Record model/adapter/tokenizer/prompt versions, real action probabilities, separate auxiliary sampling, masks, terminal reward, verifier receipt and completion reason. Keep failed and truncated attempts visible; do not optimize incomplete trajectories.
2. Freeze a small independent task pack, training/held-out family split, raw verifier-based rewards and resource budgets before live sampling. Use fresh exercises to investigate complete input-domain validation. Do not modify the earlier evaluation or train on its held-out solutions. Public development tasks are not sealed certification.
3. Run synchronous generation and a bounded LoRA update in the same pinned HF/ROCm implementation. A single resident base with small adapter snapshots avoids simultaneous full-model residency. Record initial sampler/trainer probability mismatch rather than assuming identical libraries imply identical kernels. Measure realistic context memory before the live comparison.
4. Compare unchanged, supervised and KLPO adapters under matched task and update budgets, explicitly accounting for their different supervision. Preserve all outcomes and actual compute. A one-update run establishes integration and produces descriptive observations, not statistically demonstrated improvement. Cold-process reload must be exercised separately from the prior same-process mechanics check.
5. Require sufficient diagnosed data, retention and independent quality gates before useful curriculum training or promotion. Validate adapter application/export in Vulkan separately. Native illumos validation remains deferred until VM host setup is ready.

## Reward and ownership boundaries

Independent execution checks supply coding rewards. Preserve individual failing inputs and distinguish semantics from output-format constraints. Do not reward reviewer agreement alone or increase reward by modifying the verifier. Terminal rewards do not identify which earlier action caused a failure; this first single-turn experiment avoids claiming solved multi-agent credit assignment.

A later coordination experiment may reward verified delegated outcomes, recovery and evidence-backed reports. Only Lilith's policy actions belong in Lilith's policy loss: peer replies and tool observations are external context. Peer-generated examples belong to attributed supervised teaching unless a separately justified off-policy method applies.

VSA remains a source of evidence in context, not a replacement for RL. When retrieval is enabled, pin the selected lesson envelopes and context with the trajectory so the conditioning information can be reconstructed. This first bounded experiment does not claim VSA improvement.

The deployed Vulkan GGUF and existing approval rules remain unchanged. No native VM provisioning, production adapter activation or external publication is part of this run.
