# Lilith: first bounded KLPO run — 2026-09-21

The approved [research assessment](../../../docs/ai/research/2026-09-21-lilith-klpo.md) is recorded and the probability-capture, tested loss, supervised control and actual RL-update slices have run. **Training mechanics passed; coding improvement was not demonstrated.** This is a scratch HF/ROCm experiment on the pinned Gemma 4 parent. It does not change the deployed Vulkan model or supply native illumos or Qwen-parity evidence.

## Observed behavior and reward diagnosis

[Independent replay](independent-report.json) verifies all 24 trajectory receipts against the frozen task verifier and sampler/task provenance.

| Arm | Strict training | Strict public family holdout | Logic checks, including unscored fence extraction |
|---|---:|---:|---:|
| Initial adapter | 1/4 | 0/4 | 8/8 |
| One supervised update | 1/4 | 0/4 | 8/8 |
| One KLPO update | 1/4 | 0/4 | 8/8 |

All responses completed. Each arm has seven syntax failures caused by one enclosing Markdown code fence. The independent diagnostic removes only that wrapper, without rewriting code, and all extracted programs pass their finite cases. The diagnostic is **unscored**: original rewards, failures and scores remain intact. Thus the mixed training rewards primarily distinguish formatting on this pack; they do not demonstrate a contrast in coding correctness. Finite checks are not universal proofs.

This tiny public pack, one update and paired task RNG seeds support mechanics observations only. Supervised references and RL rewards provide different information; this is not an equal-information or statistically powered comparison. Future quality work needs a separately versioned artifact/reward contract and fresh tasks that expose actual logic defects, including the previously diagnosed domain-validation failures. Do not tune against these public holdouts or silently rescore this run.

## Actual execution

The [frozen config](live/run2/config.json) and [result](live/run2/result.json) record four training tasks, four disjoint-family development tasks, full-support temperature-1 categorical sampling, eight independent auxiliary token draws per generated prefix, 128 generated-token and 384 context-token limits. Actual action probabilities and token IDs are captured at collection time; historic Vulkan text cannot substitute for them.

Both arms reset to the same initial rank-2 adapter and use fresh AdamW state, learning rate `1e-4`, and one update. SFT uses 135 canonical training-target tokens. KLPO uses all four complete initial training responses with rewards `[0, 0, 0, 1]`, beta `0.1`, and the mean over responses of the summed token objective. Its [durable consumption receipt](live/run2/klpo-consumption.json) records the single-use bank claim; failed outcomes are retained. The predeclared all-equal-reward skip is a conservative experiment rule, not an algorithm requirement.

The initial action-log-probability drift check found `0.01348149` with scoring chunks of eight. The predeclared fallback to one position reduced it to `2.2164e-7`, below the `1e-4` gate, before either update. Both updates use that scoring path. Adapter fingerprints changed; frozen base tensor mutation versions remained unchanged. The run took 402.3 seconds with peak allocation 24,297,685,504 bytes (22.63 GiB), without CPU offload.

Run 1 was cancelled during parent loading, before generation, to add the frozen task-manifest gate. Its cancellation receipt is retained. Run 2 is the sole collected comparison; source snapshots and raw trajectories are retained under [live](live/).

The [cold-process reload](live/cold-reload-run2/result.json) also passed: fresh PID 302072 loaded the updated KLPO candidate from collector PID 300816, verified file and tensor fingerprints, and reproduced the fixed-prefix log probabilities exactly. This validates the saved scratch candidate in the same HF/ROCm configuration, not Vulkan export.

## Verification and evidence

- [Pinned upstream implementation](reference.json), [frozen tasks](tasks.json), and [task tests](task-tests.log): eight authored references pass 7,667 finite cases; eight negative controls fail.
- [Trajectory/loss core](../../../scripts/session-platform/lilith_rl.py) and [17 tests](unit-tests.log): independent sampler draws, mask and completion rules, immutable provenance, holdout exclusion, one-use banks, and exhaustive tiny-vocabulary gradient agreement to `1e-12`.
- [Live runner](../../../scripts/session-platform/lilith_rl_probe.py), raw [base](live/run2/base-trajectories.json), [SFT](live/run2/sft-trajectories.json), and [KLPO](live/run2/klpo-trajectories.json) receipts.
- [Independent report tool](../../../scripts/session-platform/lilith_rl_report.py) and [six tests](report-tests.log), including forged reward receipts, missing/duplicate arms, manifest drift and unscored wrapper diagnostics.
- Regression checks: [108 Lilith tests](lilith-integration-tests.log), [four adapter preflight tests](preflight-integration-tests.log), [four legacy trainer safety tests](trainer-integration-tests.log), plus the six report tests: **122 passing tests** in total.

The next quality slice should evaluate meaningful failure corrections with fresh transfer tasks and retention checks. Vulkan adapter application/export, broader coordination behavior, Qwen comparison and native illumos validation remain separate gates.
