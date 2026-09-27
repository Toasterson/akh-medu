# ADAM/Kaine mechanism curriculum — real coding slice

This slice trains **Lilith's Gemma recipient** with mechanisms derived from the
ADAM/Kaine work. It does not claim to train an ADAM or Kaine sovereign model.
ADAM's landed model is BabyLLM `small_8k`; Kaine has no frozen architecture or
weights yet, so no Kaine weight lineage is invented here.

The replacement environment in `adam_kaine_code_environment.py` scores real
Python functions against 4,877 finite cases across twelve tasks: four training, four disjoint
development-selection, and four new-family independent-evaluation tasks. The five
reward components bind actual observations, source artifact, verifier outcome,
disposition, and bounded execution. The graded KLPO reward is 95% independently measured correct-case fraction
and 5% bounded grammar admission; exact semantic success scores one. Model text cannot mint a receipt. `native-reference-audit.json`
records all twelve canonical programs passing in the pinned illumos guest.

The transcript corpus is a separate distillation source. Its 559 rows comprise
359 train and 200 validation rows, all `supervised_imitation`; it contributes no
RL reward. The training runner accepts only a frozen train subset and refuses
over-budget examples rather than truncating them. Validation sessions never
enter fitting.

`adam_kaine_gemma_train.py` defines four same-parent arms: base, transcript SFT,
three fresh on-policy KLPO banks, and a fixed 0.5 coordinate interpolation of
the independent SFT and KLPO adapters. Candidate rollouts enable Gemma thinking and retain full reasoning-token log probabilities; the external verifier strips only complete Gemma `<|channel>thought … <channel|>` blocks and scores the remaining answer; incomplete or stray channels fail the reasoning gate. The mechanics pilot starts at 512 total/320 new tokens. If every base training rollout truncates, a protocol-frozen resource-only branch first proves gradients at 1,024 tokens, then reruns from the unchanged policy at 1,024 total/800 new. An out-of-memory result is a resource block; truncation is never coding-failure evidence. Each online bank is bound to the current adapter hash and
unique RNG namespace; durable claims prevent replay. Only the four development-holdout task scores choose among the two parents
and blend under a frozen rule; training-bank tasks never select an arm and the decision is durably frozen before the selected candidate and unchanged base receive paired independent-eval rollouts with common random numbers. Eval outcomes never change the candidate or trigger release. The
blend is actual same-parent/rank/target LoRA tensor interpolation. It is not a
mathematical claim about delta averaging and does not guarantee improvement.

The earlier `adam-kaine-reward-20260922` action-order design is rejected and
must not provide training reward.

## CPU validation

```sh
python3 scripts/session-platform/test_adam_kaine_code_environment.py
python3 scripts/session-platform/test_adam_kaine_code_rl_tasks.py
python3 scripts/session-platform/test_adam_kaine_gemma_train.py
python3 -m py_compile scripts/session-platform/adam_kaine_gemma_train.py
```

The GPU runner has not been executed by this slice. A completed run must retain
all bank trajectories, consumption claims, adapters, config, result, and a
second-process exact cold reload receipt before any training-success claim.
