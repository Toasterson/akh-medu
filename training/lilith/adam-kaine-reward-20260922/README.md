# ADAM/Kaine mechanisms for Lilith — reward curriculum

> **Rejected before training:** this first design verifies action-list order but
> does not verify a real environment effect. It must not supply policy reward.
> It is retained as negative design evidence. The replacement is
> `adam-kaine-code-env-20260922`, where generated code is checked by the frozen
> bounded interpreter and then executed under native illumos for admitted
> audit candidates.

This slice transfers the behavior and training mechanisms into Lilith's
Gemma-4 parent. It does not train the sovereign ADAM or Kaine lineages.

ADAM currently means the repository's `small_8k` BabyLLM lineage: d_model 512,
8 layers, 8 heads, d_ff 2048, RoPE, 8192 context, and recompute attention. Its
33M MobileLLM/SWA, 350M hybrid, and approximately 1.1B rungs remain P7 plans.
Kaine has no fixed size, architecture, checkpoint, or training preset yet;
ADR 111 makes those outputs of the 350M and 1.1B measurements. Inventing Kaine
weights here would erase the experiment the architecture plan requires.

## What is executable now

`adam_kaine_reward.py` implements ADR 110's five-component per-step reward:
effect, validity, provenance, disposition, and budget. Integrity, honesty, and
decency are hard gates. A failed trajectory receives zero policy reward and a
corrected sibling for the separate SFT control; the model never supplies its
own verifier receipt. The recorded CPU environment creates receipts from the
task, action, observation, and step index.

The two initial behavioral families come directly from ADR 110:

- `act_to_perceive`: observe twice, identify the invariant across changed
  observations, then report;
- `emergent_property_bug`: instrument first, test a bounded number of
  hypotheses, delegate after repeated failure, then fix and report.

These measure actions and verified effects rather than persona words. Four
development tasks and four state-distinct independent-evaluation tasks were
frozen before policy training. The manifest is
`a491b9bd86fd52b1f4ca5dd756e218bd31e9b6e4ff5baae5c9f7961d917a2734`.
No real transcript session had been admitted at freeze time, so the initial
session denylist is empty; four exact `sealed:<task-id>` family tags are frozen,
and any later evaluation-session hashes must be appended before export.

`admit_verified_transcript()` accepts the transcript exporter contract only
for `verified_outcome` records with a passed external verifier, SHA-256-bound
trajectory outcome, non-empty steps, and no denied evaluation session or
family. Supervised imitation and self-reported completion cannot create RL
reward.

## Training protocol

`training-protocol.json` freezes four same-parent arms:

1. base, without an update;
2. immutable corrected-sibling weighted SFT as the distillation control;
3. three sequential online KLPO updates, each with a fresh one-use probability
   bank and externally verified `klpo_reward`;
4. the breed arm, which starts from the SFT adapter and then takes the same
   three fresh KLPO updates.

Candidate selection may use development tasks and pre-existing frozen
canaries. All candidates must be frozen before the independent evaluation is
opened once. The evaluation may reject a candidate but may not select, merge,
or tune it. Release evidence must include the five reward components by
family, prior coding and formatting retention, and cold reload. The protocol
hash is `207026d54229f5e7c4a8fe5921a04e5428837b8a0aac6c134f8cd08bc38d85e8`.

The existing KLPO path supplies real policy updates and the actual sampled
token/q-log-probability contract. This module supplies its environment reward
and split discipline. The full ADR 110 `AkhEnvironment` with Engine, complete
ToolRegistry, Agent Board, and Anima dojo tenancy is still unbuilt; this first
curriculum is a deterministic recorded environment and is labelled as such in
the manifest.

Eleven tests cover split isolation, hard reward gates, independent receipts,
corrected siblings, transcript outcome binding, and the real-RL/SFT/breed arm
separation.
