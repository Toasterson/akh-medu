# ADR 096 — Step-Level Rewards from Engine Replay, VSA as Geometric Judge, and Recurrent Depth for Axis-A

- **Status:** Accepted (design; not built). **Superseded in part by [ADR 110](110-kaine-character-guided-reward-live-environment.md) (2026-09-14): negative/redundancy penalties and scalar advantage are replaced by the guided vector with weights in [0,1] and corrected siblings.** Affected: Decision 3 (decoupled scalar advantage) and the "register redundancy as a negative criterion" clause of Decision 4; both are marked in place. Decisions 1, 2 (as narrowed by A1), 5–9 stand. The two `088-*` files noted at the time were reconciled on 2026-09-27 (the action world model is now ADR 117).
- **Date:** 2026-08-15
- **Plan:** [Phase 51 P3-RL](../plans/2026-08-15-phase51-p3rl-step-rewards-and-depth.md)
- **Relates:** ADR 087 (Grug-Brain curriculum — this fills its unbuilt RL stage), ADR 092 (v1 register, D6 "engine replay is the scorer"), ADR 093 (dual-process handoff), ADR 085 (Phase 41 findings — VOC two-regime rule), ADR 052 (unified thinking loop)
- **Measured basis:** [P3-R results](../2026-07-11-phase51-p3r-results.md), [GWT emergence probes](../2026-07-11-phase51-gwt-emergence-probes.md)
- **Anima work item:** `01a004e0-8d5a-7e50-8e82-42b5062d1337` (project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, "Akh-Medu"), phase `draft`

## Context

P3-R (11.7M, A/B on target rendering) cleared its reasoning gate — trace supervision bought **+0.32 axis-B answer accuracy** (0.516 → 0.840) at identical model, data and compute — but left two open failures:

1. Gate 1 near-miss with a diagnostic error ordering: `accuracy 0.840 > executability 0.800 > trace validity 0.770`. Correct answers ride on imperfect traces. SFT cannot fix this: it has no channel to say *which step* was wrong.
2. Gate 3 negative and structurally inverted: axis-A (depths 5–6) trace arm 0.039 vs answer-only 0.105, with axis-A **trace validity 0.000**.

Separately, the rubric-RL literature was reviewed for transferable method. Its central apparatus — an LLM judge grading natural-language criteria — is a workaround for domains that lack a verifier. Our reasoning corpus has one.

## Decisions

1. **Engine replay stays the marking scheme. No LLM judge enters the corpus loop.** `replay_trace` already emits per-statement `(satisfied, step)` pairs — exactly what SRaR ([2605.17291](https://arxiv.org/abs/2605.17291)) pays an LLM judge to produce — deterministically and unfoolably. An LLM judge would be a strict downgrade and would import the verifier-failure taxonomy (partial-compound / implicit-as-explicit / imprecise-verification; 53,447 measured cases, [2605.12474](https://arxiv.org/abs/2605.12474)) that we do not currently have.

2. **VSA judges geometry, never semantics.** Three criteria replay structurally cannot check — step **redundancy** (bound `relation ⊗ subject ⊗ object`, Hamming-compared across steps), **goal drift** (bundle-vs-goal similarity), **coverage** — all in the existing kernel. Redundancy first: it catches the self-correction looping SRaR measured at 48.1% of rollouts, which replay scores 1.0 because each repeated step is individually valid. Semantic judgement is explicitly out of scope; binary-VSA analogy is degenerate, which is the founding reason the neural organ exists.

3. **The RL stage uses SRaR's decoupled advantage over replay-derived per-step rewards.** `Â(t) = outcome advantage + per-step rubric offset`, each normalised separately; the rubric term never enters the group baseline. Per-step normalisation is across sibling rollouts, so only steps whose validity varies produce signal — removing this collapsed training in the source work.
   > **Superseded by ADR 110 Decision 2 (2026-09-14).** No scalar advantage and no policy gradient: per-step scores land in a reward *vector* (`effect`, `validity`, `provenance`, `disposition`, `budget`) reduced to a weight in [0,1]; learning is weighted SFT on corrected siblings plus sibling-contrast preference pairs (ADR 110 rules 2, 3, 7). The replay-derived per-step scores described here remain the input to that vector.

4. **Anti-farming is designed in, not retrofitted.** We are immune to verifier failure but not to rubric-design limitation — the failure that survived even a strong verifier. Concretely, executability rewards *emitting valid statements*, so padding with unnecessary `recall` lines is the predictable exploit. Normalise per **required hop** (`example.depth`), register redundancy as a negative criterion, keep the decoupled advantage. Rubric Dropout ([2608.11669](https://arxiv.org/abs/2608.11669), 30–50%, **one mask per rollout group**) is held in reserve, to be applied only if farming is observed.
   > **Superseded in part by ADR 110 Decision 2 (2026-09-14).** Redundancy is no longer a *negative* criterion: a redundant step trains at weight 0 and its corrected (non-redundant) sibling at its earned weight; no weight leaves [0,1]. The per-hop normalisation and the redundancy measurement (A1 `TraceQualityReport`) stand; "keep the decoupled advantage" is withdrawn with Decision 3.

5. **Axis-A is two failures, and they get separate fixes.** *B-render* (cannot emit a sequence longer than trained) and *B-reason* (cannot compose 5 hops). `trace_valid = 0.000` — no well-formed output at all, not merely a wrong one — is the signature of B-render dominating. The two literatures are not interchangeable and neither may be used to close the other's branch.

6. **The positional-encoding ablation runs before any architecture change.** `TinyTransformer` uses a **learned absolute position embedding** over a `max_seq_len` table (`crates/grug-train/src/transformer.rs:290, 310`). Depth-5/6 traces are longer than anything trained, so their position rows sit at initialisation — a mechanically sufficient cause of `trace_valid = 0.000` that was not previously on the Phase 51 risk list. NoPE beats every explicit scheme on downstream length generalization at zero extra cost ([2305.19466](https://arxiv.org/abs/2305.19466)); randomized PE adds +12.0% across 6000 models / 15 algorithmic tasks by removing the OOD-index failure ([2305.16843](https://arxiv.org/abs/2305.16843)); position coupling by structural role gives 6.67× length extrapolation and maps onto our step-indexed ADR-092 grammar ([2405.20671](https://arxiv.org/abs/2405.20671)). Kohli et al. hit the same split, using APE for 2-hop and switching to NoPE for depth extrapolation.

7. **Recurrent depth is promoted to primary architectural candidate for B-reason, ahead of the Goyal/Bengio slot.** [2604.07822](https://arxiv.org/abs/2604.07822) is a near-mirror of our setup (synthetic KG, from-scratch, `L=4`/`|E|=200`/`|R|=10`, logit-lens analysis, depth extrapolation as a named challenge): vanilla transformers fail, recurrent-depth succeeds, and extrapolation is unlocked by **scaling inference-time recurrence** with no retraining. Adopt their specifics — dynamic training recurrence `R ~ clip(Poisson(λ), 2, 8)`, zero-init `c_proj` on attention and FFN so each block is identity at init, easy-to-hard threshold curriculum, and the expectation of three-stage grokking at ~10⁴ epochs. Supporting: a `k`-layer model looped `L` times ≈ `kL` layers ([2502.17416](https://arxiv.org/abs/2502.17416)) — decisive when the constraint is depth at 13M parameters. The Goyal/Bengio slot stays in the back pocket; its trigger did not fire (gate 4 moved 5→4).

8. **Recurrence halting reuses the VOC two-regime rule.** Overthinking is the named limitation of recurrent depth, corroborated by looped-LM collapse-with-depth ([2605.26733](https://arxiv.org/abs/2605.26733)). Phase 41e already solved the analogous problem and recorded *why* (ADR 085: the first sweep reproduced the illusion-of-thinking failure; halting on diminishing returns is permitted only in refinement, never during frontier search). Mechanism to borrow: training-free halt-when-stable on successive-output KL, which matched uniform depth-8 quality at 4.94 average loops ([2607.14427](https://arxiv.org/abs/2607.14427)).

9. **The founding thesis is unchanged.** Phase 41's VSA loop remains the owner of unbounded depth; `handoff()` already ships the System-2 continuation. Recurrent depth gives the organ *more* depth, not *all* depth. This is a strengthening of the kludge-brain division of labour, not a move of reasoning out of VSA space.

## Consequences

- **Positive.** The RL stage costs almost no new scoring infrastructure — `broken_by_refs` already carries step attribution. VSA gets a real, bounded judging role that plays to hyperdimensional strengths and away from its known weakness. B0 is three GPU runs that may retire a large fraction of a "negative" gate. The GWT probe finding (no revisitable middle register; crossover pinned at 5–6 of 6) becomes actionable rather than merely diagnostic: recurrent depth manufactures the revisitable middle by construction, and `probe_lens`/`probe_swap`/`hidden_from` rerun unchanged over effective depth `L × R`. Our rubric pitfalls are measured, not guessed — the weakness RaR's own ablation identified in synthetic criteria.
- **Negative / risks.** Recurrent depth costs wall-clock linearly in `R` (partly offset: Kohli et al. report `R=4` converging in less absolute wall-clock than `R=2`, on their task). Three-stage grokking means runs must not be killed early, which is expensive to get wrong. Kohli et al.'s task is **implicit** — a single tail-entity prediction, no emitted chain — so it is evidence for B-reason only and must not be allowed to substitute for B0. LOTUS ([2606.31779](https://arxiv.org/abs/2606.31779)) is the closest bridge to trace supervision in latent space but is a 3B result with no evidence at 13M. Weight-tied stacks are unstable without the zero-init discipline.
- **Rejected alternatives.** *(a)* LLM-judge rubric rewards on the reasoning corpus — strictly worse than a deterministic verifier, and imports a hacking taxonomy we are free of. *(b)* VSA as semantic judge — contradicts the degenerate-analogy finding that motivates the organ. *(c)* POW3R-style criterion reweighting as the anti-hacking measure — measured *below* no intervention at all, with the worst overclaim fraction in its sweep. *(d)* Jumping straight to the Goyal/Bengio slot — its trigger did not fire, and recurrent depth is now better evidenced for our exact failure. *(e)* Treating axis-A as a single failure and reaching for architecture first — would spend GPU weeks on B-reason while a learned-APE table plausibly explains the 0.000.

## Amendments

### 2026-08-15 — A1 built: the lab criteria are structural, not VSA

Implementing A1 (`crates/grug-data/src/reason/quality.rs`) showed **decision 2's assignment of these criteria to the VSA kernel was wrong for the lab corpus**. The criteria themselves stand; the substrate changes. Two reasons, the second decisive:

1. **Isolation.** `grug-data` declares "No ML dependency" and the workspace dependency direction is one-way — the engine depends on `grug-data` (`Cargo.toml:176`), never the reverse. Reaching `src/vsa/` from there would invert it and pull the 113K-LOC engine (oxigraph, redb, hnsw, candle) into the isolated data crate, breaking the isolation ADR 087 §3 established for exactly this reason.
2. **VSA is the wrong instrument here.** Hypervector similarity is what you reach for with no ground truth and a need for *graded* resemblance. In the lab every entity is already a discrete symbol and `ReasonExample::gold_chain` **is** the ground truth, so exact set membership answers all three questions directly — no dimension, no seed, no false-match rate. A cleanup step would only add approximation error to a comparison that is currently exact.

**The split this establishes, which is the durable point:**

- **Lab** (`grug-data`, gold chain available) → structural scorers. Exact, dependency-free, deterministic.
- **Production** (engine, Phase 41/P6, no gold chain — only a goal vector and a trace) → VSA geometry via the real kernel and `CompositeVerifier`. Unchanged, and still the only option there.

Decision 2 is therefore **narrowed, not withdrawn**: VSA judges geometry, never semantics, *and only where ground truth is absent*.

Delivered: `TraceQualityReport` / `TraceQualitySummary` with redundancy, `max_claim_repeats`, drift, `hop_coverage` (the required-hop denominator decision 4 mandates) and `wasted_valid_steps` (the padding exploit, counted). Wired into `ReasonEvalSummary`, so every `eval_reason` run reports it and the retro-score is a re-run rather than a new tool. `is_looping()` thresholds at ≥3 repeats of one claim or >25% redundancy, so ordinary single-repeat backtracking is not flagged. A test asserts the load-bearing property directly: a 3×-duplicated gold trace scores `executability == 1.0` under replay while quality reports `redundancy_rate == 2/3` and `is_looping()`.

## Open questions

- Does redundancy actually occur in our traces? A1 is a no-GPU retro-score over `runs/p3r-trace` and settles it before any guard is built.
- Does NoPE move axis-A `trace_valid` off 0.000? Splits the axis-A null cleanly between the two branches.
- Systematic generalization (composing atomic facts never composed in training) is Kohli et al.'s other axis; `grug-data` has no `C_ID`/`C_OOD` split for it. Worth a third axis, deferred.
- Whether latent-space step supervision (LOTUS shape) survives at 13M, or whether explicit trace tokens remain necessary at our scale.
