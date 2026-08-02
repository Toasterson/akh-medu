# Columbus-1 / Metacognitive Steering — Provenance Audit and Technical Assessment

> **Date:** 2026-08-02 · **Status:** Research (no decision taken) · **Author:** session read, unbranched (`main` @ `7e13f29`)
> **Subject:** Autopoiesis Sciences, "Columbus-1: The First Judgment Model Distilled From Science in Motion", published 2026-07-31
> **Primary sources:** [X thread](https://x.com/autopoiesislab/status/2083221810171224416) · [blog post](https://autopoiesis.science/blog/columbus-1/) (both retrieved 2026-08-02)
> **Relevance to us:** [ADR 085](../decisions/085-thinking-engine-findings.md) finding 3 (two-regime halting), [ADR 052](../decisions/052-unified-thinking-loop.md), [ADR 086](../decisions/086-feature-test-vm-harness-benchmarks.md) (Phase 50 harness)
> **Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu")

---

## 0. Why this is in the repo

Two reasons, and only two.

1. **One of their claimed findings independently rhymes with ours.** They report that the "Challenging" cognitive state sits 130–142° from the other two states in activation space, and that you must therefore *hard-route* into it rather than blend it. We arrived at a structurally identical rule from the opposite direction — control policy, not representation geometry — in Phase 41e (ADR 085, finding 3). If both are real, that convergence is worth a paragraph in whatever we eventually publish about the thinking loop.
2. **It is a clean worked example of the evaluation gap Phase 50 exists to close.** Their headline results have no control arm. Writing down precisely *which* experiment is missing is cheap now and useful when we scope our own benchmark claims.

This document is **not** a recommendation to adopt anything from it. Section 6 is the verdict.

---

## 1. What is claimed

Faithful summary; every sentence here is *their* claim, not an established fact.

**The method (Metacognitive Steering, "MCS").** A sparse neural controller reads the residual stream of a *frozen* trillion-parameter model at layer 36, classifies the current "cognitive regime", predicts a blend of steering modes within that regime, and writes a four-row steering matrix back into layers 36, 38, 42 and 48 on **every decode step**. No weights change; the model is stated to be bit-identical between steered and unsteered runs.

**Base model.** Kimi K2.6, disclosed as 1T total / 32B active per token, 61 layers, 384 routed experts (8 + 1 shared per token), 7,168 hidden dimensions.

**The training signal.** Not papers — interaction traces from *Aristotle*, their AI research partner, used by working scientists on real long-horizon problems. The pitch is the file-drawer argument: papers record the path that worked, judgment lives in the paths that were abandoned, and those decisions are erased before publication. From those traces they extract "tens of thousands of contrastive nudge pairs" (identical context, two completions differing only in cognitive approach), captured at layers 36/38/42/48 via a tail-30-token mean at the last token position.

**The three regimes** (unsupervised clustering of layer-36 absolute activations; functional labels assigned afterward by humans reading traces):

| State | Stance | Claimed geometry |
|---|---|---|
| Deliberative | explore | primary reasoning manifold |
| Methodical | converge & execute | ~85° from Deliberative — near-orthogonal, "safe to blend" |
| Challenging | scrutinize | 130–142° from the other two — *opposes* them |

**The routing rule.** Hard-switch between states, soft-blend among modes *within* a state. Their line: "Diluting a brake with an accelerator gives you neither."

**The system (Columbus-1).** MCS + a 32B "bias policy" that injects strategic nudges into hierarchical planning + a persistent semantic memory graph holding hypotheses, evidence, decisions, methods, facts and **rejected paths**, linked by support / contradiction / causal edges, so the system can revisit weak signals and abandoned branches.

**The two headline results.**
- **BlueZ:** 10 crash sites → 8 confirmed attacker-reachable findings against BlueZ 5.86, independently reproduced by partner firm Xchg Labs. The top one is described as a zero-click, pre-auth, pre-pairing full-chain RCE: an attacker-controlled advertised name copied into a fixed 250-byte stack buffer in the privileged `bluetoothd`. Disclosed to BlueZ maintainers 2026-07-06.
- **Rocket:** a 10-ft vehicle that lands propulsively on *non-throttleable, non-extinguishable* G-class solid motors. Columbus-1 claimed to have selected the architecture, generated parametric CAD + STL/STEP/DXF, written the flight software, and directed human fabricators over iMessage. Claimed 99% safe landings across 100 fully-dispersed 6-DoF runs, and three control principles it says nobody supplied (optimise arrival velocity *jointly* with remaining impulse; stabilise attitude before terminal propulsion; stage motors on predicted time-to-contact rather than fixed altitude).

---

## 2. Provenance audit — the paper does not exist

The blog references "the frozen analysis snapshot **described in the preprint**". That phrase carries no hyperlink. This is conspicuous because the same page links *every other* citation: seven prior dynamic-steering papers, LEACE, Popper, Dwan et al. 2013, Franco et al. 2014, the Bluetooth SIG, bluez.org. Their own preprint is the single unlinked reference on the page.

Checks run 2026-08-02, all negative:

| Check | Method | Result |
|---|---|---|
| arXiv title/abstract/full-text | `arxiv.org/search` for `"Metacognitive Steering"` | **0 results** |
| arXiv author page | `arxiv.org/a/autopoiesis` | 404 |
| Authors by name | web search: Karpf, Xing, Song, Reth + method terms | no paper; only the blog and unrelated hits |
| Their own site | `firecrawl_map` over `autopoiesis.science` | **5 URLs total**: home, blog index, and three posts (Columbus-1, Oracle partnership, GPQA Diamond). No papers/research section |
| Outbound links on the post | `firecrawl_scrape` links format | 23 links, all citations resolve *except* the preprint, which has no anchor |
| Research/PDF corpora | `firecrawl_search` with `categories: [research, pdf]` | nothing of theirs |
| Follow-up announcement | their X timeline (21 posts; visible: pinned Columbus-1 thread, Oracle post 2025-08-14, GPQA post 2025-07-31) | no paper link posted since |

**Consequence.** Every number in Section 3 exists only as prose in a marketing post. There is no methods section, no ablation table, no released code, no released data, no released steering vectors. This matches their 2025 pattern: the 92.4% GPQA Diamond claim also shipped as a blog post rather than a paper.

**Note on the vulnerability side** (checked separately, and *not* damning): nothing matching the described bug appears in the Debian BlueZ tracker, OpenCVE, or NVD. The nearest recent record, CVE-2026-64539, is a kernel-side `eir` stack OOB write when *prepending* the Flags AD — different component, different direction of data flow, not theirs. Disclosure was 2026-07-06, four weeks before the post, so an active embargo is a completely ordinary explanation. The honest statement is just: as of 2026-08-02 the strongest claim in the post rests on one partner firm's word.

---

## 3. Evidence grading, claim by claim

Ordered by how much I'd bet on each.

**Likely real, and the most interesting thing in the post — the Challenging state is oppositional, not a dial position.** Two independent lines are reported to agree: contrastive *difference* geometry and unsupervised *state* clustering both return three groups, with one of them anti-aligned. The engineering consequence (hard-route, never blend) is a specific, falsifiable design rule, and it is not the conclusion a marketing team reaches by default — the tidy story would have been one smooth exploration↔exploitation dial. See Section 5 for why I weight this up.

**Plausible but unmeasured — that steering changes trajectory shape.** Their evidence is two "small matched process probes" rendered as trajectory strips: baseline commits early and develops one synthesis; steered run keeps branches alive longer, develops one far enough to expose its failure mode, then *prunes after deep investment*. That last behaviour is the genuinely notable signature. But this is a qualitative demo at n≈2 tasks, explicitly not benchmarked for answer quality, with no inter-rater protocol on the trace labelling.

**Weaker than the prose implies — the statistics.**
- Three cross-layer SVD components clear a permutation null at **1.54 / 1.31 / 1.25×** threshold. That is a modest effect presented as a decisive one, and the fourth component doesn't clear at all. The claim "the channel is about three dimensions wide" is doing a lot of work on those margins.
- **Cramér's V = 0.993** between states and steering modes is close to tautological: both clusterings derive from the same contrastive pairs at the same layers, so the positive/negative labelling drives both. "Difference geometry and state geometry had every opportunity to disagree" overstates the independence. To their credit, the post hedges one sentence later ("not proof of a mechanism").
- **~99% cross-validated accuracy from 10 neurons** is striking but unauditable without the data, and deployment reportedly uses a broader stability-selected readout plus a separate 20-neuron interrupt detector — i.e. the headline sparse number is not the shipped configuration.

**Unsupported as stated — that MCS is what produced the BlueZ and rocket results.** This is the load-bearing gap. Columbus-1 is 1T base model + 32B bias policy + persistent memory graph + CAD/sim/search/exec tooling + MCS. The results are attributed to the steering. **No run of the same scaffold with steering disabled is reported.** Absent that arm, the parsimonious explanation for 8 bugs in `bluetoothd` is a competent agentic harness with sanitizer feedback and persistent cross-branch state — which several groups now have, and which is roughly what our own Phase 50 exec1 runner is shaped like.

**Rhetorical, not technical.** The framing tweet — "Reinforcement learning rewards convergence, and convergence is the opposite of discovery" — is half true and half strawman. Entropy/mode collapse under RLVR is real and well documented, and their "RL keeps searching, but it keeps searching the same place" is a fair description of it. But their own fix is contrastive directions distilled from expert traces: that is imitation learning of *stance*. It is a better-shaped imitation target, not an escape from imitation, and the post's framing implies otherwise.

**Ambiguity worth flagging, not an accusation.** The "32B bias policy" and Kimi K2.6's "32B active per token" share a number. The post never says whether the bias policy is a separate model, and the architecture figure doesn't disambiguate.

---

## 4. Prior art

They cite seven dynamic activation-steering works: PDS (2510.05498), DMS (2510.22099), CLAS (2604.24693), A-LQR (2604.19018), SVF (2602.01654), DAC (2024.blackboxnlp-1.34), SADI (OpenReview `8WQ7VTfPTl`), plus LEACE (2306.03819) for erasing the dominant shared direction before measuring alignment. Their novelty claim over these is *scale*: prior dynamic steering demos reached 70B, MCS runs at 1T.

**They do not cite the nearest published neighbour.** [arXiv:2605.08942](https://arxiv.org/abs/2605.08942), Li, Bai, Liu, Zhang & Zeng, "Decomposing and Steering Functional Metacognition in Large Language Models" (submitted 2026-05-09, 18pp, cs.CL) proposes that LLMs maintain a *decomposable space of functional metacognitive states* — evaluation awareness, self-assessed capability, perceived risk, computational effort allocation, audience adaptation, intentionality — shows they are linearly decodable from residual-stream activations with distinct layer-wise profiles, and demonstrates that steering along probe-derived directions **causally and dissociably** modulates reasoning behaviour. Code and data public at `github.com/xlands/meta-cognition`.

That is the same shape of claim, three months earlier, with a methods section and released code. It doesn't invalidate MCS — different states, different scale, different application — but it substantially complicates "the world's first judgment model", and its absence from an otherwise-thorough citation list is notable.

---

## 5. Where this touches Akh-Medu

**5.1 The two-regime convergence (the reason this doc exists).**

ADR 085 finding 3: our first VOC sweep applied diminishing-returns halting uniformly across the whole loop and reproduced the illusion-of-thinking failure — the controller halted mid-search because verifier quality is blind to exploration progress. The shipped fix splits search into *exploration* (no answer yet; progress = novelty / frontier consumption) and *refinement* (answer in hand; progress = verifier-score delta), and permits diminishing-returns halting **only in the second**. Finding 4 records that the convergence sub-verifier needed the identical split. ADR 085 recommendation 2 promotes this to a standing design rule for every future halting/valuation signal.

Autopoiesis report that scrutiny is 130–142° from deliberation and execution, that it must be hard-routed rather than blended, and that a dedicated 20-neuron interrupt detector overrides the primary classifier to force entry into it.

These are the same claim in two different substrates: **the mode that says "stop / this is wrong" is not a point on the same axis as the modes that make progress, and treating it as one destroys both.** We found it in control-policy space by breaking the loop; they claim to have found it in activation space by measuring angles. Neither result depends on the other, which is exactly what makes the agreement worth recording.

Caveat before we lean on it: our finding is measured and reproducible in `src/thinking/`; theirs is an unaudited number in a blog post. The correct use is "independent convergence, one side unverified", never "confirmed by Autopoiesis".

**5.2 Their memory graph is our KG, described in marketing register.** Hypotheses, evidence, decisions, methods, facts, rejected paths, linked by support / contradiction / causal edges, enabling revisiting of abandoned branches. That is the Akh KG plus `TruthMaintenanceSystem::was_retracted` (the 44g supersede filter) plus DS-massed competing hypotheses (Phase 17) plus `DerivationKind` provenance. Nothing here to import; it's mild external validation that "persist the rejected branches, not just the conclusions" is the right architecture. Our version is strictly more explicit about *why* a branch died.

**5.3 What they have that we don't: judgment traces from real scientists.** Their compounding loop is Aristotle (deployed to working researchers) → judgment traces → control signal → better investigations → more traces. That is a genuine data moat and we have no equivalent. It is also the part of the post I believe most readily, because it requires no mechanism to be true — just a deployed product and instrumentation.

**5.4 Implication for Phase 50 (ADR 086).** The missing control arm here is a direct lesson for our own benchmark claims. Columbus-1 attributes discovery to one component of a five-component system with no ablation. When our agentic illumos sysadmin corpus and fleet-orchestration bench produce results, the single-akh control arm and the strategy-off arm are what make them citable — the same discipline as the 44j deterministic A/B harness (`src/bench_phase44/`). If we ever claim "the thinking loop found X", we need the run with `deliberate()` disabled in the same report.

**5.5 Not applicable to us: the mechanism itself.** MCS requires residual-stream read/write access on every decode step. Under ADR 070 our default backing is Ollama (no activation access), and even the Candle in-process path (ADR 067) would need the hook plumbing plus a labelled contrastive corpus we don't have. More to the point, our steering already happens *above* the model — the ThinkingLoop scheduler, the VOC controller, the CompositeVerifier — where it is faithful and auditable, which is the whole ADR 052 bet. Intervening in activation space would trade that away. **No action.**

---

## 6. Verdict

**A well-produced capability announcement, not a result.** Treat it accordingly: nothing in it is replicable, and nothing in it should be cited by us as evidence for anything.

The one idea worth carrying forward is the oppositional-scrutiny finding, and only as *independent convergence on our own measured Phase 41e result* — flagged as unverified on their side every time it is mentioned.

**What would change this assessment**, in descending order of value:

1. The ablation: same Columbus-1 scaffold, MCS disabled, run on held-out CVE-bearing code. Without this the capability claims are unattributable, full stop.
2. A published preprint with the SVD/clustering methodology, the null construction, and the mode-labelling protocol.
3. A CVE with BlueZ maintainer confirmation for the zero-click chain.
4. Released steering vectors or the contrastive-pair corpus (even a sample), which would let anyone reproduce the three-regime geometry on an open model.

**Revisit trigger:** any of items 1–3 landing. Nothing here justifies a design change, a plan, or an ADR today.

---

## Appendix — retrieval log

All 2026-08-02, via `firecrawl_scrape` / `firecrawl_search` / `firecrawl_map` except where noted.

- `x.com/autopoiesislab/status/2083221810171224416` — 6-post thread, read via in-app browser
- `x.com/autopoiesislab` — profile timeline, in-app browser (21 posts, 4,941 followers, joined Jan 2025)
- `autopoiesis.science/blog/columbus-1/` — full post, markdown + links formats
- `autopoiesis.science/` — homepage (team, backers, advisory board)
- `firecrawl_map` of `autopoiesis.science` — 5 URLs
- `arxiv.org/search/?searchtype=all&query="Metacognitive Steering"` — 0 results
- `arxiv.org/a/autopoiesis` — 404
- `arxiv.org/abs/2605.08942` — full abstract page (the uncited prior art)
- Vulnerability cross-check: OpenCVE `vendor=bluez`, Debian security tracker `source-package/bluez`, NVD CVE-2026-64539

Team as listed on the post: Vincent Karpf, Jiehao Xing, Jialing Song, Joseph Reth (co-founder/CEO), with ~14 named supporting contributors. Backers: Informed Ventures (formerly GSR), Alpaca VC, VitalStage, Cross Atlantic Angels. Advisory board includes Hannes Jónsson (creator of the Nudged Elastic Band algorithm).
