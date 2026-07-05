# High Grug — A Designed Dense Reasoning Language with Custom Token IDs — akh-medu

- **Date:** 2026-07-04
- **Status:** Research draft (no implementation). Fourth doc in the symbol-language series; builds on the [two-lineage survey](2026-06-23-symbol-language-state-of-the-art.md), the [coverage-gap addendum](2026-06-23-symbol-language-coverage-gap-addendum.md), and the [Egyptian design-rationale companion](2026-06-23-egyptian-writing-design-rationale.md).
- **Scope:** Can akh-medu train its models to *reason in* a designed, token-dense, exact language ("High Grug") whose symbols are KG entities/relations minted as custom token IDs and streamable into context?
- **Method:** 3 parallel WebSearch/WebFetch subagents (127 tool calls total; Firecrawl unavailable — credits exhausted until 2026-07-07, so claims here are sourced but **not** adversarially 3-vote verified like the prior three docs). Synthesized against the verified two-lineage survey.
- **Author:** Claude (research agent), commissioned by Till.
- **Naming note:** Official name: **High Grug** (Till, 2026-07-04). "Grug speak" is community shorthand (from [grugbrain.dev](https://grugbrain.dev/)), not literature terminology — the papers say "illegible CoT," "language mixing," "neuralese." The "High" is apt twice over: in diglossia terms High Grug is the **H-variety** — the formal, written, dense reasoning register — against the low gloss (vernacular English for humans); the same register split Egyptian ran for three millennia between hieroglyphic (monumental, dense, sacred) and demotic (cursive, everyday). The name encodes the dual-channel design. Egyptian-flavored alternative held in reserve: *medu-akh*, after *medu-netjer*.

---

## 0. The vision (what this doc evaluates)

> Train the akh's model to use a **designed reasoning language**: **High Grug** — very dense per token, delivering *exact* meaning — into which we can **stream handmade symbols carrying custom token IDs** (SymbolId → token ID), so the KG's entities, relations, and classifiers become first-class vocabulary the model thinks in.

Verdict up front: **the literature supports every component, warns precisely where it breaks, and the exact combination is unpublished.** The two-lineage survey supplies the vocabulary structure and governance; the 2024–2026 token literature supplies the mechanics; akh-medu already owns the executor, the grounding layer, and the training seams.

---

## 1. The phenomenon: models already drift into Grug

RL-trained reasoning models spontaneously develop compressed, telegraphic, idiosyncratic CoT dialects:

- Karpathy, Sept 2024, days after o1: *"You can tell the RL is done properly when the models cease to speak English in their chain of thought"* ([tweet](https://x.com/karpathy/status/1835561952258723930)).
- o3's published traces (Apollo × OpenAI anti-scheming paper) show a stable idiosyncratic dialect — *"disclaim," "watchers," "vantage," "marinade," "illusions"* used in non-standard ways; OpenAI states it is investigating why ([snippets](https://www.antischeming.ai/snippets), [arXiv:2509.15541](https://arxiv.org/abs/2509.15541)). The word "marinade" appears **8,344×** above human base rate in o3 CoT ([1a3orn typology](https://1a3orn.com/sub/2025-10-weird-cot.html)).
- **The dialect is load-bearing:** forcing 14 models to rely only on the *legible* portions of their CoT raised error ~36–53%; illegibility↔accuracy correlation ≈ 0 (r=0.061) — dense segments carry reasoning, not noise ([Jozdien, arXiv:2510.27338](https://arxiv.org/abs/2510.27338)).
- **The dialect is translatable:** a trace flagged "illegible" in an Anthropic system card decoded cleanly as dense FreeCell shorthand — *"dense language, not alien language"*; even a small model could translate it ([faul_sname, LW June 2026](https://www.lesswrong.com/posts/wCSEpT3dTGz4N86Wi/even-illegible-mythos-reasoning-traces-seem-pretty-legible)).
- DeepSeek-R1 is the cleanest primary evidence of the economics: R1-Zero's raw RL CoT had *"poor readability, and language mixing"*; DeepSeek added a language-consistency reward and *accepted a measured performance degradation* for readability ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)).
- OpenAI's position: don't pressure the CoT at all — optimizing it for looking-nice produces **obfuscated** reward hacking; they coin the **"monitorability tax"** ([Baker et al., arXiv:2503.11926](https://arxiv.org/abs/2503.11926)); cross-lab position paper says scaled RL *"might no longer incentivise legibility"* and neuralese would lose CoT's safety advantages ([Korbak et al., arXiv:2507.11473](https://arxiv.org/abs/2507.11473)).

**Reading:** the demand for a dense reasoning code is empirically real — models grow one under RL whether we like it or not. Today's options are (a) let it drift (o3: opaque, spandrel-ridden), or (b) tax it back to English (R1: pay capability for legibility). **A designed language with a trained gloss channel is the unclaimed third option.**

Caution absorbed from the same literature: not every dense token means something — *vestigial* tokens get reinforced by credit assignment without carrying content ([Biddulph, LW](https://www.lesswrong.com/posts/6AxCwm334ab9kDsQ5/vestigial-reasoning-in-rl)); a grammar-constrained designed language excludes spandrels by construction.

## 2. Prior art map (what exists, with numbers)

### 2.1 Density without new vocabulary (cheap, proven, shallow)
- **Chain of Draft** — prompting only, ≤5-word steps: matched/beat CoT at **7.6% of tokens** ([arXiv:2502.18600](https://arxiv.org/abs/2502.18600)).
- **TALE** token budgets: ~68% reduction, <5% degradation ([arXiv:2412.18547](https://arxiv.org/abs/2412.18547)); **TokenSkip** trained pruning ([arXiv:2502.12067](https://arxiv.org/abs/2502.12067)); **C3oT** conditioned short-CoT ([arXiv:2412.11664](https://arxiv.org/abs/2412.11664)).

### 2.2 Latent/continuous reasoning (densest, but rejected as primary)
- **COCONUT** (last hidden state fed back; BFS-like superposed branches; [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)); **CCoT** contemplation tokens ([arXiv:2412.13171](https://arxiv.org/abs/2412.13171)); **CODI** self-distilled implicit CoT, 3.1× compression matching explicit CoT at GPT-2 scale ([arXiv:2502.21074](https://arxiv.org/abs/2502.21074)); **SoftCoT/++**, **Soft Thinking**, **Differentiable Cache Augmentation** (DeepMind, [arXiv:2412.17747](https://arxiv.org/abs/2412.17747)); surveys [arXiv:2507.06203](https://arxiv.org/abs/2507.06203), [arXiv:2505.16782](https://arxiv.org/html/2505.16782v1).
- Rejected as akh-medu's primary medium: latent thoughts are **unauditable** (filler-token paper warns of "unauditable, hidden computations," [arXiv:2404.15758](https://arxiv.org/abs/2404.15758)), **unportable across heterogeneous models**, and un-storable as KG facts. They remain interesting as a *later accelerator*.

### 2.3 Discrete custom tokens — the direct precedents
- **ToolkenGPT** — *the implementation template*: each tool = one new vocabulary token with a trainable embedding; base LLM **frozen**; only embedding rows learned ([arXiv:2305.11554](https://arxiv.org/abs/2305.11554)).
- **Token Assorted** — VQ-VAE **discrete latent codes mixed with text** in reasoning traces: +4–13% accuracy, ~17% shorter; random latent/text mixing during training "enables fast adaptation to new latent tokens" ([arXiv:2502.03275](https://arxiv.org/abs/2502.03275), ICML 2025).
- **Self-RAG** — epistemic *control tokens* (Retrieve/ISREL/ISSUP/ISUSE) as vocabulary entries ([arXiv:2310.11511](https://arxiv.org/abs/2310.11511)) — template for confidence/microtheory markers.
- **Planning tokens** — hand-designed discrete plan-symbols prefixing each reasoning step ([arXiv:2310.05707](https://arxiv.org/abs/2310.05707)).
- **DSI atomic docids** — whole documents as single T5 vocabulary tokens ([arXiv:2202.06991](https://arxiv.org/abs/2202.06991)); **DSI++** shows continual docid minting causes forgetting ([arXiv:2212.09744](https://arxiv.org/pdf/2212.09744)) — *the* caution against minting a token per KG entity.
- **LUKE** — entity vocabulary + entity-aware attention (encoder era, [EMNLP 2020](https://aclanthology.org/2020.emnlp-main.523/)); **RT-2** — actions as tokens ([arXiv:2307.15818](https://arxiv.org/abs/2307.15818)).
- **Context/objects as single tokens via projector:** **xRAG** — a document's *retrieval embedding* projected in as **one token** (175→1 tokens, +10% avg; [arXiv:2405.13792](https://arxiv.org/abs/2405.13792)); gist tokens 26× ([arXiv:2304.08467](https://arxiv.org/abs/2304.08467)); ICAE 4× ([arXiv:2307.06945](https://arxiv.org/abs/2307.06945)); Activation Beacon 4K→400K ([arXiv:2401.03462](https://arxiv.org/abs/2401.03462)).
- **Negative result to respect:** pause tokens help only when present in *pretraining*; bolt-on at finetune largely fails ([arXiv:2310.02226](https://arxiv.org/abs/2310.02226)). New tokens must be *semantically referenced* (ToolkenGPT works frozen because tool tokens are used meaningfully) or trained in.

> **The gap (agent's own conclusion):** "no paper was found where *human-authored* symbols for concepts/memories (e.g., KG node IDs) are minted as token IDs and trained as first-class vocabulary for *reasoning*… curated symbol inventory + embedding-row-only training + mixed text/symbol streams — each individually validated but not published as one system." **That is exactly the akh-medu proposal.**

### 2.4 Engineered compact languages (the cautionary shelf + the one real one)
- **ORION / "Mentalese"** ([arXiv:2511.22891](https://arxiv.org/pdf/2511.22891), Nov 2025, unreplicated preprint) — the first trained constructed reasoning language, citing the Language-of-Thought Hypothesis: "ultra-compressed, structured tokens," **4–16× fewer reasoning tokens, 90–98% of R1-Distill accuracy, 7–9× cheaper training** (Mentalese alignment + Shorter-Length Preference Optimization). Category precedent exists; the KG-grounded curated variant is still open.
- **SynthLang** — logographic prompt DSL (glyph classes `Σ ↹ ⊕` + CJK subjects + modifiers); claims 70% token cut / 233% speedup with **zero benchmark methodology, no tokenizer-level counts, no replication** ([repo](https://github.com/ruvnet/SynthLang), [spec gist](https://gist.github.com/ruvnet/8e9ade113348ecc84db24b0082554614)). The lesson is *how not to claim*.
- **Shogtongue** — GPT-4 self-compression folklore (Apr 2023, [@gfodor](https://x.com/gfodor/status/1643415357615640577)); "lossless" was a prompt wish, never measured; smaller models couldn't read it.
- **CAVEMAN** — deliberate grug-ification of agent *output* (~65% fewer output tokens, unbenchmarked; [writeup](https://www.rushis.com/caveman-does-talking-like-a-caveman-actually-make-ai-better/), [HN](https://news.ycombinator.com/item?id=47647455)); METR used a "Grug NEVER thinks" prompt as an experimental CoT register ([blog](https://metr.org/blog/2025-08-08-cot-may-be-highly-informative-despite-unfaithfulness/)).

### 2.5 Formal notations as reasoning media — the executability law
- Wins: **PAL** +15 abs over PaLM-540B CoT on GSM8K ([arXiv:2211.10435](https://arxiv.org/abs/2211.10435)); **Program-of-Thought** ~+12% ([arXiv:2211.12588](https://arxiv.org/abs/2211.12588)); **Logic-LM** +39.2%/+18.4% ([arXiv:2305.12295](https://arxiv.org/abs/2305.12295)); **LLM+P** (PDDL+planner; [arXiv:2304.11477](https://arxiv.org/abs/2304.11477)); **Grammar Prompting** (predict a minimal BNF, then generate under it; [arXiv:2305.19234](https://arxiv.org/abs/2305.19234)).
- Losses: **AMR-as-CoT hurts** ([NAACL 2024, arXiv:2405.01502](https://arxiv.org/abs/2405.01502)); **format constraints tax reasoning** ("Let Me Speak Freely?", [arXiv:2408.02442](https://arxiv.org/abs/2408.02442)); converting symbols *into* NL can help ([arXiv:2401.11725](https://arxiv.org/pdf/2401.11725)).
- **Law:** notation wins come from an **external executor** checking/running the notation, not from brevity or structure per se. AMR had no executor; PDDL/Python/FOL did. → Grug must be an *executable* notation whose executor is the akh-medu engine (assertions land in the KG; steps checked by e-graph/TMS/Dempster-Shafer).

### 2.6 Emergent-communication lessons
- Machine codes are **not naturally dense**: without an explicit length cost, agents develop *anti-efficient* codes (longest messages for the most frequent meanings; [Chaabouni et al., NeurIPS 2019](https://arxiv.org/abs/1905.12561)). Density must be paid for in the objective.
- Interpretability of a machine code is a **built artifact**: "Translating Neuralese" needed a trained translator, criterion = *induces the same belief in a listener* ([arXiv:1704.06960](https://arxiv.org/abs/1704.06960)) — the formal spec for our gloss channel's fidelity metric.

### 2.7 Density economics (why custom token IDs, exactly)
- Token density is **tokenizer-relative, not script-intrinsic**: same content up to ~15× token-cost across languages ([Petrov et al., arXiv:2305.15425](https://arxiv.org/pdf/2305.15425)); Chinese prompts are *not* more token-efficient in practice ([arXiv:2604.14210](https://arxiv.org/pdf/2604.14210)); human languages converge on uniform *information rate* (~39 bits/s), trading per-symbol density against speed ([Coupé et al., Science Advances 2019](https://www.science.org/doi/10.1126/sciadv.aaw2594)).
- → Borrowing exotic Unicode glyphs buys nothing (they BPE-split). **The only guaranteed 1-symbol=1-token is a minted token ID.** The user's instinct is the technically correct one.

### 2.8 Dense inter-model channels — and why discrete symbols win for akhs
- **CIPHER** (debate via embedding mixtures, [arXiv:2310.06272](https://arxiv.org/abs/2310.06272)), **DroidSpeak** (KV/E-cache reuse across fine-tunes of the *same base*, ~4× throughput; [arXiv:2411.02820](https://arxiv.org/abs/2411.02820)), **Cache-to-Cache** (trained KV fusion; ICLR 2026, [arXiv:2510.03215](https://arxiv.org/abs/2510.03215)).
- These are denser than any token language but **model-locked and unauditable**. akh-medu's akhs are deliberately heterogeneous (Tecton=GLM/ollama-cloud, Thoth=cloud reasoning, Gemma=Gemma-4/candle). **A shared discrete symbol language is the only dense channel that works across different backing models** — Grug doubles as the inter-akh wire format over ACP, and its utterances are storable as KG facts with provenance.

## 3. Synthesis with the two-lineage research: six design laws

1. **Vocabulary is cheap for models — the logographic path reopens.** Every lineage-A logography died on *human* learning cost (Wilkins' 4,000 radicals; the scribal class; Blissymbolics' ceiling). For a model, 500–2,000 extra embedding rows are trivial. The historical binding constraint on dense sign inventories does not bind a machine reasoner — while the *benefits* the Egyptian system demonstrated (dense text, explicit classification, dual-channel meaning) transfer intact.
2. **Density must be designed and paid for.** Emergent codes are anti-efficient (Chaabouni); RL-drifted codes are dense but spandrel-ridden and idiosyncratic (o3's "marinade"). Grug gets density by construction: closed vocabulary, grammar-constrained syntax, length-aware training (ORION's SLPO precedent).
3. **Executability is the win condition.** (§2.5.) Grug is not a compressed register; it is the surface syntax of the KG. Every Grug statement is assertable/checkable against the engine — the e-graph, TMS, event calculus, and Dempster-Shafer layers are the "interpreter" that PAL had in Python. This is what separates the proposal from SynthLang.
4. **Meaning by attached relation — the determinative, now literal.** The Egyptian classifier becomes a *classifier token*: an unpronounced-in-gloss type-tag following an entity symbol. Both verified levels transfer: **lexical classifiers** (static type) and **referent classifiers** (microtheory-relative type, pragmatics-sensitive) — the latter mapping onto akh-medu's context-bounded meaning. Graded (central/fringe) membership feeds Dempster-Shafer masses.
5. **Closed relation core + open concept lexicon** (AMR's surviving governance, mirrored by Egyptian's closed classifier set over an open word lexicon) — realized as a **two-tier token strategy**:
   - **Tier 1 (minted discrete tokens):** the governed closed core — NSM-style primes, the well-known relation predicates, classifier tokens, epistemic/control markers. Hundreds, not thousands. Standardized-against-logic + corpus-anchored (addendum §6), versioned (Leibniz's revisable postulates).
   - **Tier 2 (projected continuous tokens):** the open entity space — a symbol's **VSA hypervector projected through a trained bridge into one input-embedding slot**, xRAG-style. No vocabulary growth per entity (dodges DSI++ forgetting), and Phase 26's Neural→VSA bridge *is* the projector. Harnad satisfied: tier-2 symbols are grounded in the VSA/categorical layer by construction; tier-1 predicates carry mandated grounding anchors; composites get checked, not assumed.
6. **Dual-channel legibility — the Egyptian answer to the monitorability tax.** Don't tax the reasoning channel into English (R1's cost) and don't let it drift opaque (o3). Train **bilingual**: Grug ↔ reductive NSM-style paraphrase, both directions, from day one. Phase 33's grounded verbalizer is the decompiler; the Neuralese criterion (same induced belief) is its fidelity metric. Monitors read the gloss; the model thinks in Grug; spot-check translation faithfulness continuously.

## 4. Design sketch: High Grug v0

**Vocabulary budget (tier 1, minted):**
| Block | ~Count | Contents | Precedent |
|---|---|---|---|
| Semantic primes | 65–100 | NSM inventory + akh extensions | NSM (verified survey §2.1) |
| Relation predicates | 100–200 | the governed closed `agent:*`/KG relation set | AMR ~100 relations; CG/Common Logic |
| Classifier/type-tags | 50–100 | lexical + referent classifiers | Egyptian determinatives |
| Epistemic/control | 20–40 | confidence masses, retract/assert, microtheory push/pop, budget markers | Self-RAG reflection tokens; BudgetThinker |
| Structural | ~10 | place-structure delimiters (Lojban-style x1..xn), scope brackets | Lojban selbri/sumti |

**Syntax:** predicate-first, fixed place structure (Lojban's verified lesson: syntactic unambiguity is achievable; semantic ambiguity is deliberately handled by classifiers + microtheory context). Enforced by GBNF/constrained decoding (ADR 067's machinery). To dodge the measured format-tax ([arXiv:2408.02442](https://arxiv.org/abs/2408.02442)): free-form Grug *within* statements, hard grammar only at statement boundaries; and train (not merely constrain) the syntax in P1+.

**Token mechanics (from §2.3/§2.7 practice):**
- Embedding init for new rows: **mean-of-gloss-subtokens** (Hewitt's KL bound; EEVE recipe) or **VSA-projection** via the Phase-26 bridge — A/B these.
- Training: PEFT `trainable_token_indices` (embedding-rows-only, ToolkenGPT-style) → LoRA+`modules_to_save` (watch the tied-embedding untying pitfall) → staged continued pretraining (Chinese-LLaMA/EEVE) as commitment increases.
- GGUF/candle: mark tier-1 tokens **CONTROL**, tokenize agent-injected symbols with `parse_special=true`, keep HF-side and GGUF-side vocab in sync (the Unsloth Gemma-3 breakage is the cautionary tale). Candle backend gives us direct tokenizer control; ollama backend inherits the GGUF caveats.
- Streaming: Phase 44's context assembly emits tier-1 token IDs + tier-2 projected embeddings directly — "stream in handmade symbols" is a context-manager feature, not a model feature.

**Phased path (each gate = measurement, not vibes):**
- **P0 — no training (this month, credits-free):** Grug as a *constrained notation* over existing BPE pieces (multi-token symbols, grammar-enforced). Measure: accuracy-per-token vs English CoT and vs Chain-of-Draft baselines on akh tasks; translation-fidelity via the Phase-33 verbalizer. The bench_phase44 A/B harness + Phase 41's scaling benchmark are the instruments. *Kill criterion: if constrained-Grug at P0 costs more accuracy than CoD saves in tokens, stop.*
- **P1 — mint tier-1 tokens, embedding rows only** (ToolkenGPT recipe; Phase 27 Burn/LoRA + Phase 35 infra). Distill training data by translating existing reasoning traces into Grug (T5 Knowledge/NLU tier; Token Assorted's random-mixing trick for adaptation).
- **P2 — tier-2 projector:** train the VSA→embedding bridge (xRAG recipe; Phase 26). Entities enter as single slots; measure recall/grounding fidelity.
- **P3 — bilingual RL:** SLPO-style length-aware preference optimization (ORION precedent) + faithfulness reward on Grug↔gloss round-trip. This is where "reasoning language" becomes literal.
- **P4 — inter-akh Grug over ACP:** shared tier-1 vocabulary version-pinned across akhs; utterances logged as KG facts.

## 5. Risks (each with its source and mitigation)

| Risk | Evidence | Mitigation |
|---|---|---|
| Format constraints tax reasoning | [arXiv:2408.02442](https://arxiv.org/abs/2408.02442) | loose grammar inside statements; train the syntax (P1+), don't only constrain it (P0 measures the tax) |
| Bolt-on tokens do nothing | pause tokens ([arXiv:2310.02226](https://arxiv.org/abs/2310.02226)) | semantic referencing (ToolkenGPT), mixing trick (Token Assorted), staged pretraining (EEVE) |
| Continual minting → forgetting | DSI++ ([arXiv:2212.09744](https://arxiv.org/pdf/2212.09744)) | tier-2 projector for open entities; tier-1 stays small + versioned |
| Structured notation underperforms free text | AMRCoT ([arXiv:2405.01502](https://arxiv.org/abs/2405.01502)) | executability (law 3): Grug statements are engine-checked, AMR's weren't; P0 kill criterion |
| Monitorability loss | Baker/Korbak ([arXiv:2503.11926](https://arxiv.org/abs/2503.11926), [arXiv:2507.11473](https://arxiv.org/abs/2507.11473)) | dual-channel bilingual training; gloss-fidelity metric (Neuralese criterion); monitors read gloss |
| Density-vs-expressiveness wall | Kuhn ρ=−0.82 (verified, survey §2.1) | density from *symbols* (cheap for models), not from grammar minimalism; accept verbose gloss |
| Small-model capability tax | R1 readability-reward cost; CoD's GPT-3.5 math regression | per-akh opt-in (Gemma/candle first); benchmark before fleet rollout |
| Unmeasured hype | SynthLang | all claims measured against the *actual tokenizer*; publish the harness numbers |

## 6. Relation to the survey's paths

This is not a new substrate choice — it is **Path 2 (graph-primary) + Path 6 (grounded governance) realized at the token level, with Path 5 (controlled language) turned inward**: the controlled language is for the *model*, and humans get the gloss channel instead. The e-graph↔VSA coupling question (survey open question #2) gets a concrete candidate answer: they couple **in the token stream** — tier-1 symbols are e-graph-addressable, tier-2 slots are VSA-native, and the model is the transducer between them.

## 7. Honesty & verification status

- Unlike the prior three docs, these claims were **not** adversarially 3-vote verified (Firecrawl exhausted; single-agent sourcing). Confidence is correspondingly lower; re-verify load-bearing numbers before an ADR.
- Agent-flagged uncertainties: CCoT's exact numbers; TokenSkip's headline figure; Token Assorted codebook internals; ORION's Mentalese format details (readable? token inventory?) — unreplicated Nov 2025 preprint; SynthLang/Shogtongue/CAVEMAN numbers are self-reported or folklore; LLMLingua's "$42K→$2.1K" is vendor marketing; the o1 system-card quote and the 36% vs 53% illegibility-cost discrepancy need PDF-level checks.
- "Grug speak" as a name for o3's dialect is community shorthand; METR used it as an *induced* register.

## 8. Link list (all sources, grouped)

### Phenomenon & safety
- https://grugbrain.dev/ — origin of the grug register (Carson Gross, 2022)
- https://x.com/karpathy/status/1835561952258723930 — Karpathy: RL → models stop speaking English in CoT
- https://openai.com/index/learning-to-reason-with-llms/ — o1: CoT must remain unaltered; raw CoT hidden
- https://arxiv.org/abs/2412.16720 · https://cdn.openai.com/o1-system-card-20241205.pdf — o1 system card; CoT legibility as safety asset
- https://arxiv.org/abs/2503.11926 · https://openai.com/index/chain-of-thought-monitoring/ — Baker et al.: obfuscated reward hacking; "monitorability tax"
- https://arxiv.org/abs/2501.12948 — DeepSeek-R1: language mixing; readability reward costs performance
- https://arxiv.org/abs/2505.05410 · https://www.anthropic.com/research/reasoning-models-dont-say-think — CoT unfaithfulness (Anthropic)
- https://arxiv.org/abs/2507.11473 — cross-lab CoT monitorability position paper
- https://arxiv.org/abs/2509.15541 · https://www.apolloresearch.ai/research/stress-testing-deliberative-alignment-for-anti-scheming-training/ · https://openai.com/index/detecting-and-reducing-scheming-in-ai-models/ · https://www.antischeming.ai/snippets — o3 dialect transcripts
- https://transluce.org/investigating-o3-truthfulness — Transluce o3 investigation
- https://www.interconnects.ai/p/openais-o3-over-optimization-is-back — RLVR over-optimization ("worse at language")
- https://1a3orn.com/sub/2025-10-weird-cot.html · https://www.lesswrong.com/posts/qgvSMwRrdqoDMJJnD/towards-a-typology-of-strange-llm-chains-of-thought — typology; "marinade" 8,344×
- https://arxiv.org/abs/2510.27338 · https://www.alignmentforum.org/posts/GKyyYCs8n2goDcAe2/reasoning-models-sometimes-output-illegible-chains-of — illegible CoT is load-bearing
- https://www.lesswrong.com/posts/wCSEpT3dTGz4N86Wi/even-illegible-mythos-reasoning-traces-seem-pretty-legible — "dense language, not alien language"
- https://arxiv.org/abs/2601.06047 — academic treatment of the o3 dialect
- https://metr.org/blog/2025-08-08-cot-may-be-highly-informative-despite-unfaithfulness/ — METR's induced grug-speak probe
- https://www.lesswrong.com/posts/6AxCwm334ab9kDsQ5/vestigial-reasoning-in-rl — spandrel mechanism
- https://ai-2027.com/ · https://www.lesswrong.com/posts/zkccztuSjLshffrNr/13-arguments-about-a-transition-to-neuralese-ais — neuralese bandwidth math & transition debate

### Latent / continuous reasoning
- https://arxiv.org/abs/2412.06769 · https://github.com/facebookresearch/coconut — COCONUT
- https://arxiv.org/abs/2412.13171 — CCoT contemplation tokens
- https://arxiv.org/abs/2502.21074 · https://aclanthology.org/2025.emnlp-main.36/ · https://github.com/zhenyi4/codi — CODI
- https://arxiv.org/abs/2502.12134 · https://aclanthology.org/2025.acl-long.1137/ · https://github.com/xuyige/SoftCoT · https://arxiv.org/pdf/2505.11484 — SoftCoT/++
- https://arxiv.org/abs/2505.15778 — Soft Thinking
- https://arxiv.org/abs/2412.17747 — Differentiable Cache Augmentation
- https://arxiv.org/abs/2502.08524 — CoCoMix (SAE concepts in pretraining)
- https://arxiv.org/abs/2412.08821 — Large Concept Models (SONAR sentence space)
- https://arxiv.org/abs/2507.06203 · https://arxiv.org/html/2505.16782v1 — latent-reasoning surveys
- https://arxiv.org/pdf/2510.07745 · https://arxiv.org/html/2509.19170 · https://arxiv.org/pdf/2511.06411 · https://arxiv.org/html/2508.03440 · https://arxiv.org/abs/2509.21482 · https://arxiv.org/pdf/2511.08983 · https://arxiv.org/pdf/2511.07124 — 2025–26 follow-ups

### Discrete custom tokens
- https://arxiv.org/abs/2305.11554 — ToolkenGPT (the implementation template)
- https://arxiv.org/abs/2502.03275 · https://proceedings.mlr.press/v267/su25g.html — Token Assorted
- https://arxiv.org/abs/2310.11511 · https://selfrag.github.io/ — Self-RAG reflection tokens
- https://arxiv.org/abs/2310.05707 · https://github.com/WANGXinyiLinda/planning_tokens — planning tokens
- https://arxiv.org/abs/2202.06991 · https://arxiv.org/pdf/2212.09744 — DSI atomic docids; DSI++ forgetting
- https://aclanthology.org/2020.emnlp-main.523/ · https://arxiv.org/abs/2110.08151 — LUKE / mLUKE entity vocabularies
- https://arxiv.org/abs/2307.15818 · https://proceedings.mlr.press/v229/zitkovich23a.html — RT-2 action tokens
- https://arxiv.org/abs/2302.04761 · https://ar5iv.labs.arxiv.org/html/2302.04761 — Toolformer (avoided vocab surgery)
- https://arxiv.org/abs/2405.13792 — xRAG (document = 1 token via projector)
- https://arxiv.org/abs/2304.08467 — gist tokens (26×)
- https://arxiv.org/abs/2307.06945 — ICAE memory slots
- https://arxiv.org/abs/2401.03462 — Activation Beacon
- https://arxiv.org/abs/2305.16300 · https://github.com/epfml/landmark-attention — Landmark attention
- https://arxiv.org/abs/2207.06881 — Recurrent Memory Transformer
- https://arxiv.org/abs/2305.14788 — AutoCompressors
- https://arxiv.org/abs/2310.02226 — pause tokens (pretraining-only gains)
- https://arxiv.org/abs/2404.15758 — filler tokens / hidden computation
- https://arxiv.org/abs/2403.09629 — Quiet-STaR thought delimiters
- https://arxiv.org/abs/2402.05862 — GraphToken (KG → soft tokens)
- https://arxiv.org/pdf/2508.17196 — BudgetThinker control tokens

### Concise CoT & compression
- https://arxiv.org/abs/2502.18600 · https://github.com/sileix/chain-of-draft — Chain of Draft
- https://arxiv.org/abs/2412.18547 · https://aclanthology.org/2025.findings-acl.1274.pdf — TALE
- https://arxiv.org/abs/2502.12067 · https://aclanthology.org/2025.emnlp-main.165/ — TokenSkip
- https://arxiv.org/abs/2412.11664 — C3oT
- https://arxiv.org/html/2503.01141v1 — token-complexity analysis
- https://arxiv.org/pdf/2504.10903 · https://arxiv.org/html/2507.02076v1 · https://arxiv.org/pdf/2503.21614 · https://arxiv.org/pdf/2507.09662 · https://arxiv.org/html/2506.05256v1 · https://arxiv.org/pdf/2511.13223 — efficient-reasoning surveys & length-penalty RL
- https://arxiv.org/html/2310.05736v2 · https://github.com/microsoft/LLMLingua · https://www.llmlingua.com/ · https://arxiv.org/abs/2310.06839 · https://arxiv.org/pdf/2403.12968 · https://arxiv.org/html/2410.12388v2 — LLMLingua family & survey
- https://arxiv.org/pdf/2502.14925 — CODEPROMPTZIP

### Engineered languages & notations
- https://github.com/ruvnet/SynthLang · https://gist.github.com/ruvnet/8e9ade113348ecc84db24b0082554614 · https://synthlang.fly.dev/ — SynthLang (self-reported)
- https://www.tomsguide.com/news/chatgpt-made-its-own-shogtongue-language-to-blow-past-the-word-limit · https://www.piratewires.com/p/compression-prompts-gpt-hidden-dialects · https://x.com/gfodor/status/1643415357615640577 — Shogtongue folklore
- https://www.rushis.com/caveman-does-talking-like-a-caveman-actually-make-ai-better/ · https://news.ycombinator.com/item?id=47647455 — CAVEMAN
- https://arxiv.org/pdf/2511.22891 — ORION / Mentalese (trained constructed reasoning language)
- https://arxiv.org/abs/2211.10435 — PAL · https://arxiv.org/abs/2211.12588 — Program of Thoughts
- https://arxiv.org/abs/2304.11477 · https://arxiv.org/pdf/2305.15771 — LLM+P & critique
- https://arxiv.org/abs/2305.12295 · https://aclanthology.org/2023.findings-emnlp.248/ · https://arxiv.org/html/2502.15652v2 · https://arxiv.org/abs/2405.18357 — Logic-LM, survey, SymbCoT
- https://arxiv.org/abs/2305.19234 — Grammar Prompting
- https://arxiv.org/abs/2408.02442 · https://aclanthology.org/2024.emnlp-industry.91/ — format-restriction tax
- https://arxiv.org/abs/2405.01502 · https://aclanthology.org/2024.naacl-long.209/ · https://arxiv.org/html/2504.04745 — AMR-as-CoT negative results
- https://arxiv.org/pdf/2412.10654 · https://arxiv.org/pdf/2402.11199 · https://arxiv.org/pdf/2505.24377 — KG-as-notation studies
- https://arxiv.org/pdf/2401.11725 — Symbol-to-Language
- https://arxiv.org/pdf/2512.17967 — Memelang
- https://arxiv.org/pdf/2311.01751 · https://arxiv.org/html/2402.05868v3 — EmojiLM / EmojiPrompt
- https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output · https://tianpan.co/blog/2026-04-16-grammar-constrained-generation-output-reliability · https://www.jamesdrandall.com/posts/gbnf-constrained-generation/ — constrained-decoding practice

### Emergent communication & inter-model channels
- https://arxiv.org/abs/1905.12561 · https://papers.nips.cc/paper_files/paper/2019/hash/31ca0ca71184bbdb3de7b20a51e88e90-Abstract.html — anti-efficient encoding
- https://arxiv.org/abs/1704.06960 · https://aclanthology.org/P17-1022/ · https://github.com/jacobandreas/neuralese — Translating Neuralese
- https://arxiv.org/abs/2006.02419 — Lazaridou & Baroni survey · arXiv 1612.07182 / 1804.03984 — referential-game emergence
- https://arxiv.org/abs/2310.06272 — CIPHER · https://arxiv.org/abs/2411.02820 — DroidSpeak · https://arxiv.org/abs/2510.03215 · https://github.com/thu-nics/C2C — Cache-to-Cache · https://arxiv.org/pdf/2510.03346 — KVComm

### Density economics & tokenizers
- https://arxiv.org/pdf/2305.15425 · https://aleksandarpetrov.github.io/tokenization-fairness/ — tokenizer unfairness (~15×)
- https://arxiv.org/html/2510.21909v1 · https://arxiv.org/pdf/2605.24718 — tokenizer-inequity follow-ups
- https://arxiv.org/pdf/2604.14210 — Chinese not more token-efficient (2026) · https://eu.36kr.com/en/p/3579563285969799 — R1 "thinks in Chinese" anecdote
- https://www.science.org/doi/10.1126/sciadv.aaw2594 · https://www.cnrs.fr/en/press/similar-information-rates-across-languages-despite-divergent-speech-rates · https://gwern.net/doc/cs/algorithm/information/2011-pellegrino.pdf — uniform information rate
- https://arxiv.org/abs/2506.18341 — multilingual unification learning
- https://arxiv.org/pdf/2603.23527 — prompt-compression replication reversal · https://news.ycombinator.com/item?id=48588755 — genre skepticism

### Vocab-surgery mechanics
- https://huggingface.co/docs/transformers/en/main_classes/tokenizer · https://huggingface.co/docs/transformers/en/internal/tokenization_utils — HF add_tokens/resize
- https://www.cs.columbia.edu/~johnhew//vocab-expansion.html — mean-init KL bound
- https://arxiv.org/html/2407.05841 · https://arxiv.org/html/2402.14714v1 — init comparison; EEVE staged training
- https://arxiv.org/abs/2112.06598 — WECHSEL · https://arxiv.org/abs/2305.14481 · https://github.com/konstantinjdobler/focus — FOCUS · https://arxiv.org/html/2504.21018v2 — HyperOfa · https://arxiv.org/pdf/2505.20133 — Token Distillation
- https://arxiv.org/abs/2304.08177 — Chinese-LLaMA vocab merge
- https://huggingface.co/docs/peft/main/en/developer_guides/lora · https://github.com/huggingface/peft/issues/1750 · https://discuss.huggingface.co/t/addition-of-lm-head-and-embed-tokens-layers-to-the-lora-adapter/141080 · https://kaitchup.substack.com/p/lora-trainable-tokens-save-memory · https://langcopilot.com/posts/2025-09-23-how-to-add-special-tokens-llms — PEFT practice
- https://deepwiki.com/ggml-org/llama.cpp/3.3-tokenization-and-vocabulary · https://github.com/ggml-org/llama.cpp/discussions/9379 · https://github.com/unslothai/unsloth/issues/5070 · https://github.com/ggml-org/llama.cpp/issues/4868 — GGUF token_type / parse_special / breakage

### Conlangs as LLM probes (adjacent)
- https://arxiv.org/abs/2508.06094 — ConlangCrafter · https://arxiv.org/html/2509.00425 — Camlang · https://arxiv.org/pdf/2510.07591 — conlang grammar probes
