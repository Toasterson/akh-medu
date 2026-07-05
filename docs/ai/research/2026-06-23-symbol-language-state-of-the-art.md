# Symbol-Language State of the Art — Two-Lineage Survey — akh-medu

- **Date:** 2026-06-23
- **Status:** Research draft (no implementation). Findings split across three append-only docs (this survey + a coverage-gap addendum + an Egyptian companion) — each ingested to Seshat separately so all are vector-findable while preserving history.
- **Scope:** Design research for the akh-medu symbol model — symbols as entities + well-known relation predicates in a knowledge graph, with a VSA layer (binary hypervectors, Hamming, HNSW), an e-graph reasoning layer, Cyc-style microtheories, and spreading-activation / argumentation / Dempster-Shafer reasoning.
- **Method:** Three Firecrawl-backed `deep-research` workflow passes (fan-out search → PDF-capable scrape → 3-vote adversarial verification → synthesis). ~210 agents, ~9.3M tokens across the first two passes; coverage-gap + Egyptian passes append.
- **Author:** Claude (research agent), commissioned by Till.
- **Related:** [ADR 037](../decisions/037-llm-vsa-deep-integration.md) (LLM-VSA), [ADR 052](../decisions/052-unified-thinking-loop.md), [research: thinking in LLMs vs VSA](2026-04-27-thinking-in-llms-vs-vsa.md), [ADR 072](../decisions/072-normalized-akhomed-store.md).

---

## 0. Origin question

> "When I originally thought about [the symbols] I thought about a dedicated compute-optimized language akin to Egyptian where rules and symbols make up meaning. Here rules and connections make up meaning. What's the state of the art research here?"

The question straddles two literatures, surveyed here as halves of one question:
- **Lineage A** — the writing-system / conceptual-language / semantic-primitive tradition (Egyptian, Leibniz, Wilkins, Blissymbolics, NSM, Conceptual Dependency, Lojban, Conceptual Graphs, UNL/AMR).
- **Lineage B** — the AI knowledge-representation tradition (VSA/HDC, knowledge graphs, Cyc/microtheories, e-graphs, neuro-symbolic, conceptual spaces).

"Compute-optimized" is evaluated on four axes throughout: **(1) reasoning/inference efficiency, (2) hardware/vector efficiency, (3) compositional expressiveness, (4) authorability.**

> **Provenance convention:** §1–§2 are machine-verified (adversarial 3-vote, cited, with confidence). §3–§6 are cross-lineage synthesis by the author, flagged as such.

---

## 1. The single law both lineages obey

Researched separately, the linguistic lineage and the AI lineage reach the same conclusion from opposite directions:

> **Meaning that is *positionally enumerated* dies; meaning that is *relational/compositional over a small primitive set* survives — but composition alone never confers grounding, and no complete primitive set exists.**

- **Lineage A** proves it historically: top-down taxonomic encodings (Wilkins, Dalgarno) failed completely; combinatorial/relational designs (Leibniz's idea, NSM, Lojban) survived in niches.
- **Lineage B** proves it computationally: the same tension reappears as the four-axis tradeoff, and VSA is the modern *engineered* attempt to hold structure + vector in one substrate — bounded by capacity, not by principle.

The original "rules and connections make up meaning" intuition is the **historically correct bet.** What both lineages add — and what akh-medu's design currently lacks — is the *governance of the primitive set* and its *grounding*.

---

## 2. Verified findings

### 2.1 Lineage A — conceptual-language history (Firecrawl pass, 21/25 verified)

**Two strategies, opposite fates.** The 17th-c. philosophical languages split: Dalgarno's *Ars Signorum* (1661) — ~1,000 radicals, analytic small-set — vs. Wilkins' *Essay* (1668) — 4,000+ radicals, encyclopedic — both built on Aristotelian categories (top-down taxonomy). Both failed completely. *[3-0, high; Maat, ILLC dissertation]*

**Leibniz broke toward composition** — prioritized *combinatorial over classificatory* principles, with an arithmetic encoding: primes = primitive concepts, products = composites (*man* = rational × animal), analysis = factorization — anticipating Gödel numbering. *[3-0 / 2-1, high; Maat; Jinaru]*

**The deepest finding — Leibniz diagnosed his own program as unsolvable.** "Primitive" terms are *not* truly atomic but "postulates adopted based on considerations of calculation convenience"; one can never be certain decomposition reached genuine atoms, so the inventory must keep expanding as knowledge advances. *The inventor of the symbolic-AI dream already saw why it could not be completed.* *[3-0, high; Jinaru citing Eco, *Search for the Perfect Language* pp. 222–223]*

**Harnad — the unifying critique and the constructive answer.** Symbol grounding problem (1990): in a closed symbol system meaning is parasitic on an external interpreter — the **dictionary-go-round** that "never comes to a halt on what anything meant." Prescription: symbols must be grounded **bottom-up in iconic** (analog sensory projections) **and categorical** (learned/innate invariant feature-detectors) representations; elementary symbols are *names of categories* assigned via their categorical representations. **A primitive set cannot be chosen by top-down decomposition — it must bottom out in sensory category invariants.** *[3-0, high; Harnad, *Physica D* 42:335–346, 1990]*

**⚠️ Refuted (0-3), and load-bearing:** the claim that "once elementary names are grounded, composites inherit grounding by composition alone" was killed unanimously. Harnad does **not** endorse grounding propagating through composition. Relational-over-positional is right; *"binding grounded atoms yields a grounded composite for free"* is wrong.

**NSM — the disciplined survivor.** ~65 semantic primes (stabilized 2014, declared "complete"), explicitly tracing to "Leibniz's alphabet of human thoughts," defended against circularity by **reductive paraphrase** into indefinables with no technical terms. *[3-0, high]* But completeness is contested: Evans argues only a "solid core" is universal, with an "outer set [where] different languages compose things differently" — no complete shared alphabet. *[2-1; Wierzbicka 2021; Goddard]*

**The primitive-set problem, empirically.** Intuitively-basic words fail primality: NSM excludes *eat* (→ "doing"/"mouth"), *go*, *hot*, *bird*. Decomposition suffers **regress** (what does CAUSE mean if not "cause"?) and **non-uniqueness** ("word meaning systematically cannot be given a unique, precise definition"); the definitional theory of concepts "has few adherents today." *[3-0 / 2-1, high; Acquaviva & Panagiotidis; Fodor; Laurence & Margolis]*

**Lojban — disambiguation is syntactic, not semantic.** Machine-parseable grammar over predicate logic (each *selbri* a relation with fixed argument slots), but semantic ambiguity, metaphor, and ellipsis are *deliberately* permitted. Engineered languages can kill syntactic but not semantic ambiguity. *[3-0 / 2-1; Lojban CLL]*

**The tradeoff, quantified.** Kuhn's survey of 100 controlled natural languages: expressiveness vs. simplicity correlate at **ρ = −0.82**, precision vs. expressiveness at −0.66; no language reaches the "perfect" P5E5N5S5 class. *[3-0, high; Kuhn, *Computational Linguistics* 40(1), 2014]*

### 2.2 Lineage B — AI knowledge representation (first pass, 24/25 verified)

| Verified finding | Confidence |
|---|---|
| VSA = family (TPR/HRR/MAP/BSC) unifying structure + vector via binding/bundling/permutation; explicitly designed as this bridge | high |
| Hardware is its strength: N≫D quasi-orthogonality; similarity = Hamming (XOR + popcount) → SIMD / HNSW | high |
| Composition *requires* non-associative binding (superposition catastrophe); bundling capacity is finite (signal-detection bound) | high |
| Decoding = resonator networks ("search in superposition"), but **quadratic capacity ceiling, no convergence guarantee** | high |
| e-graphs (egg) & OWL2 profiles: reasoning-power-vs-tractability is a *deliberately engineered* axis | high |
| Hand-authored KR (Cyc) stalled on the acquisition bottleneck; 2025 consensus → neuro-symbolic fusion | medium |

---

## 3. The merged four-axis scorecard *(synthesis)*

H/M/L. **Hardware/vector is L for the entire linguistic lineage by nature** — that is the whole reason lineage B exists.

| Approach | 1. Reasoning | 2. HW/vector | 3. Compositional | 4. Authorability |
|---|:--:|:--:|:--:|:--:|
| **— Lineage A —** | | | | |
| Wilkins/Dalgarno (taxonomic) | M | L | M | **L** ← *died* |
| Leibniz characteristica | H *(aspirational)* | L | **H** | L |
| Blissymbolics | L | L | H | M *(AAC niche)* |
| NSM semantic primes | M | L | **H** | M–H |
| Schank Conceptual Dependency | M–H | L | H | M |
| Lojban | M | L | H | M *(steep)* |
| Conceptual Graphs (Sowa) | M *(FOL=slow)* | L | H | **H** |
| UNL / AMR interlingua | M | L | H | M |
| **— Lineage B —** | | | | |
| **VSA/HDC (binary, Hamming)** | M | **H** | H *(lossy)* | **L** |
| Knowledge graphs / RDF-OWL | M | M | M–H | **H** |
| e-graph / equality saturation | **H** | **L** | H | M |
| Cyc / microtheories | M | L | H | **L** |
| Conceptual Spaces (Gärdenfors) | M | **M–H** | M | M |
| Neuro-symbolic (learned bridge) | M | M–H | M–H | M |

No row is all-H. The survivors cluster where columns 3–4 are strong (NSM, Sowa, AMR); the thing that died (Wilkins) is the one with L authorability and positional composition.

---

## 4. The tradeoff geometry (both-lineage-confirmed) *(synthesis)*

Four forces pull against each other; satisfy two or three, never four in one substrate:
- **HW-fit (2) ⟂ decomposability + authorability (3,4)** — VSA's verified capacity ceiling *is* this tax. Lineage A pays the mirror tax: the most authorable systems (Sowa, NSM) have no hardware story.
- **Reasoning (1) ⟂ HW-fit (2)** — e-graphs want pointer-graphs; vectors want arithmetic. You cannot saturate an e-graph inside the vector algebra.
- **Expressiveness (3) ⟂ tractability (1)** — OWL2 profiles in B; **Kuhn's ρ = −0.82** in A. Same wall, two literatures, one now with a hard number.
- **Authorability (4) ⟂ everything machine-optimized** — and the historical killer in *both* lineages is the authoring/acquisition bottleneck.

---

## 5. Implications for akh-medu's predicate set *(synthesis)*

akh-medu's predicates accrete bottom-up via `resolve_or_create_relation("agent:…")` with no principled set. The verified findings say that is **not** automatically the Wilkins failure. Three prescriptions emerged — **complementary, not competing**:

1. **Leibniz (governance):** treat primitives as *revisable calculational postulates*, never a "complete" inventory. akh-medu's accretion is closer to Leibniz than to Wilkins — the fix is not "stop accreting" but "**accrete with discipline and version it.**"
2. **Harnad (grounding):** primitives must bottom out in **sensory/categorical** representations, not symbol-on-symbol definitions. This is exactly what Phase-24 sensory grounding + the VSA distributional layer are for — reframed from "nice features" to **the load-bearing answer** to why pure-symbolic KR died. ⚠️ The refuted claim warns: do **not** assume a bound/composed predicate is grounded just because its parts are.
3. **NSM (seed + method):** the most rigorous existing candidate principled core, but its completeness is contested — adopt the **reductive-paraphrase methodology** (decompose to indefinables, no technical terms) more than the exact 65 atoms.

**Synthesis:** don't chase a complete primitive set (Wilkins and Leibniz both proved it impossible). Keep predicates *revisable but disciplined/versioned* (Leibniz), tie them to sensory/VSA categorical grounding with composites checked separately (Harnad), and borrow NSM's reductive-paraphrase discipline as the authoring guardrail. **The hole is not "no fixed alphabet" — it is "no governance + no enforced grounding tie."**

---

## 6. Candidate paths *(synthesis)*

1. **VSA-primary** ("everything is a hypervector"). HW **H** · Reasoning M · Compositional H-lossy · Authorability **L**. Most exposed to the verified capacity ceiling.
2. **Graph-primary, vectors as index** (≈ current akh-medu). Authorability **H** · Reasoning **H** · Compositional H · HW M. Robust spine; forfeits "computing in superposition."
3. **Conceptual-spaces middle layer** (Gärdenfors). HW M–H · Reasoning M · Compositional M · Authorability M. Least-explored, highest-upside bridge; fits HNSW metric index; graded membership feeds Dempster-Shafer.
4. **Neuro-symbolic / learned bridge.** HW **H** · Authorability M (opaque). 2025-consensus direction; mismatch with pure non-backprop VSA.
5. **Controlled-language authoring front-end** (NSM/ACE-style). Authorability **H**; composes with 1–4.
6. **Grounded-predicate governance** *(new, evidence-backed)*. Versioned, paraphrase-disciplined predicate registry (Leibniz + NSM); every predicate carries a sensory/VSA categorical anchor (Harnad); composites grounding-checked, not assumed (the refuted claim). Authorability **H**; direct fix for the "accrete with no principle" gap. Composes with 1–5.

**Where the verified history leans:** Path 2 (graph-primary spine) + Path 6 (grounded governance) — the relational-compositional survivor pattern, with the grounding the pure-symbolic failures lacked. Path 1 stays exciting-but-capacity-threatened.

---

## 7. Coverage-gap pass (Schank CD, Sowa Conceptual Graphs, UNL/AMR)

Covered in the append-only addendum **`2026-06-23-symbol-language-coverage-gap-addendum.md`** (Schank's Conceptual Dependency ACTs + Scripts, Sowa's Conceptual Graphs, and the UNL/AMR interlinguas — the relational AI/NLP systems that earlier passes fetched but did not verify). Kept as a separate doc so it is independently vector-findable in Seshat without re-ingesting this survey.

---

## 8. Companion: why the Egyptians built their writing the way they did

Covered in the companion doc **`2026-06-23-egyptian-writing-design-rationale.md`** — the historical / cognitive / administrative / religious rationale for the mixed logo-phonetic + determinative structure of Egyptian writing, read as a designed information system. The determinative-as-semantic-classifier (an unpronounced type-tag attached to a token) is the live analogy for akh-medu's relation-predicate-on-entity design; *medu* ("words / speech") is itself the source of the project's name.

---

## 9. Open questions

- Does VSA's finite superposition capacity + the resonator network's quadratic, non-converging factorization ceiling survive akh-medu's intended KG scale? (Decides Path 1 vs Path 2.)
- How should the e-graph reasoning layer and the VSA layer couple — reason over *decoded* symbols (repeated unbinding cost) or *in* the vector algebra (parallel, capacity-bounded)?
- Of the three predicate-governance prescriptions (NSM adopt / Leibniz revisable / Harnad grounding), which to operationalize first — and how is "predicate grounded in a sensory/VSA category" actually enforced and measured?
- Harnad refutation: by what mechanism (if any) does grounding propagate to composite predicates, and under what conditions does it fail?

---

## 10. Key sources

**Lineage A:** Maat, *Philosophical Languages in the 17th Century* (ILLC); Harnad, "The Symbol Grounding Problem," *Physica D* 42 (1990); Wierzbicka, "Semantic Primitives, fifty years later," RUDN (2021); Goddard (NSM, Griffith); Acquaviva & Panagiotidis, *Lingue e Linguaggio* XI.2 (2012); Kuhn, "A Survey and Classification of Controlled Natural Languages," *Computational Linguistics* 40(1) (2014); Lojban CLL; Jinaru on Leibniz; Eco, *The Search for the Perfect Language*.

**Lineage B:** Kleyko et al., "A Survey on HDC/VSA," *ACM Computing Surveys* 55(6) (2022) + "VSA as a Computing Framework," *Proc. IEEE* (2022); Frady/Kent/Olshausen/Sommer, "Resonator Networks 1–2," *Neural Computation* (2020); Willsey et al., "egg," POPL 2021; W3C OWL 2 Profiles; "A Community-driven vision for a new Knowledge Resource for AI," *AI Magazine* (2025).
