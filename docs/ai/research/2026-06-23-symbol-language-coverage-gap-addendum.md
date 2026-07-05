# Symbol-Language Survey — Coverage-Gap Addendum: Schank CD, Sowa Conceptual Graphs, UNL/AMR — akh-medu

- **Date:** 2026-06-23
- **Status:** Research draft (no implementation). Append-only addendum to [the two-lineage survey](2026-06-23-symbol-language-state-of-the-art.md) §7 — closes the lineage-A coverage gap left when these AI/NLP relational systems were fetched but not verified in earlier passes.
- **Scope:** Three relational meaning-representation systems where meaning is built from primitives + typed relations, scored on the same four axes (reasoning / hardware-vector / compositional / authorability).
- **Method:** Firecrawl-backed `deep-research` pass (`wv3c0nztl`), 103 agents, 21/25 claims verified (4 refuted, shown), nearly all primary sources.
- **Author:** Claude (research agent), commissioned by Till.

---

## 1. The one-line result

All three families **confirm the survival hypothesis at the level of design intent** — they are textbook relational/compositional designs (small primitive/relation set + typed connections → canonical decomposable meaning) — **while exposing exactly where relationality alone is insufficient.**

> **Relationality is necessary but not sufficient.** It must be paired with a *principled* relation/primitive inventory, or the vocabulary accretes ad hoc — exactly the failure mode akh-medu's bottom-up `resolve_or_create_relation` currently risks. The two disciplines that actually *survived* are **standardization against a logic** (Conceptual Graphs → ISO/IEC 24707 Common Logic) and **corpus-anchoring** (AMR → PropBank).

---

## 2. Schank's Conceptual Dependency + Scripts — the cautionary twin of akh-medu

- **The design:** synonymous sentences (any language) map to **one canonical representation**, decomposed into ~11 primitive ACTs (ATRANS, PTRANS, PROPEL, MOVE, GRASP, INGEST, EXPEL, MTRANS, MBUILD, SPEAK, ATTEND) + typed conceptual roles. The *smallness* of the set is treated as what makes world-knowledge inference tractable. *[3-0, high; Schank & Abelson 1977]*
- **The primitive-CHOICE problem was live from day one, and never principled.** S&A 1977 verbatim: *"There has been much debate over whether the conceptual primitives of CD theory are the right primitives, and some criticism that the theory is ad hoc… the important criterion is whether the theory is useful… We will not be dogmatic… revisions in our scheme will no doubt be necessary."* The inventory **drifted 11 → 12 → 14 ACTs** across Schank's own papers. *[3-0, high]* **This is the precise risk akh-medu's accreting predicate set runs.**
- **Script brittleness is structural:** scripts (the restaurant script) fire *only* when a known stereotyped situation is recognized — unanticipated situations get no inference. *[3-0, high]*

**Scores:** Reasoning M–H (canonical form aids inference) · HW/vector L · Compositional H (small ACT set + roles) · Authorability M (hand-authored scripts = acquisition bottleneck). **Lesson: relational + canonical, but the primitive set was defended on "usefulness," not correctness — and that is why it wobbled.**

## 3. Sowa's Conceptual Graphs — the strongest survival case

- **The design:** bipartite graph, exactly two node kinds (rectangular *concept* nodes, circular *conceptual-relation* nodes), from Peirce's existential graphs + 1960s semantic networks; first deployed (1976) as an NL→relational-DB intermediate language. *[3-0, high; Sowa Handbook]*
- **Standardized, reversible FOL semantics — the key differentiator.** CGIF is one of three **normative dialects of ISO/IEC 24707 Common Logic** (with CLIF and XCL), all sharing one model-theoretic FOL semantics; a CG and a CLIF sentence are logically equivalent and inter-translatable. *[3-0, high; Sowa eg2cg; Delugach CEUR Vol-352]* **This is a principled, logic-grounded relation vocabulary — the opposite of ad-hoc accretion.**
- **Graph inference can be efficient + is sound/complete for FOL:** Peirce's existential-graph rules prove Leibniz's *Praeclarum Theorema* in **7 steps from a blank sheet** vs 43 steps / 5 axioms in *Principia*. *[3-0, high; treat as illustrative, not a general benchmark]*
- **Expressiveness ⟂ decidability, named explicitly:** CL/CGIF is full FOL (> RDF, > OWL DL); the comparably-expressive OWL Full is undecidable. CGs span a *"continuous range of precision"* from FOL-precise to deliberately loose for NL vagueness. *[3-0, high; Delugach]*
- **⚠️ The acquisition bottleneck is STILL OPEN — relationality + FOL did not dissolve it.** Perlis (1982): *no formal algorithm can translate informal language to formal representation* because background knowledge is unbounded. Sowa concedes context specification is *"endless"* even for humans; writing complex CLCE needs logic training (reading doesn't). *[3-0, high]* **This is the central complication: CD, CG, and the interlinguas all inherit the same bottleneck.**
- **Authorability win in practice:** the linear form was designed for *both* machine-parseability and human-readability; the deployed **Sonetto** system (Tesco.com) let domain experts with **no programming training** author ontologies + business rules with CGIF underneath (via semi-automated extraction). *[3-0, high]*

**Scores:** Reasoning M (full FOL → undecidable at the top) · HW/vector L · Compositional H · Authorability **H** (best in the lineage; dual-readable + standardized + deployed). **Lesson: standardization against a logic is one of the two disciplines that beats accretion.**

## 4. UNL / AMR interlinguas — corpus-anchoring as the other surviving discipline

- **AMR** = rooted, directed, edge-labeled DAG using **~100 relations** + PropBank framesets, equivalent to conjunctions of logical triples. Survived as an **active NLP annotation standard**; UNL (Universal Words + ~46 relations) faded. *[3-0, high; Banarescu et al. 2013]*
- **Caveat the verifier flagged:** "small primitive set" applies to AMR's ~100 **relations**, not its **concept** vocabulary (open: thousands of PropBank framesets + open English words). AMR pairs a *small relation set* with an *open concept lexicon* — and it's **corpus-anchored** (PropBank), which is the discipline that anchored its meaning.
- **⚠️ Refuted (0-3):** the tidy claim that "AMR explicitly declares itself NOT an interlingua, which is why it survived where UNL didn't" was **killed** — the primary text does not settle the comparative survival mechanism. Treat "AMR survived because it abandoned language-neutrality" as plausible-but-unconfirmed.

**Scores:** Reasoning M · HW/vector L · Compositional H (relations) · Authorability M (expert sembanking). **Lesson: a closed, corpus-anchored relation set + open concept lexicon is a viable governance model.**

## 5. Honest refutations (4 killed)

- CGs descend *directly* from Schank's CD — **refuted (1-2)**; CG's parents are Peirce + semantic networks, Schank is a sibling.
- "Core CGs" rest on a tiny fixed primitive set / "exactly four primitives" — **refuted (1-2 / 0-3)** as stated.
- AMR self-declares as non-interlingua — **refuted (0-3)**.
- **No confirmed claim addresses the hardware/vector axis** for any of the three (by design — that axis is VSA/HDC, excluded here).

---

## 6. How this updates the akh-medu predicate-governance recommendation

The parent survey's §5 named three prescriptions (Leibniz revisable / Harnad grounding / NSM method). This pass adds **two concrete, battle-tested governance disciplines** — the only ones that demonstrably survived:

1. **Standardize against a logic (Conceptual Graphs → ISO Common Logic).** Give the predicate vocabulary a formal, reversible, model-theoretic semantics so predicates aren't defined by accretion. akh-medu already has the e-graph + microtheory machinery; pinning predicates to a CL-like core would be the "standardization" discipline.
2. **Corpus-anchor (AMR → PropBank).** Fix a *small closed relation set* but allow an *open concept lexicon*, with relations anchored to an external annotated corpus rather than invented per-need. akh-medu's "well-known relation predicates" are already a closed-ish set; anchoring them to PropBank/FrameNet-style frames is the "corpus-anchoring" discipline.

**Net, combined with the parent survey:** the **acquisition/predicate-CHOICE bottleneck broke all three relational systems despite their relational design** — so akh-medu cannot expect relationality alone to save it. The defensible path is *graph-primary spine* (Path 2) + *grounded-predicate governance* (Path 6), where "governance" now concretely means **standardize-against-a-logic and/or corpus-anchor the relation set**, plus Harnad's sensory grounding — not unprincipled `resolve_or_create_relation` accretion (which is exactly Schank's 11→14 drift in miniature).

---

## 7. Key sources

Schank & Abelson, *Scripts, Plans, Goals and Understanding* (1977); Sowa, *Conceptual Graphs* handbook chapter + "From Existential Graphs to Conceptual Graphs" (eg2cg) + CLA-2022 context slides; Delugach, CEUR Vol-352 paper3 (ISO Common Logic standard editor); Banarescu et al., "Abstract Meaning Representation for Sembanking," ACL 2013 (W13-2322); ISO/IEC 24707 Common Logic; Perlis (1982) on informal→formal translation; Sarraf & Ellis (2006) on Sonetto/Tesco.

**Open follow-ups:** dedicated UNL-vs-AMR primary-source pass (survival mechanism unconfirmed); the CD↔CG lineage relationship (direct-descent refuted); and the hardware/vector axis for these notations vs VSA/HDC (the bridge back to akh-medu's substrate choice).
