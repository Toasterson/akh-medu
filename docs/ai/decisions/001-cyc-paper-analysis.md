# ADR-001: Cyc Paper Analysis — Applicable Ideas for Akh-medu

- **Date**: 2026-02-17
- **Updated**: 2026-02-17 (expanded with full CycL HOL feature catalog)
- **Status**: Accepted
- **Source**: "From Generative AI to Trustworthy AI: What LLMs might learn from Cyc" (Lenat & Marcus, 2023, arXiv:2308.04445)
- **Additional sources**: CycL Wikipedia, OpenCyc ontology documentation, "First-Orderized ResearchCyc" (Ramachandran, Reagan, Goolsbey 2005), Cyc glossary

## Context

The paper describes Cyc, a 40-year symbolic AI project with tens of millions of hand-authored axioms expressed in higher-order logic (CycL). Lenat and Marcus lay out 16 desiderata for trustworthy AI and describe how Cyc addresses each. They propose hybridizing symbolic reasoning with LLMs.

Akh-medu is a neuro-symbolic engine that already combines VSA, knowledge graphs, e-graph reasoning, and an autonomous agent. This analysis identifies what Cyc does that akh-medu does not, and which ideas are worth incorporating.

## The 16 Desiderata (Paper Summary)

1. **Explanation** — auditable reasoning chains with provenance
2. **Deduction** — modus ponens, arithmetic, exhaustive search, contradiction detection
3. **Induction** — generalization from examples, temporal projection
4. **Analogy** — far-flung analogical reasoning across domains
5. **Abduction** — inference to best explanation
6. **Theory of Mind** — model of user's knowledge, capabilities, concerns
7. **Quantifier fluency** — proper handling of ∀/∃ scoping
8. **Modal fluency** — "believes that", "hopes that", nested modalities
9. **Defeasibility** — default reasoning with exceptions, belief revision
10. **Pro/Con Arguments** — structured argumentation with meta-rules for preference
11. **Contexts (Microtheories)** — first-class reasoning scopes with inheritance
12. **Meta-knowledge/reasoning** — introspection, self-model, strategy switching
13. **Explicit ethics** — inspectable ethical constraints in contextual hierarchies
14. **Sufficient speed** — resource-bounded reasoning with competitive dispatch
15. **Lingual/Embodied** — NLU/NLG, perception, manipulation
16. **Broad/Deep knowledge** — vast common-sense KB

## Cyc's Key Architectural Innovations

### Epistemological Level / Heuristic Level split
All knowledge is stored in clean higher-order logic (CycL) — the epistemological level. For performance, the same knowledge is redundantly stored in specialized representations — the heuristic level. 1,100+ specialized reasoners each operate on their own optimized representation. They competitively bid on sub-problems; the fastest applicable one wins.

### Microtheories (Contexts)
Every assertion lives in a named context (microtheory) that carries domain assumptions as implicit conjuncts. Contexts form an inheritance hierarchy: `CanadianProfessionalHockey ⊂ CanadianSports ∩ ProfessionalSports ∩ Hockey ∩ Post1900 ∩ RealWorld`. Assertions within a context are terser because shared assumptions are factored out. Cross-context reasoning has explicit rules for when entailments propagate. The `ist(context, sentence)` modal operator reifies truth relative to a context. `genlMt` specifies inheritance (multiple inheritance allowed). Lifting rules govern when entailments propagate across microtheory boundaries.

### Defeasible argumentation
Almost everything in Cyc is default-true. When multiple arguments exist for/against an answer, meta-level rules decide: more specific trumps more general, recent trumps stale, expert trumps novice, constructive trumps non-constructive. The system collects all pro/con arguments before deciding. Monotonic rules are preferred over default rules. Truth values emerge from argumentation, not from simple lookup.

### Resource-bounded competitive reasoning
Reasoners bid time estimates. Slow ones get interrupted. The general theorem prover was empirically shown to always time out — they turned it off after a decade of zero successful completions.

### Higher-Order Logic in CycL
CycL is not merely first-order — it extends FOL with:
- **Quantification over predicates and functions** — rules about entire classes of relations
- **Reification** — predicates are first-class citizens; you can assert properties about `#$isa` itself
- **Rule macro predicates** — compact higher-order shorthands (`relationAllExists`, `genls`) backed by specialized reasoners
- **Skolem functions** — automatic witnesses for existential quantification
- **Non-atomic reified terms (NARTs)** — computed entities like `(GovernmentFn France)` as first-class terms
- **Modal operators** — `ist`, belief contexts, temporal contexts
- **Self-representation** — CycL can describe its own syntax and semantics

### Ontological Predicates
CycL's ontology is organized around a small set of core predicates:
- `isa` — instance membership (element-of)
- `genls` — subcollection (subset-of), forms the type taxonomy
- `genlPreds` — predicate subsumption (biologicalMother ⊂ mother ⊂ parent)
- `genlInverse` — argument-swapped equivalence (parent ↔ child)
- `genlMt` — microtheory inheritance
- `arity` / `arg1Isa` / `arg2Isa` — relation arity and argument type constraints

### Truth Maintenance System (TMS)
Every deduction records its full list of supporting premises. When a support is retracted, the TMS automatically retracts all conclusions that depended solely on it, re-evaluates conclusions with alternative justifications, and cascades through the dependency graph. This makes the KB self-healing.

### Circumscription and Closed World Assumption
CycL uses circumscription (the only instances of a concept are those explicitly known or derivable), the unique names assumption (different constants refer to different things), and can apply the closed world assumption per-context where appropriate.

## Full Gap Analysis: What Akh-medu Lacks

### High Priority

| # | Cyc Feature | Akh-medu Status | Gap | Impact |
|---|-------------|-----------------|-----|--------|
| 1 | **Microtheories + `ist` + `genlMt`** | Has `Compartment` (core/skill/project) | Compartments are opaque containers, not first-class reasoning objects. No inheritance, no `ist` operator, no cross-compartment inference, no factored-out domain assumptions, no lifting rules. | Critical — enables contextual truth, belief systems, fictional vs real worlds |
| 2 | **Predicate hierarchy (`genlPreds` / `genlInverse`)** | Relations are flat SymbolIds | No predicate subsumption. Can't infer `parent(X,Y)` from `biologicalMother(X,Y)`. No inverse relations. | High — massive inference amplifier, minimal code |
| 3 | **Truth Maintenance System** | Provenance records sources | No automatic retraction cascade. Removing a triple leaves downstream inferences orphaned. No re-evaluation of alternative justifications. | High — makes provenance actionable, KB self-heals |
| 4 | **Defeasible reasoning** | Has confidence scores | No specificity-based override. No exception hierarchy. Competing triples resolved by confidence alone. | High — "birds fly, penguins don't" requires manual confidence tuning |
| 5 | **Pro/Con argumentation** | Has superposition reasoning | Competing hypotheses interact via vector interference, not structured arguments. No meta-rules for preference. No human-readable argument summaries. Truth values don't emerge from argumentation. | High — core trustworthiness mechanism |
| 6 | **Competitive reasoner dispatch** | 3 inference strategies + e-graph | Not organized as a marketplace. No bidding. No resource budgets. No time-bounded interruption. | Medium-High — performance and extensibility |

### Medium Priority

| # | Cyc Feature | Akh-medu Status | Gap | Impact |
|---|-------------|-----------------|-----|--------|
| 7 | **Rule macro predicates** | No equivalent | No compact higher-order shorthands for common patterns. Each pattern must be expressed as raw triples. No specialized reasoner hooks for common patterns. | Medium — knowledge entry efficiency + specialized reasoner hooks |
| 8 | **Skolem functions** | No equivalent | Existential inference produces abstract results, not concrete witness symbols. Can't query "which X satisfies ∃X: P(X)?" | Medium — makes existential reasoning concrete and queryable |
| 9 | **Argumentation-based truth values** | Confidence is a single `f64` | Truth determination is lookup-based, not argumentation-based. No collection of supporting/opposing evidence before committing a truth value. | Medium — deepens argumentation from presentation layer to core semantics |
| 10 | **Arity + type constraints** | Relations have no declared constraints | Can assert malformed triples (wrong number of args, wrong types). No enforcement at assertion time. | Medium — catches errors early, improves KB quality |
| 11 | **Temporal projection** | Triples have timestamps | No confidence decay over time. No relation-specific half-lives. "Is X still true?" has no temporal model. | Medium — knowledge staleness is a real problem |
| 12 | **Contradiction detection** | E-graph finds equivalences | No active check when adding triples against existing knowledge. No functional-predicate violation detection. No disjointness checking. | Medium — KB integrity |

### Lower Priority

| # | Cyc Feature | Akh-medu Status | Gap | Impact |
|---|-------------|-----------------|-----|--------|
| 13 | **Circumscription / CWA** | No equivalent | Doesn't distinguish "unknown" from "false". No closed-world reasoning. No unique-names assumption. | Lower — useful but adds complexity; best as per-context toggle |
| 14 | **Second-order quantification** | E-graph has `AkhLang` | Can't write rules that quantify over predicates ("for all transitive relations R..."). Rules are first-order only. | Lower — very powerful but hardest to implement efficiently |
| 15 | **NARTs (computed entities)** | `SymbolKind::Composite` exists | Not used for functional term construction. No `(GovernmentFn France)` style computed entities with structural unification. | Lower — nice-to-have for computed entities |

### Already Strong

| Cyc Feature | Akh-medu Status |
|-------------|-----------------|
| **Explanation/Provenance** | Persistent provenance ledger with 20+ derivation kinds, multi-index |
| **Deduction** | Spreading activation, backward chaining, e-graph rewriting |
| **Analogy** | VSA bind/unbind for A:B::C:? analogical reasoning |
| **Meta-reasoning** | Agent reflection, strategy rotation, planning with backtracking |
| **Speed** | SIMD-accelerated VSA, HNSW ANN, tiered storage |
| **Contexts (basic)** | Compartments provide isolation (but not inheritance or cross-context reasoning) |

## Decision

Adopt Cyc-inspired enhancements as **Phase 9** of the roadmap, organized into 15 sub-phases ordered by priority and dependency:

**High priority (9a–9f)**:
- **9a**: Microtheories — `ist` operator, `genlMt` inheritance, lifting rules, domain assumptions
- **9b**: Predicate hierarchy — `genlPreds`, `genlInverse`, predicate subsumption inference
- **9c**: Truth Maintenance System — support tracking, retraction cascades, re-evaluation
- **9d**: Defeasible reasoning — specificity override, exception chains, default-truth semantics
- **9e**: Pro/Con argumentation — structured argument collection, meta-rules, verdict derivation
- **9f**: Competitive reasoner dispatch — `Reasoner` trait, bidding, resource budgets, specialized reasoners

**Medium priority (9g–9l)**:
- **9g**: Rule macro predicates — compact higher-order shorthands with specialized reasoner hooks
- **9h**: Skolem functions — existential witnesses as concrete symbols
- **9i**: Argumentation-based truth values — truth emerges from argument balance, not lookup
- **9j**: Arity and type constraints — relation arity declarations, argument type enforcement
- **9k**: Temporal projection — confidence decay models, relation-specific half-lives
- **9l**: Contradiction detection — functional violations, disjointness, temporal conflicts

**Lower priority (9m–9o)**:
- **9m**: Circumscription / CWA — closed-world assumption, unique names, per-context toggle
- **9n**: Second-order quantification — rules over predicates, predicate variables in inference
- **9o**: Non-atomic reified terms — functional term construction, structural unification

See `docs/ai/plans/2026-02-17-phase9-cyc-inspired.md` for the detailed implementation plan.

## Consequences

- Compartment model will need significant rework (from flat scoping to hierarchical reasoning contexts with `ist` and `genlMt`)
- Triple model may need new fields (specificity level, temporal decay parameters, support set for TMS)
- Relation model needs arity, argument types, hierarchy links (`genlPreds`, `genlInverse`)
- New subsystems: `argumentation` module, `dispatch` module, `tms` module
- Provenance ledger becomes bidirectional (not just recording but driving retraction)
- Inference subsystem will need a dispatcher/scheduler layer
- Rule macros create a new abstraction layer between raw triples and user-facing knowledge entry
- These changes build on existing infrastructure (provenance, compartments, confidence, e-graph) rather than replacing it
- Estimated total scope: ~7,000–10,500 lines across 15 sub-phases
