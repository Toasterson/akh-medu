# Phase 14 — Purpose-Driven Bootstrapping with Identity

Status: **In Progress** (14a-14b complete, 14c-14i pending)

Autonomous domain knowledge acquisition AND identity construction from operator statements
like "You are the Architect of the System based on Ptah" or "Be like Gandalf — a GCC
compiler expert". Purpose + identity parser extracts domain, competence level, seed
concepts, and character reference. Identity resolution via multi-source cascade
(Wikidata SPARQL + DBpedia categories + Wikipedia extraction with Hearst patterns)
resolves cultural references (mythology, fiction, history) into structured personality:
12 Jungian archetypes -> OCEAN Big Five -> behavioral parameters -> Psyche construction
(Persona + Shadow + ArchetypeWeights). The Ritual of Awakening: self-naming via
culture-specific morpheme composition (Egyptian, Greek, Norse, Latin patterns),
provenance-tracked as `DerivationKind::RitualOfAwakening` — the akh's creation myth. Domain expansion, prerequisite
discovery (Vygotsky ZPD), resource discovery (Semantic Scholar + OpenAlex + Open Library),
curriculum-ordered ingestion, and Dreyfus competence assessment — all shaped by the
constructed personality (Creator archetype weights building resources, Sage weights
theoretical depth). Bootstrap orchestrator runs meta-OODA with personality-adaptive
exploration-exploitation. Community purpose recipes (TOML with identity section) shared
via ActivityPub/oxifed. 9 sub-phases (14a-14i). Builds on existing Psyche model in
`compartment/psyche.rs`.

- **Implementation plan**: `docs/ai/plans/2026-02-17-phase14-bootstrapping.md`
- **Research**: `docs/ai/decisions/005-bootstrapping-research.md`, `docs/ai/decisions/006-identity-bootstrapping-research.md`

## Phase 14a — Purpose & Identity Parser

- [x] `BootstrapError` miette diagnostic enum (4 variants: EmptyInput, NoPurpose, InvalidCompetence, Engine) with `BootstrapResult<T>`
- [x] `DreyfusLevel` enum: Novice, AdvancedBeginner, Competent, Proficient, Expert — with as_label/from_label, Display, Default(Novice)
- [x] `EntityType` enum: Deity, FictionalCharacter, HistoricalFigure, Concept, Unknown — with as_label/from_label, Display, Default(Unknown)
- [x] `IdentityRef` struct: name, entity_type, source_phrase
- [x] `PurposeModel` struct: domain, competence_level, seed_concepts, description
- [x] `BootstrapIntent` struct: purpose, identity (optional)
- [x] 5 `LazyLock<Regex>` patterns: RE_BASED_ON, RE_LIKE, RE_INSPIRED, RE_AS, RE_DOMAIN
- [x] `parse_purpose()` — regex cascade + entity type classification + domain extraction + competence keywords + seed concepts
- [x] `classify_entity_type()` — static sets: ~30 deities, ~20 fictional, ~20 historical
- [x] `extract_competence()` — keyword matching to Dreyfus levels
- [x] 12 unit tests

## Phase 14b — Identity Resolution & Ritual of Awakening

- [x] `IdentityError` miette diagnostic enum (5 variants: ResolutionFailed, NoArchetypeMapping, NamingFailed, PsycheConstructionFailed, Engine) with `IdentityResult<T>`
- [x] `CultureOrigin` enum: Egyptian, Greek, Norse, Latin, Fictional, Unknown — with as_label/from_label, Display, Default(Unknown)
- [x] `CharacterKnowledge` struct: name, entity_type, culture, description, domains, traits, archetypes
- [x] `OceanProfile`, `ArchetypeProfile`, `MorphemeTable`, `NameCandidate`, `RitualResult` types
- [x] Static tables: DOMAIN_TRAITS (12 domains), TRAIT_ARCHETYPE (35 mappings), ARCHETYPE_OCEAN (13 archetypes), ARCHETYPE_SHADOWS (9 archetypes)
- [x] 4 culture morpheme tables: Egyptian (Akh/Mer/Neb...), Greek (Archi/Neo/Proto...), Norse (All/Heim/Mjo...), Latin (Arch/Magn/Prim...)
- [x] `resolve_identity()` — static tables -> Wikidata -> Wikipedia cascade
- [x] `resolve_from_wikidata()` — sync HTTP via ureq, JSON parse
- [x] `resolve_from_wikipedia()` — REST API summary extraction
- [x] `resolve_from_static_tables()` — 15 hardcoded figures (Ptah, Thoth, Ra, Anubis, Athena, Apollo, Hermes, Odin, Thor, Gandalf, Sherlock, Spock, Turing, Einstein, Curie)
- [x] `classify_culture()` — keyword matching on name + description
- [x] `build_archetype_profile()` — trait->archetype counting, top 2 -> primary/secondary
- [x] `build_ocean_profile()` — weighted average (0.7/0.3) from ARCHETYPE_OCEAN
- [x] `build_psyche()` — full Psyche construction with domain-augmented traits, culture grammar, OCEAN tone, archetype shadows
- [x] `ritual_of_awakening()` — morpheme combination, pronounceability filter, VSA scoring, provenance recording
- [x] `is_pronounceable()` — consonant/vowel alternation heuristic
- [x] `generate_candidates()` — prefix+root+suffix combinations (capped at 100)
- [x] `DerivationKind::RitualOfAwakening` (tag 59), `DerivationKind::IdentityResolved` (tag 60)
- [x] `AgentError::Bootstrap` + `AgentError::Identity` transparent variants
- [x] `UserIntent::AwakenCommand` in NLP, wired into TUI + headless
- [x] CLI: `Commands::Awaken` with 3 subcommands (Parse, Resolve, Status)
- [x] `derivation_kind_prose()` for RitualOfAwakening and IdentityResolved
- [x] 13 unit tests (+ 12 purpose = 25 total)

## Phase 14c — Domain Expansion (Skeleton Ontology)

- [x] `DomainExpandError` miette diagnostic enum (4 variants: NoSeeds, EmptyExpansion, RateLimitReached, Engine) with `ExpandResult<T>`
- [x] `ExpansionConfig` struct: similarity_threshold, max_depth, max_concepts, max_api_calls, inter_call_delay_ms, use_conceptnet
- [x] `ExpansionPredicates` struct: 9 well-known relations in `expand:` namespace (expanded_from, instance_of, subclass_of, part_of, has_part, related_to, has_prerequisite, used_for, domain_prototype)
- [x] `ExpansionPredicates::conceptnet_relation()` mapper for IsA/PartOf/HasA/RelatedTo/HasPrerequisite/UsedFor
- [x] `ExpansionRoleVectors` struct: 4 role vectors (concept, relation, source, depth)
- [x] `ConceptSource` enum: Seed, Wikidata, Wikipedia, ConceptNet — with Display
- [x] `CandidateConcept`, `DiscoveredRelation`, `ExpansionResult` data types
- [x] `DomainExpander` struct with `new()` and `expand()` methods
- [x] `build_domain_prototype()` — bundle encoded seed labels into prototype vector
- [x] `is_within_boundary()` — VSA similarity check against domain prototype
- [x] `query_wikidata()` — wbsearchentities + wbgetentities for P31/P279/P361/P527
- [x] `query_wikipedia()` — categories + categorymembers with meta-category filter
- [x] `query_conceptnet()` — edges for IsA/PartOf/HasA/RelatedTo/HasPrerequisite/UsedFor
- [x] `api_call()` — rate-limited HTTP with counter and inter-call delay
- [x] `normalize_label()` — lowercase, trim, hyphens/underscores→spaces
- [x] Deduplication by normalized label
- [x] `insert_into_kg()` — entity creation, relation triples, provenance recording
- [x] `DerivationKind::DomainExpansion` (tag 61) in provenance.rs
- [x] `derivation_kind_prose()` arm for DomainExpansion in explain.rs
- [x] `AgentError::DomainExpand` transparent variant in agent/error.rs
- [x] `AwakenAction::Expand` CLI subcommand with --seeds, --purpose, --threshold, --max-concepts, --no-conceptnet
- [x] `pub mod expand` + re-exports in bootstrap/mod.rs
- [x] ~18 unit tests (config defaults, display, normalize, dedup, JSON parsing, boundary, URL encoding)
