# Phase 14 — Purpose-Driven Bootstrapping with Identity

Status: **In Progress** (14a-14h complete, 14i-14m pending)

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

## Phase 14d — Prerequisite Discovery & ZPD Classification

- [x] `PrerequisiteError` miette diagnostic enum (4 variants: NoConcepts, CycleDetected, EmptyCurriculum, Engine) with `PrerequisiteResult<T>`
- [x] `PrerequisiteConfig` struct: known_min_triples, known_similarity_threshold, proximal_min_prereq_coverage, proximal_similarity_low/high, min_edge_confidence, weight_conceptnet/structural/vsa, max_prereqs_per_concept
- [x] `PrerequisitePredicates` struct: 5 well-known relations in `prereq:` namespace (prerequisite_of, zpd_zone, curriculum_tier, prereq_coverage, similarity_to_known)
- [x] `ZpdZone` enum: Known, Proximal, Beyond — with as_label/from_label, Display, Ord
- [x] `PrerequisiteSource` enum: ConceptNet, Structural, VsaSimilarity — with Display
- [x] `PrerequisiteEdge`, `CurriculumEntry`, `PrereqAnalysisResult` data types
- [x] `PrerequisiteAnalyzer` struct with `new()` and `analyze()` methods
- [x] `collect_domain_concepts()` — gather concepts from expansion result
- [x] `discover_conceptnet_prereqs()` — Strategy 1: read expand:has_prerequisite triples
- [x] `discover_structural_prereqs()` — Strategy 2: subclass_of/part_of/instance_of heuristics
- [x] `discover_vsa_prereqs()` — Strategy 3: VSA asymmetric bind+similarity (O(n²))
- [x] `merge_edges()` — deduplicate, combine confidences, cap max_prereqs_per_concept
- [x] `break_cycles()` — iterative DFS cycle detection, remove lowest-confidence edge
- [x] `build_known_bundle()` — VSA bundle of high-triple-count concepts
- [x] `topological_sort()` — Kahn's algorithm → tier numbers
- [x] `classify_zpd()` — bottom-up classification: Known/Proximal/Beyond
- [x] `generate_curriculum()` — sort by (tier ASC, zone ASC, similarity DESC)
- [x] `persist_to_kg()` — prerequisite_of triples + ZPD triples + provenance
- [x] `DerivationKind::PrerequisiteDiscovered` (tag 62), `DerivationKind::ZpdClassification` (tag 63) in provenance.rs
- [x] `derivation_kind_prose()` arms for both variants in explain.rs
- [x] `format_derivation_kind()` arms for both variants in main.rs
- [x] `AgentError::Prerequisite` transparent variant in agent/error.rs
- [x] `AwakenAction::Prerequisite` CLI subcommand with --seeds, --purpose, --known-threshold, --zpd-low, --zpd-high
- [x] `pub mod prerequisite` + re-exports in bootstrap/mod.rs
- [x] ~18 unit tests (config defaults, ZPD roundtrip, source display, errors, merge, cycles, toposort, classification, curriculum ordering)

## Phase 14e — Resource Discovery

- [x] `ResourceDiscoveryError` miette diagnostic enum (2 variants: NoProximalConcepts, Engine) with `ResourceResult<T>`
- [x] `ResourceDiscoveryConfig` struct: max_api_calls, delay_ms, max_per_concept, min_quality, dedup_threshold, per-API enable flags
- [x] `ResourcePredicates` struct: 9 well-known relations in `resource:` namespace (title, url, source_api, quality_score, covers_concept, difficulty, open_access, abstract_text, year)
- [x] `ResourceApi` enum: SemanticScholar, OpenAlex, OpenLibrary — with Display
- [x] `DiscoveredResource`, `ResourceDiscoveryResult` data types
- [x] `ResourceDiscoverer` struct with `new()` and `discover()` methods
- [x] `search_semantic_scholar()` — paper search API with title/url/abstract/year/citations/openAccess
- [x] `search_openalex()` — works search with inverted-index abstract reconstruction
- [x] `search_open_library()` — book search via Open Library
- [x] `api_call()` — rate-limited HTTP with counter and inter-call delay (same pattern as expand.rs)
- [x] `score_resource()` — weighted scoring: citation_impact(0.30) + recency(0.15) + open_access(0.20) + vsa_similarity(0.25) + source_type(0.10)
- [x] `estimate_difficulty()` — ZPD similarity → DreyfusLevel mapping
- [x] `deduplicate_resources()` — pairwise VSA similarity, keep higher-scored on threshold match
- [x] `limit_per_concept()` — cap resources per concept, keep highest quality
- [x] `reconstruct_openalex_abstract()` — rebuild text from OpenAlex inverted index
- [x] `store_resources()` — KG entity creation + 9 predicate triples + provenance recording per resource
- [x] `build_search_query()` — concept label + domain context, truncated to 120 chars
- [x] `url_encode()` — percent-encode for URL query parameters
- [x] `DerivationKind::ResourceDiscovery` (tag 66) in provenance.rs
- [x] `derivation_kind_prose()` arm for ResourceDiscovery in explain.rs
- [x] `format_derivation_kind()` arm for ResourceDiscovery in main.rs
- [x] `AkhError::ResourceDiscovery` transparent variant in error.rs
- [x] `AgentError::ResourceDiscovery` transparent variant in agent/error.rs
- [x] `AwakenAction::Resources` CLI subcommand with --seeds, --purpose, --min-quality, --max-api-calls, --no-semantic-scholar, --no-openalex, --no-open-library
- [x] `pub mod resources` + re-exports in bootstrap/mod.rs
- [x] 22 unit tests (config defaults, API display, error formatting, query building, difficulty boundaries, scoring, deduplication, abstract reconstruction, URL encoding, limit per concept)

## Phase 14f — Iterative Ingestion

- [x] Curriculum-ordered ingestion loop (tier ASC, similarity DESC)
- [x] Two-layer strategy: abstract text (always) + URL (open-access, best-effort)
- [x] Multi-extractor cross-validation (NELL-inspired, >=2 methods → boost)
- [x] Post-ingestion grounding via `ground_symbol()` per concept
- [x] SaturationTracker: consecutive zero-triple streak detection
- [x] Provenance: `DerivationKind::CurriculumIngestion` (tag 67)
- [x] Well-known `ingest:*` predicates (5 relations)
- [x] CLI: `akh awaken ingest --seeds "..." [--max-cycles N] [--saturation N] [--xval-boost F] [--no-url] [--catalog-dir PATH]`
- [x] Error chain: `IngestionError` → `AkhError` → `AgentError`
- [x] 17 unit tests (config, errors, saturation tracker, resource index, sort order, cross-validation, accumulation, aggregation)

## Phase 14g — Competence Assessment

- [x] `CompetenceError` miette diagnostic enum (3 variants: NoConcepts, InsufficientTriples, Engine) with `CompetenceResult<T>`
- [x] `CompetenceConfig` struct: min_triples_per_concept, bloom_max_depth, 5 weight parameters (sum to 1.0)
- [x] `CompetencePredicates` struct: 3 well-known relations in `assess:` namespace (dreyfus_level, competence_score, assessed_at)
- [x] `BloomLevel` enum: Remember, Understand, Apply, Analyze — with Ord, Display, as_label, all()
- [x] `CompetencyQuestion` struct (private): bloom_level, concept, related_concept, answered
- [x] `KnowledgeAreaAssessment` pub struct: name, dreyfus_level, score, triple_count, cq_answered/total, gap_count, relation_density, score_components
- [x] `ScoreComponents` pub struct: coverage, connectivity, type_diversity, relation_density, cross_domain
- [x] `BootstrapRecommendation` pub enum: Ready, ContinueLearning { estimated_cycles, focus_areas }, NeedsOperatorInput { question }
- [x] `CompetenceReport` pub struct: overall_dreyfus, overall_score, knowledge_areas, remaining_gaps, recommendation, provenance_ids
- [x] `CompetenceAssessor` struct with `new()` and `assess()` methods
- [x] Dreyfus score: 5-component weighted formula (coverage 0.30, connectivity 0.20, type_diversity 0.20, relation_density 0.15, cross_domain 0.15)
- [x] CQ evaluation: 4 Bloom levels per concept (Remember: lookup, Understand: shortest_path, Apply: >=2 outgoing, Analyze: all prereqs known)
- [x] Reuses `autonomous::gap::analyze_gaps` for coverage + dead-end ratio
- [x] Reuses `autonomous::schema::discover_schema` for type diversity
- [x] Reuses `graph::analytics::shortest_path` for Understand CQ
- [x] `score_to_dreyfus()` and `dreyfus_to_min_score()` mapping functions
- [x] Knowledge area grouping by tier bucket (foundational/intermediate/advanced)
- [x] Recommendation generation: Ready / ContinueLearning / NeedsOperatorInput
- [x] `DerivationKind::CompetenceAssessment` (tag 68) in provenance.rs
- [x] `derivation_kind_prose()` arm for CompetenceAssessment in explain.rs
- [x] `format_derivation_kind()` arm for CompetenceAssessment in main.rs
- [x] `AkhError::Competence` transparent variant in error.rs
- [x] `AgentError::Competence` transparent variant in agent/error.rs
- [x] `AwakenAction::Assess` CLI subcommand with --seeds, --purpose, --min-triples, --bloom-depth, --verbose
- [x] `pub mod competence` + re-exports in bootstrap/mod.rs
- [x] 15 unit tests (config defaults, error formatting, BloomLevel ordering/display, Dreyfus boundaries, weighted formula, type diversity cap, relation density normalization, recommendation variants, score components)

## Phase 14h — Bootstrap Orchestrator

- [x] `OrchestratorError` miette diagnostic enum (5 variants: EmptyPurpose, StageFailed, MaxCyclesExhausted, NoSessionToResume, Engine) with `OrchestratorResult<T>`
- [x] `From` impls for all 7 sub-phase errors (BootstrapError, IdentityError, DomainExpandError, PrerequisiteError, ResourceDiscoveryError, IngestionError, CompetenceError) wrapping into StageFailed with stage context
- [x] `OrchestratorConfig` struct: max_learning_cycles, plan_only, + sub-stage config overrides
- [x] `BootstrapStage` ordered enum (9 variants: ParsePurpose..Complete) with Serialize/Deserialize/Ord, as_label/from_label roundtrip
- [x] `BootstrapSession` persistent session struct with stage, cycle, purpose, intent, name, psyche, expansion_labels, assessment, timestamps, exploration_rate
- [x] `SessionAssessment` compact assessment snapshot (score, dreyfus, focus_areas, recommendation)
- [x] `Checkpoint` enum (4 variants: PurposeParsed, IdentityConstructed, LearningPlan, AssessmentComplete) for operator-visible progress
- [x] `OrchestrationResult` struct: intent, chosen_name, psyche, final_report, learning_cycles, stages_completed, target_reached, provenance_ids
- [x] `BootstrapOrchestrator::new()` — fresh bootstrap from purpose statement
- [x] `BootstrapOrchestrator::resume()` — reload persisted session and continue
- [x] `BootstrapOrchestrator::status()` — query session without running
- [x] `BootstrapOrchestrator::run()` — 8-stage pipeline with learning loop (stages 4–7 repeat up to max_cycles)
- [x] Session persistence via `engine.store().put_meta()/get_meta()` with bincode serialization
- [x] Expansion reconstruction from stored labels on resume past stage 3
- [x] `compute_exploration_rate()` — Dreyfus factor + personality factor (explorer archetype * 0.3)
- [x] `apply_personality_bias()` — Explorer(>0.6): broader expansion; Sage(>0.6): depth; Guardian(>0.6): conservative; Healer(>0.6): gap-focused
- [x] `DerivationKind::BootstrapOrchestration` (tag 69) with stage, learning_cycle, exploration_rate, target_dreyfus, current_dreyfus, current_score
- [x] `AkhError::Orchestrator`, `AgentError::Orchestrator` transparent variants
- [x] `derivation_kind_prose()` arm for BootstrapOrchestration
- [x] `format_derivation_kind()` arm for BootstrapOrchestration in main.rs
- [x] CLI: `AwakenAction::Bootstrap` with statement, --plan-only, --resume, --status, --max-cycles, --identity
- [x] 14 unit tests (config, stage ordering, label roundtrip, error formatting, From impls, session serialization, assessment serialization, checkpoints, exploration rate scenarios, result construction)

### Continuous Learning Idle Task (wired after 14h)

- [x] `ContinuousLearningError` miette diagnostic enum (4 variants: NoCuriosityTargets, NoExpansionConcepts, StageFailed, Engine) with `ContinuousLearningResult<T>`
- [x] `ContinuousLearningConfig`: max_targets(5), max_api_calls(5), min_gap_score(0.3)
- [x] `ContinuousLearningRunResult`: targets_found, resources_discovered, concepts_ingested, competence_score, dreyfus_level, provenance_ids
- [x] `run_continuous_learning()` — chains curiosity→prerequisite→resources→ingestion→assessment with soft failures
- [x] `DerivationKind::ContinuousLearning` (tag 70) with targets_explored, resources_found, concepts_ingested, dreyfus_level, gap_score_avg
- [x] 6th idle task in `IdleScheduler` with 2-hour interval
- [x] `derivation_kind_prose()` + `format_derivation_kind()` arms
- [x] 10 unit tests (config, errors, result, expansion builder, empty KG, serialization, provenance tag)

## Phase 14i — Community Recipe Sharing

- [ ] Purpose recipe TOML format with identity section
- [ ] Recipe generation from successful bootstrap
- [ ] Skillpack export (`SkillInstallPayload`)
- [ ] ActivityPub sharing via oxifed
- [ ] Recipe import + dependency resolution
- [ ] Unit tests

## Phase 14j — Extended Rule Parser (NLU Tier 1) ✓

- [x] `Negation { inner }` AbsTree variant + parser pattern (not/no/never + multilingual)
- [x] `Quantified { quantifier, scope }` variant (all/every/some/most/no + multilingual)
- [x] `Comparison { entity_a, entity_b, property, ordering }` variant (more/less/bigger than + multilingual)
- [x] `Conditional { condition, consequent }` variant (if/when/unless + multilingual)
- [x] `Temporal { time_expr, inner }` variant + multilingual temporal lexicons
- [x] `Modal { modality, inner }` variant (want/can/should/must/may + multilingual)
- [x] `RelativeClause { head, clause }` variant (that/which/who patterns)
- [x] `to_vsa()` encoding for all new AbsTree variants
- [x] Concrete grammar linearization (formal, terse, narrative, rust_gen, custom) for all new variants
- [x] 46+ unit tests (parser recognizers, linearization, edge cases)

## Phase 14k — Micro-ML NER & Intent Classification (NLU Tier 2) ✓

- [x] `NluError` miette diagnostic enum with `NluResult<T>` (created in Phase 14j/14m)
- [x] `src/nlu/mod.rs` — NLU pipeline orchestrator (created in Phase 14j/14m)
- [x] `MicroMlLayer` — ONNX NER session + tokenizer via `ort` crate
- [x] `EntitySpan` + `IntentClass` types
- [x] DistilBERT multilingual NER model integration (`Davlan/distilbert-base-multilingual-cased-ner-hrl`)
- [x] `augment_parse()` — feed NER entities back into entity resolution for re-parse
- [x] Model management: `$AKH_DATA_DIR/models/`, `akh init --with-models` download
- [x] Feature-gated: `nlu-ml = ["ort", "tokenizers"]`
- [x] Graceful degradation: skip Tier 2 if models not present
- [x] ~20 unit tests (mock ONNX session, entity span extraction, intent classification)

## Phase 14l — Small LLM Translator (NLU Tier 3) ✓

- [x] `LlmTranslator` struct with `llama-cpp-2` model handle
- [x] GBNF grammar definition constraining output to valid AbsTree JSON
- [x] System prompt for NL→AbsTree translation with few-shot examples
- [x] `translate()` — generate constrained AbsTree JSON, deserialize
- [x] Model: Qwen2.5-1.5B-Instruct Q4_K_M GGUF (~1.1 GB, Apache 2.0)
- [x] Model management: `$AKH_DATA_DIR/models/`, `akh init --with-llm` download
- [ ] Self-training data collection: store successful (input, AbsTree) pairs
- [x] Feature-gated: `nlu-llm = ["llama-cpp-2"]`
- [x] Graceful degradation: skip Tier 3 if model not present
- [x] ~15 unit tests (mock model, GBNF validation, JSON→AbsTree deserialization)

## Phase 14m — VSA Parse Ranker (NLU Tier 4) ✓

- [x] `ParseRanker` struct with exemplar memory (Jaccard similarity; HNSW upgrade deferred)
- [x] `record_success()` — store successful parse exemplars for future matching
- [x] `find_similar()` — find best matching exemplar by normalized text similarity
- [x] `has_similar_exemplar()` — quick check for known patterns
- [x] Self-improvement loop: more successful parses → better exemplar coverage
- [x] Persistence: exemplar memory via `put_meta`/`get_meta` on durable store (bincode serialization)
- [x] `DerivationKind::NluParsed` provenance variant (records tier, confidence, exemplar similarity)
- [x] `NluPipeline` orchestrator with 4-tier cascade (Tiers 2-3 feature-gated stubs)
- [x] Wire NLU pipeline into TUI `process_input_local` (before goal escalation)
- [x] 9 unit tests (exemplar recording, ranking, persistence roundtrip, eviction, case insensitivity)
