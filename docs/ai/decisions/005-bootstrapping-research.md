# ADR-005: Bootstrapping Research — Purpose-Driven Knowledge Acquisition, Self-Directed Learning, and Competence Assessment

- **Date**: 2026-02-17
- **Status**: Accepted
- **Context**: Deep research for Phase 14 — enabling operators to start akh instances with a purpose statement (e.g., "GCC compiler expert" or "meal planning assistant") and have the system autonomously discover, acquire, and validate the necessary knowledge

## Research Question

Can akh-medu autonomously bootstrap domain expertise from a natural-language purpose statement, without manual specification of skills, books, or knowledge sources — and without LLM dependency?

## Key Findings

### 1. Curriculum Learning and Self-Paced Ordering

**Bengio et al. (2009)** demonstrated that training on examples ordered from simple to complex significantly improves learning outcomes — applicable to ordering knowledge acquisition by prerequisite difficulty.

**Kumar et al. (2010)** introduced self-paced learning: the learner itself selects which examples to train on, preferring those with lowest current loss. Maps to akh-medu's gap analysis selecting the most-learnable concepts first.

**Mapping to akh-medu**: The topological sort of the prerequisite graph gives a natural curriculum. Within each prerequisite tier, concepts are ordered by VSA similarity to existing knowledge (most similar = easiest to integrate). The autonomous gap analysis module already measures coverage scores that serve as the "loss" signal.

### 2. Intrinsic Motivation and Compression Progress (Schmidhuber)

**Schmidhuber (2009)** formalized curiosity as the desire for compression progress — the first derivative of subjective compressibility. An agent should pursue knowledge that maximally improves its ability to compress (predict, explain) observations.

**PowerPlay (Schmidhuber 2011)** invents its own problems, finding the simplest still-unsolvable problem to work on next. Maps directly to selecting the knowledge gap with highest coverage potential from the ZPD frontier.

**Pathak et al. (2017)** operationalized curiosity as prediction error in a learned feature space.

**Mapping to akh-medu**: Compression progress = delta in KG coverage score + VSA reconstruction quality between learning episodes. The OODA utility scoring (Phase 8c) can incorporate a `curiosity_bonus` proportional to expected compression progress. The `autonomous::gap` module's `coverage_score` provides the signal.

### 3. Prerequisite Structure Discovery

**Pan et al. (2017)** pioneered prerequisite relation learning from MOOCs using concept co-occurrence, semantic similarity, and lecture ordering. Achieved 5.9-48% F1 improvement over baselines.

**Liang et al. (2019)** explored unsupervised prerequisite identification using KGs, combining document-based, Wikipedia-based, and graph-based features via voting.

**Roy et al. (2019)** mined MOOC lecture transcripts using temporal ordering of concept appearances across lecture sequences.

**Mapping to akh-medu**: When ingesting textbooks/documentation, chunk ordering provides a natural prerequisite signal (chapter 1 concepts prerequisite to chapter 5). Create `prerequisite-of` relation triples. Topological sort of the prerequisite subgraph gives learning order. VSA asymmetric similarity detects prerequisite relationships (if concept B's vector can be partially reconstructed by binding A with a "has-part" role vector).

### 4. Competency Modeling and Skill Ontologies

**ESCO** (European Skills, Competences, Qualifications) maps skills, competences, and qualifications to occupational requirements via a three-pillar taxonomy (Occupations, Skills, Qualifications).

**O\*NET** profiles occupations with specific Knowledge, Skills, and Abilities (KSAs).

**Guzman-Garcia et al. (2021)** proposed a competency ontology that explicitly models relationships between competencies, knowledge units, skills, and learning activities.

**Mapping to akh-medu**: Competencies as Entity symbols with well-known relation predicates (`requires-knowledge`, `requires-skill`, `has-subcompetency`). The `SkillManifest` extends with competency declarations. SPARQL queries over the competency subgraph find deficits. Each competency gap becomes an agent goal with success criteria derived from the competency model.

### 5. Zone of Proximal Development (Vygotsky)

**Vygotsky's ZPD (1978)** defines three zones: (1) known independently, (2) achievable with scaffolding (the ZPD), (3) beyond current reach. The ZPD is the optimal learning zone.

**Mapping to akh-medu** — operationalized via VSA + KG dual index:
- **Zone 1 (Known)**: Concepts with ≥N triples in KG and VSA similarity > 0.7 to multiple known concepts
- **Zone 2 (ZPD)**: Concepts whose prerequisites are all in Zone 1, VSA similarity 0.3-0.7 to existing knowledge
- **Zone 3 (Beyond)**: Prerequisites not met, VSA similarity < 0.3 to all known concepts

The agent should focus learning on Zone 2 concepts. As the KG grows, the ZPD naturally expands outward.

### 6. Ontology Learning Layer Cake

**Buitelaar et al. (2005)** defined the ontology learning layer cake: term extraction → synonym detection → concept formation → taxonomic relations → non-taxonomic relations → axioms.

**OntoLearn (Navigli & Velardi, 2004)**: Extracts domain terminology from corpora, places terms in existing ontology using gloss-based similarity.

**Text2Onto (Cimiano & Volker, 2005)**: Probabilistic ontology model from text with incremental updates.

**FRED (Gangemi et al., 2017)**: Machine reading for semantic web, produces RDF/OWL ontologies from text.

**Mapping to akh-medu**: The library ingestion pipeline already performs term extraction and concept formation. Extend with taxonomy detection (Hearst patterns: "X such as Y", "X including Y") and non-taxonomic relation extraction. VSA distributional similarity groups synonyms. E-graph captures concept equivalences.

### 7. Knowledge Gap Analysis

**CWA vs OWA**: The Closed World Assumption treats missing triples as false; the Open World Assumption treats them as unknown. For bootstrapping, the CWA is appropriate within a defined domain scope — if a concept *should* have a property (based on schema discovery) but doesn't, that's a gap.

**SPARQL completeness queries**: Detect missing relations, underspecified entities, dead-end nodes. The existing `autonomous::gap` module already detects dead-end entities, missing predicates, and incomplete types.

**VSA similarity gap detection**: Concepts that should be similar (based on shared context) but have low VSA similarity indicate knowledge gaps between them. The HNSW index enables efficient gap frontier detection.

**Competency Question evaluation**: Generated from the domain skeleton, CQs probe knowledge completeness at increasing Bloom's taxonomy levels (Remember → Understand → Apply → Analyze → Evaluate → Create).

### 8. Seed Concept Expansion

**Wikidata SPARQL traversal**: Given seed entity QIDs, traverse `instance-of`, `subclass-of`, `part-of`, `has-part` properties to depth 2-3. Free, structured, multilingual.

**Wikipedia category tree**: Human-curated hierarchical organization. Extract categories from seed articles, retrieve member articles, filter by VSA similarity to domain prototype.

**ConceptNet (Speer et al., 2017)**: 36 relation types across 21M assertions. Key relations: `RelatedTo`, `IsA`, `PartOf`, `HasPrerequisite`, `UsedFor`.

**VSA distributional expansion**: Bundle seed concept vectors → domain prototype. From ingested text, find terms whose vectors have high Hamming similarity to the prototype.

**Boundary detection**: Stop expanding when VSA similarity to domain prototype drops below threshold (0.6). Maximum 3 hops from any seed concept.

### 9. Resource Discovery APIs

Four free, no-auth-required APIs for autonomous resource acquisition:

| API | Endpoint | Best For |
|-----|----------|----------|
| **Semantic Scholar** | `api.semanticscholar.org/graph/v1/paper/search` | Academic papers, influential citation count, open access PDFs |
| **OpenAlex** | `api.openalex.org/works?search=...` | Scholarly works with Wikipedia-aligned concept tags |
| **CrossRef** | `api.crossref.org/works?query=...` | DOI resolution, metadata, licensing |
| **Open Library** | `openlibrary.org/search.json?q=...` | Books, textbooks, subject discovery |

**Quality heuristics**: Citation count normalized by age, venue ranking, open access priority, abstract VSA similarity to domain prototype.

### 10. KB Bootstrapping Strategies (Lessons from Cyc, NELL, YAGO, DBpedia)

**Cyc**: Manual bootstrapping over 30+ years. Key lesson: microtheories (context-dependent knowledge bundles) prevent cross-domain interference.

**NELL**: Continuous web extraction with multi-extractor cross-validation. Key lesson: multiple independent extractors constraining each other dramatically improves precision.

**YAGO**: Combined Wikipedia extraction with WordNet taxonomy. Key lesson: "small but well-crafted top-level schema + large messy taxonomy" achieves 95% precision.

**DBpedia**: Template-based extraction from Wikipedia infoboxes. Key lesson: structured sources yield much higher precision than free text.

**BOLT-K (Gupta et al., 2019)**: Bootstraps ontology learning for new domains by transferring from existing related domains.

**Mapping to akh-medu**: Start with a minimal seed schema (~50 universal predicates). Ingest 2-3 anchor documents for structured extraction. Use gap-driven expansion with NELL-style multi-extractor cross-validation. Microtheories via named graphs in oxigraph.

### 11. Goal-Directed Knowledge Acquisition (HTN + GOAP)

**HTN Planning**: Decomposes complex tasks into subtasks via methods — directly applicable to "become a GCC expert" → "learn front-end" → "learn parsing" → ... For learning, add a "prerequisite-first" planning strategy generating steps in topological order.

**GOAP (Orkin)**: STRIPS-like preconditions and effects. Learning actions modeled as: precondition "has concepts X, Y", action "ingest document about Z", effect "has concept Z".

**HTN Learning (Hogg et al., 2014)**: Learn decomposition rules from observed successful task completions.

**Mapping to akh-medu**: The existing `Plan` with `PlanStep`s is already HTN-like. Add prerequisite-first as a third planning strategy. GOAP-style preconditions checked against KG state before selecting learning actions.

### 12. Dreyfus Skill Acquisition Model

**Dreyfus & Dreyfus (1980)** defined five stages: Novice → Advanced Beginner → Competent → Proficient → Expert.

**Operationalized in akh-medu** via KG metrics:

| Level | Coverage Score | Indicators |
|-------|---------------|------------|
| Novice (< 0.2) | < 20 triples, no type clusters | Basic definitions only |
| Advanced Beginner (0.2-0.4) | Some relations, 1-2 type clusters | Recognizes patterns |
| Competent (0.4-0.6) | Multiple type clusters, co-occurring predicates | Can reason about domain |
| Proficient (0.6-0.8) | Rich relations, hierarchies, few gaps | Deep understanding |
| Expert (0.8+) | Cross-domain connections, edge cases, alternatives | Can evaluate tradeoffs |

### 13. Community Knowledge Sharing (Purpose Recipes)

**Purpose recipes**: Shareable bootstrap specifications capturing prerequisite graphs, resource URLs, validation criteria, and estimated effort. Format: TOML with purpose metadata, seed requirements, prerequisite ordering, resource lists, and validation thresholds.

**ActivityPub federation via oxifed**: Purpose recipes as ActivityPub Objects shared between akh instances. Share structure (syllabus) but not content (copyrighted material). Reputation system via Like/Announce activities.

**Skillpack distribution**: Successful bootstraps can export accumulated triples as installable skillpacks, allowing other instances to skip the ingestion phase.

### 14. Exploration-Exploitation Tradeoff in Learning

Early bootstrap (Novice/Advanced Beginner): favor breadth — build broad scaffold across prerequisite graph. Mid bootstrap (Competent): balance breadth and depth using alternating strategies. Late bootstrap (Proficient/Expert): favor depth — fill specific gaps.

Automated via Dreyfus-level-dependent exploration weight: `exploration_weight = 1.0 - (dreyfus_level * 0.7)`.

### 15. Metacognitive Self-Assessment

**Azevedo & Aleven (2013)**: Three metacognitive components — knowledge (what the system knows about what it knows), monitoring (ongoing assessment), control (using monitoring to adjust behavior).

**Mapping to akh-medu**: Metacognitive knowledge = `GapAnalysisResult.coverage_score`. Monitoring = reflection system running every N cycles. Control = `compute_adjustments()` for priority changes and goal suggestions.

**Calibration**: `usefulness(domain) = coverage_score * (1 - dead_end_ratio) * type_diversity_factor`. Thresholds: ≥0.6 basic queries, ≥0.75 reasoning tasks, ≥0.9 expert-level questions.

## Decision

Implement Phase 14 as an 8-sub-phase bootstrapping system:

### 14a — Purpose Parsing
Parse natural-language purpose statements into structured `PurposeModel` with domain, focus, competence level, and seed concepts. Pattern matching + Wikidata entity linking for validation.

### 14b — Domain Expansion (Skeleton Ontology)
Multi-source seed concept expansion (Wikidata SPARQL + Wikipedia categories + ConceptNet) with VSA similarity boundary detection. Output: concept map with 50-200 nodes and typed relations.

### 14c — Prerequisite Discovery and ZPD Classification
Build prerequisite DAG from anchor documents using temporal ordering heuristics + VSA asymmetric similarity. Classify all concepts into Known/Proximal/Beyond zones. Select ZPD concepts for curriculum.

### 14d — Resource Discovery
Multi-API search (Semantic Scholar, OpenAlex, Open Library) with quality scoring heuristics. Scaffold resource selection by difficulty level (intro material first for frontier concepts).

### 14e — Iterative Ingestion with Curriculum Ordering
Topological-sort-ordered ingestion using the library pipeline. Self-paced within each prerequisite tier (easiest-first by VSA similarity). NELL-style multi-extractor cross-validation for extraction quality.

### 14f — Competence Assessment
Dreyfus-level estimation from combined KG coverage + schema discovery + gap analysis metrics. Competency question evaluation at Bloom's taxonomy levels. VSA structural analysis for geometric competence signal.

### 14g — Bootstrap Orchestrator
Meta-OODA loop driving the 6-stage bootstrap pipeline (parse → expand → discover → acquire → assess → iterate). Uses existing goal/plan/reflect infrastructure. Termination: competence assessor reports sufficient coverage.

### 14h — Community Recipe Sharing
TOML-format purpose recipes with prerequisite graphs, resource lists, and validation criteria. ActivityPub sharing via oxifed. Skillpack export from successful bootstraps.

## Architecture Advantage: 70% Infrastructure Already Exists

The critical insight from this research: akh-medu already possesses approximately 70% of the infrastructure needed for autonomous bootstrapping. The following table maps research concepts to existing subsystems:

| Research Concept | Existing Module | Extension Needed |
|-----------------|----------------|-----------------|
| HTN decomposition | `goal::decompose_goal()` | Domain-aware HTN methods |
| Gap detection | `autonomous::gap::analyze_gaps()` | Already strong |
| Schema discovery | `autonomous::schema::discover_schema()` | Already strong |
| Prerequisite graph | `graph::KnowledgeGraph` (petgraph) | Add topological sort |
| Resource ingestion | `library/` pipeline | Already complete (PDF/EPUB/HTML) |
| Reflection | `agent::reflect()` | Add Dreyfus estimation |
| Exploration/exploitation | Utility scoring in OODA | Adjust weights by Dreyfus level |
| Session persistence | `agent::persist_session()` | Already complete |
| Microtheories | Compartments + named graphs | Already planned (Phase 9a) |
| Skillpack lifecycle | `skills::SkillManager` | Add dependency resolution |
| Operator interaction | `agent::tools::UserInteractTool` | Add plan presentation flow |

New components needed: purpose parser, domain expander tool, Dreyfus estimator, bootstrap orchestrator, purpose recipe format.

## The Bootstrap Pipeline

```
OPERATOR: "I want an akh that is a GCC compiler expert"
                          │
                [14a: Purpose Parsing]
                Parse intent → PurposeModel { domain, focus, level, seeds }
                          │
                [14b: Domain Expansion]
                Wikidata + Wikipedia + ConceptNet → skeleton ontology (50-200 nodes)
                VSA similarity boundary detection
                          │
                [14c: Prerequisite Discovery + ZPD]
                Anchor doc ingestion → prerequisite DAG
                Classify: Known / ZPD / Beyond
                          │
                [14d: Resource Discovery]
                Semantic Scholar + OpenAlex + Open Library → ranked resource list
                Quality scoring, scaffolded by difficulty
                          │
                [14e: Iterative Ingestion]
                Topological sort → curriculum order
                Library pipeline → concepts → KG triples → VSA vectors
                NELL-style multi-extractor cross-validation
                          │
                [14f: Competence Assessment]
                Dreyfus estimation + competency questions + graph completeness
                          │
                [14g: Bootstrap Orchestrator]  ←── loops back to 14c
                Meta-OODA: gap → prioritize → acquire → assess → repeat
                Terminates when competence threshold met
                          │
                [14h: Community Sharing]
                Export purpose recipe (TOML) + skillpack
                Share via ActivityPub / oxifed
```

## Sources

### Curriculum Learning and Self-Paced Learning
- [Bengio et al. 2009, "Curriculum Learning," ICML](https://dl.acm.org/doi/10.1145/1553374.1553380)
- [Kumar et al. 2010, "Self-Paced Learning for Latent Variable Models," NeurIPS](https://papers.nips.cc/paper/3923-self-paced-learning-for-latent-variable-models)
- [Jiang et al. 2015, "Self-Paced Curriculum Learning," AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/9608/9467)
- [Narvekar et al. 2020, "Curriculum Learning for RL Domains," JMLR](https://jmlr.org/papers/v21/20-212.html)
- [Soviany et al. 2021, "Curriculum Learning: A Survey," IJCV](https://arxiv.org/pdf/2101.10382)

### Intrinsic Motivation and Autodidactic AI
- [Schmidhuber 2009, "Driven by Compression Progress"](https://arxiv.org/abs/0812.4360)
- [Schmidhuber 2011, "PowerPlay"](https://arxiv.org/abs/1112.5309)
- [Pathak et al. 2017, "Curiosity-driven Exploration," ICML](https://arxiv.org/abs/1705.05363)
- [Burda et al. 2018, "Large-Scale Study of Curiosity-Driven Learning"](https://pathak22.github.io/large-scale-curiosity/)

### Prerequisite Structure Discovery
- [Pan et al. 2017, "Prerequisite Relation Learning for Concepts in MOOCs," ACL](https://aclanthology.org/P17-1133/)
- [Liang et al. 2019, "Exploring KGs for Prerequisite Identification"](https://slejournal.springeropen.com/articles/10.1186/s40561-019-0104-3)
- [Roy et al. 2019, "Mining MOOC Lecture Transcripts for Concept Dependencies," EDM](https://files.eric.ed.gov/fulltext/ED593223.pdf)
- [Jiang et al. 2023, "Continual Pre-Training for Concept Prerequisite Learning"](https://www.mdpi.com/2227-7390/11/12/2780)

### Competency Modeling
- [ESCO Ontology](https://data.europa.eu/esco/model)
- [O*NET Occupational Information Network](https://www.onetcenter.org/)
- [Guzman-Garcia et al. 2021, "Competency Ontology for Learning Environments"](https://slejournal.springeropen.com/articles/10.1186/s40561-021-00160-z)

### Zone of Proximal Development
- [Vygotsky 1978, "Mind in Society"](https://www.simplypsychology.org/zone-of-proximal-development.html)
- [Rafner et al. 2022, "Scaffolding Human Champions," Human Arenas](https://link.springer.com/article/10.1007/s42087-022-00304-8)

### Ontology Learning
- [Buitelaar et al. 2005, "Ontology Learning from Text"](https://doi.org/10.3233/978-1-60750-945-0)
- [Navigli & Velardi 2004, "OntoLearn"](https://doi.org/10.1162/089120104323093276)
- [Cimiano & Volker 2005, "Text2Onto"](https://doi.org/10.1007/11428817_21)
- [Gangemi et al. 2017, "FRED"](https://doi.org/10.3233/SW-160240)

### Knowledge Base Bootstrapping
- [Lenat 1995, "CYC," CACM](https://doi.org/10.1145/219717.219745)
- [Suchanek et al. 2007, "YAGO," WWW](https://doi.org/10.1145/1242572.1242667)
- [Lehmann et al. 2015, "DBpedia," Semantic Web Journal](https://doi.org/10.3233/SW-140134)
- [Mitchell et al. 2018, "NELL," CACM](https://doi.org/10.1145/3191513)
- [Gupta et al. 2019, "BOLT-K," WWW](https://dl.acm.org/doi/abs/10.1145/3308558.3313511)
- [Vrandecic & Krotzsch 2014, "Wikidata," CACM](https://doi.org/10.1145/2629489)

### Knowledge Gap Analysis and Completeness
- [Reiter 1978, "Closed World Assumption"](https://doi.org/10.1007/978-1-4684-3384-5_3)
- [Zaveri et al. 2016, "Linked Data Quality"](https://doi.org/10.3233/SW-150175)
- [Pipino et al. 2002, "Data Quality Assessment," CACM](https://doi.org/10.1145/505248.506010)
- [Bezerra et al. 2013, "Evaluating Ontologies with Competency Questions"](https://doi.org/10.1109/WI-IAT.2013.199)
- [Gruninger & Fox 1995, "Methodology for Design and Evaluation of Ontologies"](https://citeseerx.ist.psu.edu/document?doi=10.1.1.44.8723)

### Seed Expansion and Resource Discovery
- [Speer et al. 2017, "ConceptNet 5.5," AAAI](https://doi.org/10.1609/aaai.v31i1.11164)
- [Kinney et al. 2023, "Semantic Scholar Open Data Platform"](https://doi.org/10.48550/arXiv.2301.10140)
- [Priem et al. 2022, "OpenAlex"](https://doi.org/10.48550/arXiv.2205.01833)

### HTN Planning and Goal-Directed Acquisition
- [HTN Planning Overview](https://en.wikipedia.org/wiki/Hierarchical_task_network)
- [Orkin, "GOAP for Games"](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=0c35d00a015c93bac68475e8e1283b02701ff46b)
- [Hogg et al. 2014, "Learning HTN Domains from Traces"](https://www.cse.lehigh.edu/~munoz/Publications/AIJ14.pdf)

### Dreyfus Skill Acquisition
- [Dreyfus & Dreyfus 1980, "Five-Stage Model"](https://www.bumc.bu.edu/facdev-medicine/files/2012/03/Dreyfus-skill-level.pdf)

### Metacognition and Self-Regulated Learning
- [Azevedo & Aleven 2013, "Metacognition and Learning Technologies," Springer](https://link.springer.com/book/10.1007/978-1-4419-5546-3)
- [Schraw & Moshman 1995, "Metacognitive Theories," Educational Psychology Review](https://doi.org/10.1007/BF02212307)

### Domain Modeling
- [Schraagen et al. 2000, "Cognitive Task Analysis," CRC Press](https://doi.org/10.4324/9781410605501)
- [Card et al. 1983, "GOMS," CRC Press](https://doi.org/10.1201/9780203736166)
- [Schreiber et al. 2000, "CommonKADS," MIT Press](https://mitpress.mit.edu/9780262193009/)

### Federation
- [W3C 2018, "ActivityPub," W3C Recommendation](https://www.w3.org/TR/activitypub/)

### VSA and Neuro-Symbolic
- [Kanerva 2009, "Hyperdimensional Computing"](https://doi.org/10.1007/s12559-009-9009-8)
- [egg: Equality Saturation](https://arxiv.org/abs/2004.03082)
- [IBM Neuro-Vector-Symbolic Architecture](https://research.ibm.com/projects/neuro-vector-symbolic-architecture)

## Consequences

- Phase 14 adds 8 sub-phases (14a-14h) to the roadmap
- New agent tools needed: `purpose_parse`, `domain_expand`, `assess_competence`
- New well-known predicates: `prerequisite-of`, `has-knowledge-area`, `covers-concept`, `expanded-from`, `source-quality`, `gap-type`
- New `DerivationKind` variants: `GapAnalysis`, `DomainExpansion`, `CompetenceAssessment`, `ResourceDiscovery`
- Extends `SkillManifest` with dependency resolution (`depends: Vec<String>`)
- Purpose recipe format (TOML) for community sharing via ActivityPub/oxifed
- ~70% of infrastructure already exists; new code estimated at ~2,500-3,500 lines across 8 sub-phases
- Total Phase 14 scope: ~3,500-5,000 lines including orchestration and tool integration
- Depends on: Phase 8 (agent OODA, planning, reflection), library module, autonomous gap/schema analysis
- Builds toward: Phase 12e (ActivityPub federation), Phase 13 (personal assistant can bootstrap domain knowledge)
