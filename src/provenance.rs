//! Provenance ledger: tracks how knowledge was derived.
//!
//! Every inference, reasoning step, or extraction produces a provenance record
//! that links the derived knowledge back to its sources. Records are persisted
//! via redb with multiple indices for efficient lookup.

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

use redb::{
    Database, MultimapTableDefinition, ReadableTable, ReadableTableMetadata, TableDefinition,
};
use serde::{Deserialize, Serialize};

use crate::error::{ProvenanceError, ProvenanceResult, StoreError};
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// Primary table: provenance_id → bincode-encoded ProvenanceRecord.
const PROVENANCE_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("provenance");

/// Index: derived_symbol_id → provenance_ids that derive that symbol.
const DERIVED_INDEX: MultimapTableDefinition<u64, u64> =
    MultimapTableDefinition::new("provenance_derived_idx");

/// Reverse index: source_symbol_id → provenance_ids that depend on that source.
const SOURCE_INDEX: MultimapTableDefinition<u64, u64> =
    MultimapTableDefinition::new("provenance_source_idx");

/// Index: derivation_kind_tag → provenance_ids with that kind.
const KIND_INDEX: MultimapTableDefinition<u8, u64> =
    MultimapTableDefinition::new("provenance_kind_idx");

// ---------------------------------------------------------------------------
// DerivationKind
// ---------------------------------------------------------------------------

/// How a piece of knowledge was derived.
///
/// Unified superset covering extraction, inference, reasoning, and aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DerivationKind {
    /// Directly extracted from source material.
    Extracted,
    /// Seed symbol provided by the user at query time.
    Seed,
    /// Inferred by following a graph edge.
    GraphEdge { from: SymbolId, predicate: SymbolId },
    /// Inferred via VSA unbind + cleanup recovery.
    VsaRecovery {
        from: SymbolId,
        predicate: SymbolId,
        similarity: f32,
    },
    /// Inferred via analogy (A:B :: C:?).
    Analogy {
        a: SymbolId,
        b: SymbolId,
        c: SymbolId,
    },
    /// Recovered as a role-filler via unbind.
    FillerRecovery {
        subject: SymbolId,
        predicate: SymbolId,
    },
    /// Derived via symbolic reasoning (e-graph rewriting).
    Reasoned,
    /// Aggregated from multiple sources.
    Aggregated,
    /// The agent made a decision during an OODA cycle.
    AgentDecision { goal: SymbolId, cycle: u64 },
    /// The agent consolidated a working memory entry to episodic memory.
    AgentConsolidation {
        reason: String,
        relevance_score: f32,
    },
    /// Derived by a forward-chaining inference rule.
    RuleInference {
        rule_name: String,
        antecedents: Vec<SymbolId>,
    },
    /// Derived via multi-path confidence fusion.
    FusedInference {
        path_count: usize,
        interference_signal: f32,
    },
    /// Identified by gap analysis as missing knowledge.
    GapIdentified { gap_kind: String, severity: f32 },
    /// Discovered via schema pattern analysis.
    SchemaDiscovered { pattern_type: String },
    /// Derived by VSA-based semantic enrichment (role classification, importance, data flow).
    SemanticEnrichment { source: String },
    /// Knowledge loaded from a compartment.
    CompartmentLoaded {
        compartment_id: String,
        source_file: String,
    },
    /// An action was vetoed by a Shadow pattern.
    ShadowVeto { pattern_name: String, severity: f32 },
    /// The psyche adjusted archetype weights or persona during reflection.
    PsycheEvolution { trigger: String, cycle: u64 },
    /// A WASM tool was executed from a skill package.
    WasmToolExecution {
        tool_name: String,
        skill_id: String,
        danger_level: String,
    },
    /// A CLI tool was executed (external binary).
    CliToolExecution {
        tool_name: String,
        binary_path: String,
        danger_level: String,
    },
    /// Knowledge derived from document ingestion in the content library.
    DocumentIngested {
        document_id: String,
        format: String,
        chunk_index: u32,
    },
    /// Atomic concept extracted from document text during library ingestion.
    ConceptExtracted {
        document_id: String,
        chunk_index: u32,
        extraction_method: String,
    },
    /// Triple inherited from an ancestor context via `ctx:specializes` chain.
    ContextInheritance {
        context: SymbolId,
        ancestor: SymbolId,
    },
    /// Triple propagated from one context to another via a lifting rule.
    ContextLifting {
        from_context: SymbolId,
        to_context: SymbolId,
        condition: String,
    },
    /// Triple inferred via predicate generalization (genlPreds).
    PredicateGeneralization {
        specific: SymbolId,
        general: SymbolId,
    },
    /// Triple inferred via predicate inverse (genlInverse).
    PredicateInverse {
        predicate: SymbolId,
        inverse: SymbolId,
    },
    /// Triple selected via defeasible override resolution.
    DefeasibleOverride {
        winner: SymbolId,
        loser: SymbolId,
        reason: String,
    },
    /// Record that a reasoner was selected via competitive dispatch (Phase 9f).
    DispatchRoute {
        reasoner: String,
        problem_kind: String,
    },
    /// An argumentation verdict was reached (Phase 9e).
    ArgumentVerdict {
        winner: SymbolId,
        pro_count: usize,
        con_count: usize,
        decisive_rule: String,
    },
    /// A rule macro expanded compact assertions into concrete triples (Phase 9g).
    RuleMacroExpansion {
        macro_name: String,
        expanded_count: usize,
    },
    /// Confidence was adjusted via temporal decay (Phase 9k).
    TemporalDecay {
        profile: String,
        original_confidence: f32,
        decayed_confidence: f32,
    },
    /// A contradiction was detected between triples (Phase 9l).
    ContradictionDetected {
        kind: String,
        existing_object: SymbolId,
        incoming_object: SymbolId,
    },
    /// A Skolem symbol was created as an existential witness (Phase 9h).
    SkolemWitness {
        existential_relation: SymbolId,
        bound_var: SymbolId,
    },
    /// A Skolem symbol was grounded to a concrete entity (Phase 9h).
    SkolemGrounding {
        skolem: SymbolId,
        concrete_entity: SymbolId,
    },
    /// A negation-as-failure query returned negative under CWA (Phase 9m).
    CwaQuery {
        context: SymbolId,
        subject: SymbolId,
        predicate: SymbolId,
    },
    /// A second-order rule was instantiated for a qualifying predicate (Phase 9n).
    SecondOrderInstantiation {
        rule_name: String,
        predicate: SymbolId,
        inferred_count: usize,
    },
    /// A NART (non-atomic reified term) was created (Phase 9o).
    NartCreation {
        function: SymbolId,
        arg_count: usize,
    },

    // --- Phase 10: Code Generation -------------------------------------------

    /// Code was generated from KG structure (Phase 10b).
    CodeGenerated {
        scope: String,
        source_count: usize,
    },
    /// Code was refined after compiler feedback (Phase 10d).
    CodeRefinement {
        attempt: u32,
        error_count: usize,
    },
    /// A reusable abstraction was discovered by library learning (Phase 10h).
    LibraryLearning {
        pattern_name: String,
        occurrences: u32,
        compression: f64,
    },

    // --- Phase 11: Autonomous Task System -------------------------------------

    /// A goal was autonomously generated from drive observation (Phase 11a).
    AutonomousGoalGeneration { drive: String, strength: f32 },

    /// A goal was decomposed via HTN method (Phase 11b).
    HtnDecomposition {
        method_name: String,
        strategy: String,
        subtask_count: usize,
    },

    /// A goal's priority was re-evaluated via argumentation (Phase 11c).
    PriorityArgumentation {
        goal: SymbolId,
        old_priority: u8,
        new_priority: u8,
        audience: String,
        net_score: f64,
    },

    /// A project was created to group related goals (Phase 11d).
    ProjectCreated { name: String },

    /// A world-monitoring watch fired (Phase 11e).
    WatchFired {
        watch_id: String,
        condition_summary: String,
    },

    /// A metacognitive evaluation was performed on a goal (Phase 11f).
    MetacognitiveEvaluation {
        goal: SymbolId,
        signal: String,
        improvement_rate: f32,
        competence: f32,
    },

    /// A resource/effort assessment was performed on a goal (Phase 11g).
    ResourceAssessment {
        goal: SymbolId,
        voc: f32,
        opportunity_cost: f32,
        diminishing_returns: bool,
    },

    /// A successful strategy was compiled into a reusable method (Phase 11h).
    ProceduralLearning {
        source_goal: SymbolId,
        method_name: String,
        step_count: usize,
    },

    // --- Phase 13: Personal Assistant ------------------------------------------

    /// An email was ingested and parsed via the email channel (Phase 13a).
    EmailIngested {
        message_id: String,
        channel_id: String,
    },

    /// An email thread was constructed or updated via JWZ (Phase 13a).
    EmailThreaded {
        thread_root_id: String,
        message_count: usize,
    },

    /// Email classified as spam/ham via OnlineHD (Phase 13b).
    SpamClassification {
        email_message_id: String,
        decision: String,
        vsa_confidence: f32,
        bayesian_score: f32,
    },

    /// Email triaged and routed (Phase 13c).
    EmailTriaged {
        email_message_id: String,
        route: String,
        importance_score: f32,
    },

    /// Structured information extracted from email message (Phase 13d).
    EmailExtracted {
        email_message_id: String,
        item_count: usize,
        kinds_found: String,
    },

    /// PIM task managed (Phase 13e).
    PimTaskManaged {
        goal_id_raw: u64,
        gtd_state: String,
        quadrant: String,
    },

    /// Calendar event managed (Phase 13f).
    CalendarEventManaged {
        event_summary: String,
        dtstart: u64,
        dtend: u64,
    },

    // --- Phase 13g: Preference Learning & Proactive Assistance ----------

    /// Preference profile updated from feedback signal (Phase 13g).
    PreferenceLearned {
        signal_kind: String,
        entity_id_raw: u64,
        weight: f32,
    },

    /// JITIR suggestion surfaced (Phase 13g).
    JitirSuggestion {
        entity_id_raw: u64,
        relevance: f32,
        serendipitous: bool,
    },

    /// Proactive assistance event (Phase 13g).
    ProactiveAssistance {
        level: String,
        suggestion_count: u32,
        acceptance_rate: f32,
    },

    // --- Phase 15: Causal World Model ----------------------------------------

    /// A causal action schema was learned or updated (Phase 15a).
    CausalSchemaLearned {
        action_name: String,
        precondition_count: u32,
        effect_count: u32,
    },

    // --- Phase 14: Identity Bootstrapping ------------------------------------

    /// The Ritual of Awakening was performed — the akh chose its name (Phase 14b).
    RitualOfAwakening {
        chosen_name: String,
        culture: String,
        vsa_score: f32,
    },

    /// An identity reference was resolved to structured character knowledge (Phase 14b).
    IdentityResolved {
        name: String,
        entity_type: String,
        culture: String,
        trait_count: usize,
    },

    // --- Phase 14c: Domain Expansion -----------------------------------------

    /// Domain skeleton ontology expanded from seed concepts (Phase 14c).
    DomainExpansion {
        seed_label: String,
        concept_count: u32,
        relation_count: u32,
        source: String,
    },

    // --- Phase 14d: Prerequisite Discovery & ZPD Classification ---------------

    /// A prerequisite relationship was discovered between concepts (Phase 14d).
    PrerequisiteDiscovered {
        from_label: String,
        to_label: String,
        edge_count: u32,
        strategy_sources: String,
    },

    /// A concept was classified into a Vygotsky ZPD zone (Phase 14d).
    ZpdClassification {
        concept_label: String,
        zone: String,
        prereq_coverage: f32,
        similarity_to_known: f32,
        curriculum_tier: u32,
    },

    // --- Phase 11i: Sleep/Consolidation Cycle ---------------------------------

    /// The agent performed a sleep/consolidation cycle (Phase 11i).
    SleepConsolidation {
        phase: String,
        merged_count: usize,
        pruned_count: usize,
    },

    // --- Phase 11j: Directed Curiosity ----------------------------------------

    /// A curiosity target was identified via information-gap analysis (Phase 11j).
    DirectedCuriosity {
        concept: SymbolId,
        gap_score: f32,
        fill_ratio: f32,
    },

    // --- Phase 14e: Resource Discovery ----------------------------------------

    /// A learning resource was discovered for a ZPD-proximal concept (Phase 14e).
    ResourceDiscovery {
        concept_label: String,
        resource_count: u32,
        sources: String,
        top_title: String,
    },

    // --- Phase 14f: Curriculum Ingestion --------------------------------------

    /// A curriculum resource was ingested for a concept (Phase 14f).
    CurriculumIngestion {
        concept_label: String,
        resource_title: String,
        triples_added: u32,
        extraction_methods: String,
        tier: u32,
        cross_validated: bool,
    },

    // --- Phase 14g: Competence Assessment ------------------------------------

    /// A competence assessment was performed on the akh's domain knowledge (Phase 14g).
    CompetenceAssessment {
        overall_score: f32,
        overall_dreyfus: String,
        knowledge_area_count: u32,
        cq_answered: u32,
        cq_total: u32,
        recommendation: String,
    },

    // --- Phase 14h: Bootstrap Orchestration -----------------------------------

    /// A full bootstrap orchestration pipeline was completed or checkpointed (Phase 14h).
    BootstrapOrchestration {
        stage: String,
        learning_cycle: u32,
        exploration_rate: f32,
        target_dreyfus: String,
        current_dreyfus: String,
        current_score: f32,
    },

    // --- Continuous Learning (idle task) --------------------------------------

    /// An idle-time continuous learning cycle completed (wires curiosity→bootstrap).
    ContinuousLearning {
        targets_explored: usize,
        resources_found: usize,
        concepts_ingested: usize,
        dreyfus_level: String,
        gap_score_avg: f32,
    },

    // --- Phase 14j: NLU Parsing ---------------------------------------------------

    /// Input was successfully parsed by the NLU pipeline (Phase 14j).
    NluParsed {
        /// Which NLU tier produced the parse (1 = rule parser, 2 = micro-ML, 3 = LLM, 4 = ranker).
        source_tier: u8,
        /// Parse confidence from the producing tier.
        confidence: f32,
        /// If the VSA parse ranker contributed, the exemplar similarity score.
        exemplar_similarity: Option<f32>,
    },

    // --- Phase 25: People & Contacts -----------------------------------------

    /// A contact was resolved or created via identity resolution (Phase 25a).
    ContactResolved {
        contact_id: String,
        alias_count: u32,
    },

    /// Two contacts were merged into one (Phase 25a).
    ContactMerged {
        kept: String,
        discarded: String,
    },

    /// A relationship was recorded between two contacts (Phase 25b).
    RelationshipRecorded {
        from: String,
        to: String,
        kind: String,
    },

    /// A communication style observation was recorded (Phase 25d).
    StyleObserved {
        contact_id: String,
        formality: f32,
        verbosity: f32,
    },

    /// A calendar attendee was linked to a contact (Phase 25e).
    CalendarAttendeeLinked {
        event_id: u64,
        contact_id: String,
    },

    /// A triple was extracted from text by an LLM backend (autonomous extraction).
    LlmTripleExtraction {
        backend: String,
        source_text_hash: u64,
        model: String,
    },

    // --- Phase 15b: Event Calculus -------------------------------------------

    /// A state projection was computed via event calculus (Phase 15b).
    EventCalculusProjection {
        fluent_count: u32,
        event_count: u32,
        interval_secs: u64,
    },

    // --- Phase 15c: Counterfactual Reasoning ----------------------------------

    /// A counterfactual analysis was performed (Phase 15c).
    CounterfactualReasoning {
        actual_action: String,
        hypothetical_action: String,
        divergent_count: u32,
    },

    // --- Phase 16: Predictive Planning ----------------------------------------

    /// An MCTS planning session was executed (Phase 16b).
    MctsPlanning {
        iterations: u32,
        depth: u32,
        expected_value: f32,
    },

    // --- Phase 17: Evidence Theory --------------------------------------------

    /// Evidence was combined from multiple sources (Phase 17a).
    EvidenceCombination {
        source_count: u32,
        conflict_degree: f32,
        belief: f32,
        plausibility: f32,
    },

    // --- Phase 18: Source Reliability ------------------------------------------

    /// A source's reliability was assessed (Phase 18a).
    SourceReliabilityAssessed {
        source_id_raw: u64,
        reliability: String,
        credibility: String,
    },

    // --- Phase 19: Epistemic Logic --------------------------------------------

    /// An epistemic assessment was made (Phase 19a).
    EpistemicAssessment {
        agent_id_raw: u64,
        modality: String,
        proposition_count: u32,
    },

    /// An epistemic update was applied (Phase 19b DEL).
    EpistemicUpdate {
        event_kind: String,
        updated_agent_count: u32,
    },

    /// A theory-of-mind model was built (Phase 19c).
    TheoryOfMind {
        target_agent_raw: u64,
        tom_level: u8,
        prediction_count: u32,
    },

    // --- Phase 20: Active Inference -------------------------------------------

    /// An active inference cycle was run (Phase 20a).
    ActiveInferenceCycle {
        surprise: f32,
        pragmatic: f32,
        epistemic: f32,
        efe: f32,
    },

    // --- Phase 21: Game Theory ------------------------------------------------

    /// A game-theoretic interaction model was built (Phase 21).
    GameTheoreticModel {
        game_type: String,
        equilibria_count: u32,
        recommended_strategy: String,
    },

    // --- Phase 36: Delegation ------------------------------------------------

    /// A concept was identified via neural→VSA bridge probing (Phase 26d).
    NeuralProbe {
        layer: usize,
        similarity: f32,
        source_text: String,
    },

    // --- Phase 23: Affective System ------------------------------------------

    /// An affective appraisal was processed and mood state updated (Phase 23).
    AffectiveAppraisal {
        /// The appraisal trigger kind (e.g. "GoalCompleted", "GoalFailed").
        trigger_kind: String,
        /// Outcome valence of the appraisal (-1.0 to 1.0).
        valence: f32,
        /// Resulting mood valence after EMA update.
        resulting_mood_valence: f32,
    },

    // --- Phase 36: Delegation ------------------------------------------------

    /// A task was delegated to an external worker (Phase 36).
    Delegation {
        target: String,
        task_description: String,
        model_used: Option<String>,
        tokens_consumed: Option<u64>,
        cost_usd: Option<f64>,
        wall_time_ms: u64,
    },

    // --- Phase 22: Akhipedia ------------------------------------------------

    /// A wiki article was created (Phase 22a).
    ArticleCreated {
        title: String,
        author_raw: u64,
        block_count: usize,
    },

    /// A wiki article was updated (Phase 22b).
    ArticleUpdated {
        title: String,
        block_count: usize,
    },

    /// An article was cited by another (Phase 22b).
    ArticleCited {
        from_title: String,
        to_title: String,
    },

    /// An article was published via ActivityPub federation (Phase 22e).
    ArticleFederated {
        title: String,
        activity_type: String,
    },

    /// An opinion was expressed in a blog post (Phase 22f).
    OpinionExpressed {
        title: String,
        stance: String,
        confidence: f32,
    },

    /// A debate contribution was made (Phase 22f).
    DebateContribution {
        topic: String,
        contribution_kind: String,
    },

    // --- Cluster topology ingestion ------------------------------------------

    /// Kubernetes cluster topology was queried and asserted as triples.
    ClusterTopologyIngested {
        nodes: usize,
        storage_classes: usize,
        pods: usize,
        triples_asserted: usize,
    },

    // --- Phase 32: LLM Knowledge Elicitation ---------------------------------

    /// Structured knowledge generated by LLM elicitation (Phase 32).
    ///
    /// Records which model, elicitation pass, prompt fingerprint, and
    /// self-consistency score produced this triple, plus optional neural bridge
    /// coherence (Phase 26d) and whether a web source later corroborated it.
    /// LLM-only triples (not web-corroborated) decay via temporal projection
    /// at –0.05/month (Phase 9k).  If the source model is later found
    /// unreliable, all triples sharing the same `model_id` can be bulk-retracted
    /// via TMS (Phase 9c).
    KnowledgeElicitation {
        /// Identifier for the generative model (e.g. `"t5-knowledge-q6k"`).
        model_id: String,
        /// Elicitation pass that produced this triple
        /// (one of `"taxonomy"`, `"properties"`, `"relationships"`,
        ///  `"procedures"`, `"temporal"`).
        pass: String,
        /// FNV-64 hash of the prompt text for reproducibility.
        prompt_hash: u64,
        /// Fraction of self-consistency rounds in which this triple appeared
        /// (0.0–1.0).
        self_consistency: f32,
        /// Optional neural bridge coherence score from Phase 26d.
        /// `None` when the neural bridge is unavailable.
        neural_coherence: Option<f32>,
        /// Whether a web source (Wikidata / Wikipedia / ConceptNet) later
        /// confirmed this triple.
        web_corroborated: bool,
    },

    /// A VSA-native RL update adjusted the confidence of an action-quality triple.
    ///
    /// Records the tool that was used, the cosine-similarity reward, and the
    /// OODA cycle in which the Act→Observe transition occurred.
    VsaRlUpdate {
        tool: SymbolId,
        reward: f32,
        cycle: u64,
    },

    // --- Import/Export (MCP import_export tools) ------------------------------

    /// Triples were imported from a file (MCP import_triples tool).
    FileImport {
        source_path: String,
        format: String,
    },

    // --- KG Compaction -------------------------------------------------------

    /// Episodic scaffolding triples were removed from the KG (daemon compaction).
    ///
    /// Records how many `episode:N → agent:learned → *` and reverse child-of
    /// links were purged, and the resulting knowledge density.
    KgCompaction {
        /// Number of scaffolding triples actually deleted this run.
        scaffolding_removed: usize,
        /// Total triples in the KG before this compaction run.
        total_before: usize,
        /// Knowledge density (knowledge triples / total triples) after compaction.
        knowledge_density: f32,
    },

    // --- Phase 24: Sensory Grounding -----------------------------------------

    /// Knowledge grounded from direct sensory observation (Phase 24).
    ///
    /// The agent perceived this through image, audio, or structured sensor input.
    Perceived {
        /// Modality: "image", "audio", or "structured".
        modality: String,
        /// Extraction confidence in [0, 1].
        confidence: f32,
        /// Opaque source identifier (e.g. file path, device name, or URI).
        source_id: String,
    },

    // --- Relation enrichment (work item 019e4026) ----------------------------

    /// The relation-enrichment daemon pass injected predicate symbols and
    /// linking triples to improve the entity:relation ratio.
    RelationEnrichment {
        /// New `SymbolKind::Relation` symbols created this run.
        relations_added: usize,
        /// Linking triples added to connect sparse / orphaned entities.
        triples_added: usize,
        /// Entity:relation ratio before the pass.
        ratio_before: f32,
        /// Entity:relation ratio after the pass.
        ratio_after: f32,
    },

    // --- OpenClaw self-import (ADR 076) ---------------------------------------

    /// Knowledge the akh imported from its *own* OpenClaw past life — persona,
    /// memory journals, skills, or the SQLite/Lance stores. Distinct from the
    /// generic `FileImport` so the akh can query exactly what it absorbed from
    /// OpenClaw and dedupe on re-import.
    OpenClawImport {
        /// The OpenClaw source path the knowledge came from.
        source_path: String,
        /// What kind of artefact: `journal` | `skill` | `persona` |
        /// `sqlite_memory` | `vector_export`.
        kind: String,
    },

    // --- ADR 081 slice 4: Daily Librarian projection --------------------------

    /// A typed triple extracted by the daily Librarian pass (ADR 081 slice 4,
    /// `agent::librarian`) against the akh's in-force schema.
    ///
    /// Distinct from `KnowledgeElicitation` (Phase 32): that pass *generates*
    /// speculative concept knowledge from a model's world knowledge, whereas
    /// this pass *extracts* grounded facts from material the akh itself
    /// captured (transcript turns / note bodies) — the Librarian never
    /// invents a source, it only structures one that already exists.
    LibrarianExtraction {
        /// Identifier for the extraction model (an Ollama model tag, e.g.
        /// `"gpt-oss:120b"`, or `"fake"` for the deterministic test double).
        model_id: String,
        /// The captured material this triple was extracted from
        /// (`"turn:{seq}"` for a transcript turn, `"note:{id}"` for a note body).
        source_id: String,
        /// The installed schema type name this triple validated against, when
        /// it landed in the typed core. `None` for a frontier triple (the
        /// predicate did not match any installed type/property — asserted via
        /// `note:frontier/{predicate}` instead, never dropped).
        schema_type: Option<String>,
    },

    // --- ADR 081 slice 8: diff-supersede + multi-akh DS conflict --------------

    /// A previously-projected triple was retracted because re-projecting its
    /// source note no longer asserts it (`schema::project::reproject_note`,
    /// ADR 081 decision 12, work-item 8: "diff-based supersede... retract
    /// only changed triples via the TMS, preserving history"). Recorded
    /// *before* the retraction so the ledger keeps a durable "who/what
    /// superseded this fact" audit trail even after the TMS marks the
    /// triple's object `was_retracted`.
    NoteReprojection {
        /// The note whose re-projection caused the retraction.
        note_id: String,
        /// Why: what changed (frontmatter key removed/edited, body no longer
        /// links, etc.) — free text for the audit trail, not machine-parsed.
        reason: String,
    },

    /// A triple was asserted with an explicit source attribution, so
    /// multiple sources' (dis)agreement about the same `(subject,
    /// predicate)` can later be detected and resolved as a Dempster-Shafer
    /// belief interval instead of last-writer-wins
    /// (`schema::conflict::conflicting_assertions`, ADR 081 decision 12).
    /// `source` names who/what is asserting the fact — an akh identity, a
    /// note entity, a Librarian run marker, etc.
    SourceAttributed {
        subject: SymbolId,
        predicate: SymbolId,
        source: SymbolId,
    },
}

/// Version marker for exact RuleInference support triples encoded in the
/// variant's `antecedents` vector as
/// `[marker, subject, predicate, object, ...]`.
///
/// Older records contain an unversioned symbol walk and remain readable.  The
/// maximum `u64` is reserved for this metadata marker and is never resolved as
/// a registry symbol.
pub const RULE_SUPPORT_TRIPLES_V1_SENTINEL: u64 = u64::MAX;

/// Inner marker for a conclusion-bound RuleInference application encoded as
/// `[V1 marker, V2 marker, conclusion s, p, o, support s, p, o, ...]`.
/// Keeping the V1 marker outermost makes old readers fail closed on the
/// deliberately non-triple-aligned payload instead of misreading metadata as
/// support.
pub const RULE_APPLICATION_V2_SENTINEL: u64 = u64::MAX - 1;

impl DerivationKind {
    /// Stable snake-case names used by audit and wire protocols.
    ///
    /// The array index is the kind's persisted numeric tag.  Keep additions in
    /// lock-step with [`Self::tag`]; unlike a name derived from `Debug`, these
    /// strings are an explicit compatibility contract.
    pub const KIND_NAMES: [&'static str; 108] = [
        "extracted",
        "seed",
        "graph_edge",
        "vsa_recovery",
        "analogy",
        "filler_recovery",
        "reasoned",
        "aggregated",
        "agent_decision",
        "agent_consolidation",
        "rule_inference",
        "fused_inference",
        "gap_identified",
        "schema_discovered",
        "semantic_enrichment",
        "compartment_loaded",
        "shadow_veto",
        "psyche_evolution",
        "wasm_tool_execution",
        "cli_tool_execution",
        "document_ingested",
        "concept_extracted",
        "context_inheritance",
        "context_lifting",
        "predicate_generalization",
        "predicate_inverse",
        "defeasible_override",
        "dispatch_route",
        "argument_verdict",
        "rule_macro_expansion",
        "temporal_decay",
        "contradiction_detected",
        "skolem_witness",
        "skolem_grounding",
        "cwa_query",
        "second_order_instantiation",
        "nart_creation",
        "code_generated",
        "code_refinement",
        "library_learning",
        "autonomous_goal_generation",
        "htn_decomposition",
        "priority_argumentation",
        "project_created",
        "watch_fired",
        "metacognitive_evaluation",
        "resource_assessment",
        "procedural_learning",
        "email_ingested",
        "email_threaded",
        "spam_classification",
        "email_triaged",
        "email_extracted",
        "pim_task_managed",
        "calendar_event_managed",
        "preference_learned",
        "jitir_suggestion",
        "proactive_assistance",
        "causal_schema_learned",
        "ritual_of_awakening",
        "identity_resolved",
        "domain_expansion",
        "prerequisite_discovered",
        "zpd_classification",
        "sleep_consolidation",
        "directed_curiosity",
        "resource_discovery",
        "curriculum_ingestion",
        "competence_assessment",
        "bootstrap_orchestration",
        "continuous_learning",
        "nlu_parsed",
        "contact_resolved",
        "contact_merged",
        "relationship_recorded",
        "style_observed",
        "calendar_attendee_linked",
        "llm_triple_extraction",
        "event_calculus_projection",
        "counterfactual_reasoning",
        "mcts_planning",
        "evidence_combination",
        "source_reliability_assessed",
        "epistemic_assessment",
        "epistemic_update",
        "theory_of_mind",
        "active_inference_cycle",
        "game_theoretic_model",
        "neural_probe",
        "delegation",
        "article_created",
        "article_updated",
        "article_cited",
        "article_federated",
        "opinion_expressed",
        "debate_contribution",
        "cluster_topology_ingested",
        "knowledge_elicitation",
        "vsa_rl_update",
        "file_import",
        "kg_compaction",
        "affective_appraisal",
        "perceived",
        "relation_enrichment",
        "open_claw_import",
        "librarian_extraction",
        "note_reprojection",
        "source_attributed",
    ];

    /// Stable wire name for this derivation kind.
    pub fn kind_name(&self) -> &'static str {
        Self::KIND_NAMES[self.tag() as usize]
    }

    /// Resolve an exact stable wire name to its persisted numeric tag.
    pub fn tag_for_name(name: &str) -> Option<u8> {
        Self::KIND_NAMES
            .iter()
            .position(|candidate| *candidate == name)
            .map(|index| index as u8)
    }

    /// Numeric tag for indexing by kind.
    pub fn tag(&self) -> u8 {
        match self {
            Self::Extracted => 0,
            Self::Seed => 1,
            Self::GraphEdge { .. } => 2,
            Self::VsaRecovery { .. } => 3,
            Self::Analogy { .. } => 4,
            Self::FillerRecovery { .. } => 5,
            Self::Reasoned => 6,
            Self::Aggregated => 7,
            Self::AgentDecision { .. } => 8,
            Self::AgentConsolidation { .. } => 9,
            Self::RuleInference { .. } => 10,
            Self::FusedInference { .. } => 11,
            Self::GapIdentified { .. } => 12,
            Self::SchemaDiscovered { .. } => 13,
            Self::SemanticEnrichment { .. } => 14,
            Self::CompartmentLoaded { .. } => 15,
            Self::ShadowVeto { .. } => 16,
            Self::PsycheEvolution { .. } => 17,
            Self::WasmToolExecution { .. } => 18,
            Self::CliToolExecution { .. } => 19,
            Self::DocumentIngested { .. } => 20,
            Self::ConceptExtracted { .. } => 21,
            Self::ContextInheritance { .. } => 22,
            Self::ContextLifting { .. } => 23,
            Self::PredicateGeneralization { .. } => 24,
            Self::PredicateInverse { .. } => 25,
            Self::DefeasibleOverride { .. } => 26,
            Self::DispatchRoute { .. } => 27,
            Self::ArgumentVerdict { .. } => 28,
            Self::RuleMacroExpansion { .. } => 29,
            Self::TemporalDecay { .. } => 30,
            Self::ContradictionDetected { .. } => 31,
            Self::SkolemWitness { .. } => 32,
            Self::SkolemGrounding { .. } => 33,
            Self::CwaQuery { .. } => 34,
            Self::SecondOrderInstantiation { .. } => 35,
            Self::NartCreation { .. } => 36,
            Self::CodeGenerated { .. } => 37,
            Self::CodeRefinement { .. } => 38,
            Self::LibraryLearning { .. } => 39,
            Self::AutonomousGoalGeneration { .. } => 40,
            Self::HtnDecomposition { .. } => 41,
            Self::PriorityArgumentation { .. } => 42,
            Self::ProjectCreated { .. } => 43,
            Self::WatchFired { .. } => 44,
            Self::MetacognitiveEvaluation { .. } => 45,
            Self::ResourceAssessment { .. } => 46,
            Self::ProceduralLearning { .. } => 47,
            Self::EmailIngested { .. } => 48,
            Self::EmailThreaded { .. } => 49,
            Self::SpamClassification { .. } => 50,
            Self::EmailTriaged { .. } => 51,
            Self::EmailExtracted { .. } => 52,
            Self::PimTaskManaged { .. } => 53,
            Self::CalendarEventManaged { .. } => 54,
            Self::PreferenceLearned { .. } => 55,
            Self::JitirSuggestion { .. } => 56,
            Self::ProactiveAssistance { .. } => 57,
            Self::CausalSchemaLearned { .. } => 58,
            Self::RitualOfAwakening { .. } => 59,
            Self::IdentityResolved { .. } => 60,
            Self::DomainExpansion { .. } => 61,
            Self::PrerequisiteDiscovered { .. } => 62,
            Self::ZpdClassification { .. } => 63,
            Self::SleepConsolidation { .. } => 64,
            Self::DirectedCuriosity { .. } => 65,
            Self::ResourceDiscovery { .. } => 66,
            Self::CurriculumIngestion { .. } => 67,
            Self::CompetenceAssessment { .. } => 68,
            Self::BootstrapOrchestration { .. } => 69,
            Self::ContinuousLearning { .. } => 70,
            Self::NluParsed { .. } => 71,
            Self::ContactResolved { .. } => 72,
            Self::ContactMerged { .. } => 73,
            Self::RelationshipRecorded { .. } => 74,
            Self::StyleObserved { .. } => 75,
            Self::CalendarAttendeeLinked { .. } => 76,
            Self::LlmTripleExtraction { .. } => 77,
            Self::EventCalculusProjection { .. } => 78,
            Self::CounterfactualReasoning { .. } => 79,
            Self::MctsPlanning { .. } => 80,
            Self::EvidenceCombination { .. } => 81,
            Self::SourceReliabilityAssessed { .. } => 82,
            Self::EpistemicAssessment { .. } => 83,
            Self::EpistemicUpdate { .. } => 84,
            Self::TheoryOfMind { .. } => 85,
            Self::ActiveInferenceCycle { .. } => 86,
            Self::GameTheoreticModel { .. } => 87,
            Self::NeuralProbe { .. } => 88,
            Self::Delegation { .. } => 89,
            Self::ArticleCreated { .. } => 90,
            Self::ArticleUpdated { .. } => 91,
            Self::ArticleCited { .. } => 92,
            Self::ArticleFederated { .. } => 93,
            Self::OpinionExpressed { .. } => 94,
            Self::DebateContribution { .. } => 95,
            Self::ClusterTopologyIngested { .. } => 96,
            Self::KnowledgeElicitation { .. } => 97,
            Self::VsaRlUpdate { .. } => 98,
            Self::FileImport { .. } => 99,
            Self::KgCompaction { .. } => 100,
            Self::AffectiveAppraisal { .. } => 101,
            Self::Perceived { .. } => 102,
            Self::RelationEnrichment { .. } => 103,
            Self::OpenClawImport { .. } => 104,
            Self::LibrarianExtraction { .. } => 105,
            Self::NoteReprojection { .. } => 106,
            Self::SourceAttributed { .. } => 107,
        }
    }
}

// ---------------------------------------------------------------------------
// ProvenanceId
// ---------------------------------------------------------------------------

/// Niche-optimized provenance record identifier.
///
/// `Option<ProvenanceId>` is the same size as `ProvenanceId` thanks to `NonZeroU64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ProvenanceId(NonZeroU64);

impl ProvenanceId {
    /// Create from a raw `u64`. Returns `None` if raw is zero.
    pub fn new(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(ProvenanceId)
    }

    /// Get the underlying value.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl std::fmt::Display for ProvenanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "prov:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ProvenanceRecord
// ---------------------------------------------------------------------------

/// A single provenance record linking derived knowledge to its sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Unique ID, assigned on persistence. `None` until stored.
    pub id: Option<ProvenanceId>,
    /// The symbol that was derived / activated.
    pub derived_id: SymbolId,
    /// Source symbols this derivation depends on.
    pub sources: Vec<SymbolId>,
    /// How this was derived.
    pub kind: DerivationKind,
    /// Confidence in the derivation.
    pub confidence: f32,
    /// Inference depth at which the symbol was activated.
    pub depth: usize,
    /// Timestamp (seconds since UNIX epoch).
    pub timestamp: u64,
}

impl ProvenanceRecord {
    /// Create a new record with default values.
    pub fn new(derived_id: SymbolId, kind: DerivationKind) -> Self {
        Self {
            id: None,
            derived_id,
            sources: Vec::new(),
            kind,
            confidence: 1.0,
            depth: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Set the source symbols.
    pub fn with_sources(mut self, sources: Vec<SymbolId>) -> Self {
        self.sources = sources;
        self
    }

    /// Set the confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the depth.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }
}

// ---------------------------------------------------------------------------
// ProvenanceLedger
// ---------------------------------------------------------------------------

/// Persistent provenance ledger backed by redb.
///
/// Stores provenance records with multiple indices for efficient querying
/// by derived symbol, source symbol, and derivation kind.
///
/// The database is wrapped in `Arc<Mutex<Database>>` to serialize concurrent
/// write attempts at the Rust level, preventing OS-level lock pile-up.
pub struct ProvenanceLedger {
    db: Arc<Mutex<Database>>,
    next_id: AtomicU64,
}

impl std::fmt::Debug for ProvenanceLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProvenanceLedger")
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

fn redb_err(msg: impl std::fmt::Display) -> ProvenanceError {
    ProvenanceError::Store(StoreError::Redb {
        message: msg.to_string(),
    })
}

impl ProvenanceLedger {
    /// Open a provenance ledger on an existing database.
    ///
    /// Recovers the next ID counter by scanning the primary table.
    pub fn open(db: Arc<Mutex<Database>>) -> ProvenanceResult<Self> {
        // Ensure tables exist by opening a write txn.
        {
            let db_lock = db.lock().expect("redb mutex poisoned");
            let txn = db_lock.begin_write().map_err(redb_err)?;
            // Opening the tables creates them if absent.
            txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;
            txn.open_multimap_table(DERIVED_INDEX)
                .map_err(redb_err)?;
            txn.open_multimap_table(SOURCE_INDEX)
                .map_err(redb_err)?;
            txn.open_multimap_table(KIND_INDEX)
                .map_err(redb_err)?;
            txn.commit().map_err(redb_err)?;
        }

        // Recover max ID from the primary table.
        let max_id = {
            let db_lock = db.lock().expect("redb mutex poisoned");
            let txn = db_lock.begin_read().map_err(redb_err)?;
            drop(db_lock); // release lock — read txn is independent
            let table = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;
            let mut max = 0u64;
            let iter = table.iter().map_err(redb_err)?;
            for entry in iter {
                let (key_guard, _val_guard) = entry.map_err(redb_err)?;
                let key: u64 = key_guard.value();
                if key > max {
                    max = key;
                }
            }
            max
        };

        Ok(Self {
            db,
            next_id: AtomicU64::new(max_id + 1),
        })
    }

    /// Store a single record, assigning it a new ID.
    pub fn store(&self, record: &mut ProvenanceRecord) -> ProvenanceResult<ProvenanceId> {
        let raw_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let prov_id =
            ProvenanceId::new(raw_id).expect("provenance ID counter should never reach zero");
        record.id = Some(prov_id);

        let encoded = bincode::serialize(&record).map_err(|e| {
            ProvenanceError::Store(StoreError::Serialization {
                message: format!("provenance record serialize: {e}"),
            })
        })?;

        let db_lock = self.db.lock().expect("redb mutex poisoned");
        let txn = db_lock.begin_write().map_err(redb_err)?;
        {
            let mut table = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;
            table
                .insert(raw_id, encoded.as_slice())
                .map_err(redb_err)?;
        }
        {
            let mut idx = txn
                .open_multimap_table(DERIVED_INDEX)
                .map_err(redb_err)?;
            idx.insert(record.derived_id.get(), raw_id)
                .map_err(redb_err)?;
        }
        {
            let mut idx = txn
                .open_multimap_table(SOURCE_INDEX)
                .map_err(redb_err)?;
            for src in &record.sources {
                idx.insert(src.get(), raw_id).map_err(redb_err)?;
            }
        }
        {
            let mut idx = txn
                .open_multimap_table(KIND_INDEX)
                .map_err(redb_err)?;
            idx.insert(record.kind.tag(), raw_id)
                .map_err(redb_err)?;
        }
        txn.commit().map_err(redb_err)?;

        Ok(prov_id)
    }

    /// Store a batch of records in a single transaction.
    pub fn store_batch(
        &self,
        records: &mut [ProvenanceRecord],
    ) -> ProvenanceResult<Vec<ProvenanceId>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let mut ids = Vec::with_capacity(records.len());
        let mut encoded_batch = Vec::with_capacity(records.len());

        // Assign IDs and serialize before opening the transaction.
        for record in records.iter_mut() {
            let raw_id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let prov_id =
                ProvenanceId::new(raw_id).expect("provenance ID counter should never reach zero");
            record.id = Some(prov_id);
            ids.push(prov_id);

            let encoded = bincode::serialize(&record).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("provenance record serialize: {e}"),
                })
            })?;
            encoded_batch.push((raw_id, encoded));
        }

        let db_lock = self.db.lock().expect("redb mutex poisoned");
        let txn = db_lock.begin_write().map_err(redb_err)?;
        {
            let mut table = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;
            for (raw_id, encoded) in &encoded_batch {
                table
                    .insert(*raw_id, encoded.as_slice())
                    .map_err(redb_err)?;
            }
        }
        {
            let mut derived_idx = txn
                .open_multimap_table(DERIVED_INDEX)
                .map_err(redb_err)?;
            let mut source_idx = txn
                .open_multimap_table(SOURCE_INDEX)
                .map_err(redb_err)?;
            let mut kind_idx = txn
                .open_multimap_table(KIND_INDEX)
                .map_err(redb_err)?;

            for (i, (raw_id, _)) in encoded_batch.iter().enumerate() {
                let record = &records[i];
                derived_idx
                    .insert(record.derived_id.get(), *raw_id)
                    .map_err(redb_err)?;
                for src in &record.sources {
                    source_idx
                        .insert(src.get(), *raw_id)
                        .map_err(redb_err)?;
                }
                kind_idx
                    .insert(record.kind.tag(), *raw_id)
                    .map_err(redb_err)?;
            }
        }
        txn.commit().map_err(redb_err)?;

        Ok(ids)
    }

    /// Get a provenance record by its ID.
    pub fn get(&self, id: ProvenanceId) -> ProvenanceResult<ProvenanceRecord> {
        let db_lock = self.db.lock().expect("redb mutex poisoned");
        let txn = db_lock.begin_read().map_err(redb_err)?;
        drop(db_lock); // release lock — read txn is independent
        let table = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;
        let guard = table
            .get(id.get())
            .map_err(redb_err)?
            .ok_or(ProvenanceError::NotFound { id: id.get() })?;
        let record: ProvenanceRecord = bincode::deserialize(guard.value()).map_err(|e| {
            ProvenanceError::Store(StoreError::Serialization {
                message: format!("provenance record deserialize: {e}"),
            })
        })?;
        Ok(record)
    }

    /// Find all provenance records for a given derived symbol.
    pub fn by_derived(&self, symbol: SymbolId) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        self.records_from_multimap_index(&DERIVED_INDEX, symbol.get())
    }

    /// Find all provenance records that depend on a given source symbol.
    ///
    /// This answers "what was derived from X?" — the reverse dependency lookup.
    pub fn by_source(&self, symbol: SymbolId) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        self.records_from_multimap_index(&SOURCE_INDEX, symbol.get())
    }

    /// Find all provenance records of a given derivation kind.
    pub fn by_kind(&self, kind: &DerivationKind) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        self.records_from_kind_index(kind.tag())
    }

    /// Total number of provenance records.
    pub fn len(&self) -> ProvenanceResult<usize> {
        let db_lock = self.db.lock().expect("redb mutex poisoned");
        let txn = db_lock.begin_read().map_err(redb_err)?;
        drop(db_lock); // release lock — read txn is independent
        let table = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;
        Ok(table.len().map_err(redb_err)? as usize)
    }

    /// Whether the ledger has no records.
    pub fn is_empty(&self) -> ProvenanceResult<bool> {
        self.len().map(|n| n == 0)
    }

    /// Retrieve every provenance record in one read transaction.
    ///
    /// This is intentionally a ledger scan rather than one `get` transaction
    /// per knowledge-graph triple.  Audit surfaces such as
    /// `_akh/provenance/filter` need to join many triples to their provenance;
    /// keeping the scan inside one redb snapshot makes that join both coherent
    /// and bounded in database round trips.
    pub fn all_records(&self) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        let db_lock = self.db.lock().expect("redb mutex poisoned");
        let txn = db_lock.begin_read().map_err(redb_err)?;
        drop(db_lock); // release lock — read txn is independent
        let table = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;

        let mut records = Vec::with_capacity(table.len().map_err(redb_err)? as usize);
        for entry in table.iter().map_err(redb_err)? {
            let (_, encoded) = entry.map_err(redb_err)?;
            let record = bincode::deserialize(encoded.value()).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("provenance record deserialize: {e}"),
                })
            })?;
            records.push(record);
        }
        Ok(records)
    }

    /// Replace the complete ledger with records carrying their original IDs.
    ///
    /// This is the cold-wake counterpart to [`Self::all_records`].  The
    /// primary table and every secondary index are rebuilt in one redb
    /// transaction, so restored graph edges can never observe a partially
    /// rebuilt ledger.  Callers must invoke this before publishing the woken
    /// engine to concurrent writers.
    pub fn replace_all(&self, records: &[ProvenanceRecord]) -> ProvenanceResult<()> {
        let mut encoded = Vec::with_capacity(records.len());
        let mut seen = std::collections::HashSet::with_capacity(records.len());
        let mut max_id = 0u64;

        for record in records {
            let raw_id = record.id.ok_or_else(|| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: "restored provenance record is missing its ID".into(),
                })
            })?.get();
            if !seen.insert(raw_id) {
                return Err(ProvenanceError::Store(StoreError::Serialization {
                    message: format!("duplicate restored provenance ID {raw_id}"),
                }));
            }
            max_id = max_id.max(raw_id);
            let bytes = bincode::serialize(record).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("restored provenance record serialize: {e}"),
                })
            })?;
            encoded.push((raw_id, bytes));
        }

        let next_id = max_id.checked_add(1).ok_or_else(|| {
            ProvenanceError::Store(StoreError::Serialization {
                message: "restored provenance ID space is exhausted".into(),
            })
        })?;

        let db_lock = self.db.lock().expect("redb mutex poisoned");
        let txn = db_lock.begin_write().map_err(redb_err)?;
        txn.delete_table(PROVENANCE_TABLE).map_err(redb_err)?;
        txn.delete_multimap_table(DERIVED_INDEX).map_err(redb_err)?;
        txn.delete_multimap_table(SOURCE_INDEX).map_err(redb_err)?;
        txn.delete_multimap_table(KIND_INDEX).map_err(redb_err)?;

        {
            let mut primary = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;
            for (raw_id, bytes) in &encoded {
                primary
                    .insert(*raw_id, bytes.as_slice())
                    .map_err(redb_err)?;
            }
        }
        {
            let mut derived_idx = txn
                .open_multimap_table(DERIVED_INDEX)
                .map_err(redb_err)?;
            let mut source_idx = txn
                .open_multimap_table(SOURCE_INDEX)
                .map_err(redb_err)?;
            let mut kind_idx = txn
                .open_multimap_table(KIND_INDEX)
                .map_err(redb_err)?;
            for (record, (raw_id, _)) in records.iter().zip(&encoded) {
                derived_idx
                    .insert(record.derived_id.get(), *raw_id)
                    .map_err(redb_err)?;
                for source in &record.sources {
                    source_idx
                        .insert(source.get(), *raw_id)
                        .map_err(redb_err)?;
                }
                kind_idx
                    .insert(record.kind.tag(), *raw_id)
                    .map_err(redb_err)?;
            }
        }
        txn.commit().map_err(redb_err)?;

        // `replace_all` is a cold-wake operation performed before the engine is
        // published, so restoring the exact allocator successor is race-free.
        self.next_id.store(next_id, Ordering::SeqCst);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Retrieve records given a u64-keyed multimap index.
    fn records_from_multimap_index(
        &self,
        table_def: &MultimapTableDefinition<u64, u64>,
        key: u64,
    ) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        let db_lock = self.db.lock().expect("redb mutex poisoned");
        let txn = db_lock.begin_read().map_err(redb_err)?;
        drop(db_lock); // release lock — read txn is independent
        let idx = txn
            .open_multimap_table(*table_def)
            .map_err(redb_err)?;
        let primary = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;

        let values = idx.get(key).map_err(redb_err)?;

        let mut records = Vec::new();
        for entry in values {
            let raw_id = entry.map_err(redb_err)?.value();
            let guard = primary
                .get(raw_id)
                .map_err(redb_err)?
                .ok_or(ProvenanceError::NotFound { id: raw_id })?;
            let record: ProvenanceRecord = bincode::deserialize(guard.value()).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("provenance record deserialize: {e}"),
                })
            })?;
            records.push(record);
        }
        Ok(records)
    }

    /// Retrieve records given a u8-keyed multimap index (kind).
    fn records_from_kind_index(&self, tag: u8) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        let db_lock = self.db.lock().expect("redb mutex poisoned");
        let txn = db_lock.begin_read().map_err(redb_err)?;
        drop(db_lock); // release lock — read txn is independent
        let idx = txn
            .open_multimap_table(KIND_INDEX)
            .map_err(redb_err)?;
        let primary = txn.open_table(PROVENANCE_TABLE).map_err(redb_err)?;

        let values = idx.get(tag).map_err(redb_err)?;

        let mut records = Vec::new();
        for entry in values {
            let raw_id = entry.map_err(redb_err)?.value();
            let guard = primary
                .get(raw_id)
                .map_err(redb_err)?
                .ok_or(ProvenanceError::NotFound { id: raw_id })?;
            let record: ProvenanceRecord = bincode::deserialize(guard.value()).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("provenance record deserialize: {e}"),
                })
            })?;
            records.push(record);
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Arc<Mutex<Database>> {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.redb");
        // Leak the TempDir to keep the directory alive for the test.
        std::mem::forget(dir);
        Arc::new(Mutex::new(Database::create(path).unwrap()))
    }

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    #[test]
    fn store_and_retrieve() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut record = ProvenanceRecord::new(sym(1), DerivationKind::Seed)
            .with_confidence(1.0)
            .with_depth(0);

        let id = ledger.store(&mut record).unwrap();
        assert!(record.id.is_some());

        let retrieved = ledger.get(id).unwrap();
        assert_eq!(retrieved.derived_id, sym(1));
        assert_eq!(retrieved.kind, DerivationKind::Seed);
        assert_eq!(retrieved.confidence, 1.0);
        assert_eq!(retrieved.depth, 0);
    }

    #[test]
    fn auto_incrementing_ids() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut r1 = ProvenanceRecord::new(sym(1), DerivationKind::Seed);
        let mut r2 = ProvenanceRecord::new(sym(2), DerivationKind::Extracted);
        let mut r3 = ProvenanceRecord::new(sym(3), DerivationKind::Reasoned);

        let id1 = ledger.store(&mut r1).unwrap();
        let id2 = ledger.store(&mut r2).unwrap();
        let id3 = ledger.store(&mut r3).unwrap();

        assert_eq!(id1.get(), 1);
        assert_eq!(id2.get(), 2);
        assert_eq!(id3.get(), 3);
    }

    #[test]
    fn replace_all_restores_ids_indices_and_allocator() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut stale = ProvenanceRecord::new(sym(1), DerivationKind::Extracted);
        let stale_id = ledger.store(&mut stale).unwrap();

        let source = sym(9);
        let mut first = ProvenanceRecord::new(sym(70), DerivationKind::Seed)
            .with_sources(vec![source]);
        first.id = ProvenanceId::new(7);
        let mut second = ProvenanceRecord::new(sym(90), DerivationKind::Reasoned)
            .with_sources(vec![source]);
        second.id = ProvenanceId::new(9);

        ledger.replace_all(&[first, second]).unwrap();
        assert!(matches!(ledger.get(stale_id), Err(ProvenanceError::NotFound { .. })));
        assert_eq!(ledger.len().unwrap(), 2);
        assert_eq!(ledger.by_source(source).unwrap().len(), 2);
        assert_eq!(ledger.by_kind(&DerivationKind::Seed).unwrap().len(), 1);

        let mut next = ProvenanceRecord::new(sym(100), DerivationKind::Aggregated);
        assert_eq!(ledger.store(&mut next).unwrap().get(), 10);
    }

    #[test]
    fn replace_all_rejects_invalid_manifests_without_erasing_live_data() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();
        let mut live = ProvenanceRecord::new(sym(5), DerivationKind::Seed);
        let live_id = ledger.store(&mut live).unwrap();

        let mut duplicate_a = ProvenanceRecord::new(sym(7), DerivationKind::Reasoned);
        duplicate_a.id = ProvenanceId::new(8);
        let mut duplicate_b = ProvenanceRecord::new(sym(9), DerivationKind::Aggregated);
        duplicate_b.id = ProvenanceId::new(8);
        assert!(ledger.replace_all(&[duplicate_a, duplicate_b]).is_err());
        assert_eq!(ledger.get(live_id).unwrap().derived_id, sym(5));

        let missing_id = ProvenanceRecord::new(sym(11), DerivationKind::Extracted);
        assert!(ledger.replace_all(&[missing_id]).is_err());
        assert_eq!(ledger.get(live_id).unwrap().derived_id, sym(5));
    }

    #[test]
    fn rule_support_payloads_roundtrip_in_legacy_and_versioned_forms() {
        let marker = sym(RULE_SUPPORT_TRIPLES_V1_SENTINEL);
        for antecedents in [vec![sym(1), sym(2)], vec![marker, sym(1), sym(3), sym(2)]] {
            let record = ProvenanceRecord::new(
                sym(9),
                DerivationKind::RuleInference {
                    rule_name: "compatibility".into(),
                    antecedents: antecedents.clone(),
                },
            );
            let bytes = bincode::serialize(&record).unwrap();
            let decoded: ProvenanceRecord = bincode::deserialize(&bytes).unwrap();
            assert!(matches!(
                decoded.kind,
                DerivationKind::RuleInference { antecedents: restored, .. }
                    if restored == antecedents
            ));
        }
    }

    #[test]
    fn by_derived_index() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let target = sym(10);
        let mut r1 = ProvenanceRecord::new(target, DerivationKind::Seed);
        let mut r2 = ProvenanceRecord::new(target, DerivationKind::Reasoned);
        let mut r3 = ProvenanceRecord::new(sym(20), DerivationKind::Extracted);

        ledger.store(&mut r1).unwrap();
        ledger.store(&mut r2).unwrap();
        ledger.store(&mut r3).unwrap();

        let results = ledger.by_derived(target).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.derived_id == target));
    }

    #[test]
    fn by_source_reverse_lookup() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let source = sym(5);
        let mut r1 = ProvenanceRecord::new(
            sym(10),
            DerivationKind::GraphEdge {
                from: source,
                predicate: sym(100),
            },
        )
        .with_sources(vec![source]);
        let mut r2 = ProvenanceRecord::new(sym(20), DerivationKind::Reasoned)
            .with_sources(vec![source, sym(6)]);

        ledger.store(&mut r1).unwrap();
        ledger.store(&mut r2).unwrap();

        let dependents = ledger.by_source(source).unwrap();
        assert_eq!(dependents.len(), 2);
    }

    #[test]
    fn by_kind_index() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut r1 = ProvenanceRecord::new(sym(1), DerivationKind::Seed);
        let mut r2 = ProvenanceRecord::new(sym(2), DerivationKind::Seed);
        let mut r3 = ProvenanceRecord::new(sym(3), DerivationKind::Extracted);
        let mut r4 = ProvenanceRecord::new(sym(4), DerivationKind::Reasoned);

        ledger.store(&mut r1).unwrap();
        ledger.store(&mut r2).unwrap();
        ledger.store(&mut r3).unwrap();
        ledger.store(&mut r4).unwrap();

        let seeds = ledger.by_kind(&DerivationKind::Seed).unwrap();
        assert_eq!(seeds.len(), 2);
        let extracted = ledger.by_kind(&DerivationKind::Extracted).unwrap();
        assert_eq!(extracted.len(), 1);
        let reasoned = ledger.by_kind(&DerivationKind::Reasoned).unwrap();
        assert_eq!(reasoned.len(), 1);
    }

    #[test]
    fn store_batch_single_transaction() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut records: Vec<ProvenanceRecord> = (1..=5)
            .map(|i| ProvenanceRecord::new(sym(i), DerivationKind::Seed))
            .collect();

        let ids = ledger.store_batch(&mut records).unwrap();
        assert_eq!(ids.len(), 5);
        assert_eq!(ledger.len().unwrap(), 5);

        // All records should have IDs assigned.
        for record in &records {
            assert!(record.id.is_some());
        }
    }

    #[test]
    fn persistence_across_reopens() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("persist.redb");
        let db = Arc::new(Mutex::new(Database::create(&path).unwrap()));

        let stored_id = {
            let ledger = ProvenanceLedger::open(Arc::clone(&db)).unwrap();
            let mut record =
                ProvenanceRecord::new(sym(42), DerivationKind::Extracted).with_confidence(0.95);
            ledger.store(&mut record).unwrap()
        };

        // Drop and reopen.
        drop(db);
        let db2 = Arc::new(Mutex::new(Database::create(&path).unwrap()));
        let ledger2 = ProvenanceLedger::open(db2).unwrap();

        let retrieved = ledger2.get(stored_id).unwrap();
        assert_eq!(retrieved.derived_id, sym(42));
        assert_eq!(retrieved.confidence, 0.95);

        // Next ID should continue past the stored one.
        let mut r2 = ProvenanceRecord::new(sym(43), DerivationKind::Seed);
        let id2 = ledger2.store(&mut r2).unwrap();
        assert!(id2.get() > stored_id.get());

        let mut all = ledger2.all_records().unwrap();
        all.sort_by_key(|record| record.id);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].id, Some(stored_id));
        assert_eq!(all[1].id, Some(id2));
        assert_eq!(all[0].kind.kind_name(), "extracted");
        assert_eq!(all[1].kind.kind_name(), "seed");
    }

    #[test]
    fn provenance_id_niche_optimization() {
        assert_eq!(
            std::mem::size_of::<Option<ProvenanceId>>(),
            std::mem::size_of::<ProvenanceId>()
        );
    }
}
