//! Round-trip serialisation of representative entities, exercising the
//! `Serialize`/`Deserialize` impls SeaORM derives — JSON columns, byte vectors,
//! optionals, and composite-keyed cognitive tables — without a live database.

use akhomed_store::entity::{
    akh, akh_state, competence_tool, drive, effort_case, fact, failure_case, goal, goal_dependency,
    goal_hierarchy, learned_method_step, mutation_log, provenance, provenance_input, symbol,
    workspace, workspace_seed_pack,
};
use chrono::{DateTime, Utc};
use serde_json::json;
use uuid::Uuid;

fn ts() -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).expect("valid timestamp")
}

fn roundtrip<T>(model: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let encoded = serde_json::to_string(model).expect("serialize");
    let decoded: T = serde_json::from_str(&encoded).expect("deserialize");
    assert_eq!(model, &decoded, "round-trip mismatch for {encoded}");
}

#[test]
fn akh_and_state_roundtrip() {
    roundtrip(&akh::Model {
        id: Uuid::nil(),
        name: "tecton".into(),
        home_host: Some("m2-cellar".into()),
        awakened_at: ts(),
        last_seen: None,
    });
    roundtrip(&akh_state::Model {
        akh_id: Uuid::from_u128(1),
        watermark: 42,
        allocator_next: 1000,
        cycle_count: 7,
        session_start_cycle: 3,
        sleep_phase: "Wake".into(),
        last_sleep_cycle: 5,
        psyche: Some(json!({ "awakened": true })),
        extra: Some(json!({ "method_stats": [] })),
        harvest_watermark: 12,
        updated_at: ts(),
    });
}

#[test]
fn symbol_and_fact_roundtrip() {
    roundtrip(&symbol::Model {
        akh_id: Uuid::from_u128(1),
        id: 100,
        kind: "Relation".into(),
        glyph_codepoint: None,
        label: "is-a".into(),
    });
    roundtrip(&fact::Model {
        workspace_id: Uuid::from_u128(2),
        subject: 100,
        predicate: 200,
        object: 300,
        akh_id: Uuid::from_u128(1),
        confidence: 0.75,
        asserted_at: ts(),
        retracted_at: None,
        provenance_id: None,
        compartment: None,
    });
}

#[test]
fn json_columns_roundtrip() {
    // mutation_log payload + provenance derivation are the JSON columns we keep.
    roundtrip(&mutation_log::Model {
        seq: 1,
        akh_id: Uuid::from_u128(1),
        workspace_id: Uuid::from_u128(2),
        kind: "assert".into(),
        payload: json!({ "subject": 1, "predicate": 2, "object": 3 }),
        applied_at: ts(),
    });
    roundtrip(&provenance::Model {
        provenance_id: 1,
        akh_id: Uuid::from_u128(1),
        workspace_id: Uuid::from_u128(2),
        subject: 5,
        derivation: json!({ "kind": "AgentConsolidation" }),
        confidence: 0.9,
        created_at: ts(),
    });
    roundtrip(&workspace::Model {
        id: Uuid::from_u128(3),
        name: "tecton".into(),
        dimension: 10000,
        encoding: "bipolar".into(),
        language: "auto".into(),
        created_at: ts(),
    });
}

#[test]
fn normalised_edge_and_child_tables_roundtrip() {
    // The former JSON list-columns are now proper edge/child rows.
    roundtrip(&goal_hierarchy::Model {
        akh_id: Uuid::from_u128(1),
        parent_id: 7,
        child_id: 8,
        position: 0,
    });
    roundtrip(&goal_dependency::Model {
        akh_id: Uuid::from_u128(1),
        goal_id: 7,
        blocking_goal_id: 4,
    });
    roundtrip(&learned_method_step::Model {
        akh_id: Uuid::from_u128(1),
        method_id: 42,
        step_index: 1,
        tool: "kg_query".into(),
        arg_pattern: "find {Entity}".into(),
        expected_outcome: "kg_query_result".into(),
    });
    roundtrip(&provenance_input::Model {
        provenance_id: 1,
        position: 0,
        input_symbol_id: 9_000_000_000_i64,
    });
    roundtrip(&workspace_seed_pack::Model {
        workspace_id: Uuid::from_u128(3),
        position: 0,
        pack_name: "ci-runner".into(),
    });
}

#[test]
fn cognitive_tables_roundtrip() {
    roundtrip(&drive::Model {
        akh_id: Uuid::from_u128(1),
        kind: "curiosity".into(),
        strength: 0.6,
        last_computed: 10,
        last_triple_count: 500,
        last_symbol_count: 200,
        threshold: 0.6,
    });
    roundtrip(&goal::Model {
        akh_id: Uuid::from_u128(1),
        id: 7,
        description: "learn pyramids".into(),
        status: "Active".into(),
        priority: 5,
        success_criteria: "knows construction".into(),
        parent: None,
        created_at: 100,
        cycles_worked: 3,
        last_progress_cycle: 120,
        source: Some("DriveCuriosity".into()),
        detail: Some(json!({ "estimated_effort": { "cycles": 10 } })),
    });
    roundtrip(&failure_case::Model {
        akh_id: Uuid::from_u128(1),
        seq: 0,
        goal_description: "g".into(),
        tool_name: "code_gen".into(),
        error_message: "boom".into(),
        cycle: 9,
        resolved: false,
        resolution_tool: None,
        vector: vec![0xDE, 0xAD],
    });
    roundtrip(&effort_case::Model {
        akh_id: Uuid::from_u128(1),
        seq: 0,
        description: "d".into(),
        goal_vector: vec![1, 2, 3],
        cycles_consumed: 4,
        succeeded: true,
        tool_usage: json!({ "code_gen": 2, "search": 1 }),
        initial_coverage: 3,
    });
    roundtrip(&competence_tool::Model {
        akh_id: Uuid::from_u128(1),
        tool: "code_gen".into(),
        successes: 8,
        attempts: 10,
    });
}
