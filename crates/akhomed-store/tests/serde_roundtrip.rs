//! 38a acceptance #3: round-trip serialisation of every entity, with emphasis
//! on the JSONB columns (psyche / payload / derivation) and the Postgres array
//! columns (`provenance.inputs`, `workspace.seed_packs`). These exercise the
//! `Serialize`/`Deserialize` impls SeaORM derives onto each `Model`, without
//! needing a live database.

use akhomed_store::entity::{
    active_microtheory, akh, episode, fact, membership, mutation_log, provenance, psyche,
    snapshot, workspace,
};
use chrono::{DateTime, Utc};
use serde_json::json;
use uuid::Uuid;

/// A fixed, sub-second-free timestamp so RFC3339 round-trips are exact.
fn ts() -> DateTime<Utc> {
    DateTime::from_timestamp(1_700_000_000, 0).expect("valid timestamp")
}

fn roundtrip<T>(model: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    let encoded = serde_json::to_string(model).expect("serialize");
    let decoded: T = serde_json::from_str(&encoded).expect("deserialize");
    assert_eq!(model, &decoded, "round-trip mismatch for {encoded}");
    decoded
}

#[test]
fn akh_roundtrips() {
    roundtrip(&akh::Model {
        id: Uuid::nil(),
        name: "tecton".into(),
        home_host: Some("m2-cellar".into()),
        awakened_at: ts(),
        last_seen: None,
    });
}

#[test]
fn fact_roundtrips_with_optionals() {
    roundtrip(&fact::Model {
        fact_id: 42,
        akh_id: Uuid::from_u128(1),
        workspace_id: Uuid::from_u128(2),
        subject: 100,
        predicate: 200,
        object: 300,
        confidence: 0.75,
        asserted_at: ts(),
        retracted_at: Some(ts()),
        provenance_id: Some(7),
        compartment: Some("egyptian-math".into()),
    });
    // ...and with the nullable columns absent.
    roundtrip(&fact::Model {
        fact_id: 43,
        akh_id: Uuid::from_u128(1),
        workspace_id: Uuid::from_u128(2),
        subject: 1,
        predicate: 2,
        object: 3,
        confidence: 1.0,
        asserted_at: ts(),
        retracted_at: None,
        provenance_id: None,
        compartment: None,
    });
}

#[test]
fn mutation_log_roundtrips_jsonb_payload() {
    roundtrip(&mutation_log::Model {
        seq: 1,
        akh_id: Uuid::from_u128(1),
        workspace_id: Uuid::from_u128(2),
        kind: "assert".into(),
        payload: json!({ "subject": 1, "predicate": 2, "object": 3, "confidence": 1.0 }),
        applied_at: ts(),
    });
}

#[test]
fn snapshot_roundtrips_bytea_and_optional_jsonb() {
    roundtrip(&snapshot::Model {
        snapshot_id: 1,
        akh_id: Uuid::from_u128(1),
        created_at: ts(),
        watermark: 12345,
        working_mem: vec![0xDE, 0xAD, 0xBE, 0xEF],
        episodes: vec![1, 2, 3],
        state_blob: vec![],
        psyche: Some(json!({ "awakened": true })),
    });
    roundtrip(&snapshot::Model {
        snapshot_id: 2,
        akh_id: Uuid::from_u128(1),
        created_at: ts(),
        watermark: 0,
        working_mem: vec![],
        episodes: vec![],
        state_blob: vec![],
        psyche: None,
    });
}

#[test]
fn psyche_roundtrips_four_jsonb_columns() {
    roundtrip(&psyche::Model {
        akh_id: Uuid::from_u128(9),
        persona: json!({ "openness": 0.8 }),
        shadow: json!({ "repressed": ["doubt"] }),
        archetypes: json!([{ "name": "sage", "weight": 0.6 }]),
        self_integration: json!({ "level": 3 }),
        awakened: true,
        updated_at: ts(),
    });
}

#[test]
fn provenance_roundtrips_bigint_array() {
    roundtrip(&provenance::Model {
        provenance_id: 1,
        akh_id: Uuid::from_u128(1),
        workspace_id: Uuid::from_u128(2),
        subject: 500,
        derivation: json!({ "kind": "AgentConsolidation" }),
        inputs: vec![1, 2, 3, 9_000_000_000],
        confidence: 0.9,
        created_at: ts(),
    });
}

#[test]
fn episode_roundtrips() {
    roundtrip(&episode::Model {
        episode_id: 1,
        akh_id: Uuid::from_u128(1),
        workspace_id: Uuid::from_u128(2),
        started_at: ts(),
        ended_at: Some(ts()),
        payload: json!({ "summary": "learned about pyramids" }),
    });
}

#[test]
fn workspace_roundtrips_text_array() {
    roundtrip(&workspace::Model {
        id: Uuid::from_u128(3),
        name: "tecton".into(),
        dimension: 10000,
        encoding: "bipolar".into(),
        language: "auto".into(),
        seed_packs: vec!["core".into(), "ci-runner".into()],
        created_at: ts(),
    });
    // Empty array case.
    roundtrip(&workspace::Model {
        id: Uuid::from_u128(4),
        name: "scratch".into(),
        dimension: 10000,
        encoding: "bipolar".into(),
        language: "auto".into(),
        seed_packs: vec![],
        created_at: ts(),
    });
}

#[test]
fn membership_and_active_microtheory_roundtrip() {
    roundtrip(&membership::Model {
        workspace_id: Uuid::from_u128(3),
        akh_id: Uuid::from_u128(1),
        joined_at: ts(),
        role: "owner".into(),
    });
    roundtrip(&active_microtheory::Model {
        workspace_id: Uuid::from_u128(3),
        microtheory_id: "egyptian-math".into(),
        loaded_by: Uuid::from_u128(1),
        loaded_at: ts(),
    });
}
