//! Bounded annex of the accepted-turn store. Never a second cognitive Engine.
use crate::{Store, Transaction, key};
use akh_session_types::{AcceptedTurn, Fault, Result, neural_spans::*, validate_key};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Record {
    turn_id: String,
    capture: SpanCapture,
    text: String,
}

pub fn validate(turn: &AcceptedTurn) -> Result<()> {
    validate_key(&turn.akh)?;
    validate_key(&turn.session_id)?;
    validate_key(&turn.turn_id)?;
    if turn.span_captures.len() > MAX_CAPTURES {
        return Err(Fault::invalid("at most four spans per accepted turn"));
    }
    for capture in &turn.span_captures {
        capture.text(&turn.input.text, &turn.output)?;
    }
    Ok(())
}

/// Caller commits this with the acceptance receipt, after its journal is durable.
/// The caller owns capture policy and durable operation-ID replay protection.
pub fn append(tx: &mut Transaction<'_>, principal: &str, turn: &AcceptedTurn) -> Result<()> {
    validate_key(principal)?;
    validate(turn)?;
    if turn.span_captures.is_empty() {
        return Ok(());
    }
    let bucket = key("neural_spans_v1", &[principal, &turn.akh, &turn.session_id]);
    let mut records: Vec<Record> = tx.get(&bucket)?.unwrap_or_default();
    let incoming: Vec<Record> = turn
        .span_captures
        .iter()
        .map(|capture| {
            Ok(Record {
                turn_id: turn.turn_id.clone(),
                capture: capture.clone(),
                text: capture.text(&turn.input.text, &turn.output)?.into(),
            })
        })
        .collect::<Result<_>>()?;
    let previous: Vec<_> = records
        .iter()
        .filter(|record| record.turn_id == turn.turn_id)
        .collect();
    if !previous.is_empty() {
        if previous == incoming.iter().collect::<Vec<_>>() {
            return Ok(());
        }
        return Err(Fault::conflict("accepted span turn identity reused"));
    }
    records.extend(incoming);
    let excess = records.len().saturating_sub(MAX_SESSION_SPANS);
    records.drain(..excess);
    tx.put(&bucket, &records)
}

pub fn query(store: &Store, principal: &str, request: &SpanQuery) -> Result<Vec<SpanHit>> {
    validate_key(principal)?;
    request.validate()?;
    let records: Vec<Record> = store
        .get(&key(
            "neural_spans_v1",
            &[principal, &request.akh, &request.session_id],
        ))?
        .unwrap_or_default();
    if records.len() > MAX_SESSION_SPANS {
        return Err(Fault::conflict("span bucket exceeds admission bound"));
    }
    let mut hits = Vec::new();
    for record in records
        .into_iter()
        .filter(|record| record.capture.space == request.space)
    {
        if record.capture.vector.len() != VECTOR_BYTES {
            return Err(Fault::conflict("invalid stored span vector"));
        }
        let distance = record
            .capture
            .vector
            .iter()
            .zip(&request.vector)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        hits.push(SpanHit {
            turn_id: record.turn_id,
            source: record.capture.source,
            byte_start: record.capture.byte_start,
            byte_end: record.capture.byte_end,
            text: record.text,
            hamming_distance: distance,
        });
    }
    // Stable ordering also preserves capture order for ties within one turn.
    hits.sort_by(|a, b| {
        a.hamming_distance
            .cmp(&b.hamming_distance)
            .then_with(|| a.turn_id.cmp(&b.turn_id))
    });
    hits.truncate(request.limit);
    Ok(hits)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn turn(id: &str, byte: u8) -> AcceptedTurn {
        serde_json::from_value(serde_json::json!({
            "request_id":id,"akh":"ptah","session_id":"session","turn_id":id,
            "input":{"origin":{"kind":"human","principal":"operator"},"text":"cedar"},
            "output":"answer","model":"exact","tool_receipts":[],"tools":[],
            "span_captures":[{"space":{"model_sha256":"1".repeat(64),"tokenizer_sha256":"2".repeat(64),
              "policy_sha256":"3".repeat(64),"projector_sha256":"4".repeat(64)},
              "source":"user","byte_start":0,"byte_end":5,"vector":vec![byte;VECTOR_BYTES]}]
        })).unwrap()
    }
    fn request(turn: &AcceptedTurn) -> SpanQuery {
        SpanQuery {
            akh: turn.akh.clone(),
            session_id: turn.session_id.clone(),
            space: turn.span_captures[0].space.clone(),
            vector: vec![0; VECTOR_BYTES],
            limit: 2,
        }
    }
    #[test]
    fn accepted_spans_are_ranked_and_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("core.redb");
        let store = Store::open(&path).unwrap();
        let first = turn("first", 255);
        let second = turn("second", 0);
        store
            .transaction(|tx| {
                append(tx, "sessions", &first)?;
                append(tx, "sessions", &second)
            })
            .unwrap();
        drop(store);
        let store = Store::open(&path).unwrap();
        let hits = query(&store, "sessions", &request(&second)).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].turn_id, "second");
        assert_eq!(hits[0].text, "cedar");
        assert_eq!(hits[0].hamming_distance, 0);
        assert_eq!(hits[1].hamming_distance, 10000);
    }

    #[test]
    fn empty_capture_extension_preserves_legacy_wire_shape() {
        let mut value = serde_json::to_value(turn("legacy", 0)).unwrap();
        value.as_object_mut().unwrap().remove("span_captures");
        let legacy: AcceptedTurn = serde_json::from_value(value.clone()).unwrap();
        assert!(legacy.span_captures.is_empty());
        assert_eq!(serde_json::to_value(legacy).unwrap(), value);
    }

    #[test]
    fn queries_never_cross_service_akh_session_or_representation() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("core.redb")).unwrap();
        let turn = turn("turn", 0);
        store
            .transaction(|tx| append(tx, "sessions", &turn))
            .unwrap();
        assert!(
            query(&store, "another-service", &request(&turn))
                .unwrap()
                .is_empty()
        );
        for field in [
            "akh",
            "session",
            "model",
            "tokenizer",
            "policy",
            "projector",
        ] {
            let mut q = request(&turn);
            match field {
                "akh" => q.akh = "other".into(),
                "session" => q.session_id = "other".into(),
                "model" => q.space.model_sha256 = "a".repeat(64),
                "tokenizer" => q.space.tokenizer_sha256 = "a".repeat(64),
                "policy" => q.space.policy_sha256 = "a".repeat(64),
                _ => q.space.projector_sha256 = "a".repeat(64),
            }
            assert!(query(&store, "sessions", &q).unwrap().is_empty(), "{field}");
        }
    }

    #[test]
    fn replay_is_idempotent_conflicts_and_invalid_batches_are_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("core.redb")).unwrap();
        let original = turn("same", 0);
        store
            .transaction(|tx| append(tx, "sessions", &original))
            .unwrap();
        store
            .transaction(|tx| append(tx, "sessions", &original))
            .unwrap();
        assert_eq!(
            query(&store, "sessions", &request(&original))
                .unwrap()
                .len(),
            1
        );
        assert!(
            store
                .transaction(|tx| append(tx, "sessions", &turn("same", 1)))
                .is_err()
        );
        let mut invalid = turn("invalid", 0);
        invalid.input.text = "cédar".into();
        invalid.span_captures[0].byte_start = 2; // middle of UTF-8 é
        assert!(
            store
                .transaction(|tx| {
                    append(tx, "sessions", &turn("rolled-back", 0))?;
                    append(tx, "sessions", &invalid)
                })
                .is_err()
        );
        let hits = query(&store, "sessions", &request(&original)).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].turn_id, "same");
        assert_eq!(hits[0].hamming_distance, 0);
    }

    #[test]
    fn session_retention_evicts_oldest_and_queries_are_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("core.redb")).unwrap();
        let first = turn("first", 0);
        store
            .transaction(|tx| {
                append(tx, "sessions", &first)?;
                for i in 0..MAX_SESSION_SPANS {
                    append(tx, "sessions", &turn(&format!("turn-{i:04}"), 255))?;
                }
                Ok(())
            })
            .unwrap();
        let mut q = request(&first);
        q.limit = 16;
        let hits = query(&store, "sessions", &q).unwrap();
        assert_eq!(hits.len(), 16);
        assert!(hits.iter().all(|hit| hit.hamming_distance == 10000));
        q.limit = 17;
        assert!(query(&store, "sessions", &q).is_err());
        q.limit = 1;
        q.vector.pop();
        assert!(query(&store, "sessions", &q).is_err());
    }
}
