//! Bounded local experiment: native temporal prerequisites gate actual file effects.
use akh_medu::agent::causal::{ActionSchema, CausalPattern, PatternElement};
use akh_medu::agent::event_calculus::{Event, EventCalculusEngine};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::graph::Triple;
use serde_json::json;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::path::PathBuf::from(std::env::args().nth(1).ok_or("fresh directory required")?);
    std::fs::create_dir(&root)?;
    let engine = Engine::new(EngineConfig {
        data_dir: Some(root.join("graph")),
        ..EngineConfig::default()
    })?;
    let mut ec = EventCalculusEngine::new(&engine)?;
    let fluent = engine.resolve_or_create_entity("source:permission")?;
    let subject = engine.resolve_or_create_entity("fixture")?;
    let predicate = engine.resolve_or_create_relation("permission")?;
    let object = engine.resolve_or_create_entity("granted")?;
    let schema = ActionSchema {
        action_id: engine.resolve_or_create_entity("repair")?,
        name: "repair".into(),
        description: "Explicit temporal projection assistance".into(),
        parameters: vec![],
        preconditions: vec![CausalPattern {
            subject: PatternElement::Concrete(subject),
            predicate: PatternElement::Concrete(predicate),
            object: PatternElement::Concrete(object),
            negated: false,
        }],
        effects: vec![],
        success_rate: 0.0,
        execution_count: 0,
    };
    let mut rows = vec![];
    for (name, time, initiate, expected) in [
        ("current", 10, true, true),
        ("revoked", 20, false, false),
        ("corrected", 30, true, true),
    ] {
        ec.record_event(
            Event {
                symbol_id: engine.resolve_or_create_entity(name)?,
                name: name.into(),
                timestamp: time,
                initiates: if initiate { vec![fluent] } else { vec![] },
                terminates: if initiate { vec![] } else { vec![fluent] },
            },
            &engine,
        )?;
        ec.persist(&engine)?;
        ec = EventCalculusEngine::restore(&engine)?;
        let holding = ec.holds_at(fluent, time);
        // Declared adapter: only currently holding permission reaches ActionSchema.
        // History remains in EC; the projection does not assert a predicted effect.
        let current = if holding {
            vec![Triple::new(subject, predicate, object)]
        } else {
            vec![]
        };
        let applicable = schema.is_applicable(&current);
        let marker = root.join(format!("{name}.effect"));
        if applicable {
            assert!(
                std::process::Command::new("/usr/bin/touch")
                    .arg(&marker)
                    .status()?
                    .success()
            );
        }
        let effect = marker.is_file();
        assert_eq!(holding, expected);
        assert_eq!(applicable, expected);
        assert_eq!(effect, expected);
        rows.push(json!({"case":name,"time":time,"holding":holding,"applicable":applicable,"actual_effect":effect,"expected":expected}));
    }
    assert!(ec.holds_at(fluent, 10));
    assert!(!ec.holds_at(fluent, 20));
    assert!(ec.holds_at(fluent, 30));
    // Source timestamps may arrive late: evaluate native behavior, not an assumed fix.
    ec.record_event(
        Event {
            symbol_id: engine.resolve_or_create_entity("late-old-revocation")?,
            name: "late-old-revocation".into(),
            timestamp: 15,
            initiates: vec![],
            terminates: vec![fluent],
        },
        &engine,
    )?;
    let delayed_holding = ec.holds_at(fluent, 30);
    assert!(
        delayed_holding,
        "late old revocation must not override correction"
    );
    assert!(!ec.holds_at(fluent, 20));
    let result = json!({"ordered_cases":rows,"ordered_controls_passed":true,"late_old_revocation":{"observed_holds_at_30":delayed_holding,"expected_holds_at_30":true,"passed":delayed_holding},"native_event_calculus":true,"native_action_schema":true,"assistance":"explicit fluent-to-precondition projection","model_used":false,"new_world_credit":0});
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", result);
    Ok(())
}
