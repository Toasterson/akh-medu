use akh_medu::autonomous::{
    rule_engine::{RuleEngine, RuleEngineConfig},
    rules::{InferenceRule, RuleKind, RuleSet, TriplePattern},
};
use akh_medu::{
    engine::{Engine, EngineConfig},
    graph::Triple,
};
fn engine() -> (tempfile::TempDir, Engine) {
    let dir = tempfile::tempdir().unwrap();
    let e = Engine::new(EngineConfig {
        data_dir: Some(dir.path().into()),
        ..Default::default()
    })
    .unwrap();
    (dir, e)
}
fn fact(e: &Engine, s: &str, p: &str, o: &str) -> Triple {
    let t = Triple::new(
        e.resolve_or_create_entity(s).unwrap(),
        e.resolve_or_create_relation(p).unwrap(),
        e.resolve_or_create_entity(o).unwrap(),
    );
    e.add_triple(&t).unwrap();
    e.all_triples()
        .into_iter()
        .find(|x| x.subject == t.subject && x.predicate == t.predicate && x.object == t.object)
        .unwrap()
}
fn infer(e: &Engine, specs: &[(&[&str], &str)]) -> Vec<Triple> {
    let rules = specs
        .iter()
        .enumerate()
        .map(|(i, (a, c))| {
            InferenceRule::new(
                format!("owned-{i}"),
                RuleKind::Custom {
                    name: "owned".into(),
                },
            )
            .with_antecedents(a.iter().map(|x| TriplePattern::parse(x).unwrap()).collect())
            .with_consequents(vec![TriplePattern::parse(c).unwrap()])
        })
        .collect();
    for name in ["target", "supported", "true", "downstream"] {
        e.resolve_or_create_entity(name).unwrap();
    }
    RuleEngine::new(RuleEngineConfig {
        auto_commit: true,
        ..Default::default()
    })
    .with_rules(RuleSet {
        name: "owned".into(),
        rules,
        source: "test".into(),
    })
    .run(e)
    .unwrap()
    .derived
    .into_iter()
    .map(|d| d.triple)
    .collect()
}
#[test]
fn exact_removal() {
    let (_d, e) = engine();
    let a = fact(&e, "a", "active", "true");
    assert_eq!(e.retract_owned_rule_snapshot(&[a.clone()], &[]).unwrap(), 1);
    assert!(!e.has_triple(a.subject, a.predicate, a.object));
}
#[test]
fn unrelated_shared_object() {
    let (_d, e) = engine();
    let a = fact(&e, "a", "active", "true");
    let b = fact(&e, "b", "active", "true");
    e.retract_owned_rule_snapshot(&[a], &[]).unwrap();
    assert!(e.has_triple(b.subject, b.predicate, b.object));
}
#[test]
fn conjunction_withdrawal() {
    let (_d, e) = engine();
    let a = fact(&e, "a", "active", "true");
    fact(&e, "b", "active", "true");
    let rules = &[(
        &["(a active true)", "(b active true)"][..],
        "(target supported true)",
    )];
    let owned = infer(&e, rules);
    assert_eq!(owned.len(), 1);
    e.retract_owned_rule_snapshot(&[a], &owned).unwrap();
    assert!(infer(&e, rules).is_empty());
}
#[test]
fn alternative_recomputed() {
    let (_d, e) = engine();
    let a = fact(&e, "a", "active", "true");
    fact(&e, "b", "active", "true");
    let rules = &[
        (&["(a active true)"][..], "(target supported true)"),
        (&["(b active true)"][..], "(target supported true)"),
    ];
    let owned = infer(&e, rules);
    assert_eq!(owned.len(), 1);
    e.retract_owned_rule_snapshot(&[a], &owned).unwrap();
    assert_eq!(infer(&e, rules).len(), 1);
}
#[test]
fn downstream_invalidated() {
    let (_d, e) = engine();
    let a = fact(&e, "a", "active", "true");
    let rules = &[
        (&["(a active true)"][..], "(target supported true)"),
        (
            &["(target supported true)"][..],
            "(downstream supported true)",
        ),
    ];
    let owned = infer(&e, rules);
    assert_eq!(owned.len(), 2);
    e.retract_owned_rule_snapshot(&[a], &owned).unwrap();
    assert!(infer(&e, rules).is_empty());
}
#[test]
fn assertion_not_owned() {
    let (_d, e) = engine();
    let a = fact(&e, "a", "active", "true");
    let target = fact(&e, "target", "supported", "true");
    assert!(
        e.retract_owned_rule_snapshot(&[a.clone()], &[target.clone()])
            .is_err()
    );
    assert!(e.has_triple(a.subject, a.predicate, a.object));
    assert!(e.has_triple(target.subject, target.predicate, target.object));
}
#[test]
fn mismatched_link_rejected() {
    let (_d, e) = engine();
    let a = fact(&e, "a", "active", "true");
    let mut owned = infer(&e, &[(&["(a active true)"][..], "(target supported true)")]);
    owned[0].provenance_id = None;
    assert!(e.retract_owned_rule_snapshot(&[a.clone()], &owned).is_err());
    assert!(e.has_triple(a.subject, a.predicate, a.object));
}
#[test]
fn cold_metadata_witness() {
    let (dir, e) = engine();
    let a = fact(&e, "a", "active", "true");
    e.persist().unwrap();
    drop(e);
    let e = Engine::new(EngineConfig {
        data_dir: Some(dir.path().into()),
        ..Default::default()
    })
    .unwrap();
    let mut stale = a.clone();
    stale.timestamp += 1;
    assert!(e.retract_owned_rule_snapshot(&[stale], &[]).is_err());
    e.retract_owned_rule_snapshot(&[a], &[]).unwrap();
}
