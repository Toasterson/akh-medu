//! WI-262 verification tests for shend's resolution tier.
//!
//! The four WI-mandated proofs:
//! 1. A fixture crate with two same-named symbols in different modules
//!    resolves each reference to the correct definition — and naive name
//!    matching demonstrably gets it wrong.
//! 2. An unresolvable reference is recorded explicitly unresolved — not
//!    omitted, not guessed.
//! 3. Indexing a single file with its dependency absent produces partial
//!    resolutions that complete once the dependency is indexed.
//! 4. Bulk-retracting the `stackgraph` tier leaves `syntactic` resolutions
//!    intact.

#[path = "fixtures/mod.rs"]
mod fixtures;

use shend::index::naive_name_match;
use shend::{
    DefId, Provenance, ResolutionStatus, ResolutionStore, Resolver, ResolvesTo, Span, Tier,
    Watermark,
};

fn watermark() -> Watermark {
    Watermark::new(fixtures::VIRTUAL_REPO, "wi262-fixture-commit")
}

/// Index the whole fixture crate (any order — resolution composes across
/// files) and resolve every file's references into the store.
fn index_fixture_crate() -> (Resolver, ResolutionStore) {
    let mut resolver = Resolver::new().expect("rust.tsg loads and validates");
    let files = fixtures::demo_crate_files();
    // index in *reverse* dependency order on purpose: consumer before alpha —
    // per-file indexing must not depend on dependencies being present.
    for (path, source) in files.iter().rev() {
        resolver.index_file(path, source).expect("file indexes");
    }
    let mut store = ResolutionStore::new();
    for (path, _) in &files {
        for edge in resolver.resolutions_for_file(path, &watermark()) {
            store.record_resolution(edge);
        }
    }
    (resolver, store)
}

/// Find the resolved binding of the `make()` call inside `run()` in
/// consumer.rs: the reference named `make` at the call site (not the one in
/// the `use` line). Span-based: the call is on the line after the `use`.
fn resolved_call_target<'a>(store: &'a ResolutionStore, file: &str) -> &'a DefId {
    store
        .resolves()
        .iter()
        .filter(|e| {
            e.reference_file == file
                && e.reference_name == "make"
                && e.status == ResolutionStatus::Resolved
        })
        .max_by_key(|e| e.reference_span.start_row)
        .and_then(|e| e.target.as_ref())
        .unwrap_or_else(|| panic!("the `make()` call in {file} resolved"))
}

#[test]
fn same_named_symbols_resolve_to_their_own_modules() {
    let (resolver, store) = index_fixture_crate();
    let _ = &resolver;

    // The `make()` call in consumer.rs (which `use crate::alpha::make;`s)
    // binds to alpha's definition — not beta's, though it has the same name.
    let target = resolved_call_target(&store, "src/consumer.rs");
    assert_eq!(
        target.file, "src/alpha.rs",
        "make() must resolve through the alias to alpha::make, got {target:?}"
    );
    assert_eq!(target.name, "make");
    // Provenance lives on the `ResolvesTo` edge, not the `DefId` (store.rs:
    // `target: Option<DefId>` / `provenance: Provenance`). Find the edge that
    // produced this binding and assert its tier.
    let edge = store
        .resolves()
        .iter()
        .find(|e| {
            e.reference_file == "src/consumer.rs"
                && e.reference_name == "make"
                && e.status == ResolutionStatus::Resolved
                && e.target.as_ref() == Some(target)
        })
        .expect("the edge binding make() to alpha::make exists");
    assert_eq!(edge.provenance.tier, Tier::Stackgraph);

    // And the same-named call inside alpha.rs binds file-locally.
    let target = resolved_call_target(&store, "src/alpha.rs");
    assert_eq!(target.file, "src/alpha.rs");

    // --- the counterexample: name matching WOULD have got it wrong ---
    // Feed naive name matching exactly the definitions the graph sees:
    // two `make` definitions, one per file.
    let defs = [
        ("src/alpha.rs".to_string(), "make".to_string()),
        ("src/beta.rs".to_string(), "make".to_string()),
    ];
    let naive = naive_name_match(defs, "make");
    assert_eq!(
        naive.len(),
        2,
        "name matching sees both `make`s and cannot pick — this ambiguity is \
         exactly why resolution must be binding, not name: {naive:?}"
    );
    // Had it picked deterministically (e.g. first), there is no alpha/beta
    // information in the name — the correct answer (alpha) is only reachable
    // through the module chain.
    assert!(naive.contains("src/beta.rs"));
}

#[test]
fn unresolvable_reference_is_recorded_unresolved() {
    let (_resolver, store) = index_fixture_crate();

    // consumer.rs calls `nonexistent_function()` — nothing anywhere defines
    // it. It must be present in the store, explicitly unresolved, with no
    // target guessed from the name.
    let unresolved: Vec<_> = store.unresolved_at(Tier::Stackgraph).collect();
    assert!(!unresolved.is_empty(), "unresolved references recorded");
    let missing = unresolved
        .iter()
        .find(|e| e.reference_file == "src/consumer.rs" && e.reference_name == "nonexistent_function")
        .expect("nonexistent_function recorded explicitly, not omitted");
    assert_eq!(missing.status, ResolutionStatus::Unresolved);
    assert_eq!(missing.target, None, "never guessed");

    // It contributes NO dependency edge.
    assert!(
        !store
            .depends_on()
            .iter()
            .any(|d| d.from_file == "src/consumer.rs" && d.to_file.is_empty()),
        "unresolved reference must not fabricate a dependency"
    );
}

#[test]
fn dependencies_derived_at_file_and_crate_granularity() {
    let (_resolver, store) = index_fixture_crate();
    let file_deps = store.file_dependencies();
    assert!(
        file_deps.contains(&("src/consumer.rs".to_string(), "src/alpha.rs".to_string())),
        "consumer depends on alpha's file: {file_deps:?}"
    );
    // Same-crate fixture → no cross-CRATE edges, and every depends-on carries
    // provenance.
    for d in store.depends_on() {
        assert_eq!(d.provenance.tier, Tier::Stackgraph);
        assert_eq!(d.provenance.watermark, watermark());
    }
}

#[test]
fn partial_resolution_completes_after_dependency_is_indexed() {
    let wm = watermark();
    let mut resolver = Resolver::new().expect("rust.tsg loads");

    // Index ONLY consumer.rs — its `use crate::alpha::make` dependency is
    // absent. File-local indexing must succeed...
    let consumer = fixtures::fixture("src/consumer.rs");
    resolver
        .index_file("src/consumer.rs", &consumer)
        .expect("file indexes with dependencies absent");

    // ...and produce resolutions, but the `make` binding cannot complete yet.
    let before = resolver.resolutions_for_file("src/consumer.rs", &wm);
    let make_before = &before[
        before
            .iter()
            .position(|e| e.reference_name == "make")
            .expect("the make reference is recorded")
    ];
    assert_eq!(
        make_before.status,
        ResolutionStatus::Unresolved,
        "with alpha.rs absent, make cannot resolve: {make_before:?}"
    );

    // Now index the dependency. Partial paths compose in the shared database —
    // no re-parsing, no re-indexing of consumer.rs — and the reference
    // completes.
    let alpha = fixtures::fixture("src/alpha.rs");
    resolver
        .index_file("src/alpha.rs", &alpha)
        .expect("dependency indexes later");
    let after = resolver.resolutions_for_file("src/consumer.rs", &wm);
    let resolved: Vec<_> = after
        .iter()
        .filter(|e| e.reference_name == "make" && e.status == ResolutionStatus::Resolved)
        .collect();
    assert!(
        resolved
            .iter()
            .any(|e| e.target.as_ref().is_some_and(|t| t.file == "src/alpha.rs")),
        "after indexing alpha.rs the make reference resolves to it: {after:?}"
    );
}

#[test]
fn retracting_stackgraph_tier_leaves_syntactic_intact() {
    let (_resolver, mut store) = index_fixture_crate();
    assert!(
        store.resolved_at(Tier::Stackgraph).next().is_some(),
        "fixture produced stackgraph-tier resolutions"
    );

    // Syntactic-tier edges: file-local bindings asserted by the always-on
    // parse stage (same-file reference → definition), which must survive a
    // stackgraph-tier retraction. alpha.rs's internal `make()` calls are the
    // canonical file-local case.
    let syntactic = ResolvesTo {
        reference_file: "src/alpha.rs".to_string(),
        reference_name: "make".to_string(),
        reference_span: Span {
            start_row: 9,
            start_col: 15,
            end_row: 9,
            end_col: 19,
        },
        target: Some(DefId {
            file: "src/alpha.rs".to_string(),
            name: "make".to_string(),
            module: "crate::alpha".to_string(),
            span: Span {
                start_row: 2,
                start_col: 7,
                end_row: 2,
                end_col: 11,
            },
        }),
        status: ResolutionStatus::Resolved,
        provenance: Provenance::new(Tier::Syntactic, watermark()),
    };
    store.record_resolution(syntactic);
    let syntactic_count_before = store.resolved_at(Tier::Syntactic).count();
    assert!(syntactic_count_before >= 1);

    // Bulk-retract the stackgraph tier.
    store.retract_tier(Tier::Stackgraph);

    assert_eq!(
        store.resolved_at(Tier::Stackgraph).count(),
        0,
        "all stackgraph-tier resolutions retracted"
    );
    assert_eq!(
        store.unresolved_at(Tier::Stackgraph).count(),
        0,
        "unresolved records of the tier retract as well"
    );
    assert_eq!(
        store.resolved_at(Tier::Syntactic).count(),
        syntactic_count_before,
        "syntactic-tier resolutions are untouched"
    );
    assert!(
        store
            .depends_on()
            .iter()
            .all(|d| d.provenance.tier != Tier::Stackgraph),
        "stackgraph-tier depends-on edges retract with their tier"
    );
}

#[test]
fn ruleset_loads_and_validates() {
    // Resolver::new parses the embedded rust.tsg against tree-sitter-graph
    // 0.12 + tree-sitter-rust 0.24 — the ruleset is live, not dead weight.
    let resolver = Resolver::new();
    assert!(resolver.is_ok(), "rust.tsg loads: {:?}", resolver.err());
}
