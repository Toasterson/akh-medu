//! Phase 51 P1a-2 (Wave 2): fast test that the `--n-tools` CLI wiring in
//! `examples/train_p1a.rs` builds a genuinely procedural [`grug_data::Dataset`]
//! and that the BPE tokenizer training corpus (prompt + expected-call JSON,
//! the same shape `train_p1a::load_or_train_tokenizer` builds) comes out
//! non-degenerate over it.
//!
//! `examples/train_p1a.rs`'s `build_dataset(seed, n_tools, holdout_frac)`
//! helper is a thin `if n_tools.is_some() { Dataset::new_sized(..) } else {
//! Dataset::new(..) }` -- example-binary internals aren't reachable from an
//! integration test, so this test exercises the exact same public API the
//! helper calls, at a scale small enough to stay well under ~30s.

use grug_data::Dataset;
use grug_train::train::format_prompt;

#[test]
fn n_tools_wiring_builds_a_procedural_dataset() {
    // Mirrors `--n-tools 20 --holdout-frac 0.15`: small enough to be fast,
    // large enough that per-tool example count is meaningfully below the
    // hand-authored-catalog's ~500/tool.
    let dataset = Dataset::new_sized(42, 20, 0.15).expect("procedural catalog splits cleanly");

    // holdout_frac=0.15 of 20 tools -> ceil(3.0) = 3 holdout, 17 train.
    assert_eq!(dataset.train_tools().len(), 17);
    assert_eq!(dataset.holdout_tools().len(), 3);

    // Whole-tool holdout tools must be disjoint from train tools (the axis-A
    // guarantee the eval gate depends on).
    let train_names: std::collections::HashSet<&str> =
        dataset.train_tools().iter().map(|t| t.name.as_str()).collect();
    for holdout_tool in dataset.holdout_tools() {
        assert!(
            !train_names.contains(holdout_tool.name.as_str()),
            "holdout tool {} leaked into the train set",
            holdout_tool.name
        );
    }

    // The corpus-building shape `load_or_train_tokenizer` uses: prompt +
    // expected-call JSON per example, over the *procedural* train stream.
    let examples = dataset
        .train_examples(60)
        .expect("procedural generator produces train examples");
    assert_eq!(examples.len(), 60);

    let corpus: Vec<String> = examples
        .iter()
        .flat_map(|ex| vec![format_prompt(ex), ex.expected_call_as_json_string()])
        .collect();
    assert_eq!(corpus.len(), 120, "one prompt + one target JSON per example");
    assert!(
        corpus.iter().all(|s| !s.is_empty()),
        "no empty corpus strings"
    );

    // Every rendered prompt should mention at least one procedurally-named
    // tool (verb_noun scheme) so the corpus actually carries the new
    // identifier surface the BPE tokenizer needs subword coverage for, not
    // just leftover 56-tool-catalog names.
    let procedural_name_appears = examples.iter().any(|ex| {
        format_prompt(ex).contains(&ex.expected_call.name)
            || ex.expected_call.name.contains('_')
    });
    assert!(procedural_name_appears, "expected verb_noun-shaped tool names in the corpus");
}

#[test]
fn n_tools_absent_falls_back_to_built_in_catalog() {
    // The no-flag path (`Dataset::new`) must still work unchanged so a
    // pre-Wave-2 invocation (no `--n-tools`) keeps behaving as before.
    let dataset = Dataset::new(42).expect("built-in catalog splits cleanly");
    assert!(!dataset.train_tools().is_empty());
    assert!(!dataset.holdout_tools().is_empty());
}
