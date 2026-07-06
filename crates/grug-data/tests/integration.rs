//! Integration tests for `grug-data`: determinism, holdout disjointness (both
//! axes), eval-scorer correctness at each gated level, and the anti-collapse
//! alarm.

use std::collections::HashSet;

use grug_data::eval::{score_output, ArgMismatch, DiversityReport, EvalReport, SchemaViolation};
use grug_data::{Dataset, DistractorConfig, EvalAxis, Example, Generator, ToolCatalog};

#[test]
fn determinism_same_seed_same_stream() {
    let mut gen_a = Generator::new(1234);
    let mut gen_b = Generator::new(1234);

    let batch_a = gen_a.generate(50).unwrap();
    let batch_b = gen_b.generate(50).unwrap();

    assert_eq!(batch_a.len(), batch_b.len());
    for (a, b) in batch_a.iter().zip(batch_b.iter()) {
        assert_eq!(a.user_request, b.user_request);
        assert_eq!(a.expected_call, b.expected_call);
        assert_eq!(a.mock_response, b.mock_response);
        assert_eq!(
            a.tools.iter().map(|t| &t.name).collect::<Vec<_>>(),
            b.tools.iter().map(|t| &t.name).collect::<Vec<_>>()
        );
    }
}

#[test]
fn determinism_dataset_streams_reproduce() {
    let ds_a = Dataset::new(777).unwrap();
    let ds_b = Dataset::new(777).unwrap();

    let train_a = ds_a.train_examples(30).unwrap();
    let train_b = ds_b.train_examples(30).unwrap();
    assert_eq!(train_a, train_b);

    let holdout_a = ds_a.holdout_examples(10).unwrap();
    let holdout_b = ds_b.holdout_examples(10).unwrap();
    assert_eq!(holdout_a, holdout_b);
}

#[test]
fn different_seeds_diverge() {
    let mut gen_a = Generator::new(1);
    let mut gen_b = Generator::new(2);
    let batch_a = gen_a.generate(20).unwrap();
    let batch_b = gen_b.generate(20).unwrap();
    // Overwhelmingly likely to differ somewhere in 20 draws; if this ever
    // flakes, the generator's randomness derivation is suspect.
    assert_ne!(batch_a, batch_b);
}

#[test]
fn holdout_disjointness_no_holdout_tool_in_training_stream() {
    let dataset = Dataset::new(42).unwrap();

    let holdout_names: std::collections::HashSet<&str> = dataset
        .holdout_tools()
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(!holdout_names.is_empty(), "holdout set must be non-empty");

    let train_names: std::collections::HashSet<&str> = dataset
        .train_tools()
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(
        holdout_names.is_disjoint(&train_names),
        "train and holdout tool sets must be disjoint"
    );

    // Generate a large training stream and confirm no holdout tool name
    // appears anywhere: as the expected call, nor among offered distractors.
    let train_examples = dataset.train_examples(500).unwrap();
    for ex in &train_examples {
        assert!(
            !holdout_names.contains(ex.expected_call.name.as_str()),
            "holdout tool {} leaked into training expected_call",
            ex.expected_call.name
        );
        for offered in &ex.tools {
            assert!(
                !holdout_names.contains(offered.name.as_str()),
                "holdout tool {} leaked into training offered-tools bundle",
                offered.name
            );
        }
    }

    // And the reverse: holdout examples never draw a training tool.
    let holdout_examples = dataset.holdout_examples(200).unwrap();
    for ex in &holdout_examples {
        assert!(
            !train_names.contains(ex.expected_call.name.as_str()),
            "training tool {} leaked into holdout expected_call",
            ex.expected_call.name
        );
        for offered in &ex.tools {
            assert!(
                !train_names.contains(offered.name.as_str()),
                "training tool {} leaked into holdout offered-tools bundle",
                offered.name
            );
        }
    }
}

#[test]
fn split_rejects_degenerate_holdout_counts() {
    let catalog = ToolCatalog::built_in();
    let total = catalog.len();

    assert!(catalog.split_at(0).is_err());
    assert!(catalog.split_at(total).is_err());
    assert!(catalog.split_at(total + 5).is_err());
    assert!(catalog.split_at(1).is_ok());
}

#[test]
fn jsonl_roundtrip() {
    let mut generator = Generator::new(9);
    let example = generator.next_example();
    let line = example.to_jsonl().unwrap();
    assert!(!line.contains('\n'));
    let parsed = Example::from_jsonl(&line).unwrap();
    assert_eq!(example, parsed);
}

#[test]
fn eval_known_good_output_passes_all_levels() {
    let mut generator = Generator::new(55);
    let example = generator.next_example();
    let raw = example.expected_call_as_json_string();

    let outcome = score_output(&raw, &example);
    assert!(outcome.valid_json());
    assert!(outcome.schema_valid());
    assert!(outcome.arg_correct());
}

#[test]
fn eval_malformed_json_fails_at_json_level() {
    let mut generator = Generator::new(2);
    let example = generator.next_example();

    let raw = "{not valid json at all";
    let outcome = score_output(raw, &example);
    assert!(!outcome.valid_json());
    assert!(outcome.schema.is_none());
    assert!(outcome.arg_correctness.is_none());
}

#[test]
fn eval_wrong_tool_fails_at_schema_level() {
    let mut generator = Generator::new(3);
    let example = generator.next_example();

    // Pick a tool name guaranteed not to be the expected one, and not
    // necessarily offered either — use a name outside the whole catalog.
    let raw = r#"{"name": "definitely_not_a_real_tool", "arguments": {}}"#;
    let outcome = score_output(raw, &example);

    assert!(outcome.valid_json());
    assert!(!outcome.schema_valid());
    match &outcome.schema {
        Some(Err(violations)) => {
            assert!(violations.iter().any(|v| matches!(
                v,
                SchemaViolation::UnknownTool { name } if name == "definitely_not_a_real_tool"
            )));
        }
        other => panic!("expected schema violation, got {other:?}"),
    }
    assert!(outcome.arg_correctness.is_none());
}

#[test]
fn eval_missing_required_arg_fails_at_schema_level() {
    // get_weather requires "city".
    let catalog = ToolCatalog::built_in();
    let weather = catalog
        .tools()
        .iter()
        .find(|t| t.name == "get_weather")
        .unwrap()
        .clone();
    let example = Example {
        tools: vec![weather],
        user_request: "What's the weather?".to_string(),
        expected_call: grug_data::ToolCall::new(
            "get_weather",
            serde_json::json!({"city": "Zurich"}),
        ),
        mock_response: "{}".to_string(),
    };

    let raw = r#"{"name": "get_weather", "arguments": {"units": "celsius"}}"#;
    let outcome = score_output(raw, &example);

    assert!(outcome.valid_json());
    assert!(!outcome.schema_valid());
    match &outcome.schema {
        Some(Err(violations)) => {
            assert!(violations.iter().any(|v| matches!(
                v,
                SchemaViolation::MissingRequired { param } if param == "city"
            )));
        }
        other => panic!("expected schema violation, got {other:?}"),
    }
}

#[test]
fn eval_wrong_arg_type_fails_at_schema_level() {
    let catalog = ToolCatalog::built_in();
    let timer = catalog
        .tools()
        .iter()
        .find(|t| t.name == "set_timer")
        .unwrap()
        .clone();
    let example = Example {
        tools: vec![timer],
        user_request: "Set a timer for 10 minutes.".to_string(),
        expected_call: grug_data::ToolCall::new(
            "set_timer",
            serde_json::json!({"duration_minutes": 10}),
        ),
        mock_response: "{}".to_string(),
    };

    // duration_minutes should be an Integer, not a string.
    let raw = r#"{"name": "set_timer", "arguments": {"duration_minutes": "ten"}}"#;
    let outcome = score_output(raw, &example);

    assert!(outcome.valid_json());
    assert!(!outcome.schema_valid());
    match &outcome.schema {
        Some(Err(violations)) => {
            assert!(violations.iter().any(|v| matches!(
                v,
                SchemaViolation::WrongType { param } if param == "duration_minutes"
            )));
        }
        other => panic!("expected schema violation, got {other:?}"),
    }
}

#[test]
fn eval_hallucinated_arg_fails_at_schema_level() {
    let catalog = ToolCatalog::built_in();
    let weather = catalog
        .tools()
        .iter()
        .find(|t| t.name == "get_weather")
        .unwrap()
        .clone();
    let example = Example {
        tools: vec![weather],
        user_request: "What's the weather in Zurich?".to_string(),
        expected_call: grug_data::ToolCall::new(
            "get_weather",
            serde_json::json!({"city": "Zurich"}),
        ),
        mock_response: "{}".to_string(),
    };

    // "forecast_days" is not a declared param on get_weather.
    let raw = r#"{"name": "get_weather", "arguments": {"city": "Zurich", "forecast_days": 5}}"#;
    let outcome = score_output(raw, &example);

    assert!(outcome.valid_json());
    assert!(!outcome.schema_valid());
    match &outcome.schema {
        Some(Err(violations)) => {
            assert!(violations.iter().any(|v| matches!(
                v,
                SchemaViolation::UnknownArg { param } if param == "forecast_days"
            )));
        }
        other => panic!("expected schema violation, got {other:?}"),
    }
}

#[test]
fn eval_schema_valid_but_wrong_argument_value_fails_at_arg_level() {
    let catalog = ToolCatalog::built_in();
    let weather = catalog
        .tools()
        .iter()
        .find(|t| t.name == "get_weather")
        .unwrap()
        .clone();
    let example = Example {
        tools: vec![weather],
        user_request: "What's the weather in Zurich?".to_string(),
        expected_call: grug_data::ToolCall::new(
            "get_weather",
            serde_json::json!({"city": "Zurich"}),
        ),
        mock_response: "{}".to_string(),
    };

    // Schema-valid (city is a string, get_weather is offered) but the wrong
    // city — argument correctness must fail, not schema.
    let raw = r#"{"name": "get_weather", "arguments": {"city": "Tokyo"}}"#;
    let outcome = score_output(raw, &example);

    assert!(outcome.valid_json());
    assert!(outcome.schema_valid());
    assert!(!outcome.arg_correct());
    match &outcome.arg_correctness {
        Some(Err(mismatches)) => {
            assert!(mismatches
                .iter()
                .any(|m| matches!(m, ArgMismatch::ValueDiffers { param } if param == "city")));
        }
        other => panic!("expected arg mismatch, got {other:?}"),
    }
}

#[test]
fn eval_report_aggregates_batch_rates() {
    let mut generator = Generator::new(101);
    let examples = generator.generate(10).unwrap();

    let pairs: Vec<(String, Example)> = examples
        .iter()
        .map(|ex| (ex.expected_call_as_json_string(), ex.clone()))
        .collect();

    let report = EvalReport::from_outputs(&pairs);
    assert_eq!(report.n, 10);
    assert_eq!(report.valid_json_rate, 1.0);
    assert_eq!(report.schema_adherence_rate, 1.0);
    assert_eq!(report.arg_correct_rate, 1.0);
}

#[test]
fn anti_collapse_identical_outputs_score_near_zero_diversity() {
    let mut generator = Generator::new(202);
    let examples = generator.generate(20).unwrap();

    // Simulate a mode-collapsed model: always emits the exact same call,
    // regardless of the prompt/example.
    let collapsed_call = r#"{"name": "get_weather", "arguments": {"city": "Zurich"}}"#;
    let pairs: Vec<(String, Example)> = examples
        .iter()
        .map(|ex| (collapsed_call.to_string(), ex.clone()))
        .collect();

    let report = DiversityReport::from_outputs(&pairs);
    assert!(
        report.distinct_call_ratio < 0.2,
        "expected near-zero distinct_call_ratio, got {}",
        report.distinct_call_ratio
    );
    assert!(
        report.tool_name_entropy < 0.2,
        "expected near-zero tool_name_entropy, got {}",
        report.tool_name_entropy
    );
}

#[test]
fn anti_collapse_varied_outputs_score_high_diversity() {
    let mut generator = Generator::new(303);
    let examples = generator.generate(30).unwrap();

    // Simulate a healthy model: correctly answers each example (which draws
    // varied tools/arguments by construction).
    let pairs: Vec<(String, Example)> = examples
        .iter()
        .map(|ex| (ex.expected_call_as_json_string(), ex.clone()))
        .collect();

    let report = DiversityReport::from_outputs(&pairs);
    assert!(
        report.distinct_call_ratio > 0.5,
        "expected high distinct_call_ratio, got {}",
        report.distinct_call_ratio
    );
    assert!(
        report.tool_name_entropy > 0.5,
        "expected high tool_name_entropy, got {}",
        report.tool_name_entropy
    );
}

// ---------------------------------------------------------------------
// Phase 51 P1a: catalog scale + the phrasing-holdout (axis B) honest gate.
// ---------------------------------------------------------------------

#[test]
fn catalog_size_in_range() {
    let catalog = ToolCatalog::built_in();
    assert!(
        (40..=60).contains(&catalog.len()),
        "expected 40-60 tools, got {}",
        catalog.len()
    );
}

#[test]
fn catalog_has_varied_param_shapes() {
    let catalog = ToolCatalog::built_in();
    let tools = catalog.tools();

    let has_enum = tools
        .iter()
        .filter(|t| {
            t.params
                .iter()
                .any(|p| matches!(p.ty, grug_data::ParamType::Enum(_)))
        })
        .count();
    let has_optional = tools
        .iter()
        .filter(|t| t.params.iter().any(|p| !p.required))
        .count();
    let many_params = tools.iter().filter(|t| t.params.len() >= 5).count();
    let one_param = tools.iter().filter(|t| t.params.len() == 1).count();

    assert!(has_enum >= 5, "expected several tools with enum params, got {has_enum}");
    assert!(
        has_optional >= 10,
        "expected several tools with optional params, got {has_optional}"
    );
    assert!(
        many_params >= 2,
        "expected at least 2 tools with 5+ params, got {many_params}"
    );
    assert!(
        one_param >= 2,
        "expected at least 2 tools with exactly 1 param, got {one_param}"
    );
}

#[test]
fn every_tool_has_a_phrasing_bank_with_both_halves_nonempty() {
    let catalog = ToolCatalog::built_in();
    for tool in catalog.tools() {
        let bank = catalog
            .phrasings()
            .get(&tool.name)
            .unwrap_or_else(|_| panic!("no phrasing bank for {}", tool.name));
        assert!(
            bank.train_count() >= 1,
            "{} has no train phrasings",
            tool.name
        );
        assert!(
            bank.holdout_count() >= 1,
            "{} has no holdout phrasings",
            tool.name
        );
        assert!(
            bank.train_count() + bank.holdout_count() >= 6,
            "{} has fewer than 6 total phrasing templates ({} + {})",
            tool.name,
            bank.train_count(),
            bank.holdout_count()
        );
    }
}

#[test]
fn phrasing_holdout_disjointness_no_holdout_template_leaks_into_training() {
    // For every train-catalog tool, render every train-phrasing template and
    // every holdout-phrasing template against the SAME expected call, and
    // confirm the two rendered-string sets never intersect. This is the
    // structural guarantee: if a holdout phrasing ever produced the same
    // text as some train phrasing, the eval would be measuring memorization,
    // not generalization.
    let dataset = Dataset::new(4242).unwrap();
    let catalog = ToolCatalog::built_in();

    for tool in dataset.train_tools() {
        let bank = catalog.phrasings().get(&tool.name).unwrap();

        // A representative call: fuzz-free zero-ish values are fine here —
        // we only care that the two template halves render to disjoint text
        // for arbitrary shared input, not about argument correctness.
        let mut args = serde_json::Map::new();
        for p in &tool.params {
            let placeholder = match &p.ty {
                grug_data::ParamType::String => serde_json::Value::String("PLACEHOLDER".into()),
                grug_data::ParamType::Integer => serde_json::Value::from(7),
                grug_data::ParamType::Number => serde_json::Value::from(7.5),
                grug_data::ParamType::Boolean => serde_json::Value::Bool(true),
                grug_data::ParamType::Enum(variants) => {
                    serde_json::Value::String(variants.first().cloned().unwrap_or_default())
                }
            };
            args.insert(p.name.clone(), placeholder);
        }
        let call = grug_data::ToolCall::new(tool.name.clone(), serde_json::Value::Object(args));

        let train_renders: HashSet<String> = bank
            .train_templates()
            .iter()
            .map(|t| t.render(&call))
            .collect();
        let holdout_renders: HashSet<String> = bank
            .holdout_templates()
            .iter()
            .map(|t| t.render(&call))
            .collect();

        assert!(
            train_renders.is_disjoint(&holdout_renders),
            "tool {} has a phrasing template that renders identically in both \
             the train and holdout halves: train={:?} holdout={:?}",
            tool.name,
            train_renders,
            holdout_renders
        );
    }

    // Now the behavioral guarantee: generate a large TRAIN stream and a
    // large HELD-OUT-PHRASING stream, and confirm none of the holdout
    // stream's rendered `user_request` strings ever appear in the training
    // stream's rendered requests, across every train tool.
    let train_examples = dataset.train_examples(2000).unwrap();
    let holdout_phrasing_examples = dataset.holdout_phrasing_examples(2000).unwrap();

    let train_requests: HashSet<&str> = train_examples
        .iter()
        .map(|ex| ex.user_request.as_str())
        .collect();

    let mut leaked = 0usize;
    for ex in &holdout_phrasing_examples {
        if train_requests.contains(ex.user_request.as_str()) {
            leaked += 1;
        }
    }
    assert_eq!(
        leaked, 0,
        "{leaked} held-out-phrasing request(s) leaked verbatim into the training stream"
    );
}

#[test]
fn holdout_phrasing_examples_only_draw_train_tools() {
    let dataset = Dataset::new(99).unwrap();
    let train_names: HashSet<&str> = dataset
        .train_tools()
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    let holdout_tool_names: HashSet<&str> = dataset
        .holdout_tools()
        .iter()
        .map(|t| t.name.as_str())
        .collect();

    let examples = dataset.holdout_phrasing_examples(300).unwrap();
    assert!(!examples.is_empty());
    for ex in &examples {
        assert!(
            train_names.contains(ex.expected_call.name.as_str()),
            "held-out-phrasing example used tool {} which is not a train tool",
            ex.expected_call.name
        );
        assert!(
            !holdout_tool_names.contains(ex.expected_call.name.as_str()),
            "held-out-phrasing example used whole-tool-holdout tool {}",
            ex.expected_call.name
        );
    }
}

#[test]
fn determinism_all_three_streams_reproduce() {
    let ds_a = Dataset::new(555).unwrap();
    let ds_b = Dataset::new(555).unwrap();

    let train_a = ds_a.train_examples(200).unwrap();
    let train_b = ds_b.train_examples(200).unwrap();
    assert_eq!(train_a, train_b);

    let holdout_a = ds_a.holdout_examples(50).unwrap();
    let holdout_b = ds_b.holdout_examples(50).unwrap();
    assert_eq!(holdout_a, holdout_b);

    let phrasing_a = ds_a.holdout_phrasing_examples(50).unwrap();
    let phrasing_b = ds_b.holdout_phrasing_examples(50).unwrap();
    assert_eq!(phrasing_a, phrasing_b);

    // And the three streams must NOT be identical to each other (independent
    // RNG streams via distinct seed salts) even though train and
    // held-out-phrasing share a tool catalog.
    assert_ne!(train_a, phrasing_a);
}

#[test]
fn volume_10k_train_examples_generate_without_pathological_slowdown() {
    let dataset = Dataset::new(7).unwrap();
    let start = std::time::Instant::now();
    let examples = dataset.train_examples(10_000).unwrap();
    let elapsed = start.elapsed();
    assert_eq!(examples.len(), 10_000);
    assert!(
        elapsed.as_secs() < 5,
        "10k example generation took {elapsed:?}, expected well under 5s"
    );
}

#[test]
fn streaming_iterator_matches_vec_returning_method() {
    let dataset = Dataset::new(31).unwrap();
    let via_vec = dataset.train_examples(25).unwrap();
    let via_iter: Vec<Example> = dataset
        .train_examples_iter()
        .take(25)
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(via_vec, via_iter);
}

#[test]
fn two_axis_eval_produces_well_formed_labeled_reports() {
    let dataset = Dataset::new(123).unwrap();

    // A "perfect" model: always answers with the ground-truth call.
    let (phrasing_report, whole_tool_report) = dataset
        .eval_both_axes(20, |ex| ex.expected_call_as_json_string())
        .unwrap();

    assert_eq!(phrasing_report.axis, EvalAxis::HeldOutPhrasing);
    assert_eq!(whole_tool_report.axis, EvalAxis::WholeToolHoldout);

    assert_eq!(phrasing_report.eval.n, 20);
    assert_eq!(whole_tool_report.eval.n, 20);

    // A perfect model should hit 100% on every level, on both axes.
    assert_eq!(phrasing_report.eval.valid_json_rate, 1.0);
    assert_eq!(phrasing_report.eval.schema_adherence_rate, 1.0);
    assert_eq!(phrasing_report.eval.arg_correct_rate, 1.0);
    assert_eq!(whole_tool_report.eval.valid_json_rate, 1.0);
    assert_eq!(whole_tool_report.eval.schema_adherence_rate, 1.0);
    assert_eq!(whole_tool_report.eval.arg_correct_rate, 1.0);

    // Diversity should be well-formed (non-negative, bounded) on both axes.
    assert!(phrasing_report.diversity.distinct_call_ratio > 0.0);
    assert!(whole_tool_report.diversity.distinct_call_ratio > 0.0);
}

#[test]
fn score_axis_labels_match_the_axis_argument() {
    let dataset = Dataset::new(88).unwrap();
    let examples = dataset.holdout_phrasing_examples(10).unwrap();
    let pairs: Vec<(String, Example)> = examples
        .iter()
        .map(|ex| (ex.expected_call_as_json_string(), ex.clone()))
        .collect();

    let report = Dataset::score_axis(EvalAxis::HeldOutPhrasing, &pairs);
    assert_eq!(report.axis, EvalAxis::HeldOutPhrasing);
    assert_eq!(report.eval.n, 10);
}

#[test]
#[ignore = "manual throughput probe, not part of the fast test gate"]
fn throughput_probe_50k() {
    let dataset = Dataset::new(1).unwrap();
    let start = std::time::Instant::now();
    let n = 50_000;
    let examples = dataset.train_examples(n).unwrap();
    let elapsed = start.elapsed();
    eprintln!(
        "{n} examples in {elapsed:?} = {:.0}/sec",
        n as f64 / elapsed.as_secs_f64()
    );
    assert_eq!(examples.len(), n);
}

// ---------------------------------------------------------------------
// Phase 51 P1a-2: procedural catalog scale-up (hundreds-thousands of
// tools, few examples each, arg-copyability by construction).
// ---------------------------------------------------------------------

#[test]
fn procedural_catalog_deterministic_across_construction() {
    let a = grug_data::ToolCatalog::procedural(42, 1000);
    let b = grug_data::ToolCatalog::procedural(42, 1000);
    assert_eq!(a.len(), 1000);
    assert_eq!(b.len(), 1000);
    for (ta, tb) in a.tools().iter().zip(b.tools().iter()) {
        assert_eq!(ta, tb, "same seed must produce byte-identical tool schemas");
    }
}

#[test]
fn procedural_mean_examples_per_tool_stays_low_at_scale() {
    // The whole point of the procedural switch: with 1000 tools and 30k
    // examples, each tool should appear roughly 30 times on average, not the
    // ~500x/tool the 56-hand-authored-tool catalog produced -- well under
    // the 100x/tool threshold where memorization becomes feasible again.
    let n_tools = 1000;
    let n_examples = 30_000;
    let dataset = Dataset::new_sized(7, n_tools, 0.15).unwrap();

    let examples = dataset.train_examples(n_examples).unwrap();
    assert_eq!(examples.len(), n_examples);

    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for ex in &examples {
        *counts.entry(ex.expected_call.name.as_str()).or_insert(0) += 1;
    }

    let train_tool_count = dataset.train_tools().len();
    let mean = n_examples as f64 / train_tool_count as f64;
    assert!(
        mean < 100.0,
        "mean examples/tool = {mean:.1}, expected well under 100 (memorization threshold)"
    );

    // Every observed per-tool count should also individually be nowhere near
    // the old ~500x/tool regime -- a generous 5x-mean ceiling catches a
    // pathologically skewed draw distribution, not just the average.
    let max_count = counts.values().copied().max().unwrap_or(0);
    assert!(
        (max_count as f64) < mean * 5.0 + 20.0,
        "a single tool was drawn {max_count} times, mean is {mean:.1} -- draw is not roughly uniform"
    );

    eprintln!(
        "procedural scale probe: {train_tool_count} train tools, {n_examples} examples, \
         mean={mean:.2}/tool, max={max_count}/tool"
    );
}

#[test]
fn procedural_required_arg_values_are_copyable_from_user_request() {
    // The load-bearing invariant: for a sample of generated examples, every
    // required argument's fuzzed value string must appear verbatim in the
    // rendered `user_request` text. This is what makes argument-copying
    // learnable at all -- P1a's hand-authored catalog already guaranteed
    // this per-tool; here we check it holds at procedural scale too.
    let dataset = Dataset::new_sized(2024, 500, 0.15).unwrap();
    let examples = dataset.train_examples(3000).unwrap();
    assert_eq!(examples.len(), 3000);

    let mut checked_required_args = 0usize;
    for ex in &examples {
        let schema = ex
            .expected_schema()
            .expect("expected tool is always among offered tools");
        let args = ex.expected_call.arguments_map();
        for required in schema.required_params() {
            let value = args
                .get(&required.name)
                .unwrap_or_else(|| panic!("required param {} missing from generated call", required.name));
            let value_text = match value {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Number(n) => n.to_string(),
                other => other.to_string(),
            };
            assert!(
                ex.user_request.contains(&value_text),
                "tool {}: required arg {}={value_text:?} not found verbatim in user_request {:?}",
                schema.name,
                required.name,
                ex.user_request
            );
            checked_required_args += 1;
        }
    }
    assert!(
        checked_required_args > 1000,
        "expected to have checked a substantial number of required args, got {checked_required_args}"
    );

    // Also probe the whole-tool holdout and held-out-phrasing streams --
    // copyability must hold there too since it's a property of the phrasing
    // templates, not of which stream draws from them.
    for holdout_examples in [
        dataset.holdout_examples(300).unwrap(),
        dataset.holdout_phrasing_examples(300).unwrap(),
    ] {
        for ex in &holdout_examples {
            let schema = ex.expected_schema().unwrap();
            let args = ex.expected_call.arguments_map();
            for required in schema.required_params() {
                let value = args.get(&required.name).unwrap();
                let value_text = match value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Bool(b) => b.to_string(),
                    serde_json::Value::Number(n) => n.to_string(),
                    other => other.to_string(),
                };
                assert!(
                    ex.user_request.contains(&value_text),
                    "holdout tool {}: required arg {}={value_text:?} not verbatim in {:?}",
                    schema.name,
                    required.name,
                    ex.user_request
                );
            }
        }
    }
}

#[test]
fn procedural_whole_tool_holdout_disjoint_at_scale() {
    let dataset = Dataset::new_sized(55, 800, 0.15).unwrap();

    let holdout_names: HashSet<&str> = dataset
        .holdout_tools()
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    let train_names: HashSet<&str> = dataset
        .train_tools()
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    assert!(!holdout_names.is_empty());
    assert!(!train_names.is_empty());
    assert!(holdout_names.is_disjoint(&train_names));
    assert_eq!(holdout_names.len() + train_names.len(), 800);

    let train_examples = dataset.train_examples(3000).unwrap();
    for ex in &train_examples {
        assert!(!holdout_names.contains(ex.expected_call.name.as_str()));
        for offered in &ex.tools {
            assert!(!holdout_names.contains(offered.name.as_str()));
        }
    }

    let holdout_examples = dataset.holdout_examples(1000).unwrap();
    for ex in &holdout_examples {
        assert!(!train_names.contains(ex.expected_call.name.as_str()));
        for offered in &ex.tools {
            assert!(!train_names.contains(offered.name.as_str()));
        }
    }
}

#[test]
fn procedural_phrasing_holdout_leak_proof_at_scale() {
    // Structural check first: for every train tool, render every
    // train-phrasing template and every holdout-phrasing template against
    // the same call, and confirm the rendered text never intersects --
    // exactly the guarantee `phrasing_holdout_disjointness_...` checks for
    // the hand-authored catalog, now at procedural scale.
    let dataset = Dataset::new_sized(909, 600, 0.15).unwrap();
    let catalog = grug_data::ToolCatalog::procedural(909, 600);

    let mut checked = 0usize;
    for tool in dataset.train_tools() {
        let bank = catalog.phrasings().get(&tool.name).unwrap();

        let mut args = serde_json::Map::new();
        for p in &tool.params {
            let placeholder = match &p.ty {
                grug_data::ParamType::String => serde_json::Value::String("PLACEHOLDER".into()),
                grug_data::ParamType::Integer => serde_json::Value::from(7),
                grug_data::ParamType::Number => serde_json::Value::from(7.5),
                grug_data::ParamType::Boolean => serde_json::Value::Bool(true),
                grug_data::ParamType::Enum(variants) => {
                    serde_json::Value::String(variants.first().cloned().unwrap_or_default())
                }
            };
            args.insert(p.name.clone(), placeholder);
        }
        let call = grug_data::ToolCall::new(tool.name.clone(), serde_json::Value::Object(args));

        let train_renders: HashSet<String> = bank
            .train_templates()
            .iter()
            .map(|t| t.render(&call))
            .collect();
        let holdout_renders: HashSet<String> = bank
            .holdout_templates()
            .iter()
            .map(|t| t.render(&call))
            .collect();
        assert!(
            train_renders.is_disjoint(&holdout_renders),
            "tool {} has overlapping train/holdout phrasing renders",
            tool.name
        );
        checked += 1;
    }
    assert!(checked > 400, "expected to check most of the 600 train tools, got {checked}");

    // Behavioral check: a large held-out-phrasing stream must never produce
    // a `user_request` seen verbatim in a large training stream.
    let train_examples = dataset.train_examples(5000).unwrap();
    let holdout_phrasing_examples = dataset.holdout_phrasing_examples(5000).unwrap();

    let train_requests: HashSet<&str> = train_examples
        .iter()
        .map(|ex| ex.user_request.as_str())
        .collect();

    let leaked = holdout_phrasing_examples
        .iter()
        .filter(|ex| train_requests.contains(ex.user_request.as_str()))
        .count();
    assert_eq!(
        leaked, 0,
        "{leaked} held-out-phrasing request(s) leaked verbatim into the procedural training stream"
    );
}

#[test]
fn procedural_two_axis_eval_well_formed() {
    let dataset = Dataset::new_sized(321, 400, 0.15).unwrap();

    let (phrasing_report, whole_tool_report) = dataset
        .eval_both_axes(30, |ex| ex.expected_call_as_json_string())
        .unwrap();

    assert_eq!(phrasing_report.axis, EvalAxis::HeldOutPhrasing);
    assert_eq!(whole_tool_report.axis, EvalAxis::WholeToolHoldout);
    assert_eq!(phrasing_report.eval.n, 30);
    assert_eq!(whole_tool_report.eval.n, 30);

    // Perfect model (echoes ground truth) should hit 100% on both axes.
    assert_eq!(phrasing_report.eval.valid_json_rate, 1.0);
    assert_eq!(phrasing_report.eval.schema_adherence_rate, 1.0);
    assert_eq!(phrasing_report.eval.arg_correct_rate, 1.0);
    assert_eq!(whole_tool_report.eval.valid_json_rate, 1.0);
    assert_eq!(whole_tool_report.eval.schema_adherence_rate, 1.0);
    assert_eq!(whole_tool_report.eval.arg_correct_rate, 1.0);
}

#[test]
fn procedural_generation_is_fast_at_moderate_volume() {
    // Not the full 100k throughput probe (kept `#[ignore]`d below for manual
    // runs), but part of the fast gate: confirm procedural generation of a
    // few thousand examples over a sizable (1000-tool) catalog completes
    // comfortably within budget in an unoptimized debug build.
    //
    // Budget note (Phase 51 P1a-3 step 2): `Dataset::new_sized` now defaults
    // to `DistractorConfig::hard()`, which does real per-example work
    // (`select_nth_unstable_by` over the whole train catalog, O(n) average)
    // instead of the old O(distractor_count) uniform index sampling. That
    // moved this measured debug-build cost from ~1-2s to ~25-30s (vs. ~5s in
    // `--release`) -- still linear in n_tools * n_examples with no
    // per-comparison allocation (see `ToolFingerprint`), just a bigger
    // constant factor than before. 40s comfortably covers that in debug
    // builds under normal load; the `--release` companion test below is the
    // tighter, allocation/algorithm regression guard.
    let dataset = Dataset::new_sized(11, 1000, 0.15).unwrap();
    let start = std::time::Instant::now();
    let examples = dataset.train_examples(20_000).unwrap();
    let elapsed = start.elapsed();
    assert_eq!(examples.len(), 20_000);
    assert!(
        elapsed.as_secs() < 40,
        "20k procedural example generation took {elapsed:?}, expected well under 40s (debug build)"
    );
}

#[test]
#[ignore = "tight perf budget only holds in --release; run manually with \
            `cargo test -p grug-data --release -- --ignored hard_distractor_default_scales`"]
fn hard_distractor_default_scales_to_thousands_of_tools_in_release() {
    // The `--release` companion to `procedural_generation_is_fast_at_moderate_volume`:
    // a tighter budget that only holds with an optimized build, so a manual
    // `--release` run catches an algorithmic regression (e.g. accidentally
    // reintroducing an O(n log n) full sort or a per-comparison allocation)
    // that the necessarily-loose debug-mode gate above wouldn't catch.
    // `#[ignore]`d (like the throughput probes below) because plain
    // `cargo test` runs in debug mode, where this budget doesn't hold by
    // design -- not because the code is slow, but because unoptimized Rust
    // is ~5x slower here and this crate's fast gate must stay debug-mode
    // green.
    let dataset = Dataset::new_sized(23, 1500, 0.15).unwrap();
    let start = std::time::Instant::now();
    let examples = dataset.train_examples(20_000).unwrap();
    let elapsed = start.elapsed();
    assert_eq!(examples.len(), 20_000);
    assert!(
        elapsed.as_secs() < 15,
        "20k procedural examples over 1500 tools took {elapsed:?} -- fine in --release, \
         but investigate if this creeps up (possible algorithmic regression)"
    );
}

#[test]
#[ignore = "manual throughput probe, not part of the fast test gate"]
fn procedural_throughput_probe_100k() {
    let dataset = Dataset::new_sized(1, 1000, 0.15).unwrap();
    let start = std::time::Instant::now();
    let n = 100_000;
    let examples = dataset.train_examples(n).unwrap();
    let elapsed = start.elapsed();
    eprintln!(
        "{n} procedural examples in {elapsed:?} = {:.0}/sec",
        n as f64 / elapsed.as_secs_f64()
    );
    assert_eq!(examples.len(), n);
}

// ---------------------------------------------------------------------
// Hard distractors (Phase 51 P1a-3 step 2): force in-context schema-reading
// by offering more, harder-to-rule-out distractor tools, on BOTH the train
// stream and both eval axes.
// ---------------------------------------------------------------------

#[test]
fn hard_distractor_bundles_are_denser_than_the_old_default() {
    // The old default was 2 distractors (3 tools offered); the new default
    // must offer strictly more so there is genuinely more to discriminate.
    let dataset = Dataset::new_sized(1, 500, 0.15).unwrap();
    let examples = dataset.train_examples(50).unwrap();
    for ex in &examples {
        assert!(
            ex.tools.len() > 3,
            "default distractor config should offer more than the old 3-tool bundle, got {}",
            ex.tools.len()
        );
    }
}

#[test]
fn hard_distractor_bundles_contain_same_family_or_similar_shape_tools() {
    // Build a big-enough procedural catalog that every tool has at least a
    // few genuinely similar siblings (shared name token or param-type
    // histogram), then confirm hard-mode's offered bundles actually surface
    // that similarity rather than looking uniformly random.
    let dataset = Dataset::new_sized_with_distractors(
        7,
        800,
        0.15,
        DistractorConfig::hard_with_count(5),
    )
    .unwrap();
    let examples = dataset.train_examples(200).unwrap();

    let mut bundles_with_a_similar_distractor = 0usize;
    for ex in &examples {
        let target = ex.expected_schema().expect("target is among offered tools");
        let target_tokens: HashSet<&str> = target.name.split('_').collect();
        let target_kind_multiset: Vec<&str> = target
            .params
            .iter()
            .map(|p| match p.ty {
                grug_data::ParamType::String => "string",
                grug_data::ParamType::Integer => "integer",
                grug_data::ParamType::Number => "number",
                grug_data::ParamType::Boolean => "boolean",
                grug_data::ParamType::Enum(_) => "enum",
            })
            .collect();

        let has_similar = ex.tools.iter().any(|t| {
            if t.name == target.name {
                return false;
            }
            let shares_name_token = t.name.split('_').any(|tok| target_tokens.contains(tok));
            let same_param_count = t.params.len() == target.params.len();
            let kinds: Vec<&str> = t
                .params
                .iter()
                .map(|p| match p.ty {
                    grug_data::ParamType::String => "string",
                    grug_data::ParamType::Integer => "integer",
                    grug_data::ParamType::Number => "number",
                    grug_data::ParamType::Boolean => "boolean",
                    grug_data::ParamType::Enum(_) => "enum",
                })
                .collect();
            let shares_shape = same_param_count && kinds == target_kind_multiset;
            shares_name_token || shares_shape
        });
        if has_similar {
            bundles_with_a_similar_distractor += 1;
        }
    }

    // Hard mode should surface a similar distractor in the large majority of
    // bundles at this catalog size/distractor count — not every single one
    // (some targets are genuinely unusual), but far more than uniform-random
    // selection would produce by chance.
    assert!(
        bundles_with_a_similar_distractor as f64 / examples.len() as f64 > 0.5,
        "expected hard-mode bundles to mostly contain a same-family/similar-shape distractor, \
         got {bundles_with_a_similar_distractor}/{}",
        examples.len()
    );
}

#[test]
fn distractor_count_is_configurable_on_dataset() {
    for count in [1usize, 3, 6, 10] {
        let dataset = Dataset::new_sized_with_distractors(
            2,
            500,
            0.15,
            DistractorConfig::hard_with_count(count),
        )
        .unwrap();
        let examples = dataset.train_examples(10).unwrap();
        for ex in &examples {
            assert_eq!(ex.tools.len(), count + 1, "target + {count} distractors expected");
        }
    }
}

#[test]
fn easy_distractor_mode_is_still_reachable_for_ab_comparison() {
    let dataset =
        Dataset::new_sized_with_distractors(3, 500, 0.15, DistractorConfig::easy()).unwrap();
    let examples = dataset.train_examples(10).unwrap();
    for ex in &examples {
        assert_eq!(ex.tools.len(), DistractorConfig::easy().distractor_count + 1);
    }
}

#[test]
fn train_and_both_eval_axes_use_the_same_hard_distractor_bundle_size() {
    // The training bin's config must apply uniformly: train, axis-A
    // (whole-tool holdout), and axis-B (held-out-phrasing) should all offer
    // the same bundle width, so axis-A eval genuinely tests the same
    // schema-discrimination task the model was trained on.
    let config = DistractorConfig::hard_with_count(5);
    let dataset = Dataset::new_sized_with_distractors(9, 600, 0.15, config).unwrap();

    let train = dataset.train_examples(20).unwrap();
    let axis_a = dataset.holdout_examples(20).unwrap();
    let axis_b = dataset.holdout_phrasing_examples(20).unwrap();

    for ex in train.iter().chain(axis_a.iter()).chain(axis_b.iter()) {
        assert_eq!(ex.tools.len(), 6, "target + 5 distractors on every stream");
    }
}

#[test]
fn hard_distractors_preserve_whole_tool_holdout_disjointness() {
    // Re-run the pre-existing disjointness guarantee under hard-distractor
    // config with a wider bundle, since hard-mode selection is a new code
    // path that must still only ever draw from the generator's own (train or
    // holdout) catalog.
    let dataset = Dataset::new_sized_with_distractors(
        13,
        400,
        0.2,
        DistractorConfig::hard_with_count(6),
    )
    .unwrap();

    let holdout_names: HashSet<&str> = dataset.holdout_tools().iter().map(|t| t.name.as_str()).collect();
    let train_names: HashSet<&str> = dataset.train_tools().iter().map(|t| t.name.as_str()).collect();
    assert!(holdout_names.is_disjoint(&train_names));

    let train_examples = dataset.train_examples(300).unwrap();
    for ex in &train_examples {
        for offered in &ex.tools {
            assert!(
                !holdout_names.contains(offered.name.as_str()),
                "holdout tool {} leaked into training offered-tools bundle under hard distractors",
                offered.name
            );
        }
    }

    let holdout_examples = dataset.holdout_examples(150).unwrap();
    for ex in &holdout_examples {
        for offered in &ex.tools {
            assert!(
                !train_names.contains(offered.name.as_str()),
                "train tool {} leaked into holdout offered-tools bundle under hard distractors",
                offered.name
            );
        }
    }
}

#[test]
fn hard_distractors_preserve_arg_copyability_and_target_presence() {
    // Every required arg must still appear verbatim in the rendered request
    // (arg-copyability), and the target tool must still be the one whose
    // call is expected, regardless of how many/which distractors surround it.
    let dataset = Dataset::new_sized_with_distractors(
        21,
        400,
        0.15,
        DistractorConfig::hard_with_count(7),
    )
    .unwrap();
    let examples = dataset.train_examples(200).unwrap();

    for ex in &examples {
        let schema = ex
            .expected_schema()
            .expect("expected tool must be among the offered tools");
        assert_eq!(schema.name, ex.expected_call.name);

        for p in schema.required_params() {
            if let Some(v) = ex.expected_call.arguments_map().get(&p.name) {
                let text = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                assert!(
                    ex.user_request.contains(&text),
                    "required arg {}={:?} not copyable from rendered request {:?} \
                     (tool {}, {} distractors offered)",
                    p.name,
                    v,
                    ex.user_request,
                    schema.name,
                    ex.tools.len() - 1
                );
            }
        }
    }
}

#[test]
fn hard_distractor_determinism_holds_at_dataset_level() {
    let cfg = DistractorConfig::hard_with_count(5);
    let ds_a = Dataset::new_sized_with_distractors(444, 700, 0.15, cfg).unwrap();
    let ds_b = Dataset::new_sized_with_distractors(444, 700, 0.15, cfg).unwrap();

    let train_a = ds_a.train_examples(40).unwrap();
    let train_b = ds_b.train_examples(40).unwrap();
    assert_eq!(train_a, train_b);

    let axis_a_a = ds_a.holdout_examples(20).unwrap();
    let axis_a_b = ds_b.holdout_examples(20).unwrap();
    assert_eq!(axis_a_a, axis_a_b);

    let axis_b_a = ds_a.holdout_phrasing_examples(20).unwrap();
    let axis_b_b = ds_b.holdout_phrasing_examples(20).unwrap();
    assert_eq!(axis_b_a, axis_b_b);
}

#[test]
#[ignore = "manual stats probe, not part of the fast test gate"]
fn stats_probe_param_shapes() {
    let catalog = ToolCatalog::built_in();
    let tools = catalog.tools();
    let total = tools.len();
    let enums = tools.iter().filter(|t| t.params.iter().any(|p| matches!(p.ty, grug_data::ParamType::Enum(_)))).count();
    let optionals = tools.iter().filter(|t| t.params.iter().any(|p| !p.required)).count();
    let many = tools.iter().filter(|t| t.params.len() >= 5).count();
    let one = tools.iter().filter(|t| t.params.len() == 1).count();
    let avg_train: f64 = tools.iter().map(|t| catalog.phrasings().get(&t.name).unwrap().train_count() as f64).sum::<f64>() / total as f64;
    let avg_holdout: f64 = tools.iter().map(|t| catalog.phrasings().get(&t.name).unwrap().holdout_count() as f64).sum::<f64>() / total as f64;
    eprintln!("total={total} enums={enums} optionals={optionals} many_params(>=5)={many} one_param={one} avg_train_templates={avg_train:.2} avg_holdout_templates={avg_holdout:.2}");
}
