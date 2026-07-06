//! Integration tests for `grug-data`: determinism, holdout disjointness (both
//! axes), eval-scorer correctness at each gated level, and the anti-collapse
//! alarm.

use std::collections::HashSet;

use grug_data::eval::{score_output, ArgMismatch, DiversityReport, EvalReport, SchemaViolation};
use grug_data::{Dataset, EvalAxis, Example, Generator, ToolCatalog};

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
