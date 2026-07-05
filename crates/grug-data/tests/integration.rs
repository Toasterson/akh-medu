//! Integration tests for `grug-data`: determinism, holdout disjointness,
//! eval-scorer correctness at each gated level, and the anti-collapse alarm.

use grug_data::eval::{score_output, ArgMismatch, DiversityReport, EvalReport, SchemaViolation};
use grug_data::{Dataset, Example, Generator, ToolCatalog};

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
