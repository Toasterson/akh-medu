//! Phase 41g criterion benchmarks: dataset load, single-prompt pipeline, metrics emit.
//!
//! Run with:
//!   cargo bench --bench phase41g
//!
//! The full e2e sweep (all prompts × all step budgets) is too slow for criterion;
//! run it as an integration test or via `cargo test bench_phase41g -- --ignored`.

use std::path::Path;

use criterion::{Criterion, black_box, criterion_group, criterion_main};

use akh_medu::bench_phase41g::{
    DATASET_PATH,
    AcceptancePredicate, BenchPrompt, PromptCategory,
    dataset::load_dataset,
    metrics::{AgreementRecord, MetricsSummary, emit_agreement_csv, emit_csv, emit_plot_script},
    pipeline::{PipelineConfig, run_pipeline},
};

// ---------------------------------------------------------------------------
// Dataset load
// ---------------------------------------------------------------------------

fn bench_dataset_load(c: &mut Criterion) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(DATASET_PATH);
    c.bench_function("phase41g_dataset_load", |b| {
        b.iter(|| {
            let dataset = load_dataset(black_box(&path)).expect("dataset must load");
            black_box(dataset.len())
        })
    });
}

// ---------------------------------------------------------------------------
// Single-prompt pipeline (canonical worked example)
// ---------------------------------------------------------------------------

fn canonical_prompt() -> BenchPrompt {
    BenchPrompt {
        id: "canonical_worked_example".into(),
        category: PromptCategory::CodeAdd,
        prompt:
            "Add a Rust struct Foo with proper error handling using miette and unit tests.".into(),
        acceptance: vec![
            AcceptancePredicate::OutputContains {
                pattern: "struct Foo".into(),
            },
            AcceptancePredicate::PlanHasMinSteps { min_steps: 3 },
        ],
    }
}

fn bench_stub_pipeline_100(c: &mut Criterion) {
    let prompt = canonical_prompt();
    let config = PipelineConfig::new(100);
    c.bench_function("phase41g_pipeline_steps100", |b| {
        b.iter(|| black_box(run_pipeline(black_box(&prompt), black_box(&config))))
    });
}

fn bench_stub_pipeline_1000(c: &mut Criterion) {
    let prompt = canonical_prompt();
    let config = PipelineConfig::new(1_000);
    c.bench_function("phase41g_pipeline_steps1000", |b| {
        b.iter(|| black_box(run_pipeline(black_box(&prompt), black_box(&config))))
    });
}

fn bench_stub_pipeline_all_categories(c: &mut Criterion) {
    let prompts: Vec<BenchPrompt> = [
        (PromptCategory::CodeAdd, "Add struct Foo with error handling."),
        (PromptCategory::CodeEdit, "Change parse_config to use miette errors."),
        (PromptCategory::Refactor, "Extract validate() from save()."),
        (PromptCategory::DocWrite, "Write rustdoc for the storage module."),
        (PromptCategory::Debug, "Fix index out of bounds in parse_headers."),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (cat, prompt))| BenchPrompt {
        id: format!("cat_bench_{i}"),
        category: cat,
        prompt: prompt.into(),
        acceptance: vec![AcceptancePredicate::PlanHasMinSteps { min_steps: 1 }],
    })
    .collect();

    let config = PipelineConfig::new(300);
    c.bench_function("phase41g_pipeline_all_categories", |b| {
        b.iter(|| {
            for p in &prompts {
                black_box(run_pipeline(black_box(p), black_box(&config)));
            }
        })
    });
}

// ---------------------------------------------------------------------------
// Metrics emission
// ---------------------------------------------------------------------------

fn bench_metrics_emit(c: &mut Criterion) {
    let prompt = canonical_prompt();
    let results: Vec<_> = [100u64, 300, 1_000, 3_000, 10_000]
        .iter()
        .map(|&budget| run_pipeline(&prompt, &PipelineConfig::new(budget)))
        .collect();

    let summary = MetricsSummary::from_results(&results);

    c.bench_function("phase41g_metrics_summary", |b| {
        b.iter(|| {
            black_box(MetricsSummary::from_results(black_box(&results)));
        })
    });

    let dir = tempfile::tempdir().unwrap();
    c.bench_function("phase41g_emit_csv", |b| {
        b.iter(|| emit_csv(black_box(&results), black_box(dir.path())).unwrap())
    });

    let _ = summary;
}

fn bench_agreement_matrix(c: &mut Criterion) {
    let prompt = canonical_prompt();
    let symbolic = run_pipeline(&prompt, &PipelineConfig::new(1_000));

    // Pattern B is skipped (no API key in CI), so we fabricate a skipped result.
    let pb = akh_medu::bench_phase41g::PatternBResult {
        prompt_id: symbolic.prompt_id.clone(),
        model: "anthropic/claude-opus-4-5".into(),
        success: false,
        wall_ms: 0,
        tokens_used: 0,
        skipped: true,
        thinking_text: String::new(),
        answer_text: String::new(),
    };
    let records = vec![AgreementRecord::from_pair(&symbolic, &pb)];

    let dir = tempfile::tempdir().unwrap();
    c.bench_function("phase41g_emit_agreement_csv", |b| {
        b.iter(|| emit_agreement_csv(black_box(&records), black_box(dir.path())).unwrap())
    });

    emit_plot_script(dir.path()).unwrap();
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    dataset_benches,
    bench_dataset_load,
);

criterion_group!(
    pipeline_benches,
    bench_stub_pipeline_100,
    bench_stub_pipeline_1000,
    bench_stub_pipeline_all_categories,
);

criterion_group!(
    metrics_benches,
    bench_metrics_emit,
    bench_agreement_matrix,
);

criterion_main!(dataset_benches, pipeline_benches, metrics_benches);
