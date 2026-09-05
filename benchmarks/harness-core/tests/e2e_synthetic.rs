//! End-to-end verification: write 2-3 synthetic `RunRecord`s (deepseek, glm,
//! qwen on a fake "demo" harness), generate the leaderboard markdown AND the
//! HTML site from them, and assert both contain the expected rows — plus
//! confirm the HTML has zero external asset references, matching the W7
//! "verification deliverable" requirement.

use harness_core::hash::{TaskFile, hash_task_set};
use harness_core::schema::{
    GitSha, GradingMode, HarnessId, HostTier, JudgeInfo, ModelBacking, ModelIdentity, ModelTag,
    ResourceUsage, RunRecord, TaskId, TaskOutcome, TaskResult, WallTimeSecs,
};
use harness_core::{leaderboard, site, store};

fn task(id: &str, pass: bool, score: f64) -> TaskResult {
    TaskResult::new(
        TaskId::new(id).unwrap(),
        if pass {
            TaskOutcome::Pass
        } else {
            TaskOutcome::Fail
        },
        score,
        WallTimeSecs(2.0),
        ResourceUsage {
            tokens_used: Some(500),
            requests_used: Some(1),
        },
        None,
    )
    .unwrap()
}

fn demo_run(model: &str, day: &str, grading: GradingMode, scores: &[(bool, f64)]) -> RunRecord {
    let task_files: Vec<TaskFile> = (0..scores.len())
        .map(|i| TaskFile {
            relative_path: format!("task-{i:02}.md"),
            content: b"demo task".to_vec(),
        })
        .collect();
    let task_set_hash = hash_task_set(&task_files);

    let tasks = scores
        .iter()
        .enumerate()
        .map(|(i, (pass, score))| task(&format!("task-{i:02}"), *pass, *score))
        .collect();

    RunRecord::new(
        format!("{day}T00:00:00Z").parse().unwrap(),
        HarnessId::new("demo").unwrap(),
        GitSha::new("deadbeef00deadbeef00deadbeef00deadbeef0").unwrap(),
        task_set_hash,
        ModelIdentity {
            tag: ModelTag::new(model).unwrap(),
            backing: ModelBacking::Cloud {
                provider_tag: format!("{model}-cloud-tag"),
            },
        },
        grading,
        HostTier::Host,
        tasks,
    )
}

#[test]
fn synthetic_runs_flow_through_store_leaderboard_and_site() {
    let results_dir = tempfile::tempdir().unwrap();

    // Three models on the same fake "demo" harness, two runs for deepseek
    // (older + newer) to exercise "latest wins" and give the trend chart
    // more than one point.
    let deepseek_old = demo_run(
        "deepseek",
        "2026-06-20",
        GradingMode::Autonomous,
        &[(true, 1.0), (false, 0.0), (true, 0.6)],
    );
    let deepseek_new = demo_run(
        "deepseek",
        "2026-07-01",
        GradingMode::Autonomous,
        &[(true, 1.0), (true, 0.9), (true, 0.6)],
    );
    let glm = demo_run(
        "glm-5.2",
        "2026-07-01",
        GradingMode::Advisory(JudgeInfo {
            judge_model: ModelTag::new("qwen").unwrap(),
            rotation_seed: 7,
        }),
        &[(true, 1.0), (true, 0.8), (false, 0.2)],
    );
    let qwen = demo_run(
        "qwen",
        "2026-07-01",
        GradingMode::Autonomous,
        &[(false, 0.1), (false, 0.0), (true, 1.0)],
    );

    for record in [&deepseek_old, &deepseek_new, &glm, &qwen] {
        let dir = store::write_run(results_dir.path(), record).expect("write_run");
        assert!(dir.join("run.jsonl").is_file());
        assert!(dir.join("summary.md").is_file());
    }

    // --- storage round trip ---
    let runs = store::scan_results(results_dir.path()).expect("scan_results");
    assert_eq!(runs.len(), 4, "all four written runs should be found");

    let latest = store::latest_per_pair(&runs);
    assert_eq!(
        latest.len(),
        3,
        "one row per (harness, model) — deepseek collapses to its newer run"
    );
    let deepseek_key = (
        HarnessId::new("demo").unwrap(),
        ModelTag::new("deepseek").unwrap(),
    );
    assert_eq!(
        latest.get(&deepseek_key).unwrap().header.timestamp,
        deepseek_new.header.timestamp,
        "latest_per_pair must keep the newer deepseek run, not the older one"
    );

    // --- leaderboard markdown ---
    let md = leaderboard::render_leaderboard_markdown(&latest);
    assert!(
        md.contains("| demo | deepseek |"),
        "leaderboard missing deepseek row:\n{md}"
    );
    assert!(
        md.contains("| demo | glm-5.2 |"),
        "leaderboard missing glm-5.2 row:\n{md}"
    );
    assert!(
        md.contains("| demo | qwen |"),
        "leaderboard missing qwen row:\n{md}"
    );
    // deepseek's newer run is 3/3 passed = 100.0%.
    assert!(
        md.contains("100.0%"),
        "expected the newer all-pass deepseek run to show 100.0%:\n{md}"
    );
    // advisory grading on glm shows the rotation judge.
    assert!(
        md.contains("advisory (judge `qwen`)"),
        "leaderboard missing advisory judge annotation:\n{md}"
    );

    println!("--- generated LEADERBOARD.md ---\n{md}");

    // --- HTML site ---
    let html = site::render_html_site(&latest, &runs);
    for model in ["deepseek", "glm-5.2", "qwen"] {
        assert!(html.contains(model), "site missing model {model}");
    }
    assert!(
        html.contains("<svg"),
        "site should contain at least one inline SVG trend chart"
    );
    assert!(
        html.contains("<table"),
        "site should contain the leaderboard table"
    );

    // Zero external assets: no network-loaded resource of any kind, and no
    // script tags (this is a static, self-contained document).
    for needle in ["http://", "https://", "src=", "<script"] {
        assert!(
            !html.contains(needle),
            "site unexpectedly references an external asset via {needle:?}"
        );
    }

    // Confirm it's at least well-formed enough to be real markup: doctype,
    // one <html>, one <head>, one <body>, balanced-looking tag counts for
    // the elements we emit ourselves.
    assert!(html.starts_with("<!doctype html>"));
    assert_eq!(html.matches("<html").count(), 1);
    assert_eq!(html.matches("</html>").count(), 1);
    assert_eq!(html.matches("<head>").count(), 1);
    assert_eq!(html.matches("</head>").count(), 1);
    assert_eq!(html.matches("<body>").count(), 1);
    assert_eq!(html.matches("</body>").count(), 1);
    assert_eq!(
        html.matches("<table>").count(),
        html.matches("</table>").count()
    );
    assert_eq!(html.matches("<svg").count(), html.matches("</svg>").count());

    println!(
        "site has zero external asset references ({} bytes)",
        html.len()
    );
}
