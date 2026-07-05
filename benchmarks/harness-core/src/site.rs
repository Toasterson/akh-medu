//! Self-contained single-page HTML site generator.
//!
//! Hard constraints (ADR 086 §4 + Phase 50 W7 scope): inline CSS, inline
//! hand-emitted SVG trend charts, **no JavaScript, no CDN, no external
//! fonts/assets**. Every byte the page needs ships inside the one file this
//! module returns; `render_html_site` is a pure function of the runs passed
//! in, so regenerating from the same committed `benchmarks/results/` tree is
//! deterministic.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use chrono::{DateTime, Utc};

use crate::schema::{GradingMode, HarnessId, HostTier, ModelBacking, ModelTag, RunRecord};

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Render the full self-contained site. `latest` selects which
/// `(harness, model)` pairs get a leaderboard row (see
/// [`crate::store::latest_per_pair`]); `all_runs` supplies the full history
/// used to draw each pair's trend chart.
pub fn render_html_site(latest: &BTreeMap<(HarnessId, ModelTag), RunRecord>, all_runs: &[RunRecord]) -> String {
    let mut out = String::new();
    out.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n");
    out.push_str("<meta charset=\"utf-8\">\n");
    out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
    out.push_str("<title>akh-medu benchmark leaderboard</title>\n");
    out.push_str(&format!("<style>\n{}\n</style>\n", STYLE));
    out.push_str("</head>\n<body>\n");

    out.push_str("<main>\n");
    out.push_str("<h1>Benchmark leaderboard</h1>\n");
    out.push_str(
        "<p class=\"muted\">Generated from the latest committed run per (harness, model) under \
         <code>benchmarks/results/</code>. Regenerate with <code>bench-tool site</code> — do not \
         hand-edit this file.</p>\n",
    );

    render_leaderboard_table(&mut out, latest);
    render_trend_section(&mut out, latest, all_runs);

    out.push_str("</main>\n");
    out.push_str("</body>\n</html>\n");
    out
}

fn render_leaderboard_table(out: &mut String, latest: &BTreeMap<(HarnessId, ModelTag), RunRecord>) {
    out.push_str("<section>\n<h2>Latest runs</h2>\n");
    if latest.is_empty() {
        out.push_str("<p>No runs recorded yet.</p>\n</section>\n");
        return;
    }

    out.push_str("<table>\n<thead><tr>");
    for col in [
        "Harness",
        "Model",
        "Backing",
        "Grading",
        "Host tier",
        "Date (UTC)",
        "Tasks",
        "Pass rate",
        "Mean score",
        "Task-set hash",
        "Harness SHA",
    ] {
        let _ = write!(out, "<th>{col}</th>");
    }
    out.push_str("</tr></thead>\n<tbody>\n");

    for ((harness, model_tag), record) in latest {
        let summary = record.summary();
        let backing = match &record.header.model.backing {
            ModelBacking::Cloud { provider_tag } => format!("cloud (<code>{}</code>)", escape_html(provider_tag)),
            ModelBacking::LocalGguf { file_name, file_hash } => format!(
                "local GGUF <code>{}</code> (<code>{}</code>)",
                escape_html(file_name),
                file_hash.short()
            ),
        };
        let grading = match &record.header.grading {
            GradingMode::Autonomous => "autonomous".to_string(),
            GradingMode::Advisory(judge) => {
                format!("advisory (judge <code>{}</code>)", escape_html(judge.judge_model.as_str()))
            }
        };
        let host_tier = match &record.header.host_tier {
            HostTier::Host => "host".to_string(),
            HostTier::Vm(image) => format!("VM (<code>{}</code>)", escape_html(&image.registry)),
        };

        out.push_str("<tr>");
        let _ = write!(out, "<td>{}</td>", escape_html(harness.as_str()));
        let _ = write!(out, "<td>{}</td>", escape_html(model_tag.as_str()));
        let _ = write!(out, "<td>{backing}</td>");
        let _ = write!(out, "<td>{grading}</td>");
        let _ = write!(out, "<td>{host_tier}</td>");
        let _ = write!(out, "<td>{}</td>", record.header.timestamp.format("%Y-%m-%d"));
        let _ = write!(out, "<td>{}/{}</td>", summary.passed, summary.total_tasks);
        let _ = write!(out, "<td>{:.1}%</td>", summary.pass_rate * 100.0);
        let _ = write!(out, "<td>{:.3}</td>", summary.mean_score);
        let _ = write!(out, "<td><code>{}</code></td>", record.header.task_set_hash.short());
        let _ = write!(out, "<td><code>{}</code></td>", record.header.harness_git_sha.short());
        out.push_str("</tr>\n");
    }

    out.push_str("</tbody>\n</table>\n</section>\n");
}

fn render_trend_section(
    out: &mut String,
    latest: &BTreeMap<(HarnessId, ModelTag), RunRecord>,
    all_runs: &[RunRecord],
) {
    out.push_str("<section>\n<h2>Trends</h2>\n");
    if latest.is_empty() {
        out.push_str("<p>No history to chart yet.</p>\n</section>\n");
        return;
    }
    out.push_str("<div class=\"chart-grid\">\n");

    for (harness, model_tag) in latest.keys() {
        let mut series: Vec<(DateTime<Utc>, f64)> = all_runs
            .iter()
            .filter(|r| &r.header.harness == harness && &r.header.model.tag == model_tag)
            .map(|r| (r.header.timestamp, r.summary().mean_score))
            .collect();
        series.sort_by_key(|(ts, _)| *ts);

        out.push_str("<div class=\"chart\">\n");
        let _ = write!(
            out,
            "<h3>{} &times; {}</h3>\n",
            escape_html(harness.as_str()),
            escape_html(model_tag.as_str())
        );
        out.push_str(&render_trend_svg(&series));
        let _ = write!(
            out,
            "<p class=\"muted small\">{} run(s), mean score over time (0.0&ndash;1.0)</p>\n",
            series.len()
        );
        out.push_str("</div>\n");
    }

    out.push_str("</div>\n</section>\n");
}

/// Hand-emit an inline SVG line chart of `points` (already sorted by time).
/// No JS: every visual detail is baked into static markup at generation
/// time.
fn render_trend_svg(points: &[(DateTime<Utc>, f64)]) -> String {
    if points.is_empty() {
        return "<p class=\"muted small\">no data</p>\n".to_string();
    }

    let width = 320.0f64;
    let height = 120.0f64;
    let pad = 26.0f64;
    let plot_w = width - 2.0 * pad;
    let plot_h = height - 2.0 * pad;
    let n = points.len();

    let x_at = |i: usize| -> f64 {
        if n > 1 {
            pad + (i as f64 / (n - 1) as f64) * plot_w
        } else {
            pad + plot_w / 2.0
        }
    };
    let y_at = |score: f64| -> f64 { pad + (1.0 - score.clamp(0.0, 1.0)) * plot_h };

    let mut svg = String::new();
    let _ = write!(
        svg,
        "<svg viewBox=\"0 0 {width} {height}\" width=\"100%\" height=\"{height}\" role=\"img\" \
         aria-label=\"trend chart, mean score over {n} run(s)\">\n"
    );

    // Gridlines at score 0.0 and 1.0.
    let y_top = y_at(1.0);
    let y_bot = y_at(0.0);
    let _ = write!(
        svg,
        "<line x1=\"{pad}\" y1=\"{y_top:.1}\" x2=\"{x_end:.1}\" y2=\"{y_top:.1}\" style=\"stroke:var(--border)\" stroke-width=\"1\"/>\n",
        x_end = width - pad
    );
    let _ = write!(
        svg,
        "<line x1=\"{pad}\" y1=\"{y_bot:.1}\" x2=\"{x_end:.1}\" y2=\"{y_bot:.1}\" style=\"stroke:var(--border)\" stroke-width=\"1\"/>\n",
        x_end = width - pad
    );
    let _ = write!(svg, "<text x=\"2\" y=\"{:.1}\" font-size=\"9\" style=\"fill:var(--muted)\">1.0</text>\n", y_top + 3.0);
    let _ = write!(svg, "<text x=\"2\" y=\"{:.1}\" font-size=\"9\" style=\"fill:var(--muted)\">0.0</text>\n", y_bot + 3.0);

    if n > 1 {
        let poly_points: String = points
            .iter()
            .enumerate()
            .map(|(i, (_, score))| format!("{:.1},{:.1}", x_at(i), y_at(*score)))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = write!(
            svg,
            "<polyline points=\"{poly_points}\" fill=\"none\" style=\"stroke:var(--accent)\" stroke-width=\"2\"/>\n"
        );
    }

    for (i, (_, score)) in points.iter().enumerate() {
        let _ = write!(
            svg,
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"2.5\" style=\"fill:var(--accent)\"/>\n",
            x_at(i),
            y_at(*score)
        );
    }

    let first_date = points.first().unwrap().0.format("%Y-%m-%d");
    let last_date = points.last().unwrap().0.format("%Y-%m-%d");
    let _ = write!(
        svg,
        "<text x=\"{pad}\" y=\"{y:.1}\" font-size=\"9\" style=\"fill:var(--muted)\">{first_date}</text>\n",
        y = height - 4.0
    );
    if n > 1 {
        let _ = write!(
            svg,
            "<text x=\"{x_end:.1}\" y=\"{y:.1}\" font-size=\"9\" text-anchor=\"end\" style=\"fill:var(--muted)\">{last_date}</text>\n",
            x_end = width - pad,
            y = height - 4.0
        );
    }

    svg.push_str("</svg>\n");
    svg
}

const STYLE: &str = r#"
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --accent: #2b6cb0;
  --border: #d0d5dd;
  --muted: #6b7280;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115;
    --fg: #e5e7eb;
    --accent: #7dd3fc;
    --border: #333844;
    --muted: #9ca3af;
  }
}
* { box-sizing: border-box; }
body {
  background: var(--bg);
  color: var(--fg);
  font-family: ui-sans-serif, system-ui, sans-serif;
  margin: 0;
  padding: 24px;
  line-height: 1.5;
}
main { max-width: 960px; margin: 0 auto; }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 17px; margin-top: 28px; }
h3 { font-size: 13px; margin: 0 0 8px 0; }
p.muted, .muted { color: var(--muted); font-size: 13px; }
.small { font-size: 11px; }
code {
  font-family: ui-monospace, SFMono-Regular, monospace;
  font-size: 0.92em;
  background: rgba(127, 127, 127, 0.12);
  padding: 1px 4px;
  border-radius: 3px;
}
table { border-collapse: collapse; width: 100%; margin-top: 8px; }
th, td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; font-size: 13px; }
th { background: rgba(127, 127, 127, 0.08); }
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
  margin-top: 8px;
}
.chart { border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
.chart svg { display: block; }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        Blake3Hex, GitSha, GradingMode as GM, ModelIdentity, ResourceUsage, TaskId, TaskOutcome, TaskResult,
        TaskSetHash, WallTimeSecs,
    };

    fn record(harness: &str, model: &str, day: &str, score: f64) -> RunRecord {
        RunRecord::new(
            format!("{day}T00:00:00Z").parse().unwrap(),
            HarnessId::new(harness).unwrap(),
            GitSha::new("a".repeat(40)).unwrap(),
            TaskSetHash::from_blake3(Blake3Hex::from_hash(blake3::hash(b"tasks"))),
            ModelIdentity {
                tag: ModelTag::new(model).unwrap(),
                backing: ModelBacking::Cloud {
                    provider_tag: format!("{model}-cloud"),
                },
            },
            GM::Autonomous,
            HostTier::Host,
            vec![TaskResult::new(
                TaskId::new("t1").unwrap(),
                TaskOutcome::Pass,
                score,
                WallTimeSecs(1.0),
                ResourceUsage::none(),
                None,
            )
            .unwrap()],
        )
    }

    #[test]
    fn site_has_no_external_asset_references() {
        let all = vec![
            record("demo", "deepseek", "2026-07-01", 0.5),
            record("demo", "deepseek", "2026-07-05", 0.8),
        ];
        let latest = crate::store::latest_per_pair(&all);
        let html = render_html_site(&latest, &all);

        for needle in ["http://", "https://", "src="] {
            assert!(!html.contains(needle), "html unexpectedly contains {needle:?}");
        }
        assert!(!html.contains("<script"));
        assert!(html.contains("<svg"));
        assert!(html.contains("deepseek"));
    }

    #[test]
    fn escapes_untrusted_strings() {
        let mut r = record("demo", "deepseek", "2026-07-01", 1.0);
        r.header.model.backing = ModelBacking::Cloud {
            provider_tag: "<script>alert(1)</script>".to_string(),
        };
        let all = vec![r];
        let latest = crate::store::latest_per_pair(&all);
        let html = render_html_site(&latest, &all);
        assert!(!html.contains("<script>alert"));
        assert!(html.contains("&lt;script&gt;"));
    }
}
