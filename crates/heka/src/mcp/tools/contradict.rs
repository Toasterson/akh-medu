//! `heka_contradict` — ACH-driven contradiction analyzer for memory drawers.
//!
//! # Summary
//!
//! Given two drawer IDs, this tool:
//! 1. Loads both drawers from the Heka store.
//! 2. Constructs an ACH (Analysis of Competing Hypotheses) tableau:
//!    - H0 = drawer A's claim, H1 = drawer B's claim.
//!    - Evidence = self-assertions + temporal recency + KG facts (if engine is available).
//! 3. Runs `AchAnalysis::evaluate()` to score and rank hypotheses.
//! 4. Converts ACH consistency ratings → Dempster-Shafer mass functions →
//!    belief intervals via Dempster's rule (Phase 17).
//! 5. Returns per-side confidences, ranks, and the deciding-evidence chain.
//!
//! # Demo
//!
//! ```text
//! heka_remember(subject="temperature", claim="temperature is 37 degrees", ...)
//! heka_remember(subject="temperature", claim="temperature is 22 degrees", ...)
//! heka_contradict(drawer_a="1", drawer_b="2")
//! ```
//!
//! Returns Dempster-Shafer [Bel, Pl] intervals for each side, the deciding
//! evidence chain with drawer/KG citations, and the full ACH matrix.
//!
//! Anima WI: `019dd0c7-2557-7e40-874d-d3c99af2e239` (Phase 42i)

use akh_medu::reasoning::ach_for_drawers::{
    DrawerAchConfig, DrawerClaim, DrawerAchResult, analyze_drawer_conflict,
};
use rmcp::model::{CallToolResult, Content};
use rmcp::ErrorData as McpError;
use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;

use crate::drawer::DrawerStore;

// ═══════════════════════════════════════════════════════════════════════
// Parameter type
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HekaContradictParams {
    #[schemars(
        description = "ID of the first drawer (the claim you want to evaluate as H0)"
    )]
    pub drawer_a: String,

    #[schemars(
        description = "ID of the second drawer (the claim you want to evaluate as H1)"
    )]
    pub drawer_b: String,

    #[schemars(
        description = "Maximum KG evidence items to include (default: 5, 0 = skip KG lookup)"
    )]
    pub kg_limit: Option<usize>,
}

// ═══════════════════════════════════════════════════════════════════════
// Core logic (sync, usable from both MCP handler and CLI)
// ═══════════════════════════════════════════════════════════════════════

/// Run the full ACH contradiction analysis and format it as a human-readable report.
///
/// `engine` is optional; when `None` the KG evidence step is skipped.
pub fn run_contradict(
    store: &DrawerStore,
    params: &HekaContradictParams,
    engine: Option<&akh_medu::engine::Engine>,
) -> Result<String, McpError> {
    // Parse drawer IDs (accept either numeric or UUID-like strings).
    let id_a = parse_drawer_id(&params.drawer_a)?;
    let id_b = parse_drawer_id(&params.drawer_b)?;

    // Load both drawers.
    let rec_a = store
        .get(id_a)
        .map_err(|e| McpError::invalid_params(format!("drawer_a: {e}"), None))?;
    let rec_b = store
        .get(id_b)
        .map_err(|e| McpError::invalid_params(format!("drawer_b: {e}"), None))?;

    // Build bridge claims.
    let claim_a = DrawerClaim {
        drawer_id: rec_a.id.to_string(),
        claim: rec_a.claim.clone(),
        source: rec_a.source.clone(),
        timestamp: rec_a.timestamp,
        confidence: rec_a.confidence,
    };
    let claim_b = DrawerClaim {
        drawer_id: rec_b.id.to_string(),
        claim: rec_b.claim.clone(),
        source: rec_b.source.clone(),
        timestamp: rec_b.timestamp,
        confidence: rec_b.confidence,
    };

    let config = DrawerAchConfig {
        kg_evidence_limit: params.kg_limit.unwrap_or(5),
        max_deciding: 5,
    };

    let result = analyze_drawer_conflict(&claim_a, &claim_b, engine, &config);

    Ok(format_report(&result))
}

// ═══════════════════════════════════════════════════════════════════════
// MCP async wrapper
// ═══════════════════════════════════════════════════════════════════════

/// Async MCP handler — offloads the blocking ACH engine to a spawn_blocking task.
pub async fn heka_contradict_handler(
    store: Arc<DrawerStore>,
    engine: Option<Arc<akh_medu::engine::Engine>>,
    params: HekaContradictParams,
) -> Result<CallToolResult, McpError> {
    let result = tokio::task::spawn_blocking(move || {
        run_contradict(
            &store,
            &params,
            engine.as_deref(),
        )
    })
    .await
    .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))??;

    Ok(CallToolResult::success(vec![Content::text(result)]))
}

// ═══════════════════════════════════════════════════════════════════════
// Report formatter
// ═══════════════════════════════════════════════════════════════════════

fn format_report(r: &DrawerAchResult) -> String {
    let mut lines: Vec<String> = Vec::new();

    lines.push("═══════════════════════════════════════════════════".into());
    lines.push("heka_contradict — ACH Contradiction Analysis".into());
    lines.push("═══════════════════════════════════════════════════".into());
    lines.push(format!("Analysis ID : {}", r.analysis_id));
    lines.push(format!("Question    : {}", r.question));
    lines.push(String::new());

    // Preferred hypothesis banner
    let preferred = if r.side_a.rank < r.side_b.rank {
        &r.side_a
    } else if r.side_b.rank < r.side_a.rank {
        &r.side_b
    } else {
        // Tie: prefer the one with higher plausibility
        if r.side_a.belief.plausibility >= r.side_b.belief.plausibility {
            &r.side_a
        } else {
            &r.side_b
        }
    };
    lines.push(format!(
        "★ PREFERRED: \"{}\" (drawer {})",
        preferred.claim, preferred.drawer_id
    ));
    lines.push(String::new());

    // Per-side results
    for side in [&r.side_a, &r.side_b] {
        let marker = if side.drawer_id == preferred.drawer_id { "★" } else { "○" };
        lines.push(format!(
            "{marker} Drawer {} — \"{}\"",
            side.drawer_id, side.claim
        ));
        lines.push(format!(
            "    Rank               : #{} (lower = more credible)",
            side.rank
        ));
        lines.push(format!(
            "    Inconsistency score: {:.3} (higher = more contradicted)",
            side.inconsistency_score
        ));
        lines.push(format!(
            "    DS Belief interval : [{:.3}, {:.3}]  width={:.3}",
            side.belief.belief,
            side.belief.plausibility,
            side.belief.width()
        ));
        lines.push(format!(
            "    DS Pignistic prob  : {:.3}",
            side.belief.belief + side.belief.width() / 2.0
        ));
        lines.push(String::new());
    }

    // Deciding evidence
    if !r.deciding_evidence.is_empty() {
        lines.push("Deciding Evidence (ordered by diagnosticity):".into());
        for (i, ev) in r.deciding_evidence.iter().enumerate() {
            let mut meta = Vec::new();
            if let Some(did) = &ev.drawer_id {
                meta.push(format!("drawer={did}"));
            }
            if ev.kg_triple.is_some() {
                meta.push("source=KG".into());
            }
            if let Some(fav) = &ev.favors {
                meta.push(format!("favors=drawer-{fav}"));
            }
            let meta_str = if meta.is_empty() {
                String::new()
            } else {
                format!("  [{}]", meta.join(", "))
            };
            lines.push(format!(
                "  {}. {} (diag={:.3}){meta_str}",
                i + 1,
                ev.description,
                ev.diagnosticity,
            ));
        }
        lines.push(String::new());
    }

    // Information gaps
    if !r.information_gaps.is_empty() {
        lines.push("Information Gaps:".into());
        for gap in &r.information_gaps {
            lines.push(format!("  ! {gap}"));
        }
        lines.push(String::new());
    } else {
        lines.push("Information Gaps: none".into());
        lines.push(String::new());
    }

    // Full matrix
    lines.push("── ACH Consistency Matrix ──────────────────────────".into());
    lines.push(r.matrix_text.clone());

    lines.join("\n")
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

fn parse_drawer_id(s: &str) -> Result<u64, McpError> {
    s.trim()
        .parse::<u64>()
        .map_err(|_| {
            McpError::invalid_params(
                format!(
                    "invalid drawer ID \"{s}\" — expected a numeric ID (e.g. \"1\" or \"42\")"
                ),
                None,
            )
        })
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drawer::DrawerStore;

    fn store_with_contradictions() -> (DrawerStore, u64, u64) {
        let store = DrawerStore::in_memory().unwrap();
        // drawer 1 – Monday claim
        let id_a = store
            .insert(
                "temperature",
                "temperature is 37 degrees",
                Some("thermometer-A".into()),
                0.9,
                Some(1_716_000_000),
                None,
                vec![],
            )
            .unwrap();
        // drawer 2 – Friday claim (newer, same subject)
        let id_b = store
            .insert(
                "temperature",
                "temperature is 22 degrees",
                Some("thermometer-B".into()),
                0.85,
                Some(1_716_432_000),
                None,
                vec![],
            )
            .unwrap();
        (store, id_a, id_b)
    }

    #[test]
    fn contradict_basic_demo() {
        let (store, id_a, id_b) = store_with_contradictions();
        let params = HekaContradictParams {
            drawer_a: id_a.to_string(),
            drawer_b: id_b.to_string(),
            kg_limit: Some(0),
        };
        let report = run_contradict(&store, &params, None).unwrap();

        assert!(report.contains("heka_contradict"), "missing header");
        assert!(report.contains("temperature is 37 degrees"), "missing claim A");
        assert!(report.contains("temperature is 22 degrees"), "missing claim B");
        assert!(report.contains("PREFERRED"), "missing preferred banner");
        assert!(report.contains("DS Belief interval"), "missing DS interval");
        assert!(report.contains("Deciding Evidence"), "missing deciding evidence");
    }

    #[test]
    fn contradict_newer_preferred() {
        let (store, id_a, id_b) = store_with_contradictions();
        let params = HekaContradictParams {
            drawer_a: id_a.to_string(),
            drawer_b: id_b.to_string(),
            kg_limit: Some(0),
        };
        let report = run_contradict(&store, &params, None).unwrap();

        // Drawer B (id_b) is newer and should be preferred.
        let preferred_line = report
            .lines()
            .find(|l| l.contains("PREFERRED"))
            .expect("PREFERRED line missing");
        assert!(
            preferred_line.contains(&format!("drawer {id_b}")),
            "newer drawer (id={id_b}) should be preferred, got: {preferred_line}"
        );
    }

    #[test]
    fn missing_drawer_returns_error() {
        let store = DrawerStore::in_memory().unwrap();
        let params = HekaContradictParams {
            drawer_a: "999".into(),
            drawer_b: "1000".into(),
            kg_limit: None,
        };
        let result = run_contradict(&store, &params, None);
        assert!(result.is_err(), "expected error for missing drawers");
    }

    #[test]
    fn invalid_drawer_id_returns_error() {
        let store = DrawerStore::in_memory().unwrap();
        let params = HekaContradictParams {
            drawer_a: "not-a-number".into(),
            drawer_b: "1".into(),
            kg_limit: None,
        };
        let result = run_contradict(&store, &params, None);
        assert!(result.is_err());
    }

    #[test]
    fn report_cites_drawer_ids_in_evidence() {
        let (store, id_a, id_b) = store_with_contradictions();
        let params = HekaContradictParams {
            drawer_a: id_a.to_string(),
            drawer_b: id_b.to_string(),
            kg_limit: Some(0),
        };
        let report = run_contradict(&store, &params, None).unwrap();

        assert!(
            report.contains(&format!("drawer={id_a}")),
            "report must cite drawer {id_a}"
        );
        assert!(
            report.contains(&format!("drawer={id_b}")),
            "report must cite drawer {id_b}"
        );
    }
}
