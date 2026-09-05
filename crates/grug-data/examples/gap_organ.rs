//! Phase 51 P3a follow-up — two **model-independent** measurements on the
//! retrieval-scoped registry (no trained model, pure CPU):
//!
//! 1. **Semantic vs lexical retrieval** (the standing caveat). The lexical
//!    retriever's recall@8 = 1.0 could be an artifact of the retriever and the
//!    training label sharing one signal (verbatim name tokens). We compare it to
//!    an HDC/VSA char-trigram retriever on clean requests (both should be high —
//!    the caveat, made concrete) and under request-side name typos (exact-token
//!    lexical collapses; the trigram HDC is robust). Neither is a *learned*
//!    embedding, so neither is claimed to survive genuine paraphrase — that is
//!    the Phase 28 VSA router's job, untestable on this name-only corpus.
//!
//! 2. **Gap → organ widen/synthesize** (ADR 089 mechanism 3 back half). With a
//!    deliberately narrow offered set (k=3) so genuine retrieval misses exist,
//!    how many gaps does widening recover, and do genuine-novelty requests
//!    correctly terminate in synthesis rather than false-widening onto a
//!    confusable?
//!
//! ```text
//! cargo run -p grug-data --example gap_organ --release
//! ```

use std::collections::HashSet;

use grug_data::organ::GapRecoveryReport;
use grug_data::retriever::{SemanticRetriever, retrieval_recall_at_k, semantic_recall_at_k};
use grug_data::{Dataset, DistractorConfig};

/// Substitute one middle character in every request word equal to a `verb_noun`
/// name token — breaks an exact-token match while leaving most char trigrams.
fn corrupt_name_tokens(request: &str, name: &str) -> String {
    let toks: HashSet<String> = name.split('_').map(|s| s.to_ascii_lowercase()).collect();
    request
        .split_whitespace()
        .map(|w| {
            let key: String = w
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .collect::<String>()
                .to_ascii_lowercase();
            if toks.contains(&key) && w.chars().count() >= 2 {
                let mut cs: Vec<char> = w.chars().collect();
                let i = cs.len() / 2;
                cs[i] = if cs[i] == 'x' { 'q' } else { 'x' };
                cs.into_iter().collect()
            } else {
                w.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let n_tools = 1000usize;
    let holdout_frac = 0.15;
    let dataset = Dataset::new_sized_with_distractors(
        42,
        n_tools,
        holdout_frac,
        DistractorConfig::easy_with_count(2),
    )
    .expect("dataset splits cleanly");
    let registry = dataset.registry();
    println!(
        "gap_organ: registry={} out_of_registry(holdout)={}",
        registry.len(),
        dataset.holdout_tools().len()
    );

    // ---- Lever 2: semantic vs lexical retrieval ----
    let lexical = dataset.registry_retriever();
    let semantic = SemanticRetriever::new(registry);
    let pairs = dataset.retrieval_recall_pairs(500).expect("recall pairs");

    let clean_lex = retrieval_recall_at_k(&lexical, &pairs, 8);
    let clean_sem = semantic_recall_at_k(&semantic, &pairs, 8);

    let noisy: Vec<(String, usize)> = pairs
        .iter()
        .map(|(req, idx)| (corrupt_name_tokens(req, &registry[*idx].name), *idx))
        .collect();
    let noisy_lex = retrieval_recall_at_k(&lexical, &noisy, 8);
    let noisy_sem = semantic_recall_at_k(&semantic, &noisy, 8);

    println!("\n=== Lever 2: retrieval recall@8 (n={}) ===", pairs.len());
    println!("            lexical (exact-token)   semantic (HDC char-trigram)");
    println!("  clean   :        {clean_lex:.3}                {clean_sem:.3}");
    println!("  +typos  :        {noisy_lex:.3}                {noisy_sem:.3}");
    println!(
        "  → exact-token recall drops {:+.3} under name typos; HDC drops {:+.3} (robustness = the VSA layer's contribution)",
        noisy_lex - clean_lex,
        noisy_sem - clean_sem
    );

    // ---- Lever 3: gap → widen / synthesize ----
    // Scoped in-registry pairs (reuse `pairs`), narrow offered set k=3.
    let novelty: Vec<String> = dataset
        .retrieval_abstention_examples(8, 500)
        .expect("novelty examples")
        .into_iter()
        .map(|ex| ex.user_request)
        .collect();
    let schedule = [8usize, 16, 64, registry.len()];
    let report = GapRecoveryReport::measure(&lexical, &pairs, &novelty, 3, &schedule);

    println!(
        "\n=== Lever 3: gap → organ widen/synthesize (from_k=3, widen→{}) ===",
        registry.len()
    );
    println!(
        "  retrieval-miss gaps : {} of {} scoped  → widen recovered {} ({:.3})",
        report.retrieval_misses,
        pairs.len(),
        report.widen_recovered,
        report.recovery_rate()
    );
    println!(
        "  genuine novelty     : {} → synthesize {} ({:.3}) | false-widen {} (confusable wrongly 'covered')",
        report.novelty,
        report.novelty_synthesized,
        report.synthesis_rate(),
        report.novelty_false_widen
    );

    let out = serde_json::json!({
        "n_tools": n_tools,
        "lever2_recall": {
            "clean": { "lexical": clean_lex, "semantic": clean_sem },
            "typos": { "lexical": noisy_lex, "semantic": noisy_sem },
        },
        "lever3_gap_organ": {
            "from_k": report.from_k,
            "max_width": report.max_width,
            "retrieval_misses": report.retrieval_misses,
            "widen_recovered": report.widen_recovered,
            "recovery_rate": report.recovery_rate(),
            "novelty": report.novelty,
            "novelty_synthesized": report.novelty_synthesized,
            "novelty_false_widen": report.novelty_false_widen,
            "synthesis_rate": report.synthesis_rate(),
        },
    });
    println!("\n{}", serde_json::to_string_pretty(&out).unwrap());
}
