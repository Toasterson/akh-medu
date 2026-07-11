//! Phase 51 P3d-1 — the retrieval **recall matrix**: how much of P3a's
//! recall@8 = 1.000 was surface coupling to the tool *name*?
//!
//! For each [`SignalMode`] tier (D0 name / D1 description / D2 paraphrase) and
//! each retriever (lexical or HDC, keyed on the name or the description), report
//! recall@8 over the registry. Model-independent, pure CPU.
//!
//! Expected shape (the hypothesis P3d-1 tests):
//! - **D0**: lexical-on-name ≈ 1.0 (the name is embedded verbatim — the P3a
//!   result reproduced).
//! - **D1**: lexical-on-name collapses (name absent from the request); the
//!   signal has moved to the description, so lexical/HDC-on-**description**
//!   recover it.
//! - **D2**: request (alt gloss) and description (primary gloss) share no
//!   surface tokens → *every* surface retriever drops; only a learned embedding
//!   (P3d-2, Seshat MiniLM) can bridge. This quantifies the genuine paraphrase
//!   gap the lexical caveat pointed at.
//!
//! ```text
//! cargo run -p grug-data --example recall_matrix --release
//! ```

use grug_data::retriever::{retrieval_recall_at_k, semantic_recall_at_k, Retriever, SemanticRetriever};
use grug_data::{Dataset, DistractorConfig, SignalMode};

fn main() {
    let seed = 42u64;
    let n_tools = 1000usize;
    let holdout_frac = 0.15;
    let k = 8usize;
    let n_pairs = 500usize;

    println!("recall_matrix: seed={seed} n_tools={n_tools} k={k} n_pairs={n_pairs}\n");
    println!(
        "{:<8}  {:>12}  {:>12}  {:>12}  {:>12}",
        "tier", "lex@name", "lex@desc", "hdc@name", "hdc@desc"
    );

    let mut rows = Vec::new();
    for (label, mode) in [("D0/name", SignalMode::Name), ("D1/desc", SignalMode::Description), ("D2/para", SignalMode::Paraphrase)] {
        let dataset = Dataset::new_sized_signal(seed, n_tools, holdout_frac, DistractorConfig::easy_with_count(2), mode)
            .expect("dataset splits cleanly");
        let registry = dataset.registry();
        let names: Vec<String> = registry.iter().map(|t| t.name.clone()).collect();
        let descriptions: Vec<String> = registry.iter().map(|t| t.description.clone()).collect();

        let lex_name = dataset.registry_retriever();
        let lex_desc = Retriever::new_over(&descriptions);
        let hdc_name = SemanticRetriever::new_over(&names);
        let hdc_desc = SemanticRetriever::new_over(&descriptions);

        let pairs = dataset.retrieval_recall_pairs(n_pairs).expect("recall pairs");

        let r_lex_name = retrieval_recall_at_k(&lex_name, &pairs, k);
        let r_lex_desc = retrieval_recall_at_k(&lex_desc, &pairs, k);
        let r_hdc_name = semantic_recall_at_k(&hdc_name, &pairs, k);
        let r_hdc_desc = semantic_recall_at_k(&hdc_desc, &pairs, k);

        println!(
            "{label:<8}  {r_lex_name:>12.3}  {r_lex_desc:>12.3}  {r_hdc_name:>12.3}  {r_hdc_desc:>12.3}"
        );

        // Show one glossed example per tier so the transformation is legible.
        if let Some((req, idx)) = pairs.first() {
            println!("          e.g. tool={:<24} desc={:?}", registry[*idx].name, registry[*idx].description);
            println!("               request={req:?}");
        }

        rows.push(serde_json::json!({
            "tier": label,
            "lex_name": r_lex_name,
            "lex_desc": r_lex_desc,
            "hdc_name": r_hdc_name,
            "hdc_desc": r_hdc_desc,
        }));
    }

    println!("\nReading: D0 lex@name≈1.0 reproduces P3a. D1 lex@name should collapse while");
    println!("lex@desc / hdc@desc recover (signal moved to the description). D2 all surface");
    println!("retrievers drop — only a learned embedding (P3d-2 MiniLM) bridges request↔desc.");

    let out = serde_json::json!({ "seed": seed, "n_tools": n_tools, "k": k, "n_pairs": n_pairs, "rows": rows });
    println!("\n{}", serde_json::to_string_pretty(&out).unwrap());
}
