//! Phase 51 P3d-2 — does a **learned** embedder recover the paraphrase (D2)
//! retrieval the surface retrievers cannot?
//!
//! P3d-1's recall matrix showed that under true paraphrase (D2: request uses the
//! *alt* gloss, description the *primary* gloss — no shared surface tokens) every
//! surface retriever floors at ~0.01–0.02. This example embeds each tool
//! *description* and each *request* with Seshat's ONNX `all-MiniLM-L6-v2` and
//! ranks by cosine (via `grug_data::EmbedRetriever`, so the recall predicate is
//! identical to the surface retrievers). The question: does MiniLM materially
//! beat the surface floor on D2 — i.e. is a learned semantic router the thing
//! that makes paraphrase retrievable, as ADR 089's M1 caveat claims?
//!
//! Requires the embedding model on disk (`model.onnx` + `tokenizer.json`). By
//! default it looks in `$AKH_MINILM_DIR`, else
//! `/var/lib/akh-medu/seshat/models/all-MiniLM-L6-v2`. If absent, it prints
//! download instructions and exits non-zero (nothing is fabricated).
//!
//! ```text
//! cargo run -p seshat --example semantic_recall_minilm --release
//! # or point at a model dir:
//! AKH_MINILM_DIR=/path/to/all-MiniLM-L6-v2 cargo run -p seshat --example semantic_recall_minilm --release
//! ```

use std::path::PathBuf;

use grug_data::{
    Dataset, DistractorConfig, EmbedRetriever, Retriever, SemanticRetriever, SignalMode,
    embed_recall_at_k, retrieval_recall_at_k, semantic_recall_at_k,
};
use seshat::embedder::SentenceEmbedder;

fn model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("AKH_MINILM_DIR") {
        return PathBuf::from(d);
    }
    if let Some(arg) = std::env::args().nth(1) {
        return PathBuf::from(arg);
    }
    PathBuf::from("/var/lib/akh-medu/seshat/models/all-MiniLM-L6-v2")
}

/// Embed `texts` in chunks so a 850-item batch doesn't build one giant tensor.
fn embed_all(embedder: &mut SentenceEmbedder, texts: &[String]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(texts.len());
    for chunk in texts.chunks(64) {
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        let vecs = embedder
            .embed_batch(&refs)
            .expect("embedding a chunk should not fail once the model is loaded");
        out.extend(vecs);
    }
    out
}

fn main() {
    let dir = model_dir();
    let mut embedder = match SentenceEmbedder::load(&dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("semantic_recall_minilm: could not load the embedder: {e}\n");
            eprintln!("This P3d-2 measurement needs the ONNX all-MiniLM-L6-v2 model on disk:");
            eprintln!("  {}/model.onnx", dir.display());
            eprintln!("  {}/tokenizer.json", dir.display());
            eprintln!("\nProvision it (FLOSS, ~90 MB), e.g.:");
            eprintln!("  mkdir -p {}", dir.display());
            eprintln!(
                "  curl -L -o {}/model.onnx      https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
                dir.display()
            );
            eprintln!(
                "  curl -L -o {}/tokenizer.json  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
                dir.display()
            );
            eprintln!("\nThen re-run (or set AKH_MINILM_DIR to an existing copy).");
            std::process::exit(1);
        }
    };
    println!(
        "semantic_recall_minilm: embedder loaded from {}",
        dir.display()
    );

    let seed = 42u64;
    let n_tools = 1000usize;
    let holdout_frac = 0.15;
    let k = 8usize;
    let n_pairs = 500usize;

    println!("seed={seed} n_tools={n_tools} k={k} n_pairs={n_pairs}\n");
    println!(
        "{:<8}  {:>12}  {:>12}  {:>14}",
        "tier", "lex@desc", "hdc@desc", "minilm@desc"
    );

    let mut rows = Vec::new();
    for (label, mode) in [
        ("D0/name", SignalMode::Name),
        ("D1/desc", SignalMode::Description),
        ("D2/para", SignalMode::Paraphrase),
    ] {
        let dataset = Dataset::new_sized_signal(
            seed,
            n_tools,
            holdout_frac,
            DistractorConfig::easy_with_count(2),
            mode,
        )
        .expect("dataset splits cleanly");
        let registry = dataset.registry();
        let descriptions: Vec<String> = registry.iter().map(|t| t.description.clone()).collect();
        let pairs = dataset
            .retrieval_recall_pairs(n_pairs)
            .expect("recall pairs");

        // Surface references (identical to the P3d-1 matrix).
        let lex_desc = retrieval_recall_at_k(&Retriever::new_over(&descriptions), &pairs, k);
        let hdc_desc = semantic_recall_at_k(&SemanticRetriever::new_over(&descriptions), &pairs, k);

        // Learned embedding: embed descriptions (the registry) and requests.
        let tool_vecs = embed_all(&mut embedder, &descriptions);
        let embed_r = EmbedRetriever::new(tool_vecs);
        let req_texts: Vec<String> = pairs.iter().map(|(r, _)| r.clone()).collect();
        let req_vecs = embed_all(&mut embedder, &req_texts);
        let embed_pairs: Vec<(Vec<f32>, usize)> = req_vecs
            .into_iter()
            .zip(pairs.iter().map(|(_, i)| *i))
            .collect();
        let minilm_desc = embed_recall_at_k(&embed_r, &embed_pairs, k);

        println!("{label:<8}  {lex_desc:>12.3}  {hdc_desc:>12.3}  {minilm_desc:>14.3}");
        rows.push(serde_json::json!({
            "tier": label, "lex_desc": lex_desc, "hdc_desc": hdc_desc, "minilm_desc": minilm_desc,
        }));
    }

    println!("\nReading: the D2 row is the test — surface (lex/hdc)@desc floor at ~0.01-0.02;");
    println!("MiniLM@desc must beat that materially for a learned router to be the thing that");
    println!("makes genuine paraphrase retrievable (ADR 089 M1 caveat).");

    let out = serde_json::json!({ "seed": seed, "n_tools": n_tools, "k": k, "n_pairs": n_pairs, "model_dir": dir.display().to_string(), "rows": rows });
    println!("\n{}", serde_json::to_string_pretty(&out).unwrap());
}
