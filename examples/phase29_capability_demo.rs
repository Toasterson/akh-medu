//! Phase 29 capability-exposure demo: every new `_akh/*` group end to end.
//!
//! Spins an in-process workspace, serves it over ACP/TCP, then drives each
//! Phase-29 method with the same client call the `akh` CLI uses
//! (`ext_request_tcp`) and prints a one-line sample per verb.
//!
//! Plan: docs/ai/plans/2026-03-26-phase29-capability-exposure.md — ADR 039.
//! (The plan predates ADR 061's ACP-only transport; the "MCP tools + REST
//! dashboard" surface it names is realized here as `_akh/*` method groups +
//! `akh acp` CLI verbs.)
//!
//! Run:
//! ```sh
//! cargo run --example phase29_capability_demo --features acp
//! ```
//!
//! The demo keeps its TCP listener up for 60 seconds after printing, so you
//! can point the real CLI at it, e.g.:
//! ```sh
//! akh --remote 127.0.0.1:<printed port> acp microtheory list
//! ```

#[cfg(feature = "acp")]
fn main() -> miette::Result<()> {
    use akh_medu::acp::{AkhAcpAgent, ext_request_tcp, serve_tcp_loop};
    use akh_medu::agent::{Agent, AgentConfig};
    use akh_medu::chat::ChatProcessor;
    use akh_medu::engine::{Engine, EngineConfig};
    use akh_medu::nlu::NluPipeline;
    use akh_medu::vsa::Dimension;
    use std::sync::{Arc, Mutex};

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| miette::miette!("tokio runtime: {e}"))?;

    rt.block_on(async move {
        // ── A persistent throwaway workspace (compartments + provenance need
        //    a data_dir), seeded with one fact and one on-disk skillpack. ──
        let dir = tempfile::tempdir().map_err(|e| miette::miette!("tempdir: {e}"))?;
        let data_dir = dir.path().join("ws");
        let skill_dir = data_dir.join("skills").join("astronomy");
        std::fs::create_dir_all(&skill_dir).map_err(|e| miette::miette!("skill dir: {e}"))?;
        std::fs::write(
            skill_dir.join("skill.json"),
            serde_json::json!({
                "id": "astronomy", "name": "Astronomy Pack", "version": "1.0.0",
                "description": "Basic astronomical knowledge", "domains": ["astronomy"],
                "weight_size_bytes": 0, "triples_file": "triples.json",
                "depends": [], "conflicts": [], "provides": [], "requires": []
            })
            .to_string(),
        )
        .map_err(|e| miette::miette!("skill manifest: {e}"))?;
        std::fs::write(
            skill_dir.join("triples.json"),
            serde_json::json!([
                { "subject": "Sol", "predicate": "is-a", "object": "Star", "confidence": 1.0 }
            ])
            .to_string(),
        )
        .map_err(|e| miette::miette!("skill triples: {e}"))?;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(data_dir),
            ..Default::default()
        })?;
        engine.ingest_label_triples(&[(
            "Sun".to_string(),
            "is-a".to_string(),
            "Star".to_string(),
            1.0,
        )])?;
        let engine = Arc::new(engine);
        let processor = ChatProcessor::new(&engine, NluPipeline::new());
        let mut agent = Agent::new(Arc::clone(&engine), AgentConfig::default())?;
        agent.force_set_psyche(akh_medu::compartment::psyche::Psyche::default());
        let me = AkhAcpAgent::new(engine, Arc::new(Mutex::new(agent)), processor);

        // ── Serve over ACP/TCP (the same loop the daemon runs). ──
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .map_err(|e| miette::miette!("bind: {e}"))?;
        let addr = listener
            .local_addr()
            .map_err(|e| miette::miette!("local addr: {e}"))?
            .to_string();
        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let server = tokio::spawn(serve_tcp_loop(me, listener, shutdown_rx));
        println!("ACP data-plane at {addr}\n");

        // One bounded call per verb; print `verb  ->  one-line result`.
        let call = |cli: &'static str, method: &'static str, params: serde_json::Value| {
            let addr = addr.clone();
            async move {
                let out = match ext_request_tcp(&addr, method, params).await {
                    Ok(v) => serde_json::to_string(&v).unwrap_or_default(),
                    Err(e) => format!("error: {e}"),
                };
                let line: String = out.chars().take(120).collect();
                println!("$ akh acp {cli}\n    {method} -> {line}\n");
            }
        };

        println!("── 29a microtheories ──");
        call(
            "microtheory create cooking --description \"my cooking knowledge\"",
            "_akh/microtheory/create",
            serde_json::json!({ "id": "cooking", "description": "my cooking knowledge" }),
        )
        .await;
        call(
            "microtheory list",
            "_akh/microtheory/list",
            serde_json::json!({}),
        )
        .await;
        call(
            "microtheory inspect cooking",
            "_akh/microtheory/inspect",
            serde_json::json!({ "id": "cooking" }),
        )
        .await;
        call(
            "microtheory activate cooking",
            "_akh/microtheory/activate",
            serde_json::json!({ "id": "cooking" }),
        )
        .await;
        call(
            "microtheory deactivate cooking --unload",
            "_akh/microtheory/deactivate",
            serde_json::json!({ "id": "cooking", "unload": true }),
        )
        .await;

        println!("── 29b skillpacks ──");
        call(
            "skill activate astronomy",
            "_akh/skills/activate",
            serde_json::json!({ "id": "astronomy" }),
        )
        .await;
        call("skill list", "_akh/skills", serde_json::json!({})).await;
        call(
            "skill inspect astronomy",
            "_akh/skills/inspect",
            serde_json::json!({ "id": "astronomy" }),
        )
        .await;
        call(
            "skill deactivate astronomy",
            "_akh/skills/deactivate",
            serde_json::json!({ "id": "astronomy" }),
        )
        .await;

        println!("── 29c psyche ──");
        call("psyche show", "_akh/psyche", serde_json::json!({})).await;
        call(
            "psyche archetypes --sage 0.9",
            "_akh/psyche/archetypes",
            serde_json::json!({ "sage": 0.9 }),
        )
        .await;

        println!("── 29d argumentation ──");
        call(
            "argue Sun is-a",
            "_akh/argue",
            serde_json::json!({ "subject": "Sun", "predicate": "is-a" }),
        )
        .await;

        println!("── 29e reasoning traces ──");
        call(
            "explain Sun",
            "_akh/explain",
            serde_json::json!({ "subject": "Sun" }),
        )
        .await;
        call(
            "explain --question \"what do you know about Sun?\"",
            "_akh/explain",
            serde_json::json!({ "question": "what do you know about Sun?" }),
        )
        .await;
        call(
            "provenance Sun",
            "_akh/provenance",
            serde_json::json!({ "symbol": "Sun" }),
        )
        .await;

        println!("── 29h evidence ──");
        call("evidence summary", "_akh/evidence", serde_json::json!({})).await;
        call(
            "evidence assess Sun",
            "_akh/evidence/assess",
            serde_json::json!({ "claim": "Sun" }),
        )
        .await;

        println!("── 29i source trust ──");
        call(
            "trust --top 5",
            "_akh/trust",
            serde_json::json!({ "top_k": 5 }),
        )
        .await;

        println!("listener stays up 60s — try: akh --remote {addr} acp microtheory list");
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        let _ = shutdown_tx.send(true);
        let _ = server.await;
        Ok(())
    })
}

#[cfg(not(feature = "acp"))]
fn main() {
    eprintln!("this demo requires --features acp");
}
