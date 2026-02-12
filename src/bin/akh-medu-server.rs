//! akh-medu pre-processor HTTP server for the Eleutherios integration.
//!
//! Provides three endpoints:
//! - `POST /preprocess` — accepts `{ chunks: [{ id, text, language? }] }`, returns structured output
//! - `GET /health` — status, version, supported languages
//! - `GET /languages` — list languages with pattern counts
//!
//! Build and run with: `cargo run --features server --bin akh-medu-server`

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::grammar::concrete::ParseContext;
use akh_medu::grammar::lexer::Language;
use akh_medu::grammar::preprocess::{
    preprocess_batch, PreProcessRequest, PreProcessResponse,
};
use akh_medu::vsa::Dimension;

struct AppState {
    engine: Engine,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    supported_languages: Vec<String>,
}

#[derive(Serialize)]
struct LanguageInfo {
    code: String,
    name: String,
    pattern_count: usize,
}

#[derive(Serialize)]
struct LanguagesResponse {
    languages: Vec<LanguageInfo>,
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        supported_languages: vec![
            "en".to_string(),
            "ru".to_string(),
            "ar".to_string(),
            "fr".to_string(),
            "es".to_string(),
            "auto".to_string(),
        ],
    })
}

async fn languages() -> Json<LanguagesResponse> {
    use akh_medu::grammar::lexer::Lexicon;

    let langs = vec![
        (Language::English, "English"),
        (Language::Russian, "Russian"),
        (Language::Arabic, "Arabic"),
        (Language::French, "French"),
        (Language::Spanish, "Spanish"),
    ];

    let languages = langs
        .into_iter()
        .map(|(lang, name)| {
            let lexicon = Lexicon::for_language(lang);
            LanguageInfo {
                code: lang.bcp47().to_string(),
                name: name.to_string(),
                pattern_count: lexicon.relational_patterns().len(),
            }
        })
        .collect();

    Json(LanguagesResponse { languages })
}

async fn preprocess(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PreProcessRequest>,
) -> Result<Json<PreProcessResponse>, (StatusCode, String)> {
    let ctx = ParseContext::with_engine(
        state.engine.registry(),
        state.engine.ops(),
        state.engine.item_memory(),
    );

    let start = Instant::now();
    let results = preprocess_batch(&request.chunks, &ctx);
    let elapsed = start.elapsed().as_millis() as u64;

    Ok(Json(PreProcessResponse {
        results,
        processing_time_ms: elapsed,
    }))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")),
        )
        .init();

    let config = EngineConfig {
        dimension: Dimension::DEFAULT,
        ..Default::default()
    };

    let engine = Engine::new(config).expect("failed to initialize engine");
    tracing::info!("akh-medu pre-processor engine initialized");

    let state = Arc::new(AppState { engine });

    let app = Router::new()
        .route("/health", get(health))
        .route("/languages", get(languages))
        .route("/preprocess", post(preprocess))
        .with_state(state);

    let bind = "0.0.0.0:8200";
    tracing::info!("akh-medu pre-processor server listening on {bind}");

    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .expect("failed to bind");
    axum::serve(listener, app)
        .await
        .expect("server error");
}
