//! akh-medu pre-processor HTTP server for the Eleutherios integration.
//!
//! Provides endpoints for text preprocessing and equivalence management:
//! - `POST /preprocess` — accepts `{ chunks: [{ id, text, language? }] }`, returns structured output
//! - `GET /health` — status, version, supported languages
//! - `GET /languages` — list languages with pattern counts
//! - `GET /equivalences` — list all learned equivalences
//! - `GET /equivalences/stats` — counts by source
//! - `POST /equivalences/learn` — trigger learning, return new count
//! - `POST /equivalences/import` — bulk import from JSON body
//!
//! Build and run with: `cargo run --features server --bin akh-medu-server`

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Serialize;
use tokio::sync::RwLock;

use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::grammar::concrete::ParseContext;
use akh_medu::grammar::entity_resolution::{
    EquivalenceStats, LearnedEquivalence,
};
use akh_medu::grammar::lexer::Language;
use akh_medu::grammar::preprocess::{
    preprocess_batch, PreProcessRequest, PreProcessResponse,
};
use akh_medu::vsa::Dimension;

struct AppState {
    engine: RwLock<Engine>,
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

#[derive(Serialize)]
struct LearnResponse {
    discovered: usize,
    total_learned: usize,
}

#[derive(Serialize)]
struct ImportResponse {
    imported: usize,
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
    let engine = state.engine.read().await;
    let ctx = ParseContext::with_engine(
        engine.registry(),
        engine.ops(),
        engine.item_memory(),
    );

    let start = Instant::now();
    let results = preprocess_batch(&request.chunks, &ctx);
    let elapsed = start.elapsed().as_millis() as u64;

    Ok(Json(PreProcessResponse {
        results,
        processing_time_ms: elapsed,
    }))
}

async fn equivalences_list(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<LearnedEquivalence>> {
    let engine = state.engine.read().await;
    Json(engine.export_equivalences())
}

async fn equivalences_stats(
    State(state): State<Arc<AppState>>,
) -> Json<EquivalenceStats> {
    let engine = state.engine.read().await;
    Json(engine.equivalence_stats())
}

async fn equivalences_learn(
    State(state): State<Arc<AppState>>,
) -> Result<Json<LearnResponse>, (StatusCode, String)> {
    let mut engine = state.engine.write().await;
    let discovered = engine.learn_equivalences().map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("learning failed: {e}"))
    })?;
    let stats = engine.equivalence_stats();
    Ok(Json(LearnResponse {
        discovered,
        total_learned: stats.learned_total,
    }))
}

async fn equivalences_import(
    State(state): State<Arc<AppState>>,
    Json(equivs): Json<Vec<LearnedEquivalence>>,
) -> Result<Json<ImportResponse>, (StatusCode, String)> {
    let count = equivs.len();
    let mut engine = state.engine.write().await;
    engine.import_equivalences(&equivs).map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("import failed: {e}"))
    })?;
    Ok(Json(ImportResponse { imported: count }))
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

    let state = Arc::new(AppState {
        engine: RwLock::new(engine),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/languages", get(languages))
        .route("/preprocess", post(preprocess))
        .route("/equivalences", get(equivalences_list))
        .route("/equivalences/stats", get(equivalences_stats))
        .route("/equivalences/learn", post(equivalences_learn))
        .route("/equivalences/import", post(equivalences_import))
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
