//! Sokar's agent surface (Anima WI 52a.1; docs/ai/phases/phase-52-sokar/52a1-mcp-brief.md).
mod tools;

use crate::{Sokar, SokarError, authz::Principal};
use axum::{
    body::Body,
    extract::Request,
    response::{IntoResponse, Response},
};
use rmcp::{
    handler::server::tool::ToolRouter,
    model::*,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService,
        session::local::{LocalSessionManager, SessionConfig},
    },
};
use std::{collections::HashMap, convert::Infallible, sync::Arc, time::Duration};
use tower::{Service, ServiceExt};

#[derive(Clone)]
pub struct SokarMcp {
    state: Arc<Sokar>,
    principal: Principal,
    tool_router: ToolRouter<Self>,
}
/// The rmcp factory takes no request argument. Route the authenticated extension
/// to a principal-specific service/manager, then create a handler per session.
/// This also isolates POST, GET/resume and DELETE session IDs across principals.
/// Bearer files are still reloaded by api_boundary on EVERY request.
pub fn service(
    state: Arc<Sokar>,
) -> impl Service<Request, Response = Response, Error = Infallible, Future: Send>
+ Clone
+ Send
+ Sync
+ 'static {
    let services: HashMap<_, _> = state
        .config
        .principals
        .iter()
        .map(|config| {
            let principal = Principal {
                id: config.id.clone(),
            };
            let manager = LocalSessionManager {
                session_config: SessionConfig {
                    keep_alive: Some(Duration::from_secs(state.config.mcp.session_idle_secs)),
                    ..Default::default()
                },
                ..Default::default()
            };
            let state = state.clone();
            let service = StreamableHttpService::new(
                move || Ok(SokarMcp::new(state.clone(), principal.clone())),
                Arc::new(manager),
                StreamableHttpServerConfig::default(),
            );
            (config.id.clone(), service)
        })
        .collect();
    let services = Arc::new(services);
    tower::service_fn(move |request: Request| {
        let service = request
            .extensions()
            .get::<Principal>()
            .and_then(|principal| services.get(&principal.id))
            .cloned();
        async move {
            match service {
                Some(service) => Ok(service.oneshot(request).await?.map(Body::new)),
                None => Ok(SokarError::Authn.into_response()),
            }
        }
    })
}

#[rmcp::tool_handler]
impl rmcp::ServerHandler for SokarMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation { name: "sokar".into(), ..Implementation::from_build_env() },
            instructions: Some("Sokar stores shared artifacts. Bytes are content-addressed and immutable per version. A path is unique within a collection with latest-wins lookup; replacing it creates a new version. Use sokar_collection before uploading. content_url requires the same bearer token as this MCP endpoint. Inline tools are bounded; use the HTTP API for larger artifacts.".into()),
            ..Default::default()
        }
    }
}
