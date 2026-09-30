//! Prompt-prefix stability of the requests an akh sends to an
//! OpenAI-compatible model server (ADR 123 amendment 2026-09-30, ADR 121).
//!
//! llama.cpp's `llama-server` (Lilith on hephaistos) and every other
//! prefix-caching server reuse the KV cache only up to the first token that
//! differs from the previous request. Gemma 4's pinned chat template renders
//! the request as `system content`, then every tool declaration, then the
//! conversation, so anything per-turn inside the system message sits in front
//! of the ~25k-token tool block and forces a full re-prefill of it. With a
//! sliding-window model (Gemma 4: 1024-token window on 40 of 48 layers) the
//! server can only restore the cache at a context checkpoint, which it places
//! at user-message starts, so a divergence anywhere before the first user
//! message costs the whole prompt.
//!
//! These tests drive the real [`OpenAiChatBackend`] through the real agentic
//! turn against a local mock server, capture the request bodies, and compare
//! them in Gemma 4 render order.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::agent::agent::{Agent, AgentConfig};
use crate::agent::conversation::ConversationState;
use crate::agent::openai_backend::{OpenAiChatBackend, OpenAiConfig};
use crate::engine::{Engine, EngineConfig};
use crate::graph::Triple;

/// A minimal OpenAI-compatible server: `GET /models` and `POST
/// /chat/completions` (streaming or not), answering every chat with one short
/// sentence and recording every chat request body.
struct MockOpenAi {
    base_url: String,
    bodies: Arc<Mutex<Vec<Value>>>,
}

impl MockOpenAi {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock OpenAI server");
        let base_url = format!("http://{}/v1", listener.local_addr().expect("local addr"));
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&bodies);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().expect("clone stream"));
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).is_err() {
                    continue;
                }
                let mut content_length = 0usize;
                loop {
                    let mut header = String::new();
                    if reader.read_line(&mut header).is_err()
                        || header == "\r\n"
                        || header.is_empty()
                    {
                        break;
                    }
                    if let Some((name, value)) = header.split_once(':')
                        && name.eq_ignore_ascii_case("content-length")
                    {
                        content_length = value.trim().parse().unwrap_or(0);
                    }
                }
                let mut body = vec![0u8; content_length];
                if reader.read_exact(&mut body).is_err() {
                    continue;
                }
                let response = if request_line.contains("/models") {
                    let json = r#"{"data":[{"id":"lilith"}]}"#;
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                        json.len()
                    )
                } else {
                    let parsed: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
                    let stream_requested = parsed["stream"].as_bool().unwrap_or(false);
                    recorded
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(parsed);
                    if stream_requested {
                        let events = concat!(
                            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Ready.\"},\"finish_reason\":null}]}\n\n",
                            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
                            "data: [DONE]\n\n"
                        );
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{events}"
                        )
                    } else {
                        let json = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"Ready."},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{json}",
                            json.len()
                        )
                    }
                };
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
        Self { base_url, bodies }
    }

    fn bodies(&self) -> Vec<Value> {
        self.bodies
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

/// A request body rendered in the order Gemma 4's pinned template emits it:
/// the leading system message's text, then every tool declaration, then the
/// rest of the conversation. Returns the render and the byte length of its
/// `system + tools` head (everything before the first conversation turn).
fn gemma_order(body: &Value) -> (String, usize) {
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    let mut rendered = String::new();
    let mut rest = messages.as_slice();
    if let Some(first) = messages.first()
        && first["role"] == "system"
    {
        rendered.push_str("<|turn>system\n");
        rendered.push_str(first["content"].as_str().unwrap_or_default());
        rest = &messages[1..];
    }
    for tool in body["tools"].as_array().into_iter().flatten() {
        rendered.push_str("<|tool>");
        rendered.push_str(&tool.to_string());
        rendered.push_str("<tool|>");
    }
    rendered.push_str("<turn|>\n");
    let head = rendered.len();
    for message in rest {
        rendered.push_str("<|turn>");
        rendered.push_str(message["role"].as_str().unwrap_or_default());
        rendered.push('\n');
        match &message["content"] {
            Value::String(text) => rendered.push_str(text),
            other => rendered.push_str(&other.to_string()),
        }
        rendered.push_str("<turn|>\n");
    }
    (rendered, head)
}

/// Byte offset of the first difference between two renders (`None` when one
/// is a prefix of the other).
fn first_difference(a: &str, b: &str) -> Option<usize> {
    let offset = a.bytes().zip(b.bytes()).position(|(x, y)| x != y);
    match offset {
        Some(offset) => Some(offset),
        None if a.len() == b.len() => None,
        None => Some(a.len().min(b.len())),
    }
}

fn snippet(text: &str, at: usize) -> String {
    let start = text.floor_char_boundary(at.saturating_sub(60));
    let end = text.ceil_char_boundary((at + 120).min(text.len()));
    text[start..end].replace('\n', "\\n")
}

fn engine_with_facts() -> Arc<Engine> {
    let engine = Arc::new(Engine::new(EngineConfig::default()).expect("engine"));
    let host = engine.resolve_or_create_entity("hephaistos").unwrap();
    let serves = engine.resolve_or_create_entity("serves-model").unwrap();
    let model = engine.resolve_or_create_entity("gemma-4-12b").unwrap();
    engine
        .add_triple(&Triple::new(host, serves, model))
        .unwrap();
    engine
}

fn lilith_like_agent(engine: &Arc<Engine>, base_url: &str) -> Agent {
    let mut agent = Agent::new(Arc::clone(engine), AgentConfig::default()).expect("agent");
    agent.set_capture_transcripts(false);
    agent.set_context_manager(Arc::new(crate::agent::context::CompactingManager));
    agent.set_chat_backend(Arc::new(OpenAiChatBackend::new(
        OpenAiConfig {
            base_url: base_url.to_string(),
            api_key: Some("test-key".into()),
            model: "lilith".into(),
            context_window: Some(49_152),
            ..Default::default()
        },
        Some("You are Lilith, the operator's Chief of Staff.".into()),
    )));
    agent
}

fn run_turn(agent: &Agent, engine: &Engine, prompt: &str, state: &Mutex<ConversationState>) {
    let sink = crate::message::VecSink::new();
    let cancel = std::sync::atomic::AtomicBool::new(false);
    agent
        .run_agentic_turn_with_conversation_state(
            engine, "lilith", prompt, &sink, &cancel, false, state,
        )
        .expect("the agentic turn ran against the mock server");
}

/// The regression test for the 2026-09-30 Lilith C2 latency (every ACP turn
/// re-prefilled ~28k tokens): two turns of one session and the first turn of
/// a second session must send byte-identical `system + tools` heads, so the
/// first difference falls inside the conversation, after the tool block.
#[test]
fn openai_request_head_is_byte_identical_across_turns_and_sessions() {
    let server = MockOpenAi::start();
    let engine = engine_with_facts();
    let agent = lilith_like_agent(&engine, &server.base_url);

    let session_a = Mutex::new(ConversationState::new("session-a", "narrative"));
    let session_b = Mutex::new(ConversationState::new("session-b", "narrative"));
    run_turn(&agent, &engine, "One sentence: are you ready?", &session_a);
    run_turn(
        &agent,
        &engine,
        "Which model does hephaistos serve?",
        &session_a,
    );
    run_turn(
        &agent,
        &engine,
        "Which model does hephaistos serve?",
        &session_b,
    );

    let bodies = server.bodies();
    assert_eq!(bodies.len(), 3, "one request per turn");
    let renders: Vec<(String, usize)> = bodies.iter().map(gemma_order).collect();
    let (first, head) = &renders[0];
    assert!(
        bodies[0]["tools"].as_array().is_some_and(|t| !t.is_empty()),
        "the turn offers tools"
    );
    let mut failures = Vec::new();
    for (label, (other, other_head)) in [("turn 2", &renders[1]), ("new session", &renders[2])] {
        let diff = first_difference(first, other);
        eprintln!(
            "{label}: head {head} / {other_head} bytes, first difference at {diff:?}: …{}…",
            diff.map(|d| snippet(other, d)).unwrap_or_default()
        );
        if head != other_head || diff.is_some_and(|d| d < *head) {
            failures.push(format!(
                "{label}: the request diverges at byte {} inside the system + tools head \
                 ({head} bytes), so a prefix cache re-prefills the whole tool block",
                diff.unwrap_or_default()
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["function"]["name"].as_str().map(str::to_string))
        .collect()
}

/// An agent whose model profile comes from an `akh.toml` with a per-akh
/// `max_tools_per_turn` override and a `pinned_tools` core — the same path
/// akhomed takes at load.
fn scoped_agent(engine: &Arc<Engine>, base_url: &str, cap: u32, pinned: &[&str]) -> Agent {
    let pinned_toml = pinned
        .iter()
        .map(|name| format!("\"{name}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let toml_src = format!(
        r#"
        name = "lilith"
        backend = "openai"
        model = "lilith"
        pinned_tools = [{pinned_toml}]

        [openai_endpoint]
        base_url = "{base_url}"
        api_key = "test-key"
        context_window = 49152

        [model_profile]
        max_tools_per_turn = {cap}
        "#
    );
    let config: crate::akh_config::AkhConfig = toml::from_str(&toml_src).expect("akh.toml parses");
    let profile = config.resolve_model_profile().expect("profile resolves");
    assert_eq!(
        profile.max_tools_per_turn.as_ref().map(|s| s.value),
        Some(cap),
        "the per-akh override wins over the catalog row"
    );
    let mut agent = lilith_like_agent(engine, base_url);
    agent.set_model_profile(profile);
    agent.set_pinned_tools(config.pinned_tools.iter().cloned());
    agent
}

/// The catalog row for Lilith's model caps interactive turns at 16 tools.
#[test]
fn gemma_catalog_row_caps_interactive_turns() {
    let config: crate::akh_config::AkhConfig = toml::from_str(
        r#"
        name = "lilith"
        backend = "openai"
        model = "lilith"

        [openai_endpoint]
        base_url = "http://127.0.0.1:1/v1"
        api_key = "test-key"
        "#,
    )
    .expect("akh.toml parses");
    let profile = config.resolve_model_profile().expect("profile resolves");
    assert_eq!(profile.max_tools_per_turn.map(|s| s.value), Some(16));
}

/// A pinned core that fills the cap makes the whole tool block — and so the
/// whole `system + tools` head — identical for any prompt; the system
/// prompt's tool section lists exactly the offered tools.
#[test]
fn pinned_core_filling_the_cap_keeps_the_tool_block_identical() {
    let server = MockOpenAi::start();
    let engine = engine_with_facts();
    let pinned = ["user_interact", "kg_query", "tool_query", "memory_recall"];
    let agent = scoped_agent(
        &engine,
        &server.base_url,
        4,
        &[&pinned[..], &["mcp__anima__get_work_item"]].concat(),
    );
    let session = Mutex::new(ConversationState::new("session-a", "narrative"));
    run_turn(&agent, &engine, "One sentence: are you ready?", &session);
    run_turn(
        &agent,
        &engine,
        "Which model does hephaistos serve?",
        &session,
    );

    let bodies = server.bodies();
    assert_eq!(bodies.len(), 2);
    let expected = vec!["kg_query", "memory_recall", "tool_query", "user_interact"];
    assert_eq!(tool_names(&bodies[0]), expected, "pinned core, name-sorted");
    assert_eq!(tool_names(&bodies[1]), expected);
    let (first, head) = gemma_order(&bodies[0]);
    let (second, second_head) = gemma_order(&bodies[1]);
    assert_eq!(head, second_head);
    assert_eq!(
        first[..head],
        second[..head],
        "system + tools head is byte-identical"
    );

    let system = bodies[0]["messages"][0]["content"]
        .as_str()
        .unwrap_or_default();
    let listed = system.lines().filter(|l| l.starts_with("- **")).count();
    assert_eq!(
        listed,
        expected.len(),
        "the prompt lists only offered tools"
    );
}

/// With room for relevance picks, the pinned core still leads, name-sorted,
/// and the picks follow it name-sorted, within the cap.
#[test]
fn relevance_picks_follow_the_pinned_core_within_the_cap() {
    let server = MockOpenAi::start();
    let engine = engine_with_facts();
    let agent = scoped_agent(&engine, &server.base_url, 6, &["user_interact", "kg_query"]);
    let session = Mutex::new(ConversationState::new("session-a", "narrative"));
    run_turn(
        &agent,
        &engine,
        "Search the web for the llama.cpp release notes",
        &session,
    );
    run_turn(
        &agent,
        &engine,
        "Recall what you stored about hephaistos",
        &session,
    );

    for body in server.bodies() {
        let names = tool_names(&body);
        assert_eq!(names.len(), 6, "{names:?}");
        assert_eq!(names[..2], ["kg_query", "user_interact"], "{names:?}");
        let mut picks = names[2..].to_vec();
        picks.sort();
        assert_eq!(names[2..], picks[..], "picks are name-sorted: {names:?}");
    }
}

/// `scope_turn_catalog` in isolation: pins first whatever order they are
/// configured in, unavailable pins reported, picks sorted by name whatever
/// order the ranking returns, and no ranking at all once pins fill the cap.
#[test]
fn scope_turn_catalog_orders_pins_then_sorted_picks() {
    use crate::agent::agentic_chat::scope_turn_catalog;
    let tool = |name: &str| serde_json::json!({"type": "function", "function": {"name": name}});
    let full: Vec<Value> = ["alpha", "bravo", "charlie", "delta", "echo"]
        .into_iter()
        .map(tool)
        .collect();
    let names = |tools: &[Value]| -> Vec<String> {
        tools
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_string())
            .collect()
    };

    let scoped = scope_turn_catalog(
        &full,
        &["delta".into(), "missing".into(), "bravo".into()],
        4,
        |rest, k| {
            assert_eq!(k, 2);
            assert_eq!(names(rest), ["alpha", "charlie", "echo"]);
            vec![tool("echo"), tool("alpha")]
        },
    );
    assert_eq!(names(&scoped.tools), ["bravo", "delta", "alpha", "echo"]);
    assert_eq!(scoped.pinned, ["bravo", "delta"]);
    assert_eq!(scoped.unavailable, ["missing"]);

    let filled = scope_turn_catalog(&full, &["echo".into(), "alpha".into()], 2, |_, _| {
        panic!("no ranking once the pinned core fills the cap")
    });
    assert_eq!(names(&filled.tools), ["alpha", "echo"]);

    let over = scope_turn_catalog(
        &full,
        &["echo".into(), "alpha".into(), "bravo".into()],
        2,
        |_, _| panic!("pins past the cap are kept, never ranked away"),
    );
    assert_eq!(names(&over.tools), ["alpha", "bravo", "echo"]);
}

/// A registered stand-in for a static MCP proxy tool: the manifest the real
/// `StaticMcpTool` builds from a `tools/list` descriptor, never executed.
struct CapturedMcpTool {
    server: String,
    descriptor: crate::agent::tools::mcp::McpToolDescriptor,
}

impl crate::agent::tool::Tool for CapturedMcpTool {
    fn signature(&self) -> crate::agent::tool::ToolSignature {
        self.descriptor.signature(&format!(
            "mcp__{}__{}",
            self.server, self.descriptor.remote_name
        ))
    }

    fn execute(
        &self,
        _engine: &Engine,
        _input: crate::agent::tool::ToolInput,
    ) -> crate::agent::error::AgentResult<crate::agent::tool::ToolOutput> {
        Ok(crate::agent::tool::ToolOutput::err("measurement stand-in"))
    }

    fn manifest(&self) -> crate::agent::tool_manifest::ToolManifest {
        use crate::agent::tool_manifest::{
            DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolSource,
        };
        ToolManifest {
            name: format!("mcp__{}__{}", self.server, self.descriptor.remote_name),
            description: self.descriptor.description.clone(),
            parameters: self.descriptor.param_schemas(),
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Cautious,
                capabilities: std::collections::HashSet::new(),
                description: String::new(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Mcp {
                server: self.server.clone(),
            },
        }
    }
}

/// Measurement harness (run by hand, `--ignored`): renders the request a
/// Lilith turn sends before and after interactive scoping from her rendered
/// `akh.toml` (read-only) and captured MCP `tools/list` payloads, and writes
/// both bodies to `$PREFIX_MEASURE_OUT/{before,after}.json`.
///
/// `PREFIX_MEASURE_TOML`: the rendered workspace TOML (only `system_prompt`
/// and `pinned_tools` are read). `PREFIX_MEASURE_MCP`: a directory of
/// `<server>.json` JSON-RPC `tools/list` responses. `PREFIX_MEASURE_PINNED`:
/// optional comma-separated pinned core overriding the TOML's.
#[test]
#[ignore = "measurement harness; needs captured inputs"]
fn measure_lilith_request_before_and_after_scoping() {
    let toml_path = std::env::var("PREFIX_MEASURE_TOML").expect("PREFIX_MEASURE_TOML");
    let mcp_dir = std::env::var("PREFIX_MEASURE_MCP").expect("PREFIX_MEASURE_MCP");
    let out_dir = std::env::var("PREFIX_MEASURE_OUT").expect("PREFIX_MEASURE_OUT");
    let rendered: toml::Value =
        toml::from_str(&std::fs::read_to_string(&toml_path).expect("read toml")).expect("toml");
    let system_prompt = rendered["system_prompt"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let pinned: Vec<String> = match std::env::var("PREFIX_MEASURE_PINNED") {
        Ok(list) => list.split(',').map(str::to_string).collect(),
        Err(_) => rendered
            .get("pinned_tools")
            .and_then(toml::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default(),
    };

    let build = |cap: Option<u32>, base_url: &str| -> (Arc<Engine>, Agent) {
        let engine = Arc::new(Engine::new(EngineConfig::default()).expect("engine"));
        let mut agent = Agent::new(Arc::clone(&engine), AgentConfig::default()).expect("agent");
        agent.set_capture_transcripts(false);
        agent.set_context_manager(Arc::new(crate::agent::context::CompactingManager));
        agent.set_chat_backend(Arc::new(OpenAiChatBackend::new(
            OpenAiConfig {
                base_url: base_url.to_string(),
                api_key: Some("test-key".into()),
                model: "lilith".into(),
                context_window: Some(49_152),
                ..Default::default()
            },
            Some(system_prompt.clone()),
        )));
        // exec kind = none: akhomed registers no shell/file tools.
        for name in [
            "shell_exec",
            "file_io",
            "edit",
            "glob",
            "grep",
            "sandbox_exec",
        ] {
            agent.unregister_tool(name);
        }
        let registry = agent.tool_registry_handle();
        for entry in std::fs::read_dir(&mcp_dir).expect("mcp dir") {
            let path = entry.expect("entry").path();
            let server = path.file_stem().unwrap().to_string_lossy().to_string();
            let listing: Value =
                serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
            for raw in listing["result"]["tools"].as_array().into_iter().flatten() {
                if let Some(descriptor) = crate::agent::tools::mcp::McpToolDescriptor::parse(raw) {
                    registry.register(Box::new(CapturedMcpTool {
                        server: server.clone(),
                        descriptor,
                    }));
                }
            }
        }
        if let Some(cap) = cap {
            let config: crate::akh_config::AkhConfig = toml::from_str(&format!(
                "name = \"lilith\"\nbackend = \"openai\"\nmodel = \"lilith\"\n\
                 [openai_endpoint]\nbase_url = \"{base_url}\"\napi_key = \"k\"\n\
                 [model_profile]\nmax_tools_per_turn = {cap}\n"
            ))
            .expect("akh.toml");
            agent.set_model_profile(config.resolve_model_profile().expect("profile"));
            agent.set_pinned_tools(pinned.iter().cloned());
        }
        (engine, agent)
    };

    for (label, cap) in [("before", None), ("after", Some(16))] {
        let server = MockOpenAi::start();
        let (engine, agent) = build(cap, &server.base_url);
        let session = Mutex::new(ConversationState::new("session", "narrative"));
        run_turn(&agent, &engine, "One sentence: are you ready?", &session);
        let body = server.bodies().remove(0);
        eprintln!(
            "{label}: {} tools, system {} bytes, body {} bytes",
            tool_names(&body).len(),
            body["messages"][0]["content"]
                .as_str()
                .unwrap_or_default()
                .len(),
            body.to_string().len()
        );
        std::fs::write(
            format!("{out_dir}/{label}.json"),
            serde_json::to_string_pretty(&body).expect("json"),
        )
        .expect("write body");
    }
}
