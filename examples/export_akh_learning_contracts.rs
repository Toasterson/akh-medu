//! Export a content-addressed snapshot of the tools BabyLLM will see.
//!
//! Seshat (or another MCP server) descriptors can be merged through the real
//! Akh proxy conversion:
//!
//! cargo run -p seshat --example export_mcp_tools > /tmp/seshat-tools.json
//! cargo run --example export_akh_learning_contracts --features daemon -- \
//!   --mcp seshat=/tmp/seshat-tools.json --check-learning-loop
//!
//! The same proof can avoid a temporary file:
//!
//! cargo run -p seshat --example export_mcp_tools --no-default-features |
//!   cargo run --example export_akh_learning_contracts --features daemon -- \
//!   --mcp-stdin seshat --check-learning-loop

#[cfg(feature = "daemon")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;

    use akh_medu::acp::mcp_bridge::{DiscoveredServerTools, build_tools_from_discovered};
    use akh_medu::agent::tools::{
        ArchivalSearchTool, StudyNoteAppendTool, StudyNoteListTool, StudyNoteReadTool,
        StudyNoteReviseTool,
    };
    use akh_medu::agent::{Agent, AgentConfig};
    use akh_medu::engine::{Engine, EngineConfig};
    use akh_medu::library::PersistentNoteCatalog;
    use akh_medu::vsa::Dimension;
    use grug_data::{KnowledgeToolBindings, KnowledgeWorkflowGenerator, ToolContractSnapshot};

    let mut mcp_specs = Vec::new();
    let mut mcp_stdin = None;
    let mut check_learning_loop = false;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--mcp" => mcp_specs.push(
                args.next()
                    .ok_or("--mcp requires SERVER=DESCRIPTORS.json")?,
            ),
            "--mcp-stdin" => {
                if mcp_stdin.is_some() {
                    return Err("--mcp-stdin may only be provided once".into());
                }
                mcp_stdin = Some(args.next().ok_or("--mcp-stdin requires SERVER")?);
            }
            "--check-learning-loop" => check_learning_loop = true,
            _ => return Err(format!("unknown argument {argument:?}").into()),
        }
    }

    let temp = tempfile::tempdir()?;
    let engine = Arc::new(Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })?);
    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default())?;
    let notes = Arc::new(PersistentNoteCatalog::open(temp.path().join("notes"))?);
    agent.set_note_catalog(notes.clone());
    agent.register_tool(Box::new(StudyNoteAppendTool::new(notes.clone())));
    agent.register_tool(Box::new(StudyNoteReviseTool::new(notes.clone())));
    agent.register_tool(Box::new(StudyNoteListTool::new(notes.clone())));
    agent.register_tool(Box::new(StudyNoteReadTool::new(notes.clone())));
    agent.register_tool(Box::new(ArchivalSearchTool::with_notes(notes)));

    let mut discovered = Vec::new();
    for spec in mcp_specs {
        let (server, path) = spec
            .split_once('=')
            .ok_or("--mcp must be SERVER=DESCRIPTORS.json")?;
        if server.trim().is_empty() || path.trim().is_empty() {
            return Err("--mcp server and path must not be empty".into());
        }
        let value: serde_json::Value = serde_json::from_reader(std::fs::File::open(path)?)?;
        let descriptors = if let Some(tools) = value.as_array() {
            tools.clone()
        } else if let Some(tools) = value.get("tools").and_then(serde_json::Value::as_array) {
            tools.clone()
        } else {
            return Err(format!("{path:?} must contain a JSON tool array").into());
        };
        discovered.push(DiscoveredServerTools {
            name: server.into(),
            // Never contacted: build_tools_from_discovered only converts the
            // descriptor. A deliberately invalid URL makes accidental calls
            // fail closed during future refactors.
            url: "http://127.0.0.1:1/contract-snapshot".into(),
            auth_header: None,
            descriptors,
        });
    }
    if let Some(server) = mcp_stdin {
        if server.trim().is_empty() {
            return Err("--mcp-stdin server must not be empty".into());
        }
        let value: serde_json::Value = serde_json::from_reader(std::io::stdin().lock())?;
        let descriptors = if let Some(tools) = value.as_array() {
            tools.clone()
        } else if let Some(tools) = value.get("tools").and_then(serde_json::Value::as_array) {
            tools.clone()
        } else {
            return Err("stdin must contain a JSON tool array".into());
        };
        discovered.push(DiscoveredServerTools {
            name: server,
            url: "http://127.0.0.1:1/contract-snapshot".into(),
            auth_header: None,
            descriptors,
        });
    }
    for tool in build_tools_from_discovered(&discovered) {
        agent.register_tool(tool);
    }

    let snapshot = ToolContractSnapshot::from_serializable(&agent.list_tool_manifests())?;
    if check_learning_loop {
        KnowledgeWorkflowGenerator::new(snapshot.clone(), KnowledgeToolBindings::default())?;
    }
    serde_json::to_writer_pretty(std::io::stdout().lock(), &snapshot)?;
    println!();
    Ok(())
}

#[cfg(not(feature = "daemon"))]
fn main() {
    eprintln!("this example requires --features daemon");
    std::process::exit(2);
}
