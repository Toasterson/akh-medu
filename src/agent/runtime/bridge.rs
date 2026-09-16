//! Compatibility host for the shared provider package; akhomed core builds omit it.
use super::*;
use akh_session_types as wire;

pub(super) fn run(turn: &mut Turn<'_>) -> Result<String, String> {
    let kind = match turn.runtime.kind {
        RuntimeKind::Codex => wire::RuntimeKind::Codex,
        RuntimeKind::Claude => wire::RuntimeKind::Claude,
    };
    let config = akh_runtime::ProviderConfig {
        isolation: None,
        model_endpoint: None,
        revision: "embedded-compat-v1".into(),
        kind,
        working_dir: turn.runtime.config.working_dir.clone(),
        binary: turn
            .runtime
            .config
            .binary
            .clone()
            .unwrap_or_else(|| turn.runtime.kind.label().into()),
        node_binary: turn.runtime.config.node_binary.clone(),
        bridge_script: turn.runtime.config.bridge_script.clone(),
        environment: Default::default(),
        launcher: vec![],
        timeout_secs: turn.runtime.config.timeout_secs,
        execution_mode: wire::ExecutionMode::Colocated,
        conformance_file: None,
        native_aliases: Default::default(),
        effort: turn.runtime.effort.clone(),
    };
    let request = wire::RuntimeJobRequest {
        id: wire::new_id(),
        session_id: turn
            .session
            .as_ref()
            .map(|s| s.id.clone())
            .unwrap_or_else(wire::new_id),
        turn_id: wire::new_id(),
        generation: 1,
        binding: wire::RuntimeBinding {
            runtime: "embedded".into(),
            profile: "embedded".into(),
            revision: config.revision.clone(),
            kind,
            model: turn.runtime.model.clone(),
            execution_mode: wire::ExecutionMode::Colocated,
        },
        workspace: wire::WorkspaceBinding {
            id: "embedded".into(),
            sandbox_id: "embedded".into(),
            executor: "embedded".into(),
            generation: 1,
            site: "local".into(),
            root: turn.cwd.to_string_lossy().into_owned(),
            platform: std::env::consts::OS.into(),
            confinement: "legacy-in-process".into(),
        },
        provider_session: turn.session.as_ref().map(|s| s.id.clone()),
        input: wire::TurnInput {
            origin: wire::InputOrigin::Human {
                principal: "embedded-client".into(),
            },
            text: turn.prompt.clone(),
            data: None,
            artifacts: vec![],
        },
        instructions: turn.instructions.clone(),
        tools: turn
            .tools
            .iter()
            .map(|t| wire::ToolSpec {
                name: t["function"]["name"].as_str().unwrap_or_default().into(),
                description: t["function"]["description"]
                    .as_str()
                    .unwrap_or_default()
                    .into(),
                input_schema: t["function"]["parameters"].clone(),
                read_only: false,
                destination: wire::ToolDestination::Client,
            })
            .collect(),
        approval_policy: if turn.registry.permission_mode()
            == super::super::permission::PermissionMode::Plan
        {
            wire::ApprovalPolicy::ReadOnly
        } else {
            wire::ApprovalPolicy::AskWrites
        },
    };
    let cancel = turn.cancel;
    akh_runtime::run(&config, &request, turn, cancel)
}

impl akh_runtime::Host for Turn<'_> {
    fn event(&mut self, event: wire::RuntimeEvent) -> Result<(), String> {
        match event {
            wire::RuntimeEvent::Bound {
                provider_session,
                model,
            } => self.bind(&provider_session, model.as_deref()),
            wire::RuntimeEvent::Text { text } => self.text(&text),
            wire::RuntimeEvent::Reasoning { text } => self.sink.emit(&AkhMessage::Reasoning {
                step: text, expression: None,
            }),
            wire::RuntimeEvent::ToolStarted { id, name } => {
                self.sink.emit(&AkhMessage::ToolCallStarted {
                    tool_id: id,
                    kind: crate::message::ToolKind::Other,
                    title: name,
                    locations: vec![],
                })
            }
            wire::RuntimeEvent::ToolCompleted {
                id,
                success,
                details,
            } => self.sink.emit(&AkhMessage::ToolCallCompleted {
                tool_id: id,
                success,
                output: details.map(|value| value.to_string()),
                diff: None,
            }),
        }
        Ok(())
    }
    fn tool(
        &mut self,
        _id: &str,
        name: &str,
        arguments: Value,
    ) -> Result<wire::ToolResult, String> {
        let value = Turn::tool(self, name, arguments);
        Ok(wire::ToolResult {
            success: value["success"] == true,
            text: value["text"].as_str().unwrap_or_default().into(),
            data: None,
            receipt: None,
        })
    }
    fn approval(&mut self, _id: &str, name: &str, arguments: &Value) -> Result<bool, String> {
        Ok(self.approve(name, arguments))
    }
    fn question(&mut self, _id: &str, _questions: &Value) -> Result<Value, String> {
        Err("structured runtime questions require the akhsessd input bridge".into())
    }
}
