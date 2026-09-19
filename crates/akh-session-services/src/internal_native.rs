//! One process-wide resident model and compute lane. The HIP current stream is
//! process-global; even separate Runtime instances must never generate concurrently.
use super::*;
use akh_medu::nlu::{
    candle_backend::{
        CandleBackend, CandleConfig, CandleDevice, CandleFinishReason, CaptureRequest,
        ModelFamily, RecallDecision,
    },
    error::NluError,
    span_projection::{CapturePolicy, Projector},
};
use akh_session_types::neural_spans::{
    MAX_TEXT_BYTES, SpanCapture, SpanSource, SpanSpace,
};
use std::sync::{Mutex, MutexGuard, TryLockError};

/// At most this many recalled spans are materialized before the user turn.
const MAX_RECALL_HITS: usize = 4;

/// First-iteration neural context for one turn: the accepted input text that
/// captures are byte ranges of, and the callback identity for recall/delivery.
pub(super) struct RecallContext<'a> {
    pub(super) input_text: &'a str,
    pub(super) call_id: String,
}

/// Everything decided before prefill about capturing the user span.
struct CapturePlan {
    policy: CapturePolicy,
    projector: Projector,
    space: SpanSpace,
    /// Byte range inside the rendered prompt.
    prompt_bytes: std::ops::Range<usize>,
    /// Byte range inside the accepted input text.
    input_bytes: std::ops::Range<usize>,
}

/// Capture the leading (trimmed, bounded) user input as rendered in the prompt.
/// `None` skips capture silently: a non-Qwen family, an empty input, or a
/// renderer whose user content no longer starts with the accepted text.
fn capture_plan(
    config: &NativeCandle,
    backend: &CandleBackend,
    prompt: &str,
    user_span: Option<&std::ops::Range<usize>>,
    input_text: &str,
) -> Option<CapturePlan> {
    let user_span = user_span?;
    let (blocks, hidden) = backend.qwen35_shape()?;
    let policy = CapturePolicy::for_model(blocks, hidden)?;
    let lead = input_text.len() - input_text.trim_start().len();
    let core = input_text.trim();
    let mut len = core.len().min(MAX_TEXT_BYTES);
    while len > 0 && !core.is_char_boundary(len) {
        len -= 1;
    }
    if len == 0 {
        return None;
    }
    let prompt_bytes = user_span.start..user_span.start + len;
    if prompt.get(prompt_bytes.clone()) != Some(&core[..len]) {
        return None;
    }
    let projector = Projector::default();
    let space = SpanSpace {
        model_sha256: config.model_sha256.to_ascii_lowercase(),
        tokenizer_sha256: config.tokenizer_sha256.to_ascii_lowercase(),
        policy_sha256: policy.sha256(),
        projector_sha256: projector.sha256(),
    };
    space.validate().ok()?;
    Some(CapturePlan {
        policy,
        projector,
        space,
        prompt_bytes,
        input_bytes: lead..lead + len,
    })
}

/// Weight residency is independent of per-turn limits and logical profile names.
/// Durable continuations separately retain the complete NativeCandle snapshot.
#[derive(Debug, PartialEq, Eq)]
struct ResidentIdentity {
    model_path: std::path::PathBuf,
    model_sha256: String,
    tokenizer_path: std::path::PathBuf,
    tokenizer_sha256: String,
    device: NativeCandleDevice,
}

impl From<&NativeCandle> for ResidentIdentity {
    fn from(config: &NativeCandle) -> Self {
        Self {
            model_path: config.model_path.clone(),
            model_sha256: config.model_sha256.clone(),
            tokenizer_path: config.tokenizer_path.clone(),
            tokenizer_sha256: config.tokenizer_sha256.clone(),
            device: config.device.clone(),
        }
    }
}

static MODEL: Mutex<Option<(ResidentIdentity, CandleBackend)>> = Mutex::new(None);
#[cfg(test)]
#[path = "native_instruction_gate.rs"]
mod instruction_gate;
#[cfg(test)]
pub(super) static GENERATION_CHECKS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

pub(super) fn validate_device(device: &NativeCandleDevice) -> Result<()> {
    match device {
        NativeCandleDevice::Cpu => Ok(()),
        NativeCandleDevice::Hip { ordinal: 0 } if cfg!(feature = "hip") => Ok(()),
        NativeCandleDevice::Vulkan { .. } if cfg!(feature = "vulkan") => Ok(()),
        NativeCandleDevice::Metal { .. } if cfg!(all(feature = "metal", target_os = "macos")) => {
            Ok(())
        }
        _ => Err(Fault::new(
            "unsupported",
            "requested native device is unavailable in this build; HIP supports ordinal zero only",
        )),
    }
}

fn acquire_model(
    cancel: &AtomicBool,
    deadline: Instant,
) -> Result<MutexGuard<'static, Option<(ResidentIdentity, CandleBackend)>>> {
    loop {
        check(cancel, deadline)?;
        match MODEL.try_lock() {
            Ok(guard) => return Ok(guard),
            Err(TryLockError::Poisoned(_)) => {
                return Err(Fault::new(
                    "model",
                    "native compute lane is poisoned; restart required",
                ));
            }
            Err(TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(10)),
        }
    }
}

pub(super) fn completion(
    config: &NativeCandle,
    session_key: &str,
    messages: &[Value],
    tools: &[Value],
    host: &mut dyn Host,
    cancel: &AtomicBool,
    deadline: Instant,
    recall: Option<RecallContext<'_>>,
) -> Result<(Value, String, Option<Value>)> {
    let mut resident = acquire_model(cancel, deadline)?;
    let identity = ResidentIdentity::from(config);
    if resident
        .as_ref()
        .is_none_or(|(loaded, _)| loaded != &identity)
    {
        // Dropping tensors alone can leave old HIP weights in its process-wide
        // allocator cache. Release that unused memory before another identity
        // (including Vulkan on the same card) attempts a multi-gigabyte load.
        if let Some((_, previous)) = resident.take() {
            let released = previous
                .unload()
                .map_err(|e| Fault::new("model_unload", e.to_string()))?;
            tracing::debug!(released_bytes = released, "unloaded previous native model");
        }
        verify_artifact(&config.model_path, &config.model_sha256, cancel, deadline)?;
        verify_artifact(
            &config.tokenizer_path,
            &config.tokenizer_sha256,
            cancel,
            deadline,
        )?;
        let device = match config.device {
            NativeCandleDevice::Cpu => CandleDevice::Cpu,
            NativeCandleDevice::Hip { ordinal } => CandleDevice::Hip { ordinal },
            NativeCandleDevice::Cuda { ordinal } => CandleDevice::Cuda { ordinal },
            NativeCandleDevice::Metal { ordinal } => CandleDevice::Metal { ordinal },
            NativeCandleDevice::Vulkan { ordinal } => CandleDevice::Vulkan { ordinal },
        };
        let backend = CandleBackend::load_on_device(
            &config.model_path,
            &config.tokenizer_path,
            CandleConfig {
                max_tokens: config.max_output_tokens,
                ..Default::default()
            },
            device,
        )
        .map_err(|e| Fault::new("model_load", e.to_string()))?;
        *resident = Some((identity, backend));
    }
    check(cancel, deadline)?;
    let backend = &mut resident.as_mut().unwrap().1;
    // Always install the current cap, including cache hits. A rejected limit
    // must stop this call before prefill, never reuse the prior call's budget.
    backend
        .set_context_limit(config.max_context_tokens)
        .map_err(|e| Fault::new("model_context", e.to_string()))?;
    let family = backend.family();
    let count_prompt = backend.completion_prompt_counter();
    let prompt_budget = config.max_context_tokens.checked_sub(config.max_output_tokens as usize)
        .ok_or_else(|| Fault::invalid("output reservation exceeds native context"))?;
    let count_window = |candidate: &[Value]| {
        let rendered = render_prompt(candidate, tools, family)?;
        count_prompt(&rendered).map_err(|e| Fault::new("model_context", e.to_string()))
    };
    // akhomed owns eviction once it assembles to budget; the runtime then only
    // checks the fit against the model's own tokenizer and fails closed. The
    // local selector remains for payloads from a core that has not moved yet.
    let budgeted_by_core = crate::native_window::core_budgeted(messages);
    let window = if budgeted_by_core {
        crate::native_window::verify(messages, prompt_budget, count_window)?
    } else {
        crate::native_window::select(messages, prompt_budget, count_window)?
    };
    tracing::debug!(budgeted_by_core, dropped_messages = window.dropped_messages,
        prompt_tokens = window.prompt_tokens,
        "selected native prompt window; durable history retained");
    let messages = window.messages.as_slice();
    let (prompt, user_span) = render_prompt_with_user_span(messages, tools, family)?;
    if prompt.len() > MAX_BYTES {
        return Err(Fault::new(
            "capacity",
            "native prompt exceeds context byte budget",
        ));
    }
    let plan = recall
        .as_ref()
        .and_then(|ctx| capture_plan(config, backend, &prompt, user_span.as_ref(), ctx.input_text));
    let mut captured: Option<SpanCapture> = None;
    let mut recalled_message: Option<Value> = None;
    let result = {
        let capture_request = plan.as_ref().map(|plan| CaptureRequest {
            byte_span: plan.prompt_bytes.clone(),
            layers: &plan.policy.layers,
        });
        let plan = plan.as_ref();
        let recall = recall.as_ref();
        let prompt_text = prompt.as_str();
        backend.generate_completion_for_session_with_capture(
            session_key,
            &prompt,
            capture_request,
            config.max_output_tokens,
            &mut || {
                #[cfg(test)]
                GENERATION_CHECKS.fetch_add(1, Ordering::SeqCst);
                check(cancel, deadline).map_err(|e| NluError::LlmGenerationFailed {
                    reason: e.to_string(),
                })
            },
            &mut |taps| {
                let (Some(plan), Some(ctx), Some(taps)) = (plan, recall, taps) else {
                    return Ok(RecallDecision::Continue);
                };
                let vector = match plan.projector.project(&plan.policy, &taps.layers) {
                    Ok(vector) => vector,
                    Err(error) => {
                        tracing::warn!(%error, "neural span projection rejected the pooled taps");
                        return Ok(RecallDecision::Continue);
                    }
                };
                captured = Some(SpanCapture {
                    space: plan.space.clone(),
                    source: SpanSource::User,
                    byte_start: plan.input_bytes.start,
                    byte_end: plan.input_bytes.end,
                    vector: vector.clone(),
                });
                // A recall exchange is bounded network work under the compute
                // lane: honour cancellation before it, and never fail the turn on it.
                check(cancel, deadline).map_err(|e| NluError::LlmGenerationFailed {
                    reason: e.to_string(),
                })?;
                let hits = match host.recall(&ctx.call_id, &plan.space, &vector, MAX_RECALL_HITS) {
                    Ok(Some(hits)) => hits,
                    Ok(None) => {
                        tracing::debug!("session daemon does not serve neural recall");
                        return Ok(RecallDecision::Continue);
                    }
                    Err(error) => {
                        tracing::warn!(error, "neural recall failed; continuing without it");
                        return Ok(RecallDecision::Continue);
                    }
                };
                let recalled = crate::native_recall::assemble(
                    messages, &hits, &plan.space, &prompt_text[..plan.prompt_bytes.start],
                    prompt_budget, MAX_BYTES, |candidate| {
                        let rendered = render_prompt(candidate, tools, family)?;
                        let tokens = count_prompt(&rendered).map_err(|e| Fault::new("model_context", e.to_string()))?;
                        Ok((rendered, tokens))
                    },
                ).map_err(|e| NluError::LlmGenerationFailed { reason: e.to_string() })?;
                let Some(recalled) = recalled else { return Ok(RecallDecision::Continue); };
                tracing::debug!(recalled_spans = recalled.message["neural_recall"]["turn_ids"]
                    .as_array().map_or(0, Vec::len), "native recall materialized");
                recalled_message = Some(recalled.message);
                Ok(RecallDecision::Reprompt(recalled.prompt))
            },
        )
    };
    // Preserve canonical cancellation/timeout faults even when the backend wraps them.
    if let Err(error) = check(cancel, deadline) {
        backend.discard_completion_state();
        return Err(error);
    }
    let result = result.map_err(|e| Fault::new("model", e.to_string()))?;
    tracing::debug!(
        prompt_tokens = result.prompt_tokens,
        reused_prompt_tokens = result.reused_prompt_tokens,
        "native prompt prepared"
    );
    if !matches!(result.finish_reason, CandleFinishReason::Stop) {
        backend.discard_completion_state();
        return Err(Fault::new(
            "model_incomplete",
            "native model exhausted its output token budget",
        ));
    }
    let (message, finish) = match parse_response_with_tools(&result.text, tools) {
        Ok(parsed) => parsed,
        Err(error) => {
            backend.discard_completion_state();
            return Err(error);
        }
    };
    drop(resident);
    if let (Some(ctx), Some(capture)) = (recall.as_ref(), captured) {
        match host.spans(&ctx.call_id, &[capture]) {
            Ok(true) => tracing::debug!("neural span capture delivered"),
            Ok(false) => tracing::debug!("session daemon does not retain neural span captures"),
            Err(error) => tracing::warn!(error, "neural span delivery failed; turn continues"),
        }
    }
    if finish == "stop" {
        let text = message["content"].as_str().unwrap().to_owned();
        if !text.is_empty() {
            host.event(RuntimeEvent::Text { text })
                .map_err(Fault::invalid)?;
        }
    }
    Ok((message, finish, recalled_message))
}

fn verify_artifact(
    path: &std::path::Path,
    expected: &str,
    cancel: &AtomicBool,
    deadline: Instant,
) -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|_| Fault::new("model_load", "cannot read pinned native model artifact"))?;
    let mut hash = Sha256::new();
    let mut bytes = [0u8; 65536];
    loop {
        check(cancel, deadline)?;
        let count = file
            .read(&mut bytes)
            .map_err(|_| Fault::new("model_load", "cannot hash pinned native model artifact"))?;
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    if !format!("{:x}", hash.finalize()).eq_ignore_ascii_case(expected) {
        return Err(Fault::new(
            "model_identity",
            "native model artifact does not match its configured SHA256",
        ));
    }
    Ok(())
}

fn render_prompt(messages: &[Value], tools: &[Value], family: ModelFamily) -> Result<String> {
    render_prompt_with_user_span(messages, tools, family).map(|(prompt, _)| prompt)
}

/// Render and also report the byte range of the final message's content when
/// that message is the user turn; captures and recall are scoped to it.
fn render_prompt_with_user_span(
    messages: &[Value],
    tools: &[Value],
    family: ModelFamily,
) -> Result<(String, Option<std::ops::Range<usize>>)> {
    let (prompt, spans) = if matches!(family, ModelFamily::Qwen35) {
        render_qwen_xml_prompt(messages, tools)?
    } else {
        render_generic_prompt(messages, tools, family)?
    };
    let last_is_user = messages.last().is_some_and(|m| m["role"] == "user");
    Ok((prompt, spans.last().cloned().filter(|_| last_is_user)))
}

fn render_generic_prompt(
    messages: &[Value],
    tools: &[Value],
    family: ModelFamily,
) -> Result<(String, Vec<std::ops::Range<usize>>)> {
    let mut user_spans = Vec::new();
    let (system, user, assistant, close) = match family {
        ModelFamily::Qwen2 | ModelFamily::Qwen35 => (
            "<|im_start|>system\n",
            "<|im_start|>user\n",
            "<|im_start|>assistant\n",
            "<|im_end|>\n",
        ),
        ModelFamily::Gemma => (
            "<start_of_turn>user\n",
            "<start_of_turn>user\n",
            "<start_of_turn>model\n",
            "<end_of_turn>\n",
        ),
        ModelFamily::Gemma4 => (
            "<|turn>user\n",
            "<|turn>user\n",
            "<|turn>model\n",
            "<turn|>\n",
        ),
    };
    let mut prompt = String::new();
    let mut folded_system = false;
    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("user");
        if !folded_system {
            prompt.push_str(match role {
                "system" => system,
                "assistant" => assistant,
                _ => user,
            });
        }
        folded_system = false;
        if role == "tool" {
            prompt.push_str("<tool_response>\n");
        }
        let content_start = prompt.len();
        prompt.push_str(msg["content"].as_str().unwrap_or(""));
        if role == "user" {
            user_spans.push(content_start..prompt.len());
        }
        if role == "system" && !tools.is_empty() {
            prompt.push_str("\n\n# Tools\nAvailable functions (JSON schema):\n<tools>\n");
            for tool in tools {
                prompt.push_str(&tool["function"].to_string());
                prompt.push('\n');
            }
            prompt.push_str("</tools>\nTo call a function emit <tool_call>{\"name\":\"function_name\",\"arguments\":{}}</tool_call> with an arguments object matching its schema. Otherwise answer directly.\n");
        }
        if let Some(calls) = msg["tool_calls"].as_array() {
            for call in calls {
                let arguments: Value = serde_json::from_str(
                    call["function"]["arguments"]
                        .as_str()
                        .ok_or_else(|| Fault::invalid("invalid stored tool arguments"))?,
                )
                .map_err(|_| Fault::invalid("invalid stored tool arguments"))?;
                prompt.push_str("<tool_call>");
                prompt.push_str(
                    &json!({"name":call["function"]["name"],"arguments":arguments}).to_string(),
                );
                prompt.push_str("</tool_call>");
            }
        }
        if role == "tool" {
            prompt.push_str("\n</tool_response>");
        }
        if role == "system" && matches!(family, ModelFamily::Gemma | ModelFamily::Gemma4) {
            prompt.push_str("\n\n");
            folded_system = true;
        } else {
            prompt.push_str(close);
        }
    }
    prompt.push_str(assistant);
    Ok((prompt, user_spans))
}

/// Text-only projection of the pinned Qwen3.8 template, with thinking disabled.
/// Keep full tool wrappers, assistant thinking frames, and grouped tool results.
/// A later `system` message renders as its own system turn: that is how a
/// bounded recalled-context block is placed before the user turn.
fn render_qwen_xml_prompt(
    messages: &[Value],
    tools: &[Value],
) -> Result<(String, Vec<std::ops::Range<usize>>)> {
    let mut user_spans = Vec::new();
    let mut prompt = String::new();
    let system = messages
        .first()
        .filter(|m| m["role"] == "system")
        .and_then(|m| m["content"].as_str())
        .unwrap_or("")
        .trim();
    if !tools.is_empty() || !system.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        if !tools.is_empty() {
            prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
            for tool in tools {
                prompt.push('\n');
                prompt.push_str(&tool.to_string());
            }
            prompt.push_str("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>");
            if !system.is_empty() {
                prompt.push_str("\n\n");
            }
        }
        prompt.push_str(system);
        prompt.push_str("<|im_end|>\n");
    }
    for (index, msg) in messages.iter().enumerate() {
        let content = msg["content"].as_str().unwrap_or("").trim();
        match msg["role"].as_str() {
            Some("system") if index == 0 => {}
            Some("system") => {
                prompt.push_str("<|im_start|>system\n");
                prompt.push_str(content);
                prompt.push_str("<|im_end|>\n");
            }
            Some("user") => {
                prompt.push_str("<|im_start|>user\n");
                let start = prompt.len();
                prompt.push_str(content);
                user_spans.push(start..prompt.len());
                prompt.push_str("<|im_end|>\n");
            }
            Some("assistant") => {
                prompt.push_str("<|im_start|>assistant\n<think>\n");
                prompt.push_str(msg["reasoning_content"].as_str().unwrap_or("").trim());
                prompt.push_str("\n</think>\n\n");
                prompt.push_str(content);
                if let Some(calls) = msg["tool_calls"].as_array() {
                    for (i, call) in calls.iter().enumerate() {
                        if i != 0 {
                            prompt.push('\n');
                        } else if !content.is_empty() {
                            prompt.push_str("\n\n");
                        }
                        let arguments: serde_json::Map<String, Value> = serde_json::from_str(
                            call["function"]["arguments"]
                                .as_str()
                                .ok_or_else(|| Fault::invalid("stored tool arguments missing"))?,
                        )
                        .map_err(|_| Fault::invalid("stored tool arguments must be an object"))?;
                        prompt.push_str(&format!(
                            "<tool_call>\n<function={}>\n",
                            call["function"]["name"]
                                .as_str()
                                .ok_or_else(|| Fault::invalid("stored tool name missing"))?
                        ));
                        for (name, value) in arguments {
                            prompt.push_str(&format!(
                                "<parameter={name}>\n{}\n</parameter>\n",
                                value
                                    .as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| value.to_string())
                            ));
                        }
                        prompt.push_str("</function>\n</tool_call>");
                    }
                }
                prompt.push_str("<|im_end|>\n");
            }
            Some("tool") => {
                if index == 0 || messages[index - 1]["role"] != "tool" {
                    prompt.push_str("<|im_start|>user");
                }
                prompt.push_str("\n<tool_response>\n");
                prompt.push_str(content);
                prompt.push_str("\n</tool_response>");
                if messages.get(index + 1).is_none_or(|m| m["role"] != "tool") {
                    prompt.push_str("<|im_end|>\n");
                }
            }
            _ => return Err(Fault::invalid("invalid native conversation role")),
        }
    }
    prompt.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");
    Ok((prompt, user_spans))
}

/// Parse the entire output before exposing text or allowing any broker callback.
/// A malformed second call invalidates the whole response, including its first call.
#[cfg(test)]
fn parse_response(raw: &str) -> Result<(Value, String)> {
    parse_response_with_tools(raw, &[])
}

fn parse_response_with_tools(raw: &str, tools: &[Value]) -> Result<(Value, String)> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Call {
        name: String,
        arguments: serde_json::Map<String, Value>,
    }
    let invalid = || {
        Fault::new(
            "model_incomplete",
            "native model emitted malformed tool markup",
        )
    };
    if raw.len() > MAX_BYTES {
        return Err(invalid());
    }
    let answer = answer_only(raw)?;
    let mut rest = answer.as_str();
    let mut content = String::new();
    let mut calls = Vec::new();
    while let Some(start) = rest.find("<tool_call>") {
        content.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let end = after.find("</tool_call>").ok_or_else(invalid)?;
        let block = after[..end].trim();
        let call: Call = if block.starts_with("<function=") {
            let (name, arguments) = parse_xml_call(block, tools)?;
            Call { name, arguments }
        } else {
            serde_json::from_str(block).map_err(|_| invalid())?
        };
        if call.name.trim().is_empty() || call.name.len() > 512 {
            return Err(invalid());
        }
        calls.push(json!({"id":format!("native-{}", new_id()),"type":"function","function":{"name":call.name,"arguments":Value::Object(call.arguments).to_string()}}));
        rest = &after[end + "</tool_call>".len()..];
    }
    content.push_str(rest);
    if [
        "<tool",
        "</tool",
        "<function=",
        "</function",
        "<parameter=",
        "</parameter",
    ]
    .iter()
    .any(|marker| content.contains(marker))
        || content.trim_end().ends_with('<')
    {
        return Err(invalid());
    }
    let content = content.trim();
    if calls.is_empty() && content.is_empty() {
        return Err(Fault::new(
            "model_incomplete",
            "native model returned no final answer or tool call",
        ));
    }
    let finish = if calls.is_empty() {
        "stop"
    } else {
        "tool_calls"
    };
    Ok((
        json!({"role":"assistant","content":content,"tool_calls":calls}),
        finish.into(),
    ))
}

fn parse_xml_call(
    block: &str,
    tools: &[Value],
) -> Result<(String, serde_json::Map<String, Value>)> {
    let invalid = || {
        Fault::new(
            "model_incomplete",
            "native model emitted malformed XML tool call",
        )
    };
    let body = block.strip_prefix("<function=").ok_or_else(invalid)?;
    let (name, body) = body.split_once('>').ok_or_else(invalid)?;
    if name.is_empty()
        || name
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '<' | '>'))
    {
        return Err(invalid());
    }
    let mut rest = body.strip_suffix("</function>").ok_or_else(invalid)?.trim();
    let schema = tools
        .iter()
        .find(|t| t["function"]["name"] == name)
        .map(|t| &t["function"]["parameters"]);
    let mut arguments = serde_json::Map::new();
    while !rest.is_empty() {
        let body = rest.strip_prefix("<parameter=").ok_or_else(invalid)?;
        let (parameter, body) = body.split_once('>').ok_or_else(invalid)?;
        if parameter.is_empty()
            || parameter
                .chars()
                .any(|c| c.is_whitespace() || matches!(c, '<' | '>'))
        {
            return Err(invalid());
        }
        let (value, tail) = body.split_once("</parameter>").ok_or_else(invalid)?;
        if value.contains("<parameter=") || value.contains("<function=") {
            return Err(invalid());
        }
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        let kind = schema.and_then(|s| s["properties"][parameter]["type"].as_str());
        let value = if kind == Some("string") {
            Value::String(value.into())
        } else {
            serde_json::from_str(value)
                .or_else(|_| {
                    if kind.is_none() {
                        Ok(Value::String(value.into()))
                    } else {
                        Err(())
                    }
                })
                .map_err(|_| invalid())?
        };
        if arguments.insert(parameter.into(), value).is_some() {
            return Err(invalid());
        }
        rest = tail.trim();
    }
    Ok((name.into(), arguments))
}

/// Reasoning is never an executable action plan. Require complete channel
/// framing; tolerant display parsers are inappropriate at this effect boundary.
fn answer_only(raw: &str) -> Result<String> {
    let invalid = || {
        Fault::new(
            "model_incomplete",
            "native model emitted incomplete reasoning/channel markup",
        )
    };
    let mut answer = String::new();
    let mut rest = raw;
    let stray = |text: &str| {
        ["<think", "</think", "<|channel", "<channel|"]
            .iter()
            .any(|marker| text.contains(marker))
    };
    while !rest.is_empty() {
        let think = rest.find("<think>").map(|i| (i, true));
        let channel = rest.find("<|channel>").map(|i| (i, false));
        let next = match (think, channel) {
            (Some(a), Some(b)) => Some(if a.0 < b.0 { a } else { b }),
            (a, b) => a.or(b),
        };
        // Tool arguments are data. Do not strip reasoning-looking text inside
        // their envelope (e.g. a source file containing literal <think> tags).
        if let Some(tool_start) = rest.find("<tool_call>") {
            if next.is_none_or(|(start, _)| tool_start < start) {
                if stray(&rest[..tool_start]) {
                    return Err(invalid());
                }
                let after = &rest[tool_start + "<tool_call>".len()..];
                let end = after.find("</tool_call>").ok_or_else(invalid)?;
                let consumed = tool_start + "<tool_call>".len() + end + "</tool_call>".len();
                answer.push_str(&rest[..consumed]);
                rest = &rest[consumed..];
                continue;
            }
        }
        let Some((start, thinking)) = next else {
            if stray(rest) {
                return Err(invalid());
            }
            answer.push_str(rest);
            break;
        };
        if stray(&rest[..start]) {
            return Err(invalid());
        }
        answer.push_str(&rest[..start]);
        let after = &rest[start
            + if thinking {
                "<think>".len()
            } else {
                "<|channel>".len()
            }..];
        let close = if thinking { "</think>" } else { "<channel|>" };
        let end = after.find(close).ok_or_else(invalid)?;
        let block = &after[..end];
        if block.contains("<think>") || block.contains("<|channel>") {
            return Err(invalid());
        }
        if !thinking {
            let (name, body) = block.split_once('\n').ok_or_else(invalid)?;
            if name.trim() == "final" {
                answer.push_str(body);
            }
        }
        rest = &after[end + close.len()..];
    }
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn residency_ignores_call_limits_but_preserves_every_artifact_identity_field() {
        let original = NativeCandle {
            model: "ptah".into(),
            model_path: "/models/model.gguf".into(),
            model_sha256: "0".repeat(64),
            tokenizer_path: "/models/tokenizer.json".into(),
            tokenizer_sha256: "1".repeat(64),
            device: NativeCandleDevice::Cpu,
            max_context_tokens: 2048,
            max_iterations: 4,
            max_output_tokens: 128,
        };
        let identity = ResidentIdentity::from(&original);
        let mut another_call = original.clone();
        another_call.model = "same-weights-another-profile".into();
        another_call.max_context_tokens = 4096;
        another_call.max_iterations = 8;
        another_call.max_output_tokens = 512;
        assert_ne!(
            original, another_call,
            "durable continuation snapshots remain distinct"
        );
        assert_eq!(identity, ResidentIdentity::from(&another_call));
        let changes: [fn(&mut NativeCandle); 5] = [
            |c| c.model_path = "/models/other.gguf".into(),
            |c| c.model_sha256 = "2".repeat(64),
            |c| c.tokenizer_path = "/models/other-tokenizer.json".into(),
            |c| c.tokenizer_sha256 = "3".repeat(64),
            |c| c.device = NativeCandleDevice::Hip { ordinal: 0 },
        ];
        for change in changes {
            let mut different_artifact = original.clone();
            change(&mut different_artifact);
            assert_ne!(identity, ResidentIdentity::from(&different_artifact));
        }
    }
    #[test]
    fn device_admission_matches_compiled_features_without_initializing_hardware() {
        assert!(validate_device(&NativeCandleDevice::Cpu).is_ok());
        assert_eq!(
            validate_device(&NativeCandleDevice::Vulkan { ordinal: 0 }).is_ok(),
            cfg!(feature = "vulkan")
        );
        assert_eq!(
            validate_device(&NativeCandleDevice::Hip { ordinal: 0 }).is_ok(),
            cfg!(feature = "hip")
        );
        assert_eq!(
            validate_device(&NativeCandleDevice::Hip { ordinal: 1 })
                .unwrap_err()
                .code,
            "unsupported"
        );
        assert_eq!(
            validate_device(&NativeCandleDevice::Cuda { ordinal: 0 })
                .unwrap_err()
                .code,
            "unsupported"
        );
    }
    #[test]
    fn artifact_pins_reject_mismatch_and_unreadable_paths() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("artifact");
        std::fs::write(&path, b"abc").unwrap();
        let cancel = AtomicBool::new(false);
        let deadline = Instant::now() + Duration::from_secs(1);
        verify_artifact(&path, &crate::digest(b"abc"), &cancel, deadline).unwrap();
        assert_eq!(
            verify_artifact(&path, &crate::digest(b"other"), &cancel, deadline)
                .unwrap_err()
                .code,
            "model_identity"
        );
        assert_eq!(
            verify_artifact(
                &dir.path().join("missing"),
                &crate::digest(b"abc"),
                &cancel,
                deadline
            )
            .unwrap_err()
            .code,
            "model_load"
        );
        cancel.store(true, Ordering::SeqCst);
        assert_eq!(
            verify_artifact(&path, &crate::digest(b"abc"), &cancel, deadline)
                .unwrap_err()
                .code,
            "cancelled"
        );
    }
    #[test]
    fn qwen_template_groups_results_and_preserves_assistant_framing() {
        let tools = vec![
            json!({"type":"function","function":{"name":"read","parameters":{"type":"object"}}}),
        ];
        let messages = vec![
            json!({"role":"system","content":"  Be precise.  "}),
            json!({"role":"user","content":"Read both."}),
            json!({"role":"assistant","content":"Checking.","tool_calls":[{"function":{"name":"read","arguments":"{\"path\":\"a\"}"}},{"function":{"name":"read","arguments":"{\"path\":\"b\"}"}}]}),
            json!({"role":"tool","content":"A"}),
            json!({"role":"tool","content":"B"}),
        ];
        let prompt = render_prompt(&messages, &tools, ModelFamily::Qwen35).unwrap();
        assert!(prompt.starts_with("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"));
        assert!(prompt.contains(&tools[0].to_string()));
        assert!(prompt.contains("</IMPORTANT>\n\nBe precise.<|im_end|>\n"));
        assert!(
            prompt
                .contains("<|im_start|>assistant\n<think>\n\n</think>\n\nChecking.\n\n<tool_call>")
        );
        assert!(prompt.contains("<|im_start|>user\n<tool_response>\nA\n</tool_response>\n<tool_response>\nB\n</tool_response><|im_end|>\n"));
    }
    #[test]
    fn occupied_compute_lane_still_observes_cancellation_and_deadline() {
        let _occupied = MODEL.lock().unwrap();
        let cancel = AtomicBool::new(true);
        let fault = acquire_model(&cancel, Instant::now() + Duration::from_secs(1))
            .err()
            .unwrap();
        assert_eq!(fault.code, "cancelled");
        cancel.store(false, Ordering::SeqCst);
        let fault = acquire_model(&cancel, Instant::now()).err().unwrap();
        assert_eq!(fault.code, "timeout");
    }
    #[test]
    fn tool_output_is_all_or_nothing() {
        let good = "<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"x\"}}</tool_call>";
        let (message, finish) = parse_response(good).unwrap();
        assert_eq!(finish, "tool_calls");
        assert_eq!(
            message["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"x\"}"
        );
        for bad in [
            format!("{good}<tool_call>{{}}"),
            format!("{good}</tool_call>"),
            "<tool_call>{\"name\":\"read\",\"arguments\":\"{}\"}</tool_call>".into(),
        ] {
            assert!(parse_response(&bad).is_err());
        }
        assert_eq!(
            parse_response("First paragraph.\n\nSecond paragraph.")
                .unwrap()
                .0["content"],
            "First paragraph.\n\nSecond paragraph."
        );
    }
    #[test]
    fn hypothetical_tools_in_reasoning_never_execute() {
        let call = "<tool_call>{\"name\":\"read\",\"arguments\":{}}</tool_call>";
        for raw in [
            format!("<think>Consider {call}</think>No action."),
            format!("<|channel>thought\nConsider {call}<channel|>No action."),
        ] {
            let (message, finish) = parse_response(&raw).unwrap();
            assert_eq!(finish, "stop");
            assert_eq!(message["content"], "No action.");
        }
        assert!(parse_response(&format!("<think>{call}")).is_err());
        assert!(parse_response(&format!("<|channel>thought\n{call}")).is_err());
        let literal = "<tool_call>{\"name\":\"write\",\"arguments\":{\"text\":\"<think>literal</think>\"}}</tool_call>";
        let (message, _) = parse_response(literal).unwrap();
        let arguments: Value = serde_json::from_str(
            message["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(arguments["text"], "<think>literal</think>");
    }
    #[test]
    fn native_history_preserves_tool_arguments_as_objects() {
        let messages = vec![
            json!({"role":"assistant","tool_calls":[{"function":{"name":"read","arguments":"{\"path\":\"x\"}"}}]}),
            json!({"role":"tool","content":"result"}),
        ];
        let prompt = render_prompt(&messages, &[], ModelFamily::Qwen2).unwrap();
        assert!(prompt.contains("\"arguments\":{\"path\":\"x\"}"));
        assert!(prompt.contains("<tool_response>\nresult\n</tool_response>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        let prompt = render_prompt(&messages, &[], ModelFamily::Qwen35).unwrap();
        assert!(prompt.contains("<function=read>\n<parameter=path>\nx\n</parameter>\n</function>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
    #[test]
    fn qwen_xml_parameters_keep_schema_types_and_reject_duplicates() {
        let tools = vec![
            json!({"function":{"name":"read","parameters":{"properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}}}),
        ];
        let raw = "<tool_call>\n<function=read>\n<parameter=path>\n123\n</parameter>\n<parameter=limit>\n4\n</parameter>\n</function>\n</tool_call>";
        let (message, finish) = parse_response_with_tools(raw, &tools).unwrap();
        assert_eq!(finish, "tool_calls");
        let args: Value = serde_json::from_str(
            message["tool_calls"][0]["function"]["arguments"]
                .as_str()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(args, json!({"path":"123","limit":4}));
        assert!(
            parse_response_with_tools(&raw.replace("\n4\n", "\nnot-a-number\n"), &tools).is_err()
        );
        assert!(
            parse_response_with_tools(
                &raw.replace("<parameter=limit>", "<parameter=path>"),
                &tools
            )
            .is_err()
        );
        assert!(parse_response_with_tools(&raw.replace("</function>", ""), &tools).is_err());
    }
}
