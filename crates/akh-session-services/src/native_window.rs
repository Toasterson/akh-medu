//! A bounded prompt view, never a mutation of the durable conversation.
use crate::types::{Fault, Result};
use serde_json::Value;

pub(crate) struct Window {
    pub messages: Vec<Value>,
    pub dropped_messages: usize,
    pub prompt_tokens: usize,
}

/// Count the fully rendered prompt, including tool schemas and template tokens.
/// Unmarked system messages are pinned; recalled system messages belong to
/// the following user turn. The newest user turn and all its tool traffic stay.
pub(crate) fn select(
    messages: &[Value],
    budget: usize,
    mut count: impl FnMut(&[Value]) -> Result<usize>,
) -> Result<Window> {
    let mut selected = messages.to_vec();
    let mut tokens = count(&selected)?;
    let mut starts = Vec::new();
    for (i, message) in messages.iter().enumerate() {
        if message["role"] == "user" {
            let mut start = i;
            while start > 0
                && messages[start - 1]["role"] == "system"
                && messages[start - 1].get("neural_recall").is_some()
            {
                start -= 1;
            }
            starts.push(start);
        }
    }
    let mut keep = vec![true; messages.len()];
    for turns in starts.windows(2) {
        if tokens <= budget {
            break;
        }
        for i in turns[0]..turns[1] {
            let pinned =
                messages[i]["role"] == "system" && messages[i].get("neural_recall").is_none();
            if !pinned {
                keep[i] = false;
            }
        }
        selected = messages
            .iter()
            .zip(&keep)
            .filter(|(_, keep)| **keep)
            .map(|(message, _)| message.clone())
            .collect();
        tokens = count(&selected)?;
    }
    if tokens > budget {
        return Err(Fault::new(
            "capacity",
            "pinned native context and current turn exceed prompt budget",
        ));
    }
    Ok(Window {
        dropped_messages: messages.len() - selected.len(),
        messages: selected,
        prompt_tokens: tokens,
    })
}

/// True when akhomed marked this message list as already assembled to budget.
///
/// Absent on the legacy payload, which keeps a mixed akhomed/akhrtd pair
/// interoperating while the seam moves, the same discipline the span annex used.
pub(crate) fn core_budgeted(messages: &[Value]) -> bool {
    messages
        .iter()
        .any(|message| message.get(akh_session_types::CONTEXT_BUDGETED) == Some(&Value::Bool(true)))
}

/// Check a prompt the cognitive core already budgeted, without ever dropping.
///
/// akhomed owns which turns and recalled spans survive; a runtime that quietly
/// evicts on top of that decision is a second context policy. This verifies the
/// core's decision against the model's own tokenizer and fails the turn when it
/// does not fit, so an overflow surfaces instead of silently losing context.
pub(crate) fn verify(
    messages: &[Value],
    budget: usize,
    mut count: impl FnMut(&[Value]) -> Result<usize>,
) -> Result<Window> {
    let tokens = count(messages)?;
    if tokens > budget {
        return Err(Fault::new(
            "capacity",
            format!("assembled context is {tokens} tokens against a {budget} token prompt budget"),
        ));
    }
    Ok(Window {
        messages: messages.to_vec(),
        dropped_messages: 0,
        prompt_tokens: tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn evicts_a_whole_old_turn_without_mutating_history() {
        let history = vec![
            json!({"role":"system","content":"rules"}),
            json!({"role":"user","content":"old fact"}),
            json!({"role":"assistant","tool_calls":[{"id":"a"}]}),
            json!({"role":"tool","tool_call_id":"a","content":"result"}),
            json!({"role":"assistant","content":"done"}),
            json!({"role":"user","content":"current"}),
        ];
        let window = select(&history, 2, |m| Ok(m.len())).unwrap();
        assert_eq!(
            window.messages,
            vec![history[0].clone(), history[5].clone()]
        );
        assert_eq!(window.dropped_messages, 4);
        assert_eq!(window.prompt_tokens, 2);
        assert_eq!(history.len(), 6);
    }

    #[test]
    fn preserves_current_tools_and_all_unmarked_system_instructions() {
        let history = vec![
            json!({"role":"system"}),
            json!({"role":"user"}),
            json!({"role":"system","content":"new safety rule"}),
            json!({"role":"user"}),
            json!({"role":"assistant","tool_calls":[{}]}),
            json!({"role":"tool"}),
        ];
        assert_eq!(
            select(&history, 4, |m| Ok(m.len())).err().unwrap().code,
            "capacity"
        );
        let result = select(&history, 5, |m| Ok(m.len())).unwrap();
        assert_eq!(
            result.messages,
            vec![
                history[0].clone(),
                history[2].clone(),
                history[3].clone(),
                history[4].clone(),
                history[5].clone()
            ]
        );
    }

    #[test]
    fn core_budgeted_marker_is_opt_in_and_absent_on_legacy_payloads() {
        assert!(!core_budgeted(&[json!({"role":"user","content":"hi"})]));
        assert!(!core_budgeted(&[
            json!({"role":"system","context_budgeted":false})
        ]));
        assert!(core_budgeted(&[
            json!({"role":"system","context_budgeted":true}),
            json!({"role":"user","content":"hi"})
        ]));
    }

    #[test]
    fn verify_accepts_a_budgeted_prompt_and_never_drops() {
        let history = vec![
            json!({"role":"system","content":"rules"}),
            json!({"role":"user","content":"current"}),
        ];
        let window = verify(&history, 2, |m| Ok(m.len())).unwrap();
        assert_eq!(window.messages, history);
        assert_eq!(window.dropped_messages, 0);
        assert_eq!(window.prompt_tokens, 2);
    }

    #[test]
    fn verify_fails_closed_instead_of_evicting_what_the_core_chose() {
        let history = vec![
            json!({"role":"system","content":"rules"}),
            json!({"role":"user","content":"old"}),
            json!({"role":"assistant","content":"done"}),
            json!({"role":"user","content":"current"}),
        ];
        // select() would drop the old turn to fit; verify() must not.
        assert_eq!(
            select(&history, 2, |m| Ok(m.len()))
                .unwrap()
                .dropped_messages,
            2
        );
        let fault = verify(&history, 2, |m| Ok(m.len())).err().unwrap();
        assert_eq!(fault.code, "capacity");
        assert!(fault.message.contains("4 tokens"), "{}", fault.message);
        assert_eq!(
            verify(&history, 100, |_| Err(Fault::invalid("counter failed")))
                .err()
                .unwrap()
                .message,
            "counter failed"
        );
    }

    #[test]
    fn recall_moves_with_its_user_turn_and_count_errors_fail_closed() {
        let history = vec![
            json!({"role":"system","neural_recall":{}}),
            json!({"role":"user","content":"old"}),
            json!({"role":"assistant"}),
            json!({"role":"system","neural_recall":{}}),
            json!({"role":"user","content":"new"}),
        ];
        assert_eq!(
            select(&history, 2, |m| Ok(m.len())).unwrap().messages,
            history[3..]
        );
        assert_eq!(
            select(&history, 100, |_| Err(Fault::invalid("counter failed")))
                .err()
                .unwrap()
                .message,
            "counter failed"
        );
        assert_eq!(
            select(&history, 5, |m| Ok(m.len())).unwrap().messages,
            history
        );
    }
}
