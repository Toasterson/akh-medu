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
