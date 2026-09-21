//! Ranked recall and window selection are one decision: dedupe cannot outlive eviction.
use crate::{
    native_window,
    types::{
        Fault, Result,
        neural_spans::{SpanHit, SpanSource, SpanSpace},
    },
};
use serde_json::{Value, json};

pub(crate) struct RecalledPrompt {
    pub prompt: String,
    pub message: Value,
}

pub(crate) fn assemble(
    messages: &[Value],
    hits: &[SpanHit],
    space: &SpanSpace,
    visible: &str,
    prompt_budget: usize,
    max_prompt_bytes: usize,
    mut render: impl FnMut(&[Value]) -> Result<(String, usize)>,
) -> Result<Option<RecalledPrompt>> {
    if messages
        .last()
        .is_none_or(|message| message["role"] != "user")
    {
        return Err(Fault::invalid("recall requires a current user turn"));
    }
    for count in (1..=hits.len().min(4)).rev() {
        let selected = &hits[..count];
        // A deduped attempt is cheap when history remains unchanged. If it
        // evicts history, the second attempt pins all selected evidence.
        for visible in [visible, ""] {
            let message = match materialize(selected, space, visible) {
                Ok(Some(message)) => message,
                Ok(None) => return Ok(None),
                Err(error) if error.code == "capacity" => break,
                Err(error) => return Err(error),
            };
            let mut rewritten = messages.to_vec();
            rewritten.insert(rewritten.len() - 1, message.clone());
            let window = match native_window::select(&rewritten, prompt_budget, |candidate| {
                render(candidate).map(|(_, count)| count)
            }) {
                Ok(window) => window,
                Err(error) if error.code == "capacity" => break,
                Err(error) => return Err(error),
            };
            if window.dropped_messages > 0 && !visible.is_empty() {
                continue;
            }
            let (prompt, tokens) = render(&window.messages)?;
            if prompt.len() > max_prompt_bytes || tokens > prompt_budget {
                break;
            }
            if selected
                .iter()
                .any(|hit| !hit.text.trim().is_empty() && !prompt.contains(hit.text.trim()))
            {
                break;
            }
            return Ok(Some(RecalledPrompt { prompt, message }));
        }
    }
    Ok(None)
}

fn materialize(hits: &[SpanHit], space: &SpanSpace, visible: &str) -> Result<Option<Value>> {
    let mut block =
        "Recalled from earlier in this conversation (verbatim, for context only):\n".to_owned();
    let mut ids = Vec::new();
    for hit in hits {
        let text = hit.text.trim();
        if text.is_empty() || visible.contains(text) {
            continue;
        }
        let source = match hit.source {
            SpanSource::User => "user",
            SpanSource::Assistant => "assistant",
        };
        block.push_str(&format!("- [{source}] {text}\n"));
        if block.len() > 4096 {
            return Err(Fault::new("capacity", "recalled notes exceed byte budget"));
        }
        ids.push(hit.turn_id.clone());
    }
    if ids.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        json!({"role":"system","content":block,"neural_recall":{
            "turn_ids":ids,"policy_sha256":space.policy_sha256,"projector_sha256":space.projector_sha256
        }}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn space() -> SpanSpace {
        SpanSpace {
            model_sha256: "1".repeat(64),
            tokenizer_sha256: "2".repeat(64),
            policy_sha256: "3".repeat(64),
            projector_sha256: "4".repeat(64),
        }
    }
    fn hit(id: &str, text: &str) -> SpanHit {
        SpanHit {
            turn_id: id.into(),
            source: SpanSource::User,
            byte_start: 0,
            byte_end: text.len(),
            text: text.into(),
            hamming_distance: 0,
        }
    }
    fn render(messages: &[Value]) -> Result<(String, usize)> {
        let text = messages
            .iter()
            .map(|m| m["content"].as_str().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("|");
        let count = text.len();
        Ok((text, count))
    }
    #[test]
    fn retains_the_top_fact_when_all_notes_do_not_fit() {
        let messages = vec![
            json!({"role":"system","content":"rules"}),
            json!({"role":"user","content":"question"}),
        ];
        let hits = vec![hit("best", "port 4317"), hit("lower", &"x".repeat(100))];
        let result = assemble(&messages, &hits, &space(), "rules", 120, 4096, render)
            .unwrap()
            .unwrap();
        assert!(result.prompt.contains("port 4317"));
        assert!(!result.prompt.contains(&"x".repeat(100)));
        assert_eq!(result.message["neural_recall"]["turn_ids"], json!(["best"]));
    }

    #[test]
    fn deduped_fact_is_repinned_when_other_notes_evict_its_turn() {
        let messages = vec![
            json!({"role":"system","content":"rules"}),
            json!({"role":"user","content":"port 4317"}),
            json!({"role":"assistant","content":"a".repeat(70)}),
            json!({"role":"user","content":"question"}),
        ];
        let hits = vec![hit("best", "port 4317"), hit("next", "new evidence")];
        let result = assemble(
            &messages,
            &hits,
            &space(),
            "rules port 4317",
            145,
            4096,
            render,
        )
        .unwrap()
        .unwrap();
        assert!(result.prompt.contains("port 4317"));
        assert!(result.prompt.contains("new evidence"));
        assert!(!result.prompt.contains(&"a".repeat(70)));
        assert_eq!(
            result.message["neural_recall"]["turn_ids"],
            json!(["best", "next"])
        );
    }

    #[test]
    fn byte_limit_prunes_lower_notes_and_never_splits_the_top_note() {
        let messages = vec![json!({"role":"user","content":"question"})];
        let hits = vec![hit("best", "port 4317"), hit("huge", &"x".repeat(5000))];
        let result = assemble(&messages, &hits, &space(), "", 20000, 4096, render)
            .unwrap()
            .unwrap();
        assert_eq!(result.message["neural_recall"]["turn_ids"], json!(["best"]));
        assert!(
            assemble(&messages, &hits[1..], &space(), "", 20000, 4096, render)
                .unwrap()
                .is_none()
        );
        assert!(
            assemble(&messages, &hits[..1], &space(), "", 20, 4096, render)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn visible_notes_are_noops_and_render_errors_propagate() {
        let messages = vec![json!({"role":"user","content":"question"})];
        let hits = vec![hit("best", "port 4317")];
        assert!(
            assemble(
                &messages,
                &hits,
                &space(),
                "port 4317",
                200,
                4096,
                |_| panic!("visible recall must not rerender")
            )
            .unwrap()
            .is_none()
        );
        let error = assemble(&messages, &hits, &space(), "", 200, 4096, |_| {
            Err(Fault::invalid("bad encoding"))
        })
        .err()
        .unwrap();
        assert_eq!(error.message, "bad encoding");
    }
}
