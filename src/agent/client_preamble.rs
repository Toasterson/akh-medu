//! Client-injected prompt preamble (Lilith C2 operator decision 2026-10-01).
//!
//! Some ACP clients (the T3 app) append provider text to every operator
//! prompt as a trailing run of top-level XML-tag blocks, for example
//! `<runtime_info>...</runtime_info>` and
//! `<pull_request_linking>...</pull_request_linking>`. An akh that opts in
//! (`[acp] strip_client_preamble = true`) does not give that run to the
//! model: it is cut off the user text before the turn, and the turn's captured
//! transcript keeps the raw prompt plus the block names and size as client
//! metadata.
//!
//! The rule is structural, never a list of known tag names, and it mirrors
//! `operator_text` in `scripts/lilith/c2/harvest_sessions.py` so the harvester
//! and the core agree on what the operator's own words are:
//!
//! - a block is `<name>...</name>` where `name` is a lowercase
//!   `snake_case` identifier (`[a-z][a-z0-9]*(_[a-z0-9]+)*`) and the closing
//!   tag is the first `</name>` after the opening tag;
//! - the opening tag starts the text or follows whitespace;
//! - from that tag to the end, the text is blocks and whitespace only;
//! - the *earliest* such tag starts the preamble; tags earlier in the text
//!   that do not lead to the end stay;
//! - a prompt that is nothing but preamble is left alone (the turn must keep
//!   some text).

use std::cell::RefCell;

/// What was cut from a prompt, kept as client metadata (never the content).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ClientPreambleMeta {
    /// Tag names of the removed top-level blocks, in order.
    pub tags: Vec<String>,
    /// Bytes removed, including the whitespace between the operator text and
    /// the first block.
    pub bytes: usize,
}

/// A prompt split into the operator's words and the client preamble.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreambleSplit<'a> {
    /// No trailing block run, or nothing would remain: the prompt is
    /// unchanged.
    Untouched,
    /// The prompt's own text (non-empty, trailing whitespace trimmed) and what
    /// was cut.
    Stripped {
        operator_text: &'a str,
        meta: ClientPreambleMeta,
    },
}

/// A lowercase `snake_case` tag name at the start of `text` (after `<`),
/// returned with the length of the whole opening tag `<name>`.
fn open_tag(text: &str) -> Option<(&str, usize)> {
    let rest = text.strip_prefix('<')?;
    let end = rest.find('>')?;
    let name = &rest[..end];
    is_snake_case(name).then_some((name, end + 2))
}

fn is_snake_case(name: &str) -> bool {
    let mut segments = name.split('_');
    let first_ok = segments.next().is_some_and(|first| {
        first.starts_with(|c: char| c.is_ascii_lowercase())
            && first
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
    });
    first_ok
        && segments.all(|segment| {
            !segment.is_empty()
                && segment
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        })
}

/// The tag names of the blocks that run from `start` to the end of `text`,
/// or `None` when anything but whitespace follows a block, a block is not
/// closed, or no block starts at `start`.
fn blocks_to_end(text: &str, start: usize) -> Option<Vec<String>> {
    let mut tags = Vec::new();
    let mut pos = start;
    loop {
        match open_tag(&text[pos..]) {
            None => {
                return (text[pos..].trim().is_empty() && !tags.is_empty()).then_some(tags);
            }
            Some((name, open_len)) => {
                let body_start = pos + open_len;
                let close = format!("</{name}>");
                let at = text[body_start..].find(&close)?;
                pos = body_start + at + close.len();
                tags.push(name.to_string());
                pos += text[pos..].len() - text[pos..].trim_start().len();
            }
        }
    }
}

/// Split the trailing client preamble off `prompt`.
pub fn split_client_preamble(prompt: &str) -> PreambleSplit<'_> {
    for (at, _) in prompt.match_indices('<') {
        let at_boundary = prompt[..at]
            .chars()
            .next_back()
            .is_none_or(char::is_whitespace);
        if !at_boundary || open_tag(&prompt[at..]).is_none() {
            continue;
        }
        let Some(tags) = blocks_to_end(prompt, at) else {
            continue;
        };
        let operator_text = prompt[..at].trim_end();
        if operator_text.is_empty() {
            return PreambleSplit::Untouched;
        }
        return PreambleSplit::Stripped {
            operator_text,
            meta: ClientPreambleMeta {
                tags,
                bytes: prompt.len() - operator_text.len(),
            },
        };
    }
    PreambleSplit::Untouched
}

// ── Per-turn raw capture ────────────────────────────────────────────────────

/// The raw prompt of the turn running on this thread, with what was cut.
#[derive(Debug, Clone)]
struct StrippedTurn {
    /// The text the model got.
    operator_text: String,
    /// The prompt as the client sent it.
    raw: String,
    meta: ClientPreambleMeta,
}

thread_local! {
    static STRIPPED_TURN: RefCell<Option<StrippedTurn>> = const { RefCell::new(None) };
}

/// Restores the previous per-turn record when dropped.
#[must_use = "the record lasts until the guard drops"]
pub struct StrippedTurnGuard {
    previous: Option<StrippedTurn>,
}

impl Drop for StrippedTurnGuard {
    fn drop(&mut self) {
        STRIPPED_TURN.with(|cell| *cell.borrow_mut() = self.previous.take());
    }
}

/// Apply the opt-in policy to one turn's prompt.
///
/// Returns the text the turn runs on and, when something was cut, a guard
/// that makes the transcript capture on this thread record the raw prompt
/// and the client metadata. Hold the guard for the whole turn. An ACP turn
/// runs on one `spawn_blocking` thread (see `crate::fs::set_session_cwd`),
/// so the record cannot leak into another session's turn.
pub fn apply_client_preamble_policy(
    strip: bool,
    prompt: String,
) -> (String, Option<StrippedTurnGuard>) {
    if !strip {
        return (prompt, None);
    }
    let PreambleSplit::Stripped {
        operator_text,
        meta,
    } = split_client_preamble(&prompt)
    else {
        return (prompt, None);
    };
    tracing::info!(
        tags = ?meta.tags,
        bytes = meta.bytes,
        "stripped client-injected preamble from the operator prompt"
    );
    let record = StrippedTurn {
        operator_text: operator_text.to_string(),
        raw: prompt.clone(),
        meta,
    };
    let text = record.operator_text.clone();
    let previous = STRIPPED_TURN.with(|cell| cell.borrow_mut().replace(record));
    (text, Some(StrippedTurnGuard { previous }))
}

/// For the capture hook: the raw prompt and metadata when `turn_prompt` is
/// the text this thread's turn was stripped to.
pub(crate) fn raw_prompt_for(turn_prompt: &str) -> Option<(String, ClientPreambleMeta)> {
    STRIPPED_TURN.with(|cell| {
        cell.borrow()
            .as_ref()
            .filter(|turn| turn.operator_text == turn_prompt)
            .map(|turn| (turn.raw.clone(), turn.meta.clone()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RUNTIME_INFO: &str = "<runtime_info>In case you're asked: you are running in T3 Code through the Grok harness.</runtime_info>";
    const PR_LINKING: &str = "<pull_request_linking>\nWhen the t3-code MCP server exposes link_pull_request, use it.\n</pull_request_linking>";

    fn stripped(prompt: &str) -> (&str, Vec<String>) {
        match split_client_preamble(prompt) {
            PreambleSplit::Stripped {
                operator_text,
                meta,
            } => (operator_text, meta.tags),
            PreambleSplit::Untouched => panic!("expected a strip of {prompt:?}"),
        }
    }

    #[test]
    fn the_two_real_t3_blocks_are_stripped_with_their_names_and_size() {
        let prompt = format!("Yes handle and upark 4 items {RUNTIME_INFO}\n\n{PR_LINKING}");
        let PreambleSplit::Stripped {
            operator_text,
            meta,
        } = split_client_preamble(&prompt)
        else {
            panic!("not stripped");
        };
        assert_eq!(operator_text, "Yes handle and upark 4 items");
        assert_eq!(meta.tags, ["runtime_info", "pull_request_linking"]);
        assert_eq!(meta.bytes, prompt.len() - operator_text.len());
    }

    #[test]
    fn one_block_and_any_snake_case_name_are_stripped() {
        assert_eq!(
            stripped(&format!("Hello lilith {RUNTIME_INFO}")),
            ("Hello lilith", vec!["runtime_info".to_string()])
        );
        assert_eq!(
            stripped("Fix it <some_new_block>anything</some_new_block>").1,
            ["some_new_block"]
        );
    }

    #[test]
    fn tags_that_are_not_trailing_are_untouched() {
        for prompt in [
            "Compare <runtime_info>x</runtime_info> with the old output, then stop",
            "No preamble at all",
            "",
        ] {
            assert_eq!(
                split_client_preamble(prompt),
                PreambleSplit::Untouched,
                "{prompt:?}"
            );
        }
    }

    #[test]
    fn malformed_or_non_snake_case_blocks_are_untouched() {
        for prompt in [
            "Fix it <runtime_info>never closed",
            "Fix it <RuntimeInfo>x</RuntimeInfo>",
            "Fix it <run-time>x</run-time>",
            "Fix it <_x>y</_x>",
            "Fix it <a__b>y</a__b>",
            "a<runtime_info>x</runtime_info>",
            "Fix it <runtime_info>x</runtime_info> and then more words",
        ] {
            assert_eq!(
                split_client_preamble(prompt),
                PreambleSplit::Untouched,
                "{prompt:?}"
            );
        }
    }

    #[test]
    fn a_nested_block_is_one_top_level_block() {
        let prompt = "Do it <outer><inner>x</inner> y</outer>";
        assert_eq!(stripped(prompt), ("Do it", vec!["outer".to_string()]));
    }

    #[test]
    fn an_earlier_inline_block_stays_when_only_the_last_leads_to_the_end() {
        let prompt = "See <a_b>x</a_b> then <runtime_info>y</runtime_info>";
        assert_eq!(
            stripped(prompt),
            ("See <a_b>x</a_b> then", vec!["runtime_info".to_string()])
        );
    }

    #[test]
    fn a_prompt_that_is_only_preamble_is_never_emptied() {
        assert_eq!(
            split_client_preamble(RUNTIME_INFO),
            PreambleSplit::Untouched
        );
        let both = format!("  {RUNTIME_INFO}\n{PR_LINKING}");
        assert_eq!(split_client_preamble(&both), PreambleSplit::Untouched);
    }

    #[test]
    fn the_policy_is_off_by_default_and_a_no_op_without_preamble() {
        let prompt = format!("Hi {RUNTIME_INFO}");
        let (text, guard) = apply_client_preamble_policy(false, prompt.clone());
        assert_eq!(text, prompt);
        assert!(guard.is_none());
        let (plain, guard) = apply_client_preamble_policy(true, "Hi".to_string());
        assert_eq!(plain, "Hi");
        assert!(guard.is_none());
    }

    #[test]
    fn the_capture_sees_the_raw_prompt_only_for_the_stripped_turn() {
        let prompt = format!("Hi {RUNTIME_INFO}");
        let (text, guard) = apply_client_preamble_policy(true, prompt.clone());
        assert_eq!(text, "Hi");
        let (raw, meta) = raw_prompt_for("Hi").expect("recorded while the guard lives");
        assert_eq!(raw, prompt);
        assert_eq!(meta.tags, ["runtime_info"]);
        assert!(raw_prompt_for("a different prompt").is_none());
        drop(guard);
        assert!(raw_prompt_for("Hi").is_none());
    }
}
