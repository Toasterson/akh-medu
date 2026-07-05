//! Typed parsing of Gemma 4's channel-formatted responses (ADR 054).
//!
//! Gemma 4 emits a "harmony"-style channel stream that interleaves the model's
//! internal reasoning with the user-facing answer, e.g.
//!
//! ```text
//! <|channel>thought
//! Let me consider the three primary colors…<channel|>Red, blue, and green.
//! ```
//!
//! Earlier code simply stripped these markers and dropped the reasoning. That
//! reasoning is valuable: the chat layer wants to annotate the model's thought
//! process separately from its answer (collapsible "thinking" sections, trace
//! views, etc.). This module parses the raw output into a typed [`ChannelStream`]
//! of ordered [`ChannelSegment`]s so callers can render each channel distinctly
//! instead of discarding it.
//!
//! The parser is deliberately tolerant: malformed/unterminated channels and text
//! outside any channel block are still captured (as [`Channel::Final`]), so no
//! generated content is ever silently lost.

/// The open/close delimiters Gemma 4 uses around a channel block.
const OPEN: &str = "<|channel>";
const CLOSE: &str = "<channel|>";

/// The kind of a response channel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Channel {
    /// Internal reasoning (`<|channel>thought`). Not user-facing by default;
    /// the chat UI renders it as a collapsible "thinking" annotation.
    Thought,
    /// The user-facing answer (`<|channel>final`, or any text outside a block).
    Final,
    /// Any other named channel the model emits (kept verbatim for forward-compat).
    Other(String),
}

impl Channel {
    /// Classify a raw channel name. Unknown names are preserved as [`Channel::Other`].
    fn from_name(name: &str) -> Self {
        match name.trim() {
            "thought" | "thinking" | "analysis" | "reasoning" => Channel::Thought,
            "final" | "answer" | "response" | "" => Channel::Final,
            other => Channel::Other(other.to_string()),
        }
    }

    /// True for the internal-reasoning channel.
    pub fn is_thought(&self) -> bool {
        matches!(self, Channel::Thought)
    }

    /// A short stable label (for UI / serialization).
    pub fn label(&self) -> &str {
        match self {
            Channel::Thought => "thought",
            Channel::Final => "final",
            Channel::Other(s) => s.as_str(),
        }
    }
}

/// One typed segment of a channel-formatted response, in emission order.
#[derive(Debug, Clone)]
pub struct ChannelSegment {
    pub channel: Channel,
    pub text: String,
}

/// A parsed channel stream: the model's output decomposed into ordered, typed
/// segments. Construct with [`ChannelStream::parse`].
#[derive(Debug, Clone, Default)]
pub struct ChannelStream {
    pub segments: Vec<ChannelSegment>,
}

impl ChannelStream {
    /// Parse raw model output into typed segments.
    ///
    /// Text outside any `<|channel>…<channel|>` block is captured as
    /// [`Channel::Final`] (the answer). Unterminated blocks run to end-of-input.
    pub fn parse(raw: &str) -> Self {
        let mut segments = Vec::new();
        let mut cursor = 0usize;

        while let Some(rel) = raw[cursor..].find(OPEN) {
            let open_abs = cursor + rel;
            // Any text before this channel block belongs to the default (final) answer.
            let pre = &raw[cursor..open_abs];
            push_trimmed(&mut segments, Channel::Final, pre);

            let after_open = open_abs + OPEN.len();
            let rest = &raw[after_open..];
            // The channel name runs to the first newline, or to the close marker.
            let (name, content_start_rel) = match rest.find('\n') {
                Some(nl) => (&rest[..nl], nl + 1),
                None => match rest.find(CLOSE) {
                    Some(c) => (&rest[..c], c),
                    None => (rest, rest.len()),
                },
            };
            let content_start = after_open + content_start_rel;
            // Content runs to the close marker, or to end-of-input if unterminated.
            let (content, next_cursor) = match raw[content_start..].find(CLOSE) {
                Some(c) => (
                    &raw[content_start..content_start + c],
                    content_start + c + CLOSE.len(),
                ),
                None => (&raw[content_start..], raw.len()),
            };
            push_trimmed(&mut segments, Channel::from_name(name), content);
            cursor = next_cursor;
        }

        // Trailing default text after the last block.
        if cursor < raw.len() {
            push_trimmed(&mut segments, Channel::Final, &raw[cursor..]);
        }
        ChannelStream { segments }
    }

    /// The user-facing answer: only [`Channel::Final`] segments (text outside any
    /// block is `Final`). Thought and other side-channels are excluded.
    pub fn answer(&self) -> String {
        join(
            self.segments
                .iter()
                .filter(|s| matches!(s.channel, Channel::Final))
                .map(|s| s.text.as_str()),
            " ",
        )
    }

    /// The model's reasoning: all thought segments, joined by newlines.
    pub fn thoughts(&self) -> String {
        join(
            self.segments
                .iter()
                .filter(|s| s.channel.is_thought())
                .map(|s| s.text.as_str()),
            "\n",
        )
    }

    /// Whether the model emitted any internal reasoning.
    pub fn has_thoughts(&self) -> bool {
        self.segments.iter().any(|s| s.channel.is_thought())
    }
}

/// Push a segment only if its trimmed text is non-empty (drops empty channels).
fn push_trimmed(segments: &mut Vec<ChannelSegment>, channel: Channel, text: &str) {
    let t = text.trim();
    if !t.is_empty() {
        segments.push(ChannelSegment {
            channel,
            text: t.to_string(),
        });
    }
}

fn join<'a>(parts: impl Iterator<Item = &'a str>, sep: &str) -> String {
    parts.collect::<Vec<_>>().join(sep).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_thought_then_answer() {
        let raw = "<|channel>thought\nConsider primary colors.<channel|>Red, blue, and green.";
        let s = ChannelStream::parse(raw);
        assert_eq!(s.segments.len(), 2);
        assert_eq!(s.segments[0].channel, Channel::Thought);
        assert_eq!(s.segments[0].text, "Consider primary colors.");
        assert_eq!(s.segments[1].channel, Channel::Final);
        assert_eq!(s.answer(), "Red, blue, and green.");
        assert_eq!(s.thoughts(), "Consider primary colors.");
        assert!(s.has_thoughts());
    }

    #[test]
    fn empty_thought_channel_then_answer() {
        // The common Gemma 4 greeting shape: empty thought, then the answer.
        let raw = "<|channel>thought\n<channel|>Hello there, I am Akh.";
        let s = ChannelStream::parse(raw);
        assert_eq!(s.answer(), "Hello there, I am Akh.");
        assert!(!s.has_thoughts()); // empty thought dropped
    }

    #[test]
    fn plain_text_no_channels_is_final() {
        let s = ChannelStream::parse("Paris is the capital of France.");
        assert_eq!(s.answer(), "Paris is the capital of France.");
        assert!(!s.has_thoughts());
    }

    #[test]
    fn unterminated_channel_runs_to_end() {
        let s = ChannelStream::parse("<|channel>thought\nstill thinking");
        assert_eq!(s.thoughts(), "still thinking");
        assert_eq!(s.answer(), "");
    }

    #[test]
    fn unknown_channel_preserved() {
        let s = ChannelStream::parse("<|channel>critic\nhmm<channel|>ok");
        assert_eq!(s.segments[0].channel, Channel::Other("critic".to_string()));
        assert_eq!(s.answer(), "ok");
    }
}
