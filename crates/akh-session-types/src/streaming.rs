//! Paragraph-granularity coalescing of native-runtime text streams.
//!
//! Providers emit a text or reasoning delta every few bytes. Forwarding each
//! one costs a durable store transaction on the runtime, a persisted session
//! event, an ACP notification and an OTLP log record. [`ParagraphCoalescer`]
//! buffers consecutive deltas of one [`DeltaKind`] and releases them as
//! [`StreamChunk`]s at *structural* boundaries only:
//!
//! - a paragraph boundary: a line break followed by an empty line (`\n\n`,
//!   `\r\n\r\n`, or `\n\r\n`), cut after the whole run of line breaks;
//! - a kind switch (text to reasoning or back);
//! - a size cap (`max_bytes`), cut on a UTF-8 character boundary and after the
//!   last line break inside the cap when there is one;
//! - a latency cap: `idle_flush_ms` without a new delta, or `max_latency_ms`
//!   since the oldest buffered byte arrived;
//! - an explicit [`ParagraphCoalescer::flush`] (a non-text event is about to
//!   be forwarded, or the turn ends).
//!
//! Boundaries never depend on words or punctuation, so the behaviour is the
//! same for every natural language. The coalescer is a pure state machine:
//! time is passed in by the caller, so tests drive it with a fabricated clock.
use crate::RuntimeEvent;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// How the runtime host forwards provider text and reasoning deltas.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StreamingMode {
    /// Coalesce deltas into paragraphs (see the module documentation).
    #[default]
    Paragraph,
    /// Forward every provider delta unchanged, exactly as it arrived.
    Token,
}

/// The `[streaming]` section of a runtime service configuration.
///
/// Every field has a default, so existing configurations without the section
/// keep loading and get paragraph streaming.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct StreamingConfig {
    #[serde(default)]
    pub mode: StreamingMode,
    /// Largest chunk the coalescer releases, in bytes.
    #[serde(default = "StreamingConfig::default_max_bytes")]
    pub max_bytes: usize,
    /// Release a buffered partial paragraph after this long without a delta.
    #[serde(default = "StreamingConfig::default_idle_flush_ms")]
    pub idle_flush_ms: u64,
    /// Release a buffered partial paragraph at most this long after its oldest
    /// byte arrived, even while deltas keep arriving.
    #[serde(default = "StreamingConfig::default_max_latency_ms")]
    pub max_latency_ms: u64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            mode: StreamingMode::default(),
            max_bytes: Self::DEFAULT_MAX_BYTES,
            idle_flush_ms: Self::DEFAULT_IDLE_FLUSH_MS,
            max_latency_ms: Self::DEFAULT_MAX_LATENCY_MS,
        }
    }
}

impl StreamingConfig {
    pub const DEFAULT_MAX_BYTES: usize = 4096;
    pub const DEFAULT_IDLE_FLUSH_MS: u64 = 1500;
    pub const DEFAULT_MAX_LATENCY_MS: u64 = 3000;
    /// Upper bound of the flush tick, so an idle deadline is observed at most
    /// this late.
    const MAX_TICK: Duration = Duration::from_millis(250);
    /// Lower bound of the flush tick, so tiny test deadlines do not spin.
    const MIN_TICK: Duration = Duration::from_millis(5);

    fn default_max_bytes() -> usize {
        Self::DEFAULT_MAX_BYTES
    }
    fn default_idle_flush_ms() -> u64 {
        Self::DEFAULT_IDLE_FLUSH_MS
    }
    fn default_max_latency_ms() -> u64 {
        Self::DEFAULT_MAX_LATENCY_MS
    }

    /// Token mode, the pre-coalescing behaviour.
    pub fn token() -> Self {
        Self {
            mode: StreamingMode::Token,
            ..Self::default()
        }
    }

    pub fn validate(&self) -> Result<(), StreamingConfigError> {
        if self.max_bytes == 0 {
            return Err(StreamingConfigError::ZeroMaxBytes);
        }
        if self.idle_flush_ms == 0 {
            return Err(StreamingConfigError::ZeroIdleFlush);
        }
        if self.max_latency_ms == 0 {
            return Err(StreamingConfigError::ZeroMaxLatency);
        }
        if self.idle_flush_ms > self.max_latency_ms {
            return Err(StreamingConfigError::IdleExceedsMaxLatency {
                idle_flush_ms: self.idle_flush_ms,
                max_latency_ms: self.max_latency_ms,
            });
        }
        Ok(())
    }

    pub fn idle_flush(&self) -> Duration {
        Duration::from_millis(self.idle_flush_ms)
    }

    pub fn max_latency(&self) -> Duration {
        Duration::from_millis(self.max_latency_ms)
    }

    /// How often a host without its own event loop should check deadlines:
    /// a quarter of the idle deadline, clamped to 5..=250 ms. A deadline is
    /// therefore observed at most one tick late.
    pub fn tick_interval(&self) -> Duration {
        (self.idle_flush().min(self.max_latency()) / 4).clamp(Self::MIN_TICK, Self::MAX_TICK)
    }
}

/// A `[streaming]` section the runtime cannot honour.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, miette::Diagnostic)]
pub enum StreamingConfigError {
    #[error("streaming.max_bytes must be greater than zero")]
    #[diagnostic(
        code(akh::runtime::streaming::zero_max_bytes),
        help(
            "set streaming.max_bytes to a positive byte count (default 4096), or set \
             streaming.mode = \"token\" to forward every provider delta unchanged"
        )
    )]
    ZeroMaxBytes,
    #[error("streaming.idle_flush_ms must be greater than zero")]
    #[diagnostic(
        code(akh::runtime::streaming::zero_idle_flush),
        help(
            "set streaming.idle_flush_ms to a positive number of milliseconds (default 1500), \
             or set streaming.mode = \"token\" to disable buffering"
        )
    )]
    ZeroIdleFlush,
    #[error("streaming.max_latency_ms must be greater than zero")]
    #[diagnostic(
        code(akh::runtime::streaming::zero_max_latency),
        help(
            "set streaming.max_latency_ms to a positive number of milliseconds (default 3000), \
             or set streaming.mode = \"token\" to disable buffering"
        )
    )]
    ZeroMaxLatency,
    #[error(
        "streaming.idle_flush_ms ({idle_flush_ms}) exceeds streaming.max_latency_ms ({max_latency_ms})"
    )]
    #[diagnostic(
        code(akh::runtime::streaming::idle_exceeds_max_latency),
        help(
            "the idle deadline can never fire after the latency cap; lower idle_flush_ms \
             or raise max_latency_ms so that idle_flush_ms <= max_latency_ms"
        )
    )]
    IdleExceedsMaxLatency {
        idle_flush_ms: u64,
        max_latency_ms: u64,
    },
}

/// The two streamed event kinds. They are buffered separately: a switch
/// flushes the other kind first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeltaKind {
    Text,
    Reasoning,
}

impl DeltaKind {
    /// The kind and text of a streamed event; `None` for every other event.
    pub fn of(event: &RuntimeEvent) -> Option<(Self, &str)> {
        match event {
            RuntimeEvent::Text { text } => Some((Self::Text, text)),
            RuntimeEvent::Reasoning { text } => Some((Self::Reasoning, text)),
            _ => None,
        }
    }

    /// Stable, bounded label for telemetry.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Reasoning => "reasoning",
        }
    }

    pub fn event(self, text: String) -> RuntimeEvent {
        match self {
            Self::Text => RuntimeEvent::Text { text },
            Self::Reasoning => RuntimeEvent::Reasoning { text },
        }
    }
}

/// Why a chunk was released.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlushReason {
    /// Token mode: the delta itself.
    Passthrough,
    /// A paragraph boundary was reached.
    Paragraph,
    /// The buffer reached `max_bytes`.
    SizeCap,
    /// The other delta kind arrived.
    KindSwitch,
    /// No delta for `idle_flush_ms`.
    Idle,
    /// The oldest buffered byte is `max_latency_ms` old.
    MaxLatency,
    /// A non-text event is about to be forwarded.
    Barrier,
    /// The turn ended.
    Finish,
}

impl FlushReason {
    /// Stable, bounded label for telemetry.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Passthrough => "passthrough",
            Self::Paragraph => "paragraph",
            Self::SizeCap => "size_cap",
            Self::KindSwitch => "kind_switch",
            Self::Idle => "idle",
            Self::MaxLatency => "max_latency",
            Self::Barrier => "barrier",
            Self::Finish => "finish",
        }
    }
}

/// One coalesced event, ready to forward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamChunk {
    pub kind: DeltaKind,
    pub text: String,
    /// Provider deltas attributed to this chunk. Each non-empty delta is
    /// counted exactly once, in the first chunk that carries its bytes, so the
    /// sum over all chunks equals the number of deltas received.
    pub deltas: usize,
    pub reason: FlushReason,
}

impl StreamChunk {
    pub fn into_event(self) -> RuntimeEvent {
        self.kind.event(self.text)
    }
}

#[derive(Debug)]
struct Pending {
    kind: DeltaKind,
    text: String,
    deltas: usize,
    /// When the oldest buffered byte arrived (conservative after a split).
    first_at: Instant,
    last_at: Instant,
    /// Offset where the most recent delta's bytes begin.
    last_delta_start: usize,
    /// Offset from which the next boundary scan starts.
    scanned: usize,
}

/// Buffers text and reasoning deltas and releases them at paragraph
/// boundaries. See the module documentation for the flush rules.
#[derive(Debug)]
pub struct ParagraphCoalescer {
    config: StreamingConfig,
    pending: Option<Pending>,
}

impl ParagraphCoalescer {
    pub fn new(config: StreamingConfig) -> Result<Self, StreamingConfigError> {
        config.validate()?;
        Ok(Self {
            config,
            pending: None,
        })
    }

    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Whether any bytes are buffered.
    pub fn is_empty(&self) -> bool {
        self.pending.is_none()
    }

    /// Accept one provider delta and return the chunks it releases, in order.
    ///
    /// In token mode the delta is returned unchanged (even when empty). In
    /// paragraph mode overdue deadlines are honoured first, then empty deltas
    /// are dropped and the rest is buffered and split.
    pub fn push(&mut self, kind: DeltaKind, text: &str, now: Instant) -> Vec<StreamChunk> {
        if self.config.mode == StreamingMode::Token {
            return vec![StreamChunk {
                kind,
                text: text.to_owned(),
                deltas: 1,
                reason: FlushReason::Passthrough,
            }];
        }
        let mut out = Vec::new();
        out.extend(self.poll(now));
        if text.is_empty() {
            return out;
        }
        if self.pending.as_ref().is_some_and(|p| p.kind != kind) {
            out.extend(self.flush(FlushReason::KindSwitch));
        }
        let pending = self.pending.get_or_insert_with(|| Pending {
            kind,
            text: String::new(),
            deltas: 0,
            first_at: now,
            last_at: now,
            last_delta_start: 0,
            scanned: 0,
        });
        pending.last_delta_start = pending.text.len();
        pending.text.push_str(text);
        pending.deltas += 1;
        pending.last_at = now;
        self.split(&mut out);
        out
    }

    /// Release the buffer if its idle or latency deadline has passed.
    pub fn poll(&mut self, now: Instant) -> Option<StreamChunk> {
        let pending = self.pending.as_ref()?;
        if now >= pending.first_at + self.config.max_latency() {
            self.flush(FlushReason::MaxLatency)
        } else if now >= pending.last_at + self.config.idle_flush() {
            self.flush(FlushReason::Idle)
        } else {
            None
        }
    }

    /// The next instant at which [`Self::poll`] would release the buffer.
    pub fn deadline(&self) -> Option<Instant> {
        self.pending.as_ref().map(|p| {
            (p.first_at + self.config.max_latency()).min(p.last_at + self.config.idle_flush())
        })
    }

    /// Release everything buffered, whatever its boundaries.
    pub fn flush(&mut self, reason: FlushReason) -> Option<StreamChunk> {
        self.pending.take().map(|p| StreamChunk {
            kind: p.kind,
            text: p.text,
            deltas: p.deltas,
            reason,
        })
    }

    fn split(&mut self, out: &mut Vec<StreamChunk>) {
        let max_bytes = self.config.max_bytes;
        while let Some(pending) = self.pending.as_mut() {
            let (cut, reason) = if let Some(cut) = paragraph_cut(&pending.text, pending.scanned) {
                (cut, FlushReason::Paragraph)
            } else if pending.text.len() >= max_bytes {
                (size_cut(&pending.text, max_bytes), FlushReason::SizeCap)
            } else {
                // A boundary is at most three bytes (`\n\r\n`), so one that is
                // still incomplete starts within the last two bytes.
                pending.scanned = pending.text.len().saturating_sub(2);
                return;
            };
            let rest = pending.text.split_off(cut);
            let chunk = StreamChunk {
                kind: pending.kind,
                text: std::mem::replace(&mut pending.text, rest),
                deltas: std::mem::take(&mut pending.deltas),
                reason,
            };
            out.push(chunk);
            if pending.text.is_empty() {
                self.pending = None;
            } else {
                // The remainder came entirely from the latest delta only when
                // the cut lies inside it; otherwise keep the older timestamp so
                // the latency cap is never exceeded.
                if cut >= pending.last_delta_start {
                    pending.first_at = pending.last_at;
                }
                pending.last_delta_start = pending.last_delta_start.saturating_sub(cut);
                pending.scanned = 0;
            }
        }
    }
}

/// End of the first paragraph boundary at or after `from`: a `\n`, an
/// optional `\r`, another `\n`, then any further run of line-break bytes.
fn paragraph_cut(text: &str, from: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut index = from;
    while index < bytes.len() {
        if bytes[index] == b'\n' {
            let mut next = index + 1;
            if bytes.get(next) == Some(&b'\r') {
                next += 1;
            }
            if bytes.get(next) == Some(&b'\n') {
                let mut end = next + 1;
                while matches!(bytes.get(end), Some(b'\n' | b'\r')) {
                    end += 1;
                }
                return Some(end);
            }
        }
        index += 1;
    }
    None
}

/// Where to cut an oversized buffer: after the last `\n` within `max_bytes`,
/// otherwise at the last character boundary within `max_bytes`. A cap smaller
/// than the first character releases that one character whole.
fn size_cut(text: &str, max_bytes: usize) -> usize {
    let mut window = max_bytes.min(text.len());
    while !text.is_char_boundary(window) {
        window -= 1;
    }
    if let Some(line) = text[..window].rfind('\n') {
        return line + 1;
    }
    if window > 0 {
        return window;
    }
    text.chars().next().map_or(text.len(), char::len_utf8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paragraph() -> ParagraphCoalescer {
        ParagraphCoalescer::new(StreamingConfig::default()).unwrap()
    }

    fn texts(chunks: &[StreamChunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    /// Stream `text` in `size`-byte pieces (on char boundaries), then finish.
    fn stream(coalescer: &mut ParagraphCoalescer, text: &str, size: usize) -> Vec<StreamChunk> {
        let now = Instant::now();
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut cut = size.min(rest.len());
            while !rest.is_char_boundary(cut) {
                cut += 1;
            }
            out.extend(coalescer.push(DeltaKind::Text, &rest[..cut], now));
            rest = &rest[cut..];
        }
        out.extend(coalescer.flush(FlushReason::Finish));
        out
    }

    #[test]
    fn defaults_are_paragraph_mode_and_valid() {
        let config: StreamingConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config, StreamingConfig::default());
        assert_eq!(config.mode, StreamingMode::Paragraph);
        assert_eq!(config.max_bytes, 4096);
        assert_eq!(config.idle_flush_ms, 1500);
        assert_eq!(config.max_latency_ms, 3000);
        config.validate().unwrap();
        assert_eq!(config.tick_interval(), Duration::from_millis(250));
        let token: StreamingConfig = serde_json::from_str(r#"{"mode":"token"}"#).unwrap();
        assert_eq!(token, StreamingConfig::token());
        assert!(serde_json::from_str::<StreamingConfig>(r#"{"typo":1}"#).is_err());
    }

    #[test]
    fn validation_reports_each_invalid_setting_with_a_diagnostic() {
        use miette::Diagnostic;
        let cases = [
            (
                StreamingConfig {
                    max_bytes: 0,
                    ..StreamingConfig::default()
                },
                StreamingConfigError::ZeroMaxBytes,
            ),
            (
                StreamingConfig {
                    idle_flush_ms: 0,
                    ..StreamingConfig::default()
                },
                StreamingConfigError::ZeroIdleFlush,
            ),
            (
                StreamingConfig {
                    max_latency_ms: 0,
                    idle_flush_ms: 0,
                    ..StreamingConfig::default()
                },
                StreamingConfigError::ZeroIdleFlush,
            ),
            (
                StreamingConfig {
                    idle_flush_ms: 4000,
                    ..StreamingConfig::default()
                },
                StreamingConfigError::IdleExceedsMaxLatency {
                    idle_flush_ms: 4000,
                    max_latency_ms: 3000,
                },
            ),
        ];
        for (config, expected) in cases {
            let error = config.validate().unwrap_err();
            assert_eq!(error, expected);
            assert!(error.code().is_some() && error.help().is_some());
            assert!(ParagraphCoalescer::new(config).is_err());
        }
        let zero_latency = StreamingConfig {
            max_latency_ms: 0,
            ..StreamingConfig::default()
        };
        assert_eq!(
            zero_latency.validate().unwrap_err(),
            StreamingConfigError::ZeroMaxLatency
        );
    }

    #[test]
    fn paragraph_boundaries_split_and_keep_the_remainder() {
        let mut c = paragraph();
        let now = Instant::now();
        let out = c.push(DeltaKind::Text, "one\n\ntwo", now);
        assert_eq!(texts(&out), ["one\n\n"]);
        assert_eq!(out[0].reason, FlushReason::Paragraph);
        let out = c.push(DeltaKind::Text, " more\r\n\r\nthree\n\n\nfour", now);
        assert_eq!(texts(&out), ["two more\r\n\r\n", "three\n\n\n"]);
        assert_eq!(
            c.flush(FlushReason::Finish).map(|c| c.text).as_deref(),
            Some("four")
        );
        assert!(c.is_empty());
    }

    #[test]
    fn a_boundary_split_across_deltas_is_found() {
        for pieces in [
            &["para\n", "\nnext"][..],
            &["para\r\n", "\r\nnext"],
            &["para\r", "\n\r", "\nnext"],
            &["para\n\r", "\nnext"],
        ] {
            let mut c = paragraph();
            let now = Instant::now();
            let mut out = Vec::new();
            for piece in pieces {
                out.extend(c.push(DeltaKind::Text, piece, now));
            }
            assert_eq!(out.len(), 1, "{pieces:?}");
            assert!(out[0].text.starts_with("para") && out[0].text.ends_with('\n'));
            assert_eq!(out[0].deltas, pieces.len());
            assert_eq!(
                c.flush(FlushReason::Finish).unwrap().text,
                "next",
                "{pieces:?}"
            );
        }
    }

    #[test]
    fn a_single_line_break_is_not_a_paragraph() {
        let mut c = paragraph();
        let now = Instant::now();
        assert!(c.push(DeltaKind::Text, "a\nb\r\nc\n", now).is_empty());
        assert_eq!(c.flush(FlushReason::Finish).unwrap().text, "a\nb\r\nc\n");
    }

    #[test]
    fn the_size_cap_prefers_the_last_line_break_and_respects_utf8() {
        let config = StreamingConfig {
            max_bytes: 10,
            ..StreamingConfig::default()
        };
        let mut c = ParagraphCoalescer::new(config).unwrap();
        let now = Instant::now();
        let out = c.push(DeltaKind::Text, "abc\ndefghijkl", now);
        assert_eq!(texts(&out), ["abc\n"]);
        assert_eq!(out[0].reason, FlushReason::SizeCap);
        c.flush(FlushReason::Finish);

        // "é" is two bytes and "€" three: the cap must never split them.
        let text = "ééééé€€€€ü";
        let mut c = ParagraphCoalescer::new(config).unwrap();
        let out = stream(&mut c, text, 3);
        assert!(out.iter().all(|chunk| chunk.text.len() <= 10), "{out:?}");
        assert_eq!(
            out.iter().map(|c| c.text.as_str()).collect::<String>(),
            text
        );
        assert_eq!(out[0].text, "ééééé");

        // A cap smaller than one character still releases whole characters.
        let tiny = StreamingConfig {
            max_bytes: 1,
            ..StreamingConfig::default()
        };
        let mut c = ParagraphCoalescer::new(tiny).unwrap();
        let out = c.push(DeltaKind::Text, "€a", now);
        assert_eq!(texts(&out), ["€", "a"]);
    }

    #[test]
    fn a_kind_switch_flushes_the_other_kind_first() {
        let mut c = paragraph();
        let now = Instant::now();
        assert!(c.push(DeltaKind::Reasoning, "think", now).is_empty());
        assert!(c.push(DeltaKind::Reasoning, "ing", now).is_empty());
        let out = c.push(DeltaKind::Text, "answer", now);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, DeltaKind::Reasoning);
        assert_eq!(out[0].text, "thinking");
        assert_eq!(out[0].deltas, 2);
        assert_eq!(out[0].reason, FlushReason::KindSwitch);
        let rest = c.flush(FlushReason::Barrier).unwrap();
        assert_eq!((rest.kind, rest.text.as_str()), (DeltaKind::Text, "answer"));
    }

    #[test]
    fn idle_and_max_latency_deadlines_release_partial_paragraphs() {
        let mut c = paragraph();
        let start = Instant::now();
        c.push(DeltaKind::Text, "partial", start);
        assert_eq!(c.deadline(), Some(start + Duration::from_millis(1500)));
        assert!(c.poll(start + Duration::from_millis(1499)).is_none());
        let idle = c.poll(start + Duration::from_millis(1500)).unwrap();
        assert_eq!(
            (idle.text.as_str(), idle.reason),
            ("partial", FlushReason::Idle)
        );
        assert!(c.deadline().is_none());

        // Steady deltas never go idle; the latency cap still releases them.
        let mut c = paragraph();
        let mut released = Vec::new();
        for step in 0..=30 {
            let now = start + Duration::from_millis(step * 100);
            released.extend(c.push(DeltaKind::Text, "x", now));
        }
        assert_eq!(released.len(), 1);
        assert_eq!(released[0].reason, FlushReason::MaxLatency);
        assert_eq!(
            released[0].text.len(),
            30,
            "flushed at t=3000 before the delta"
        );
        assert_eq!(c.flush(FlushReason::Finish).unwrap().text, "x");
    }

    #[test]
    fn a_remainder_from_an_old_delta_keeps_its_latency_clock() {
        let config = StreamingConfig {
            max_bytes: 4,
            ..StreamingConfig::default()
        };
        let mut c = ParagraphCoalescer::new(config).unwrap();
        let start = Instant::now();
        assert!(c.push(DeltaKind::Text, "abc", start).is_empty());
        let later = start + Duration::from_millis(1000);
        let out = c.push(DeltaKind::Text, "defgh", later);
        assert_eq!(texts(&out), ["abcd", "efgh"]);
        assert!(c.is_empty());

        let mut c = ParagraphCoalescer::new(StreamingConfig::default()).unwrap();
        c.push(DeltaKind::Text, "a\n\nb", start);
        // The remainder "b" arrived with the latest delta, so its clock is
        // that delta's arrival, not the older paragraph's.
        assert_eq!(c.deadline(), Some(start + Duration::from_millis(1500)));
    }

    #[test]
    fn finishing_never_loses_bytes_and_counts_each_delta_once() {
        let text = "Erster Absatz mit Umlauten: äöü.\n\nSecond paragraph — with a dash.\r\n\r\n第三段。\n\nlast line without a boundary";
        for size in 1..=9 {
            let mut c = paragraph();
            let out = stream(&mut c, text, size);
            assert_eq!(
                out.iter().map(|c| c.text.as_str()).collect::<String>(),
                text
            );
            let deltas = {
                let mut n = 0;
                let mut rest = text;
                while !rest.is_empty() {
                    let mut cut = size.min(rest.len());
                    while !rest.is_char_boundary(cut) {
                        cut += 1;
                    }
                    rest = &rest[cut..];
                    n += 1;
                }
                n
            };
            assert_eq!(out.iter().map(|c| c.deltas).sum::<usize>(), deltas);
            assert_eq!(out.len(), 4, "size {size}: {out:?}");
        }
    }

    #[test]
    fn token_mode_is_an_exact_passthrough() {
        let mut c = ParagraphCoalescer::new(StreamingConfig::token()).unwrap();
        let now = Instant::now();
        let pieces = ["a", "", "b\n\n", "c"];
        let mut out = Vec::new();
        for piece in pieces {
            out.extend(c.push(DeltaKind::Text, piece, now));
        }
        assert_eq!(texts(&out), pieces);
        assert!(
            out.iter()
                .all(|c| c.deltas == 1 && c.reason == FlushReason::Passthrough)
        );
        assert!(c.is_empty());
        assert!(c.flush(FlushReason::Finish).is_none());
        assert!(c.poll(now + Duration::from_secs(60)).is_none());
    }

    #[test]
    fn empty_deltas_are_dropped_in_paragraph_mode() {
        let mut c = paragraph();
        let now = Instant::now();
        assert!(c.push(DeltaKind::Reasoning, "", now).is_empty());
        assert!(c.is_empty());
    }
}
