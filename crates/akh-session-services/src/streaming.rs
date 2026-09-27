//! Paragraph streaming at the runtime host boundary.
//!
//! Every native-runtime event (Codex, Claude, internal) reaches akhrtd through
//! one `Host::event` call. [`DeltaStream`] sits there, in front of telemetry
//! and the durable job event log, so everything downstream (the persisted
//! runtime events, akhsessd's session events, ACP `agent_message_chunk` /
//! `agent_thought_chunk` notifications, OTLP delta records) sees coalesced
//! paragraphs instead of provider tokens. The flush rules live in
//! [`ParagraphCoalescer`].
//!
//! Ordering: text and reasoning are the only buffered events. Any other event
//! and every other host interaction (tool callback, approval, question, usage,
//! recall, context assembly) first releases the buffer, under the same lock
//! that appends the event, so no event ever overtakes text produced before it
//! and nothing non-text waits on a deadline.
//!
//! Timers: the host is synchronous — a provider that goes quiet mid-paragraph
//! makes no further `Host` calls. Deadlines are therefore checked on every
//! incoming event *and* by a lightweight flush ticker, one thread per active
//! turn owned by the runtime worker. It wakes every
//! [`StreamingConfig::tick_interval`] (a quarter of the idle deadline, at most
//! 250 ms), polls the coalescer under the lock, and exits as soon as the turn
//! finishes. A turn's end always flushes; a sink failure seen by the ticker is
//! kept and returned by the next host call, exactly as an inline failure would
//! have been.
use akh_session_types::{
    RuntimeEvent,
    streaming::{DeltaKind, FlushReason, ParagraphCoalescer, StreamChunk, StreamingMode},
};
use std::{
    sync::{Arc, Mutex, MutexGuard, mpsc},
    thread::JoinHandle,
    time::Instant,
};

/// Source of "now" for deadline checks; injectable for tests.
pub trait Clock: Send + Sync + 'static {
    fn now(&self) -> Instant;
}

/// Wall-clock monotonic time.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// One item leaving the coalescer, in stream order.
#[derive(Debug, Clone, PartialEq)]
pub enum Outgoing {
    /// A released text or reasoning chunk.
    Chunk(StreamChunk),
    /// Any other runtime event, forwarded unchanged.
    Event(RuntimeEvent),
}

impl Outgoing {
    pub fn into_event(self) -> RuntimeEvent {
        match self {
            Self::Chunk(chunk) => chunk.into_event(),
            Self::Event(event) => event,
        }
    }
}

/// Ordered, durable destination of the coalesced stream.
pub trait EventSink: Send + 'static {
    fn append(&mut self, item: Outgoing) -> Result<(), String>;
}

struct Shared<S> {
    coalescer: ParagraphCoalescer,
    sink: S,
    /// First sink failure. Sticky: once the durable log refused an event,
    /// nothing later may be appended out of order after the gap.
    failed: Option<String>,
}

impl<S: EventSink> Shared<S> {
    fn check(&self) -> Result<(), String> {
        self.failed.clone().map_or(Ok(()), Err)
    }
    fn append(&mut self, item: Outgoing) -> Result<(), String> {
        self.sink.append(item).inspect_err(|error| {
            self.failed.get_or_insert_with(|| error.clone());
        })
    }
    fn release(&mut self, chunk: Option<StreamChunk>) -> Result<(), String> {
        chunk.map_or(Ok(()), |chunk| self.append(Outgoing::Chunk(chunk)))
    }
    fn poll(&mut self, now: Instant) -> Result<(), String> {
        self.check()?;
        let chunk = self.coalescer.poll(now);
        self.release(chunk)
    }
}

struct FlushTicker {
    stop: mpsc::Sender<()>,
    thread: JoinHandle<()>,
}

/// The coalescing front of a runtime host's event path. See the module docs.
pub struct DeltaStream<S: EventSink> {
    shared: Arc<Mutex<Shared<S>>>,
    clock: Arc<dyn Clock>,
    ticker: Option<FlushTicker>,
    finished: bool,
}

impl<S: EventSink> DeltaStream<S> {
    pub fn new(coalescer: ParagraphCoalescer, sink: S, clock: Arc<dyn Clock>) -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                coalescer,
                sink,
                failed: None,
            })),
            clock,
            ticker: None,
            finished: false,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Shared<S>> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Start the flush ticker (paragraph mode only; token mode never buffers).
    /// If the thread cannot be spawned, deadlines are still honoured on the
    /// next incoming event and the turn end still flushes.
    pub fn start_ticker(&mut self, span: tracing::Span) {
        if self.ticker.is_some() || self.finished {
            return;
        }
        let interval = {
            let shared = self.lock();
            let config = shared.coalescer.config();
            if config.mode == StreamingMode::Token {
                return;
            }
            config.tick_interval()
        };
        let (stop, stopped) = mpsc::channel::<()>();
        let shared = self.shared.clone();
        let clock = self.clock.clone();
        let spawned = std::thread::Builder::new()
            .name("akh-stream-flush".into())
            .spawn(move || {
                while let Err(mpsc::RecvTimeoutError::Timeout) = stopped.recv_timeout(interval) {
                    let mut shared = shared.lock().unwrap_or_else(|e| e.into_inner());
                    if shared.failed.is_some() {
                        continue;
                    }
                    if let Err(error) = span.in_scope(|| shared.poll(clock.now())) {
                        tracing::warn!(
                            event.name = "runtime.delta.flush_failed",
                            error,
                            "deadline flush failed; the next runtime event reports it"
                        );
                    }
                }
            });
        match spawned {
            Ok(thread) => self.ticker = Some(FlushTicker { stop, thread }),
            Err(error) => tracing::warn!(
                event.name = "runtime.delta.ticker_unavailable",
                error = %error,
                "stream flush ticker unavailable; deadlines are checked on the next event"
            ),
        }
    }

    /// Route one runtime event. Text and reasoning are coalesced; any other
    /// event flushes the buffer first and is then appended unchanged.
    pub fn event(&self, event: RuntimeEvent) -> Result<(), String> {
        let mut shared = self.lock();
        shared.check()?;
        let now = self.clock.now();
        match DeltaKind::of(&event) {
            Some((kind, text)) => {
                for chunk in shared.coalescer.push(kind, text, now) {
                    shared.append(Outgoing::Chunk(chunk))?;
                }
                Ok(())
            }
            None => {
                let chunk = shared.coalescer.flush(FlushReason::Barrier);
                shared.release(chunk)?;
                shared.append(Outgoing::Event(event))
            }
        }
    }

    /// Release buffered text before a host interaction that is not an event
    /// (tool callback, approval, question, usage, recall, context assembly).
    pub fn barrier(&self) -> Result<(), String> {
        let mut shared = self.lock();
        shared.check()?;
        let chunk = shared.coalescer.flush(FlushReason::Barrier);
        shared.release(chunk)
    }

    /// Honour idle and latency deadlines now. The ticker calls the same logic.
    pub fn tick(&self) -> Result<(), String> {
        let now = self.clock.now();
        self.lock().poll(now)
    }

    /// Stop the ticker and release everything still buffered. Idempotent.
    pub fn finish(&mut self) -> Result<(), String> {
        if let Some(ticker) = self.ticker.take() {
            drop(ticker.stop);
            if ticker.thread.join().is_err() {
                tracing::warn!("stream flush ticker panicked");
            }
        }
        if self.finished {
            return self.lock().check();
        }
        self.finished = true;
        let mut shared = self.lock();
        shared.check()?;
        let chunk = shared.coalescer.flush(FlushReason::Finish);
        shared.release(chunk)
    }
}

impl<S: EventSink> Drop for DeltaStream<S> {
    fn drop(&mut self) {
        if let Err(error) = self.finish() {
            tracing::warn!(
                event.name = "runtime.delta.flush_failed",
                error,
                "final stream flush failed while dropping the runtime host"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use akh_session_types::streaming::StreamingConfig;
    use std::time::Duration;

    #[derive(Clone, Default)]
    struct Recorder(Arc<Mutex<Vec<Outgoing>>>);
    impl Recorder {
        fn items(&self) -> Vec<Outgoing> {
            self.0.lock().unwrap().clone()
        }
        fn events(&self) -> Vec<RuntimeEvent> {
            self.items().into_iter().map(Outgoing::into_event).collect()
        }
    }
    impl EventSink for Recorder {
        fn append(&mut self, item: Outgoing) -> Result<(), String> {
            self.0.lock().unwrap().push(item);
            Ok(())
        }
    }

    struct ManualClock(Mutex<Instant>);
    impl ManualClock {
        fn advance(&self, by: Duration) {
            *self.0.lock().unwrap() += by;
        }
    }
    impl Clock for ManualClock {
        fn now(&self) -> Instant {
            *self.0.lock().unwrap()
        }
    }

    fn stream(config: StreamingConfig) -> (DeltaStream<Recorder>, Recorder, Arc<ManualClock>) {
        let recorder = Recorder::default();
        let clock = Arc::new(ManualClock(Mutex::new(Instant::now())));
        let stream = DeltaStream::new(
            ParagraphCoalescer::new(config).unwrap(),
            recorder.clone(),
            clock.clone(),
        );
        (stream, recorder, clock)
    }

    fn text(text: &str) -> RuntimeEvent {
        RuntimeEvent::Text { text: text.into() }
    }

    #[test]
    fn a_tool_event_flushes_buffered_text_first() {
        let (stream, recorder, _) = stream(StreamingConfig::default());
        stream.event(text("I will read ")).unwrap();
        stream.event(text("the file.")).unwrap();
        assert!(recorder.items().is_empty());
        let started = RuntimeEvent::ToolStarted {
            id: "1".into(),
            name: "read".into(),
        };
        stream.event(started.clone()).unwrap();
        stream.event(text("Done")).unwrap();
        assert_eq!(
            recorder.events(),
            vec![text("I will read the file."), started]
        );
        match &recorder.items()[0] {
            Outgoing::Chunk(chunk) => {
                assert_eq!((chunk.deltas, chunk.reason), (2, FlushReason::Barrier))
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_barrier_releases_text_before_approvals_and_usage() {
        let (stream, recorder, _) = stream(StreamingConfig::default());
        stream.event(text("may I write?")).unwrap();
        stream.barrier().unwrap();
        assert_eq!(recorder.events(), vec![text("may I write?")]);
        stream.barrier().unwrap();
        assert_eq!(recorder.items().len(), 1);
    }

    #[test]
    fn ticks_release_idle_and_overdue_text() {
        let (stream, recorder, clock) = stream(StreamingConfig::default());
        stream.event(text("partial")).unwrap();
        clock.advance(Duration::from_millis(1499));
        stream.tick().unwrap();
        assert!(recorder.items().is_empty());
        clock.advance(Duration::from_millis(1));
        stream.tick().unwrap();
        assert_eq!(recorder.events(), vec![text("partial")]);

        // An overdue buffer is released by the next event before it is buffered.
        stream.event(text("old")).unwrap();
        clock.advance(Duration::from_secs(5));
        stream.event(text("new")).unwrap();
        assert_eq!(recorder.events(), vec![text("partial"), text("old")]);
    }

    #[test]
    fn the_ticker_flushes_without_any_further_event() {
        let recorder = Recorder::default();
        let config = StreamingConfig {
            idle_flush_ms: 20,
            max_latency_ms: 40,
            ..StreamingConfig::default()
        };
        let mut stream = DeltaStream::new(
            ParagraphCoalescer::new(config).unwrap(),
            recorder.clone(),
            Arc::new(SystemClock),
        );
        stream.start_ticker(tracing::Span::none());
        stream.event(text("quiet provider")).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while recorder.items().is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(recorder.events(), vec![text("quiet provider")]);
        stream.finish().unwrap();
    }

    #[test]
    fn finishing_and_dropping_flush_everything() {
        let (mut stream, recorder, _) = stream(StreamingConfig::default());
        stream.event(text("tail")).unwrap();
        stream.finish().unwrap();
        stream.finish().unwrap();
        assert_eq!(recorder.events(), vec![text("tail")]);

        let (stream, recorder, _) = self::stream(StreamingConfig::default());
        stream.event(text("dropped mid-turn")).unwrap();
        drop(stream);
        assert_eq!(recorder.events(), vec![text("dropped mid-turn")]);
    }

    #[test]
    fn token_mode_forwards_each_delta_immediately() {
        let (mut stream, recorder, _) = stream(StreamingConfig::token());
        let events = vec![
            text("a"),
            RuntimeEvent::Reasoning { text: "".into() },
            text("b\n\n"),
        ];
        for event in &events {
            stream.event(event.clone()).unwrap();
        }
        stream.start_ticker(tracing::Span::none());
        assert!(stream.ticker.is_none());
        stream.finish().unwrap();
        assert_eq!(recorder.events(), events);
    }

    #[test]
    fn forwarded_bytes_and_delta_counts_are_exact() {
        let answer = "Erster Absatz: äöü.\n\nSecond one\r\n\r\n第三段。\n\ntail";
        let (mut stream, recorder, _) = stream(StreamingConfig::default());
        let mut sent = 0;
        let mut deltas = 0;
        let mut rest = answer;
        for size in [3, 5, 8, 4, 7, 6].into_iter().cycle() {
            if rest.is_empty() {
                break;
            }
            let mut cut = size.min(rest.len());
            while !rest.is_char_boundary(cut) {
                cut += 1;
            }
            stream.event(text(&rest[..cut])).unwrap();
            sent += cut;
            deltas += 1;
            rest = &rest[cut..];
            if deltas == 9 {
                stream
                    .event(RuntimeEvent::ToolCompleted {
                        id: "1".into(),
                        success: true,
                        details: None,
                    })
                    .unwrap();
            }
        }
        stream.finish().unwrap();
        let chunks: Vec<StreamChunk> = recorder
            .items()
            .into_iter()
            .filter_map(|item| match item {
                Outgoing::Chunk(chunk) => Some(chunk),
                Outgoing::Event(_) => None,
            })
            .collect();
        assert_eq!(chunks.iter().map(|c| c.text.len()).sum::<usize>(), sent);
        assert_eq!(chunks.iter().map(|c| c.deltas).sum::<usize>(), deltas);
        assert_eq!(
            chunks.iter().map(|c| c.text.as_str()).collect::<String>(),
            answer
        );
        assert!(
            chunks.len() < deltas / 2,
            "{} chunks for {deltas} deltas",
            chunks.len()
        );
    }

    #[test]
    fn a_sink_failure_is_sticky() {
        struct Failing;
        impl EventSink for Failing {
            fn append(&mut self, _: Outgoing) -> Result<(), String> {
                Err("runtime event byte budget exceeded".into())
            }
        }
        let mut stream = DeltaStream::new(
            ParagraphCoalescer::new(StreamingConfig::default()).unwrap(),
            Failing,
            Arc::new(SystemClock),
        );
        stream.event(text("one\n\n")).unwrap_err();
        assert!(stream.event(text("two")).is_err());
        assert!(stream.barrier().is_err());
        assert!(stream.finish().is_err());
    }
}
