//! TUI message sink: channels AkhMessages to the TUI event loop.

use std::sync::Mutex;

use crate::message::{AkhMessage, MessageSink};

/// Collects messages for TUI rendering (thread-safe).
pub struct TuiSink {
    pending: Mutex<Vec<AkhMessage>>,
}

impl TuiSink {
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(Vec::new()),
        }
    }

    /// Drain all pending messages for the TUI to render.
    pub fn drain(&self) -> Vec<AkhMessage> {
        let mut pending = self.pending.lock().unwrap();
        std::mem::take(&mut *pending)
    }
}

impl MessageSink for TuiSink {
    fn emit(&self, msg: &AkhMessage) {
        self.pending.lock().unwrap().push(msg.clone());
    }
}
