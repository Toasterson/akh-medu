//! Grammar-constrained JSON decoding for Candle tool calling (ADR 070).
//!
//! Small GGUF models emit *mostly*-valid tool-call JSON, but "mostly" is not
//! good enough to dispatch a tool reliably. Because the Candle backend owns the
//! token sampling loop (unlike an HTTP backend), we can *constrain* generation:
//! at each step we keep only the next-token candidates that keep the output a
//! valid prefix of a JSON value. The model still chooses *which* valid token
//! (semantics); the grammar guarantees the output *parses* (syntax). This is the
//! same mechanism llama.cpp/Ollama apply via GBNF, implemented natively over
//! Candle's logits — see `tool_call.gbnf` for the readable grammar.
//!
//! [`JsonConstraint`] is a minimal pushdown automaton over the JSON value
//! grammar. It is fed the decoded text of each accepted token and reports
//! whether a candidate continuation is still grammar-valid and whether a
//! complete value has been produced (so decoding can stop).

/// Where we are inside the JSON value grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ctx {
    /// Inside an object, expecting `"` (a key) or `}`.
    ObjectKeyOrEnd,
    /// Inside an object, read key+`:`, expecting a value.
    ObjectValue,
    /// Inside an object, read a value, expecting `,` or `}`.
    ObjectCommaOrEnd,
    /// Inside an array, expecting a value or `]`.
    ArrayValueOrEnd,
    /// Inside an array, read a value, expecting `,` or `]`.
    ArrayCommaOrEnd,
}

/// A pushdown automaton that validates a stream of characters against the JSON
/// value grammar, one accepted chunk at a time.
///
/// Construct with [`JsonConstraint::object`] (tool-call arguments are always a
/// JSON object). Use [`JsonConstraint::accepts`] to test a candidate token's
/// text before committing it, and [`JsonConstraint::push`] to advance the state
/// with the chosen token. [`JsonConstraint::is_complete`] reports when a full
/// top-level value has closed.
#[derive(Debug, Clone)]
pub struct JsonConstraint {
    stack: Vec<Ctx>,
    /// Inside a string literal (key or value).
    in_string: bool,
    /// The current string literal is an object key (a `:` must follow it).
    string_is_key: bool,
    /// The previous char was a backslash inside a string.
    escape: bool,
    /// After an object key string closed, the next non-ws char must be `:`.
    awaiting_colon: bool,
    /// Currently reading a number / `true` / `false` / `null` bareword.
    in_bareword: bool,
    /// We have consumed the opening `{`.
    started: bool,
    /// A complete top-level value has been read.
    done: bool,
}

impl JsonConstraint {
    /// A constraint that requires the output to be a single JSON object — the
    /// shape of tool-call arguments.
    pub fn object() -> Self {
        Self {
            stack: Vec::new(),
            in_string: false,
            string_is_key: false,
            escape: false,
            awaiting_colon: false,
            in_bareword: false,
            started: false,
            done: false,
        }
    }

    /// Whether a complete top-level JSON value has been read (decoding may stop).
    pub fn is_complete(&self) -> bool {
        self.done
    }

    /// Would feeding `chunk` keep the output a valid JSON prefix? Does not mutate.
    pub fn accepts(&self, chunk: &str) -> bool {
        let mut probe = self.clone();
        probe.feed(chunk).is_ok()
    }

    /// Advance the automaton by `chunk` (the decoded text of an accepted token).
    /// Returns `Err(())` if the chunk is not a valid continuation — callers
    /// should only `push` chunks they first checked with [`accepts`].
    pub fn push(&mut self, chunk: &str) -> Result<(), ()> {
        self.feed(chunk)
    }

    fn feed(&mut self, chunk: &str) -> Result<(), ()> {
        for ch in chunk.chars() {
            self.step(ch)?;
        }
        Ok(())
    }

    fn step(&mut self, ch: char) -> Result<(), ()> {
        // Once a complete value has closed, only trailing whitespace is allowed.
        if self.done {
            return if ch.is_whitespace() { Ok(()) } else { Err(()) };
        }

        // ── Inside a string literal ───────────────────────────────────────
        if self.in_string {
            if self.escape {
                self.escape = false;
                return Ok(());
            }
            match ch {
                '\\' => self.escape = true,
                '"' => {
                    self.in_string = false;
                    if self.string_is_key {
                        self.string_is_key = false;
                        self.awaiting_colon = true;
                    } else {
                        self.close_value();
                    }
                }
                _ => {}
            }
            return Ok(());
        }

        // ── Inside a bareword (number / true / false / null) ──────────────
        if self.in_bareword {
            if is_bareword_char(ch) {
                return Ok(());
            }
            self.in_bareword = false;
            self.close_value();
            // fall through: reprocess `ch` in the post-value state
        }

        // ── Awaiting the `:` after an object key ──────────────────────────
        if self.awaiting_colon {
            if ch.is_whitespace() {
                return Ok(());
            }
            if ch == ':' {
                self.awaiting_colon = false;
                *self.stack.last_mut().ok_or(())? = Ctx::ObjectValue;
                return Ok(());
            }
            return Err(());
        }

        if ch.is_whitespace() {
            return Ok(());
        }

        match self.stack.last().copied() {
            // Top level: the only valid opener is `{` (tool args are an object).
            None => {
                if self.started {
                    return Err(());
                }
                if ch == '{' {
                    self.started = true;
                    self.stack.push(Ctx::ObjectKeyOrEnd);
                    Ok(())
                } else {
                    Err(())
                }
            }
            Some(Ctx::ObjectKeyOrEnd) => match ch {
                '"' => {
                    self.in_string = true;
                    self.string_is_key = true;
                    Ok(())
                }
                '}' => {
                    self.pop_container();
                    Ok(())
                }
                _ => Err(()),
            },
            Some(Ctx::ObjectValue) => self.open_value(ch),
            Some(Ctx::ObjectCommaOrEnd) => match ch {
                ',' => {
                    *self.stack.last_mut().ok_or(())? = Ctx::ObjectKeyOrEnd;
                    Ok(())
                }
                '}' => {
                    self.pop_container();
                    Ok(())
                }
                _ => Err(()),
            },
            Some(Ctx::ArrayValueOrEnd) => {
                if ch == ']' {
                    self.pop_container();
                    Ok(())
                } else {
                    self.open_value(ch)
                }
            }
            Some(Ctx::ArrayCommaOrEnd) => match ch {
                ',' => {
                    *self.stack.last_mut().ok_or(())? = Ctx::ArrayValueOrEnd;
                    Ok(())
                }
                ']' => {
                    self.pop_container();
                    Ok(())
                }
                _ => Err(()),
            },
        }
    }

    /// Begin a value in a value-expecting position.
    fn open_value(&mut self, ch: char) -> Result<(), ()> {
        match ch {
            '"' => {
                self.in_string = true;
                Ok(())
            }
            '{' => {
                self.stack.push(Ctx::ObjectKeyOrEnd);
                Ok(())
            }
            '[' => {
                self.stack.push(Ctx::ArrayValueOrEnd);
                Ok(())
            }
            c if c == '-' || c.is_ascii_digit() || c.is_ascii_alphabetic() => {
                self.in_bareword = true;
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// A non-string value (or container) just completed; advance the parent.
    fn close_value(&mut self) {
        match self.stack.last_mut() {
            Some(slot) if *slot == Ctx::ObjectValue => *slot = Ctx::ObjectCommaOrEnd,
            Some(slot) if *slot == Ctx::ArrayValueOrEnd => *slot = Ctx::ArrayCommaOrEnd,
            _ => {}
        }
    }

    fn pop_container(&mut self) {
        self.stack.pop();
        if self.stack.is_empty() {
            self.done = true;
        } else {
            self.close_value();
        }
    }
}

/// Characters that may appear in a JSON number / `true` / `false` / `null`.
fn is_bareword_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '-' | '+' | '.' | 'e' | 'E')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a full string through the constraint, returning the final state.
    fn run(s: &str) -> Result<JsonConstraint, ()> {
        let mut c = JsonConstraint::object();
        c.push(s)?;
        Ok(c)
    }

    #[test]
    fn accepts_simple_object() {
        let c = run(r#"{"location": "Paris"}"#).unwrap();
        assert!(c.is_complete());
    }

    #[test]
    fn accepts_nested_and_arrays_and_numbers() {
        let c = run(r#"{"a": [1, 2, true], "b": {"c": -3.5e2}, "d": null}"#).unwrap();
        assert!(c.is_complete());
    }

    #[test]
    fn incomplete_object_is_valid_prefix_but_not_complete() {
        let c = run(r#"{"location": "Par"#).unwrap();
        assert!(!c.is_complete());
    }

    #[test]
    fn rejects_non_object_top_level() {
        assert!(run("[1,2,3]").is_err());
        assert!(run("\"hi\"").is_err());
        assert!(run("42").is_err());
    }

    #[test]
    fn rejects_missing_colon_and_trailing_garbage() {
        assert!(run(r#"{"a" 1}"#).is_err());
        assert!(run(r#"{"a": 1} x"#).is_err());
    }

    #[test]
    fn accepts_checks_candidate_without_mutating() {
        let mut c = JsonConstraint::object();
        c.push("{").unwrap();
        // After `{`, a quote (key) is valid; a digit is not.
        assert!(c.accepts("\""));
        assert!(!c.accepts("1"));
        // `accepts` must not have advanced the state.
        assert!(c.accepts("}")); // empty object is valid
    }

    #[test]
    fn escaped_quotes_inside_strings() {
        let c = run(r#"{"q": "she said \"hi\""}"#).unwrap();
        assert!(c.is_complete());
    }
}
