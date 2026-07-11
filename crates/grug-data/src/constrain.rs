//! Schema-constrained decoding grammar (Phase 51 P1a-4 Step A).
//!
//! A byte-level acceptor over the *exact* JSON tool-call surface this project
//! trains on — the string produced by
//! [`crate::schema::Example::expected_call_as_json_string`]. Because
//! `serde_json`'s default `Map` is a `BTreeMap` (no `preserve_order` feature
//! in this workspace), object keys serialize in **sorted** order, so the
//! ground-truth target is:
//!
//! ```text
//! {"arguments":{"<p1>":<v1>,"<p2>":<v2>,...},"name":"<tool>"}
//! ```
//!
//! with `arguments` before `name` (`"a" < "n"`) and the parameter keys in
//! ascending order. The grammar mirrors that layout: a fixed `{"arguments":{`
//! prefix, a comma-separated body of `"<param>":<value>` entries, a `}` that
//! closes the arguments object, the fixed `,"name":"` middle, the tool name,
//! and a closing `"}`.
//!
//! # What it constrains (and deliberately does not)
//!
//! Step A's job is to isolate *"invalid parameter names"* from *"wrong
//! values"* (see the P1a-4 plan). So the grammar constrains **structure**,
//! **parameter keys**, and the **tool name** to the *offered* tools — a call
//! is accepted iff there exists an offered tool whose schema contains every
//! emitted parameter key *and* whose name is the emitted name. Parameter
//! **values** are left free (any well-formed JSON string, or any non-delimiter
//! "bare" token for numbers/booleans): forcing valid *names* is the treatment;
//! copying correct *values* is what Step A shows constrained decoding cannot
//! fix (that needs the copy/pointer mechanism, Step B).
//!
//! Because the tool name comes *last*, the acceptor cannot localise the schema
//! from an already-emitted name. Instead it carries a shrinking set of
//! **viable tools** (those consistent with every parameter key emitted so
//! far); completing a key `p` intersects that set with `{tools having p}`, and
//! the final name must lie in the surviving set. This makes the constraint
//! "consistent with some single offered tool", not the weaker "each key is a
//! real key of *some* tool".
//!
//! # Determinism
//!
//! The only sources of branching — which tool name is being typed, which
//! parameter key is being typed — are folded into candidate lists carried
//! *inside* a single [`GrammarState`], so every state has at most one
//! successor per input byte. The acceptor is therefore a DFA (with list-valued
//! state), not an NFA with a multi-state frontier: [`ToolCallGrammar::step`]
//! returns `Option<GrammarState>`, and driving it needs no set bookkeeping.

use crate::schema::ToolSchema;

/// The fixed prefix every accepted call starts with: `{"arguments":{`.
const PRE: &[u8] = b"{\"arguments\":{";
/// The fixed middle between the closed arguments object and the tool name:
/// `,"name":"`.
const MID: &[u8] = b",\"name\":\"";

/// The set of offered tools still consistent with every parameter key emitted
/// so far, as indices into [`ToolCallGrammar::tools`]. Small (bounded by the
/// number of tools offered in one prompt, typically 3–7), sorted, and deduped.
type Viable = Vec<u16>;

/// A position in the constrained-decoding acceptor. Cheap to clone (the
/// candidate/viable lists it carries are bounded by the offered-tool count).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrammarState {
    /// Matching the fixed [`PRE`] literal at byte offset `usize`.
    Pre(usize),
    /// After `{` (or a closed value + `,`... no — see [`GrammarState::AfterComma`]):
    /// at the start of the arguments body, where either a key (`"`) may open
    /// or the object may close (`}`), carrying the currently-viable tools.
    ArgsStart(Viable),
    /// Immediately after a `,` inside the arguments body: a key **must** open
    /// (`"`), the object may not close here.
    AfterComma(Viable),
    /// Typing a parameter key: `cands` are the `(tool, param)` pairs whose
    /// parameter name still matches the bytes typed so far, `pos` is how many
    /// name bytes have been consumed.
    Key { cands: Vec<(u16, u16)>, pos: usize },
    /// Consumed a key's closing `"`; expecting `:`. Carries the viable tools
    /// narrowed to those having the just-completed key.
    Colon(Viable),
    /// Expecting the first byte of a value (`"` opens a string; any
    /// non-delimiter byte starts a bare number/boolean token).
    ValueStart(Viable),
    /// Inside a string value; `escaped` tracks a pending `\` escape.
    Str { viable: Viable, escaped: bool },
    /// Inside a bare (number/boolean/null) value.
    Bare(Viable),
    /// After a value: expecting `,` (another entry) or `}` (close arguments).
    AfterValue(Viable),
    /// Matching the fixed [`MID`] literal at byte offset `pos`, carrying the
    /// tools consistent with all emitted keys (the name must be one of them).
    Mid { viable: Viable, pos: usize },
    /// Typing the tool name: `cands` are the viable tools whose name still
    /// matches the bytes typed so far.
    Name { cands: Vec<u16>, pos: usize },
    /// Consumed the name's closing `"`; expecting the final `}`.
    CloseOuter,
    /// A complete, accepted call. No byte is accepted here — only `EOS` (see
    /// [`ToolCallGrammar::accepts_eos`]).
    End,
}

/// A byte-level acceptor for schema-constrained tool-call decoding, built from
/// the tools offered in one prompt.
#[derive(Debug, Clone)]
pub struct ToolCallGrammar {
    /// Tool names as raw bytes, indexed by tool.
    tool_names: Vec<Vec<u8>>,
    /// Parameter-name bytes per tool: `params[tool][param]`.
    params: Vec<Vec<Vec<u8>>>,
}

impl ToolCallGrammar {
    /// Build the grammar for the tools offered in one example
    /// ([`crate::schema::Example::tools`]).
    pub fn new(tools: &[ToolSchema]) -> Self {
        let tool_names = tools.iter().map(|t| t.name.clone().into_bytes()).collect();
        let params = tools
            .iter()
            .map(|t| t.params.iter().map(|p| p.name.clone().into_bytes()).collect())
            .collect();
        Self { tool_names, params }
    }

    /// The start state: matching [`PRE`] from offset 0.
    pub fn start(&self) -> GrammarState {
        GrammarState::Pre(0)
    }

    /// All tools viable at the start of an arguments body (every offered tool).
    fn all_viable(&self) -> Viable {
        (0..self.tool_names.len() as u16).collect()
    }

    /// The `(tool, param)` candidate list opening a fresh key under `viable`:
    /// every parameter of every still-viable tool.
    fn cands_for(&self, viable: &[u16]) -> Vec<(u16, u16)> {
        let mut out = Vec::new();
        for &t in viable {
            for p in 0..self.params[t as usize].len() as u16 {
                out.push((t, p));
            }
        }
        out
    }

    /// Advance the acceptor by one input byte, returning the successor state or
    /// `None` if `byte` is not a legal continuation of any accepted call.
    pub fn step(&self, state: &GrammarState, byte: u8) -> Option<GrammarState> {
        use GrammarState::*;
        match state {
            Pre(i) => {
                if byte == PRE[*i] {
                    if *i + 1 < PRE.len() {
                        Some(Pre(*i + 1))
                    } else {
                        Some(ArgsStart(self.all_viable()))
                    }
                } else {
                    None
                }
            }
            ArgsStart(v) => match byte {
                b'"' => Some(Key {
                    cands: self.cands_for(v),
                    pos: 0,
                }),
                b'}' => Some(Mid {
                    viable: v.clone(),
                    pos: 0,
                }),
                _ => None,
            },
            AfterComma(v) => {
                if byte == b'"' {
                    Some(Key {
                        cands: self.cands_for(v),
                        pos: 0,
                    })
                } else {
                    None
                }
            }
            Key { cands, pos } => {
                if byte == b'"' {
                    // Closing quote: keep only candidates whose name is fully
                    // typed, and narrow viable tools to their tools.
                    let mut tools: Vec<u16> = cands
                        .iter()
                        .filter(|(t, p)| self.params[*t as usize][*p as usize].len() == *pos)
                        .map(|(t, _)| *t)
                        .collect();
                    tools.sort_unstable();
                    tools.dedup();
                    if tools.is_empty() {
                        None
                    } else {
                        Some(Colon(tools))
                    }
                } else {
                    let nc: Vec<(u16, u16)> = cands
                        .iter()
                        .copied()
                        .filter(|(t, p)| self.params[*t as usize][*p as usize].get(*pos) == Some(&byte))
                        .collect();
                    if nc.is_empty() {
                        None
                    } else {
                        Some(Key { cands: nc, pos: *pos + 1 })
                    }
                }
            }
            Colon(v) => {
                if byte == b':' {
                    Some(ValueStart(v.clone()))
                } else {
                    None
                }
            }
            ValueStart(v) => match byte {
                b'"' => Some(Str {
                    viable: v.clone(),
                    escaped: false,
                }),
                // A value must be non-empty, so it cannot start with a
                // delimiter. Backslash outside a string is not valid JSON.
                b',' | b'}' | b'\\' => None,
                _ => Some(Bare(v.clone())),
            },
            Str { viable, escaped } => {
                if *escaped {
                    Some(Str {
                        viable: viable.clone(),
                        escaped: false,
                    })
                } else {
                    match byte {
                        b'\\' => Some(Str {
                            viable: viable.clone(),
                            escaped: true,
                        }),
                        b'"' => Some(AfterValue(viable.clone())),
                        _ => Some(Str {
                            viable: viable.clone(),
                            escaped: false,
                        }),
                    }
                }
            }
            Bare(v) => match byte {
                b',' => Some(AfterComma(v.clone())),
                b'}' => Some(Mid {
                    viable: v.clone(),
                    pos: 0,
                }),
                // A bare (number/boolean/null) token never contains a quote or
                // backslash; rejecting them keeps the value well-formed.
                b'"' | b'\\' => None,
                _ => Some(Bare(v.clone())),
            },
            AfterValue(v) => match byte {
                b',' => Some(AfterComma(v.clone())),
                b'}' => Some(Mid {
                    viable: v.clone(),
                    pos: 0,
                }),
                _ => None,
            },
            Mid { viable, pos } => {
                if byte == MID[*pos] {
                    if *pos + 1 < MID.len() {
                        Some(Mid {
                            viable: viable.clone(),
                            pos: *pos + 1,
                        })
                    } else {
                        // Consumed the final `"` of `,"name":"`; ready to type
                        // the name, constrained to viable tools.
                        Some(Name {
                            cands: viable.clone(),
                            pos: 0,
                        })
                    }
                } else {
                    None
                }
            }
            Name { cands, pos } => {
                if byte == b'"' {
                    let complete = cands
                        .iter()
                        .any(|&t| self.tool_names[t as usize].len() == *pos);
                    if complete {
                        Some(CloseOuter)
                    } else {
                        None
                    }
                } else {
                    let nc: Vec<u16> = cands
                        .iter()
                        .copied()
                        .filter(|&t| self.tool_names[t as usize].get(*pos) == Some(&byte))
                        .collect();
                    if nc.is_empty() {
                        None
                    } else {
                        Some(Name { cands: nc, pos: *pos + 1 })
                    }
                }
            }
            CloseOuter => {
                if byte == b'}' {
                    Some(End)
                } else {
                    None
                }
            }
            End => None,
        }
    }

    /// Fold `bytes` through [`ToolCallGrammar::step`] starting from `state`,
    /// returning the final state or `None` if any byte is rejected. Used to
    /// test a whole candidate token (whose surface is a byte sequence) in one
    /// call: `Some` means the token keeps the running output on a path to an
    /// accepted call.
    pub fn advance(&self, state: &GrammarState, bytes: &[u8]) -> Option<GrammarState> {
        let mut cur = state.clone();
        for &b in bytes {
            cur = self.step(&cur, b)?;
        }
        Some(cur)
    }

    /// Whether `EOS` (end of generation) is legal in `state`: only once a
    /// complete call has been formed.
    pub fn accepts_eos(&self, state: &GrammarState) -> bool {
        matches!(state, GrammarState::End)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{ParamSpec, ParamType, ToolCall, ToolSchema};
    use crate::Dataset;

    fn tool(name: &str, params: &[(&str, ParamType)]) -> ToolSchema {
        ToolSchema::new(
            name,
            format!("{name} description"),
            params
                .iter()
                .map(|(n, t)| ParamSpec::new(*n, t.clone(), true))
                .collect(),
        )
    }

    /// Drive the grammar over a whole byte string; return the end state.
    fn run(grammar: &ToolCallGrammar, s: &str) -> Option<GrammarState> {
        grammar.advance(&grammar.start(), s.as_bytes())
    }

    #[test]
    fn accepts_a_hand_written_call() {
        let tools = vec![
            tool("book_table", &[("party_size", ParamType::Integer), ("time", ParamType::String)]),
            tool("get_weather", &[("city", ParamType::String)]),
        ];
        let g = ToolCallGrammar::new(&tools);
        // Keys sorted (party_size < time), arguments before name.
        let s = r#"{"arguments":{"party_size":4,"time":"20:00"},"name":"book_table"}"#;
        let end = run(&g, s).expect("valid call must be accepted");
        assert_eq!(end, GrammarState::End);
        assert!(g.accepts_eos(&end));
    }

    #[test]
    fn accepts_empty_arguments() {
        let tools = vec![tool("ping", &[])];
        let g = ToolCallGrammar::new(&tools);
        let s = r#"{"arguments":{},"name":"ping"}"#;
        let end = run(&g, s).expect("empty-args call must be accepted");
        assert_eq!(end, GrammarState::End);
    }

    #[test]
    fn accepts_string_value_containing_delimiters() {
        let tools = vec![tool("note", &[("text", ParamType::String)])];
        let g = ToolCallGrammar::new(&tools);
        // The string value contains `,` and `}` which must NOT end the value.
        let s = r#"{"arguments":{"text":"a,b}c"},"name":"note"}"#;
        assert_eq!(run(&g, s), Some(GrammarState::End));
    }

    #[test]
    fn rejects_unknown_param_name() {
        let tools = vec![tool("get_weather", &[("city", ParamType::String)])];
        let g = ToolCallGrammar::new(&tools);
        // "region" is not a param of get_weather (nor any offered tool).
        let s = r#"{"arguments":{"region":"x"},"name":"get_weather"}"#;
        assert_eq!(run(&g, s), None);
    }

    #[test]
    fn rejects_name_inconsistent_with_emitted_params() {
        // Two tools with disjoint params. Emitting tool A's param then naming
        // tool B must be rejected — the viable set is narrowed to A by the key.
        let tools = vec![
            tool("alpha", &[("aaa", ParamType::String)]),
            tool("beta", &[("bbb", ParamType::String)]),
        ];
        let g = ToolCallGrammar::new(&tools);
        let s = r#"{"arguments":{"aaa":"x"},"name":"beta"}"#;
        assert_eq!(run(&g, s), None);
        // ...but naming the consistent tool is accepted.
        let ok = r#"{"arguments":{"aaa":"x"},"name":"alpha"}"#;
        assert_eq!(run(&g, ok), Some(GrammarState::End));
    }

    #[test]
    fn rejects_stray_whitespace_at_structural_position() {
        let tools = vec![tool("get_weather", &[("city", ParamType::String)])];
        let g = ToolCallGrammar::new(&tools);
        // A space after the first `{` is not part of the compact target.
        let s = r#"{ "arguments":{"city":"x"},"name":"get_weather"}"#;
        assert_eq!(run(&g, s), None);
    }

    #[test]
    fn eos_only_accepted_at_end() {
        let tools = vec![tool("ping", &[])];
        let g = ToolCallGrammar::new(&tools);
        assert!(!g.accepts_eos(&g.start()));
        let mid = g.advance(&g.start(), br#"{"arguments":{"#).unwrap();
        assert!(!g.accepts_eos(&mid));
    }

    /// The golden test: for real generator output, the grammar must accept the
    /// *exact* string the training target uses. If the JSON key ordering
    /// assumption (arguments-before-name, sorted params) were wrong, this
    /// fails — which is the whole point of pinning it against real data.
    #[test]
    fn accepts_real_generator_targets_across_many_examples() {
        // Procedural catalog so we exercise many tool/param shapes, including
        // multi-param calls whose sorted key order matters.
        let dataset = Dataset::new_sized(42, 200, 0.15).unwrap();
        let examples = dataset.train_examples(300).unwrap();
        let mut checked = 0;
        for ex in &examples {
            let g = ToolCallGrammar::new(&ex.tools);
            let target = ex.expected_call_as_json_string();
            let end = g.advance(&g.start(), target.as_bytes());
            assert_eq!(
                end,
                Some(GrammarState::End),
                "grammar rejected a real training target:\n  {target}"
            );
            checked += 1;
        }
        assert!(checked > 100, "expected to check many examples, got {checked}");
    }

    /// A call naming a tool that was not offered must be rejected even if its
    /// structure is otherwise fine (the name isn't in any viable set).
    #[test]
    fn rejects_unoffered_tool_name() {
        let tools = vec![tool("get_weather", &[("city", ParamType::String)])];
        let g = ToolCallGrammar::new(&tools);
        let s = r#"{"arguments":{"city":"x"},"name":"send_email"}"#;
        assert_eq!(run(&g, s), None);
    }

    /// Sanity: a `ToolCall` built the same way the generator builds them
    /// round-trips through `expected_call_as_json_string` into an accepted
    /// string (guards against the scorer and grammar disagreeing on shape).
    #[test]
    fn grammar_agrees_with_expected_call_json_shape() {
        use crate::schema::Example;
        let tools = vec![tool("x", &[("b", ParamType::Integer), ("a", ParamType::String)])];
        let call = ToolCall::new(
            "x",
            serde_json::json!({"a": "hi", "b": 3}),
        );
        let ex = Example {
            tools: tools.clone(),
            user_request: "irrelevant".to_string(),
            expected_call: call,
            mock_response: String::new(),
        };
        let g = ToolCallGrammar::new(&tools);
        let s = ex.expected_call_as_json_string();
        assert_eq!(g.advance(&g.start(), s.as_bytes()), Some(GrammarState::End), "target was: {s}");
    }
}
