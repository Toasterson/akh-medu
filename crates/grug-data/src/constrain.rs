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

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use miette::Diagnostic;
use serde_json::Value;
use thiserror::Error;

use crate::schema::ToolSchema;

/// The fixed prefix every accepted call starts with: `{"arguments":{`.
const PRE: &[u8] = b"{\"arguments\":{";
/// The fixed middle between the closed arguments object and the tool name:
/// `,"name":"`.
const MID: &[u8] = b",\"name\":\"";
/// The literal envelope Akh-Medu's runtime wraps a tool call in
/// ([`crate::knowledge_workflow::KnowledgeAction::render_runtime`]).
const RUNTIME_TAG_OPEN: &[u8] = b"<tool_call>\n";
const RUNTIME_TAG_CLOSE: &[u8] = b"\n</tool_call>";

#[derive(Debug, Error, Diagnostic)]
pub enum ToolCallGrammarError {
    #[error("tool definition at index {index} has an incompatible shape: {message}")]
    #[diagnostic(
        code(grug_data::constrain::invalid_shape),
        help(
            "tool definitions must be OpenAI-style {{\"type\":\"function\",\"function\":{{...}}}} objects"
        )
    )]
    InvalidShape { index: usize, message: String },
}

/// The set of offered tools still consistent with every parameter key emitted
/// so far, as indices into [`ToolCallGrammar::tools`]. Small (bounded by the
/// number of tools offered in one prompt, typically 3–7), sorted, and deduped.
type Viable = Vec<u16>;

/// A position in the constrained-decoding acceptor. Cheap to clone (the
/// candidate/viable lists it carries are bounded by the offered-tool count,
/// and [`ValueParser`]'s nesting is bounded by [`MAX_VALUE_DEPTH`]/
/// [`MAX_ARRAY_ITEMS`] and shares its schema fragments via `Rc`).
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
    /// narrowed to those having the just-completed key, and that key's exact
    /// name bytes (needed once `:` is consumed, to resolve the parameter's
    /// declared schema — see [`ToolCallGrammar::resolve_structural_schema`]).
    Colon { viable: Viable, param: Vec<u8> },
    /// Typing a JSON value governed by the just-completed key's declared
    /// schema (Phase 51 / ADR 104 D1 nested grammar): a recursive descent
    /// over `type: array`/`object`/scalar, bounded by [`MAX_VALUE_DEPTH`] and
    /// [`MAX_ARRAY_ITEMS`]. `viable` is carried through unchanged so
    /// completing the whole value (however deeply nested) lands back on
    /// [`GrammarState::AfterValue`] with the same bookkeeping as a scalar
    /// value always had.
    Value { viable: Viable, parser: ValueParser },
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

// ── Recursive JSON-value grammar (Phase 51 / ADR 104 D1 nested support) ────
//
// The structural acceptor above forces valid keys/tool names but originally
// left *every* value byte-unconstrained past its opening delimiter (see the
// module docs). Measurement on the 4096-context checkpoint showed that gap is
// unsafe once a value's own JSON-Schema type is `array` or `object`: nothing
// stopped the decoder from emitting a stray `,`, digit, or even a nested
// `<tool_call>` tag right after `[`/`{`, producing unparseable garbage. This
// section adds a small recursive-descent grammar driven by the parameter's
// declared JSON-Schema fragment, so a value's own shape (scalar, array of X,
// object with required properties Y) is enforced the same way parameter names
// already were.
//
// A schema fragment missing `items` (for `type: array`) or `properties` (for
// `type: object`) — or with no recognized `type` at all — falls back to
// [`SchemaKind::Any`]: a generic, still well-formed JSON value grammar (any
// scalar, or arbitrarily nested arrays/objects with free-form keys), rather
// than degrading to fully unconstrained bytes. This is what `frontmatter`-
// and `metadata`-shaped parameters (`{"type":"object","additionalProperties":
// true}`, no `properties`) get: free-form nested JSON, but never garbage.

/// Depth bound: how many `[`/`{` may be open at once while typing one
/// parameter's value. Guards against a schema (or the `Any` fallback)
/// recursing forever; 8 comfortably covers every real tool schema in this
/// project (citations are array→object→string, depth 2).
const MAX_VALUE_DEPTH: usize = 8;

/// Length bound: how many elements one array value may contain before the
/// grammar refuses `,` and forces `]`. Guards against a runaway array from an
/// undertrained model looping forever inside the mask.
const MAX_ARRAY_ITEMS: usize = 64;

/// A JSON-Schema fragment, shared cheaply (`Rc::clone` is O(1) regardless of
/// schema size) across the many [`ValueParser`] clones one decode step's
/// per-vocabulary-token mask sweep produces.
type SchemaRef = Rc<Value>;

/// One property of a schema-constrained JSON object, materialized once when
/// the object's `{` is opened rather than re-derived from the raw schema on
/// every key byte.
#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectProp {
    name: Vec<u8>,
    schema: SchemaRef,
    required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ObjectSchema {
    props: Vec<ObjectProp>,
}

/// Does every property `obj` declares `required` appear in `seen`?
fn required_satisfied(obj: &ObjectSchema, seen: &[u16]) -> bool {
    obj.props
        .iter()
        .enumerate()
        .filter(|(_, prop)| prop.required)
        .all(|(index, _)| seen.contains(&(index as u16)))
}

/// What kind of structural value a schema fragment describes, resolved once
/// per `{`/`[`/scalar entry point rather than re-inspecting the raw [`Value`]
/// on every byte.
enum SchemaKind {
    String,
    Number,
    Boolean,
    Null,
    Array { items: SchemaRef },
    Object(Rc<ObjectSchema>),
    /// No usable `items`/`properties` (or no recognized `type` at all): see
    /// the module-level fallback note above.
    Any,
}

/// Classify a JSON-Schema fragment for the structural grammar. A
/// [`crate::schema::ParamType::Enum`]'s synthesized `{"enum": [...]}`
/// fragment (no `"type"`) is treated as a string, matching how
/// [`derive_value_constraint`] already treats `enum` as a string-valued
/// constraint.
fn classify(schema: &Value) -> SchemaKind {
    if schema.get("enum").and_then(Value::as_array).is_some() {
        return SchemaKind::String;
    }
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => SchemaKind::String,
        Some("integer") | Some("number") => SchemaKind::Number,
        Some("boolean") => SchemaKind::Boolean,
        Some("null") => SchemaKind::Null,
        Some("array") => schema
            .get("items")
            .map(|items| SchemaKind::Array {
                items: Rc::new(items.clone()),
            })
            .unwrap_or(SchemaKind::Any),
        Some("object") => schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| {
                let required: std::collections::BTreeSet<&str> = schema
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|values| values.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                let props = properties
                    .iter()
                    .map(|(name, sub_schema)| ObjectProp {
                        name: name.clone().into_bytes(),
                        schema: Rc::new(sub_schema.clone()),
                        required: required.contains(name.as_str()),
                    })
                    .collect();
                SchemaKind::Object(Rc::new(ObjectSchema { props }))
            })
            .unwrap_or(SchemaKind::Any),
        _ => SchemaKind::Any,
    }
}

/// An enclosing container the value parser is currently inside: what must
/// eventually close it, and the bookkeeping needed to know when it may.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Frame {
    /// A schema-typed array (`items` known); `count` is how many elements
    /// have been completed so far (bounded by [`MAX_ARRAY_ITEMS`]).
    Array { items: SchemaRef, count: usize },
    /// An `Any`-typed array (no `items` schema): elements may be anything.
    AnyArray { count: usize },
    /// A schema-typed object; `seen` are the property indices already typed
    /// (into `obj.props`), used to gate `}` on required-property coverage
    /// and to reject re-typing an already-seen key.
    Object { obj: Rc<ObjectSchema>, seen: Vec<u16> },
    /// An `Any`-typed object (no `properties` schema): free-form keys/values.
    AnyObject,
}

/// Where the recursive value parser is, relative to the innermost open
/// [`Frame`] (or, with an empty stack, relative to nothing — the top of the
/// whole parameter value).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Leaf {
    /// Expecting the first byte of a value governed by `schema`.
    Start(SchemaRef),
    /// Inside a string value; `escaped` tracks a pending `\` escape.
    Str { schema: SchemaRef, escaped: bool },
    /// Inside a bare (number/boolean/null) token.
    Bare,
    /// At the very start of the innermost frame's body (just after `{`/`[`):
    /// a key/item may open, or (only here, not after a `,`) the frame may
    /// close if legal.
    FrameStart,
    /// Immediately after a `,` inside the innermost frame: a key/item
    /// **must** open; the frame may not close here.
    FrameComma,
    /// After a complete element/property value inside the innermost frame:
    /// expecting `,` (another element/property) or the frame's closer.
    FrameAfterValue,
    /// Typing a schema-declared object key: `cands` are the still-matching
    /// property indices, `pos` is how many name bytes have been consumed.
    ObjectKey { cands: Vec<u16>, pos: usize },
    /// Consumed a schema-declared key's closing `"`; expecting `:`.
    ObjectColon { prop: u16 },
    /// Typing a free-form (`Any`-object) key; `escaped` tracks a pending `\`.
    AnyKey { escaped: bool },
    /// Consumed a free-form key's closing `"`; expecting `:`.
    AnyColon,
}

/// A recursive-descent position while typing one parameter's JSON value.
/// `stack` holds every currently-open enclosing array/object; `leaf` is the
/// position relative to the innermost one (or, with an empty stack, the
/// position at the top of the whole value).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueParser {
    stack: Vec<Frame>,
    leaf: Leaf,
}

/// What legally follows the first byte of a fresh value under `schema`.
enum StartOutcome {
    /// The byte fully determines a leaf (string open, or a bare token's
    /// first byte); no new frame is pushed.
    Leaf(Leaf),
    /// The byte opened a container (`[`/`{`); push `Frame` and enter its
    /// body at [`Leaf::FrameStart`].
    PushFrame(Frame),
}

/// Route the first byte of a value governed by `schema`. `depth` is the
/// number of frames already open (`stack.len()` before any push this call
/// might make), checked against [`MAX_VALUE_DEPTH`].
fn start_value(schema: &SchemaRef, byte: u8, depth: usize) -> Option<StartOutcome> {
    match classify(schema) {
        SchemaKind::String => match byte {
            b'"' => Some(StartOutcome::Leaf(Leaf::Str {
                schema: schema.clone(),
                escaped: false,
            })),
            _ => None,
        },
        // A bare token's *first* byte is restricted to a plausible JSON
        // literal opener (digit/`-` for a number, `t`/`f`/`n` for
        // true/false/null) so a declared `number`/`boolean`/`null` type
        // cannot open with arbitrary garbage (e.g. a stray `<`). The
        // continuation byte (`Leaf::Bare`'s own `step`) stays as permissive
        // as this project's scalar values always were -- see the module
        // docs' "what it constrains (and deliberately does not)" note.
        SchemaKind::Number => match byte {
            b'-' | b'0'..=b'9' => Some(StartOutcome::Leaf(Leaf::Bare)),
            _ => None,
        },
        SchemaKind::Boolean => match byte {
            b't' | b'f' => Some(StartOutcome::Leaf(Leaf::Bare)),
            _ => None,
        },
        SchemaKind::Null => match byte {
            b'n' => Some(StartOutcome::Leaf(Leaf::Bare)),
            _ => None,
        },
        SchemaKind::Array { items } => match byte {
            b'[' if depth < MAX_VALUE_DEPTH => {
                Some(StartOutcome::PushFrame(Frame::Array { items, count: 0 }))
            }
            _ => None,
        },
        SchemaKind::Object(obj) => match byte {
            b'{' if depth < MAX_VALUE_DEPTH => Some(StartOutcome::PushFrame(Frame::Object {
                obj,
                seen: Vec::new(),
            })),
            _ => None,
        },
        SchemaKind::Any => match byte {
            b'"' => Some(StartOutcome::Leaf(Leaf::Str {
                schema: schema.clone(),
                escaped: false,
            })),
            b'[' if depth < MAX_VALUE_DEPTH => {
                Some(StartOutcome::PushFrame(Frame::AnyArray { count: 0 }))
            }
            b'{' if depth < MAX_VALUE_DEPTH => Some(StartOutcome::PushFrame(Frame::AnyObject)),
            // Same plausible-JSON-literal restriction as the typed
            // Number/Boolean/Null arms above -- the whole point of the
            // `Any` fallback is a grammar that stays *well-formed*, not one
            // that degrades to unconstrained bytes (see the module note on
            // schemas missing `items`/`properties`).
            b'-' | b'0'..=b'9' | b't' | b'f' | b'n' => Some(StartOutcome::Leaf(Leaf::Bare)),
            _ => None,
        },
    }
}

/// What a completed value (a closed string, a closed container, or a bare
/// token that just hit its terminating delimiter) transitions to.
enum Completion {
    /// The *entire* top-level parameter value is done (the stack that was
    /// passed in was already empty): the caller must produce
    /// [`GrammarState::AfterValue`].
    Done,
    /// Still inside an enclosing frame: continue parsing at
    /// [`Leaf::FrameAfterValue`], with that frame's bookkeeping updated.
    Parser(ValueParser),
}

/// Route "a value nested directly under the top of `stack` just finished" to
/// what follows: pops nothing (closing a *container* pops before calling
/// this), but does bump an enclosing array's element count.
fn after_value(mut stack: Vec<Frame>) -> Completion {
    match stack.last_mut() {
        None => Completion::Done,
        Some(Frame::Array { count, .. }) | Some(Frame::AnyArray { count }) => {
            *count += 1;
            Completion::Parser(ValueParser {
                stack,
                leaf: Leaf::FrameAfterValue,
            })
        }
        Some(Frame::Object { .. }) | Some(Frame::AnyObject) => Completion::Parser(ValueParser {
            stack,
            leaf: Leaf::FrameAfterValue,
        }),
    }
}

/// What one byte does to a [`ValueParser`] in progress.
enum ValueOutcome {
    /// Still typing the value; the parser advanced.
    Continue(ValueParser),
    /// The whole top-level value finished on a byte that was itself the
    /// value's own terminator (a string's closing `"`, or a container's
    /// closing `]`/`}`) — the caller enters [`GrammarState::AfterValue`].
    Done,
    /// The whole top-level value was a *bare* token (number/boolean/null),
    /// which has no terminator of its own: `byte` both ended the token and
    /// is the outer arguments object's `,` delimiter — the caller enters
    /// [`GrammarState::AfterComma`].
    DoneComma,
    /// As [`ValueOutcome::DoneComma`], but `byte` was the outer arguments
    /// object's closing `}` — the caller enters [`GrammarState::Mid`].
    DoneClose,
}

fn value_completed(stack: Vec<Frame>) -> ValueOutcome {
    match after_value(stack) {
        Completion::Done => ValueOutcome::Done,
        Completion::Parser(parser) => ValueOutcome::Continue(parser),
    }
}

/// Advance a [`ValueParser`] by one byte. Purely schema-driven — once inside
/// a value, tool/parameter viability no longer matters, so this needs no
/// [`ToolCallGrammar`] access.
fn step_value(parser: &ValueParser, byte: u8) -> Option<ValueOutcome> {
    let ValueParser { stack, leaf } = parser;
    match leaf {
        Leaf::Start(schema) => match start_value(schema, byte, stack.len())? {
            StartOutcome::Leaf(new_leaf) => Some(ValueOutcome::Continue(ValueParser {
                stack: stack.clone(),
                leaf: new_leaf,
            })),
            StartOutcome::PushFrame(frame) => {
                let mut stack = stack.clone();
                stack.push(frame);
                Some(ValueOutcome::Continue(ValueParser {
                    stack,
                    leaf: Leaf::FrameStart,
                }))
            }
        },
        Leaf::Str { schema, escaped } => {
            if *escaped {
                Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::Str {
                        schema: schema.clone(),
                        escaped: false,
                    },
                }))
            } else {
                match byte {
                    b'\\' => Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::Str {
                            schema: schema.clone(),
                            escaped: true,
                        },
                    })),
                    b'"' => Some(value_completed(stack.clone())),
                    _ => Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::Str {
                            schema: schema.clone(),
                            escaped: false,
                        },
                    })),
                }
            }
        }
        Leaf::Bare => match byte {
            b'"' | b'\\' => None,
            b',' | b'}' | b']' => match after_value(stack.clone()) {
                Completion::Done => match byte {
                    b',' => Some(ValueOutcome::DoneComma),
                    b'}' => Some(ValueOutcome::DoneClose),
                    // `]` with nothing enclosing: no array to close.
                    _ => None,
                },
                Completion::Parser(next) => step_value(&next, byte),
            },
            _ => Some(ValueOutcome::Continue(ValueParser {
                stack: stack.clone(),
                leaf: Leaf::Bare,
            })),
        },
        Leaf::FrameStart => match stack.last() {
            Some(Frame::Array { items, .. }) => {
                if byte == b']' {
                    let mut stack = stack.clone();
                    stack.pop();
                    Some(value_completed(stack))
                } else {
                    let items = items.clone();
                    step_value(
                        &ValueParser {
                            stack: stack.clone(),
                            leaf: Leaf::Start(items),
                        },
                        byte,
                    )
                }
            }
            Some(Frame::AnyArray { .. }) => {
                if byte == b']' {
                    let mut stack = stack.clone();
                    stack.pop();
                    Some(value_completed(stack))
                } else {
                    step_value(
                        &ValueParser {
                            stack: stack.clone(),
                            leaf: Leaf::Start(Rc::new(Value::Null)),
                        },
                        byte,
                    )
                }
            }
            Some(Frame::Object { obj, seen }) => match byte {
                b'"' => {
                    let cands: Vec<u16> = (0..obj.props.len() as u16)
                        .filter(|index| !seen.contains(index))
                        .collect();
                    Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::ObjectKey { cands, pos: 0 },
                    }))
                }
                b'}' if required_satisfied(obj, seen) => {
                    let mut stack = stack.clone();
                    stack.pop();
                    Some(value_completed(stack))
                }
                _ => None,
            },
            Some(Frame::AnyObject) => match byte {
                b'"' => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::AnyKey { escaped: false },
                })),
                b'}' => {
                    let mut stack = stack.clone();
                    stack.pop();
                    Some(value_completed(stack))
                }
                _ => None,
            },
            None => None,
        },
        Leaf::FrameComma => match stack.last() {
            Some(Frame::Array { items, .. }) => {
                let items = items.clone();
                step_value(
                    &ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::Start(items),
                    },
                    byte,
                )
            }
            Some(Frame::AnyArray { .. }) => step_value(
                &ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::Start(Rc::new(Value::Null)),
                },
                byte,
            ),
            Some(Frame::Object { obj, seen }) => match byte {
                b'"' => {
                    let cands: Vec<u16> = (0..obj.props.len() as u16)
                        .filter(|index| !seen.contains(index))
                        .collect();
                    Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::ObjectKey { cands, pos: 0 },
                    }))
                }
                _ => None,
            },
            Some(Frame::AnyObject) => match byte {
                b'"' => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::AnyKey { escaped: false },
                })),
                _ => None,
            },
            None => None,
        },
        Leaf::FrameAfterValue => match stack.last() {
            Some(Frame::Array { count, .. }) => match byte {
                b',' if *count < MAX_ARRAY_ITEMS => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::FrameComma,
                })),
                b']' => {
                    let mut stack = stack.clone();
                    stack.pop();
                    Some(value_completed(stack))
                }
                _ => None,
            },
            Some(Frame::AnyArray { count }) => match byte {
                b',' if *count < MAX_ARRAY_ITEMS => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::FrameComma,
                })),
                b']' => {
                    let mut stack = stack.clone();
                    stack.pop();
                    Some(value_completed(stack))
                }
                _ => None,
            },
            Some(Frame::Object { obj, seen }) => match byte {
                // A `,` is only legal if some declared property remains
                // unseen -- otherwise (every property already typed) it can
                // never be followed by a legal key, which would strand
                // generation with no further legal byte at all.
                b',' if seen.len() < obj.props.len() => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::FrameComma,
                })),
                b'}' if required_satisfied(obj, seen) => {
                    let mut stack = stack.clone();
                    stack.pop();
                    Some(value_completed(stack))
                }
                _ => None,
            },
            Some(Frame::AnyObject) => match byte {
                b',' => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::FrameComma,
                })),
                b'}' => {
                    let mut stack = stack.clone();
                    stack.pop();
                    Some(value_completed(stack))
                }
                _ => None,
            },
            None => None,
        },
        Leaf::ObjectKey { cands, pos } => {
            let Some(Frame::Object { obj, .. }) = stack.last() else {
                return None;
            };
            if byte == b'"' {
                let matches: Vec<u16> = cands
                    .iter()
                    .copied()
                    .filter(|&index| obj.props[index as usize].name.len() == *pos)
                    .collect();
                match matches.as_slice() {
                    [only] => Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::ObjectColon { prop: *only },
                    })),
                    _ => None,
                }
            } else {
                let narrowed: Vec<u16> = cands
                    .iter()
                    .copied()
                    .filter(|&index| obj.props[index as usize].name.get(*pos) == Some(&byte))
                    .collect();
                if narrowed.is_empty() {
                    None
                } else {
                    Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::ObjectKey {
                            cands: narrowed,
                            pos: pos + 1,
                        },
                    }))
                }
            }
        }
        Leaf::ObjectColon { prop } => {
            if byte != b':' {
                return None;
            }
            let Some(Frame::Object { obj, .. }) = stack.last() else {
                return None;
            };
            let schema = obj.props[*prop as usize].schema.clone();
            let mut stack = stack.clone();
            if let Some(Frame::Object { seen, .. }) = stack.last_mut() {
                seen.push(*prop);
            }
            Some(ValueOutcome::Continue(ValueParser {
                stack,
                leaf: Leaf::Start(schema),
            }))
        }
        Leaf::AnyKey { escaped } => {
            if *escaped {
                Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::AnyKey { escaped: false },
                }))
            } else {
                match byte {
                    b'\\' => Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::AnyKey { escaped: true },
                    })),
                    b'"' => Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::AnyColon,
                    })),
                    _ => Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::AnyKey { escaped: false },
                    })),
                }
            }
        }
        Leaf::AnyColon => {
            if byte != b':' {
                return None;
            }
            Some(ValueOutcome::Continue(ValueParser {
                stack: stack.clone(),
                leaf: Leaf::Start(Rc::new(Value::Null)),
            }))
        }
    }
}

/// A byte-level acceptor for schema-constrained tool-call decoding, built from
/// the tools offered in one prompt.
#[derive(Debug, Clone)]
pub struct ToolCallGrammar {
    /// Tool names as raw bytes, indexed by tool.
    tool_names: Vec<Vec<u8>>,
    /// Parameter-name bytes per tool: `params[tool][param]`.
    params: Vec<Vec<Vec<u8>>>,
    /// Raw JSON-Schema fragment for each parameter (`Value::Null` when none
    /// is known), aligned 1:1 with `params[tool][param]`. Used only for
    /// decode-time value-copy constraints (Phase 51 / ADR 104 D1); the
    /// structural acceptor above never looks at this.
    param_schemas: Vec<Vec<Value>>,
}

impl ToolCallGrammar {
    /// Build the grammar for the tools offered in one example
    /// ([`crate::schema::Example::tools`]). `Enum` parameter types carry
    /// their declared variants forward as a synthesized `{"enum": [...]}`
    /// schema fragment, so the value-copy layer treats them identically to a
    /// JSON-Schema `enum` even though this typed catalog has no raw schema.
    pub fn new(tools: &[ToolSchema]) -> Self {
        let tool_names = tools.iter().map(|t| t.name.clone().into_bytes()).collect();
        let params = tools
            .iter()
            .map(|t| {
                t.params
                    .iter()
                    .map(|p| p.name.clone().into_bytes())
                    .collect()
            })
            .collect();
        let param_schemas = tools
            .iter()
            .map(|t| {
                t.params
                    .iter()
                    .map(|p| match &p.ty {
                        crate::schema::ParamType::Enum(variants) => {
                            serde_json::json!({ "enum": variants })
                        }
                        _ => Value::Null,
                    })
                    .collect()
            })
            .collect();
        Self {
            tool_names,
            params,
            param_schemas,
        }
    }

    /// Build the grammar from OpenAI-style tool definitions (the shape
    /// [`crate::tool_contract::ToolContract::openai_definition`] produces and
    /// the deployed runtime actually offers) — the JSON-Schema `parameters`
    /// carry `pattern`/`enum`/length fragments the typed [`ToolSchema`] path
    /// above cannot express, which is exactly what the value-copy layer needs.
    pub fn from_openai_tools(tools: &[Value]) -> Result<Self, ToolCallGrammarError> {
        let mut tool_names = Vec::with_capacity(tools.len());
        let mut params = Vec::with_capacity(tools.len());
        let mut param_schemas = Vec::with_capacity(tools.len());
        for (index, definition) in tools.iter().enumerate() {
            let function =
                definition
                    .get("function")
                    .ok_or_else(|| ToolCallGrammarError::InvalidShape {
                        index,
                        message: "tool definition has no function object".into(),
                    })?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| ToolCallGrammarError::InvalidShape {
                    index,
                    message: "tool function has no string name".into(),
                })?;
            let properties = function
                .get("parameters")
                .and_then(|schema| schema.get("properties"))
                .and_then(Value::as_object);
            let mut names = Vec::new();
            let mut schemas = Vec::new();
            if let Some(properties) = properties {
                for (name, schema) in properties {
                    names.push(name.clone().into_bytes());
                    schemas.push(schema.clone());
                }
            }
            tool_names.push(name.as_bytes().to_vec());
            params.push(names);
            param_schemas.push(schemas);
        }
        Ok(Self {
            tool_names,
            params,
            param_schemas,
        })
    }

    /// The start state: matching [`PRE`] from offset 0.
    pub fn start(&self) -> GrammarState {
        GrammarState::Pre(0)
    }

    pub fn tool_count(&self) -> usize {
        self.tool_names.len()
    }

    pub fn tool_name(&self, tool: u16) -> &str {
        std::str::from_utf8(&self.tool_names[tool as usize]).unwrap_or("")
    }

    pub fn tool_index(&self, name: &str) -> Option<u16> {
        self.tool_names
            .iter()
            .position(|candidate| candidate.as_slice() == name.as_bytes())
            .map(|index| index as u16)
    }

    pub fn param_count(&self, tool: u16) -> usize {
        self.params[tool as usize].len()
    }

    pub fn param_name(&self, tool: u16, param: u16) -> &str {
        std::str::from_utf8(&self.params[tool as usize][param as usize]).unwrap_or("")
    }

    pub fn param_index(&self, tool: u16, name: &[u8]) -> Option<u16> {
        self.params[tool as usize]
            .iter()
            .position(|candidate| candidate.as_slice() == name)
            .map(|index| index as u16)
    }

    /// The raw JSON-Schema fragment declared for `tool`'s parameter named
    /// `param` (exact byte match), if any.
    pub fn param_schema(&self, tool: u16, param: &[u8]) -> Option<&Value> {
        let index = self.param_index(tool, param)?;
        Some(&self.param_schemas[tool as usize][index as usize])
    }

    /// If `state` is inside a string value (at any nesting depth), the exact
    /// schema fragment governing it (Phase 51 / ADR 104 D1 — lets the decoder
    /// look up that string's [`ValueConstraint`] regardless of whether it is
    /// a top-level parameter or nested inside an array/object).
    pub fn current_string_field<'a>(&self, state: &'a GrammarState) -> Option<&'a Value> {
        match state {
            GrammarState::Value { parser, .. } => match &parser.leaf {
                Leaf::Str { schema, .. } => Some(schema.as_ref()),
                _ => None,
            },
            _ => None,
        }
    }

    /// Resolve the JSON-Schema fragment governing `param`'s value while it
    /// could still belong to any of `viable`'s tools: if every viable tool
    /// declares the exact same fragment (the overwhelmingly common case —
    /// most prompts offer tools with disjoint parameter names, or one tool),
    /// use it; if they disagree, or none declare a schema at all, fall back
    /// to [`SchemaKind::Any`] (via a `null` fragment) rather than risk
    /// constraining the decoder against the wrong tool's shape.
    fn resolve_structural_schema(&self, viable: &[u16], param: &[u8]) -> SchemaRef {
        let mut resolved: Option<&Value> = None;
        for &tool in viable {
            let Some(index) = self.param_index(tool, param) else {
                continue;
            };
            let schema = &self.param_schemas[tool as usize][index as usize];
            match resolved {
                None => resolved = Some(schema),
                Some(previous) if previous == schema => {}
                Some(_) => return Rc::new(Value::Null),
            }
        }
        Rc::new(resolved.cloned().unwrap_or(Value::Null))
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
                    // typed, and narrow viable tools to their tools. Every
                    // surviving (tool, param) pair's name is byte-identical
                    // (that is exactly what "fully typed at this position"
                    // means), so any one of them gives the exact key bytes.
                    let mut tools: Vec<u16> = Vec::new();
                    let mut param_name: Option<Vec<u8>> = None;
                    for (t, p) in cands {
                        if self.params[*t as usize][*p as usize].len() == *pos {
                            tools.push(*t);
                            if param_name.is_none() {
                                param_name = Some(self.params[*t as usize][*p as usize].clone());
                            }
                        }
                    }
                    tools.sort_unstable();
                    tools.dedup();
                    if tools.is_empty() {
                        None
                    } else {
                        Some(Colon {
                            viable: tools,
                            param: param_name.unwrap_or_default(),
                        })
                    }
                } else {
                    let nc: Vec<(u16, u16)> = cands
                        .iter()
                        .copied()
                        .filter(|(t, p)| {
                            self.params[*t as usize][*p as usize].get(*pos) == Some(&byte)
                        })
                        .collect();
                    if nc.is_empty() {
                        None
                    } else {
                        Some(Key {
                            cands: nc,
                            pos: *pos + 1,
                        })
                    }
                }
            }
            Colon { viable, param } => {
                if byte == b':' {
                    let schema = self.resolve_structural_schema(viable, param);
                    Some(Value {
                        viable: viable.clone(),
                        parser: ValueParser {
                            stack: Vec::new(),
                            leaf: Leaf::Start(schema),
                        },
                    })
                } else {
                    None
                }
            }
            Value { viable, parser } => match step_value(parser, byte)? {
                ValueOutcome::Continue(next) => Some(Value {
                    viable: viable.clone(),
                    parser: next,
                }),
                ValueOutcome::Done => Some(AfterValue(viable.clone())),
                ValueOutcome::DoneComma => Some(AfterComma(viable.clone())),
                ValueOutcome::DoneClose => Some(Mid {
                    viable: viable.clone(),
                    pos: 0,
                }),
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
                    if complete { Some(CloseOuter) } else { None }
                } else {
                    let nc: Vec<u16> = cands
                        .iter()
                        .copied()
                        .filter(|&t| self.tool_names[t as usize].get(*pos) == Some(&byte))
                        .collect();
                    if nc.is_empty() {
                        None
                    } else {
                        Some(Name {
                            cands: nc,
                            pos: *pos + 1,
                        })
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

// ── Value-copy decoding (Phase 51 / ADR 104 D1) ────────────────────────────
//
// The structural grammar above forces valid keys/tool names but leaves
// *values* free (see the module docs). Measurement showed the residual
// failure once names are right is the model mis-copying identifier-shaped
// string values (hex revision tokens, note ids) from the context -- it drops
// or corrupts a character inside a run of doubled hex digits. Since these
// values are grounded facts already present verbatim in the prompt (the tool
// result / prior turn that produced them), the fix is to make the *decoder*
// copy them rather than hoping the model spells them correctly.

/// What governs the legal completions of one JSON string argument value, once
/// decode-time context is available. Distinct from [`crate::schema::ParamType`]
/// (a structural type check on the finished value) -- this is what the
/// decoder is allowed to *emit* while typing the value's bytes.
#[derive(Debug, Clone, PartialEq)]
pub enum ValueConstraint {
    /// The value must be exactly one of these strings (JSON-Schema `enum`,
    /// or a [`crate::schema::ParamType::Enum`]'s declared variants).
    Enum { candidates: Vec<String> },
    /// The value must be copied byte-for-byte from one of these context
    /// substrings: a `pattern` (and/or fixed-length identifier-shaped)
    /// property, with at least one match found in the rendered context.
    Copy { candidates: Vec<String> },
    /// A `pattern`/fixed-length property was declared, but no substring of
    /// the context matched it -- there is nothing to copy from, so the value
    /// is left byte-unconstrained during decoding (the pattern can still be
    /// checked post-hoc; refusing to offer *any* continuation here would
    /// strand generation with no legal next byte, which is worse).
    PatternFallback { pattern: String },
    /// No decode-time constraint (free-form string: note bodies, queries).
    Free,
}

/// A byte-trie over a small set of candidate strings. Built once per
/// generation call and then queried once per vocabulary token per decode
/// step, so lookup cost must not scale with candidate count (only with the
/// query's own byte length) -- a per-step linear scan over candidates times
/// an 8k-token vocabulary would be the wrong complexity class.
#[derive(Debug, Clone, Default)]
pub struct CopyTrie {
    children: Vec<BTreeMap<u8, usize>>,
    terminal: Vec<bool>,
}

impl CopyTrie {
    pub fn new(candidates: &[String]) -> Self {
        let mut trie = Self {
            children: vec![BTreeMap::new()],
            terminal: vec![false],
        };
        for candidate in candidates {
            let mut node = 0usize;
            for &byte in candidate.as_bytes() {
                node = match trie.children[node].get(&byte) {
                    Some(&next) => next,
                    None => {
                        trie.children.push(BTreeMap::new());
                        trie.terminal.push(false);
                        let next = trie.children.len() - 1;
                        trie.children[node].insert(byte, next);
                        next
                    }
                };
            }
            trie.terminal[node] = true;
        }
        trie
    }

    fn walk(&self, bytes: &[u8]) -> Option<usize> {
        let mut node = 0usize;
        for &byte in bytes {
            node = *self.children[node].get(&byte)?;
        }
        Some(node)
    }

    /// Is `bytes` a prefix of at least one candidate (including being a full
    /// candidate itself)?
    pub fn is_prefix_of_any(&self, bytes: &[u8]) -> bool {
        self.walk(bytes).is_some()
    }

    /// Does `bytes` exactly equal one whole candidate?
    pub fn is_exact_match(&self, bytes: &[u8]) -> bool {
        self.walk(bytes).is_some_and(|node| self.terminal[node])
    }
}

/// Strip a leading `^`/trailing `$` (this project's patterns are always
/// fully anchored) and compile the remainder unanchored, so [`regex::Regex::find_iter`]
/// can scan for occurrences anywhere in the context rather than requiring the
/// whole context to match.
fn compile_unanchored(pattern: &str) -> Option<regex::Regex> {
    let mut body = pattern.trim();
    if let Some(rest) = body.strip_prefix('^') {
        body = rest;
    }
    if let Some(rest) = body.strip_suffix('$') {
        body = rest;
    }
    regex::Regex::new(body).ok()
}

/// Derive the decode-time [`ValueConstraint`] for one property from its
/// JSON-Schema fragment and the rendered prompt context.
///
/// - `enum` → [`ValueConstraint::Enum`].
/// - `pattern`, or an identifier-shaped fixed length (`minLength == maxLength`)
///   with no explicit character class → scan `context` for matches and copy
///   from them; zero matches degrades to [`ValueConstraint::PatternFallback`]
///   rather than leaving the decoder with no legal continuation.
/// - anything else → [`ValueConstraint::Free`].
pub fn derive_value_constraint(context: &str, schema: &Value) -> ValueConstraint {
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let candidates: Vec<String> = values
            .iter()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect();
        if !candidates.is_empty() {
            return ValueConstraint::Enum { candidates };
        }
    }

    let pattern = schema.get("pattern").and_then(Value::as_str);
    let min_len = schema.get("minLength").and_then(Value::as_u64);
    let max_len = schema.get("maxLength").and_then(Value::as_u64);
    let fixed_length = matches!((min_len, max_len), (Some(lo), Some(hi)) if lo == hi && lo > 0);

    if pattern.is_none() && !fixed_length {
        return ValueConstraint::Free;
    }

    // No explicit pattern but a pinned exact length: best-effort identifier
    // detection using a generic non-whitespace token of that length (the
    // character class itself is unknown without a pattern).
    let effective_pattern = pattern
        .map(str::to_string)
        .unwrap_or_else(|| format!("\\S{{{}}}", min_len.unwrap_or_default()));

    let Some(compiled) = compile_unanchored(&effective_pattern) else {
        return ValueConstraint::PatternFallback {
            pattern: effective_pattern,
        };
    };

    let mut seen = std::collections::BTreeSet::new();
    let mut candidates = Vec::new();
    for occurrence in compiled.find_iter(context) {
        let text = occurrence.as_str().to_string();
        if seen.insert(text.clone()) {
            candidates.push(text);
        }
    }
    if candidates.is_empty() {
        ValueConstraint::PatternFallback {
            pattern: effective_pattern,
        }
    } else {
        ValueConstraint::Copy { candidates }
    }
}

/// Precomputed [`ValueConstraint`]s (and their [`CopyTrie`], when applicable)
/// for one generation call. Deriving a regex and scanning the context is done
/// once up front rather than per decode step, so decoding cost stays
/// independent of how long generation runs.
///
/// Two lookup paths share the same up-front work:
/// - `entries` is keyed by (tool, top-level parameter) — used only by
///   [`build_decode_report`]'s post-hoc, name-based lookup once generation
///   has finished.
/// - `by_schema` is keyed by the *exact schema fragment itself* (content
///   equality, via [`Value`]'s `Eq`) and covers every fragment reachable from
///   any offered tool's parameters, including nested `items`/`properties`
///   fragments — this is what decode-time [`ConstrainedDecodeState::step`]
///   uses via [`ToolCallGrammar::current_string_field`], so a citation's
///   `source_id` gets the same pattern-copy treatment a top-level parameter
///   would, wherever it is nested (Phase 51 / ADR 104 D1).
#[derive(Debug, Clone, Default)]
pub struct ValueConstraintTable {
    entries: Vec<Vec<(ValueConstraint, Option<CopyTrie>)>>,
    by_schema: HashMap<Value, (ValueConstraint, Option<CopyTrie>)>,
}

impl ValueConstraintTable {
    pub fn build(grammar: &ToolCallGrammar, context: &str) -> Self {
        let entries = (0..grammar.tool_count() as u16)
            .map(|tool| {
                (0..grammar.param_count(tool) as u16)
                    .map(|param| {
                        let schema =
                            grammar.param_schema(tool, grammar.param_name(tool, param).as_bytes());
                        let constraint = schema
                            .filter(|schema| !schema.is_null())
                            .map(|schema| derive_value_constraint(context, schema))
                            .unwrap_or(ValueConstraint::Free);
                        let trie = match &constraint {
                            ValueConstraint::Enum { candidates }
                            | ValueConstraint::Copy { candidates } => {
                                Some(CopyTrie::new(candidates))
                            }
                            _ => None,
                        };
                        (constraint, trie)
                    })
                    .collect()
            })
            .collect();

        let mut by_schema = HashMap::new();
        for tool in 0..grammar.tool_count() as u16 {
            for param in 0..grammar.param_count(tool) as u16 {
                if let Some(schema) =
                    grammar.param_schema(tool, grammar.param_name(tool, param).as_bytes())
                {
                    collect_schema_leaves(schema, context, &mut by_schema);
                }
            }
        }

        Self { entries, by_schema }
    }

    pub fn entry(&self, tool: u16, param: u16) -> Option<&(ValueConstraint, Option<CopyTrie>)> {
        self.entries.get(tool as usize)?.get(param as usize)
    }

    /// Resolve the constraint governing the string value currently governed
    /// by `schema` (whatever nesting depth it came from — see
    /// [`ToolCallGrammar::current_string_field`]). Always succeeds:
    /// a schema this table never saw (or `null`, used for the unresolvable/
    /// disagreeing-tools case) falls back to [`ValueConstraint::Free`] rather
    /// than stranding generation.
    pub fn resolve_schema(&self, schema: &Value) -> (&ValueConstraint, Option<&CopyTrie>) {
        match self.by_schema.get(schema) {
            Some((constraint, trie)) => (constraint, trie.as_ref()),
            None => (&FREE_CONSTRAINT, None),
        }
    }
}

/// A static [`ValueConstraint::Free`] to hand back a `&ValueConstraint` when
/// [`ValueConstraintTable::resolve_schema`] deliberately declines to
/// constrain.
static FREE_CONSTRAINT: ValueConstraint = ValueConstraint::Free;

/// Recursively derive and cache the [`ValueConstraint`] for `schema` and
/// every fragment nested under its `items`/`properties`, keyed by the exact
/// fragment ([`Value`]'s content equality) so [`ValueConstraintTable::resolve_schema`]
/// can look any of them up directly from the [`ValueParser`] leaf that is
/// typing them. Schemas in this project are trees, not graphs (no
/// self-reference), so plain recursion cannot loop.
fn collect_schema_leaves(
    schema: &Value,
    context: &str,
    out: &mut HashMap<Value, (ValueConstraint, Option<CopyTrie>)>,
) {
    if out.contains_key(schema) {
        return;
    }
    let constraint = derive_value_constraint(context, schema);
    let trie = match &constraint {
        ValueConstraint::Enum { candidates } | ValueConstraint::Copy { candidates } => {
            Some(CopyTrie::new(candidates))
        }
        _ => None,
    };
    out.insert(schema.clone(), (constraint, trie));

    if let Some(items) = schema.get("items") {
        collect_schema_leaves(items, context, out);
    }
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for sub_schema in properties.values() {
            collect_schema_leaves(sub_schema, context, out);
        }
    }
}

/// What a value-copy-constrained decode did for one generated call —
/// telemetry for the daemon and eval evidence, not used to change scoring.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecodeReport {
    /// Whether a grammar was supplied at all (false for pure free-text turns
    /// with no offered tools, which decode exactly like the unconstrained
    /// path always did).
    pub constrained: bool,
    pub copy_fields: Vec<CopyFieldOutcome>,
    /// Phase 51 / ADR 104 D2 mechanism 1: how many string values the
    /// identifier hard-copy lock engaged on during this decode (a
    /// content-so-far became an unambiguous identifier prefix).
    pub identifier_copy_locks: u32,
    /// How many of those locks were subsequently released mid-value because
    /// no vocabulary token could extend the match (the documented
    /// no-deadlock escape hatch), rather than the value simply completing
    /// while still locked.
    pub identifier_copy_releases: u32,
    /// Phase 51 / ADR 104 D2 mechanism 2: how many decode steps the soft
    /// suffix-copy bias picked a within-margin alternative over the masked
    /// argmax because it continued a verbatim context match.
    pub copy_bias_overrides: u32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CopyFieldOutcome {
    pub tool: String,
    pub property: String,
    pub candidates_in_context: usize,
    pub copied: bool,
    pub fallback: bool,
}

/// Recover a [`DecodeReport`] from the finished decode by parsing the emitted
/// tool call (if any) back out of the runtime envelope, rather than
/// bookkeeping every decode step live — the finished text already carries
/// everything needed to say which fields were copy-constrained and whether
/// the emitted value landed on a candidate.
pub fn build_decode_report(
    grammar: &ToolCallGrammar,
    table: &ValueConstraintTable,
    output: &str,
) -> DecodeReport {
    let mut report = DecodeReport {
        constrained: true,
        ..DecodeReport::default()
    };
    let Some(body) = output
        .strip_prefix(std::str::from_utf8(RUNTIME_TAG_OPEN).unwrap_or_default())
        .and_then(|rest| {
            rest.strip_suffix(std::str::from_utf8(RUNTIME_TAG_CLOSE).unwrap_or_default())
        })
    else {
        return report; // Answer/Gap text -- no tool-call fields to report on.
    };
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return report;
    };
    let Some(name) = value.get("name").and_then(Value::as_str) else {
        return report;
    };
    let Some(tool) = grammar.tool_index(name) else {
        return report;
    };
    let Some(arguments) = value.get("arguments").and_then(Value::as_object) else {
        return report;
    };
    for (key, emitted) in arguments {
        let Some(param) = grammar.param_index(tool, key.as_bytes()) else {
            continue;
        };
        let Some((constraint, _)) = table.entry(tool, param) else {
            continue;
        };
        match constraint {
            ValueConstraint::Enum { candidates } | ValueConstraint::Copy { candidates } => {
                let copied = emitted
                    .as_str()
                    .is_some_and(|text| candidates.iter().any(|candidate| candidate == text));
                report.copy_fields.push(CopyFieldOutcome {
                    tool: name.to_string(),
                    property: key.clone(),
                    candidates_in_context: candidates.len(),
                    copied,
                    fallback: false,
                });
            }
            ValueConstraint::PatternFallback { .. } => {
                report.copy_fields.push(CopyFieldOutcome {
                    tool: name.to_string(),
                    property: key.clone(),
                    candidates_in_context: 0,
                    copied: false,
                    fallback: true,
                });
            }
            ValueConstraint::Free => {}
        }
    }
    report
}

/// A position in the *runtime* envelope acceptor: Akh-Medu's deployed decoder
/// wraps a tool call in `<tool_call>\n...\n</tool_call>` (see
/// [`crate::knowledge_workflow::KnowledgeAction::render_runtime`]), and must
/// also allow plain free text (an `Answer` or the `Knowledge gap:` abstention
/// envelope) with no structure imposed at all. Only once the exact tag prefix
/// has been fully typed does the strict [`ToolCallGrammar`] apply; any
/// divergence commits to free text for the rest of generation, which is
/// exactly as permissive as the pre-existing unconstrained decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeCallState {
    /// No commitment yet; `usize` is how many bytes of the opening tag match
    /// so far.
    Undecided(usize),
    /// Fully inside the tag; delegates to the inner JSON grammar.
    InCall(GrammarState),
    /// The inner call reached [`GrammarState::End`]; matching the closing tag.
    Closing(usize),
    /// Diverged from the opening tag -- free text for the remainder of
    /// generation (this is deliberately a strict superset of "gap" and
    /// "answer" text: no structure is imposed on either).
    Free,
    /// A complete, accepted tool call (closing tag matched). Only `EOS` is
    /// legal from here.
    Done,
}

/// Wraps [`ToolCallGrammar`] with the runtime's `<tool_call>` envelope. See
/// [`RuntimeCallState`] for why free text is never blocked.
#[derive(Debug, Clone)]
pub struct RuntimeCallGrammar {
    inner: ToolCallGrammar,
}

impl RuntimeCallGrammar {
    pub fn new(tools: &[ToolSchema]) -> Self {
        Self {
            inner: ToolCallGrammar::new(tools),
        }
    }

    pub fn from_openai_tools(tools: &[Value]) -> Result<Self, ToolCallGrammarError> {
        Ok(Self {
            inner: ToolCallGrammar::from_openai_tools(tools)?,
        })
    }

    pub fn inner(&self) -> &ToolCallGrammar {
        &self.inner
    }

    pub fn start(&self) -> RuntimeCallState {
        RuntimeCallState::Undecided(0)
    }

    pub fn step(&self, state: &RuntimeCallState, byte: u8) -> Option<RuntimeCallState> {
        use RuntimeCallState::*;
        match state {
            Undecided(i) => {
                if *i < RUNTIME_TAG_OPEN.len() && byte == RUNTIME_TAG_OPEN[*i] {
                    if i + 1 == RUNTIME_TAG_OPEN.len() {
                        Some(InCall(self.inner.start()))
                    } else {
                        Some(Undecided(i + 1))
                    }
                } else {
                    Some(Free)
                }
            }
            InCall(inner_state) => {
                if matches!(inner_state, GrammarState::End) {
                    // The JSON call is complete; only the closing tag may
                    // follow (the inner grammar itself accepts no more bytes
                    // in `End`, so it must not be asked to).
                    if byte == RUNTIME_TAG_CLOSE[0] {
                        if RUNTIME_TAG_CLOSE.len() == 1 {
                            Some(Done)
                        } else {
                            Some(Closing(1))
                        }
                    } else {
                        None
                    }
                } else {
                    self.inner.step(inner_state, byte).map(InCall)
                }
            }
            Closing(i) => {
                if byte == RUNTIME_TAG_CLOSE[*i] {
                    if i + 1 == RUNTIME_TAG_CLOSE.len() {
                        Some(Done)
                    } else {
                        Some(Closing(i + 1))
                    }
                } else {
                    None
                }
            }
            Free => Some(Free),
            Done => None,
        }
    }

    pub fn advance(&self, state: &RuntimeCallState, bytes: &[u8]) -> Option<RuntimeCallState> {
        let mut current = state.clone();
        for &byte in bytes {
            current = self.step(&current, byte)?;
        }
        Some(current)
    }

    /// `EOS` is legal once free text has been committed to, or a call has
    /// fully closed -- never mid-structure.
    pub fn accepts_eos(&self, state: &RuntimeCallState) -> bool {
        matches!(state, RuntimeCallState::Free | RuntimeCallState::Done)
    }

    pub fn current_string_field<'a>(&self, state: &'a RuntimeCallState) -> Option<&'a Value> {
        match state {
            RuntimeCallState::InCall(inner) => self.inner.current_string_field(inner),
            _ => None,
        }
    }
}

// ── General copy-from-context (Phase 51 / ADR 104 D2) ──────────────────────
//
// The schema pattern/enum copy above only fires when a parameter declares a
// `pattern`/fixed-length/`enum` fragment. Measurement on the A100 checkpoint
// showed the *next* residual failure is identical in kind (a dropped/doubled/
// inserted character while copying an identifier-shaped string verbatim from
// the prompt) but on fields with **no** such schema fragment at all --
// `source_id`/`source_path` values, and even free-text `body`/`quote` runs
// that happen to reproduce a digit-bearing token from the retrieved
// evidence. Since the schema gives no candidate list to build a [`CopyTrie`]
// from for these, this section adds two context-*derived* mechanisms
// instead of a schema-derived one:
//
// 1. [`IdentifierIndex`] -- a hard lock: once a string value's emitted bytes
//    become an unambiguous prefix of a whitespace-free, digit/`:/_-.`-bearing
//    substring of the context, only bytes preserving that prefix (or, at the
//    closing `"`, completing it exactly) are legal. If no vocabulary token
//    can preserve it (checked once per decode step, not per candidate --
//    see [`ConstrainedDecodeState::identifier_lock_engaged`] and
//    [`ConstrainedDecodeState::with_identifier_lock_released`]), the lock is
//    released for the rest of that value rather than deadlocking generation.
// 2. [`SuffixCopyIndex`] -- a soft bias: the longest suffix of a string
//    value's emitted content that recurs in the context predicts what byte
//    comes next (an LZ-style copy prediction); `grug-train`'s decode loop
//    uses it to prefer an already-legal, near-argmax token that continues
//    that match over one that doesn't. This is a *ranking* choice, not a
//    legality one, so it has no presence in [`ConstrainedDecodeState::step`]
//    -- only [`ConstrainedDecodeState::active_string_content`] exposes what
//    it needs.
//
// Both derive their working set from the *unescaped* value content (a JSON
// `\"`/`\n`/`\\` escape is not what appears in the context, which is plain
// text), tracked alongside the D1 schema-trie bookkeeping in
// [`ConstrainedDecodeState::step`].

/// The first byte of how raw content byte `c` would be emitted inside a JSON
/// string: `"`, `\`, and control characters require a leading `\` escape, so
/// the literal output's first byte is `\` rather than `c` itself. Used to
/// translate a context byte (raw text) into what a *matching* generated
/// token's first output byte would have to be.
fn json_escape_first_byte(c: u8) -> u8 {
    match c {
        b'"' | b'\\' | 0x00..=0x1F => b'\\',
        other => other,
    }
}

/// The inverse mapping for one escaped byte *inside* a `\`-prefixed pair
/// (the byte immediately after a `\` in the raw decode stream): what
/// semantic character it contributes to the value's unescaped content.
/// Anything not one of JSON's short escapes (`"`, `\`, `/`, `b`, `f`, `n`,
/// `r`, `t`) is passed through literally -- `\u` unicode escapes are rare in
/// this project's training data and are approximated as their literal `u`
/// rather than decoded, which only affects the copy mechanisms' precision on
/// that rare path, never correctness of the surrounding grammar.
fn unescape_byte(byte: u8) -> u8 {
    match byte {
        b'n' => b'\n',
        b't' => b'\t',
        b'r' => b'\r',
        b'b' => 0x08,
        b'f' => 0x0C,
        other => other,
    }
}

/// Split `context` into whitespace-free "identifier tokens" for
/// [`IdentifierIndex`]: cut on Unicode whitespace and the JSON structural
/// delimiters `"` `{` `}` `[` `]` `,` -- deliberately **not** `:`, `/`, `_`,
/// `-`, `.`, since those are exactly the characters a compound identifier
/// like `textbook:484a5a7ec4e0ae11:edition-2` is built from; splitting on
/// them would shred the very substring this mechanism needs to copy whole.
/// Keeps tokens of byte length >= 8 containing at least one ASCII digit or
/// one of `:/_-.`.
fn identifier_tokens(context: &str) -> Vec<String> {
    fn push_if_qualifies(tokens: &mut Vec<String>, candidate: &str) {
        if candidate.len() < 8 {
            return;
        }
        let qualifies = candidate.chars().any(|c| c.is_ascii_digit())
            || candidate.chars().any(|c| matches!(c, ':' | '/' | '_' | '-' | '.'));
        if qualifies {
            tokens.push(candidate.to_string());
        }
    }

    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in context.chars() {
        if c.is_whitespace() || matches!(c, '"' | '{' | '}' | '[' | ']' | ',') {
            push_if_qualifies(&mut tokens, &current);
            current.clear();
        } else {
            current.push(c);
        }
    }
    push_if_qualifies(&mut tokens, &current);
    tokens.sort();
    tokens.dedup();
    tokens
}

/// The identifier-shaped substrings of one generation call's context,
/// indexed once ([`identifier_tokens`]) for repeated prefix/exact-match
/// queries during decoding (Phase 51 / ADR 104 D2 mechanism 1). Reuses
/// [`CopyTrie`] -- an identifier candidate list is exactly the same
/// "small set of strings, queried by prefix" shape the schema-pattern copy
/// trie already needed.
#[derive(Debug, Clone, Default)]
pub struct IdentifierIndex {
    trie: CopyTrie,
}

impl IdentifierIndex {
    pub fn build(context: &str) -> Self {
        Self {
            trie: CopyTrie::new(&identifier_tokens(context)),
        }
    }

    fn is_prefix_of_any(&self, bytes: &[u8]) -> bool {
        self.trie.is_prefix_of_any(bytes)
    }

    fn is_exact_match(&self, bytes: &[u8]) -> bool {
        self.trie.is_exact_match(bytes)
    }
}

/// Byte length of the hashed window [`SuffixCopyIndex`] indexes the context
/// by. A fixed window means its byte value is *invariant* across every
/// candidate suffix length being considered (they all end at the same
/// position, `content`'s end), so one hash lookup finds every candidate
/// occurrence regardless of how far it turns out to extend backward.
const SUFFIX_WINDOW: usize = 8;

/// Cap on how far a suffix match is extended backward, per the task spec.
const MAX_SUFFIX_LEN: usize = 64;

/// A hashed 8-byte-window index over one generation call's rendered context,
/// built once ([`SuffixCopyIndex::build`]) and queried once per decode step
/// (not per vocabulary token -- see the module docs' efficiency note) for
/// the soft suffix-copy bias (Phase 51 / ADR 104 D2 mechanism 2).
#[derive(Debug, Clone, Default)]
pub struct SuffixCopyIndex {
    context: Vec<u8>,
    windows: HashMap<[u8; SUFFIX_WINDOW], Vec<usize>>,
}

impl SuffixCopyIndex {
    pub fn build(context: &str) -> Self {
        let context = context.as_bytes().to_vec();
        let mut windows: HashMap<[u8; SUFFIX_WINDOW], Vec<usize>> = HashMap::new();
        if context.len() >= SUFFIX_WINDOW {
            for start in 0..=context.len() - SUFFIX_WINDOW {
                let mut key = [0u8; SUFFIX_WINDOW];
                key.copy_from_slice(&context[start..start + SUFFIX_WINDOW]);
                windows.entry(key).or_default().push(start);
            }
        }
        Self { context, windows }
    }

    /// The bytes (mapped through [`json_escape_first_byte`]) observed to
    /// follow the longest (>= [`SUFFIX_WINDOW`], capped at
    /// [`MAX_SUFFIX_LEN`]) suffix of `content` (the value's *unescaped*
    /// emitted bytes so far) that occurs anywhere in the context. `None`
    /// when even the shortest (8-byte) window has no occurrence at all --
    /// novel text, where the bias is inactive.
    pub fn continuations(&self, content: &[u8]) -> Option<std::collections::HashSet<u8>> {
        if content.len() < SUFFIX_WINDOW {
            return None;
        }
        let cap = content.len().min(MAX_SUFFIX_LEN);
        let mut key = [0u8; SUFFIX_WINDOW];
        key.copy_from_slice(&content[content.len() - SUFFIX_WINDOW..]);
        let positions = self.windows.get(&key)?;

        let mut best_len = 0usize;
        for &window_start in positions {
            let end = window_start + SUFFIX_WINDOW;
            let mut matched = SUFFIX_WINDOW;
            while matched < cap && matched < end {
                let context_index = end - matched - 1;
                let content_index = content.len() - matched - 1;
                if self.context[context_index] == content[content_index] {
                    matched += 1;
                } else {
                    break;
                }
            }
            best_len = best_len.max(matched);
        }
        if best_len < SUFFIX_WINDOW {
            return None;
        }

        let needle = &content[content.len() - best_len..];
        let mut next_bytes = std::collections::HashSet::new();
        for &window_start in positions {
            let end = window_start + SUFFIX_WINDOW;
            if end < best_len {
                continue;
            }
            let start = end - best_len;
            if self.context.get(start..end) == Some(needle)
                && let Some(&next) = self.context.get(end)
            {
                next_bytes.insert(json_escape_first_byte(next));
            }
        }
        if next_bytes.is_empty() {
            None
        } else {
            Some(next_bytes)
        }
    }
}

/// Configuration plus precomputed indices for one generation call's
/// context-derived copy mechanisms (Phase 51 / ADR 104 D1 schema-pattern
/// copy + D2 general copy-from-context): bundles [`ValueConstraintTable`]
/// (schema `pattern`/`enum`), [`IdentifierIndex`] (D2 mechanism 1), and
/// [`SuffixCopyIndex`] (D2 mechanism 2) so [`ConstrainedDecodeState::step`]
/// and `grug-train`'s decode loop each take one reference instead of
/// threading three separately, and so building all three (each a single
/// pass over the context) happens exactly once per generation call.
#[derive(Debug, Clone)]
pub struct CopyConstraints {
    pub table: ValueConstraintTable,
    pub identifiers: IdentifierIndex,
    pub suffix: SuffixCopyIndex,
    /// Phase 51 / ADR 104 D2 mechanism 1 on/off switch (default `true`).
    pub identifier_copy: bool,
    /// Phase 51 / ADR 104 D2 mechanism 2 margin in nats, `None` to disable.
    pub copy_bias_margin: Option<f32>,
}

impl CopyConstraints {
    pub fn build(
        grammar: &ToolCallGrammar,
        context: &str,
        identifier_copy: bool,
        copy_bias_margin: Option<f32>,
    ) -> Self {
        Self {
            table: ValueConstraintTable::build(grammar, context),
            identifiers: IdentifierIndex::build(context),
            suffix: SuffixCopyIndex::build(context),
            identifier_copy,
            copy_bias_margin,
        }
    }
}

/// Tracking for the D2 general copy mechanisms while inside one string
/// value; [`CopyState::Idle`] outside a string, or inside one the D1
/// schema-trie already hard-locks (see [`ConstrainedDecodeState::step`]).
#[derive(Debug, Clone, PartialEq, Eq)]
enum CopyState {
    Idle,
    Active {
        /// The value's *unescaped* content bytes typed so far.
        content: Vec<u8>,
        /// A pending `\` escape in the raw (possibly-escaped) decode stream.
        escaped: bool,
        /// Once true (an ASCII whitespace byte has appeared in `content`),
        /// the identifier lock can never engage for the rest of this value
        /// (the task spec: "do not apply this to values that begin with
        /// whitespace-containing text" -- since `content` only grows, "has
        /// whitespace anywhere so far" is equivalent to "began with
        /// whitespace" for the purpose of "never lock", and is O(1) to
        /// maintain incrementally instead of rescanning `content` every
        /// step).
        has_whitespace: bool,
        lock: IdLock,
    },
}

/// Phase 51 / ADR 104 D2 mechanism 1's per-value lock phase.
#[derive(Debug, Clone, PartialEq, Eq)]
enum IdLock {
    /// Not yet locked: too short, contains whitespace, or not (yet) a
    /// prefix of any identifier candidate.
    Unlocked,
    /// `content` is a strict prefix of >= 1 identifier candidate (or exactly
    /// equals one): only bytes preserving that stay legal.
    Locked,
    /// The strict lock would have deadlocked generation (no vocabulary
    /// token could extend the match) for at least one decode step -- see
    /// [`ConstrainedDecodeState::with_identifier_lock_released`] -- and has
    /// been released for the remainder of this value.
    Released,
}

/// Combined decode-time state: the structural [`RuntimeCallState`] plus the
/// bytes committed so far to the current value-copy-constrained string for
/// the D1 schema-trie (`Enum`/`Copy`; empty outside one, and whenever the
/// active field has no such constraint), plus [`CopyState`], the analogous
/// tracking for the D2 general context-copy mechanisms, which run whenever
/// the D1 schema-trie is not already hard-locking the value. This is the
/// single seam `generate.rs`'s decode loop drives one candidate token at a
/// time, so structural legality and value-copy legality can never drift
/// apart into separate, inconsistent checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstrainedDecodeState {
    pub frontier: RuntimeCallState,
    value_bytes: Vec<u8>,
    copy: CopyState,
}

impl ConstrainedDecodeState {
    pub fn start(grammar: &RuntimeCallGrammar) -> Self {
        Self {
            frontier: grammar.start(),
            value_bytes: Vec::new(),
            copy: CopyState::Idle,
        }
    }

    /// Advance by one byte, honoring the structural grammar, the D1
    /// schema-trie (`Enum`/`Copy`), and -- for values the schema-trie does
    /// not already hard-lock -- the D2 identifier hard-copy lock. `None`
    /// means `byte` is illegal under any of these. The D2 soft suffix-copy
    /// *bias* (mechanism 2) is a ranking choice, not a legality one, so it
    /// has no presence here -- see `grug-train`'s decode loop and
    /// [`Self::active_string_content`].
    pub fn step(
        &self,
        grammar: &RuntimeCallGrammar,
        copy_constraints: &CopyConstraints,
        byte: u8,
    ) -> Option<Self> {
        let table = &copy_constraints.table;
        let active = grammar
            .current_string_field(&self.frontier)
            .map(|schema| table.resolve_schema(schema));
        let was_in_string = grammar.current_string_field(&self.frontier).is_some();

        let next_frontier = grammar.step(&self.frontier, byte)?;
        let now_in_string = grammar.current_string_field(&next_frontier).is_some();
        let still_in_same_string = was_in_string && now_in_string;

        let mut value_bytes = self.value_bytes.clone();
        let schema_locked = matches!(
            active,
            Some((ValueConstraint::Enum { .. } | ValueConstraint::Copy { .. }, _))
        );
        match active {
            Some((ValueConstraint::Enum { .. } | ValueConstraint::Copy { .. }, Some(trie))) => {
                if still_in_same_string {
                    value_bytes.push(byte);
                    if !trie.is_prefix_of_any(&value_bytes) {
                        return None;
                    }
                } else if trie.is_exact_match(&value_bytes) {
                    value_bytes.clear();
                } else {
                    return None;
                }
            }
            _ => {
                if !still_in_same_string {
                    value_bytes.clear();
                }
            }
        }

        let copy = self.step_copy(copy_constraints, schema_locked, was_in_string, now_in_string, byte)?;

        Some(Self {
            frontier: next_frontier,
            value_bytes,
            copy,
        })
    }

    /// The D2 half of [`Self::step`], split out for readability: advances
    /// [`CopyState`] and enforces the identifier hard-copy lock. Returns
    /// `None` exactly when the lock rejects `byte` (a closing quote on an
    /// incomplete copy, or a byte that would break prefix-hood while
    /// `identifier_copy` is enabled and [`IdLock::Locked`]).
    fn step_copy(
        &self,
        copy_constraints: &CopyConstraints,
        schema_locked: bool,
        was_in_string: bool,
        now_in_string: bool,
        byte: u8,
    ) -> Option<CopyState> {
        if schema_locked {
            return Some(CopyState::Idle);
        }
        let identifiers = &copy_constraints.identifiers;
        let identifier_copy = copy_constraints.identifier_copy;

        if !was_in_string && now_in_string {
            // Opening quote just consumed: fresh tracking for the new value.
            return Some(CopyState::Active {
                content: Vec::new(),
                escaped: false,
                has_whitespace: false,
                lock: IdLock::Unlocked,
            });
        }

        if was_in_string && now_in_string {
            let (mut content, mut escaped, mut has_whitespace, mut lock) = match &self.copy {
                CopyState::Active {
                    content,
                    escaped,
                    has_whitespace,
                    lock,
                } => (content.clone(), *escaped, *has_whitespace, lock.clone()),
                // The schema-trie governed the previous byte (or this is the
                // first byte after some other transition); start fresh
                // rather than assume a mismatched variant.
                CopyState::Idle => (Vec::new(), false, false, IdLock::Unlocked),
            };

            if escaped {
                content.push(unescape_byte(byte));
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else {
                content.push(byte);
                if byte.is_ascii_whitespace() {
                    has_whitespace = true;
                }
            }

            if identifier_copy && !has_whitespace && !escaped {
                match lock {
                    IdLock::Unlocked => {
                        if content.len() >= 4 && identifiers.is_prefix_of_any(&content) {
                            lock = IdLock::Locked;
                        }
                    }
                    IdLock::Locked => {
                        if !identifiers.is_prefix_of_any(&content) {
                            return None;
                        }
                    }
                    IdLock::Released => {}
                }
            }

            return Some(CopyState::Active {
                content,
                escaped,
                has_whitespace,
                lock,
            });
        }

        // Closing byte (`was_in_string && !now_in_string`), or not in a
        // string either before or after.
        if was_in_string
            && !now_in_string
            && identifier_copy
            && let CopyState::Active {
                content,
                lock: IdLock::Locked,
                ..
            } = &self.copy
            && !identifiers.is_exact_match(content)
        {
            return None; // closing on an incomplete copy is illegal
        }
        Some(CopyState::Idle)
    }

    /// Whether the currently-active string value's identifier hard-copy lock
    /// (D2 mechanism 1) is engaged. `grug-train`'s decode loop uses this to
    /// detect a would-be deadlock (every vocabulary token illegal under the
    /// strict lock) and retry the same decode step with
    /// [`Self::with_identifier_lock_released`].
    pub fn identifier_lock_engaged(&self) -> bool {
        matches!(&self.copy, CopyState::Active { lock: IdLock::Locked, .. })
    }

    /// `self` with the active string's identifier lock forced to
    /// [`IdLock::Released`] (a no-op if not currently [`IdLock::Locked`]) --
    /// the documented escape hatch for "no identifier matches": never
    /// deadlock generation.
    pub fn with_identifier_lock_released(&self) -> Self {
        let mut copy = self.copy.clone();
        if let CopyState::Active { lock, .. } = &mut copy {
            *lock = IdLock::Released;
        }
        Self {
            frontier: self.frontier.clone(),
            value_bytes: self.value_bytes.clone(),
            copy,
        }
    }

    /// The unescaped content bytes of the string value currently being
    /// typed (`None` outside one), for `grug-train`'s soft suffix-copy bias
    /// (mechanism 2), which runs for every string value regardless of
    /// whether the D2 hard lock or the D1 schema-trie also governs it.
    pub fn active_string_content(&self) -> Option<&[u8]> {
        match &self.copy {
            CopyState::Active { content, .. } => Some(content),
            CopyState::Idle => None,
        }
    }

    /// Fold a whole candidate token's bytes through [`Self::step`]. `None`
    /// means the token is illegal starting from `self` (structurally, or by
    /// diverging from the active value-copy constraint).
    pub fn advance(
        &self,
        grammar: &RuntimeCallGrammar,
        copy_constraints: &CopyConstraints,
        bytes: &[u8],
    ) -> Option<Self> {
        let mut current = self.clone();
        for &byte in bytes {
            current = current.step(grammar, copy_constraints, byte)?;
        }
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;
    use crate::schema::{ParamSpec, ParamType, ToolCall, ToolSchema};

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
            tool(
                "book_table",
                &[
                    ("party_size", ParamType::Integer),
                    ("time", ParamType::String),
                ],
            ),
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
        assert!(
            checked > 100,
            "expected to check many examples, got {checked}"
        );
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
        let tools = vec![tool(
            "x",
            &[("b", ParamType::Integer), ("a", ParamType::String)],
        )];
        let call = ToolCall::new("x", serde_json::json!({"a": "hi", "b": 3}));
        let ex = Example {
            tools: tools.clone(),
            user_request: "irrelevant".to_string(),
            expected_call: call,
            mock_response: String::new(),
        };
        let g = ToolCallGrammar::new(&tools);
        let s = ex.expected_call_as_json_string();
        assert_eq!(
            g.advance(&g.start(), s.as_bytes()),
            Some(GrammarState::End),
            "target was: {s}"
        );
    }

    // ── Value-copy decoding (Phase 51 / ADR 104 D1) ────────────────────────

    fn openai_tool(name: &str, properties: Value, required: &[&str]) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "description": format!("{name} description"),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    }

    fn revise_tool() -> Value {
        openai_tool(
            "study_note_revise",
            serde_json::json!({
                "expected_revision": {
                    "type": "string",
                    "minLength": 64,
                    "maxLength": 64,
                    "pattern": "^[0-9a-f]{64}$"
                }
            }),
            &["expected_revision"],
        )
    }

    /// Drive a whole runtime-envelope call through the combined structural +
    /// value-copy stepper; `None` means some byte was illegal.
    fn advance_constrained(
        grammar: &RuntimeCallGrammar,
        copy: &CopyConstraints,
        text: &str,
    ) -> Option<ConstrainedDecodeState> {
        ConstrainedDecodeState::start(grammar).advance(grammar, copy, text.as_bytes())
    }

    /// Build the default (both D2 mechanisms enabled) [`CopyConstraints`]
    /// for a test's grammar + context.
    fn copy_constraints(grammar: &RuntimeCallGrammar, context: &str) -> CopyConstraints {
        CopyConstraints::build(grammar.inner(), context, true, Some(4.0))
    }

    #[test]
    fn copy_constraint_accepts_the_exact_context_hex_and_rejects_a_dropped_char() {
        // The measured failure mode: the model drops one character out of a
        // run of doubled hex digits (`c7927279` -> `c79279`).
        let hex = "a1b1c7927279d1e1f1a2b2c2d2e2f2a3b3c3d3e3f3a4b4c4d4e4d5b5c5d5e5b6";
        assert_eq!(hex.len(), 64);
        let context = format!(
            "<|im_start|>user\n<tool_response>\nrevision {hex}\n</tool_response><|im_end|>\n"
        );
        let tools = vec![revise_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let correct = format!(
            "<tool_call>\n{{\"arguments\":{{\"expected_revision\":\"{hex}\"}},\"name\":\"study_note_revise\"}}\n</tool_call>"
        );
        let end = advance_constrained(&grammar, &copy, &correct)
            .expect("the exact context hex must be a legal decode path");
        assert!(grammar.accepts_eos(&end.frontier));

        // Drop one character from the middle of the run of doubled digits --
        // exactly the measured corruption. This must have NO legal decode
        // path once the copy constraint is active.
        let corrupted_hex = format!("{}{}", &hex[..8], &hex[9..]); // 63 chars
        let corrupted =
            format!("<tool_call>\n{{\"arguments\":{{\"expected_revision\":\"{corrupted_hex}");
        assert_eq!(
            advance_constrained(&grammar, &copy, &corrupted),
            None,
            "a char-dropped copy must have no legal continuation"
        );
    }

    #[test]
    fn closing_quote_is_legal_only_once_the_full_candidate_is_typed() {
        let hex: String = "0123456789abcdef".chars().cycle().take(64).collect();
        let hex = hex.as_str();
        assert_eq!(hex.len(), 64);
        let context = format!("prior revision {hex} was recorded");
        let tools = vec![revise_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let prefix = format!(
            "<tool_call>\n{{\"arguments\":{{\"expected_revision\":\"{}",
            &hex[..63]
        );
        // Closing the string one byte early (63 of 64 chars) must be illegal.
        assert_eq!(
            advance_constrained(&grammar, &copy, &format!("{prefix}\"")),
            None,
            "closing quote must be rejected before the candidate is complete"
        );
        // The 64th byte followed by the closing quote is legal.
        let complete = format!("{prefix}{}\"", &hex[63..64]);
        assert!(advance_constrained(&grammar, &copy, &complete).is_some());
    }

    #[test]
    fn enum_constraint_allows_only_declared_variants() {
        let tools = vec![openai_tool(
            "set_mode",
            serde_json::json!({"mode_level": {"type": "string", "enum": ["low", "high"]}}),
            &["mode_level"],
        )];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, "please switch to high");

        let entry = copy.table.entry(0, 0).unwrap();
        assert_eq!(
            entry.0,
            ValueConstraint::Enum {
                candidates: vec!["low".to_string(), "high".to_string()]
            }
        );

        let good = "<tool_call>\n{\"arguments\":{\"mode_level\":\"high\"},\"name\":\"set_mode\"}\n</tool_call>";
        assert!(advance_constrained(&grammar, &copy, good).is_some());

        let bad = "<tool_call>\n{\"arguments\":{\"mode_level\":\"medium\"";
        assert_eq!(advance_constrained(&grammar, &copy, bad), None);
    }

    #[test]
    fn pattern_with_no_context_match_falls_back_instead_of_blocking_generation() {
        let tools = vec![revise_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        // No 64-hex substring anywhere in the context.
        let copy = copy_constraints(&grammar, "no revision has ever been seen");
        let entry = copy.table.entry(0, 0).unwrap();
        assert!(matches!(&entry.0, ValueConstraint::PatternFallback { .. }));
        assert!(entry.1.is_none(), "fallback must not build a copy trie");

        // Any string value must still have a legal decode path (fallback is
        // permissive, not a dead end).
        let arbitrary = "<tool_call>\n{\"arguments\":{\"expected_revision\":\"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\"},\"name\":\"study_note_revise\"}\n</tool_call>";
        assert!(advance_constrained(&grammar, &copy, arbitrary).is_some());

        let report = build_decode_report(
            grammar.inner(),
            &copy.table,
            "<tool_call>\n{\"arguments\":{\"expected_revision\":\"zzzz\"},\"name\":\"study_note_revise\"}\n</tool_call>",
        );
        assert_eq!(report.copy_fields.len(), 1);
        assert!(report.copy_fields[0].fallback);
        assert!(!report.copy_fields[0].copied);
    }

    #[test]
    fn gap_envelope_and_free_answer_text_pass_the_runtime_grammar_unconstrained() {
        let tools = vec![revise_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, "irrelevant context");

        let gap = "Knowledge gap: expected_revision\nReason: no verified evidence is available.";
        let end = advance_constrained(&grammar, &copy, gap)
            .expect("gap envelope must have a legal decode path");
        assert!(
            grammar.accepts_eos(&end.frontier),
            "EOS must be legal immediately after free text"
        );

        let answer = "The capital of France is Paris.";
        let end = advance_constrained(&grammar, &copy, answer)
            .expect("a free-text answer must have a legal decode path");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    #[test]
    fn from_openai_tools_rejects_a_malformed_definition() {
        let bad = serde_json::json!({"not": "a tool"});
        assert!(matches!(
            ToolCallGrammar::from_openai_tools(&[bad]),
            Err(ToolCallGrammarError::InvalidShape { .. })
        ));
    }

    // ── Nested (array/object) structural grammar (Phase 51 / ADR 104 D1) ───
    //
    // Reproduces the A100 4096-context checkpoint's observed failure: once
    // the old grammar's byte-unconstrained `ValueStart`/`Str`/`Bare` states
    // entered an array (`[`) or a nested object, *any* byte was legal,
    // producing raw output like `{"arguments":{"body":"...","citations":
    // [,"body":...` (a stray `,` right after `[`) and a nested
    // `<tool_call>` tag inside an unfinished citation object. The real
    // `study_note_append` tool contract (title/body/citations required;
    // citations is an array of objects with required source_id/locator/
    // quote; tags is an array of strings; frontmatter is a free-form
    // object) is reconstructed below from the shape captured in
    // `tool_contract.rs`'s and `knowledge_workflow.rs`'s own fixtures.

    fn study_note_append_tool() -> Value {
        openai_tool(
            "study_note_append",
            serde_json::json!({
                "title": {"type": "string"},
                "body": {"type": "string"},
                "citations": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "locator": {"type": "string"},
                            "quote": {"type": "string"}
                        },
                        "required": ["source_id", "locator", "quote"],
                        "additionalProperties": false
                    }
                },
                "challenge_id": {"type": "string"},
                "language": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "frontmatter": {
                    "type": "object",
                    "additionalProperties": true
                }
            }),
            &["title", "body", "citations"],
        )
    }

    fn allowed_bytes(
        grammar: &ToolCallGrammar,
        state: &GrammarState,
    ) -> std::collections::BTreeSet<u8> {
        (0u8..=255)
            .filter(|&byte| grammar.step(state, byte).is_some())
            .collect()
    }

    /// The failing test the bug report asks for: after `"citations":[`, the
    /// mask must exclude `,`/`_`/`<` (each observed verbatim in the raw
    /// garbage output) and permit only what the `items` schema (an object)
    /// allows to open one (`{`) or close the still-empty array (`]`). One
    /// byte further, after `[{`, only a key (`"`) may open — no required
    /// property has been typed yet, so `}` is also illegal there.
    #[test]
    fn nested_array_open_and_object_open_reject_structural_garbage() {
        let tools = vec![study_note_append_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();

        let prefix = br#"{"arguments":{"citations":["#;
        let after_open_bracket = grammar
            .advance(&grammar.start(), prefix)
            .expect("a legal prefix must have a legal decode path");
        let allowed = allowed_bytes(&grammar, &after_open_bracket);
        assert!(!allowed.contains(&b','), "`,` must not open an array item");
        assert!(!allowed.contains(&b'_'), "`_` must not open an array item");
        assert!(!allowed.contains(&b'<'), "`<` must not open an array item");
        assert_eq!(
            allowed,
            std::collections::BTreeSet::from(*b"{]"),
            "citations' items are objects: only `{{` opens one, or `]` closes \
             the still-empty array"
        );

        let after_open_brace = grammar.step(&after_open_bracket, b'{').unwrap();
        let allowed = allowed_bytes(&grammar, &after_open_brace);
        assert_eq!(
            allowed,
            std::collections::BTreeSet::from(*b"\""),
            "a fresh citation object with no required property typed yet \
             must only accept a key opening"
        );
    }

    #[test]
    fn array_of_strings_only_permits_a_string_open_or_empty_close() {
        let tools = vec![study_note_append_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();

        let prefix = br#"{"arguments":{"tags":["#;
        let state = grammar.advance(&grammar.start(), prefix).unwrap();
        let allowed = allowed_bytes(&grammar, &state);
        assert_eq!(allowed, std::collections::BTreeSet::from(*b"\"]"));
    }

    #[test]
    fn frontmatter_any_object_stays_well_formed_while_permitting_free_form_content() {
        let tools = vec![study_note_append_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();

        let call = r#"{"arguments":{"body":"b","citations":[{"locator":"l","quote":"q","source_id":"s"}],"frontmatter":{"list":["a","b"],"nested":{"note":"free text"}},"title":"t"},"name":"study_note_append"}"#;
        assert_eq!(
            grammar.advance(&grammar.start(), call.as_bytes()),
            Some(GrammarState::End),
            "well-formed free-form frontmatter content must be accepted"
        );

        // Right after `"frontmatter":{`, only a fresh key (`"`) or the
        // empty-object close (`}`) may follow -- a schema missing
        // `properties` degrades to a generic *valid-JSON* grammar, not to
        // unconstrained bytes.
        let prefix = br#"{"arguments":{"body":"b","citations":[{"locator":"l","quote":"q","source_id":"s"}],"frontmatter":{"#;
        let state = grammar.advance(&grammar.start(), prefix).unwrap();
        let allowed = allowed_bytes(&grammar, &state);
        assert_eq!(allowed, std::collections::BTreeSet::from(*b"\"}"));
    }

    #[test]
    fn array_length_bound_forces_close_instead_of_looping_forever() {
        let tools = vec![study_note_append_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();

        let mut state = grammar
            .advance(&grammar.start(), br#"{"arguments":{"tags":["#)
            .unwrap();
        for index in 0..MAX_ARRAY_ITEMS {
            if index > 0 {
                state = grammar.step(&state, b',').expect("comma before item");
            }
            state = grammar
                .advance(&state, format!("\"t{index}\"").as_bytes())
                .expect("one more tag item");
        }
        // The bound is reached: another comma-separated item is illegal, but
        // closing the array is still legal.
        assert_eq!(grammar.step(&state, b','), None);
        assert!(grammar.step(&state, b']').is_some());
    }

    /// Is `state` positioned inside genuinely unbounded free text (a string
    /// value, or a free-form `Any`-object key)? Both accept *any* byte
    /// indefinitely (until their own closing `"`), so the adversarial driver
    /// below must never prefer a byte there — `,` would always be "legal"
    /// and the driver would loop forever instead of ever closing the text.
    fn is_unbounded_free_text(state: &GrammarState) -> bool {
        match state {
            GrammarState::Value { parser, .. } => {
                matches!(parser.leaf, Leaf::Str { .. } | Leaf::AnyKey { .. })
            }
            _ => false,
        }
    }

    /// Greedily drive `grammar` byte-by-byte, always preferring an
    /// "adversarial" byte (`,`, `<`, `_` — exactly the garbage the bug
    /// report's raw model output produced) whenever it is legal at a
    /// *structural* position (outside an open string, where content is
    /// unconstrained and irrelevant to well-formedness), falling back to
    /// `hint`'s byte at the same position otherwise. `hint_pos` advances one
    /// byte per step regardless of which byte was actually chosen, so an
    /// adversarial substitution never desynchronizes the driver from the
    /// rest of `hint`'s structure. This models "argmax always wants
    /// garbage, but the mask only ever offers legal continuations" — the
    /// mask, not the model, is what must keep the output well-formed.
    ///
    /// Substitution is skipped when `hint` itself wants a closing delimiter
    /// (`}`/`]`): the top-level arguments object (like plain JSON-Schema
    /// `additionalProperties` objects generally) does not bound *how many
    /// times* a duplicate key may be re-typed, so an adversarial `,` is
    /// always structurally legal there and would loop forever chasing a
    /// close that never comes — a pre-existing property of the top-level
    /// envelope, not the nested-value gap this grammar fixes.
    fn adversarial_drive(grammar: &ToolCallGrammar, hint: &str) -> String {
        let hint_bytes = hint.as_bytes();
        let mut state = grammar.start();
        let mut out = Vec::new();
        let mut hint_pos = 0usize;
        while !grammar.accepts_eos(&state) {
            let hint_byte = hint_bytes.get(hint_pos).copied();
            let inside_string = is_unbounded_free_text(&state);
            let closing = matches!(hint_byte, Some(b'}') | Some(b']'));
            let mut chosen = None;
            if !inside_string && !closing {
                for &byte in b",<_" {
                    if Some(byte) == hint_byte {
                        continue;
                    }
                    if let Some(next) = grammar.step(&state, byte) {
                        chosen = Some((byte, next));
                        break;
                    }
                }
            }
            if chosen.is_none()
                && let Some(byte) = hint_byte
            {
                chosen = grammar.step(&state, byte).map(|next| (byte, next));
            }
            if chosen.is_none() {
                for byte in 0u8..=255 {
                    if let Some(next) = grammar.step(&state, byte) {
                        chosen = Some((byte, next));
                        break;
                    }
                }
            }
            let (byte, next) = chosen.expect("dead end: no legal continuation at all");
            out.push(byte);
            state = next;
            hint_pos += 1;
            assert!(out.len() < 8192, "runaway generation, EOS never reached");
        }
        String::from_utf8(out).expect("the grammar only ever emits valid UTF-8 in this test")
    }

    #[test]
    fn adversarial_argmax_preferring_garbage_bytes_still_produces_a_schema_valid_call() {
        let tools = vec![study_note_append_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();

        // A hand-written, fully schema-valid target: exercises an array of
        // objects (citations), an array of strings (tags), and a free-form
        // nested object (frontmatter) all in one call. No bare
        // (number/boolean) values anywhere, so every byte position is either
        // a structural delimiter or inside a string.
        let hint = r#"{"arguments":{"body":"the study note body","citations":[{"locator":"p.4","quote":"an exact quote","source_id":"book-1"}],"frontmatter":{"note":"ok","quality":"reviewed"},"tags":["textbook","keph-index"],"title":"A Title"},"name":"study_note_append"}"#;

        let out = adversarial_drive(&grammar, hint);

        let value: Value = serde_json::from_str(&out)
            .unwrap_or_else(|error| panic!("mask-forced output was not valid JSON: {error}\n{out}"));
        let name = value["name"]
            .as_str()
            .expect("accepted call must carry a string tool name");
        let arguments = value
            .get("arguments")
            .cloned()
            .expect("accepted call must carry an arguments object");

        let contracts = crate::tool_contract::ToolContractSnapshot::from_openai_tools(&tools)
            .expect("the same tool definition must convert to a validator contract");
        let validation = contracts.validate_call(name, &arguments);
        assert!(
            validation.valid,
            "mask-forced call failed schema validation: {:?}\n{out}",
            validation.issues
        );
    }

    #[test]
    fn adversarial_argmax_still_produces_a_valid_call_for_array_of_strings_and_nested_object() {
        // Narrower companion to the test above, isolating just the "array of
        // strings" (tags) and "nested object" (frontmatter) shapes the task
        // calls out explicitly, independent of the citations array-of-objects
        // path.
        let tools = vec![study_note_append_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();
        let hint = r#"{"arguments":{"body":"b","citations":[{"locator":"l","quote":"q","source_id":"s"}],"frontmatter":{"status":"active","tags":["x","y"]},"tags":["alpha","beta","gamma"],"title":"t"},"name":"study_note_append"}"#;

        let out = adversarial_drive(&grammar, hint);
        let value: Value = serde_json::from_str(&out)
            .unwrap_or_else(|error| panic!("mask-forced output was not valid JSON: {error}\n{out}"));
        let name = value["name"].as_str().unwrap();
        let arguments = value.get("arguments").cloned().unwrap();
        let contracts = crate::tool_contract::ToolContractSnapshot::from_openai_tools(&tools).unwrap();
        let validation = contracts.validate_call(name, &arguments);
        assert!(validation.valid, "{:?}\n{out}", validation.issues);
    }

    // ── General copy-from-context (Phase 51 / ADR 104 D2) ──────────────────

    /// A tool with a plain `source_id` string parameter and **no** declared
    /// `pattern`/`enum` -- exactly the measured residual-failure shape
    /// (schema-pattern copy, D1, cannot help these fields at all).
    fn cite_source_tool() -> Value {
        openai_tool(
            "cite_source",
            serde_json::json!({ "source_id": { "type": "string" } }),
            &["source_id"],
        )
    }

    /// A tool with a free-text `body` string parameter, no schema
    /// constraint at all.
    fn note_body_tool() -> Value {
        openai_tool(
            "note_body",
            serde_json::json!({ "body": { "type": "string" } }),
            &["body"],
        )
    }

    /// (a) The measured failure this mechanism targets: `source_id` has no
    /// schema pattern, so D1 offers nothing, but the exact value is present
    /// verbatim in the context as a compound (colon/hyphen-bearing)
    /// identifier. The doubled-run corruption from the task report
    /// (`484a5a7ec4e0ae11` -> `484a5a7a5a7ec4e0ae11`) must have no legal
    /// decode path once the identifier lock engages.
    #[test]
    fn identifier_hard_copy_reproduces_the_exact_context_id_and_rejects_a_doubled_run() {
        let id = "textbook:484a5a7ec4e0ae11:edition-2";
        let corrupted_id = "textbook:484a5a7a5a7ec4e0ae11:edition-2";
        let context =
            format!("<tool_response>\n{{\"source_id\":\"{id}\",\"locator\":\"p.1\"}}\n</tool_response>");
        let tools = vec![cite_source_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        // Sanity: with no `pattern`/`enum`, D1 leaves this field `Free`.
        assert_eq!(copy.table.entry(0, 0).unwrap().0, ValueConstraint::Free);

        let correct = format!(
            "<tool_call>\n{{\"arguments\":{{\"source_id\":\"{id}\"}},\"name\":\"cite_source\"}}\n</tool_call>"
        );
        let end = advance_constrained(&grammar, &copy, &correct)
            .expect("the exact context identifier must be a legal decode path");
        assert!(grammar.accepts_eos(&end.frontier));

        let corrupted = format!(
            "<tool_call>\n{{\"arguments\":{{\"source_id\":\"{corrupted_id}\""
        );
        assert_eq!(
            advance_constrained(&grammar, &copy, &corrupted),
            None,
            "a doubled-run copy of an un-patterned identifier must have no legal continuation"
        );
    }

    /// (c) A value that never matches any context identifier (novel text)
    /// must decode identically whether or not `identifier_copy` is enabled
    /// -- the lock must never even attempt to engage.
    #[test]
    fn identifier_hard_copy_stays_permissive_for_novel_free_text() {
        let context = "<tool_response>\nno identifiers appear anywhere in here\n</tool_response>";
        let tools = vec![note_body_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let with_lock = CopyConstraints::build(grammar.inner(), context, true, Some(4.0));
        let without_lock = CopyConstraints::build(grammar.inner(), context, false, Some(4.0));

        let call = "<tool_call>\n{\"arguments\":{\"body\":\"a wholly novel sentence about nothing in particular\"},\"name\":\"note_body\"}\n</tool_call>";
        let with_end = advance_constrained(&grammar, &with_lock, call)
            .expect("novel text must still have a legal decode path");
        let without_end = advance_constrained(&grammar, &without_lock, call)
            .expect("disabling identifier_copy must not change legality either");
        assert!(grammar.accepts_eos(&with_end.frontier));
        assert!(grammar.accepts_eos(&without_end.frontier));
    }

    /// (d) A `\"` escape inside a copy-tracked value must decode: the
    /// unescaping [`ConstrainedDecodeState::step_copy`] performs internally
    /// must not desynchronize from the grammar's own escape handling.
    #[test]
    fn escaped_quote_inside_a_copy_tracked_value_does_not_break_decoding() {
        let tools = vec![note_body_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, "irrelevant context with no identifiers at all");
        let call = "<tool_call>\n{\"arguments\":{\"body\":\"a \\\"quoted\\\" phrase\"},\"name\":\"note_body\"}\n</tool_call>";
        let end = advance_constrained(&grammar, &copy, call)
            .expect("an escaped quote inside the value must decode");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// [`SuffixCopyIndex`] must translate a raw context byte that needs JSON
    /// escaping (here, a literal `"`) into the *escape lead byte* `\` --
    /// the byte a matching generated token would actually have to start
    /// with, per the task spec's escaping requirement.
    #[test]
    fn suffix_copy_index_maps_a_quote_continuation_to_its_escape_lead_byte() {
        let context = "the id is \"abc12345\" exactly";
        let index = SuffixCopyIndex::build(context);
        let continuations = index
            .continuations(b"abc12345")
            .expect("an 8-byte suffix present verbatim in the context must be found");
        assert!(continuations.contains(&b'\\'));
        assert!(!continuations.contains(&b'"'));
    }

    /// (c) Novel content with no occurrence anywhere in the context (not
    /// even the 8-byte floor) leaves the suffix-copy bias inactive.
    #[test]
    fn suffix_copy_index_is_inactive_for_content_with_no_context_occurrence() {
        let index = SuffixCopyIndex::build("nothing relevant appears in this context at all");
        assert!(index.continuations(b"zzzzzzzzzzzz").is_none());
    }

    /// (e) A field with a declared `pattern` (D1) must never let D2's
    /// identifier index leak in, even when the context also contains a
    /// substring that would otherwise look like a valid identifier
    /// containing the same hex run.
    #[test]
    fn schema_pattern_lock_takes_precedence_over_the_identifier_index() {
        let hex = "0123456789abcdef".repeat(4);
        assert_eq!(hex.len(), 64);
        let context = format!("revision {hex} plus an unrelated id shard-{hex}-copy");
        let tools = vec![revise_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let via_decoy =
            format!("<tool_call>\n{{\"arguments\":{{\"expected_revision\":\"shard-{hex}-copy");
        assert_eq!(
            advance_constrained(&grammar, &copy, &via_decoy),
            None,
            "the decoy identifier substring must not leak into the schema-pattern trie"
        );

        let correct = format!(
            "<tool_call>\n{{\"arguments\":{{\"expected_revision\":\"{hex}\"}},\"name\":\"study_note_revise\"}}\n</tool_call>"
        );
        assert!(advance_constrained(&grammar, &copy, &correct).is_some());
    }

    /// The documented no-deadlock escape hatch: a byte that breaks an
    /// engaged identifier lock is illegal while locked, but legal again once
    /// `grug-train`'s decode loop (simulated here directly) force-releases
    /// the lock via [`ConstrainedDecodeState::with_identifier_lock_released`].
    #[test]
    fn identifier_lock_can_be_released_to_avoid_deadlocking_generation() {
        let id = "textbook:484a5a7ec4e0ae11:edition-2";
        let context = format!("\"source_id\":\"{id}\"");
        let tools = vec![cite_source_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let prefix = "<tool_call>\n{\"arguments\":{\"source_id\":\"textbook";
        let state = ConstrainedDecodeState::start(&grammar)
            .advance(&grammar, &copy, prefix.as_bytes())
            .expect("a real identifier prefix must be legal");
        assert!(
            state.identifier_lock_engaged(),
            "8 clean bytes matching the context identifier must have locked"
        );

        // `z` is not the identifier's next byte (`:`) -- illegal while locked.
        assert_eq!(state.step(&grammar, &copy, b'z'), None);

        let released = state.with_identifier_lock_released();
        assert!(!released.identifier_lock_engaged());
        assert!(released.step(&grammar, &copy, b'z').is_some());
    }
}
