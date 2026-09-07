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
        copy_fields: Vec::new(),
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

/// Combined decode-time state: the structural [`RuntimeCallState`] plus the
/// bytes committed so far to the current value-copy-constrained string
/// (empty outside one, and whenever the active field has no such constraint).
/// This is the single seam `generate.rs`'s decode loop drives one candidate
/// token at a time, so structural legality and value-copy legality can never
/// drift apart into two separate, inconsistent checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstrainedDecodeState {
    pub frontier: RuntimeCallState,
    value_bytes: Vec<u8>,
}

impl ConstrainedDecodeState {
    pub fn start(grammar: &RuntimeCallGrammar) -> Self {
        Self {
            frontier: grammar.start(),
            value_bytes: Vec::new(),
        }
    }

    /// Advance by one byte, honoring both the structural grammar and — when
    /// the byte lands inside an `Enum`/`Copy`-constrained string — the copy
    /// trie. `None` means `byte` is illegal under either.
    pub fn step(
        &self,
        grammar: &RuntimeCallGrammar,
        table: &ValueConstraintTable,
        byte: u8,
    ) -> Option<Self> {
        let active = grammar
            .current_string_field(&self.frontier)
            .map(|schema| table.resolve_schema(schema));

        let next_frontier = grammar.step(&self.frontier, byte)?;
        let still_in_same_string = grammar.current_string_field(&self.frontier).is_some()
            && grammar.current_string_field(&next_frontier).is_some();

        let mut value_bytes = self.value_bytes.clone();
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

        Some(Self {
            frontier: next_frontier,
            value_bytes,
        })
    }

    /// Fold a whole candidate token's bytes through [`Self::step`]. `None`
    /// means the token is illegal starting from `self` (structurally, or by
    /// diverging from the active value-copy constraint).
    pub fn advance(
        &self,
        grammar: &RuntimeCallGrammar,
        table: &ValueConstraintTable,
        bytes: &[u8],
    ) -> Option<Self> {
        let mut current = self.clone();
        for &byte in bytes {
            current = current.step(grammar, table, byte)?;
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
        table: &ValueConstraintTable,
        text: &str,
    ) -> Option<ConstrainedDecodeState> {
        ConstrainedDecodeState::start(grammar).advance(grammar, table, text.as_bytes())
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
        let table = ValueConstraintTable::build(grammar.inner(), &context);

        let correct = format!(
            "<tool_call>\n{{\"arguments\":{{\"expected_revision\":\"{hex}\"}},\"name\":\"study_note_revise\"}}\n</tool_call>"
        );
        let end = advance_constrained(&grammar, &table, &correct)
            .expect("the exact context hex must be a legal decode path");
        assert!(grammar.accepts_eos(&end.frontier));

        // Drop one character from the middle of the run of doubled digits --
        // exactly the measured corruption. This must have NO legal decode
        // path once the copy constraint is active.
        let corrupted_hex = format!("{}{}", &hex[..8], &hex[9..]); // 63 chars
        let corrupted =
            format!("<tool_call>\n{{\"arguments\":{{\"expected_revision\":\"{corrupted_hex}");
        assert_eq!(
            advance_constrained(&grammar, &table, &corrupted),
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
        let table = ValueConstraintTable::build(grammar.inner(), &context);

        let prefix = format!(
            "<tool_call>\n{{\"arguments\":{{\"expected_revision\":\"{}",
            &hex[..63]
        );
        // Closing the string one byte early (63 of 64 chars) must be illegal.
        assert_eq!(
            advance_constrained(&grammar, &table, &format!("{prefix}\"")),
            None,
            "closing quote must be rejected before the candidate is complete"
        );
        // The 64th byte followed by the closing quote is legal.
        let complete = format!("{prefix}{}\"", &hex[63..64]);
        assert!(advance_constrained(&grammar, &table, &complete).is_some());
    }

    #[test]
    fn enum_constraint_allows_only_declared_variants() {
        let tools = vec![openai_tool(
            "set_mode",
            serde_json::json!({"mode_level": {"type": "string", "enum": ["low", "high"]}}),
            &["mode_level"],
        )];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let table = ValueConstraintTable::build(grammar.inner(), "please switch to high");

        let entry = table.entry(0, 0).unwrap();
        assert_eq!(
            entry.0,
            ValueConstraint::Enum {
                candidates: vec!["low".to_string(), "high".to_string()]
            }
        );

        let good = "<tool_call>\n{\"arguments\":{\"mode_level\":\"high\"},\"name\":\"set_mode\"}\n</tool_call>";
        assert!(advance_constrained(&grammar, &table, good).is_some());

        let bad = "<tool_call>\n{\"arguments\":{\"mode_level\":\"medium\"";
        assert_eq!(advance_constrained(&grammar, &table, bad), None);
    }

    #[test]
    fn pattern_with_no_context_match_falls_back_instead_of_blocking_generation() {
        let tools = vec![revise_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        // No 64-hex substring anywhere in the context.
        let table = ValueConstraintTable::build(grammar.inner(), "no revision has ever been seen");
        let entry = table.entry(0, 0).unwrap();
        assert!(matches!(&entry.0, ValueConstraint::PatternFallback { .. }));
        assert!(entry.1.is_none(), "fallback must not build a copy trie");

        // Any string value must still have a legal decode path (fallback is
        // permissive, not a dead end).
        let arbitrary = "<tool_call>\n{\"arguments\":{\"expected_revision\":\"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz\"},\"name\":\"study_note_revise\"}\n</tool_call>";
        assert!(advance_constrained(&grammar, &table, arbitrary).is_some());

        let report = build_decode_report(
            grammar.inner(),
            &table,
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
        let table = ValueConstraintTable::build(grammar.inner(), "irrelevant context");

        let gap = "Knowledge gap: expected_revision\nReason: no verified evidence is available.";
        let end = advance_constrained(&grammar, &table, gap)
            .expect("gap envelope must have a legal decode path");
        assert!(
            grammar.accepts_eos(&end.frontier),
            "EOS must be legal immediately after free text"
        );

        let answer = "The capital of France is Paris.";
        let end = advance_constrained(&grammar, &table, answer)
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
}
