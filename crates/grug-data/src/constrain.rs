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
    /// After `{`: at the start of the arguments body, where either a key
    /// (`"`) may open or the object may close (`}`), carrying the
    /// currently-viable tools. `seen` is always empty here (this is the very
    /// first byte of the body, before any key has been typed) — kept as a
    /// bare [`Viable`] rather than pairing it with an always-empty `seen` for
    /// that reason; [`GrammarState::AfterComma`] is where `seen` starts
    /// mattering.
    ArgsStart(Viable),
    /// Immediately after a `,` inside the arguments body: a key **must** open
    /// (`"`), the object may not close here. `seen` is every top-level
    /// parameter name (exact bytes) already emitted, across however many
    /// still-viable tools declare it — see [`ToolCallGrammar::cands_for`] /
    /// [`ToolCallGrammar::required_satisfied_top`] for how it gates
    /// duplicate-key rejection and the `}` required-property check. This is
    /// the top-level analogue of nested [`Frame::Object`]'s `seen`, but keyed
    /// by name rather than by index into one fixed [`ObjectSchema`] — the
    /// tool (hence which schema/indices apply) is not resolved until the
    /// name at the very end, so tracking exact key bytes is what stays valid
    /// across a shrinking multi-tool [`Viable`] set.
    AfterComma { viable: Viable, seen: Vec<Vec<u8>> },
    /// Typing a parameter key: `cands` are the `(tool, param)` pairs whose
    /// parameter name still matches the bytes typed so far (already filtered
    /// to exclude every name in `seen`, so a duplicate key has zero
    /// candidates from its very first byte), `pos` is how many name bytes
    /// have been consumed.
    Key {
        cands: Vec<(u16, u16)>,
        pos: usize,
        seen: Vec<Vec<u8>>,
    },
    /// Consumed a key's closing `"`; expecting `:`. Carries the viable tools
    /// narrowed to those having the just-completed key, that key's exact
    /// name bytes (needed once `:` is consumed, to resolve the parameter's
    /// declared schema — see [`ToolCallGrammar::resolve_structural_schema`]),
    /// and `seen` as of *before* this key (the key is only added to `seen`
    /// once `:` is consumed, mirroring when nested [`Leaf::ObjectColon`]
    /// pushes into its own `seen`).
    Colon {
        viable: Viable,
        param: Vec<u8>,
        seen: Vec<Vec<u8>>,
    },
    /// Typing a JSON value governed by the just-completed key's declared
    /// schema (Phase 51 / ADR 104 D1 nested grammar): a recursive descent
    /// over `type: array`/`object`/scalar, bounded by [`MAX_VALUE_DEPTH`] and
    /// [`MAX_ARRAY_ITEMS`]. `viable` and `seen` (now including this key) are
    /// carried through unchanged so completing the whole value (however
    /// deeply nested) lands back on [`GrammarState::AfterValue`] with the
    /// same bookkeeping a scalar value always had.
    Value {
        viable: Viable,
        seen: Vec<Vec<u8>>,
        parser: ValueParser,
    },
    /// After a value: expecting `,` (another entry, only if some declared
    /// property of a still-viable tool remains unseen) or `}` (close
    /// arguments, only once some still-viable tool's every `required`
    /// property is in `seen`).
    AfterValue { viable: Viable, seen: Vec<Vec<u8>> },
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

/// Where a JSON string leaf sits relative to a `\` escape sequence
/// (Phase 51 / ADR 104 D2 v8 bug fix): restricts what byte legally follows a
/// `\` to JSON's own escape set, rather than the previous unconditional
/// "any byte closes the escape" rule, which let an illegal byte slip through
/// disguised as e.g. `\?` and produce invalid JSON (measured: the identifier
/// lock's post-match release left `\` as the highest-logit *legal* byte
/// because it bypasses the lock's own run tracking -- see
/// [`ConstrainedDecodeState::step_copy`] -- and this grammar gap then let
/// anything follow it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EscapeState {
    /// Not mid-escape: an ordinary content byte, `"` (closes the string), or
    /// `\` (opens an escape) may come next.
    Plain,
    /// Just consumed a `\`: the next byte must be one of JSON's short escape
    /// bytes (`"` `\` `/` `b` `f` `n` `r` `t`) or `u` (opens a 4-hex-digit
    /// unicode escape) -- anything else is illegal JSON.
    Pending,
    /// Inside a `\uXXXX` escape with this many hex digits still to come
    /// (starts at 4, counts down to 0 which returns to [`Self::Plain`]).
    Unicode(u8),
}

/// Where the recursive value parser is, relative to the innermost open
/// [`Frame`] (or, with an empty stack, relative to nothing — the top of the
/// whole parameter value).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Leaf {
    /// Expecting the first byte of a value governed by `schema`.
    Start(SchemaRef),
    /// Inside a string value; `escape` tracks position relative to a
    /// pending `\` escape sequence.
    Str { schema: SchemaRef, escape: EscapeState },
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
    /// Typing a free-form (`Any`-object) key; `escape` tracks position
    /// relative to a pending `\` escape sequence.
    AnyKey { escape: EscapeState },
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
                escape: EscapeState::Plain,
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
                escape: EscapeState::Plain,
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
        Leaf::Str { schema, escape } => match escape {
            EscapeState::Pending => match byte {
                // JSON's short escapes: the byte after `\` fully resolves
                // the escape and returns to plain string content.
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::Str {
                            schema: schema.clone(),
                            escape: EscapeState::Plain,
                        },
                    }))
                }
                // `\u` opens a 4-hex-digit unicode escape.
                b'u' => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::Str {
                        schema: schema.clone(),
                        escape: EscapeState::Unicode(4),
                    },
                })),
                // Anything else (e.g. `?`) is not a legal JSON escape byte.
                _ => None,
            },
            EscapeState::Unicode(remaining) => {
                if byte.is_ascii_hexdigit() {
                    let escape = if *remaining <= 1 {
                        EscapeState::Plain
                    } else {
                        EscapeState::Unicode(remaining - 1)
                    };
                    Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::Str {
                            schema: schema.clone(),
                            escape,
                        },
                    }))
                } else {
                    None
                }
            }
            EscapeState::Plain => match byte {
                b'\\' => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::Str {
                        schema: schema.clone(),
                        escape: EscapeState::Pending,
                    },
                })),
                b'"' => Some(value_completed(stack.clone())),
                _ => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::Str {
                        schema: schema.clone(),
                        escape: EscapeState::Plain,
                    },
                })),
            },
        },
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
                    leaf: Leaf::AnyKey { escape: EscapeState::Plain },
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
                    leaf: Leaf::AnyKey { escape: EscapeState::Plain },
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
        Leaf::AnyKey { escape } => match escape {
            EscapeState::Pending => match byte {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::AnyKey { escape: EscapeState::Plain },
                    }))
                }
                b'u' => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::AnyKey { escape: EscapeState::Unicode(4) },
                })),
                _ => None,
            },
            EscapeState::Unicode(remaining) => {
                if byte.is_ascii_hexdigit() {
                    let escape = if *remaining <= 1 {
                        EscapeState::Plain
                    } else {
                        EscapeState::Unicode(remaining - 1)
                    };
                    Some(ValueOutcome::Continue(ValueParser {
                        stack: stack.clone(),
                        leaf: Leaf::AnyKey { escape },
                    }))
                } else {
                    None
                }
            }
            EscapeState::Plain => match byte {
                b'\\' => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::AnyKey { escape: EscapeState::Pending },
                })),
                b'"' => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::AnyColon,
                })),
                _ => Some(ValueOutcome::Continue(ValueParser {
                    stack: stack.clone(),
                    leaf: Leaf::AnyKey { escape: EscapeState::Plain },
                })),
            },
        },
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
    /// Whether each parameter is declared `required`, aligned 1:1 with
    /// `params[tool][param]`. Gates the top-level arguments object's `}` —
    /// see [`Self::required_satisfied_top`] — the same way nested
    /// [`ObjectSchema`]'s `required` field gates a nested object's `}`.
    required: Vec<Vec<bool>>,
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
        let required = tools
            .iter()
            .map(|t| t.params.iter().map(|p| p.required).collect())
            .collect();
        Self {
            tool_names,
            params,
            param_schemas,
            required,
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
        let mut required = Vec::with_capacity(tools.len());
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
            let parameters = function.get("parameters");
            let properties = parameters
                .and_then(|schema| schema.get("properties"))
                .and_then(Value::as_object);
            let required_names: std::collections::BTreeSet<&str> = parameters
                .and_then(|schema| schema.get("required"))
                .and_then(Value::as_array)
                .map(|values| values.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let mut names = Vec::new();
            let mut schemas = Vec::new();
            let mut tool_required = Vec::new();
            if let Some(properties) = properties {
                for (name, schema) in properties {
                    names.push(name.clone().into_bytes());
                    schemas.push(schema.clone());
                    tool_required.push(required_names.contains(name.as_str()));
                }
            }
            tool_names.push(name.as_bytes().to_vec());
            params.push(names);
            param_schemas.push(schemas);
            required.push(tool_required);
        }
        Ok(Self {
            tool_names,
            params,
            param_schemas,
            required,
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

    /// The `(tool, param)` candidate list opening a fresh key under `viable`,
    /// excluding any parameter whose exact name bytes are already in `seen`
    /// — this is what makes a duplicate top-level key have zero legal
    /// candidates from its very first byte (Phase 51 / ADR 104 D1 bug fix:
    /// previously this had no `seen` filter at all, so `"tags"` could be
    /// retyped indefinitely). `seen` is empty at the very start of the
    /// arguments body ([`GrammarState::ArgsStart`]), so this degrades to
    /// "every parameter of every still-viable tool" there exactly as before.
    fn cands_for(&self, viable: &[u16], seen: &[Vec<u8>]) -> Vec<(u16, u16)> {
        let mut out = Vec::new();
        for &t in viable {
            for p in 0..self.params[t as usize].len() as u16 {
                let name = &self.params[t as usize][p as usize];
                if !seen.iter().any(|s| s == name) {
                    out.push((t, p));
                }
            }
        }
        out
    }

    /// Does `tool` have every `required` parameter's exact name bytes present
    /// in `seen`? The top-level analogue of nested [`required_satisfied`],
    /// but by name (since the top-level acceptor tracks emitted keys by their
    /// literal bytes, not by index into one fixed [`ObjectSchema`] — see
    /// [`GrammarState::AfterComma`]'s doc comment for why).
    fn required_satisfied_top(&self, tool: u16, seen: &[Vec<u8>]) -> bool {
        (0..self.params[tool as usize].len() as u16)
            .filter(|&p| self.required[tool as usize][p as usize])
            .all(|p| {
                let name = &self.params[tool as usize][p as usize];
                seen.iter().any(|s| s == name)
            })
    }

    /// Does *some* still-viable tool have all its required parameters in
    /// `seen`? Gates the top-level arguments object's `}` — legal iff at
    /// least one tool consistent with every key emitted so far could
    /// legitimately be the call's tool (its required parameters are all
    /// present), mirroring how [`Frame::Object`]'s `}` is gated by
    /// [`required_satisfied`] one level down.
    fn any_viable_required_satisfied(&self, viable: &[u16], seen: &[Vec<u8>]) -> bool {
        viable.iter().any(|&t| self.required_satisfied_top(t, seen))
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
                // `seen` is empty here (the very first byte of the body), so
                // this candidate list is every parameter of every offered
                // tool, exactly as before the duplicate-key fix.
                b'"' => Some(Key {
                    cands: self.cands_for(v, &[]),
                    pos: 0,
                    seen: Vec::new(),
                }),
                // Closing an empty arguments object is only legal if some
                // still-viable tool has no required parameters at all (e.g.
                // a zero-parameter tool: `required_satisfied_top` is
                // vacuously true over an empty parameter list).
                b'}' if self.any_viable_required_satisfied(v, &[]) => Some(Mid {
                    viable: v.clone(),
                    pos: 0,
                }),
                _ => None,
            },
            AfterComma { viable, seen } => {
                if byte == b'"' {
                    Some(Key {
                        cands: self.cands_for(viable, seen),
                        pos: 0,
                        seen: seen.clone(),
                    })
                } else {
                    None
                }
            }
            Key { cands, pos, seen } => {
                if byte == b'"' {
                    // Closing quote: keep only candidates whose name is fully
                    // typed, and narrow viable tools to their tools. Every
                    // surviving (tool, param) pair's name is byte-identical
                    // (that is exactly what "fully typed at this position"
                    // means), so any one of them gives the exact key bytes.
                    // `cands` was already filtered against `seen` when this
                    // `Key` state opened, so a name already in `seen` can
                    // never reach here (Phase 51 / ADR 104 D1 bug fix: this
                    // is what makes retyping an already-emitted key illegal).
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
                            seen: seen.clone(),
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
                            seen: seen.clone(),
                        })
                    }
                }
            }
            Colon {
                viable,
                param,
                seen,
            } => {
                if byte == b':' {
                    let schema = self.resolve_structural_schema(viable, param);
                    let mut seen = seen.clone();
                    seen.push(param.clone());
                    Some(Value {
                        viable: viable.clone(),
                        seen,
                        parser: ValueParser {
                            stack: Vec::new(),
                            leaf: Leaf::Start(schema),
                        },
                    })
                } else {
                    None
                }
            }
            Value {
                viable,
                seen,
                parser,
            } => match step_value(parser, byte)? {
                ValueOutcome::Continue(next) => Some(Value {
                    viable: viable.clone(),
                    seen: seen.clone(),
                    parser: next,
                }),
                ValueOutcome::Done => Some(AfterValue {
                    viable: viable.clone(),
                    seen: seen.clone(),
                }),
                // A bare (number/boolean/null) value's own terminator *is*
                // the arguments-body `,` — same "some unseen property
                // remains" gate as `AfterValue`'s `,` arm below.
                ValueOutcome::DoneComma if !self.cands_for(viable, seen).is_empty() => {
                    Some(AfterComma {
                        viable: viable.clone(),
                        seen: seen.clone(),
                    })
                }
                ValueOutcome::DoneComma => None,
                // Same for the bare value's terminator being the arguments
                // object's closing `}`: gated on required-property coverage.
                ValueOutcome::DoneClose if self.any_viable_required_satisfied(viable, seen) => {
                    Some(Mid {
                        viable: viable.clone(),
                        pos: 0,
                    })
                }
                ValueOutcome::DoneClose => None,
            },
            AfterValue { viable, seen } => match byte {
                // Legal only if some still-viable tool has a declared
                // parameter not yet in `seen` — otherwise a `,` could never
                // be followed by a legal key, stranding generation with no
                // further legal byte (Phase 51 / ADR 104 D1 bug fix: this is
                // the check that used to be entirely absent at the top
                // level, unlike the nested `Frame::Object` it mirrors).
                b',' if !self.cands_for(viable, seen).is_empty() => Some(AfterComma {
                    viable: viable.clone(),
                    seen: seen.clone(),
                }),
                // Legal only once some still-viable tool's every `required`
                // property is in `seen` (Phase 51 / ADR 104 D1 bug fix: this
                // used to be unconditional, so `note_id` could be missing
                // entirely when the call closed).
                b'}' if self.any_viable_required_satisfied(viable, seen) => Some(Mid {
                    viable: viable.clone(),
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

    /// The root node index -- always `0` by construction (`new` seeds
    /// `children`/`terminal` with exactly one entry before inserting any
    /// candidate). Exposed so an incremental byte-by-byte walk (e.g.
    /// [`IdentifierIndex::advance_prefix`]) can start one without first
    /// calling [`Self::is_prefix_of_any`] on an empty query.
    const ROOT: usize = 0;

    fn walk(&self, bytes: &[u8]) -> Option<usize> {
        let mut node = Self::ROOT;
        for &byte in bytes {
            node = *self.children[node].get(&byte)?;
        }
        Some(node)
    }

    /// One step of [`Self::walk`] from an already-reached `node` rather than
    /// the root -- the incremental-walk primitive [`IdentifierIndex::advance_prefix`]
    /// uses so a caller tracking its own cursor across many single-byte
    /// steps (e.g. [`CopyState::Active::run_prefix_node`]) never re-walks
    /// bytes it has already consumed.
    fn child(&self, node: usize, byte: u8) -> Option<usize> {
        self.children[node].get(&byte).copied()
    }

    /// Is `bytes` a prefix of at least one candidate (including being a full
    /// candidate itself)?
    pub fn is_prefix_of_any(&self, bytes: &[u8]) -> bool {
        self.walk(bytes).is_some()
    }

    /// Does `node` have at least one child edge whose byte satisfies
    /// `predicate`? Used by [`IdentifierIndex::extendable`] to ask "could
    /// this run still be continued by a *run byte*", independent of the
    /// literal `terminal` flag -- a node whose only children are non-run
    /// bytes (e.g. a candidate token that absorbed trailing punctuation like
    /// `Zorim-meru-74?`, since [`identifier_tokens`] only splits on a fixed
    /// delimiter set) behaves exactly like a terminal for the run lock, even
    /// when `terminal[node]` is `false`.
    fn has_child_matching(&self, node: usize, predicate: impl Fn(u8) -> bool) -> bool {
        self.children[node].keys().any(|&byte| predicate(byte))
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
    /// no vocabulary token could extend the match under either the trie lock
    /// or the left anchor (the documented total-deadlock escape hatch;
    /// Phase 51 / ADR 104 D2 v7 removed the earlier proactive release, so
    /// this now counts *only* true deadlock releases), rather than the value
    /// simply completing while still locked.
    pub identifier_copy_releases: u32,
    /// Phase 51 / ADR 104 D2 mechanism 2: how many decode steps the soft
    /// suffix-copy bias picked a within-margin alternative over the masked
    /// argmax because it continued a verbatim context match.
    pub copy_bias_overrides: u32,
    /// Phase 51 / ADR 104 D2 v7: how many run bytes the left-anchored
    /// run-start forcing (see the module notes) hard-restricted during this
    /// decode -- the mechanism that protects a run's first 1-2 bytes, before
    /// `MIN_IDENTIFIER_RUN_LEN` lets the trie lock counted in
    /// `identifier_copy_locks` engage.
    pub anchored_run_bytes: u32,
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
// 1. [`IdentifierIndex`] -- a hard lock over the identifier-shaped *suffix
//    run* of a string value's emitted bytes (not the whole value -- see the
//    redesign note below), so a value the runtime *composes* around a
//    copied identifier (a fixed URI prefix, a distilled sentence) still gets
//    that identifier locked even though the whole value itself never occurs
//    verbatim anywhere in the prompt. Once the run becomes an unambiguous
//    prefix of a digit-bearing identifier token in the context, only bytes
//    preserving that prefix are legal, until the run either completes (the
//    lock releases -- see [`Self::identifier_lock_engaged`]) or diverges. If
//    no vocabulary token can preserve an engaged lock (checked once per
//    decode step, not per candidate -- see
//    [`ConstrainedDecodeState::identifier_lock_engaged`] and
//    [`ConstrainedDecodeState::with_identifier_lock_released`]), the lock is
//    released for the rest of *that run* (not the whole value) rather than
//    forcing the "longest match" completion.
// 2. [`SuffixCopyIndex`] -- a soft bias over the **whole value** emitted so
//    far (not a recurring short suffix -- see the struct docs for why that
//    was root cause 2 of a runaway repetition bug): while the entire value
//    typed so far occurs verbatim in the context, the bytes observed to
//    follow those occurrences predict what byte comes next (an LZ-style copy
//    prediction); `grug-train`'s decode loop uses it to prefer an
//    already-legal, near-argmax token that continues that match over one
//    that doesn't. This is a *ranking* choice, not a legality one, so it has
//    no presence in [`ConstrainedDecodeState::step`] -- only
//    [`ConstrainedDecodeState::active_string_content`] exposes what it needs.
//    This mechanism is untouched by the suffix-run redesign below -- it
//    already operates on the whole value and stays as is.
//
// # Suffix-run redesign (measured on decoder 957589e1, paired v5)
//
// The original mechanism 1 tracked the identifier lock over the **whole**
// value typed so far, on the theory that a value copying an identifier from
// the prompt would itself be that identifier, verbatim. That is false
// whenever the runtime *composes* the value around the copied identifier --
// `"source_path":"synthetic://textbook:8d87cddac3f4fccf:edition-2"` (the
// `synthetic://textbook:` prefix and `:edition-2` suffix are template text,
// not drawn from the prompt) or `"body":"... Namar-vanta-43 is MY-159."` (a
// distilled sentence wrapping a copied token). Because the *whole value*
// never occurs in the prompt, the old whole-content prefix check never found
// a match, and the lock never engaged at all -- the model was free to
// corrupt `8d87cddac3f4fccf` or `MY-159` mid-copy with no mechanism watching
// it.
//
// The fix operates on the *suffix run* of the value's unescaped content: the
// maximal trailing span of identifier-run bytes ([`is_identifier_run_byte`];
// `[A-Za-z0-9_.-]`), reset to empty every time a non-run byte (`:`, `/`,
// whitespace, a JSON structural byte) is emitted. Tracking just the current
// run -- rather than the whole value -- means a composed prefix like
// `synthetic://textbook:` is invisible to the mechanism (it is all run
// resets and a no-digit run that never engages), while the embedded
// `8d87cddac3f4fccf` run is tracked and locked exactly as it would be if it
// were the entire value. `:` and `/` are deliberately run *delimiters* now
// (not run bytes) -- unlike the old whole-value design, a compound
// identifier built from multiple `:`/`/`-joined pieces is now several
// separate runs, each independently locked while it is being typed. `-`,
// `_`, `.` stay inside runs (splitting `note-<32hex>` on `-`, or
// `Namar-vanta-43` at all, would shred the very identifier this mechanism
// needs to protect).
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

/// Minimum byte length a context substring must have to qualify as an
/// [`IdentifierIndex`] candidate token (Phase 51 / ADR 104 D2 suffix-run
/// redesign, rule 1).
const MIN_IDENTIFIER_TOKEN_LEN: usize = 6;

/// Split `context` into candidate identifier tokens for [`IdentifierIndex`]:
/// cut on Unicode whitespace, the JSON structural delimiters `"` `{` `}` `[`
/// `]` `,`, and -- unlike the pre-redesign version of this function -- the
/// compound-identifier joiners `:` and `/` too, since those are now run
/// *delimiters* ([`is_identifier_run_byte`]) rather than run bytes: a value
/// like `synthetic://textbook:8d87cddac3f4fccf:edition-2` is typed as
/// several independent runs, so the context tokens it can lock against must
/// be split the same way. Deliberately keeps `-`, `_`, `.` inside tokens
/// (splitting `note-<32hex>` on `-`, or `Namar-vanta-43` at all, would shred
/// the very identifier this mechanism needs to protect). Keeps tokens of
/// byte length >= [`MIN_IDENTIFIER_TOKEN_LEN`] that contain at least one
/// ASCII digit -- a plain compound word like `keph-index` must never
/// qualify just because it shares its alphabet with real identifiers (the
/// old `:/_-.`-or-digit alternative let exactly this kind of false positive
/// through).
fn identifier_tokens(context: &str) -> Vec<String> {
    fn push_if_qualifies(tokens: &mut Vec<String>, candidate: &str) {
        if candidate.len() < MIN_IDENTIFIER_TOKEN_LEN {
            return;
        }
        if candidate.chars().any(|c| c.is_ascii_digit()) {
            tokens.push(candidate.to_string());
        }
    }

    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in context.chars() {
        if c.is_whitespace() || matches!(c, '"' | '{' | '}' | '[' | ']' | ',' | ':' | '/') {
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

/// Minimum run length before the identifier hard-copy lock may engage at all
/// (Phase 51 / ADR 104 D2 suffix-run redesign, rule 2): a byte or two is not
/// enough to distinguish "coincidentally shares a prefix with some
/// identifier" from "is actually typing that identifier". Lower than the
/// pre-redesign whole-value threshold (was 4) because a *run* is already a
/// narrower, more specific signal than a whole value was.
const MIN_IDENTIFIER_RUN_LEN: usize = 3;

/// Is `byte` part of an identifier's own run body (`[A-Za-z0-9_.-]`)? `:`
/// and `/` are deliberately *not* run bytes (Phase 51 / ADR 104 D2
/// suffix-run redesign) -- they, whitespace, and every JSON structural byte
/// reset the current run to empty (see [`ConstrainedDecodeState::step_copy`]).
fn is_identifier_run_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-')
}

// ── Left-anchored run start (Phase 51 / ADR 104 D2 v7) ─────────────────────
//
// The trie lock above only *engages* once a run reaches
// [`MIN_IDENTIFIER_RUN_LEN`] and has seen a digit -- by design (rule (a):
// never lock a plain word). That leaves the run's first 1-2 bytes
// unprotected, which measurement showed is exploitable: a run can start with
// a byte that prefixes no real identifier at all (e.g. the model emits `a`
// as a run's 2nd byte when the real token continues with `8`), and nothing
// catches it because the trie lock has no opinion yet. The left anchor closes
// that gap by looking *backward* instead of forward: at the moment a run
// starts, take the value's own last few bytes (the literal text immediately
// preceding the run -- e.g. `//textbook:`) and find where *that* occurs in
// the raw prompt. If it occurs, the byte(s) that follow it there are exactly
// what a faithful copy should type next, independent of whether the run has
// accumulated enough length/digits for the trie lock to have an opinion.
//
// The anchor is deliberately narrower in scope than the trie lock: it only
// forces bytes while at least one anchored occurrence's continuation is
// itself an identifier-run byte (never a delimiter), and it gives up
// entirely once the left context doesn't occur in the prompt at all (a novel
// composition, where there is nothing to anchor against). Once established,
// it narrows monotonically as the run grows -- the same incremental
// filter-by-appended-byte shape [`SuffixCopyIndex`] already uses for the
// whole-value bias -- and is capped at [`MAX_ANCHORED_RUN_BYTES`] bytes so a
// pathological case (a left context that matches a very long stretch of the
// prompt) cannot force forever.

/// How many trailing bytes of the value's content *before* a fresh run may
/// anchor it (Phase 51 / ADR 104 D2 v7). Long enough to capture a real
/// distinguishing prefix (`//textbook:` is 11 bytes) without pulling in
/// unrelated preceding text.
const ANCHOR_LEFT_CONTEXT_LEN: usize = 12;

/// Hard cap on how many bytes of one run the left anchor may force before
/// giving up and letting the rest of the run decode unconstrained by it (the
/// trie lock, if it has engaged by then, still applies independently).
/// Guards against an anchor that, by coincidence, keeps finding *some*
/// matching occurrence indefinitely.
const MAX_ANCHORED_RUN_BYTES: usize = 64;

/// The left-anchored occurrence set for one identifier run in progress
/// (Phase 51 / ADR 104 D2 v7): every offset in the prompt where the run's
/// left context is followed by exactly the run bytes emitted so far.
/// Invariant: whenever this exists (`Some`), `positions` is non-empty *and*
/// at least one of those positions' next byte is an [`is_identifier_run_byte`]
/// continuation -- the moment that stops being true, the anchor is dropped
/// (`None`) rather than kept around inert, so "is the anchor forcing
/// anything right now" is just `.is_some()`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct AnchorRun {
    /// Byte length of the left context this anchor was computed from --
    /// fixed for the run's lifetime, needed to locate each position's
    /// continuation byte (`context[position + left_len + consumed]`).
    left_len: usize,
    /// Prompt offsets where `left context ++ run-so-far` currently matches,
    /// narrowed by one byte per call to [`advance_anchor`].
    positions: Vec<usize>,
    /// How many run bytes have been force-matched under this anchor so far
    /// (including the run's first byte, which is what establishes it) --
    /// gates [`MAX_ANCHORED_RUN_BYTES`].
    consumed: usize,
}

/// Every offset in `context` where `needle` occurs -- empty for an empty
/// `needle`. A run that starts with no left context at all (the run is the
/// very first bytes of its value, e.g. one bare array element like
/// `"textbook"`) has nothing to anchor against: matching a single byte
/// "anywhere in the prompt" is true of nearly every letter in ordinary text
/// (measured: this degenerate case forced spurious rejections on wholly
/// unrelated free text), so an empty left context leaves the anchor
/// inactive from the start, exactly like a non-empty left context that does
/// not occur in the prompt.
fn positions_of(context: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() {
        return Vec::new();
    }
    SuffixCopyIndex::scan_positions(context, needle)
}

/// Advance a left-anchored occurrence set by one run byte `sem`, where
/// `consumed` is how many run bytes were already force-matched under this
/// anchor (so the continuation byte for `position` sits at
/// `position + left_len + consumed`).
///
/// - `None`: the anchor is still actively forcing (some remaining occurrence
///   continues with an identifier-run byte) but `sem` matches none of
///   them -- `sem` is illegal.
/// - `Some(None)`: the anchor deactivates -- either no occurrence's
///   continuation is an identifier-run byte at all (every one is a
///   delimiter: the copied token already ended in the source, or there were
///   no occurrences to begin with -- novel composition), so `sem` is
///   unconstrained by the anchor for the rest of this run.
/// - `Some(Some(next))`: the anchor stays active, narrowed to the
///   occurrences whose continuation byte is exactly `sem`.
fn advance_anchor(
    context: &[u8],
    positions: &[usize],
    left_len: usize,
    consumed: usize,
    sem: u8,
) -> Option<Option<Vec<usize>>> {
    let continuation_at = |p: &usize, offset: usize| context.get(p + left_len + offset).copied();
    let has_run_continuation_at = |positions: &[usize], offset: usize| {
        positions
            .iter()
            .any(|p| continuation_at(p, offset).is_some_and(is_identifier_run_byte))
    };
    if !has_run_continuation_at(positions, consumed) {
        return Some(None);
    }
    let filtered: Vec<usize> = positions
        .iter()
        .copied()
        .filter(|p| continuation_at(p, consumed) == Some(sem))
        .collect();
    if filtered.is_empty() {
        return None;
    }
    // `sem` matched -- but look one byte further ahead before committing to
    // keeping the anchor active: if every surviving occurrence's *next*
    // continuation is itself a delimiter, the copied token completed exactly
    // on this byte, so deactivate immediately (the invariant documented on
    // [`AnchorRun`] is "`Some` implies a live forcing continuation exists
    // *right now*", not "existed as of the last byte"). Without this
    // lookahead, a run that exactly completes an anchored token would still
    // report `anchor.is_some()` for the byte immediately after -- rejecting
    // a legal close/delimiter one byte too aggressively.
    if has_run_continuation_at(&filtered, consumed + 1) {
        Some(Some(filtered))
    } else {
        Some(None)
    }
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

    /// Could the run that reached `node` still be extended by a *further
    /// run byte* toward a longer candidate? Used by
    /// [`ConstrainedDecodeState::step_copy`] to decide when a run has
    /// finished copying an identifier (the lock releases) versus is still
    /// mid-copy (rule 2 of the Phase 51 / ADR 104 D2 suffix-run redesign).
    ///
    /// Deliberately broader than the trie's own literal-leaf flag:
    /// [`identifier_tokens`] only splits candidates on a fixed delimiter set (whitespace, JSON
    /// structural bytes, `:`/`/`), so a context sentence like `"...
    /// Zorim-meru-74?"` produces the candidate `Zorim-meru-74?` -- the
    /// trailing `?` is glued on, even though `?` is not itself an
    /// [`is_identifier_run_byte`] and the run mechanism can therefore never
    /// actually walk that edge. Using literal `terminal` here left the run
    /// "engaged" forever once it reached `Zorim-meru-74` (a real, complete
    /// identifier, just not a trie leaf), permanently rejecting the value's
    /// own next byte (`?`) with no legal run-byte continuation to replace it
    /// -- a bug measured on decoder v7 (Phase 51 / ADR 104 D2, paired v7). A
    /// node with no run-byte-keyed child at all can never be extended by the
    /// run mechanism regardless of what non-run-byte children it has, so it
    /// is treated the same as a literal terminal: the lock releases.
    fn extendable(&self, node: usize) -> bool {
        self.trie.has_child_matching(node, is_identifier_run_byte)
    }

    /// Advance an incremental identifier-run cursor
    /// ([`CopyState::Active::run_prefix_node`]) by one byte, without
    /// re-walking the run's bytes already consumed to reach `node` (Phase 51
    /// perf: this is the same trie [`CopyTrie::is_prefix_of_any`] would walk
    /// from the root, but walked one edge at a time so a caller tracking its
    /// own cursor across many single-byte steps never re-walks bytes it has
    /// already consumed). `node` is `Some(`[`CopyTrie::ROOT`]`)` for an empty
    /// run, or `None` once the run has left every candidate -- mirroring
    /// `is_prefix_of_any`'s "false forever after" guarantee once broken,
    /// since `None.and_then(..)` stays `None`.
    fn advance_prefix(&self, node: Option<usize>, byte: u8) -> Option<usize> {
        node.and_then(|n| self.trie.child(n, byte))
    }
}

/// A byte-level index over one generation call's rendered context, built
/// once ([`SuffixCopyIndex::build`]) and queried once per decode step (not
/// per vocabulary token -- see the module docs' efficiency note) for the
/// soft suffix-copy bias (Phase 51 / ADR 104 D2 mechanism 2).
///
/// Matches on the **whole value emitted so far**, not a recurring short
/// suffix (Phase 51 / ADR 104 D2 bug fix, root cause 2): the original
/// fixed-window suffix design re-engaged on *any* short phrase that happened
/// to recur elsewhere in the context (e.g. "keph index" appearing once in
/// retrieved evidence), which could steer the model into replaying that
/// phrase indefinitely -- the measured `": keph index: keph index-2: ..."`
/// runaway. Matching the whole value instead is self-limiting: `content`
/// only grows one byte at a time, and a longer string can be a substring of
/// the context only if every one of its prefixes already was one too. So
/// once [`Self::continuations`] returns `None` for a given value (the value
/// has diverged from every context occurrence), it is mathematically
/// guaranteed to keep returning `None` for the rest of that same value --
/// the bias never re-engages mid-value.
///
/// # Incremental matching (Phase 51 perf)
///
/// `grug-train`'s decode loop calls [`Self::continuations`] once per decode
/// step with `content` one (occasionally a few, for a multi-byte vocabulary
/// piece) byte(s) longer than the previous call -- naively re-scanning the
/// *entire* context for the *entire* value from scratch every step is
/// O(context length x value length) per step, which dominated a measured
/// 15-20x regression on long (~300 byte) copied values. Instead, an interior
/// [`MatchCache`] remembers the match-start positions found for the
/// previous call's content: when the new `content` extends that exact
/// prefix (the overwhelmingly common case -- see the struct docs' monotonic
/// growth guarantee), the next call only has to re-check the *newly
/// appended* bytes at each already-known position (O(previous matches x new
/// bytes) instead of O(context length x value length)) rather than
/// research the whole context. Any call whose `content` is not a simple
/// extension of the cached one (a fresh value, or a value that skipped
/// calls while [`crate`]'s per-value override cap was exhausted) falls back
/// to the exact original full-context scan, so the result is identical to
/// the from-scratch algorithm in every case -- this is a pure
/// caching/complexity optimization, not a behavior change.
#[derive(Debug, Clone, Default)]
pub struct SuffixCopyIndex {
    context: Vec<u8>,
    cache: std::cell::RefCell<MatchCache>,
}

/// [`SuffixCopyIndex`]'s incremental-matching cache: the match-start
/// positions found the last time [`SuffixCopyIndex::continuations`] was
/// called, and the exact `content` bytes they correspond to (so the next
/// call can tell whether it is a simple extension of this one).
#[derive(Debug, Clone, Default)]
struct MatchCache {
    content: Vec<u8>,
    positions: Vec<usize>,
}

impl SuffixCopyIndex {
    pub fn build(context: &str) -> Self {
        Self {
            context: context.as_bytes().to_vec(),
            cache: std::cell::RefCell::new(MatchCache::default()),
        }
    }

    /// Every starting offset in `context` where `content` occurs verbatim,
    /// found by the same exhaustive (overlap-permitting) scan the original
    /// implementation used: advance one byte past each match rather than
    /// past the whole match length, so overlapping occurrences are all
    /// found.
    fn scan_positions(context: &[u8], content: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        let mut search_from = 0usize;
        while search_from + content.len() <= context.len() {
            let Some(offset) = context[search_from..]
                .windows(content.len())
                .position(|window| window == content)
            else {
                break;
            };
            let match_start = search_from + offset;
            positions.push(match_start);
            search_from = match_start + 1;
        }
        positions
    }

    /// The bytes (mapped through [`json_escape_first_byte`]) observed to
    /// follow every verbatim occurrence, anywhere in the context, of the
    /// *entire* value emitted so far (`content`, the value's *unescaped*
    /// bytes). `None` when `content` is empty or occurs nowhere in the
    /// context -- novel text (or a value that has diverged from every
    /// context occurrence), where the bias is inactive for the remainder of
    /// this value (see the struct docs for why that "off" is permanent).
    pub fn continuations(&self, content: &[u8]) -> Option<std::collections::HashSet<u8>> {
        if content.is_empty() || content.len() > self.context.len() {
            return None;
        }
        let mut cache = self.cache.borrow_mut();
        // Incremental path: `content` extends the previously-cached content
        // by >= 0 bytes -- every position that matched the shorter prefix is
        // a superset of the positions that can match `content` (a longer
        // string can only be a substring where its own prefix already was),
        // so filtering by just the newly appended bytes at each previously
        // known position is exactly equivalent to (but far cheaper than) a
        // full re-scan. Requires a non-empty cached prefix: an empty cache
        // trivially "starts with" anything, but carries no position set to
        // filter (bootstrap case below).
        let positions: Vec<usize> = if !cache.content.is_empty()
            && content.len() >= cache.content.len()
            && content[..cache.content.len()] == cache.content[..]
        {
            let prev_len = cache.content.len();
            let new_suffix = &content[prev_len..];
            cache
                .positions
                .iter()
                .copied()
                .filter(|&p| self.context.get(p + prev_len..p + content.len()) == Some(new_suffix))
                .collect()
        } else {
            Self::scan_positions(&self.context, content)
        };

        cache.content = content.to_vec();
        cache.positions = positions.clone();
        drop(cache);

        let mut next_bytes = std::collections::HashSet::new();
        for match_start in positions {
            let match_end = match_start + content.len();
            if let Some(&next) = self.context.get(match_end) {
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
        /// The value's *unescaped* content bytes typed so far -- the whole
        /// value, for [`SuffixCopyIndex`] (mechanism 2), independent of the
        /// suffix-run tracking below (mechanism 1).
        content: Vec<u8>,
        /// A pending `\` escape in the raw (possibly-escaped) decode stream.
        escaped: bool,
        /// How many bytes long the current identifier run (the maximal
        /// trailing span of [`is_identifier_run_byte`] bytes) is. Reset to
        /// `0` whenever a non-run byte is emitted (Phase 51 / ADR 104 D2
        /// suffix-run redesign).
        run_len: usize,
        /// Whether the current run has seen an ASCII digit yet -- the lock
        /// may only engage over a run that already looks identifier-shaped
        /// (rule (a) of the redesign: a plain compound word like
        /// `keph-index` must never lock). Reset alongside `run_len`.
        run_has_digit: bool,
        /// Incremental cursor into [`IdentifierIndex`]'s prefix trie
        /// ([`IdentifierIndex::advance_prefix`]) for the *current run*: the
        /// node reached by walking the run's bytes from the trie root, or
        /// `None` once the run has left every identifier candidate. Lets the
        /// identifier lock's per-byte legality check run in O(1) instead of
        /// re-walking the run on every byte of every vocabulary candidate
        /// during mask-building (Phase 51 perf). Reset to
        /// `Some(`[`CopyTrie::ROOT`]`)` alongside `run_len`.
        run_prefix_node: Option<usize>,
        /// Cached "is the lock currently engaged over this run" as of the
        /// last processed byte -- kept as a field, rather than derived on
        /// demand from `run_len`/`run_has_digit`/`run_prefix_node` (which
        /// would need [`IdentifierIndex::extendable`], i.e. a
        /// [`CopyConstraints`] reference), because
        /// [`ConstrainedDecodeState::identifier_lock_engaged`] and
        /// [`ConstrainedDecodeState::with_identifier_lock_released`] must
        /// stay `&self`-only to match `grug-train`'s existing call sites.
        /// `true` iff `run_len >= `[`MIN_IDENTIFIER_RUN_LEN`]`,
        /// `run_has_digit`, and `run_prefix_node` is a *proper* (non-
        /// terminal) prefix of >= 1 candidate -- once the run completes a
        /// candidate exactly, this goes back to `false` (rule 2: "the lock
        /// releases").
        engaged: bool,
        /// The left-anchored occurrence set for the *current* run (Phase 51
        /// / ADR 104 D2 v7), or `None` when nothing is anchoring it (never
        /// established, the left context doesn't occur in the prompt, or
        /// every remaining occurrence's continuation has stopped being an
        /// identifier-run byte). See [`AnchorRun`] and [`advance_anchor`].
        /// Reset to `None` alongside `run_len` at every run boundary, and
        /// (re-)established from the value's own preceding bytes the moment
        /// a fresh run starts.
        anchor: Option<AnchorRun>,
        /// Whether the *current* run started immediately after a `:` or `/`
        /// byte in the value's own content -- the same gate `anchor`'s
        /// establishment uses (Phase 51 / ADR 104 D2 v8: composed-value
        /// second-byte bug fix). When `true`, [`engaged`](Self::Active::engaged)
        /// drops the `run_len >= `[`MIN_IDENTIFIER_RUN_LEN`]`/`run_has_digit`
        /// requirements and may go `true` after just the run's first byte,
        /// because a run right after `:`/`/` is already a strong signal that
        /// what follows is a composed identifier (the same signal the left
        /// anchor already trusts), not merely "any 3+-byte digit-bearing
        /// span". Runs that do *not* start after `:`/`/` are unaffected --
        /// they keep the original length/digit-gated rule. Reset to `false`
        /// alongside `run_len` at every run boundary, and (re-)computed from
        /// the value's own preceding byte the moment a fresh run starts.
        run_after_colon_slash: bool,
    },
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
    /// [`CopyState`] and enforces the identifier hard-copy lock over the
    /// value's current identifier *run* (Phase 51 / ADR 104 D2 suffix-run
    /// redesign). Returns `None` exactly when the lock rejects `byte` (a
    /// non-run byte or closing quote that would truncate an incomplete run,
    /// or a run byte that would break prefix-hood) while `identifier_copy`
    /// is enabled and the run is engaged -- see [`Self::identifier_lock_engaged`].
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
                run_len: 0,
                run_has_digit: false,
                run_prefix_node: Some(CopyTrie::ROOT),
                engaged: false,
                anchor: None,
                run_after_colon_slash: false,
            });
        }

        if was_in_string && now_in_string {
            let (
                mut content,
                mut escaped,
                mut run_len,
                mut run_has_digit,
                mut run_prefix_node,
                mut engaged,
                mut anchor,
                mut run_after_colon_slash,
            ) = match &self.copy {
                CopyState::Active {
                    content,
                    escaped,
                    run_len,
                    run_has_digit,
                    run_prefix_node,
                    engaged,
                    anchor,
                    run_after_colon_slash,
                } => (
                    content.clone(),
                    *escaped,
                    *run_len,
                    *run_has_digit,
                    *run_prefix_node,
                    *engaged,
                    anchor.clone(),
                    *run_after_colon_slash,
                ),
                // The schema-trie governed the previous byte (or this is
                // the first byte after some other transition); start
                // fresh rather than assume a mismatched variant.
                CopyState::Idle => (
                    Vec::new(),
                    false,
                    0,
                    false,
                    Some(CopyTrie::ROOT),
                    false,
                    None,
                    false,
                ),
            };

            // Resolve the semantic (unescaped) byte, mirroring the D1
            // schema-trie's own escape handling. `None` means this byte only
            // opened or continued a `\` escape -- no semantic byte to feed
            // the run tracking below yet.
            let semantic = if escaped {
                let semantic = unescape_byte(byte);
                content.push(semantic);
                escaped = false;
                Some(semantic)
            } else if byte == b'\\' {
                escaped = true;
                None
            } else {
                content.push(byte);
                Some(byte)
            };

            if identifier_copy
                && let Some(sem) = semantic
            {
                let context: &[u8] = &copy_constraints.suffix.context;
                if is_identifier_run_byte(sem) {
                    // Left-anchored run start (Phase 51 / ADR 104 D2 v7): a
                    // run beginning right now is anchored on the value's own
                    // preceding bytes (`content` already has `sem` appended,
                    // so exclude it); an in-progress run's anchor is instead
                    // narrowed by this one byte. Both paths must reject `sem`
                    // the same way -- "still actively forcing, but `sem`
                    // isn't one of the forced continuations".
                    if run_len == 0 {
                        let before_run = &content[..content.len() - 1];
                        // Only anchor a run whose left context itself ends on
                        // a compound-identifier joiner (`:`/`/` -- the same
                        // bytes the suffix-run redesign treats as run
                        // *delimiters* rather than run bytes, see
                        // `is_identifier_run_byte`'s docs). This is a
                        // necessary safety gate beyond the plain "does the
                        // left context occur in the prompt" check: without
                        // it, an ordinary word boundary (a space) anchors
                        // just as readily, and short common words/phrases
                        // (`"the "`, `"see "`) coincidentally recur in
                        // unrelated context text often enough to force wrong
                        // continuations on wholly unrelated free text or on
                        // prose that diverges from the context's own
                        // phrasing around a copied identifier (a distilled
                        // sentence) -- measured while implementing this: it
                        // broke plain free-text decoding outright. A `:`/`/`
                        // boundary is a much stronger signal that what
                        // follows is a composed compound identifier rather
                        // than incidental prose.
                        //
                        // The same gate now also controls the trie lock's own
                        // early engagement (Phase 51 / ADR 104 D2 v8): a run
                        // starting right after `:`/`/` may engage the lock
                        // from its very first byte -- see where
                        // `run_after_colon_slash` feeds `engaged` below.
                        let starts_after_colon_slash =
                            matches!(before_run.last(), Some(b':') | Some(b'/'));
                        run_after_colon_slash = starts_after_colon_slash;
                        let left = if starts_after_colon_slash {
                            let left_start =
                                before_run.len().saturating_sub(ANCHOR_LEFT_CONTEXT_LEN);
                            Some(&before_run[left_start..])
                        } else {
                            None
                        };
                        anchor = match left {
                            None => None,
                            Some(left) => {
                                let left_len = left.len();
                                let occurrences = positions_of(context, left);
                                match advance_anchor(context, &occurrences, left_len, 0, sem) {
                                    None => return None,
                                    Some(None) => None,
                                    Some(Some(positions)) => Some(AnchorRun {
                                        left_len,
                                        positions,
                                        consumed: 1,
                                    }),
                                }
                            }
                        };
                    } else if let Some(active) = anchor.take() {
                        anchor = if active.consumed >= MAX_ANCHORED_RUN_BYTES {
                            None
                        } else {
                            match advance_anchor(
                                context,
                                &active.positions,
                                active.left_len,
                                active.consumed,
                                sem,
                            ) {
                                None => return None,
                                Some(None) => None,
                                Some(Some(positions)) => Some(AnchorRun {
                                    left_len: active.left_len,
                                    positions,
                                    consumed: active.consumed + 1,
                                }),
                            }
                        };
                    }

                    if engaged {
                        // Engaged implies `run_prefix_node` is a *proper*
                        // (non-terminal) prefix of >= 1 candidate, so
                        // extending it must stay on a path toward one.
                        match identifiers.advance_prefix(run_prefix_node, sem) {
                            Some(next) => run_prefix_node = Some(next),
                            None => {
                                // Not extendable means no candidate has a
                                // *run-byte* child from here -- the run has
                                // effectively completed (whether or not this
                                // node is a literal trie leaf; see
                                // `IdentifierIndex::extendable`'s docs for why
                                // "literal terminal" alone is not enough).
                                let effectively_complete = run_prefix_node
                                    .is_some_and(|node| !identifiers.extendable(node));
                                if effectively_complete {
                                    // The run already equals a complete
                                    // candidate and this byte doesn't extend
                                    // toward a *longer* one (rule 2: "extend
                                    // only if a longer candidate exists") --
                                    // fall back to untracked instead of
                                    // rejecting, since a trailing `.`/prose
                                    // byte right after a copied id must stay
                                    // legal.
                                    run_prefix_node = None;
                                } else {
                                    return None; // mid-identifier corruption
                                }
                            }
                        }
                    } else {
                        // Not (yet) engaged: track passively, never reject.
                        run_prefix_node = identifiers.advance_prefix(run_prefix_node, sem);
                    }
                    run_len += 1;
                    run_has_digit = run_has_digit || sem.is_ascii_digit();
                } else {
                    // A non-run byte while still inside the string: a run
                    // boundary. `engaged` already implies the run is
                    // non-terminal (incomplete), so this would truncate a
                    // copy in progress -- illegal. An active anchor
                    // (`anchor.is_some()`) means the same thing for the same
                    // reason: it only survives while it is still forcing an
                    // identifier-run-byte continuation, so a non-run byte
                    // here would abandon that forced copy mid-way.
                    if engaged || anchor.is_some() {
                        return None;
                    }
                    run_len = 0;
                    run_has_digit = false;
                    run_prefix_node = Some(CopyTrie::ROOT);
                    anchor = None;
                    run_after_colon_slash = false;
                }
                // A run that started right after `:`/`/` may engage from its
                // first byte on: `run_prefix_node` being a proper prefix of
                // >= 1 candidate is itself already a strong signal at that
                // position (Phase 51 / ADR 104 D2 v8 -- see
                // `run_after_colon_slash`'s docs). Every other run keeps the
                // original length/digit-gated rule unchanged.
                engaged = if run_after_colon_slash {
                    run_prefix_node.is_some_and(|node| identifiers.extendable(node))
                } else {
                    run_len >= MIN_IDENTIFIER_RUN_LEN
                        && run_has_digit
                        && run_prefix_node.is_some_and(|node| identifiers.extendable(node))
                };
            }

            return Some(CopyState::Active {
                content,
                escaped,
                run_len,
                run_has_digit,
                run_prefix_node,
                engaged,
                anchor,
                run_after_colon_slash,
            });
        }

        // Closing byte (`was_in_string && !now_in_string`), or not in a
        // string either before or after.
        if was_in_string
            && !now_in_string
            && identifier_copy
            && let CopyState::Active {
                engaged: locked,
                anchor,
                ..
            } = &self.copy
            && (*locked || anchor.is_some())
        {
            return None; // closing on an incomplete run is illegal
        }
        Some(CopyState::Idle)
    }

    /// Whether the currently-active string value's identifier hard-copy lock
    /// (D2 mechanism 1) is engaged over its current run. `grug-train`'s
    /// decode loop uses this to detect a would-be deadlock (every
    /// vocabulary token illegal under the strict lock) and retry the same
    /// decode step with [`Self::with_identifier_lock_released`].
    pub fn identifier_lock_engaged(&self) -> bool {
        matches!(&self.copy, CopyState::Active { engaged: true, .. })
    }

    /// Whether the left-anchored run-start forcing (Phase 51 / ADR 104 D2
    /// v7) is currently restricting the next byte of the active run. Folded
    /// into `grug-train`'s deadlock check alongside
    /// [`Self::identifier_lock_engaged`], since the anchor can force a byte
    /// before the trie lock has even engaged (the run's first 1-2 bytes).
    pub fn identifier_anchor_active(&self) -> bool {
        matches!(&self.copy, CopyState::Active { anchor: Some(_), .. })
    }

    /// How many run bytes the left anchor has force-matched so far in the
    /// active run (`0` when no anchor is active) -- `grug-train`'s decode
    /// loop diffs this before/after committing a token's bytes to build the
    /// `anchored_run_bytes` telemetry count in [`DecodeReport`].
    pub fn identifier_anchor_consumed(&self) -> usize {
        match &self.copy {
            CopyState::Active {
                anchor: Some(active),
                ..
            } => active.consumed,
            _ => 0,
        }
    }

    /// `self` with the active string's identifier run lock and left-anchored
    /// run-start forcing both forced open (a no-op if neither is currently
    /// active) -- the documented escape hatch for "no legal continuation at
    /// all": never deadlock generation. Scoped to the *current run* (Phase 51
    /// / ADR 104 D2 suffix-run redesign): once this run ends (a delimiter
    /// byte is next accepted), both are free to (re-)engage on whatever run
    /// follows, since run boundaries always reset tracking from scratch.
    pub fn with_identifier_lock_released(&self) -> Self {
        let mut copy = self.copy.clone();
        if let CopyState::Active {
            run_prefix_node,
            engaged,
            anchor,
            ..
        } = &mut copy
        {
            *run_prefix_node = None;
            *engaged = false;
            *anchor = None;
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
    /// (`}`/`]`): even now that the top-level arguments object rejects
    /// duplicate keys and gates `}` on required-property coverage (Phase 51
    /// / ADR 104 D1 bug fix — see [`ToolCallGrammar::cands_for`] /
    /// [`ToolCallGrammar::any_viable_required_satisfied`]), an adversarial
    /// `,` can still be *legally* substituted there whenever an optional
    /// property remains unemitted, which would desync the driver from a
    /// `hint` that intended to close at that exact byte; skipping keeps this
    /// driver's contract simple ("`hint`'s own closes always close") rather
    /// than making it reason about which closes are still substitutable.
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

    // ── Top-level arguments-object key dedup + required gating ────────────
    // (Phase 51 / ADR 104 D1 bug fix.) Reproduces the A100 checkpoint's
    // observed failure on `study_note_revise`: the top-level `arguments`
    // object had no `seen`-key tracking at all (unlike the nested
    // `Frame::Object`/`ObjectSchema` machinery just above, which already got
    // this right), so a key like `tags`/`title` could be retyped
    // indefinitely, and `}` could close the object while a `required`
    // property (`note_id`) was never emitted.

    /// The real production `study_note_revise` contract (mirrors
    /// `knowledge_workflow.rs`'s own tool-contract fixture): `note_id`,
    /// `expected_revision`, `body`, `citations` required; `title`,
    /// `challenge_id`, `language`, `tags`, `frontmatter` optional.
    fn study_note_revise_tool() -> Value {
        openai_tool(
            "study_note_revise",
            serde_json::json!({
                "title": {"type": "string"},
                "note_id": {"type": "string", "pattern": "^note-[0-9a-f]{32}$"},
                "expected_revision": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
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
            &["note_id", "expected_revision", "body", "citations"],
        )
    }

    /// (a) Once `tags` has been emitted at the top level, it must drop out of
    /// the set of keys legal to open next -- while properties that have not
    /// been emitted yet (`title`, `note_id`) must still be legal.
    #[test]
    fn top_level_key_state_excludes_already_emitted_property_and_keeps_others() {
        let tools = vec![study_note_revise_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();
        let prefix = br#"{"arguments":{"tags":["x"],""#;
        let state = grammar
            .advance(&grammar.start(), prefix)
            .expect("legal prefix up to the next key's opening quote");
        let GrammarState::Key { ref cands, .. } = state else {
            panic!("expected Key state after opening a fresh key, got {state:?}");
        };
        let names: std::collections::BTreeSet<&str> = cands
            .iter()
            .map(|&(t, p)| grammar.param_name(t, p))
            .collect();
        assert!(
            !names.contains("tags"),
            "tags was already emitted and must not be re-typable: {names:?}"
        );
        assert!(
            names.contains("title"),
            "title has not been emitted yet and must stay legal: {names:?}"
        );
        assert!(
            names.contains("note_id"),
            "note_id has not been emitted yet and must stay legal: {names:?}"
        );

        // Driving the literal duplicate `"tags"` key bytes through the
        // grammar must fail (it diverges from every remaining candidate
        // partway through, since `tags` itself was filtered out).
        assert_eq!(
            grammar.advance(&state, br#""tags""#),
            None,
            "retyping an already-emitted top-level key must be illegal"
        );
    }

    /// (b) `}` must be illegal while a `required` property (`note_id`) has
    /// not been emitted, and legal once it has.
    #[test]
    fn top_level_close_requires_every_required_property_first() {
        let tools = vec![study_note_revise_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();

        let missing_note_id = br#"{"arguments":{"body":"b","citations":[{"locator":"l","quote":"q","source_id":"s"}],"expected_revision":"deadbeef"}"#;
        let before_close = &missing_note_id[..missing_note_id.len() - 1];
        let state = grammar
            .advance(&grammar.start(), before_close)
            .expect("legal prefix with three of four required properties present");
        assert_eq!(
            grammar.step(&state, b'}'),
            None,
            "`}}` must be illegal while required `note_id` has not been emitted"
        );

        let with_note_id = br#"{"arguments":{"body":"b","citations":[{"locator":"l","quote":"q","source_id":"s"}],"expected_revision":"deadbeef","note_id":"note-00000000000000000000000000000000"}"#;
        let before_close = &with_note_id[..with_note_id.len() - 1];
        let state = grammar
            .advance(&grammar.start(), before_close)
            .expect("legal prefix with every required property present");
        assert!(
            grammar.step(&state, b'}').is_some(),
            "`}}` must be legal once every required property has been emitted"
        );
    }

    /// (c) A tool with zero declared properties must still allow the empty
    /// `{}` arguments object (required-property coverage is vacuously true
    /// over an empty parameter list).
    #[test]
    fn zero_declared_properties_still_allows_empty_arguments_object() {
        let tools = vec![openai_tool("ping", serde_json::json!({}), &[])];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();
        let s = r#"{"arguments":{},"name":"ping"}"#;
        assert_eq!(
            grammar.advance(&grammar.start(), s.as_bytes()),
            Some(GrammarState::End)
        );
    }

    /// (d) The eae9e849 adversarial-argmax driver, run against
    /// `study_note_revise`, must still land on a schema-valid call: no
    /// `MissingRequired` (the `}}`-gating fix) and no duplicate top-level key
    /// (the `seen`-tracking fix). `serde_json::Value` silently keeps only the
    /// last of a duplicate key, so the duplicate check counts raw-text
    /// occurrences of each declared property's `"<name>":` substring rather
    /// than trusting a parsed map.
    #[test]
    fn adversarial_argmax_on_study_note_revise_has_no_missing_required_or_duplicate_keys() {
        let tools = vec![study_note_revise_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();
        let hex64 = "0123456789abcdef".repeat(4);
        let hex32 = "0123456789abcdef0123456789abcdef";
        let hint = format!(
            r#"{{"arguments":{{"body":"the revised body","citations":[{{"locator":"p.9","quote":"an exact quote","source_id":"book-1"}}],"expected_revision":"{hex64}","note_id":"note-{hex32}","tags":["textbook","keph-index"],"title":"A Title"}},"name":"study_note_revise"}}"#
        );

        let out = adversarial_drive(&grammar, &hint);

        for name in [
            "title",
            "note_id",
            "expected_revision",
            "body",
            "citations",
            "challenge_id",
            "language",
            "tags",
            "frontmatter",
        ] {
            let needle = format!("\"{name}\":");
            let count = out.matches(needle.as_str()).count();
            assert!(
                count <= 1,
                "top-level key `{name}` appears {count} times in mask-forced output:\n{out}"
            );
        }

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
            !validation
                .issues
                .iter()
                .any(|issue| issue.kind == crate::tool_contract::SchemaIssueKind::MissingRequired),
            "mask-forced call is missing a required property: {:?}\n{out}",
            validation.issues
        );
        assert!(
            validation.valid,
            "mask-forced call failed schema validation: {:?}\n{out}",
            validation.issues
        );
    }

    /// Item 2: the outer envelope's fixed `,"name":"..."` / closing `}`
    /// literal segments make a repeated `"arguments"` or `"name"` key
    /// structurally unreachable -- there is no state the acceptor can be in
    /// that offers a byte path back into either literal a second time. This
    /// pins that down for both keys, and confirms the tool name stays
    /// constrained to the offered set at that position.
    #[test]
    fn envelope_forbids_repeated_arguments_or_name_and_unoffered_names() {
        let tools = vec![study_note_revise_tool()];
        let grammar = ToolCallGrammar::from_openai_tools(&tools).unwrap();

        // A second `"arguments":{...}` where `"name"` was expected.
        let repeated_arguments = br#"{"arguments":{"body":"b","citations":[{"locator":"l","quote":"q","source_id":"s"}],"expected_revision":"e","note_id":"n"},"arguments":{}}"#;
        assert_eq!(grammar.advance(&grammar.start(), repeated_arguments), None);

        // A second `"name"` after the first has already closed the call.
        let repeated_name = br#"{"arguments":{"body":"b","citations":[{"locator":"l","quote":"q","source_id":"s"}],"expected_revision":"e","note_id":"n"},"name":"study_note_revise","name":"study_note_revise"}"#;
        assert_eq!(grammar.advance(&grammar.start(), repeated_name), None);

        // A name outside the offered set is still rejected in that position.
        let unoffered_name = br#"{"arguments":{"body":"b","citations":[{"locator":"l","quote":"q","source_id":"s"}],"expected_revision":"e","note_id":"n"},"name":"delete_everything"}"#;
        assert_eq!(grammar.advance(&grammar.start(), unoffered_name), None);
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

    /// A tool with a plain `source_path` string parameter and **no**
    /// declared `pattern`/`enum` -- used by the suffix-run redesign tests
    /// below, where the *value* the runtime composes wraps a copied id in
    /// template text (`synthetic://textbook:<id>:edition-2`) that never
    /// occurs verbatim in the context.
    fn source_path_tool() -> Value {
        openai_tool(
            "cite_source_path",
            serde_json::json!({ "source_path": { "type": "string" } }),
            &["source_path"],
        )
    }

    /// A tool with a free-text `query` string parameter -- mirrors the exact
    /// JSON shape (`{"query": "..."}`) from the measured deadlock regression
    /// on decoder v7 (Phase 51 / ADR 104 D2 amendment).
    fn query_tool() -> Value {
        openai_tool(
            "answer_query",
            serde_json::json!({ "query": { "type": "string" } }),
            &["query"],
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
            .expect("the whole value present verbatim in the context must be found");
        assert!(continuations.contains(&b'\\'));
        assert!(!continuations.contains(&b'"'));
    }

    /// (c) Novel content with no occurrence anywhere in the context leaves
    /// the suffix-copy bias inactive.
    #[test]
    fn suffix_copy_index_is_inactive_for_content_with_no_context_occurrence() {
        let index = SuffixCopyIndex::build("nothing relevant appears in this context at all");
        assert!(index.continuations(b"zzzzzzzzzzzz").is_none());
    }

    /// Phase 51 / ADR 104 D2 bug fix, root cause 2: whole-value matching
    /// never re-engages once the value has diverged from every context
    /// occurrence, even when a shorter phrase inside the diverged value
    /// still recurs elsewhere in the context (the measured
    /// `": keph index: keph index-2: ..."` runaway).
    #[test]
    fn suffix_copy_index_never_reengages_once_the_value_diverges() {
        let context = "the guidance references keph index in section 2";
        let index = SuffixCopyIndex::build(context);

        // "keph index" alone recurs verbatim -- the bias is active for it.
        assert!(index.continuations(b"keph index").is_some());

        // But once the *whole value* has grown to include a continuation
        // that never occurred in the context (a colon repeating the
        // phrase), the whole value is no longer a substring anywhere, and
        // the bias must report `None` -- permanently, for any further
        // extension of this same value.
        assert!(index.continuations(b"keph index: keph index").is_none());
        assert!(index.continuations(b"keph index: keph index-2").is_none());
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

        // `textbook` alone is a plain word (no digit) and must never lock on
        // its own (rule (a)); `:` is a run delimiter (Phase 51 / ADR 104 D2
        // suffix-run redesign) so it resets tracking, and the *run* only
        // becomes identifier-shaped -- and long enough (>= 3 bytes) to
        // engage -- once three digits into the hex id after the colon.
        let prefix = "<tool_call>\n{\"arguments\":{\"source_id\":\"textbook:484";
        let state = ConstrainedDecodeState::start(&grammar)
            .advance(&grammar, &copy, prefix.as_bytes())
            .expect("a real identifier prefix must be legal");
        assert!(
            state.identifier_lock_engaged(),
            "identifier-shaped run matching the context identifier must have locked"
        );

        // `z` is not the identifier's next byte (`a`) -- illegal while locked.
        assert_eq!(state.step(&grammar, &copy, b'z'), None);

        let released = state.with_identifier_lock_released();
        assert!(!released.identifier_lock_engaged());
        assert!(released.step(&grammar, &copy, b'z').is_some());
    }

    /// Root cause 1's exact scenario: a legitimate plain-word tag (`textbook`)
    /// happens to be a prefix of an unrelated compound identifier elsewhere
    /// in the context (`textbook:e89c...:edition-2`) but must never lock --
    /// rule (a) -- and must close normally once complete, even though it is
    /// only a prefix of that longer identifier -- rule (b).
    #[test]
    fn plain_word_tag_that_prefixes_an_unrelated_identifier_stays_closable() {
        let context = concat!(
            "<tool_response>\n",
            "{\"source_id\":\"textbook:e89c5074c30874cc:edition-2\",\"locator\":\"p.1\"}\n",
            "</tool_response>"
        );
        let tools = vec![study_note_append_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        // Drive up to (and including) the closing quote of the first tag,
        // "textbook" -- this must be legal. `:` is a run delimiter (Phase 51
        // / ADR 104 D2 suffix-run redesign), so "textbook" is not even a
        // prefix of the compound identifier's own run in the new scheme; it
        // is also digit-free, so rule (a) would keep it from ever locking
        // either way.
        let up_to_tag_close = "<tool_call>\n{\"arguments\":{\"body\":\"b\",\"citations\":[{\"locator\":\"l\",\"quote\":\"q\",\"source_id\":\"s\"}],\"tags\":[\"textbook";
        let state = advance_constrained(&grammar, &copy, up_to_tag_close)
            .expect("typing the plain word tag must never lock");
        assert!(
            !state.identifier_lock_engaged(),
            "a plain alphabetic word must never engage the identifier lock"
        );
        let closed = state
            .step(&grammar, &copy, b'"')
            .expect("closing the complete tag must be legal even though it prefixes an unrelated identifier");

        // The rest of the call (second tag, title) must decode normally,
        // proving the array was never derailed by a stuck lock.
        let rest = ",\"keph-index\"],\"title\":\"t\"},\"name\":\"study_note_append\"}\n</tool_call>";
        let end = closed
            .advance(&grammar, &copy, rest.as_bytes())
            .expect("the rest of the call must decode to completion");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    // ── Suffix-run redesign (Phase 51 / ADR 104 D2, decoder 957589e1) ──────
    //
    // The mechanism above (whole-value hard lock) never engages once the
    // *value* the runtime emits is composed around a copied identifier
    // rather than being that identifier verbatim -- a fixed URI prefix, or a
    // distilled sentence. These tests target that residual gap directly:
    // the identifier lock now tracks the value's current identifier *run*
    // (a maximal `[A-Za-z0-9_.-]` span, reset by `:`/`/`/whitespace/JSON
    // delimiters) rather than the whole value.

    /// (a) The measured failure: `source_path` has no schema pattern, and
    /// the *whole* emitted value (`synthetic://textbook:<id>:edition-2`)
    /// never occurs verbatim in the context -- only the bare id does. The
    /// pre-redesign whole-value lock could never engage here at all; the
    /// run-suffix lock must still catch a corrupted copy of the embedded id.
    #[test]
    fn identifier_hard_copy_locks_an_embedded_id_inside_a_runtime_composed_value() {
        let id = "8d87cddac3f4fccf";
        let context = format!("<tool_response>\n{{\"id\":\"{id}\"}}\n</tool_response>");
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        // Sanity: the whole composed value never occurs in the context, so
        // this is not a D1 schema-pattern/enum field either.
        assert_eq!(copy.table.entry(0, 0).unwrap().0, ValueConstraint::Free);

        let correct = format!(
            "<tool_call>\n{{\"arguments\":{{\"source_path\":\"synthetic://textbook:{id}:edition-2\"}},\"name\":\"cite_source_path\"}}\n</tool_call>"
        );
        let end = advance_constrained(&grammar, &copy, &correct).expect(
            "a value composed around a verbatim context id must be a legal decode path",
        );
        assert!(grammar.accepts_eos(&end.frontier));

        // The measured corruption: the argmax wants to insert an extra
        // digit partway into the run (dropping a trailing byte elsewhere to
        // keep the same overall length) -- reproduced here as the point of
        // divergence itself, which must have no legal continuation.
        let corrupted =
            "<tool_call>\n{\"arguments\":{\"source_path\":\"synthetic://textbook:8d87c9";
        assert_eq!(
            advance_constrained(&grammar, &copy, corrupted),
            None,
            "a corrupted copy of an id embedded in a composed value must have no legal continuation"
        );
    }

    /// (b) The measured failure: `body` is a free-text, distilled sentence
    /// (never verbatim in the context) that embeds a verbatim numeric token
    /// from the retrieved evidence (`MY-159`). The lock must engage on that
    /// embedded run, reject closing it one digit early, and release once it
    /// completes so the sentence-ending period stays legal.
    #[test]
    fn identifier_hard_copy_locks_a_run_embedded_in_a_distilled_sentence() {
        let context = "<tool_response>\nNamar-vanta-43 was catalogued as MY-159 in the ledger.\n</tool_response>";
        let tools = vec![note_body_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        let correct = "<tool_call>\n{\"arguments\":{\"body\":\"the note says Namar-vanta-43 is MY-159.\"},\"name\":\"note_body\"}\n</tool_call>";
        let end = advance_constrained(&grammar, &copy, correct)
            .expect("a distilled sentence embedding verbatim context tokens must decode");
        assert!(grammar.accepts_eos(&end.frontier));

        // The measured failure: the argmax wants to close the number one
        // digit early (`.` right after `MY-15` instead of the real `9`).
        let truncated =
            "<tool_call>\n{\"arguments\":{\"body\":\"the note says Namar-vanta-43 is MY-15";
        let state = advance_constrained(&grammar, &copy, truncated)
            .expect("a real prefix of the context token must be legal");
        assert!(
            state.identifier_lock_engaged(),
            "MY-15 must still be locked -- MY-159 is not yet complete"
        );
        assert_eq!(
            state.step(&grammar, &copy, b'.'),
            None,
            "closing the number one digit early must be illegal"
        );

        // The real next digit is legal, and completing the number releases
        // the lock so the sentence-ending period becomes legal again.
        let completed = state
            .step(&grammar, &copy, b'9')
            .expect("the real next digit must be legal");
        assert!(
            !completed.identifier_lock_engaged(),
            "the run now equals the complete context token; the lock must have released"
        );
        assert!(
            completed.step(&grammar, &copy, b'.').is_some(),
            "a sentence-ending period right after a completed id must stay legal"
        );
    }

    /// (c) Rule 1 drops the old ":/_-."-or-digit alternative: a compound
    /// word with no digit at all (`keph-index`) must never engage the lock,
    /// even when it is a genuine prefix of a real, digit-bearing context
    /// identifier (`keph-index-9000`).
    #[test]
    fn tag_with_no_digit_never_engages_the_identifier_lock() {
        let context = "<tool_response>\nsee also keph-index-9000 for details\n</tool_response>";
        let tools = vec![note_body_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        let call = "<tool_call>\n{\"arguments\":{\"body\":\"see keph-index";
        let state = advance_constrained(&grammar, &copy, call)
            .expect("typing a plain, digit-free compound word must never be blocked");
        assert!(
            !state.identifier_lock_engaged(),
            "keph-index has no digit and must never engage the lock, even as a real prefix"
        );
        // Closing here (a different word than the context's
        // `keph-index-9000`) must still be legal.
        assert!(state.step(&grammar, &copy, b'"').is_some());
    }

    /// (d) Once a run exactly completes a context identifier token
    /// (`edition-2`), the lock releases: the value's own next delimiter
    /// (`:`) is legal, and so is closing the value immediately.
    #[test]
    fn identifier_run_completes_and_permits_the_values_own_delimiter_afterward() {
        let context = "<tool_response>\nsee edition-2 of the volume\n</tool_response>";
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        let up_to_run = "<tool_call>\n{\"arguments\":{\"source_path\":\"synthetic://textbook:edition-2";
        let state = advance_constrained(&grammar, &copy, up_to_run)
            .expect("a real context token must be a legal decode path");
        assert!(
            !state.identifier_lock_engaged(),
            "edition-2 exactly completes the context token; the lock must have released"
        );

        let after_colon = state
            .step(&grammar, &copy, b':')
            .expect("`:` right after a completed identifier run must be legal");
        assert!(
            state.step(&grammar, &copy, b'"').is_some(),
            "closing the value immediately after a completed run must also be legal"
        );

        let end = after_colon
            .advance(
                &grammar,
                &copy,
                b"more\"},\"name\":\"cite_source_path\"}\n</tool_call>",
            )
            .expect("the rest of the call must decode to completion");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// (v8 bug fix) The exact measured regression on the A100 (paired
    /// decoder v7, commit a72b2837): `revise_from_new_edition` collapsed
    /// 80->26/100 because the context sentence containing the copied
    /// identifier ends in `?` with no separating whitespace
    /// (`"...Zorim-meru-74?"`). [`identifier_tokens`] does not split on `?`,
    /// so the *candidate* it derives is `Zorim-meru-74?`, not
    /// `Zorim-meru-74`. Once the run exactly reaches `Zorim-meru-74`, no
    /// further [`is_identifier_run_byte`] byte can ever continue toward that
    /// candidate (`?` is not a run byte) -- the run is, for the lock's own
    /// purposes, complete. It must release for **any** next byte, including
    /// the value's own literal `?`, rather than staying engaged forever with
    /// no legal run-byte continuation (the literal-`terminal`-only check
    /// missed exactly this case, deadlocking the strict mask; a `\` escape
    /// opener then falsely appeared "legal" because it bypasses the run
    /// lock's own semantic-byte check, hiding the deadlock from
    /// `grug-train`'s release detector and producing invalid `\?` JSON).
    #[test]
    fn identifier_run_releases_when_only_a_trailing_non_run_byte_extends_the_candidate() {
        let context = "What is the keph index of Zorim-meru-74? Please compute it.";
        let tools = vec![query_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        let up_to_run =
            "<tool_call>\n{\"arguments\":{\"query\":\"What is the keph index of Zorim-meru-74";
        let state = advance_constrained(&grammar, &copy, up_to_run)
            .expect("typing the identifier itself must be a legal decode path");
        assert!(
            !state.identifier_lock_engaged(),
            "Zorim-meru-74 has no run-byte continuation toward the `?`-suffixed \
             candidate, so the lock must have released"
        );

        // The value's own real next byte, bare `?`, must be directly legal
        // -- no escape required, and no deadlock.
        let with_question_mark = state
            .step(&grammar, &copy, b'?')
            .expect("a released lock must accept the value's own literal `?`");

        let end = with_question_mark
            .advance(
                &grammar,
                &copy,
                b"\"},\"name\":\"answer_query\"}\n</tool_call>",
            )
            .expect("the rest of the call must decode to completion");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// (v8 bug fix regression guard) A genuinely longer real candidate
    /// (`Zorim-meru-742`, not just trailing punctuation) must still lock and
    /// force its continuation -- the fix above (releasing when only a
    /// non-run byte extends the trie candidate) must not also let a run that
    /// is a proper prefix of a *real*, longer identifier go free early.
    #[test]
    fn identifier_run_still_locks_toward_a_longer_real_candidate() {
        let context = "context mentions Zorim-meru-742 exactly once, nothing else";
        let tools = vec![query_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        let up_to_run = "<tool_call>\n{\"arguments\":{\"query\":\"Zorim-meru-74";
        let state = advance_constrained(&grammar, &copy, up_to_run)
            .expect("a prefix of the real, longer candidate must be legal");
        assert!(
            state.identifier_lock_engaged(),
            "Zorim-meru-74 is a proper prefix of the longer Zorim-meru-742 \
             candidate and must stay locked"
        );

        assert!(
            state.step(&grammar, &copy, b'?').is_none(),
            "a byte that doesn't continue the longer candidate must be \
             rejected while the lock is engaged"
        );
        let continued = state
            .step(&grammar, &copy, b'2')
            .expect("`2` completes the longer candidate and must be legal");
        assert!(
            !continued.identifier_lock_engaged(),
            "Zorim-meru-742 is now a complete candidate with no further \
             run-byte continuation; the lock must release"
        );
    }

    /// (v8 bug fix) Inside a JSON string, a `\` may only be followed by one
    /// of JSON's own escape bytes -- this is a grammar-level restriction
    /// ([`GrammarState`]/[`ValueParser`]'s `Str`/`AnyKey` leaves previously
    /// accepted *any* byte after `\`), independent of the identifier lock.
    /// `\?` is not legal JSON and must have no decode path -- this is what
    /// let the identifier-lock deadlock above actually reach the model as
    /// invalid output (`\?` in place of the correct `?`) instead of being
    /// caught structurally.
    #[test]
    fn backslash_inside_a_string_only_accepts_a_legal_json_escape_byte() {
        let tools = vec![note_body_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, "nothing identifier-shaped here");
        let prefix = "<tool_call>\n{\"arguments\":{\"body\":\"abc\\";

        for good in *b"\"\\/bfnrt" {
            let state = advance_constrained(&grammar, &copy, prefix)
                .expect("the escape opener itself must be legal");
            assert!(
                state.step(&grammar, &copy, good).is_some(),
                "{good} must be a legal JSON escape byte"
            );
        }

        // `\u` opens a 4-hex-digit unicode escape; exactly four hex digits
        // must follow before the string can close.
        let state = advance_constrained(&grammar, &copy, prefix).unwrap();
        let after_u = state
            .step(&grammar, &copy, b'u')
            .expect("`u` must open a unicode escape");
        let mut cur = after_u;
        for hex in *b"0041" {
            cur = cur
                .step(&grammar, &copy, hex)
                .expect("hex digits must be legal inside a \\u escape");
        }
        assert!(
            cur.step(&grammar, &copy, b'"').is_some(),
            "the string may close once all four hex digits are consumed"
        );
        let non_hex = state
            .step(&grammar, &copy, b'u')
            .unwrap()
            .step(&grammar, &copy, b'g');
        assert!(non_hex.is_none(), "a non-hex byte inside \\u must be rejected");

        // The measured regression: `\?` is not valid JSON.
        let state = advance_constrained(&grammar, &copy, prefix).unwrap();
        assert!(
            state.step(&grammar, &copy, b'?').is_none(),
            "\\? is not a legal JSON escape and must be rejected"
        );
    }

    /// (e) The documented escape hatch (a would-be deadlock releases the
    /// lock) is now scoped to the *current run only* -- once that run ends
    /// (a delimiter is accepted) and a later, independent run begins in the
    /// same value, the lock must be free to engage again on it.
    #[test]
    fn identifier_lock_release_is_scoped_to_the_current_run_only() {
        let context = "<tool_response>\nsee foo-9999 and also bar-1234 here\n</tool_response>";
        let tools = vec![note_body_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        let up_to_first_run = "<tool_call>\n{\"arguments\":{\"body\":\"foo-99";
        let state = advance_constrained(&grammar, &copy, up_to_first_run)
            .expect("a real context-token prefix must be legal");
        assert!(
            state.identifier_lock_engaged(),
            "foo-99 must be locked -- foo-9999 is not yet complete"
        );

        // Simulate the documented escape hatch: the argmax would leave
        // every legal candidate for the rest of this run.
        let released = state.with_identifier_lock_released();
        assert!(!released.identifier_lock_engaged());

        // Diverge from the context id entirely for the rest of *this* run,
        // then start a brand new run (`bar-12`, a real prefix of the
        // context's `bar-1234`) after a word-boundary space.
        let diverged = released
            .advance(&grammar, &copy, b"zzz bar-12")
            .expect("a released run must accept an arbitrary continuation");
        assert!(
            diverged.identifier_lock_engaged(),
            "a later run in the same value must still be able to lock, independent of an earlier run's release"
        );
    }

    // ── Left-anchored run start (Phase 51 / ADR 104 D2 v7) ──────────────────

    /// (B) Regression for measured failure shape (B) in the v7 task report:
    /// `synthetic://textbook:b851eb11b1b27ab9:edition-2` -- the argmax wants
    /// `a` as the run's *2nd* byte (the real id starts `b8...`). At that
    /// point `run_len` is only 1; the left anchor on `textbook:` (found
    /// verbatim in the prompt, immediately followed by the real id) hard-
    /// restricts the run's first bytes -- and, since v8, so does the trie
    /// lock itself: the run starts right after ':' and `b` is a real prefix
    /// of the context id, so the lock engages from this very first byte too.
    /// Both mechanisms agreeing here is expected; the case where the trie
    /// lock *alone* has to carry this (the anchor cannot engage, because the
    /// id occurs in the prompt only in prose form) is covered separately by
    /// `first_run_byte_after_colon_engages_the_trie_lock_when_the_anchor_cannot`.
    #[test]
    fn anchor_hard_restricts_the_runs_first_bytes_before_the_trie_lock_can_engage() {
        let id = "b851eb11b1b27ab9";
        let context = format!(
            "<tool_response>\n{{\"source_path\":\"synthetic://textbook:{id}:edition-2\"}}\n</tool_response>"
        );
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let up_to_first_byte =
            "<tool_call>\n{\"arguments\":{\"source_path\":\"synthetic://textbook:b";
        let state = advance_constrained(&grammar, &copy, up_to_first_byte)
            .expect("the real first byte of the context id must be legal");
        assert!(
            state.identifier_lock_engaged(),
            "since v8, the trie lock also engages from the run's first byte after ':' \
             when that byte is a real candidate prefix"
        );
        assert!(
            state.identifier_anchor_active(),
            "the left anchor on `textbook:` must be forcing the run's first byte"
        );

        assert_eq!(
            state.step(&grammar, &copy, b'a'),
            None,
            "a wrong 2nd byte must have no legal continuation before the trie lock can even engage"
        );

        let rest = format!(
            "{}:edition-2\"}},\"name\":\"cite_source_path\"}}\n</tool_call>",
            &id[1..]
        );
        let end = state
            .advance(&grammar, &copy, rest.as_bytes())
            .expect("typing the real identifier to completion must stay legal");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// (C) A composed prefix (`://textbook:`) that never occurs anywhere in
    /// the prompt (a wholly novel composition) must leave the anchor
    /// inactive -- decoding proceeds exactly as unconstrained free text
    /// would, with no forced continuation at all.
    #[test]
    fn anchor_stays_inactive_for_a_novel_composition_never_seen_in_the_prompt() {
        let context = "<tool_response>\nno identifiers appear anywhere in here\n</tool_response>";
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        let up_to_prefix = "<tool_call>\n{\"arguments\":{\"source_path\":\"synthetic://textbook:";
        let state = advance_constrained(&grammar, &copy, up_to_prefix)
            .expect("the fixed template prefix must always be legal");

        let mid = state
            .advance(&grammar, &copy, b"f")
            .expect("typing a wholly novel value must stay legal");
        assert!(
            !mid.identifier_anchor_active(),
            "`://textbook:` never occurs in this context -- the anchor must never engage"
        );

        let end = mid
            .advance(
                &grammar,
                &copy,
                b"oo\"},\"name\":\"cite_source_path\"}\n</tool_call>",
            )
            .expect("a novel composition must decode exactly as unconstrained free text would");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// (D) A plain, digit-free word (`textbook`) anchored on a `:`-joined
    /// left context that genuinely occurs in the prompt must be forced
    /// *within* the token (a wrong byte mid-word is illegal), but the anchor
    /// must deactivate the instant the token completes -- the value is free
    /// to continue with anything afterward, even though the context's own
    /// occurrence is followed by a delimiter (`,`) the value never types.
    #[test]
    fn anchored_plain_word_is_forced_within_the_token_but_not_beyond_it() {
        let context = "<tool_response>\nsee ref:textbook, page 3 for details\n</tool_response>";
        let tools = vec![note_body_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, context);

        let up_to_ref_colon = "<tool_call>\n{\"arguments\":{\"body\":\"ref:";
        let state = advance_constrained(&grammar, &copy, up_to_ref_colon)
            .expect("a ':'-anchored prefix must be legal");
        assert!(
            !state.identifier_anchor_active(),
            "the anchor engages on the run's first byte, not before it"
        );

        let mid = state
            .advance(&grammar, &copy, b"textboo")
            .expect("typing a real anchored continuation must stay legal");
        assert!(
            mid.identifier_anchor_active(),
            "'textboo' must still be anchored -- 'textbook' is not yet complete"
        );

        // The context's real 8th letter is `k`; a divergent byte must be
        // illegal while the anchor is still forcing.
        assert_eq!(mid.step(&grammar, &copy, b'z'), None);

        let completed = mid
            .step(&grammar, &copy, b'k')
            .expect("the real 8th letter must be legal");
        assert!(
            !completed.identifier_anchor_active(),
            "'textbook' exactly completes the anchored token in the context; the anchor must deactivate"
        );

        // Not forced beyond the token: the context's own occurrence
        // continues with `,`, but the value instead continues with
        // unrelated free prose, and this must still be legal.
        let end = completed
            .advance(&grammar, &copy, b" is relevant.\"},\"name\":\"note_body\"}\n</tool_call>")
            .expect(
                "the rest of the call must decode to completion, unconstrained by the completed anchor",
            );
        assert!(grammar.accepts_eos(&end.frontier));
    }

    // ── First-run-byte trie-lock engagement after ':'/'/' (Phase 51 / ADR 104
    //    D2 v8) ───────────────────────────────────────────────────────────
    //
    // Measured on the A100 (paired v8): the remaining `acquire_textbook`
    // failures are all a wrong *2nd* byte of an identifier run that starts
    // right after ':' in a composed value, *before* the trie lock could
    // engage (it needed `MIN_IDENTIFIER_RUN_LEN` == 3 bytes) -- and the left
    // anchor (v7) didn't help either, because the id occurs in the prompt
    // only in prose form (`Manual <id>`), never directly preceded by
    // `textbook:`. These tests cover the fix: a run starting right after
    // ':'/'/' may engage the trie lock from its very first byte, as long as
    // that byte is itself a prefix of >= 1 context identifier candidate.

    /// (a) The exact measured failure shape: `b` is a real prefix of the
    /// context id, so the trie lock must engage immediately (the left anchor
    /// cannot help -- `textbook:` never occurs verbatim in this context) and
    /// reject the wrong 2nd byte the model wants (`a` instead of the real
    /// `8`).
    #[test]
    fn first_run_byte_after_colon_engages_the_trie_lock_when_the_anchor_cannot() {
        let id = "b851eb11b1b27ab9";
        let context =
            format!("<tool_response>\nSynthetic Keph Manual {id} edition-2\n</tool_response>");
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let up_to_first_byte =
            "<tool_call>\n{\"arguments\":{\"source_path\":\"synthetic://textbook:b";
        let state = advance_constrained(&grammar, &copy, up_to_first_byte)
            .expect("the real first byte of the context id must be legal");
        assert_eq!(
            state.identifier_anchor_consumed(),
            0,
            "the left anchor must not have engaged -- `textbook:` never occurs verbatim in this context"
        );
        assert!(
            state.identifier_lock_engaged(),
            "the trie lock must engage from the run's first byte when the run starts right after ':'"
        );

        assert_eq!(
            state.step(&grammar, &copy, b'a'),
            None,
            "a wrong 2nd byte must be rejected even though run_len is only 1"
        );

        let rest = format!(
            "{}:edition-2\"}},\"name\":\"cite_source_path\"}}\n</tool_call>",
            &id[1..]
        );
        let end = state
            .advance(&grammar, &copy, rest.as_bytes())
            .expect("typing the real identifier (and the unrelated 'edition-2' run after it) to completion must stay legal");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// (b) A second ':'-anchored run in the *same* value (`edition-2`) that
    /// does have a matching context candidate must decode to completion --
    /// the v8 engagement rule is not one-shot-per-value, it re-applies to
    /// every run that starts right after ':'/'/'.
    #[test]
    fn first_run_byte_after_colon_engages_and_completes_a_second_run_in_the_same_value() {
        let id = "b851eb11b1b27ab9";
        let context =
            format!("<tool_response>\nSynthetic Keph Manual {id} edition-2\n</tool_response>");
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let up_to_edition_first_byte = format!(
            "<tool_call>\n{{\"arguments\":{{\"source_path\":\"synthetic://textbook:{id}:e"
        );
        let state = advance_constrained(&grammar, &copy, &up_to_edition_first_byte)
            .expect("the real first byte of the 'edition-2' run must be legal");
        assert!(
            state.identifier_lock_engaged(),
            "'e' is a real prefix of the context's 'edition-2' candidate -- the trie lock must engage"
        );

        let end = state
            .advance(
                &grammar,
                &copy,
                b"dition-2\"},\"name\":\"cite_source_path\"}\n</tool_call>",
            )
            .expect("typing the real 'edition-2' suffix to completion must stay legal");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// (c) A run starting right after ':' whose first byte prefixes *no*
    /// context candidate at all must stay fully unconstrained -- there is
    /// nothing for the trie lock to engage on, so the run decodes exactly as
    /// ordinary free text would.
    #[test]
    fn run_after_colon_with_no_matching_candidate_prefix_stays_unconstrained() {
        let id = "b851eb11b1b27ab9";
        let context =
            format!("<tool_response>\nSynthetic Keph Manual {id} edition-2\n</tool_response>");
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let up_to_colon = "<tool_call>\n{\"arguments\":{\"source_path\":\"synthetic://textbook:";
        let state = advance_constrained(&grammar, &copy, up_to_colon)
            .expect("the fixed template prefix must always be legal");

        let after_z = state
            .advance(&grammar, &copy, b"zzz-unrelated")
            .expect("a run whose first byte prefixes no candidate must stay unconstrained");
        assert!(
            !after_z.identifier_lock_engaged(),
            "'z' prefixes no candidate identifier -- the trie lock must never engage"
        );
        assert!(!after_z.identifier_anchor_active());

        let end = after_z
            .advance(
                &grammar,
                &copy,
                b"\"},\"name\":\"cite_source_path\"}\n</tool_call>",
            )
            .expect("an unconstrained run must decode to completion like ordinary free text");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// (d) A novel id sharing only its first byte with a real context
    /// candidate engages the lock (byte 1 is a real prefix) but then has no
    /// legal continuation at its 2nd byte -- the documented, acceptable
    /// deadlock outcome for a genuinely novel composition, not a bug.
    /// `grug-train`'s decode loop detects exactly this ("no legal byte at
    /// all") and retries via [`ConstrainedDecodeState::with_identifier_lock_released`],
    /// after which the rest of the run decodes unconstrained.
    #[test]
    fn novel_id_after_colon_diverges_and_releases_via_the_deadlock_escape_hatch() {
        let id = "b851eb11b1b27ab9";
        let context =
            format!("<tool_response>\nSynthetic Keph Manual {id} edition-2\n</tool_response>");
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let up_to_first_byte =
            "<tool_call>\n{\"arguments\":{\"source_path\":\"synthetic://textbook:b";
        let state = advance_constrained(&grammar, &copy, up_to_first_byte)
            .expect("the real first byte of the context id must be legal");
        assert!(state.identifier_lock_engaged());

        assert_eq!(
            state.step(&grammar, &copy, b'q'),
            None,
            "a novel id sharing only the first byte with a real candidate must deadlock at its 2nd byte"
        );

        let released = state.with_identifier_lock_released();
        assert!(!released.identifier_lock_engaged());

        let end = released
            .advance(
                &grammar,
                &copy,
                b"qqq-novel-99\"},\"name\":\"cite_source_path\"}\n</tool_call>",
            )
            .expect("after release, an arbitrary novel continuation must decode to completion");
        assert!(grammar.accepts_eos(&end.frontier));
    }

    /// (e) The same digit-bearing candidate id, typed as the *entire* value
    /// with nothing preceding it (no ':'/'/' immediately before the run):
    /// `run_after_colon_slash` must be `false` here, so the original
    /// length/digit-gated rule still applies unchanged -- a single byte
    /// (`run_len == 1`) must not engage the lock.
    #[test]
    fn run_not_after_colon_or_slash_still_needs_min_len_and_a_digit_to_engage() {
        let id = "b851eb11b1b27ab9";
        let context =
            format!("<tool_response>\nSynthetic Keph Manual {id} edition-2\n</tool_response>");
        let tools = vec![source_path_tool()];
        let grammar = RuntimeCallGrammar::from_openai_tools(&tools).unwrap();
        let copy = copy_constraints(&grammar, &context);

        let up_to_first_byte = "<tool_call>\n{\"arguments\":{\"source_path\":\"b";
        let state = advance_constrained(&grammar, &copy, up_to_first_byte)
            .expect("the real first byte of the context id must be legal");
        assert!(
            !state.identifier_lock_engaged(),
            "run_len is only 1 and the run starts at the value's own beginning (no ':'/'/' \
             before it) -- the original length/digit-gated rule must still apply"
        );

        let end = state
            .advance(
                &grammar,
                &copy,
                format!(
                    "{}\"}},\"name\":\"cite_source_path\"}}\n</tool_call>",
                    &id[1..]
                )
                .as_bytes(),
            )
            .expect("typing the real identifier to completion must stay legal");
        assert!(grammar.accepts_eos(&end.frontier));
    }
}
