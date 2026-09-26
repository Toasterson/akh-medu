//! Data-driven event route filters (Phase 53g PR 6, ADR 113 decision 10).
//!
//! An [`EventRoute`](crate::config::EventRoute) may carry `when`, a predicate
//! over the event's `data`. The route delivers only events whose data
//! satisfies it; a route without `when` delivers every event of its source and
//! type, exactly as before. The filters exist so routing policy stays in
//! configuration (rendered by the deployment, e.g. the Roman `full_akh.py`),
//! never in Rust, and never on prose: they compare machine fields (phases,
//! reason codes, counts) that a JSON pointer addresses.
//!
//! The language, in its JSON form:
//!
//! | Form | True when |
//! |---|---|
//! | `{"eq": {"pointer": P, "value": V}}` | `P` resolves to a value JSON-equal to `V` |
//! | `{"in": {"pointer": P, "values": [V…]}}` | `P` resolves to a value JSON-equal to one of the `V` |
//! | `{"prefix": {"pointer": P, "value": S}}` | `P` resolves to a string starting with `S` |
//! | `{"exists": {"pointer": P}}` | `P` resolves to a non-null value |
//! | `{"not": F}` | `F` is false |
//! | `{"all": [F…]}` | every `F` is true |
//! | `{"any": [F…]}` | at least one `F` is true |
//!
//! Pointers are RFC 6901 JSON pointers into the event `data` (`""` is the
//! whole document). A pointer that does not resolve makes `eq`, `in`,
//! `prefix` and `exists` false, so `not` of them is true: a missing field is
//! "not equal", never an error. JSON equality is exact: `0` and `0.0` differ,
//! as do `1` and `true`.
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Nesting depth bound of one filter.
pub const MAX_DEPTH: usize = 16;
/// Node count bound of one filter.
pub const MAX_NODES: usize = 256;
/// Bound on the members of one `all`/`any` group or `in` set.
pub const MAX_MEMBERS: usize = 256;
/// Byte bound on a pointer or a `prefix` value.
pub const MAX_TEXT: usize = 512;

/// An RFC 6901 JSON pointer. Checked by [`RouteFilter::validate`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct JsonPointer(String);

impl JsonPointer {
    pub fn new(pointer: impl Into<String>) -> Self {
        Self(pointer.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
    /// Whether the text is a well-formed pointer: empty, or `/`-prefixed
    /// with every `~` escaped as `~0` or `~1`.
    pub fn is_well_formed(&self) -> bool {
        let text = self.0.as_str();
        if text.len() > MAX_TEXT || text.chars().any(char::is_control) {
            return false;
        }
        if !text.is_empty() && !text.starts_with('/') {
            return false;
        }
        let bytes = text.as_bytes();
        bytes
            .iter()
            .enumerate()
            .filter(|(_, b)| **b == b'~')
            .all(|(i, _)| matches!(bytes.get(i + 1), Some(b'0' | b'1')))
    }
    /// The value the pointer addresses in `data`, if any.
    pub fn resolve<'a>(&self, data: &'a Value) -> Option<&'a Value> {
        data.pointer(&self.0)
    }
}

/// A predicate over an event's `data`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum RouteFilter {
    Eq {
        pointer: JsonPointer,
        value: Value,
    },
    In {
        pointer: JsonPointer,
        values: Vec<Value>,
    },
    Prefix {
        pointer: JsonPointer,
        value: String,
    },
    Exists {
        pointer: JsonPointer,
    },
    Not(Box<RouteFilter>),
    All(Vec<RouteFilter>),
    Any(Vec<RouteFilter>),
}

/// A filter that cannot be evaluated as written. `at` is the path of the
/// offending node, e.g. `when.all[1].not.in`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, miette::Diagnostic)]
pub enum RouteFilterError {
    #[error("{at}: `{pointer}` is not a JSON pointer")]
    #[diagnostic(
        code(akh::events::route_filter::pointer),
        help(
            "Use an RFC 6901 pointer into the event data: empty for the whole document, otherwise \
             `/`-separated, e.g. `/current/current_phase`; escape `~` as `~0` and `/` as `~1`; \
             at most 512 bytes."
        )
    )]
    Pointer { at: String, pointer: String },
    #[error("{at}: `{operator}` has no members")]
    #[diagnostic(
        code(akh::events::route_filter::empty_group),
        help(
            "Give `all`/`any` at least one filter. An empty `all` would match everything and an \
             empty `any` nothing, which is never what a route means; drop `when` instead."
        )
    )]
    EmptyGroup { at: String, operator: &'static str },
    #[error("{at}: `in` lists no values")]
    #[diagnostic(
        code(akh::events::route_filter::empty_set),
        help("List at least one value, or use `exists`/`not` if presence is what matters.")
    )]
    EmptySet { at: String },
    #[error("{at}: {count} members exceed the bound of {limit}")]
    #[diagnostic(
        code(akh::events::route_filter::members),
        help("Split the group, or route on a coarser machine field.")
    )]
    Members {
        at: String,
        count: usize,
        limit: usize,
    },
    #[error("{at}: the `prefix` value is empty or longer than 512 bytes")]
    #[diagnostic(
        code(akh::events::route_filter::prefix),
        help("An empty prefix matches every string; use `exists` for that.")
    )]
    Prefix { at: String },
    #[error("{at}: nesting exceeds {limit} levels")]
    #[diagnostic(
        code(akh::events::route_filter::depth),
        help("Flatten nested `all`/`any` groups; they are associative.")
    )]
    Depth { at: String, limit: usize },
    #[error("the filter has more than {limit} nodes")]
    #[diagnostic(
        code(akh::events::route_filter::size),
        help("Split the route in two, or route on a coarser machine field.")
    )]
    Size { limit: usize },
}

impl RouteFilter {
    /// Static checks. `at` names the filter's root in diagnostics.
    pub fn validate(&self, at: &str) -> std::result::Result<(), RouteFilterError> {
        let mut nodes = 0;
        self.check(at, 1, &mut nodes)
    }

    fn check(
        &self,
        at: &str,
        depth: usize,
        nodes: &mut usize,
    ) -> std::result::Result<(), RouteFilterError> {
        *nodes += 1;
        if *nodes > MAX_NODES {
            return Err(RouteFilterError::Size { limit: MAX_NODES });
        }
        if depth > MAX_DEPTH {
            return Err(RouteFilterError::Depth {
                at: at.into(),
                limit: MAX_DEPTH,
            });
        }
        let pointer = |at: String, pointer: &JsonPointer| {
            if pointer.is_well_formed() {
                Ok(())
            } else {
                Err(RouteFilterError::Pointer {
                    at,
                    pointer: pointer.as_str().into(),
                })
            }
        };
        let members = |at: &str, count: usize| {
            if count > MAX_MEMBERS {
                Err(RouteFilterError::Members {
                    at: at.into(),
                    count,
                    limit: MAX_MEMBERS,
                })
            } else {
                Ok(())
            }
        };
        match self {
            Self::Eq { pointer: p, .. } => pointer(format!("{at}.eq"), p),
            Self::Exists { pointer: p } => pointer(format!("{at}.exists"), p),
            Self::In { pointer: p, values } => {
                let at = format!("{at}.in");
                pointer(at.clone(), p)?;
                if values.is_empty() {
                    return Err(RouteFilterError::EmptySet { at });
                }
                members(&at, values.len())
            }
            Self::Prefix { pointer: p, value } => {
                let at = format!("{at}.prefix");
                pointer(at.clone(), p)?;
                if value.is_empty() || value.len() > MAX_TEXT {
                    return Err(RouteFilterError::Prefix { at });
                }
                Ok(())
            }
            Self::Not(inner) => inner.check(&format!("{at}.not"), depth + 1, nodes),
            Self::All(filters) | Self::Any(filters) => {
                let operator = if matches!(self, Self::All(_)) {
                    "all"
                } else {
                    "any"
                };
                if filters.is_empty() {
                    return Err(RouteFilterError::EmptyGroup {
                        at: format!("{at}.{operator}"),
                        operator,
                    });
                }
                members(&format!("{at}.{operator}"), filters.len())?;
                for (index, filter) in filters.iter().enumerate() {
                    filter.check(&format!("{at}.{operator}[{index}]"), depth + 1, nodes)?;
                }
                Ok(())
            }
        }
    }

    /// Whether `data` satisfies the filter. Total: never fails, a missing
    /// field is simply not equal.
    pub fn matches(&self, data: &Value) -> bool {
        match self {
            Self::Eq { pointer, value } => pointer.resolve(data) == Some(value),
            Self::In { pointer, values } => pointer
                .resolve(data)
                .is_some_and(|found| values.contains(found)),
            Self::Prefix { pointer, value } => pointer
                .resolve(data)
                .and_then(Value::as_str)
                .is_some_and(|found| found.starts_with(value.as_str())),
            Self::Exists { pointer } => pointer.resolve(data).is_some_and(|found| !found.is_null()),
            Self::Not(inner) => !inner.matches(data),
            Self::All(filters) => filters.iter().all(|filter| filter.matches(data)),
            Self::Any(filters) => filters.iter().any(|filter| filter.matches(data)),
        }
    }
}

impl RouteFilterError {
    /// This diagnostic as a wire [`Fault`](crate::types::Fault), prefixed with
    /// `context` (e.g. ``route `vesta-anima` ``).
    pub fn into_fault(self, context: &str) -> crate::types::Fault {
        use miette::Diagnostic;
        let code = self.code().map(|c| c.to_string()).unwrap_or_default();
        let help = self.help().map(|h| h.to_string()).unwrap_or_default();
        crate::types::Fault::invalid(format!("{context}: {self} [{code}] help: {help}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn filter(value: Value) -> RouteFilter {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn operators_evaluate_over_json_pointers() {
        let data = json!({"current":{"current_phase":"parked","parked_reason":"lane-circuit-breaker: 3",
            "live_sessions":0,"flags":{"a/b":true,"t~x":1},"nothing":null}});
        let cases = [
            (
                json!({"eq":{"pointer":"/current/current_phase","value":"parked"}}),
                true,
            ),
            (
                json!({"eq":{"pointer":"/current/current_phase","value":"working"}}),
                false,
            ),
            (
                json!({"eq":{"pointer":"/current/live_sessions","value":0}}),
                true,
            ),
            (
                json!({"eq":{"pointer":"/current/live_sessions","value":0.0}}),
                false,
            ),
            (
                json!({"eq":{"pointer":"/current/live_sessions","value":false}}),
                false,
            ),
            (
                json!({"in":{"pointer":"/current/parked_reason","values":["too-big","x"]}}),
                false,
            ),
            (
                json!({"prefix":{"pointer":"/current/parked_reason","value":"lane-circuit-breaker"}}),
                true,
            ),
            (
                json!({"prefix":{"pointer":"/current/live_sessions","value":"0"}}),
                false,
            ),
            (json!({"exists":{"pointer":"/current/flags/a~1b"}}), true),
            (
                json!({"eq":{"pointer":"/current/flags/t~0x","value":1}}),
                true,
            ),
            (json!({"exists":{"pointer":"/current/nothing"}}), false),
            (json!({"exists":{"pointer":"/current/missing"}}), false),
            (
                json!({"not":{"eq":{"pointer":"/current/missing","value":"x"}}}),
                true,
            ),
            (
                json!({"not":{"in":{"pointer":"/current/missing","values":["x"]}}}),
                true,
            ),
            (
                json!({"all":[{"eq":{"pointer":"/current/current_phase","value":"parked"}},
                {"exists":{"pointer":"/current/parked_reason"}}]}),
                true,
            ),
            (
                json!({"all":[{"eq":{"pointer":"/current/current_phase","value":"parked"}},
                {"exists":{"pointer":"/current/missing"}}]}),
                false,
            ),
            (
                json!({"any":[{"exists":{"pointer":"/current/missing"}},
                {"eq":{"pointer":"/current/live_sessions","value":0}}]}),
                true,
            ),
            (json!({"eq":{"pointer":"","value":data.clone()}}), true),
        ];
        for (raw, expected) in cases {
            let f = filter(raw.clone());
            f.validate("when").unwrap();
            assert_eq!(f.matches(&data), expected, "{raw}");
        }
    }

    #[test]
    fn invalid_filters_are_refused_with_located_diagnostics() {
        let cases = [
            (
                json!({"eq":{"pointer":"current","value":1}}),
                "pointer",
                "when.eq",
            ),
            (
                json!({"exists":{"pointer":"/a~2"}}),
                "pointer",
                "when.exists",
            ),
            (json!({"all":[]}), "empty_group", "when.all"),
            (
                json!({"any":[{"not":{"any":[]}}]}),
                "empty_group",
                "when.any[0].not.any",
            ),
            (
                json!({"in":{"pointer":"/a","values":[]}}),
                "empty_set",
                "when.in",
            ),
            (
                json!({"prefix":{"pointer":"/a","value":""}}),
                "prefix",
                "when.prefix",
            ),
        ];
        for (raw, code, at) in cases {
            let error = filter(raw.clone()).validate("when").unwrap_err();
            let fault = error.clone().into_fault("route `r`");
            assert!(
                fault
                    .message
                    .contains(&format!("akh::events::route_filter::{code}]")),
                "{raw}: {}",
                fault.message
            );
            assert!(fault.message.contains(at), "{raw}: {}", fault.message);
            assert!(fault.message.contains("help: "), "{}", fault.message);
        }
        let mut deep = json!({"exists":{"pointer":"/a"}});
        for _ in 0..MAX_DEPTH {
            deep = json!({ "not": deep });
        }
        assert!(matches!(
            filter(deep).validate("when"),
            Err(RouteFilterError::Depth { .. })
        ));
        let wide = json!({"any": (0..=MAX_MEMBERS).map(|_| json!({"exists":{"pointer":"/a"}})).collect::<Vec<_>>()});
        assert!(filter(wide).validate("when").is_err());
        // Unknown operators and unknown operand keys are refused by serde.
        assert!(
            serde_json::from_value::<RouteFilter>(json!({"gt":{"pointer":"/a","value":1}}))
                .is_err()
        );
        assert!(
            serde_json::from_value::<RouteFilter>(
                json!({"eq":{"pointer":"/a","value":1,"extra":true}})
            )
            .is_err()
        );
    }

    /// The fixture shared with the Python evaluator
    /// (`deploy/roman-mini/route_filter.py`): both must agree on every case.
    #[test]
    fn shared_fixture_cases_agree_with_the_python_evaluator() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../deploy/roman-mini/fixtures/route_filter_cases.json"
        ))
        .unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        assert!(cases.len() >= 10);
        for case in cases {
            let f = filter(case["filter"].clone());
            f.validate("when").unwrap();
            assert_eq!(
                f.matches(&case["data"]),
                case["expected"].as_bool().unwrap(),
                "{}",
                case["name"]
            );
        }
    }
}
