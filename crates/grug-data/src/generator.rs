//! Deterministic, seeded generation of tool-calling training examples.
//!
//! The generator owns ground truth: for every example it produces, the
//! `expected_call` is synthesized *first* (valid by construction against the
//! chosen tool's schema), and the natural-language `user_request` is a
//! template that references those same fuzzed values. There is no parsing or
//! inference involved in labeling — the label is the source of truth.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{Map, Value};

use crate::catalog::ToolCatalog;
use crate::error::GrugResult;
use crate::schema::{Example, ParamSpec, ParamType, ToolCall, ToolSchema};

/// A deterministic generator of tool-calling [`Example`]s.
///
/// Construct with [`Generator::new`] and a seed; the same seed always
/// produces the same example stream (see the `determinism` test). Internally
/// this wraps a single [`StdRng`] that is advanced example-by-example, so
/// calling `next_example` repeatedly on one `Generator` gives a reproducible
/// but non-repeating sequence — *not* the same as calling `new` again for
/// each example.
pub struct Generator {
    rng: StdRng,
    catalog: ToolCatalog,
}

impl Generator {
    /// Build a generator over the built-in catalog, seeded deterministically.
    pub fn new(seed: u64) -> Self {
        Self::with_catalog(seed, ToolCatalog::built_in())
    }

    /// Build a generator over a caller-supplied catalog (e.g. a train-only
    /// sub-catalog produced by [`ToolCatalog::split_at`]).
    pub fn with_catalog(seed: u64, catalog: ToolCatalog) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            catalog,
        }
    }

    /// Generate `n` examples, drawing tools uniformly from this generator's
    /// catalog.
    pub fn generate(&mut self, n: usize) -> GrugResult<Vec<Example>> {
        if self.catalog.is_empty() {
            return Err(crate::error::GrugError::EmptyCatalog);
        }
        Ok((0..n).map(|_| self.next_example()).collect())
    }

    /// Generate one example. Panics only if the catalog is empty — callers
    /// going through [`Generator::generate`] get a `GrugResult` instead;
    /// this method is exposed for callers that already checked non-emptiness
    /// (e.g. iterating with a known-good catalog).
    pub fn next_example(&mut self) -> Example {
        let tools = self.catalog.tools();
        debug_assert!(!tools.is_empty(), "catalog must be non-empty");
        let tool = tools
            .choose(&mut self.rng)
            .expect("catalog non-empty (checked by caller)")
            .clone();

        let mut args = Map::new();
        for param in &tool.params {
            // Required params always get a value; optional params are
            // included about half the time so the eval scorer sees both
            // "omitted optional" (still valid) and "provided optional".
            let include = param.required || self.rng.gen_bool(0.5);
            if include {
                args.insert(param.name.clone(), fuzz_value(&mut self.rng, &param.ty));
            }
        }
        let arguments = Value::Object(args);
        let expected_call = ToolCall::new(tool.name.clone(), arguments);
        let user_request = render_request(&mut self.rng, &tool, &expected_call);
        let mock_response = render_mock_response(&tool, &expected_call);

        // Offer a small bundle of tools (the target tool plus a few
        // distractors from the same catalog) rather than just one, so the
        // eval scorer's "tool name is one of the offered tools" check and
        // the model's tool-selection task are both non-trivial.
        let offered = offer_bundle(&mut self.rng, tools, &tool.name);

        Example {
            tools: offered,
            user_request,
            expected_call,
            mock_response,
        }
    }
}

/// Build the list of tools offered alongside `chosen_name`: the chosen tool
/// plus up to 2 distractors, order-shuffled.
fn offer_bundle(rng: &mut StdRng, tools: &[ToolSchema], chosen_name: &str) -> Vec<ToolSchema> {
    let chosen = tools
        .iter()
        .find(|t| t.name == chosen_name)
        .expect("chosen tool is in this catalog")
        .clone();

    let mut distractors: Vec<ToolSchema> = tools
        .iter()
        .filter(|t| t.name != chosen_name)
        .cloned()
        .collect();
    distractors.shuffle(rng);
    distractors.truncate(2);

    let mut offered = vec![chosen];
    offered.extend(distractors);
    offered.shuffle(rng);
    offered
}

/// Fuzz a single value respecting `ty`.
fn fuzz_value(rng: &mut StdRng, ty: &ParamType) -> Value {
    match ty {
        ParamType::String => Value::String(fuzz_string(rng)),
        ParamType::Integer => Value::from(rng.gen_range(1..=500) as i64),
        ParamType::Number => {
            let whole = rng.gen_range(1..=1000) as f64;
            let frac = (rng.gen_range(0..100) as f64) / 100.0;
            Value::from(whole + frac)
        }
        ParamType::Boolean => Value::Bool(rng.gen_bool(0.5)),
        ParamType::Enum(variants) => Value::String(
            variants
                .choose(rng)
                .cloned()
                .unwrap_or_else(|| "unknown".to_string()),
        ),
    }
}

/// A small bank of plausible string fillers, picked by param name so the
/// generated request text reads naturally (a "city" fuzzes to a place name,
/// a "query" to a short phrase, etc.) with a generic fallback for anything
/// else.
fn fuzz_string(rng: &mut StdRng) -> String {
    const PLACES: &[&str] = &[
        "Zurich", "Tokyo", "Nairobi", "Lima", "Oslo", "Seoul", "Denver", "Cairo",
    ];
    const WORDS: &[&str] = &[
        "quarterly report",
        "the new proposal",
        "grandma's recipe",
        "the budget review",
        "our trip plans",
        "the release notes",
    ];
    if rng.gen_bool(0.5) {
        PLACES.choose(rng).unwrap().to_string()
    } else {
        WORDS.choose(rng).unwrap().to_string()
    }
}

/// Render a natural-language user request that references the fuzzed
/// argument values from `call`, via a small per-tool template bank.
///
/// Falls back to a generic template for tools not covered by a bespoke
/// phrasing (keeps this function total over any `ToolSchema`, including
/// caller-supplied ones outside the built-in catalog).
fn render_request(rng: &mut StdRng, tool: &ToolSchema, call: &ToolCall) -> String {
    let args = call.arguments_map();
    let get = |k: &str| args.get(k).and_then(Value::as_str).map(str::to_string);
    let get_num = |k: &str| args.get(k).and_then(|v| v.as_f64());

    let templates: Vec<String> = match tool.name.as_str() {
        "get_weather" => {
            let city = get("city").unwrap_or_else(|| "my city".into());
            vec![
                format!("What's the weather like in {city} right now?"),
                format!("Can you check the weather for {city}?"),
            ]
        }
        "send_email" => {
            let to = get("to").unwrap_or_else(|| "someone".into());
            let subject = get("subject").unwrap_or_else(|| "a quick note".into());
            vec![
                format!("Send an email to {to} about {subject}."),
                format!("Email {to} regarding {subject}, please."),
            ]
        }
        "search_flights" => {
            let origin = get("origin").unwrap_or_else(|| "my city".into());
            let dest = get("destination").unwrap_or_else(|| "somewhere".into());
            vec![format!("Find me flights from {origin} to {dest}.")]
        }
        "set_timer" => {
            let mins = get_num("duration_minutes").unwrap_or(10.0) as i64;
            vec![format!("Set a timer for {mins} minutes.")]
        }
        "convert_currency" => {
            let amount = get_num("amount").unwrap_or(100.0);
            let from = get("from_currency").unwrap_or_else(|| "USD".into());
            let to = get("to_currency").unwrap_or_else(|| "EUR".into());
            vec![format!("Convert {amount:.2} {from} to {to}.")]
        }
        "create_calendar_event" => {
            let title = get("title").unwrap_or_else(|| "a meeting".into());
            let date = get("date").unwrap_or_else(|| "tomorrow".into());
            vec![format!("Create a calendar event called {title} on {date}.")]
        }
        "search_restaurants" => {
            let loc = get("location").unwrap_or_else(|| "downtown".into());
            vec![format!("Find restaurants near {loc}.")]
        }
        "translate_text" => {
            let text = get("text").unwrap_or_else(|| "hello".into());
            let lang = get("target_language").unwrap_or_else(|| "French".into());
            vec![format!("Translate \"{text}\" into {lang}.")]
        }
        "play_music" => {
            let query = get("query").unwrap_or_else(|| "some music".into());
            vec![format!("Play {query}.")]
        }
        "book_hotel" => {
            let city = get("city").unwrap_or_else(|| "the city".into());
            let nights = get_num("nights").unwrap_or(1.0) as i64;
            vec![format!("Book a hotel in {city} for {nights} nights.")]
        }
        "set_reminder" => {
            let text = get("text").unwrap_or_else(|| "something".into());
            let at = get("remind_at").unwrap_or_else(|| "later".into());
            vec![format!("Remind me to {text} at {at}.")]
        }
        "search_web" => {
            let query = get("query").unwrap_or_else(|| "something".into());
            vec![format!("Search the web for {query}.")]
        }
        "get_stock_price" => {
            let ticker = get("ticker").unwrap_or_else(|| "a stock".into());
            vec![format!("What's the current price of {ticker}?")]
        }
        _ => {
            vec![format!(
                "Please use {} to handle this: {}",
                tool.name, tool.description
            )]
        }
    };

    templates
        .choose(rng)
        .cloned()
        .unwrap_or_else(|| format!("Please call {}.", tool.name))
}

/// Render a plausible mock tool response for `call` against `tool`.
fn render_mock_response(tool: &ToolSchema, call: &ToolCall) -> String {
    match tool.name.as_str() {
        "get_weather" => "{\"temperature\": 21, \"condition\": \"partly cloudy\"}".to_string(),
        "send_email" => "{\"status\": \"sent\"}".to_string(),
        "search_flights" => "{\"flights\": [{\"flight_no\": \"AA123\", \"price\": 412}]}".to_string(),
        "set_timer" => "{\"status\": \"timer_set\"}".to_string(),
        "convert_currency" => "{\"converted_amount\": 91.23}".to_string(),
        "create_calendar_event" => "{\"status\": \"event_created\"}".to_string(),
        "search_restaurants" => "{\"results\": [{\"name\": \"Trattoria Bella\", \"rating\": 4.5}]}".to_string(),
        "translate_text" => "{\"translated\": \"bonjour\"}".to_string(),
        "play_music" => "{\"status\": \"now_playing\"}".to_string(),
        "book_hotel" => "{\"status\": \"booked\", \"confirmation\": \"HTL-9921\"}".to_string(),
        "set_reminder" => "{\"status\": \"reminder_set\"}".to_string(),
        "search_web" => "{\"results\": [{\"title\": \"Example result\", \"url\": \"https://example.com\"}]}".to_string(),
        "get_stock_price" => "{\"price\": 187.42}".to_string(),
        _ => {
            let _ = call; // no bespoke fields to echo for unknown tools
            "{\"status\": \"ok\"}".to_string()
        }
    }
}

/// Ensure every required param actually got a value (defensive; exercised
/// implicitly by `next_example`, exposed for tests that want to assert
/// well-formedness of a generated call against its schema).
pub fn call_satisfies_required(schema: &ToolSchema, call: &ToolCall) -> bool {
    let args = call.arguments_map();
    schema
        .required_params()
        .all(|p: &ParamSpec| args.contains_key(&p.name))
}
