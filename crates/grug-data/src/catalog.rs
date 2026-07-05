//! The built-in fake tool catalog and the whole-tool train/holdout split.

use crate::error::{GrugError, GrugResult};
use crate::schema::{ParamSpec, ParamType, ToolSchema};

/// The default number of tools reserved for the whole-tool holdout split out
/// of `ToolCatalog::built_in()`'s dozen-ish tools.
pub const DEFAULT_HOLDOUT_COUNT: usize = 3;

/// A named collection of [`ToolSchema`]s.
///
/// This is the unit the whole-tool holdout split operates on: `split_at`
/// partitions the catalog into two *disjoint* sub-catalogs by tool identity,
/// never by example — so a holdout tool's name, description, and params never
/// appear anywhere in the training stream.
#[derive(Debug, Clone, Default)]
pub struct ToolCatalog {
    tools: Vec<ToolSchema>,
}

impl ToolCatalog {
    pub fn new(tools: Vec<ToolSchema>) -> Self {
        Self { tools }
    }

    pub fn push(&mut self, tool: ToolSchema) {
        self.tools.push(tool);
    }

    pub fn tools(&self) -> &[ToolSchema] {
        &self.tools
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Split into `(train, holdout)` sub-catalogs, disjoint by tool name.
    ///
    /// The split is a deterministic function of catalog order (the last
    /// `holdout_count` tools become holdout) — determinism here matters more
    /// than randomness, since `Generator` already derives its own randomness
    /// from the seed, and a stable split makes holdout tool identity
    /// reproducible across runs for the same catalog.
    pub fn split_at(&self, holdout_count: usize) -> GrugResult<(ToolCatalog, ToolCatalog)> {
        if self.tools.is_empty() {
            return Err(GrugError::EmptyCatalog);
        }
        if holdout_count == 0 || holdout_count >= self.tools.len() {
            return Err(GrugError::DegenerateSplit {
                holdout: holdout_count,
                total: self.tools.len(),
            });
        }
        let split_point = self.tools.len() - holdout_count;
        let train = ToolCatalog::new(self.tools[..split_point].to_vec());
        let holdout = ToolCatalog::new(self.tools[split_point..].to_vec());
        Ok((train, holdout))
    }

    /// The built-in dozen-ish varied fake tools: mixed param types including
    /// enums and optionals, spanning a range of everyday-assistant domains so
    /// templates can generate plausible natural-language requests for each.
    pub fn built_in() -> Self {
        Self::new(vec![
            ToolSchema::new(
                "get_weather",
                "Get the current weather for a city.",
                vec![
                    ParamSpec::new("city", ParamType::String, true),
                    ParamSpec::new(
                        "units",
                        ParamType::Enum(vec!["celsius".into(), "fahrenheit".into()]),
                        false,
                    ),
                ],
            ),
            ToolSchema::new(
                "send_email",
                "Send an email to a recipient.",
                vec![
                    ParamSpec::new("to", ParamType::String, true),
                    ParamSpec::new("subject", ParamType::String, true),
                    ParamSpec::new("body", ParamType::String, true),
                    ParamSpec::new("urgent", ParamType::Boolean, false),
                ],
            ),
            ToolSchema::new(
                "search_flights",
                "Search for flights between two airports.",
                vec![
                    ParamSpec::new("origin", ParamType::String, true),
                    ParamSpec::new("destination", ParamType::String, true),
                    ParamSpec::new("passengers", ParamType::Integer, false),
                    ParamSpec::new(
                        "cabin_class",
                        ParamType::Enum(vec![
                            "economy".into(),
                            "business".into(),
                            "first".into(),
                        ]),
                        false,
                    ),
                ],
            ),
            ToolSchema::new(
                "set_timer",
                "Set a countdown timer.",
                vec![
                    ParamSpec::new("duration_minutes", ParamType::Integer, true),
                    ParamSpec::new("label", ParamType::String, false),
                ],
            ),
            ToolSchema::new(
                "convert_currency",
                "Convert an amount from one currency to another.",
                vec![
                    ParamSpec::new("amount", ParamType::Number, true),
                    ParamSpec::new("from_currency", ParamType::String, true),
                    ParamSpec::new("to_currency", ParamType::String, true),
                ],
            ),
            ToolSchema::new(
                "create_calendar_event",
                "Create a calendar event.",
                vec![
                    ParamSpec::new("title", ParamType::String, true),
                    ParamSpec::new("date", ParamType::String, true),
                    ParamSpec::new("duration_minutes", ParamType::Integer, false),
                    ParamSpec::new("all_day", ParamType::Boolean, false),
                ],
            ),
            ToolSchema::new(
                "search_restaurants",
                "Search for restaurants near a location.",
                vec![
                    ParamSpec::new("location", ParamType::String, true),
                    ParamSpec::new(
                        "cuisine",
                        ParamType::Enum(vec![
                            "italian".into(),
                            "japanese".into(),
                            "mexican".into(),
                            "indian".into(),
                            "thai".into(),
                        ]),
                        false,
                    ),
                    ParamSpec::new("max_price", ParamType::Integer, false),
                ],
            ),
            ToolSchema::new(
                "translate_text",
                "Translate text into a target language.",
                vec![
                    ParamSpec::new("text", ParamType::String, true),
                    ParamSpec::new("target_language", ParamType::String, true),
                ],
            ),
            ToolSchema::new(
                "play_music",
                "Play a song or artist on the speaker.",
                vec![
                    ParamSpec::new("query", ParamType::String, true),
                    ParamSpec::new("shuffle", ParamType::Boolean, false),
                ],
            ),
            ToolSchema::new(
                "book_hotel",
                "Book a hotel room.",
                vec![
                    ParamSpec::new("city", ParamType::String, true),
                    ParamSpec::new("check_in", ParamType::String, true),
                    ParamSpec::new("nights", ParamType::Integer, true),
                    ParamSpec::new(
                        "room_type",
                        ParamType::Enum(vec!["single".into(), "double".into(), "suite".into()]),
                        false,
                    ),
                ],
            ),
            ToolSchema::new(
                "set_reminder",
                "Set a reminder for a future time.",
                vec![
                    ParamSpec::new("text", ParamType::String, true),
                    ParamSpec::new("remind_at", ParamType::String, true),
                    ParamSpec::new("repeat", ParamType::Boolean, false),
                ],
            ),
            ToolSchema::new(
                "search_web",
                "Search the web for a query.",
                vec![
                    ParamSpec::new("query", ParamType::String, true),
                    ParamSpec::new("max_results", ParamType::Integer, false),
                ],
            ),
            ToolSchema::new(
                "get_stock_price",
                "Get the current price of a stock ticker.",
                vec![
                    ParamSpec::new("ticker", ParamType::String, true),
                    ParamSpec::new(
                        "exchange",
                        ParamType::Enum(vec!["NYSE".into(), "NASDAQ".into(), "LSE".into()]),
                        false,
                    ),
                ],
            ),
        ])
    }
}
