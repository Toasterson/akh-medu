//! The actual 45 built-in tool definitions + their phrasing-template banks.
//!
//! Split out of `catalog.rs` purely for file-size hygiene — this module is
//! all *data* (tool schemas + template closures), no new types. Each tool
//! contributes one `(ToolSchema, Vec<PhrasingTemplate>)` pair to
//! `all_tools()`; `ToolCatalog::built_in` wires each pair into a
//! `PhrasingBank` and pushes it into the catalog.
//!
//! Param-shape variety is deliberate: tools here range from 1 required
//! string param (`flip_coin`-adjacent simplicity) to 6+ mixed-type params
//! (`book_hotel`, `create_git_branch`'s sibling `deploy_service`), with enums,
//! optionals, integers, floats, and booleans distributed across domains so a
//! model can't shortcut schema-reading by memorizing one shape.
//!
//! Each tool gets 6-10 phrasing templates (`PhrasingBank::new_default` then
//! reserves ~30% of them, rounded up, as held-out phrasings).

use crate::catalog::{enum_of, get_bool_or, get_num_or, get_str_or, param};
use crate::phrasing::PhrasingTemplate;
use crate::schema::{ParamType, ToolSchema};

type ToolDef = (ToolSchema, Vec<PhrasingTemplate>);

/// Every built-in tool definition, in declaration order (this order is what
/// `ToolCatalog::split_at`'s "last N are holdout" rule operates on for the
/// whole-tool axis).
pub(crate) fn all_tools() -> Vec<ToolDef> {
    vec![
        get_weather(),
        get_forecast(),
        search_flights(),
        book_hotel(),
        search_restaurants(),
        get_directions(),
        check_flight_status(),
        book_train_ticket(),
        rent_car(),
        find_parking(),
        convert_currency(),
        get_stock_price(),
        get_crypto_price(),
        calculate_loan_payment(),
        transfer_money(),
        get_account_balance(),
        calculate_tip(),
        create_calendar_event(),
        delete_calendar_event(),
        list_calendar_events(),
        set_reminder(),
        set_timer(),
        set_alarm(),
        send_email(),
        send_sms(),
        send_slack_message(),
        make_phone_call(),
        schedule_meeting(),
        create_file(),
        delete_file(),
        search_files(),
        compress_files(),
        upload_file(),
        play_music(),
        set_volume(),
        search_podcast(),
        download_video(),
        set_thermostat(),
        toggle_light(),
        lock_door(),
        start_vacuum(),
        set_alarm_system(),
        run_shell_command(),
        create_git_branch(),
        deploy_service(),
        check_build_status(),
        create_ticket(),
        search_web(),
        translate_text(),
        define_word(),
        unit_convert(),
        get_news_headlines(),
        search_recipes(),
        book_ride(),
        order_food(),
        flip_coin(),
    ]
}

// ---------------------------------------------------------------------
// Weather / travel
// ---------------------------------------------------------------------

fn get_weather() -> ToolDef {
    let schema = ToolSchema::new(
        "get_weather",
        "Get the current weather for a city.",
        vec![
            param("city", ParamType::String, true),
            param("units", enum_of(&["celsius", "fahrenheit"]), false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("What's the weather like in {} right now?", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Can you check the weather for {}?", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Tell me the current conditions in {}.", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Is it raining in {} at the moment?", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("How hot is it in {} today?", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Give me a weather update for {}.", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(6, |c| {
            format!("What should I wear outside in {}?", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(7, |c| {
            format!("Weather check for {}, please.", get_str_or(c, "city", "my city"))
        }),
    ];
    (schema, templates)
}

fn get_forecast() -> ToolDef {
    let schema = ToolSchema::new(
        "get_forecast",
        "Get a multi-day weather forecast for a city.",
        vec![
            param("city", ParamType::String, true),
            param("days", ParamType::Integer, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "What's the {}-day forecast for {}?",
                get_num_or(c, "days", 3.0) as i64,
                get_str_or(c, "city", "my city")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Will it rain in {} this week?", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Give me the extended forecast for {}.", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "How's the weather looking in {} for the next {} days?",
                get_str_or(c, "city", "my city"),
                get_num_or(c, "days", 3.0) as i64
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Should I pack an umbrella for my trip to {}?", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Forecast for {}, please.", get_str_or(c, "city", "my city"))
        }),
        PhrasingTemplate::new(6, |c| {
            format!("What's the weather trend in {} looking like?", get_str_or(c, "city", "my city"))
        }),
    ];
    (schema, templates)
}

fn search_flights() -> ToolDef {
    let schema = ToolSchema::new(
        "search_flights",
        "Search for flights between two airports.",
        vec![
            param("origin", ParamType::String, true),
            param("destination", ParamType::String, true),
            param("passengers", ParamType::Integer, false),
            param(
                "cabin_class",
                enum_of(&["economy", "business", "first"]),
                false,
            ),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Find me flights from {} to {}.",
                get_str_or(c, "origin", "my city"),
                get_str_or(c, "destination", "somewhere")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "Search for flights between {} and {}.",
                get_str_or(c, "origin", "my city"),
                get_str_or(c, "destination", "somewhere")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "What flights go from {} to {}?",
                get_str_or(c, "origin", "my city"),
                get_str_or(c, "destination", "somewhere")
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "I need to fly from {} to {}, what are my options?",
                get_str_or(c, "origin", "my city"),
                get_str_or(c, "destination", "somewhere")
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!(
                "Book me a flight from {} to {}.",
                get_str_or(c, "origin", "my city"),
                get_str_or(c, "destination", "somewhere")
            )
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "Look up flight options {} -> {}.",
                get_str_or(c, "origin", "my city"),
                get_str_or(c, "destination", "somewhere")
            )
        }),
        PhrasingTemplate::new(6, |c| {
            format!(
                "Any flights from {} heading to {}?",
                get_str_or(c, "origin", "my city"),
                get_str_or(c, "destination", "somewhere")
            )
        }),
    ];
    (schema, templates)
}

fn book_hotel() -> ToolDef {
    let schema = ToolSchema::new(
        "book_hotel",
        "Book a hotel room.",
        vec![
            param("city", ParamType::String, true),
            param("check_in", ParamType::String, true),
            param("nights", ParamType::Integer, true),
            param(
                "room_type",
                enum_of(&["single", "double", "suite"]),
                false,
            ),
            param("breakfast_included", ParamType::Boolean, false),
            param("max_price", ParamType::Number, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Book a hotel in {} for {} nights.",
                get_str_or(c, "city", "the city"),
                get_num_or(c, "nights", 1.0) as i64
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "I need a room in {} starting {}.",
                get_str_or(c, "city", "the city"),
                get_str_or(c, "check_in", "tomorrow")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "Find me a hotel in {} for {} nights starting on {}.",
                get_str_or(c, "city", "the city"),
                get_num_or(c, "nights", 1.0) as i64,
                get_str_or(c, "check_in", "tomorrow")
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Reserve accommodation in {}.", get_str_or(c, "city", "the city"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!(
                "Where can I stay in {} for {} nights?",
                get_str_or(c, "city", "the city"),
                get_num_or(c, "nights", 1.0) as i64
            )
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Get me a hotel booking in {}, checking in {}.", get_str_or(c, "city", "the city"), get_str_or(c, "check_in", "tomorrow"))
        }),
        PhrasingTemplate::new(6, |c| {
            format!("Set up a hotel stay for {}.", get_str_or(c, "city", "the city"))
        }),
    ];
    (schema, templates)
}

fn search_restaurants() -> ToolDef {
    let schema = ToolSchema::new(
        "search_restaurants",
        "Search for restaurants near a location.",
        vec![
            param("location", ParamType::String, true),
            param(
                "cuisine",
                enum_of(&["italian", "japanese", "mexican", "indian", "thai"]),
                false,
            ),
            param("max_price", ParamType::Integer, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Find restaurants near {}.", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Where can I eat near {}?", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Recommend a place to eat around {}.", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("I'm hungry, what's good near {}?", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Search for dinner spots in {}.", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Any restaurant recommendations for {}?", get_str_or(c, "location", "downtown"))
        }),
    ];
    (schema, templates)
}

fn get_directions() -> ToolDef {
    let schema = ToolSchema::new(
        "get_directions",
        "Get directions between two locations.",
        vec![
            param("origin", ParamType::String, true),
            param("destination", ParamType::String, true),
            param("mode", enum_of(&["driving", "walking", "transit", "cycling"]), false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "How do I get from {} to {}?",
                get_str_or(c, "origin", "here"),
                get_str_or(c, "destination", "there")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "Give me directions from {} to {}.",
                get_str_or(c, "origin", "here"),
                get_str_or(c, "destination", "there")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!("What's the route to {}?", get_str_or(c, "destination", "there"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "Navigate me from {} to {}.",
                get_str_or(c, "origin", "here"),
                get_str_or(c, "destination", "there")
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!("What's the fastest way to {}?", get_str_or(c, "destination", "there"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "Show me how to walk from {} to {}.",
                get_str_or(c, "origin", "here"),
                get_str_or(c, "destination", "there")
            )
        }),
    ];
    (schema, templates)
}

fn check_flight_status() -> ToolDef {
    let schema = ToolSchema::new(
        "check_flight_status",
        "Check the current status of a flight by its number.",
        vec![param("flight_number", ParamType::String, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Is flight {} on time?", get_str_or(c, "flight_number", "AA123"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Check the status of flight {}.", get_str_or(c, "flight_number", "AA123"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Has flight {} been delayed?", get_str_or(c, "flight_number", "AA123"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("What gate is flight {} departing from?", get_str_or(c, "flight_number", "AA123"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Any updates on flight {}?", get_str_or(c, "flight_number", "AA123"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Track flight {} for me.", get_str_or(c, "flight_number", "AA123"))
        }),
    ];
    (schema, templates)
}

fn book_train_ticket() -> ToolDef {
    let schema = ToolSchema::new(
        "book_train_ticket",
        "Book a train ticket between two stations.",
        vec![
            param("origin_station", ParamType::String, true),
            param("destination_station", ParamType::String, true),
            param("departure_date", ParamType::String, true),
            param("first_class", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Book a train from {} to {} on {}.",
                get_str_or(c, "origin_station", "Central Station"),
                get_str_or(c, "destination_station", "North Station"),
                get_str_or(c, "departure_date", "tomorrow")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "I need a train ticket from {} to {}.",
                get_str_or(c, "origin_station", "Central Station"),
                get_str_or(c, "destination_station", "North Station")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Get me on a train to {}.", get_str_or(c, "destination_station", "North Station"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "Reserve a train seat from {} to {} departing {}.",
                get_str_or(c, "origin_station", "Central Station"),
                get_str_or(c, "destination_station", "North Station"),
                get_str_or(c, "departure_date", "tomorrow")
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Purchase a rail ticket to {}.", get_str_or(c, "destination_station", "North Station"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "Can you get me a train from {} to {}?",
                get_str_or(c, "origin_station", "Central Station"),
                get_str_or(c, "destination_station", "North Station")
            )
        }),
    ];
    (schema, templates)
}

fn rent_car() -> ToolDef {
    let schema = ToolSchema::new(
        "rent_car",
        "Rent a car from a location for a number of days.",
        vec![
            param("location", ParamType::String, true),
            param("days", ParamType::Integer, true),
            param("car_type", enum_of(&["economy", "suv", "luxury", "van"]), false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Rent a car in {} for {} days.",
                get_str_or(c, "location", "the city"),
                get_num_or(c, "days", 1.0) as i64
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("I need a rental car in {}.", get_str_or(c, "location", "the city"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "Book a car rental for {} days starting in {}.",
                get_num_or(c, "days", 1.0) as i64,
                get_str_or(c, "location", "the city")
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Get me wheels in {}.", get_str_or(c, "location", "the city"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Reserve a rental vehicle in {}.", get_str_or(c, "location", "the city"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "Set up a car rental, {} days, picking up in {}.",
                get_num_or(c, "days", 1.0) as i64,
                get_str_or(c, "location", "the city")
            )
        }),
    ];
    (schema, templates)
}

fn find_parking() -> ToolDef {
    let schema = ToolSchema::new(
        "find_parking",
        "Find available parking near a location.",
        vec![param("location", ParamType::String, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Where can I park near {}?", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Find parking near {}.", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Is there a parking garage close to {}?", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("I need to park near {}.", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Any parking spots available around {}?", get_str_or(c, "location", "downtown"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Locate parking options for {}.", get_str_or(c, "location", "downtown"))
        }),
    ];
    (schema, templates)
}

// ---------------------------------------------------------------------
// Finance
// ---------------------------------------------------------------------

fn convert_currency() -> ToolDef {
    let schema = ToolSchema::new(
        "convert_currency",
        "Convert an amount from one currency to another.",
        vec![
            param("amount", ParamType::Number, true),
            param("from_currency", ParamType::String, true),
            param("to_currency", ParamType::String, true),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Convert {:.2} {} to {}.",
                get_num_or(c, "amount", 100.0),
                get_str_or(c, "from_currency", "USD"),
                get_str_or(c, "to_currency", "EUR")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "How much is {:.2} {} in {}?",
                get_num_or(c, "amount", 100.0),
                get_str_or(c, "from_currency", "USD"),
                get_str_or(c, "to_currency", "EUR")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "What's {} {} worth in {}?",
                get_num_or(c, "amount", 100.0),
                get_str_or(c, "from_currency", "USD"),
                get_str_or(c, "to_currency", "EUR")
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "Exchange {:.2} {} for {}.",
                get_num_or(c, "amount", 100.0),
                get_str_or(c, "from_currency", "USD"),
                get_str_or(c, "to_currency", "EUR")
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!(
                "{}-to-{} rate for {:.2}?",
                get_str_or(c, "from_currency", "USD"),
                get_str_or(c, "to_currency", "EUR"),
                get_num_or(c, "amount", 100.0)
            )
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "Convert my {:.2} {} into {} please.",
                get_num_or(c, "amount", 100.0),
                get_str_or(c, "from_currency", "USD"),
                get_str_or(c, "to_currency", "EUR")
            )
        }),
    ];
    (schema, templates)
}

fn get_stock_price() -> ToolDef {
    let schema = ToolSchema::new(
        "get_stock_price",
        "Get the current price of a stock ticker.",
        vec![
            param("ticker", ParamType::String, true),
            param("exchange", enum_of(&["NYSE", "NASDAQ", "LSE"]), false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("What's the current price of {}?", get_str_or(c, "ticker", "a stock"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("How much is {} trading at?", get_str_or(c, "ticker", "a stock"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Check the stock price for {}.", get_str_or(c, "ticker", "a stock"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("What's {} worth right now?", get_str_or(c, "ticker", "a stock"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Give me a quote for {}.", get_str_or(c, "ticker", "a stock"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Look up {} on the market.", get_str_or(c, "ticker", "a stock"))
        }),
    ];
    (schema, templates)
}

fn get_crypto_price() -> ToolDef {
    let schema = ToolSchema::new(
        "get_crypto_price",
        "Get the current price of a cryptocurrency.",
        vec![
            param("symbol", ParamType::String, true),
            param("vs_currency", ParamType::String, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("What's the price of {} right now?", get_str_or(c, "symbol", "BTC"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("How much is {} worth?", get_str_or(c, "symbol", "BTC"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Check the {} exchange rate.", get_str_or(c, "symbol", "BTC"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Give me the current {} price.", get_str_or(c, "symbol", "BTC"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("What's {} trading at today?", get_str_or(c, "symbol", "BTC"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Look up {} crypto price.", get_str_or(c, "symbol", "BTC"))
        }),
    ];
    (schema, templates)
}

fn calculate_loan_payment() -> ToolDef {
    let schema = ToolSchema::new(
        "calculate_loan_payment",
        "Calculate the monthly payment for a loan.",
        vec![
            param("principal", ParamType::Number, true),
            param("annual_rate_percent", ParamType::Number, true),
            param("term_months", ParamType::Integer, true),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "What would my monthly payment be on a {:.0} loan over {} months at {:.1}%?",
                get_num_or(c, "principal", 10000.0),
                get_num_or(c, "term_months", 36.0) as i64,
                get_num_or(c, "annual_rate_percent", 5.0)
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "Calculate the payment on a {:.0} loan.",
                get_num_or(c, "principal", 10000.0)
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "How much would I pay each month for {:.0} at {:.1}% interest?",
                get_num_or(c, "principal", 10000.0),
                get_num_or(c, "annual_rate_percent", 5.0)
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "Work out my loan payment: {:.0} principal, {} month term.",
                get_num_or(c, "principal", 10000.0),
                get_num_or(c, "term_months", 36.0) as i64
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Estimate monthly installments for a {:.0} loan.", get_num_or(c, "principal", 10000.0))
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "If I borrow {:.0} at {:.1}%, what's the monthly cost?",
                get_num_or(c, "principal", 10000.0),
                get_num_or(c, "annual_rate_percent", 5.0)
            )
        }),
    ];
    (schema, templates)
}

fn transfer_money() -> ToolDef {
    let schema = ToolSchema::new(
        "transfer_money",
        "Transfer money to another account.",
        vec![
            param("recipient", ParamType::String, true),
            param("amount", ParamType::Number, true),
            param("currency", ParamType::String, false),
            param("memo", ParamType::String, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Send {:.2} to {}.",
                get_num_or(c, "amount", 50.0),
                get_str_or(c, "recipient", "my friend")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "Transfer {:.2} to {}'s account.",
                get_num_or(c, "amount", 50.0),
                get_str_or(c, "recipient", "my friend")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Pay {} {:.2}.", get_str_or(c, "recipient", "my friend"), get_num_or(c, "amount", 50.0))
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "Move {:.2} over to {}.",
                get_num_or(c, "amount", 50.0),
                get_str_or(c, "recipient", "my friend")
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Wire {} some money: {:.2}.", get_str_or(c, "recipient", "my friend"), get_num_or(c, "amount", 50.0))
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "I owe {} {:.2}, send it over.",
                get_str_or(c, "recipient", "my friend"),
                get_num_or(c, "amount", 50.0)
            )
        }),
    ];
    (schema, templates)
}

fn get_account_balance() -> ToolDef {
    let schema = ToolSchema::new(
        "get_account_balance",
        "Get the current balance of an account.",
        vec![param(
            "account_type",
            enum_of(&["checking", "savings", "credit"]),
            false,
        )],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("What's my {} balance?", get_str_or(c, "account_type", "account"))
        }),
        PhrasingTemplate::new(1, |_c| "How much money do I have?".to_string()),
        PhrasingTemplate::new(2, |c| {
            format!("Check my {} account balance.", get_str_or(c, "account_type", "account"))
        }),
        PhrasingTemplate::new(3, |_c| "What's my current balance?".to_string()),
        PhrasingTemplate::new(4, |c| {
            format!("Show me the balance on my {} account.", get_str_or(c, "account_type", "account"))
        }),
        PhrasingTemplate::new(5, |_c| "How much do I have in the bank right now?".to_string()),
    ];
    (schema, templates)
}

fn calculate_tip() -> ToolDef {
    let schema = ToolSchema::new(
        "calculate_tip",
        "Calculate a tip amount for a bill.",
        vec![
            param("bill_amount", ParamType::Number, true),
            param("tip_percent", ParamType::Number, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "How much tip should I leave on a {:.2} bill?",
                get_num_or(c, "bill_amount", 40.0)
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "Calculate a {:.0}% tip on {:.2}.",
                get_num_or(c, "tip_percent", 18.0),
                get_num_or(c, "bill_amount", 40.0)
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!("What's {:.0}% of {:.2}?", get_num_or(c, "tip_percent", 18.0), get_num_or(c, "bill_amount", 40.0))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Work out the tip for a {:.2} check.", get_num_or(c, "bill_amount", 40.0))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("How much should I tip on {:.2}?", get_num_or(c, "bill_amount", 40.0))
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "Split the tip math for a {:.2} bill at {:.0}%.",
                get_num_or(c, "bill_amount", 40.0),
                get_num_or(c, "tip_percent", 18.0)
            )
        }),
    ];
    (schema, templates)
}

// ---------------------------------------------------------------------
// Calendar / time
// ---------------------------------------------------------------------

fn create_calendar_event() -> ToolDef {
    let schema = ToolSchema::new(
        "create_calendar_event",
        "Create a calendar event.",
        vec![
            param("title", ParamType::String, true),
            param("date", ParamType::String, true),
            param("duration_minutes", ParamType::Integer, false),
            param("all_day", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Create a calendar event called {} on {}.",
                get_str_or(c, "title", "a meeting"),
                get_str_or(c, "date", "tomorrow")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "Schedule {} for {}.",
                get_str_or(c, "title", "a meeting"),
                get_str_or(c, "date", "tomorrow")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Add {} to my calendar on {}.", get_str_or(c, "title", "a meeting"), get_str_or(c, "date", "tomorrow"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Put {} on the schedule for {}.", get_str_or(c, "title", "a meeting"), get_str_or(c, "date", "tomorrow"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Block out time for {} on {}.", get_str_or(c, "title", "a meeting"), get_str_or(c, "date", "tomorrow"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("I need an event named {} on {}.", get_str_or(c, "title", "a meeting"), get_str_or(c, "date", "tomorrow"))
        }),
    ];
    (schema, templates)
}

fn delete_calendar_event() -> ToolDef {
    let schema = ToolSchema::new(
        "delete_calendar_event",
        "Delete an existing calendar event by title.",
        vec![param("title", ParamType::String, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Cancel the {} event.", get_str_or(c, "title", "meeting"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Delete {} from my calendar.", get_str_or(c, "title", "meeting"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Remove the calendar entry for {}.", get_str_or(c, "title", "meeting"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("I need to cancel {}.", get_str_or(c, "title", "meeting"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Take {} off my schedule.", get_str_or(c, "title", "meeting"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Clear {} from the calendar.", get_str_or(c, "title", "meeting"))
        }),
    ];
    (schema, templates)
}

fn list_calendar_events() -> ToolDef {
    let schema = ToolSchema::new(
        "list_calendar_events",
        "List calendar events for a given date.",
        vec![param("date", ParamType::String, false)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("What's on my calendar for {}?", get_str_or(c, "date", "today"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Show me my schedule for {}.", get_str_or(c, "date", "today"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("What do I have going on {}?", get_str_or(c, "date", "today"))
        }),
        PhrasingTemplate::new(3, |_c| "List my upcoming events.".to_string()),
        PhrasingTemplate::new(4, |c| {
            format!("Any meetings {}?", get_str_or(c, "date", "today"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Pull up my agenda for {}.", get_str_or(c, "date", "today"))
        }),
    ];
    (schema, templates)
}

fn set_reminder() -> ToolDef {
    let schema = ToolSchema::new(
        "set_reminder",
        "Set a reminder for a future time.",
        vec![
            param("text", ParamType::String, true),
            param("remind_at", ParamType::String, true),
            param("repeat", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Remind me to {} at {}.",
                get_str_or(c, "text", "something"),
                get_str_or(c, "remind_at", "later")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Set a reminder for {}.", get_str_or(c, "text", "something"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "Don't let me forget to {} at {}.",
                get_str_or(c, "text", "something"),
                get_str_or(c, "remind_at", "later")
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Ping me at {} to {}.", get_str_or(c, "remind_at", "later"), get_str_or(c, "text", "something"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("I need a nudge to {}.", get_str_or(c, "text", "something"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Alert me about {} at {}.", get_str_or(c, "text", "something"), get_str_or(c, "remind_at", "later"))
        }),
    ];
    (schema, templates)
}

fn set_timer() -> ToolDef {
    let schema = ToolSchema::new(
        "set_timer",
        "Set a countdown timer.",
        vec![
            param("duration_minutes", ParamType::Integer, true),
            param("label", ParamType::String, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Set a timer for {} minutes.", get_num_or(c, "duration_minutes", 10.0) as i64)
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Start a {}-minute countdown.", get_num_or(c, "duration_minutes", 10.0) as i64)
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Time {} minutes for me.", get_num_or(c, "duration_minutes", 10.0) as i64)
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Count down from {} minutes.", get_num_or(c, "duration_minutes", 10.0) as i64)
        }),
        PhrasingTemplate::new(4, |c| {
            format!("I need a {} minute timer for {}.", get_num_or(c, "duration_minutes", 10.0) as i64, get_str_or(c, "label", "cooking"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Timer, {} minutes, go.", get_num_or(c, "duration_minutes", 10.0) as i64)
        }),
    ];
    (schema, templates)
}

fn set_alarm() -> ToolDef {
    let schema = ToolSchema::new(
        "set_alarm",
        "Set an alarm for a specific time.",
        vec![
            param("time", ParamType::String, true),
            param("label", ParamType::String, false),
            param("repeat_daily", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Wake me up at {}.", get_str_or(c, "time", "7am"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Set an alarm for {}.", get_str_or(c, "time", "7am"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("I need to get up at {}.", get_str_or(c, "time", "7am"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Alarm for {}, please.", get_str_or(c, "time", "7am"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Set my alarm clock to {}.", get_str_or(c, "time", "7am"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Get me up at {} tomorrow.", get_str_or(c, "time", "7am"))
        }),
    ];
    (schema, templates)
}

// ---------------------------------------------------------------------
// Messaging
// ---------------------------------------------------------------------

fn send_email() -> ToolDef {
    let schema = ToolSchema::new(
        "send_email",
        "Send an email to a recipient.",
        vec![
            param("to", ParamType::String, true),
            param("subject", ParamType::String, true),
            param("body", ParamType::String, true),
            param("urgent", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Send an email to {} about {}.",
                get_str_or(c, "to", "someone"),
                get_str_or(c, "subject", "a quick note")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "Email {} regarding {}, please.",
                get_str_or(c, "to", "someone"),
                get_str_or(c, "subject", "a quick note")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Write to {} about {}.", get_str_or(c, "to", "someone"), get_str_or(c, "subject", "a quick note"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Draft an email to {}.", get_str_or(c, "to", "someone"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!(
                "Shoot {} a message about {}.",
                get_str_or(c, "to", "someone"),
                get_str_or(c, "subject", "a quick note")
            )
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Let {} know about {} by email.", get_str_or(c, "to", "someone"), get_str_or(c, "subject", "a quick note"))
        }),
    ];
    (schema, templates)
}

fn send_sms() -> ToolDef {
    let schema = ToolSchema::new(
        "send_sms",
        "Send a text message to a phone number.",
        vec![
            param("phone_number", ParamType::String, true),
            param("message", ParamType::String, true),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Text {} saying {}.",
                get_str_or(c, "phone_number", "555-0100"),
                get_str_or(c, "message", "hello")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Send a text to {}.", get_str_or(c, "phone_number", "555-0100"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "Message {}: {}",
                get_str_or(c, "phone_number", "555-0100"),
                get_str_or(c, "message", "hello")
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!("SMS {} with a quick note.", get_str_or(c, "phone_number", "555-0100"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!(
                "Can you text {} and say {}?",
                get_str_or(c, "phone_number", "555-0100"),
                get_str_or(c, "message", "hello")
            )
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Fire off a text to {}.", get_str_or(c, "phone_number", "555-0100"))
        }),
    ];
    (schema, templates)
}

fn send_slack_message() -> ToolDef {
    let schema = ToolSchema::new(
        "send_slack_message",
        "Send a message to a Slack channel.",
        vec![
            param("channel", ParamType::String, true),
            param("message", ParamType::String, true),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Post {} in #{}.", get_str_or(c, "message", "an update"), get_str_or(c, "channel", "general"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Send a Slack message to #{}.", get_str_or(c, "channel", "general"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "Message #{} saying {}.",
                get_str_or(c, "channel", "general"),
                get_str_or(c, "message", "an update")
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Drop a note in #{} on Slack.", get_str_or(c, "channel", "general"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Let the #{} channel know: {}.", get_str_or(c, "channel", "general"), get_str_or(c, "message", "an update"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Ping #{} on Slack.", get_str_or(c, "channel", "general"))
        }),
    ];
    (schema, templates)
}

fn make_phone_call() -> ToolDef {
    let schema = ToolSchema::new(
        "make_phone_call",
        "Place a phone call to a contact or number.",
        vec![param("contact", ParamType::String, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Call {}.", get_str_or(c, "contact", "someone"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Phone {} for me.", get_str_or(c, "contact", "someone"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Dial {}.", get_str_or(c, "contact", "someone"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Get {} on the phone.", get_str_or(c, "contact", "someone"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Ring up {}.", get_str_or(c, "contact", "someone"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Place a call to {}.", get_str_or(c, "contact", "someone"))
        }),
    ];
    (schema, templates)
}

fn schedule_meeting() -> ToolDef {
    let schema = ToolSchema::new(
        "schedule_meeting",
        "Schedule a meeting with participants.",
        vec![
            param("participants", ParamType::String, true),
            param("date", ParamType::String, true),
            param("duration_minutes", ParamType::Integer, false),
            param("location", ParamType::String, false),
            param("video_call", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Schedule a meeting with {} on {}.",
                get_str_or(c, "participants", "the team"),
                get_str_or(c, "date", "tomorrow")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Set up a meeting with {}.", get_str_or(c, "participants", "the team"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Get {} together for a call on {}.", get_str_or(c, "participants", "the team"), get_str_or(c, "date", "tomorrow"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Arrange a meeting with {} for {}.", get_str_or(c, "participants", "the team"), get_str_or(c, "date", "tomorrow"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Book time with {}.", get_str_or(c, "participants", "the team"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("I want to meet with {} on {}.", get_str_or(c, "participants", "the team"), get_str_or(c, "date", "tomorrow"))
        }),
    ];
    (schema, templates)
}

// ---------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------

fn create_file() -> ToolDef {
    let schema = ToolSchema::new(
        "create_file",
        "Create a new file with given content.",
        vec![
            param("path", ParamType::String, true),
            param("content", ParamType::String, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Create a file at {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Make a new file called {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Write a file to {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("I need a new file: {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Set up a file named {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Generate a file at path {}.", get_str_or(c, "path", "notes.txt"))
        }),
    ];
    (schema, templates)
}

fn delete_file() -> ToolDef {
    let schema = ToolSchema::new(
        "delete_file",
        "Delete a file by path.",
        vec![param("path", ParamType::String, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Delete {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Remove the file {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Get rid of {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Trash the file at {}.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("I don't need {} anymore, delete it.", get_str_or(c, "path", "notes.txt"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Erase {} from disk.", get_str_or(c, "path", "notes.txt"))
        }),
    ];
    (schema, templates)
}

fn search_files() -> ToolDef {
    let schema = ToolSchema::new(
        "search_files",
        "Search for files matching a query.",
        vec![
            param("query", ParamType::String, true),
            param("directory", ParamType::String, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Find files matching {}.", get_str_or(c, "query", "report"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Search for {} in my files.", get_str_or(c, "query", "report"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Where's the file about {}?", get_str_or(c, "query", "report"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Look for files named {}.", get_str_or(c, "query", "report"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Can you find anything matching {}?", get_str_or(c, "query", "report"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Search my documents for {}.", get_str_or(c, "query", "report"))
        }),
    ];
    (schema, templates)
}

fn compress_files() -> ToolDef {
    let schema = ToolSchema::new(
        "compress_files",
        "Compress a directory into an archive.",
        vec![
            param("directory", ParamType::String, true),
            param("format", enum_of(&["zip", "tar_gz", "seven_zip"]), false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Zip up {}.", get_str_or(c, "directory", "the folder"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Compress {} into an archive.", get_str_or(c, "directory", "the folder"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Make a zip of {}.", get_str_or(c, "directory", "the folder"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Archive the {} directory.", get_str_or(c, "directory", "the folder"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Package up {} for me.", get_str_or(c, "directory", "the folder"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Bundle {} into a zip file.", get_str_or(c, "directory", "the folder"))
        }),
    ];
    (schema, templates)
}

fn upload_file() -> ToolDef {
    let schema = ToolSchema::new(
        "upload_file",
        "Upload a file to cloud storage.",
        vec![
            param("path", ParamType::String, true),
            param("destination", ParamType::String, false),
            param("public", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Upload {} to the cloud.", get_str_or(c, "path", "the file"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Send {} to storage.", get_str_or(c, "path", "the file"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Back up {} online.", get_str_or(c, "path", "the file"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Push {} to my drive.", get_str_or(c, "path", "the file"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Sync {} to the cloud.", get_str_or(c, "path", "the file"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Store {} remotely.", get_str_or(c, "path", "the file"))
        }),
    ];
    (schema, templates)
}

// ---------------------------------------------------------------------
// Media
// ---------------------------------------------------------------------

fn play_music() -> ToolDef {
    let schema = ToolSchema::new(
        "play_music",
        "Play a song or artist on the speaker.",
        vec![
            param("query", ParamType::String, true),
            param("shuffle", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Play {}.", get_str_or(c, "query", "some music"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Put on {}.", get_str_or(c, "query", "some music"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Can you play {} for me?", get_str_or(c, "query", "some music"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Start playing {}.", get_str_or(c, "query", "some music"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Queue up {}.", get_str_or(c, "query", "some music"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("I want to hear {}.", get_str_or(c, "query", "some music"))
        }),
    ];
    (schema, templates)
}

fn set_volume() -> ToolDef {
    let schema = ToolSchema::new(
        "set_volume",
        "Set the playback volume level.",
        vec![param("level_percent", ParamType::Integer, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Set the volume to {}%.", get_num_or(c, "level_percent", 50.0) as i64)
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Turn the volume up to {}.", get_num_or(c, "level_percent", 50.0) as i64)
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Make it {}% volume.", get_num_or(c, "level_percent", 50.0) as i64)
        }),
        PhrasingTemplate::new(3, |_c| "Turn the volume down.".to_string()),
        PhrasingTemplate::new(4, |c| {
            format!("Adjust volume to {}.", get_num_or(c, "level_percent", 50.0) as i64)
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Set speaker level to {}%.", get_num_or(c, "level_percent", 50.0) as i64)
        }),
    ];
    (schema, templates)
}

fn search_podcast() -> ToolDef {
    let schema = ToolSchema::new(
        "search_podcast",
        "Search for a podcast by name or topic.",
        vec![param("query", ParamType::String, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Find a podcast about {}.", get_str_or(c, "query", "science"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Search podcasts for {}.", get_str_or(c, "query", "science"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Any good podcasts on {}?", get_str_or(c, "query", "science"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Look up podcast: {}.", get_str_or(c, "query", "science"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("I want to listen to something about {}.", get_str_or(c, "query", "science"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Show me podcasts related to {}.", get_str_or(c, "query", "science"))
        }),
    ];
    (schema, templates)
}

fn download_video() -> ToolDef {
    let schema = ToolSchema::new(
        "download_video",
        "Download a video from a URL.",
        vec![
            param("url", ParamType::String, true),
            param("quality", enum_of(&["480p", "720p", "1080p", "4k"]), false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Download this video: {}.", get_str_or(c, "url", "https://example.com/video"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Save the video at {}.", get_str_or(c, "url", "https://example.com/video"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Grab a copy of {}.", get_str_or(c, "url", "https://example.com/video"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Pull down the video from {}.", get_str_or(c, "url", "https://example.com/video"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Fetch and save {}.", get_str_or(c, "url", "https://example.com/video"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("I want to keep this video: {}.", get_str_or(c, "url", "https://example.com/video"))
        }),
    ];
    (schema, templates)
}

// ---------------------------------------------------------------------
// Smart home
// ---------------------------------------------------------------------

fn set_thermostat() -> ToolDef {
    let schema = ToolSchema::new(
        "set_thermostat",
        "Set the home thermostat to a target temperature.",
        vec![
            param("temperature", ParamType::Number, true),
            param("unit", enum_of(&["celsius", "fahrenheit"]), false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Set the thermostat to {:.0}.", get_num_or(c, "temperature", 21.0))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Make it {:.0} degrees in here.", get_num_or(c, "temperature", 21.0))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Turn the heat to {:.0}.", get_num_or(c, "temperature", 21.0))
        }),
        PhrasingTemplate::new(3, |_c| "It's cold, warm it up.".to_string()),
        PhrasingTemplate::new(4, |c| {
            format!("Adjust the temperature to {:.0} degrees.", get_num_or(c, "temperature", 21.0))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Set home temp to {:.0}.", get_num_or(c, "temperature", 21.0))
        }),
    ];
    (schema, templates)
}

fn toggle_light() -> ToolDef {
    let schema = ToolSchema::new(
        "toggle_light",
        "Turn a light on or off in a room.",
        vec![
            param("room", ParamType::String, true),
            param("on", ParamType::Boolean, true),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            let action = if get_bool_or(c, "on", true) { "on" } else { "off" };
            format!("Turn the {} lights {}.", get_str_or(c, "room", "living room"), action)
        }),
        PhrasingTemplate::new(1, |c| {
            let action = if get_bool_or(c, "on", true) { "on" } else { "off" };
            format!("Switch the {} light {}.", get_str_or(c, "room", "living room"), action)
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Lights {}.", get_str_or(c, "room", "living room"))
        }),
        PhrasingTemplate::new(3, |c| {
            let action = if get_bool_or(c, "on", true) { "" } else { "off " };
            format!("Flip the {}light in the {}.", action, get_str_or(c, "room", "living room"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("It's dark in the {}, fix the lights.", get_str_or(c, "room", "living room"))
        }),
        PhrasingTemplate::new(5, |c| {
            let action = if get_bool_or(c, "on", true) { "on" } else { "off" };
            format!("Please turn {} the {} light.", action, get_str_or(c, "room", "living room"))
        }),
    ];
    (schema, templates)
}

fn lock_door() -> ToolDef {
    let schema = ToolSchema::new(
        "lock_door",
        "Lock or unlock a named door.",
        vec![
            param("door", ParamType::String, true),
            param("lock", ParamType::Boolean, true),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            let action = if get_bool_or(c, "lock", true) { "Lock" } else { "Unlock" };
            format!("{} the {} door.", action, get_str_or(c, "door", "front"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Is the {} door locked?", get_str_or(c, "door", "front"))
        }),
        PhrasingTemplate::new(2, |c| {
            let action = if get_bool_or(c, "lock", true) { "secure" } else { "open up" };
            format!("Please {} the {} door.", action, get_str_or(c, "door", "front"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Make sure the {} door is locked.", get_str_or(c, "door", "front"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Let me in through the {} door.", get_str_or(c, "door", "front"))
        }),
        PhrasingTemplate::new(5, |c| {
            let action = if get_bool_or(c, "lock", true) { "lock" } else { "unlock" };
            format!("Can you {} the {} door?", action, get_str_or(c, "door", "front"))
        }),
    ];
    (schema, templates)
}

fn start_vacuum() -> ToolDef {
    let schema = ToolSchema::new(
        "start_vacuum",
        "Start the robot vacuum, optionally in one room.",
        vec![param("room", ParamType::String, false)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Vacuum the {}.", get_str_or(c, "room", "living room"))
        }),
        PhrasingTemplate::new(1, |_c| "Start the robot vacuum.".to_string()),
        PhrasingTemplate::new(2, |c| {
            format!("Clean the {} floor.", get_str_or(c, "room", "living room"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Send the vacuum to the {}.", get_str_or(c, "room", "living room"))
        }),
        PhrasingTemplate::new(4, |_c| "Run a cleaning cycle.".to_string()),
        PhrasingTemplate::new(5, |c| {
            format!("Tidy up the {} with the robot.", get_str_or(c, "room", "living room"))
        }),
    ];
    (schema, templates)
}

fn set_alarm_system() -> ToolDef {
    let schema = ToolSchema::new(
        "set_alarm_system",
        "Arm or disarm the home security system.",
        vec![param("armed", ParamType::Boolean, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            if get_bool_or(c, "armed", true) { "Arm the security system.".to_string() } else { "Disarm the alarm.".to_string() }
        }),
        PhrasingTemplate::new(1, |c| {
            if get_bool_or(c, "armed", true) { "Set the alarm before we leave.".to_string() } else { "Turn off the house alarm.".to_string() }
        }),
        PhrasingTemplate::new(2, |c| {
            if get_bool_or(c, "armed", true) { "Lock everything down.".to_string() } else { "We're home, disarm it.".to_string() }
        }),
        PhrasingTemplate::new(3, |_c| "Is the security system armed?".to_string()),
        PhrasingTemplate::new(4, |c| {
            if get_bool_or(c, "armed", true) { "Activate home security.".to_string() } else { "Deactivate the alarm system.".to_string() }
        }),
        PhrasingTemplate::new(5, |c| {
            if get_bool_or(c, "armed", true) { "Turn on the alarm, we're heading out.".to_string() } else { "Switch off the security system.".to_string() }
        }),
    ];
    (schema, templates)
}

// ---------------------------------------------------------------------
// Dev tools
// ---------------------------------------------------------------------

fn run_shell_command() -> ToolDef {
    let schema = ToolSchema::new(
        "run_shell_command",
        "Run a shell command on the host.",
        vec![
            param("command", ParamType::String, true),
            param("working_directory", ParamType::String, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Run `{}`.", get_str_or(c, "command", "ls"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Execute {} in the shell.", get_str_or(c, "command", "ls"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Can you run this command: {}", get_str_or(c, "command", "ls"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Type {} at the terminal.", get_str_or(c, "command", "ls"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("I need to run `{}`.", get_str_or(c, "command", "ls"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Please execute: {}", get_str_or(c, "command", "ls"))
        }),
    ];
    (schema, templates)
}

fn create_git_branch() -> ToolDef {
    let schema = ToolSchema::new(
        "create_git_branch",
        "Create a new git branch from a base branch.",
        vec![
            param("branch_name", ParamType::String, true),
            param("base_branch", ParamType::String, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Create a branch called {}.", get_str_or(c, "branch_name", "feature/new"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Make a new git branch: {}.", get_str_or(c, "branch_name", "feature/new"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Cut a branch named {} off {}.", get_str_or(c, "branch_name", "feature/new"), get_str_or(c, "base_branch", "main"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("I need a new branch, {}.", get_str_or(c, "branch_name", "feature/new"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Branch off {} into {}.", get_str_or(c, "base_branch", "main"), get_str_or(c, "branch_name", "feature/new"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Set up {} as a new branch.", get_str_or(c, "branch_name", "feature/new"))
        }),
    ];
    (schema, templates)
}

fn deploy_service() -> ToolDef {
    let schema = ToolSchema::new(
        "deploy_service",
        "Deploy a service to an environment.",
        vec![
            param("service_name", ParamType::String, true),
            param("environment", enum_of(&["dev", "staging", "production"]), true),
            param("version", ParamType::String, false),
            param("rollback_on_failure", ParamType::Boolean, false),
            param("replicas", ParamType::Integer, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Deploy {} to {}.",
                get_str_or(c, "service_name", "the api"),
                get_str_or(c, "environment", "staging")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Ship {} to {}.", get_str_or(c, "service_name", "the api"), get_str_or(c, "environment", "staging"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Push {} out to {}.", get_str_or(c, "service_name", "the api"), get_str_or(c, "environment", "staging"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "Release version {} of {} to {}.",
                get_str_or(c, "version", "latest"),
                get_str_or(c, "service_name", "the api"),
                get_str_or(c, "environment", "staging")
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Roll out {} on {}.", get_str_or(c, "service_name", "the api"), get_str_or(c, "environment", "staging"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Go ahead and deploy {} to the {} environment.", get_str_or(c, "service_name", "the api"), get_str_or(c, "environment", "staging"))
        }),
    ];
    (schema, templates)
}

fn check_build_status() -> ToolDef {
    let schema = ToolSchema::new(
        "check_build_status",
        "Check the status of a CI build.",
        vec![param("build_id", ParamType::String, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Is build {} passing?", get_str_or(c, "build_id", "#123"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Check the status of build {}.", get_str_or(c, "build_id", "#123"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Did build {} succeed?", get_str_or(c, "build_id", "#123"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("What's happening with CI run {}?", get_str_or(c, "build_id", "#123"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Look up build {}.", get_str_or(c, "build_id", "#123"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Give me the status of {}.", get_str_or(c, "build_id", "#123"))
        }),
    ];
    (schema, templates)
}

fn create_ticket() -> ToolDef {
    let schema = ToolSchema::new(
        "create_ticket",
        "Create a tracking ticket/issue.",
        vec![
            param("title", ParamType::String, true),
            param("description", ParamType::String, false),
            param(
                "priority",
                enum_of(&["low", "medium", "high", "critical"]),
                false,
            ),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Open a ticket for {}.", get_str_or(c, "title", "the bug"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("File an issue: {}.", get_str_or(c, "title", "the bug"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Create a ticket titled {}.", get_str_or(c, "title", "the bug"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Log a {} priority ticket for {}.", get_str_or(c, "priority", "medium"), get_str_or(c, "title", "the bug"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("We need to track: {}.", get_str_or(c, "title", "the bug"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Report an issue about {}.", get_str_or(c, "title", "the bug"))
        }),
    ];
    (schema, templates)
}

// ---------------------------------------------------------------------
// Misc / knowledge
// ---------------------------------------------------------------------

fn search_web() -> ToolDef {
    let schema = ToolSchema::new(
        "search_web",
        "Search the web for a query.",
        vec![
            param("query", ParamType::String, true),
            param("max_results", ParamType::Integer, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Search the web for {}.", get_str_or(c, "query", "something"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Google {}.", get_str_or(c, "query", "something"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Look up {} online.", get_str_or(c, "query", "something"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Find information about {}.", get_str_or(c, "query", "something"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Can you search for {}?", get_str_or(c, "query", "something"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("What does the internet say about {}?", get_str_or(c, "query", "something"))
        }),
    ];
    (schema, templates)
}

fn translate_text() -> ToolDef {
    let schema = ToolSchema::new(
        "translate_text",
        "Translate text into a target language.",
        vec![
            param("text", ParamType::String, true),
            param("target_language", ParamType::String, true),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Translate \"{}\" into {}.",
                get_str_or(c, "text", "hello"),
                get_str_or(c, "target_language", "French")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("How do you say \"{}\" in {}?", get_str_or(c, "text", "hello"), get_str_or(c, "target_language", "French"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Convert \"{}\" to {}.", get_str_or(c, "text", "hello"), get_str_or(c, "target_language", "French"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("What's \"{}\" in {}?", get_str_or(c, "text", "hello"), get_str_or(c, "target_language", "French"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Please translate this to {}: \"{}\"", get_str_or(c, "target_language", "French"), get_str_or(c, "text", "hello"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("I need \"{}\" in {}.", get_str_or(c, "text", "hello"), get_str_or(c, "target_language", "French"))
        }),
    ];
    (schema, templates)
}

fn define_word() -> ToolDef {
    let schema = ToolSchema::new(
        "define_word",
        "Look up the definition of a word.",
        vec![param("word", ParamType::String, true)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("What does {} mean?", get_str_or(c, "word", "ephemeral"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Define {}.", get_str_or(c, "word", "ephemeral"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Look up the word {}.", get_str_or(c, "word", "ephemeral"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("What's the definition of {}?", get_str_or(c, "word", "ephemeral"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("I don't know what {} means.", get_str_or(c, "word", "ephemeral"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Explain the word {} to me.", get_str_or(c, "word", "ephemeral"))
        }),
    ];
    (schema, templates)
}

fn unit_convert() -> ToolDef {
    let schema = ToolSchema::new(
        "unit_convert",
        "Convert a value from one unit to another.",
        vec![
            param("value", ParamType::Number, true),
            param("from_unit", ParamType::String, true),
            param("to_unit", ParamType::String, true),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Convert {} {} to {}.",
                get_num_or(c, "value", 1.0),
                get_str_or(c, "from_unit", "miles"),
                get_str_or(c, "to_unit", "kilometers")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!(
                "How many {} is {} {}?",
                get_str_or(c, "to_unit", "kilometers"),
                get_num_or(c, "value", 1.0),
                get_str_or(c, "from_unit", "miles")
            )
        }),
        PhrasingTemplate::new(2, |c| {
            format!(
                "What's {} {} in {}?",
                get_num_or(c, "value", 1.0),
                get_str_or(c, "from_unit", "miles"),
                get_str_or(c, "to_unit", "kilometers")
            )
        }),
        PhrasingTemplate::new(3, |c| {
            format!(
                "{} {} to {}, please.",
                get_num_or(c, "value", 1.0),
                get_str_or(c, "from_unit", "miles"),
                get_str_or(c, "to_unit", "kilometers")
            )
        }),
        PhrasingTemplate::new(4, |c| {
            format!(
                "Change {} {} into {}.",
                get_num_or(c, "value", 1.0),
                get_str_or(c, "from_unit", "miles"),
                get_str_or(c, "to_unit", "kilometers")
            )
        }),
        PhrasingTemplate::new(5, |c| {
            format!(
                "I have {} {}, convert to {}.",
                get_num_or(c, "value", 1.0),
                get_str_or(c, "from_unit", "miles"),
                get_str_or(c, "to_unit", "kilometers")
            )
        }),
    ];
    (schema, templates)
}

fn get_news_headlines() -> ToolDef {
    let schema = ToolSchema::new(
        "get_news_headlines",
        "Get the latest news headlines for a topic.",
        vec![
            param("topic", ParamType::String, false),
            param("max_results", ParamType::Integer, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("What's the news on {}?", get_str_or(c, "topic", "today"))
        }),
        PhrasingTemplate::new(1, |_c| "Give me the latest headlines.".to_string()),
        PhrasingTemplate::new(2, |c| {
            format!("Any news about {}?", get_str_or(c, "topic", "today"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Show me {} news.", get_str_or(c, "topic", "today"))
        }),
        PhrasingTemplate::new(4, |_c| "What's happening in the world today?".to_string()),
        PhrasingTemplate::new(5, |c| {
            format!("Catch me up on {}.", get_str_or(c, "topic", "today"))
        }),
    ];
    (schema, templates)
}

fn search_recipes() -> ToolDef {
    let schema = ToolSchema::new(
        "search_recipes",
        "Search for recipes matching an ingredient or dish.",
        vec![
            param("query", ParamType::String, true),
            param("max_time_minutes", ParamType::Integer, false),
            param("vegetarian", ParamType::Boolean, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!("Find a recipe for {}.", get_str_or(c, "query", "pasta"))
        }),
        PhrasingTemplate::new(1, |c| {
            format!("What can I make with {}?", get_str_or(c, "query", "pasta"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("Search for {} recipes.", get_str_or(c, "query", "pasta"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("I want to cook {} tonight.", get_str_or(c, "query", "pasta"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Give me a recipe using {}.", get_str_or(c, "query", "pasta"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("How do I make {}?", get_str_or(c, "query", "pasta"))
        }),
    ];
    (schema, templates)
}

fn book_ride() -> ToolDef {
    let schema = ToolSchema::new(
        "book_ride",
        "Book a rideshare from pickup to dropoff.",
        vec![
            param("pickup", ParamType::String, true),
            param("dropoff", ParamType::String, true),
            param("ride_type", enum_of(&["standard", "premium", "shared"]), false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Get me a ride from {} to {}.",
                get_str_or(c, "pickup", "home"),
                get_str_or(c, "dropoff", "the airport")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Book a car from {} to {}.", get_str_or(c, "pickup", "home"), get_str_or(c, "dropoff", "the airport"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("I need a lift from {} to {}.", get_str_or(c, "pickup", "home"), get_str_or(c, "dropoff", "the airport"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Call a ride to {}.", get_str_or(c, "dropoff", "the airport"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Order a car pickup at {}, heading to {}.", get_str_or(c, "pickup", "home"), get_str_or(c, "dropoff", "the airport"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Can you book me a rideshare to {}?", get_str_or(c, "dropoff", "the airport"))
        }),
    ];
    (schema, templates)
}

fn order_food() -> ToolDef {
    let schema = ToolSchema::new(
        "order_food",
        "Order food for delivery from a restaurant.",
        vec![
            param("restaurant", ParamType::String, true),
            param("items", ParamType::String, true),
            param("delivery_address", ParamType::String, false),
        ],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |c| {
            format!(
                "Order {} from {}.",
                get_str_or(c, "items", "a pizza"),
                get_str_or(c, "restaurant", "the usual place")
            )
        }),
        PhrasingTemplate::new(1, |c| {
            format!("Get me {} delivered from {}.", get_str_or(c, "items", "a pizza"), get_str_or(c, "restaurant", "the usual place"))
        }),
        PhrasingTemplate::new(2, |c| {
            format!("I'm hungry, order {} from {}.", get_str_or(c, "items", "a pizza"), get_str_or(c, "restaurant", "the usual place"))
        }),
        PhrasingTemplate::new(3, |c| {
            format!("Place an order for {} at {}.", get_str_or(c, "items", "a pizza"), get_str_or(c, "restaurant", "the usual place"))
        }),
        PhrasingTemplate::new(4, |c| {
            format!("Send over {} from {}.", get_str_or(c, "items", "a pizza"), get_str_or(c, "restaurant", "the usual place"))
        }),
        PhrasingTemplate::new(5, |c| {
            format!("Can you order food, {} from {}?", get_str_or(c, "items", "a pizza"), get_str_or(c, "restaurant", "the usual place"))
        }),
    ];
    (schema, templates)
}

fn flip_coin() -> ToolDef {
    let schema = ToolSchema::new(
        "flip_coin",
        "Flip a virtual coin and return heads or tails.",
        vec![param("count", ParamType::Integer, false)],
    );
    let templates: Vec<PhrasingTemplate> = vec![
        PhrasingTemplate::new(0, |_c| "Flip a coin.".to_string()),
        PhrasingTemplate::new(1, |_c| "Heads or tails?".to_string()),
        PhrasingTemplate::new(2, |c| {
            format!("Flip {} coins.", get_num_or(c, "count", 1.0) as i64)
        }),
        PhrasingTemplate::new(3, |_c| "Can you toss a coin for me?".to_string()),
        PhrasingTemplate::new(4, |_c| "Let's decide with a coin flip.".to_string()),
        PhrasingTemplate::new(5, |_c| "I need a random heads-or-tails call.".to_string()),
    ];
    (schema, templates)
}
