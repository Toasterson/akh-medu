//! akheventd Anima source (Phase 53g PR 6, ADR 113 decision 10; plan
//! `docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md`, Anima WI
//! `01a0d569-1cf1` item 5).
//!
//! Replaces the Roman observer's Anima scan and `laya_controller`'s decision
//! queue. Each poll:
//!
//! 1. reuses (or, every `projects_refresh_secs`, re-lists) the projects that
//!    match `project_filter`, and visits `projects_per_tick` of them round
//!    robin;
//! 2. lists each project's work items and plans with cursor pagination,
//!    bounded by `max_pages` (a repeated cursor or a longer listing fails the
//!    pass rather than silently truncating it);
//! 3. projects every entity onto the configured `fields` and, for work items
//!    matching `hydrate_parked.when`, overlays the detail read plus three
//!    hydrated fields: `park_episode`, `live_sessions` and
//!    `latest_thread_message_id` (see [`hydrated`]);
//! 4. compares the projection digest with the stored one. The first pass in
//!    which every listed entity of a project was recorded is a silent
//!    baseline; afterwards each change is one event whose ID is
//!    `<entity>@blake3(projection, revision)`.
//!
//! Emissions per (entity, park episode) are capped at
//! `max_emissions_per_episode`; later changes in that episode are suppressed
//! and listed as attention for the watchdog. The cap state commits in the
//! same transaction as the event. In `shadow` mode the source records
//! [`ShadowRecord`]s (with the routes that would have delivered) instead of
//! ingesting, and advances projections and caps exactly as in live mode.
//!
//! Missing rows are never deletions (a listing can be capped), and nothing
//! here reads text for meaning: every Anima vocabulary (phases, reason codes,
//! session statuses, sender types) lives in the configured filters.
use crate::{
    config::{AnimaSource, Endpoint, ParkedHydration},
    ingress::Events,
    route_filter::JsonPointer,
    store::{Transaction, key},
    types::anima::{
        AnimaChange, AnimaEntityKind, AnimaSourceStatus, EmissionAttention, ProjectCoverage,
        ShadowRecord, ShadowRoute, hydrated,
    },
    types::*,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, time::Duration};

const PROJECTION_DOMAIN: &[u8] = b"akh-anima-projection/v1\n";
const EVENT_DOMAIN: &[u8] = b"akh-anima-event/v1\n";
const EPISODE_DOMAIN: &[u8] = b"akh-anima-episode/v1\n";
const CONTENT_DOMAIN: &[u8] = b"akh-anima-plan-content/v1\n";
/// Plan field values longer than this are represented only by
/// `content_digest`, keeping events small.
const MAX_INLINE_PLAN_VALUE: usize = 256;

/// Anima MCP tool names and response keys this source reads (the API
/// surface, not vocabulary).
mod anima_api {
    pub const LIST_PROJECTS: &str = "list_projects";
    pub const LIST_WORK_ITEMS: &str = "list_work_items";
    pub const LIST_PLANS: &str = "list_plans";
    pub const WORK_ITEM_DETAIL: &str = "get_work_item_detail";
    pub const WORK_ITEM_FORENSICS: &str = "get_work_item_forensics";
    pub const LIST_SESSIONS: &str = "list_sessions";
    pub const PROJECTS: &str = "projects";
    pub const WORK_ITEMS: &str = "work_items";
    pub const PLANS: &str = "plans";
    pub const WORK_ITEM: &str = "work_item";
    pub const THREAD_MESSAGES: &str = "thread_messages";
    pub const EVENTS: &str = "events";
    pub const SESSIONS: &str = "sessions";
    pub const NEXT_PAGE_TOKEN: &str = "next_page_token";
    pub const TOTAL: &str = "total";
}

/// `anima_entity/<source>/<entity>`: the entity's last recorded state.
pub fn entity_key(source: &str, entity: &str) -> String {
    key("anima_entity", &[source, entity])
}
/// `anima_status/<source>`: the latest [`AnimaSourceStatus`].
pub fn status_key(source: &str) -> String {
    key("anima_status", &[source])
}
/// Prefix of the source's [`ShadowRecord`]s, ordered by record time.
pub fn shadow_prefix(source: &str) -> String {
    key("anima_shadow", &[source])
}
fn shadow_key(source: &str, recorded_at_ms: u64, event_id: &str) -> String {
    let tag = blake3::hash(event_id.as_bytes()).to_hex();
    key(
        "anima_shadow",
        &[source, &format!("{recorded_at_ms:020}-{}", &tag[..16])],
    )
}
fn shadow_count_key(source: &str) -> String {
    key("anima_shadow_count", &[source])
}
fn attention_key(source: &str) -> String {
    key("anima_attention", &[source])
}
fn projects_key(source: &str) -> String {
    key("anima_projects", &[source])
}
fn offset_key(source: &str) -> String {
    key("anima_offset", &[source])
}
fn baseline_key(source: &str, project: &str) -> String {
    key("anima_baseline", &[source, project])
}

/// One entity's last recorded state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityState {
    pub projection: Value,
    pub digest: String,
    /// Digest of the listing part alone: an unchanged listing row need not
    /// be hydrated again before `rehydrate_secs`.
    pub listing_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hydrated_at_ms: Option<u64>,
    pub revision: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub episode: Option<EpisodeState>,
    /// Listed in the source's attention map (the episode hit the cap).
    #[serde(default)]
    pub attention: bool,
}

/// Emission accounting of one park episode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeState {
    pub id: String,
    pub emitted: u32,
    pub suppressed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectRef {
    id: String,
    name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProjectsCache {
    at_ms: u64,
    projects: Vec<ProjectRef>,
}

/// What one poll did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PollReport {
    /// Projects visited, in order.
    pub projects: Vec<String>,
    /// Projects whose baseline completed in this poll.
    pub baselined: Vec<String>,
    /// Ingested event IDs.
    pub emitted: Vec<String>,
    /// Shadow-recorded event IDs.
    pub shadowed: Vec<String>,
    pub suppressed: u64,
    pub hydrated: u64,
    pub deferred: u64,
    coverage: BTreeMap<String, ProjectCoverage>,
}

/// Poll `source` forever (spawned by [`Events::sources`]).
pub async fn run(service: Events, source: AnimaSource) {
    let interval = Duration::from_secs(source.interval_secs);
    loop {
        match poll(&service, &source).await {
            Ok(report) if !(report.emitted.is_empty() && report.shadowed.is_empty()) => {
                tracing::info!(
                    source = %source.source,
                    emitted = report.emitted.len(),
                    shadowed = report.shadowed.len(),
                    suppressed = report.suppressed,
                    "anima changes recorded"
                )
            }
            Ok(_) => {}
            Err(error) => tracing::warn!(
                source = %source.source,
                code = %error.code,
                message = %error.message,
                "anima poll deferred; projections are kept"
            ),
        }
        tokio::time::sleep(interval).await;
    }
}

/// Scan once and record the outcome in `anima_status/<source>`.
pub async fn poll(service: &Events, source: &AnimaSource) -> Result<PollReport> {
    let endpoint = service.config.endpoint(&source.endpoint)?.clone();
    let reader = Reader {
        endpoint: &endpoint,
        source: &source.source,
    };
    let mut report = PollReport::default();
    let outcome = scan(service, source, &reader, &mut report).await;
    record_status(service, source, &report, outcome.as_ref().err())?;
    outcome.map(|()| report)
}

/// The source's recorded status, if it has polled.
pub fn status(service: &Events, source: &str) -> Result<Option<AnimaSourceStatus>> {
    service.store.get(&status_key(source))
}

/// The source's shadow records, oldest first; `since_ms` keeps later ones.
pub fn shadow_records(
    service: &Events,
    source: &str,
    since_ms: Option<u64>,
    limit: usize,
) -> Result<Vec<ShadowRecord>> {
    let records: Vec<ShadowRecord> = service.store.scan(&shadow_prefix(source))?;
    let records: Vec<ShadowRecord> = records
        .into_iter()
        .filter(|r| since_ms.is_none_or(|since| r.recorded_at_ms > since))
        .collect();
    let skip = records.len().saturating_sub(limit);
    Ok(records.into_iter().skip(skip).collect())
}

fn record_status(
    service: &Events,
    source: &AnimaSource,
    report: &PollReport,
    error: Option<&Fault>,
) -> Result<()> {
    let k = status_key(&source.source);
    service.store.transaction(|tx| {
        let previous: Option<AnimaSourceStatus> = tx.get(&k)?;
        let attention: BTreeMap<String, EmissionAttention> =
            tx.get(&attention_key(&source.source))?.unwrap_or_default();
        let projects: Option<ProjectsCache> = tx.get(&projects_key(&source.source))?;
        let now = now_ms();
        let mut counters = previous
            .as_ref()
            .map(|s| s.counters.clone())
            .unwrap_or_default();
        counters.emitted += report.emitted.len() as u64;
        counters.shadowed += report.shadowed.len() as u64;
        counters.suppressed += report.suppressed;
        counters.hydrated += report.hydrated;
        counters.hydration_deferred += report.deferred;
        let mut coverage = previous
            .as_ref()
            .map(|s| s.coverage.clone())
            .unwrap_or_default();
        coverage.extend(report.coverage.clone());
        let status = AnimaSourceStatus {
            source: source.source.clone(),
            shadow: source.shadow,
            ok: error.is_none(),
            at_ms: now,
            last_ok_at_ms: if error.is_none() {
                Some(now)
            } else {
                previous.as_ref().and_then(|s| s.last_ok_at_ms)
            },
            error: error.cloned(),
            projects: projects.map_or(0, |p| p.projects.len() as u64),
            coverage,
            counters,
            attention: attention.into_values().collect(),
        };
        tx.put(&k, &status)
    })
}

/// Reads through the Anima MCP endpoint; every call is a fresh bounded
/// exchange (`crate::mcp::call`).
struct Reader<'a> {
    endpoint: &'a Endpoint,
    source: &'a str,
}

impl Reader<'_> {
    async fn call(&self, tool: &str, arguments: Value) -> Result<Value> {
        let id = format!("{}:{}", self.source, new_id());
        let result = crate::mcp::call(self.endpoint, &id, tool, arguments).await?;
        if !result.success {
            return Err(Fault::new(
                "anima_unavailable",
                format!("Anima refused {tool}; the pass is retried next interval"),
            ));
        }
        tool_payload(result.data.as_ref()).ok_or_else(|| {
            Fault::new(
                "anima_unavailable",
                format!("Anima {tool} returned no JSON payload"),
            )
        })
    }
}

/// The JSON payload of an MCP tool result: `structuredContent`, else the
/// first text content parsed as JSON.
fn tool_payload(result: Option<&Value>) -> Option<Value> {
    let result = result?;
    if let Some(value) = result.get("structuredContent").filter(|v| !v.is_null()) {
        return Some(value.clone());
    }
    result
        .get("content")?
        .as_array()?
        .iter()
        .find(|c| c.get("type").and_then(Value::as_str) == Some("text"))
        .and_then(|c| c.get("text")?.as_str())
        .and_then(|text| serde_json::from_str(text).ok())
}

/// A bounded cursor-paginated listing.
struct Listing {
    rows: Vec<Value>,
    pages: u16,
    total: Option<u64>,
    complete: bool,
}

async fn list_paged(
    reader: &Reader<'_>,
    tool: &str,
    base: Value,
    rows_key: &str,
    page_size: Option<u16>,
    max_pages: u16,
) -> Result<Listing> {
    let mut rows = Vec::new();
    let mut ids = std::collections::HashSet::new();
    let mut seen = std::collections::HashSet::new();
    let mut cursor: Option<String> = None;
    let mut pages = 0u16;
    let mut total = None;
    loop {
        let mut arguments = base.clone();
        if let Some(size) = page_size {
            arguments["page_size"] = json!(size);
        }
        if let Some(token) = &cursor {
            arguments["page_token"] = json!(token);
        }
        let page = reader.call(tool, arguments).await?;
        let batch = page
            .get(rows_key)
            .and_then(Value::as_array)
            .ok_or_else(|| {
                Fault::new(
                    "anima_unavailable",
                    format!("Anima {tool} response has no `{rows_key}` array"),
                )
            })?;
        if pages == 0 {
            total = page.get(anima_api::TOTAL).and_then(Value::as_u64);
        }
        pages += 1;
        for row in batch {
            match row.get("id").and_then(Value::as_str) {
                // A later page wins for an ID seen twice.
                Some(id) if !ids.insert(id.to_string()) => {
                    if let Some(slot) = rows
                        .iter_mut()
                        .find(|r: &&mut Value| r.get("id").and_then(Value::as_str) == Some(id))
                    {
                        *slot = row.clone();
                    }
                }
                Some(_) => rows.push(row.clone()),
                None => {}
            }
        }
        cursor = page
            .get(anima_api::NEXT_PAGE_TOKEN)
            .and_then(Value::as_str)
            .filter(|token| !token.is_empty())
            .map(String::from);
        let Some(token) = &cursor else { break };
        if !seen.insert(token.clone()) {
            return Err(Fault::new(
                "anima_pagination",
                format!("Anima {tool} repeated a page cursor; the pass stops instead of looping"),
            ));
        }
        if pages >= max_pages {
            return Err(Fault::new(
                "anima_pagination",
                format!(
                    "Anima {tool} needs more than max_pages = {max_pages} pages; raise it or narrow the project"
                ),
            ));
        }
    }
    let observed = rows.len() as u64;
    Ok(Listing {
        complete: total.is_none_or(|t| t <= observed),
        rows,
        pages,
        total,
    })
}

/// Recursively key-sorted JSON bytes (independent of serde_json's map order).
fn canonical_bytes(value: &Value) -> Vec<u8> {
    fn write(value: &Value, out: &mut Vec<u8>) {
        match value {
            Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by(|a, b| a.0.cmp(b.0));
                out.push(b'{');
                for (index, (k, v)) in entries.into_iter().enumerate() {
                    if index > 0 {
                        out.push(b',');
                    }
                    out.extend(serde_json::to_vec(k).unwrap_or_default());
                    out.push(b':');
                    write(v, out);
                }
                out.push(b'}');
            }
            Value::Array(items) => {
                out.push(b'[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(b',');
                    }
                    write(item, out);
                }
                out.push(b']');
            }
            other => out.extend(serde_json::to_vec(other).unwrap_or_default()),
        }
    }
    let mut out = Vec::new();
    write(value, &mut out);
    out
}

fn hex_digest(domain: &[u8], parts: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for part in parts {
        hasher.update(&(part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hasher.finalize().to_hex().to_string()
}

/// Hex blake3 digest of a projection's canonical bytes.
pub fn projection_digest(projection: &Value) -> String {
    hex_digest(PROJECTION_DOMAIN, &[&canonical_bytes(projection)])
}

/// The event ID of `entity`'s `revision`-th projection: the entity, `@`, and
/// blake3 over the projection and the revision (so a projection that recurs
/// gets a fresh ID instead of being refused as a reused one).
pub fn event_id(entity: &str, projection: &Value, revision: u64) -> String {
    let digest = hex_digest(
        EVENT_DOMAIN,
        &[
            entity.as_bytes(),
            &canonical_bytes(projection),
            &revision.to_be_bytes(),
        ],
    );
    format!("{entity}@{digest}")
}

fn episode_digest(entity: &str, anchor: &Value) -> String {
    hex_digest(
        EPISODE_DOMAIN,
        &[entity.as_bytes(), &canonical_bytes(anchor)],
    )[..32]
        .to_string()
}

fn select(row: &Value, fields: &[String]) -> Map<String, Value> {
    let mut out = Map::new();
    if let Some(id) = row.get("id") {
        out.insert("id".into(), id.clone());
    }
    for field in fields {
        if let Some(value) = row.get(field) {
            out.insert(field.clone(), value.clone());
        }
    }
    out
}

/// A plan's projection: its `id`, the digest of its `plan_fields`, and those
/// of them that are small scalars.
fn plan_projection(row: &Value, fields: &[String]) -> Value {
    let selected = select(row, fields);
    let mut projection = Map::new();
    projection.insert(
        "content_digest".into(),
        json!(hex_digest(
            CONTENT_DOMAIN,
            &[&canonical_bytes(&Value::Object(selected.clone()))]
        )),
    );
    for (field, value) in selected {
        let inline = match &value {
            Value::String(text) => text.len() <= MAX_INLINE_PLAN_VALUE,
            Value::Array(_) | Value::Object(_) => false,
            _ => true,
        };
        if inline {
            projection.insert(field, value);
        }
    }
    Value::Object(projection)
}

/// A total order over optional JSON sort keys: numbers, then strings.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
enum OrderKey {
    Missing,
    Number(f64),
    Text(String),
}

fn order_key(item: &Value, pointer: Option<&JsonPointer>) -> OrderKey {
    match pointer.and_then(|p| p.resolve(item)) {
        Some(Value::Number(n)) => n.as_f64().map_or(OrderKey::Missing, OrderKey::Number),
        Some(Value::String(s)) => OrderKey::Text(s.clone()),
        _ => OrderKey::Missing,
    }
}

/// The greatest item by `pointer` (ties and unordered items: the later one).
fn latest<'a>(
    items: impl Iterator<Item = &'a Value>,
    pointer: Option<&JsonPointer>,
) -> Option<&'a Value> {
    items.fold(None, |best: Option<&Value>, item| match best {
        Some(current)
            if order_key(item, pointer)
                .partial_cmp(&order_key(current, pointer))
                .is_some_and(|o| o.is_lt()) =>
        {
            Some(current)
        }
        _ => Some(item),
    })
}

/// Read the work item's detail, forensics and sessions and overlay them on
/// the listing projection.
async fn hydrate(
    reader: &Reader<'_>,
    entity: &str,
    id: &str,
    listing: Map<String, Value>,
    source: &AnimaSource,
    hydration: &ParkedHydration,
) -> Result<Value> {
    let detail = reader
        .call(
            anima_api::WORK_ITEM_DETAIL,
            json!({"id": id, "message_page_size": 100, "change_page_size": 100}),
        )
        .await?;
    let work_item = detail
        .get(anima_api::WORK_ITEM)
        .filter(|v| v.is_object())
        .ok_or_else(|| {
            Fault::new(
                "anima_unavailable",
                "work-item detail has no `work_item` object",
            )
        })?;
    let forensics = reader
        .call(
            anima_api::WORK_ITEM_FORENSICS,
            json!({"work_item_id": id, "max_events": hydration.max_forensic_events}),
        )
        .await?;
    let sessions = reader
        .call(anima_api::LIST_SESSIONS, json!({"work_item_id": id}))
        .await?;

    let mut projection = listing;
    for field in &source.fields {
        if let Some(value) = work_item.get(field) {
            projection.insert(field.clone(), value.clone());
        }
    }
    let events = forensics
        .get(anima_api::EVENTS)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let anchor = latest(
        events.iter().filter(|e| hydration.park_event.matches(e)),
        hydration.park_event_order.as_ref(),
    );
    let episode = match anchor {
        Some(event) => episode_digest(entity, &json!({"park_event": event})),
        None => episode_digest(
            entity,
            &json!({"episode_fields": Value::Object(select(work_item, &hydration.episode_fields))}),
        ),
    };
    projection.insert(hydrated::PARK_EPISODE.into(), json!(episode));
    let sessions = match &sessions {
        Value::Array(items) => items.as_slice(),
        other => other
            .get(anima_api::SESSIONS)
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default(),
    };
    let live = sessions
        .iter()
        .filter(|s| hydration.live_session.matches(s))
        .count();
    projection.insert(hydrated::LIVE_SESSIONS.into(), json!(live));
    let messages = detail
        .get(anima_api::THREAD_MESSAGES)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let newest = latest(
        messages.iter().filter(|m| {
            hydration
                .thread_message
                .as_ref()
                .is_none_or(|filter| filter.matches(m))
        }),
        hydration.thread_message_order.as_ref(),
    );
    if let Some(message_id) = newest.and_then(|m| m.get("id")).filter(|v| !v.is_null()) {
        projection.insert(
            hydrated::LATEST_THREAD_MESSAGE_ID.into(),
            message_id.clone(),
        );
    }
    Ok(Value::Object(projection))
}

async fn projects(
    service: &Events,
    source: &AnimaSource,
    reader: &Reader<'_>,
    now: u64,
) -> Result<Vec<ProjectRef>> {
    let k = projects_key(&source.source);
    if let Some(cache) = service.store.get::<ProjectsCache>(&k)?
        && !cache.projects.is_empty()
        && now.saturating_sub(cache.at_ms) < source.projects_refresh_secs * 1000
    {
        return Ok(cache.projects);
    }
    let listing = list_paged(
        reader,
        anima_api::LIST_PROJECTS,
        json!({}),
        anima_api::PROJECTS,
        Some(source.page_size),
        source.max_pages,
    )
    .await?;
    let projects: Vec<ProjectRef> = listing
        .rows
        .iter()
        .filter(|row| {
            source
                .project_filter
                .as_ref()
                .is_none_or(|filter| filter.matches(row))
        })
        .filter_map(|row| {
            Some(ProjectRef {
                id: row.get("id")?.as_str()?.to_string(),
                name: row
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            })
        })
        .collect();
    service.store.transaction(|tx| {
        tx.put(
            &k,
            &ProjectsCache {
                at_ms: now,
                projects: projects.clone(),
            },
        )
    })?;
    Ok(projects)
}

async fn scan(
    service: &Events,
    source: &AnimaSource,
    reader: &Reader<'_>,
    report: &mut PollReport,
) -> Result<()> {
    let now = now_ms();
    let projects = projects(service, source, reader, now).await?;
    if projects.is_empty() {
        return Ok(());
    }
    let offset: u64 = service.store.get(&offset_key(&source.source))?.unwrap_or(0);
    let mut budget = source.hydrate_parked.as_ref().map_or(0, |h| h.max_per_tick);
    let visits = usize::from(source.projects_per_tick).min(projects.len());
    for step in 0..visits {
        let index = (offset as usize + step) % projects.len();
        let project = &projects[index];
        report.projects.push(project.id.clone());
        scan_project(service, source, reader, project, &mut budget, report).await?;
        let next = ((index + 1) % projects.len()) as u64;
        service
            .store
            .transaction(|tx| tx.put(&offset_key(&source.source), &next))?;
    }
    Ok(())
}

/// Context of one project pass.
struct ProjectPass<'a> {
    service: &'a Events,
    source: &'a AnimaSource,
    project_id: &'a str,
    baselined: bool,
    /// Silent state updates, committed with the project's coverage.
    silent: Vec<(String, EntityState)>,
    /// An entity could not be recorded (hydration deferred): the baseline
    /// is not complete yet.
    incomplete: bool,
}

async fn scan_project(
    service: &Events,
    source: &AnimaSource,
    reader: &Reader<'_>,
    project: &ProjectRef,
    budget: &mut u16,
    report: &mut PollReport,
) -> Result<()> {
    let items = list_paged(
        reader,
        anima_api::LIST_WORK_ITEMS,
        json!({"project_id": project.id}),
        anima_api::WORK_ITEMS,
        Some(source.page_size),
        source.max_pages,
    )
    .await?;
    let plans = list_paged(
        reader,
        anima_api::LIST_PLANS,
        json!({"project_id": project.id}),
        anima_api::PLANS,
        None,
        source.max_pages,
    )
    .await?;
    let bkey = baseline_key(&source.source, &project.id);
    let mut pass = ProjectPass {
        service,
        source,
        project_id: &project.id,
        baselined: service.store.get::<bool>(&bkey)?.unwrap_or(false),
        silent: Vec::new(),
        incomplete: false,
    };
    for row in &items.rows {
        let Some(id) = row.get("id").and_then(Value::as_str) else {
            continue;
        };
        let entity = AnimaEntityKind::WorkItem.entity(id);
        let listing = select(row, &source.fields);
        let listing_digest = projection_digest(&Value::Object(listing.clone()));
        let stored: Option<EntityState> =
            service.store.get(&entity_key(&source.source, &entity))?;
        let listing_value = Value::Object(listing.clone());
        let (projection, hydrated_at) = match &source.hydrate_parked {
            Some(hydration) if hydration.when.matches(&listing_value) => {
                let now = now_ms();
                let fresh = stored.as_ref().filter(|s| {
                    s.listing_digest == listing_digest
                        && s.hydrated_at_ms.is_some_and(|at| {
                            now.saturating_sub(at) < hydration.rehydrate_secs * 1000
                        })
                });
                if let Some(state) = fresh {
                    (state.projection.clone(), state.hydrated_at_ms)
                } else if *budget == 0 {
                    report.deferred += 1;
                    pass.incomplete = true;
                    continue;
                } else {
                    *budget -= 1;
                    match hydrate(reader, &entity, id, listing, source, hydration).await {
                        Ok(projection) => {
                            report.hydrated += 1;
                            (projection, Some(now))
                        }
                        Err(error) => {
                            tracing::warn!(source=%source.source,entity=%entity,code=%error.code,
                                "work-item hydration deferred; the previous projection is kept");
                            report.deferred += 1;
                            pass.incomplete = true;
                            continue;
                        }
                    }
                }
            }
            _ => (listing_value, None),
        };
        pass.observe(
            AnimaEntityKind::WorkItem,
            entity,
            projection,
            listing_digest,
            hydrated_at,
            stored,
            report,
        )?;
    }
    for row in &plans.rows {
        let Some(id) = row.get("id").and_then(Value::as_str) else {
            continue;
        };
        let entity = AnimaEntityKind::Plan.entity(id);
        let projection = plan_projection(row, &source.plan_fields);
        let digest = projection_digest(&projection);
        let stored = service.store.get(&entity_key(&source.source, &entity))?;
        pass.observe(
            AnimaEntityKind::Plan,
            entity,
            projection,
            digest,
            None,
            stored,
            report,
        )?;
    }
    let baselined = pass.baselined || !pass.incomplete;
    if baselined && !pass.baselined {
        report.baselined.push(project.id.clone());
    }
    let coverage = ProjectCoverage {
        name: project.name.clone(),
        at_ms: now_ms(),
        observed: items.rows.len() as u64,
        total: items.total,
        pages: items.pages,
        plans: plans.rows.len() as u64,
        complete: items.complete,
        baselined,
    };
    let silent = std::mem::take(&mut pass.silent);
    service.store.transaction(|tx| {
        for (entity, state) in &silent {
            tx.put(&entity_key(&source.source, entity), state)?;
        }
        tx.put(&bkey, &baselined)
    })?;
    report.coverage.insert(project.id.clone(), coverage);
    Ok(())
}

impl ProjectPass<'_> {
    #[allow(clippy::too_many_arguments)]
    fn observe(
        &mut self,
        kind: AnimaEntityKind,
        entity: String,
        projection: Value,
        listing_digest: String,
        hydrated_at_ms: Option<u64>,
        stored: Option<EntityState>,
        report: &mut PollReport,
    ) -> Result<()> {
        let digest = projection_digest(&projection);
        let episode_id = projection
            .get(hydrated::PARK_EPISODE)
            .and_then(Value::as_str)
            .map(String::from);
        match stored {
            None if !self.baselined => {
                self.silent.push((
                    entity,
                    EntityState {
                        projection,
                        digest,
                        listing_digest,
                        hydrated_at_ms,
                        revision: 0,
                        episode: episode_id.map(|id| EpisodeState {
                            id,
                            emitted: 0,
                            suppressed: 0,
                        }),
                        attention: false,
                    },
                ));
                Ok(())
            }
            Some(state) if state.digest == digest => {
                if state.hydrated_at_ms != hydrated_at_ms || state.listing_digest != listing_digest
                {
                    self.silent.push((
                        entity,
                        EntityState {
                            hydrated_at_ms,
                            listing_digest,
                            ..state
                        },
                    ));
                }
                Ok(())
            }
            previous => self.emit(
                kind,
                entity,
                projection,
                digest,
                listing_digest,
                hydrated_at_ms,
                episode_id,
                previous,
                report,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit(
        &mut self,
        kind: AnimaEntityKind,
        entity: String,
        projection: Value,
        digest: String,
        listing_digest: String,
        hydrated_at_ms: Option<u64>,
        episode_id: Option<String>,
        previous: Option<EntityState>,
        report: &mut PollReport,
    ) -> Result<()> {
        let source = self.source;
        let store = &self.service.store;
        let k = entity_key(&source.source, &entity);
        let revision = previous.as_ref().map_or(0, |p| p.revision + 1);
        let had_attention = previous.as_ref().is_some_and(|p| p.attention);
        let mut episode = match (
            episode_id,
            previous.as_ref().and_then(|p| p.episode.clone()),
        ) {
            (Some(id), Some(current)) if current.id == id => Some(current),
            (Some(id), _) => Some(EpisodeState {
                id,
                emitted: 0,
                suppressed: 0,
            }),
            (None, _) => None,
        };
        let cap = source.max_emissions_per_episode;
        let state = |episode: Option<EpisodeState>, attention: bool| EntityState {
            projection: projection.clone(),
            digest: digest.clone(),
            listing_digest: listing_digest.clone(),
            hydrated_at_ms,
            revision,
            episode,
            attention,
        };

        if let Some(current) = episode.as_mut()
            && current.emitted >= cap
        {
            current.suppressed += 1;
            let entry = EmissionAttention {
                entity: entity.clone(),
                park_episode: current.id.clone(),
                emitted: current.emitted,
                suppressed: current.suppressed,
            };
            let next = state(episode, true);
            store.transaction(|tx| {
                tx.put(&k, &next)?;
                update_attention(tx, &source.source, &entity, Some(entry))
            })?;
            report.suppressed += 1;
            return Ok(());
        }
        if let Some(current) = episode.as_mut() {
            current.emitted += 1;
        }
        let change = AnimaChange {
            kind,
            entity: entity.clone(),
            project_id: self.project_id.to_string(),
            previous: previous.map(|p| p.projection),
            current: projection.clone(),
            projection_digest: digest.clone(),
            revision,
            park_episode: episode.as_ref().map(|e| e.id.clone()),
            emission: episode.as_ref().map(|e| e.emitted),
            max_emissions: episode.as_ref().map(|_| cap),
            observed_at_utc: utc(now_ms()),
        };
        let event = EventEnvelope {
            schema_version: SCHEMA_VERSION,
            id: event_id(&entity, &projection, revision),
            source: source.source.clone(),
            event_type: match kind {
                AnimaEntityKind::WorkItem => source.work_item_event_type.clone(),
                AnimaEntityKind::Plan => source.plan_event_type.clone(),
            },
            occurred_at: now_ms(),
            subject: Some(entity.clone()),
            correlation_id: change.park_episode.clone(),
            data: serde_json::to_value(&change).map_err(|e| Fault::invalid(e.to_string()))?,
        };
        let id = event.id.clone();
        let next = state(episode, false);
        let also = |tx: &mut Transaction<'_>| {
            tx.put(&k, &next)?;
            if had_attention {
                update_attention(tx, &source.source, &entity, None)?;
            }
            Ok(())
        };
        if source.shadow {
            let routes = self
                .service
                .matching_routes(&event)
                .map(|route| {
                    let (akh, automation) = match route.resolved_target() {
                        Ok(crate::config::RouteTarget::Akh { akh, .. }) => (Some(akh), None),
                        Ok(crate::config::RouteTarget::Automation { automation }) => {
                            (None, Some(automation))
                        }
                        Err(_) => (None, None),
                    };
                    ShadowRoute {
                        route: route.id.clone(),
                        akh,
                        automation,
                    }
                })
                .collect();
            let recorded_at_ms = now_ms();
            let record = ShadowRecord {
                source: source.source.clone(),
                recorded_at_ms,
                event,
                routes,
            };
            store.transaction(|tx| {
                also(tx)?;
                tx.put(&shadow_key(&source.source, recorded_at_ms, &id), &record)?;
                prune_shadow(tx, &source.source, source.shadow_retention)
            })?;
            report.shadowed.push(id);
        } else {
            let principal = self.service.source_principal(&source.source);
            self.service.ingest_with(&principal, event, also)?;
            report.emitted.push(id);
        }
        Ok(())
    }
}

fn update_attention(
    tx: &mut Transaction<'_>,
    source: &str,
    entity: &str,
    entry: Option<EmissionAttention>,
) -> Result<()> {
    let k = attention_key(source);
    let mut map: BTreeMap<String, EmissionAttention> = tx.get(&k)?.unwrap_or_default();
    match entry {
        Some(entry) => {
            map.insert(entity.to_string(), entry);
        }
        None => {
            map.remove(entity);
        }
    }
    tx.put(&k, &map)
}

/// Keep at most `retention` shadow records, trimming in batches of a tenth.
fn prune_shadow(tx: &mut Transaction<'_>, source: &str, retention: u32) -> Result<()> {
    let count_key = shadow_count_key(source);
    let count: u64 = tx.get::<u64>(&count_key)?.unwrap_or(0) + 1;
    let retention = u64::from(retention);
    let slack = (retention / 10).max(1);
    if count <= retention + slack {
        return tx.put(&count_key, &count);
    }
    let records: Vec<ShadowRecord> = tx.scan(&shadow_prefix(source))?;
    let excess = records.len().saturating_sub(retention as usize);
    for record in records.iter().take(excess) {
        tx.delete(&shadow_key(source, record.recorded_at_ms, &record.event.id))?;
    }
    tx.put(&count_key, &((records.len() - excess) as u64))
}

fn utc(ms: u64) -> String {
    i64::try_from(ms)
        .ok()
        .and_then(chrono::DateTime::from_timestamp_millis)
        .map(|at| at.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
