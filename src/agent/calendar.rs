//! Calendar & Temporal Reasoning — Phase 13f.
//!
//! Calendar event management, Allen interval algebra for temporal reasoning,
//! iCalendar import, CalDAV sync, scheduling conflict detection, and VSA
//! temporal pattern encoding. Events are KG entities with well-known predicates
//! in the `cal:` and `time:` namespaces.

use std::collections::HashMap;
use std::fmt;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::encode::encode_token;
use crate::vsa::ops::VsaOps;
use crate::vsa::HyperVec;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

/// Errors specific to the calendar subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum CalendarError {
    #[error("calendar event not found: {event_id}")]
    #[diagnostic(
        code(akh::agent::calendar::event_not_found),
        help("Use `cal add` to create a new calendar event.")
    )]
    EventNotFound { event_id: u64 },

    #[error("scheduling conflict: \"{a}\" overlaps with \"{b}\"")]
    #[diagnostic(
        code(akh::agent::calendar::conflict),
        help("Reschedule one of the events to resolve the time overlap.")
    )]
    Conflict { a: String, b: String },

    #[error("iCalendar parse error: {message}")]
    #[diagnostic(
        code(akh::agent::calendar::parse_error),
        help("Ensure the input is a valid RFC 5545 iCalendar document.")
    )]
    ParseError { message: String },

    #[error("CalDAV sync error: {message}")]
    #[diagnostic(
        code(akh::agent::calendar::sync_error),
        help("Check the CalDAV URL, credentials, and network connectivity.")
    )]
    SyncError { message: String },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::calendar::engine),
        help("Engine-level error during calendar operation.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for CalendarError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type CalendarResult<T> = std::result::Result<T, CalendarError>;

// ═══════════════════════════════════════════════════════════════════════
// Allen Interval Algebra
// ═══════════════════════════════════════════════════════════════════════

/// The 13 Allen interval relations between two time intervals [s1,e1) and [s2,e2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AllenRelation {
    /// [s1,e1) entirely before [s2,e2): e1 < s2.
    Before,
    /// [s2,e2) entirely before [s1,e1): e2 < s1.
    After,
    /// e1 == s2 — first interval meets second.
    Meets,
    /// e2 == s1 — second interval meets first.
    MetBy,
    /// s1 < s2 < e1 < e2.
    Overlaps,
    /// s2 < s1 < e2 < e1.
    OverlappedBy,
    /// s2 < s1 and e1 < e2 — first is during second.
    During,
    /// s1 < s2 and e2 < e1 — first contains second.
    Contains,
    /// s1 == s2 and e1 < e2.
    Starts,
    /// s1 == s2 and e2 < e1.
    StartedBy,
    /// e1 == e2 and s2 < s1.
    Finishes,
    /// e1 == e2 and s1 < s2.
    FinishedBy,
    /// s1 == s2 and e1 == e2.
    Equals,
}

impl AllenRelation {
    /// Compute the Allen relation between two intervals.
    ///
    /// Intervals are half-open: [start, end).
    pub fn compute(s1: u64, e1: u64, s2: u64, e2: u64) -> Self {
        if s1 == s2 && e1 == e2 {
            Self::Equals
        } else if s1 == s2 {
            if e1 < e2 { Self::Starts } else { Self::StartedBy }
        } else if e1 == e2 {
            if s1 > s2 { Self::Finishes } else { Self::FinishedBy }
        } else if e1 < s2 {
            Self::Before
        } else if e2 < s1 {
            Self::After
        } else if e1 == s2 {
            Self::Meets
        } else if e2 == s1 {
            Self::MetBy
        } else if s1 < s2 && e1 > s2 && e1 < e2 {
            Self::Overlaps
        } else if s2 < s1 && e2 > s1 && e2 < e1 {
            Self::OverlappedBy
        } else if s2 < s1 && e1 < e2 {
            Self::During
        } else {
            // s1 < s2 && e2 < e1
            Self::Contains
        }
    }

    /// Inverse (converse) relation.
    pub fn inverse(self) -> Self {
        match self {
            Self::Before => Self::After,
            Self::After => Self::Before,
            Self::Meets => Self::MetBy,
            Self::MetBy => Self::Meets,
            Self::Overlaps => Self::OverlappedBy,
            Self::OverlappedBy => Self::Overlaps,
            Self::During => Self::Contains,
            Self::Contains => Self::During,
            Self::Starts => Self::StartedBy,
            Self::StartedBy => Self::Starts,
            Self::Finishes => Self::FinishedBy,
            Self::FinishedBy => Self::Finishes,
            Self::Equals => Self::Equals,
        }
    }

    /// Whether this relation implies temporal overlap.
    pub fn is_overlapping(self) -> bool {
        matches!(
            self,
            Self::Overlaps
                | Self::OverlappedBy
                | Self::During
                | Self::Contains
                | Self::Starts
                | Self::StartedBy
                | Self::Finishes
                | Self::FinishedBy
                | Self::Equals
        )
    }

    /// Stable string label for KG storage.
    pub fn as_label(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::After => "after",
            Self::Meets => "meets",
            Self::MetBy => "met-by",
            Self::Overlaps => "overlaps",
            Self::OverlappedBy => "overlapped-by",
            Self::During => "during",
            Self::Contains => "contains",
            Self::Starts => "starts",
            Self::StartedBy => "started-by",
            Self::Finishes => "finishes",
            Self::FinishedBy => "finished-by",
            Self::Equals => "equals",
        }
    }

    /// Parse from a string label.
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "before" => Some(Self::Before),
            "after" => Some(Self::After),
            "meets" => Some(Self::Meets),
            "met-by" => Some(Self::MetBy),
            "overlaps" => Some(Self::Overlaps),
            "overlapped-by" => Some(Self::OverlappedBy),
            "during" => Some(Self::During),
            "contains" => Some(Self::Contains),
            "starts" => Some(Self::Starts),
            "started-by" => Some(Self::StartedBy),
            "finishes" => Some(Self::Finishes),
            "finished-by" => Some(Self::FinishedBy),
            "equals" => Some(Self::Equals),
            _ => None,
        }
    }
}

impl fmt::Display for AllenRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CalendarEvent
// ═══════════════════════════════════════════════════════════════════════

/// A calendar event stored as a KG entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEvent {
    /// Symbol ID of the event entity in the KG.
    pub symbol_id: SymbolId,
    /// Short summary / title.
    pub summary: String,
    /// Start time (UNIX seconds).
    pub dtstart: u64,
    /// End time (UNIX seconds).
    pub dtend: u64,
    /// Optional location.
    pub location: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Recurrence (reuses PIM Recurrence).
    pub recurrence: Option<super::pim::Recurrence>,
    /// iCalendar UID for dedup during sync.
    pub ical_uid: Option<String>,
    /// Whether the event is confirmed.
    pub confirmed: bool,
}

impl CalendarEvent {
    /// Duration in seconds.
    pub fn duration_secs(&self) -> u64 {
        self.dtend.saturating_sub(self.dtstart)
    }

    /// Whether this event overlaps another.
    pub fn overlaps(&self, other: &CalendarEvent) -> bool {
        AllenRelation::compute(self.dtstart, self.dtend, other.dtstart, other.dtend)
            .is_overlapping()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CalendarPredicates — well-known KG relations
// ═══════════════════════════════════════════════════════════════════════

/// Well-known predicate SymbolIds for calendar metadata and Allen relations.
pub struct CalendarPredicates {
    // Allen relations (time: namespace)
    pub time_before: SymbolId,
    pub time_after: SymbolId,
    pub time_meets: SymbolId,
    pub time_met_by: SymbolId,
    pub time_overlaps: SymbolId,
    pub time_overlapped_by: SymbolId,
    pub time_during: SymbolId,
    pub time_contains: SymbolId,
    pub time_starts: SymbolId,
    pub time_started_by: SymbolId,
    pub time_finishes: SymbolId,
    pub time_finished_by: SymbolId,
    pub time_equals: SymbolId,
    // Calendar metadata (cal: namespace)
    pub cal_dtstart: SymbolId,
    pub cal_dtend: SymbolId,
    pub cal_location: SymbolId,
    pub cal_summary: SymbolId,
    pub cal_conflicts_with: SymbolId,
    pub cal_requires_resource: SymbolId,
}

impl CalendarPredicates {
    /// Resolve or create all calendar predicates.
    pub fn init(engine: &Engine) -> CalendarResult<Self> {
        Ok(Self {
            time_before: engine.resolve_or_create_relation("time:before")?,
            time_after: engine.resolve_or_create_relation("time:after")?,
            time_meets: engine.resolve_or_create_relation("time:meets")?,
            time_met_by: engine.resolve_or_create_relation("time:met-by")?,
            time_overlaps: engine.resolve_or_create_relation("time:overlaps")?,
            time_overlapped_by: engine.resolve_or_create_relation("time:overlapped-by")?,
            time_during: engine.resolve_or_create_relation("time:during")?,
            time_contains: engine.resolve_or_create_relation("time:contains")?,
            time_starts: engine.resolve_or_create_relation("time:starts")?,
            time_started_by: engine.resolve_or_create_relation("time:started-by")?,
            time_finishes: engine.resolve_or_create_relation("time:finishes")?,
            time_finished_by: engine.resolve_or_create_relation("time:finished-by")?,
            time_equals: engine.resolve_or_create_relation("time:equals")?,
            cal_dtstart: engine.resolve_or_create_relation("cal:dtstart")?,
            cal_dtend: engine.resolve_or_create_relation("cal:dtend")?,
            cal_location: engine.resolve_or_create_relation("cal:location")?,
            cal_summary: engine.resolve_or_create_relation("cal:summary")?,
            cal_conflicts_with: engine.resolve_or_create_relation("cal:conflicts-with")?,
            cal_requires_resource: engine.resolve_or_create_relation("cal:requires-resource")?,
        })
    }

    /// Map an Allen relation to its predicate SymbolId.
    pub fn allen_predicate(&self, relation: AllenRelation) -> SymbolId {
        match relation {
            AllenRelation::Before => self.time_before,
            AllenRelation::After => self.time_after,
            AllenRelation::Meets => self.time_meets,
            AllenRelation::MetBy => self.time_met_by,
            AllenRelation::Overlaps => self.time_overlaps,
            AllenRelation::OverlappedBy => self.time_overlapped_by,
            AllenRelation::During => self.time_during,
            AllenRelation::Contains => self.time_contains,
            AllenRelation::Starts => self.time_starts,
            AllenRelation::StartedBy => self.time_started_by,
            AllenRelation::Finishes => self.time_finishes,
            AllenRelation::FinishedBy => self.time_finished_by,
            AllenRelation::Equals => self.time_equals,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CalendarRoleVectors — VSA role vectors for temporal encoding
// ═══════════════════════════════════════════════════════════════════════

/// Deterministic role hypervectors for encoding temporal patterns.
pub struct CalendarRoleVectors {
    pub day_of_week: HyperVec,
    pub time_of_day: HyperVec,
    pub activity_type: HyperVec,
    pub duration: HyperVec,
}

impl CalendarRoleVectors {
    /// Create role vectors via deterministic token encoding.
    pub fn new(ops: &VsaOps) -> Self {
        Self {
            day_of_week: encode_token(ops, "cal-role:day-of-week"),
            time_of_day: encode_token(ops, "cal-role:time-of-day"),
            activity_type: encode_token(ops, "cal-role:activity-type"),
            duration: encode_token(ops, "cal-role:duration"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CalendarManager
// ═══════════════════════════════════════════════════════════════════════

/// Manages calendar events, Allen relations, conflict detection, and VSA encoding.
pub struct CalendarManager {
    events: HashMap<u64, CalendarEvent>,
    predicates: Option<CalendarPredicates>,
    role_vectors: Option<CalendarRoleVectors>,
}

impl Serialize for CalendarManager {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Only serialize events — predicates/role_vectors are re-initialized on restore.
        self.events.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CalendarManager {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let events = HashMap::<u64, CalendarEvent>::deserialize(deserializer)?;
        Ok(Self {
            events,
            predicates: None,
            role_vectors: None,
        })
    }
}

impl Default for CalendarManager {
    fn default() -> Self {
        Self {
            events: HashMap::new(),
            predicates: None,
            role_vectors: None,
        }
    }
}

impl fmt::Debug for CalendarManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CalendarManager")
            .field("event_count", &self.events.len())
            .finish()
    }
}

impl CalendarManager {
    /// Create a new manager, initializing predicates and role vectors.
    pub fn new(engine: &Engine) -> CalendarResult<Self> {
        let predicates = CalendarPredicates::init(engine)?;
        let role_vectors = CalendarRoleVectors::new(engine.ops());
        Ok(Self {
            events: HashMap::new(),
            predicates: Some(predicates),
            role_vectors: Some(role_vectors),
        })
    }

    /// Ensure predicates and role vectors are initialized (for post-deserialization).
    pub fn ensure_init(&mut self, engine: &Engine) -> CalendarResult<()> {
        if self.predicates.is_none() {
            self.predicates = Some(CalendarPredicates::init(engine)?);
        }
        if self.role_vectors.is_none() {
            self.role_vectors = Some(CalendarRoleVectors::new(engine.ops()));
        }
        Ok(())
    }

    /// Number of events.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    // ── CRUD ──────────────────────────────────────────────────────────

    /// Add a new calendar event.
    ///
    /// Creates a KG entity, syncs metadata triples, computes Allen relations
    /// against existing events, and records provenance.
    pub fn add_event(
        &mut self,
        engine: &Engine,
        summary: &str,
        dtstart: u64,
        dtend: u64,
        location: Option<&str>,
        description: Option<&str>,
        recurrence: Option<super::pim::Recurrence>,
        ical_uid: Option<&str>,
    ) -> CalendarResult<SymbolId> {
        let label = format!("cal-event:{summary}");
        let sym = engine.resolve_or_create_entity(&label)?;

        let event = CalendarEvent {
            symbol_id: sym,
            summary: summary.to_string(),
            dtstart,
            dtend,
            location: location.map(|s| s.to_string()),
            description: description.map(|s| s.to_string()),
            recurrence,
            ical_uid: ical_uid.map(|s| s.to_string()),
            confirmed: true,
        };

        self.sync_to_kg(engine, &event)?;
        self.record_provenance(engine, &event);
        self.events.insert(sym.get(), event);

        // Compute Allen relations against all existing events.
        self.compute_allen_relations(engine, sym)?;

        Ok(sym)
    }

    /// Remove a calendar event.
    pub fn remove_event(&mut self, event_id: u64) -> CalendarResult<CalendarEvent> {
        self.events
            .remove(&event_id)
            .ok_or(CalendarError::EventNotFound { event_id })
    }

    /// Get an event by symbol ID.
    pub fn get_event(&self, event_id: u64) -> Option<&CalendarEvent> {
        self.events.get(&event_id)
    }

    /// All events.
    pub fn events(&self) -> &HashMap<u64, CalendarEvent> {
        &self.events
    }

    // ── Query ─────────────────────────────────────────────────────────

    /// Events in a time range [start, end).
    pub fn events_in_range(&self, start: u64, end: u64) -> Vec<&CalendarEvent> {
        self.events
            .values()
            .filter(|e| e.dtend > start && e.dtstart < end)
            .collect()
    }

    /// Events occurring today (snap to midnight boundaries).
    pub fn today_events(&self, now_ts: u64) -> Vec<&CalendarEvent> {
        let day_start = now_ts - (now_ts % 86_400);
        let day_end = day_start + 86_400;
        self.events_in_range(day_start, day_end)
    }

    /// Events occurring in the next 7 days.
    pub fn week_events(&self, now_ts: u64) -> Vec<&CalendarEvent> {
        let day_start = now_ts - (now_ts % 86_400);
        let week_end = day_start + 7 * 86_400;
        self.events_in_range(day_start, week_end)
    }

    // ── Conflict Detection ────────────────────────────────────────────

    /// Detect scheduling conflicts using sweep-line algorithm.
    ///
    /// Returns pairs of overlapping events. O(n log n) average case.
    pub fn detect_conflicts(&self) -> Vec<(SymbolId, SymbolId)> {
        let mut sorted: Vec<&CalendarEvent> = self.events.values().collect();
        sorted.sort_by_key(|e| e.dtstart);

        let mut conflicts = Vec::new();
        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                // Once the next event starts after the current ends, no more overlaps.
                if sorted[j].dtstart >= sorted[i].dtend {
                    break;
                }
                conflicts.push((sorted[i].symbol_id, sorted[j].symbol_id));
            }
        }
        conflicts
    }

    // ── Allen Relations ───────────────────────────────────────────────

    /// Compute and store Allen relations between `event_id` and all other events.
    pub fn compute_allen_relations(
        &self,
        engine: &Engine,
        event_id: SymbolId,
    ) -> CalendarResult<()> {
        let predicates = match &self.predicates {
            Some(p) => p,
            None => return Ok(()),
        };

        let event = match self.events.get(&event_id.get()) {
            Some(e) => e,
            None => return Err(CalendarError::EventNotFound { event_id: event_id.get() }),
        };

        for other in self.events.values() {
            if other.symbol_id == event_id {
                continue;
            }
            let relation = AllenRelation::compute(
                event.dtstart,
                event.dtend,
                other.dtstart,
                other.dtend,
            );
            let pred = predicates.allen_predicate(relation);
            let triple = Triple::new(event_id, pred, other.symbol_id);
            let _ = engine.add_triple(&triple);
        }
        Ok(())
    }

    // ── VSA Encoding ──────────────────────────────────────────────────

    /// Encode a temporal pattern for an event as a VSA hypervector.
    ///
    /// Pattern: bind(day_role, day_filler) ⊕ bind(time_role, time_filler)
    ///        ⊕ bind(activity_role, summary_filler) ⊕ bind(duration_role, dur_filler)
    pub fn encode_temporal_pattern(&self, ops: &VsaOps, event: &CalendarEvent) -> Option<HyperVec> {
        let roles = self.role_vectors.as_ref()?;

        // Day of week: 0=Mon .. 6=Sun from UNIX timestamp.
        let day_num = ((event.dtstart / 86_400) + 3) % 7; // epoch was Thursday
        let day_filler = encode_token(ops, &format!("cal-day:{day_num}"));
        let day_bound = ops.bind(&roles.day_of_week, &day_filler).ok()?;

        // Time of day bucket: morning/afternoon/evening/night.
        let hour = (event.dtstart % 86_400) / 3600;
        let bucket = match hour {
            6..=11 => "morning",
            12..=17 => "afternoon",
            18..=22 => "evening",
            _ => "night",
        };
        let time_filler = encode_token(ops, &format!("cal-time:{bucket}"));
        let time_bound = ops.bind(&roles.time_of_day, &time_filler).ok()?;

        // Activity type from summary.
        let activity_filler = encode_token(ops, &format!("cal-activity:{}", event.summary));
        let activity_bound = ops.bind(&roles.activity_type, &activity_filler).ok()?;

        // Duration bucket.
        let dur_secs = event.duration_secs();
        let dur_bucket = match dur_secs {
            0..=1800 => "short",
            1801..=3600 => "medium",
            3601..=7200 => "long",
            _ => "extended",
        };
        let dur_filler = encode_token(ops, &format!("cal-duration:{dur_bucket}"));
        let dur_bound = ops.bind(&roles.duration, &dur_filler).ok()?;

        // Bundle all bindings.
        let pattern = ops.bundle(&[&day_bound, &time_bound, &activity_bound, &dur_bound]).ok()?;
        Some(pattern)
    }

    // ── KG Sync ───────────────────────────────────────────────────────

    /// Sync calendar event metadata to the KG as triples.
    fn sync_to_kg(&self, engine: &Engine, event: &CalendarEvent) -> CalendarResult<()> {
        let predicates = match &self.predicates {
            Some(p) => p,
            None => return Ok(()),
        };

        // Summary triple.
        let summary_obj = engine.resolve_or_create_entity(&event.summary)?;
        let _ = engine.add_triple(&Triple::new(
            event.symbol_id,
            predicates.cal_summary,
            summary_obj,
        ));

        // dtstart triple.
        let start_obj = engine.resolve_or_create_entity(&format!("ts:{}", event.dtstart))?;
        let _ = engine.add_triple(&Triple::new(
            event.symbol_id,
            predicates.cal_dtstart,
            start_obj,
        ));

        // dtend triple.
        let end_obj = engine.resolve_or_create_entity(&format!("ts:{}", event.dtend))?;
        let _ = engine.add_triple(&Triple::new(
            event.symbol_id,
            predicates.cal_dtend,
            end_obj,
        ));

        // Location triple (if present).
        if let Some(ref loc) = event.location {
            let loc_obj = engine.resolve_or_create_entity(loc)?;
            let _ = engine.add_triple(&Triple::new(
                event.symbol_id,
                predicates.cal_location,
                loc_obj,
            ));
        }

        Ok(())
    }

    // ── Provenance ────────────────────────────────────────────────────

    fn record_provenance(&self, engine: &Engine, event: &CalendarEvent) {
        let mut record = ProvenanceRecord::new(
            event.symbol_id,
            DerivationKind::CalendarEventManaged {
                event_summary: event.summary.clone(),
                dtstart: event.dtstart,
                dtend: event.dtend,
            },
        );
        let _ = engine.store_provenance(&mut record);
    }

    // ── Persistence ──────────────────────────────────────────────────

    /// Persist calendar state to the engine's durable store.
    pub fn persist(&self, engine: &Engine) -> CalendarResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            CalendarError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("failed to serialize calendar manager: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:calendar_manager", &bytes)
            .map_err(|e| CalendarError::Engine(Box::new(crate::error::AkhError::Store(e))))?;
        Ok(())
    }

    /// Restore calendar state from the engine's durable store.
    pub fn restore(engine: &Engine) -> CalendarResult<Self> {
        let bytes = engine
            .store()
            .get_meta(b"agent:calendar_manager")
            .map_err(|e| CalendarError::Engine(Box::new(crate::error::AkhError::Store(e))))?
            .ok_or(CalendarError::EventNotFound { event_id: 0 })?;
        let mut manager: Self = bincode::deserialize(&bytes).map_err(|e| {
            CalendarError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("failed to deserialize calendar manager: {e}"),
                },
            )))
        })?;
        manager.ensure_init(engine)?;
        Ok(manager)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// iCalendar Import (feature-gated)
// ═══════════════════════════════════════════════════════════════════════

/// Import events from an iCalendar (RFC 5545) string.
///
/// Parses VEVENT components, extracts timestamps, and adds them via CalendarManager.
#[cfg(feature = "calendar")]
pub fn import_ical(
    manager: &mut CalendarManager,
    engine: &Engine,
    data: &str,
) -> CalendarResult<Vec<SymbolId>> {
    use icalendar::{Calendar, Component, EventLike};

    let calendar: Calendar = data
        .parse()
        .map_err(|_| CalendarError::ParseError {
            message: "failed to parse iCalendar data".to_string(),
        })?;

    let mut imported = Vec::new();
    for component in calendar.components {
        if let Some(event) = component.as_event() {
            let summary = event
                .get_summary()
                .unwrap_or("Untitled")
                .to_string();

            let dtstart = event
                .get_start()
                .and_then(|dp| extract_timestamp(&dp))
                .unwrap_or(0);

            let dtend = event
                .get_end()
                .and_then(|dp| extract_timestamp(&dp))
                .unwrap_or(dtstart + 3600); // default 1h

            let location = event.get_location().map(|s| s.to_string());
            let description = event.get_description().map(|s| s.to_string());

            let uid = event
                .property_value("UID")
                .map(|s| s.to_string());

            // Skip if already imported (dedup by ical_uid).
            if let Some(ref uid_str) = uid {
                let exists = manager.events.values().any(|e| {
                    e.ical_uid.as_deref() == Some(uid_str.as_str())
                });
                if exists {
                    continue;
                }
            }

            let sym = manager.add_event(
                engine,
                &summary,
                dtstart,
                dtend,
                location.as_deref(),
                description.as_deref(),
                None,
                uid.as_deref(),
            )?;
            imported.push(sym);
        }
    }
    Ok(imported)
}

/// Extract a UNIX timestamp from an icalendar `DatePerhapsTime`.
#[cfg(feature = "calendar")]
fn extract_timestamp(dp: &icalendar::DatePerhapsTime) -> Option<u64> {
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime, TimeZone, Utc};
    match dp {
        icalendar::DatePerhapsTime::DateTime(cdt) => match cdt {
            icalendar::CalendarDateTime::Floating(ndt) => {
                Some(ndt.and_utc().timestamp() as u64)
            }
            icalendar::CalendarDateTime::Utc(dt) => Some(dt.timestamp() as u64),
            icalendar::CalendarDateTime::WithTimezone { date_time, tzid: _ } => {
                // Best-effort: treat as UTC when we can't resolve timezone.
                Some(date_time.and_utc().timestamp() as u64)
            }
        },
        icalendar::DatePerhapsTime::Date(date) => {
            let ndt = NaiveDateTime::new(*date, NaiveTime::from_hms_opt(0, 0, 0)?);
            Some(ndt.and_utc().timestamp() as u64)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CalDAV Sync (feature-gated)
// ═══════════════════════════════════════════════════════════════════════

/// Sync calendar events from a CalDAV server.
///
/// Fetches the calendar resource via HTTP GET, parses as iCalendar, and
/// deduplicates by `ical_uid`.
#[cfg(feature = "calendar")]
pub fn sync_caldav(
    manager: &mut CalendarManager,
    engine: &Engine,
    url: &str,
    user: &str,
    pass: &str,
) -> CalendarResult<Vec<SymbolId>> {
    let response = ureq::get(url)
        .set("Authorization", &format!(
            "Basic {}",
            base64_encode(&format!("{user}:{pass}"))
        ))
        .call()
        .map_err(|e| CalendarError::SyncError {
            message: format!("CalDAV request failed: {e}"),
        })?;

    let body = response
        .into_string()
        .map_err(|e| CalendarError::SyncError {
            message: format!("failed to read CalDAV response: {e}"),
        })?;

    import_ical(manager, engine, &body)
}

/// Minimal base64 encoder for Basic auth (avoids adding a base64 crate dep).
#[cfg(feature = "calendar")]
fn base64_encode(input: &str) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Allen Relation Tests ──────────────────────────────────────────

    #[test]
    fn allen_before() {
        assert_eq!(AllenRelation::compute(1, 3, 5, 7), AllenRelation::Before);
    }

    #[test]
    fn allen_after() {
        assert_eq!(AllenRelation::compute(5, 7, 1, 3), AllenRelation::After);
    }

    #[test]
    fn allen_meets() {
        assert_eq!(AllenRelation::compute(1, 3, 3, 5), AllenRelation::Meets);
    }

    #[test]
    fn allen_met_by() {
        assert_eq!(AllenRelation::compute(3, 5, 1, 3), AllenRelation::MetBy);
    }

    #[test]
    fn allen_overlaps() {
        assert_eq!(AllenRelation::compute(1, 4, 3, 6), AllenRelation::Overlaps);
    }

    #[test]
    fn allen_overlapped_by() {
        assert_eq!(
            AllenRelation::compute(3, 6, 1, 4),
            AllenRelation::OverlappedBy
        );
    }

    #[test]
    fn allen_during() {
        assert_eq!(AllenRelation::compute(3, 5, 1, 7), AllenRelation::During);
    }

    #[test]
    fn allen_contains() {
        assert_eq!(AllenRelation::compute(1, 7, 3, 5), AllenRelation::Contains);
    }

    #[test]
    fn allen_starts() {
        assert_eq!(AllenRelation::compute(1, 3, 1, 5), AllenRelation::Starts);
    }

    #[test]
    fn allen_started_by() {
        assert_eq!(AllenRelation::compute(1, 5, 1, 3), AllenRelation::StartedBy);
    }

    #[test]
    fn allen_finishes() {
        assert_eq!(AllenRelation::compute(3, 5, 1, 5), AllenRelation::Finishes);
    }

    #[test]
    fn allen_finished_by() {
        assert_eq!(
            AllenRelation::compute(1, 5, 3, 5),
            AllenRelation::FinishedBy
        );
    }

    #[test]
    fn allen_equals() {
        assert_eq!(AllenRelation::compute(1, 5, 1, 5), AllenRelation::Equals);
    }

    #[test]
    fn allen_inverse_symmetry() {
        for &rel in &[
            AllenRelation::Before,
            AllenRelation::After,
            AllenRelation::Meets,
            AllenRelation::MetBy,
            AllenRelation::Overlaps,
            AllenRelation::OverlappedBy,
            AllenRelation::During,
            AllenRelation::Contains,
            AllenRelation::Starts,
            AllenRelation::StartedBy,
            AllenRelation::Finishes,
            AllenRelation::FinishedBy,
            AllenRelation::Equals,
        ] {
            assert_eq!(rel.inverse().inverse(), rel, "double inverse of {rel}");
        }
    }

    #[test]
    fn allen_is_overlapping() {
        assert!(!AllenRelation::Before.is_overlapping());
        assert!(!AllenRelation::After.is_overlapping());
        assert!(!AllenRelation::Meets.is_overlapping());
        assert!(!AllenRelation::MetBy.is_overlapping());
        assert!(AllenRelation::Overlaps.is_overlapping());
        assert!(AllenRelation::During.is_overlapping());
        assert!(AllenRelation::Equals.is_overlapping());
        assert!(AllenRelation::Contains.is_overlapping());
        assert!(AllenRelation::Starts.is_overlapping());
    }

    #[test]
    fn allen_label_roundtrip() {
        for &rel in &[
            AllenRelation::Before,
            AllenRelation::After,
            AllenRelation::Meets,
            AllenRelation::MetBy,
            AllenRelation::Overlaps,
            AllenRelation::OverlappedBy,
            AllenRelation::During,
            AllenRelation::Contains,
            AllenRelation::Starts,
            AllenRelation::StartedBy,
            AllenRelation::Finishes,
            AllenRelation::FinishedBy,
            AllenRelation::Equals,
        ] {
            let label = rel.as_label();
            let parsed = AllenRelation::from_label(label);
            assert_eq!(parsed, Some(rel), "roundtrip failed for {label}");
        }
    }

    #[test]
    fn allen_from_label_unknown() {
        assert_eq!(AllenRelation::from_label("unknown"), None);
    }

    // ── CalendarEvent Tests ───────────────────────────────────────────

    #[test]
    fn event_duration() {
        let event = CalendarEvent {
            symbol_id: SymbolId::new(1).unwrap(),
            summary: "test".into(),
            dtstart: 1000,
            dtend: 4600,
            location: None,
            description: None,
            recurrence: None,
            ical_uid: None,
            confirmed: true,
        };
        assert_eq!(event.duration_secs(), 3600);
    }

    #[test]
    fn event_overlaps_true() {
        let a = CalendarEvent {
            symbol_id: SymbolId::new(1).unwrap(),
            summary: "a".into(),
            dtstart: 100,
            dtend: 200,
            location: None,
            description: None,
            recurrence: None,
            ical_uid: None,
            confirmed: true,
        };
        let b = CalendarEvent {
            symbol_id: SymbolId::new(2).unwrap(),
            summary: "b".into(),
            dtstart: 150,
            dtend: 250,
            location: None,
            description: None,
            recurrence: None,
            ical_uid: None,
            confirmed: true,
        };
        assert!(a.overlaps(&b));
    }

    #[test]
    fn event_overlaps_false() {
        let a = CalendarEvent {
            symbol_id: SymbolId::new(1).unwrap(),
            summary: "a".into(),
            dtstart: 100,
            dtend: 200,
            location: None,
            description: None,
            recurrence: None,
            ical_uid: None,
            confirmed: true,
        };
        let b = CalendarEvent {
            symbol_id: SymbolId::new(2).unwrap(),
            summary: "b".into(),
            dtstart: 300,
            dtend: 400,
            location: None,
            description: None,
            recurrence: None,
            ical_uid: None,
            confirmed: true,
        };
        assert!(!a.overlaps(&b));
    }

    // ── CalendarManager Tests ─────────────────────────────────────────

    #[test]
    fn manager_default_empty() {
        let mgr = CalendarManager::default();
        assert_eq!(mgr.event_count(), 0);
    }

    #[test]
    fn manager_add_and_get() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        let sym = mgr
            .add_event(&engine, "Meeting", 1000, 2000, Some("Room A"), None, None, None)
            .unwrap();
        assert_eq!(mgr.event_count(), 1);

        let event = mgr.get_event(sym.get()).unwrap();
        assert_eq!(event.summary, "Meeting");
        assert_eq!(event.dtstart, 1000);
        assert_eq!(event.dtend, 2000);
        assert_eq!(event.location.as_deref(), Some("Room A"));
    }

    #[test]
    fn manager_remove() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        let sym = mgr
            .add_event(&engine, "Test", 1000, 2000, None, None, None, None)
            .unwrap();
        assert_eq!(mgr.event_count(), 1);
        mgr.remove_event(sym.get()).unwrap();
        assert_eq!(mgr.event_count(), 0);
    }

    #[test]
    fn manager_remove_not_found() {
        let mgr = CalendarManager::default();
        assert!(mgr.events.get(&999).is_none());
    }

    #[test]
    fn manager_detect_conflicts() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        let a = mgr
            .add_event(&engine, "A", 1000, 2000, None, None, None, None)
            .unwrap();
        let b = mgr
            .add_event(&engine, "B", 1500, 2500, None, None, None, None)
            .unwrap();
        let _c = mgr
            .add_event(&engine, "C", 3000, 4000, None, None, None, None)
            .unwrap();

        let conflicts = mgr.detect_conflicts();
        assert_eq!(conflicts.len(), 1);
        // The conflict pair should contain both A and B.
        let (c1, c2) = conflicts[0];
        assert!((c1 == a && c2 == b) || (c1 == b && c2 == a));
    }

    #[test]
    fn manager_no_conflicts() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        mgr.add_event(&engine, "A", 1000, 2000, None, None, None, None)
            .unwrap();
        mgr.add_event(&engine, "B", 3000, 4000, None, None, None, None)
            .unwrap();

        assert!(mgr.detect_conflicts().is_empty());
    }

    #[test]
    fn manager_events_in_range() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        mgr.add_event(&engine, "Early", 100, 200, None, None, None, None)
            .unwrap();
        mgr.add_event(&engine, "Mid", 500, 700, None, None, None, None)
            .unwrap();
        mgr.add_event(&engine, "Late", 900, 1100, None, None, None, None)
            .unwrap();

        let in_range = mgr.events_in_range(400, 800);
        assert_eq!(in_range.len(), 1);
        assert_eq!(in_range[0].summary, "Mid");
    }

    #[test]
    fn manager_today_events() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        let now = 1708000000u64; // some timestamp
        let day_start = now - (now % 86_400);

        mgr.add_event(
            &engine,
            "Today",
            day_start + 3600,
            day_start + 7200,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        mgr.add_event(
            &engine,
            "Yesterday",
            day_start - 86_400,
            day_start - 82_800,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let today = mgr.today_events(now);
        assert_eq!(today.len(), 1);
        assert_eq!(today[0].summary, "Today");
    }

    #[test]
    fn manager_week_events() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        let now = 1708000000u64;
        let day_start = now - (now % 86_400);

        mgr.add_event(
            &engine,
            "This week",
            day_start + 2 * 86_400,
            day_start + 2 * 86_400 + 3600,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        mgr.add_event(
            &engine,
            "Next month",
            day_start + 30 * 86_400,
            day_start + 30 * 86_400 + 3600,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let week = mgr.week_events(now);
        assert_eq!(week.len(), 1);
        assert_eq!(week[0].summary, "This week");
    }

    #[test]
    fn manager_persist_restore() {
        let dir = tempfile::tempdir().unwrap();
        let config = crate::engine::EngineConfig {
            data_dir: Some(dir.path().to_path_buf()),
            ..crate::engine::EngineConfig::default()
        };
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        mgr.add_event(&engine, "Persisted", 1000, 2000, Some("Loc"), None, None, None)
            .unwrap();
        mgr.persist(&engine).unwrap();

        let restored = CalendarManager::restore(&engine).unwrap();
        assert_eq!(restored.event_count(), 1);
        let event = restored.events.values().next().unwrap();
        assert_eq!(event.summary, "Persisted");
        assert_eq!(event.location.as_deref(), Some("Loc"));
    }

    // ── iCalendar Import Tests (feature-gated) ───────────────────────

    #[cfg(feature = "calendar")]
    #[test]
    fn import_ical_basic() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        let ical_data = "\
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:test-uid-001
DTSTART:20240215T090000Z
DTEND:20240215T100000Z
SUMMARY:Test Meeting
LOCATION:Conference Room
END:VEVENT
END:VCALENDAR";

        let imported = import_ical(&mut mgr, &engine, ical_data).unwrap();
        assert_eq!(imported.len(), 1);
        assert_eq!(mgr.event_count(), 1);

        let event = mgr.get_event(imported[0].get()).unwrap();
        assert_eq!(event.summary, "Test Meeting");
        assert_eq!(event.location.as_deref(), Some("Conference Room"));
        assert_eq!(event.ical_uid.as_deref(), Some("test-uid-001"));
    }

    #[cfg(feature = "calendar")]
    #[test]
    fn import_ical_dedup() {
        let config = crate::engine::EngineConfig::default();
        let engine = Engine::new(config).unwrap();
        let mut mgr = CalendarManager::new(&engine).unwrap();

        let ical_data = "\
BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:dedup-uid-001
DTSTART:20240215T090000Z
DTEND:20240215T100000Z
SUMMARY:Dedup Test
END:VEVENT
END:VCALENDAR";

        let first = import_ical(&mut mgr, &engine, ical_data).unwrap();
        assert_eq!(first.len(), 1);

        // Import again — should skip the duplicate.
        let second = import_ical(&mut mgr, &engine, ical_data).unwrap();
        assert_eq!(second.len(), 0);
        assert_eq!(mgr.event_count(), 1);
    }
}
