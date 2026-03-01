//! Calendar-Contact Integration — Phase 25e.
//!
//! Bridges the calendar and contact subsystems: links events to attendees,
//! queries co-attendance, computes meeting frequency, and infers availability
//! patterns from event history.

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::calendar::CalendarManager;
use super::contact::{ContactError, ContactManager, ContactResult};

// ═══════════════════════════════════════════════════════════════════════
// CalendarContactPredicates
// ═══════════════════════════════════════════════════════════════════════

/// Additional predicates for calendar-contact integration.
pub struct CalendarContactPredicates {
    /// `cal:has-attendee` — event ↔ contact link.
    pub has_attendee: SymbolId,
    /// `cal:organizer` — event ↔ contact organizer link.
    pub organizer: SymbolId,
}

impl CalendarContactPredicates {
    pub fn init(engine: &Engine) -> ContactResult<Self> {
        Ok(Self {
            has_attendee: engine.resolve_or_create_relation("cal:has-attendee")?,
            organizer: engine.resolve_or_create_relation("cal:organizer")?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ContactCalendar — stateless utility
// ═══════════════════════════════════════════════════════════════════════

/// Stateless utility for calendar-contact integration queries.
pub struct ContactCalendar;

impl ContactCalendar {
    /// Add a contact as attendee of a calendar event.
    pub fn add_attendee(
        engine: &Engine,
        preds: &CalendarContactPredicates,
        event_symbol: SymbolId,
        contact_symbol: SymbolId,
        event_id: u64,
        contact_id: &str,
    ) -> ContactResult<()> {
        engine.add_triple(&Triple::new(
            event_symbol,
            preds.has_attendee,
            contact_symbol,
        ))?;

        // Record provenance.
        let mut record = ProvenanceRecord::new(
            event_symbol,
            DerivationKind::CalendarAttendeeLinked {
                event_id,
                contact_id: contact_id.to_string(),
            },
        );
        let _ = engine.store_provenance(&mut record);

        Ok(())
    }

    /// Set the organizer of an event.
    pub fn set_organizer(
        engine: &Engine,
        preds: &CalendarContactPredicates,
        event_symbol: SymbolId,
        contact_symbol: SymbolId,
    ) -> ContactResult<()> {
        engine.add_triple(&Triple::new(
            event_symbol,
            preds.organizer,
            contact_symbol,
        ))?;
        Ok(())
    }

    /// Get all events a contact attends (by scanning KG triples).
    pub fn events_with(
        cal_mgr: &CalendarManager,
        engine: &Engine,
        preds: &CalendarContactPredicates,
        contact_symbol: SymbolId,
    ) -> Vec<u64> {
        // Find all events where contact is an attendee (contact is the object).
        let triples = engine.triples_to(contact_symbol);
        triples
            .iter()
            .filter(|t| t.predicate == preds.has_attendee)
            .filter_map(|t| {
                let eid = t.subject.get();
                cal_mgr.get_event(eid).map(|_| eid)
            })
            .collect()
    }

    /// Events both contacts attend (co-attendance).
    pub fn shared_events(
        cal_mgr: &CalendarManager,
        engine: &Engine,
        preds: &CalendarContactPredicates,
        contact_a: SymbolId,
        contact_b: SymbolId,
    ) -> Vec<u64> {
        let events_a: std::collections::HashSet<u64> =
            Self::events_with(cal_mgr, engine, preds, contact_a)
                .into_iter()
                .collect();
        Self::events_with(cal_mgr, engine, preds, contact_b)
            .into_iter()
            .filter(|eid| events_a.contains(eid))
            .collect()
    }

    /// Number of meetings with a contact within a time window.
    pub fn meeting_frequency(
        cal_mgr: &CalendarManager,
        engine: &Engine,
        preds: &CalendarContactPredicates,
        contact_symbol: SymbolId,
        window_start: u64,
        window_end: u64,
    ) -> usize {
        let event_ids = Self::events_with(cal_mgr, engine, preds, contact_symbol);
        event_ids
            .iter()
            .filter_map(|eid| cal_mgr.get_event(*eid))
            .filter(|ev| ev.dtstart >= window_start && ev.dtstart < window_end)
            .count()
    }

    /// Next upcoming event with a contact after `now`.
    pub fn next_event_with(
        cal_mgr: &CalendarManager,
        engine: &Engine,
        preds: &CalendarContactPredicates,
        contact_symbol: SymbolId,
        now: u64,
    ) -> Option<u64> {
        let event_ids = Self::events_with(cal_mgr, engine, preds, contact_symbol);
        event_ids
            .iter()
            .filter_map(|eid| cal_mgr.get_event(*eid).map(|ev| (*eid, ev.dtstart)))
            .filter(|(_, start)| *start > now)
            .min_by_key(|(_, start)| *start)
            .map(|(eid, _)| eid)
    }

    /// Resolve email addresses to contacts and add them as attendees.
    pub fn import_ical_attendees(
        engine: &Engine,
        preds: &CalendarContactPredicates,
        contact_mgr: &mut ContactManager,
        event_symbol: SymbolId,
        event_id: u64,
        emails: &[String],
    ) -> ContactResult<Vec<String>> {
        let mut contact_ids = Vec::new();
        for email in emails {
            let cid = contact_mgr.resolve_or_create(engine, email, None)?;
            let contact = contact_mgr
                .get(&cid)
                .ok_or_else(|| ContactError::NotFound {
                    contact_id: cid.clone(),
                })?;
            Self::add_attendee(
                engine,
                preds,
                event_symbol,
                contact.symbol_id,
                event_id,
                &cid,
            )?;
            contact_ids.push(cid);
        }
        Ok(contact_ids)
    }

    /// Availability heatmap: number of events per hour-of-day for a contact.
    ///
    /// Returns a 24-element array where index `i` is the count of events
    /// that start during hour `i` (00:00 to 23:59).
    pub fn availability_heatmap(
        cal_mgr: &CalendarManager,
        engine: &Engine,
        preds: &CalendarContactPredicates,
        contact_symbol: SymbolId,
    ) -> [u32; 24] {
        let mut heatmap = [0u32; 24];
        let event_ids = Self::events_with(cal_mgr, engine, preds, contact_symbol);
        for eid in event_ids {
            if let Some(ev) = cal_mgr.get_event(eid) {
                // Convert UNIX timestamp to hour of day (UTC).
                let hour = ((ev.dtstart % 86400) / 3600) as usize;
                if hour < 24 {
                    heatmap[hour] += 1;
                }
            }
        }
        heatmap
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::calendar::CalendarManager;
    use crate::agent::contact::ContactManager;
    use crate::engine::Engine;

    fn test_engine() -> Engine {
        use crate::engine::EngineConfig;
        use crate::vsa::Dimension;
        Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .expect("in-memory engine")
    }

    fn setup() -> (
        Engine,
        CalendarManager,
        ContactManager,
        CalendarContactPredicates,
    ) {
        let engine = test_engine();
        let cal = CalendarManager::new(&engine).unwrap();
        let contacts = ContactManager::new(&engine).unwrap();
        let preds = CalendarContactPredicates::init(&engine).unwrap();
        (engine, cal, contacts, preds)
    }

    #[test]
    fn add_attendee_and_query() {
        let (engine, mut cal, mut contacts, preds) = setup();

        let event_sym = cal
            .add_event(&engine, "Team Meeting", 1000, 2000, None, None, None, None)
            .unwrap();

        let cid = contacts
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let contact = contacts.get(&cid).unwrap();

        ContactCalendar::add_attendee(
            &engine,
            &preds,
            event_sym,
            contact.symbol_id,
            event_sym.get(),
            &cid,
        )
        .unwrap();

        let events = ContactCalendar::events_with(&cal, &engine, &preds, contact.symbol_id);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], event_sym.get());
    }

    #[test]
    fn shared_events_between_contacts() {
        let (engine, mut cal, mut contacts, preds) = setup();

        let event_sym = cal
            .add_event(&engine, "Sync", 1000, 2000, None, None, None, None)
            .unwrap();

        let alice_id = contacts
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let bob_id = contacts
            .create_contact(&engine, "Bob", &["bob@b.com".to_string()])
            .unwrap();

        let alice_sym = contacts.get(&alice_id).unwrap().symbol_id;
        let bob_sym = contacts.get(&bob_id).unwrap().symbol_id;

        ContactCalendar::add_attendee(
            &engine,
            &preds,
            event_sym,
            alice_sym,
            event_sym.get(),
            &alice_id,
        )
        .unwrap();
        ContactCalendar::add_attendee(
            &engine,
            &preds,
            event_sym,
            bob_sym,
            event_sym.get(),
            &bob_id,
        )
        .unwrap();

        let shared =
            ContactCalendar::shared_events(&cal, &engine, &preds, alice_sym, bob_sym);
        assert_eq!(shared.len(), 1);
    }

    #[test]
    fn meeting_frequency() {
        let (engine, mut cal, mut contacts, preds) = setup();

        let alice_id = contacts
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let alice_sym = contacts.get(&alice_id).unwrap().symbol_id;

        // Add 3 events within a 30-day window.
        for i in 0..3 {
            let start = 1000 + i * 86400;
            let ev = cal
                .add_event(
                    &engine,
                    &format!("Meeting {i}"),
                    start,
                    start + 3600,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
            ContactCalendar::add_attendee(&engine, &preds, ev, alice_sym, ev.get(), &alice_id)
                .unwrap();
        }

        let freq =
            ContactCalendar::meeting_frequency(&cal, &engine, &preds, alice_sym, 0, 1_000_000);
        assert_eq!(freq, 3);
    }

    #[test]
    fn import_ical_attendees() {
        let (engine, mut cal, mut contacts, preds) = setup();

        let event_sym = cal
            .add_event(&engine, "Party", 5000, 9000, None, None, None, None)
            .unwrap();

        let cids = ContactCalendar::import_ical_attendees(
            &engine,
            &preds,
            &mut contacts,
            event_sym,
            event_sym.get(),
            &["alice@a.com".to_string(), "bob@b.com".to_string()],
        )
        .unwrap();

        assert_eq!(cids.len(), 2);
        assert_eq!(contacts.contact_count(), 2);
    }

    #[test]
    fn availability_heatmap() {
        let (engine, mut cal, mut contacts, preds) = setup();

        let alice_id = contacts
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let alice_sym = contacts.get(&alice_id).unwrap().symbol_id;

        // Event at hour 10 (UTC): 10 * 3600 = 36000
        let ev = cal
            .add_event(&engine, "10am", 36000, 39600, None, None, None, None)
            .unwrap();
        ContactCalendar::add_attendee(&engine, &preds, ev, alice_sym, ev.get(), &alice_id)
            .unwrap();

        let heatmap = ContactCalendar::availability_heatmap(&cal, &engine, &preds, alice_sym);
        assert_eq!(heatmap[10], 1);
        assert_eq!(heatmap[0], 0);
    }

    #[test]
    fn next_event_with() {
        let (engine, mut cal, mut contacts, preds) = setup();

        let alice_id = contacts
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let alice_sym = contacts.get(&alice_id).unwrap().symbol_id;

        let ev1 = cal
            .add_event(&engine, "Past", 100, 200, None, None, None, None)
            .unwrap();
        let ev2 = cal
            .add_event(&engine, "Future", 5000, 6000, None, None, None, None)
            .unwrap();

        ContactCalendar::add_attendee(&engine, &preds, ev1, alice_sym, ev1.get(), &alice_id)
            .unwrap();
        ContactCalendar::add_attendee(&engine, &preds, ev2, alice_sym, ev2.get(), &alice_id)
            .unwrap();

        let next = ContactCalendar::next_event_with(&cal, &engine, &preds, alice_sym, 300);
        assert_eq!(next, Some(ev2.get()));
    }
}
