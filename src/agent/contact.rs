//! Contact Entity & Identity Resolution — Phase 25a.
//!
//! Unified contact management that links interlocutors, email senders,
//! and calendar attendees into coherent person entities. Supports identity
//! resolution (mapping multiple aliases to one contact), merging, and
//! KG-backed persistence.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

/// Errors specific to the contact subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum ContactError {
    #[error("contact not found: \"{contact_id}\"")]
    #[diagnostic(
        code(akh::agent::contact::not_found),
        help("Create a contact first with `contact_manager.create_contact()`, or use `resolve_or_create()` for auto-creation.")
    )]
    NotFound { contact_id: String },

    #[error("duplicate alias: \"{alias}\" already belongs to contact \"{existing_contact}\"")]
    #[diagnostic(
        code(akh::agent::contact::duplicate_alias),
        help("Merge the two contacts with `contact_manager.merge()` if they are the same person, or use a different alias.")
    )]
    DuplicateAlias {
        alias: String,
        existing_contact: String,
    },

    #[error("cannot merge contact \"{contact_id}\" with itself")]
    #[diagnostic(
        code(akh::agent::contact::self_merge),
        help("Provide two different contact IDs to merge.")
    )]
    SelfMerge { contact_id: String },

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::contact::engine),
        help("Engine-level error during contact operation.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for ContactError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type ContactResult<T> = std::result::Result<T, ContactError>;

// ═══════════════════════════════════════════════════════════════════════
// ContactPredicates — well-known KG relations
// ═══════════════════════════════════════════════════════════════════════

/// Well-known predicate SymbolIds for contact metadata.
#[derive(Debug, Clone)]
pub struct ContactPredicates {
    /// `contact:is-person` — marks an entity as a person.
    pub is_person: SymbolId,
    /// `contact:has-alias` — an alias (email, handle, etc.).
    pub has_alias: SymbolId,
    /// `contact:display-name` — preferred display name.
    pub display_name: SymbolId,
    /// `contact:has-organization` — employer / org.
    pub has_organization: SymbolId,
    /// `contact:has-phone` — phone number.
    pub has_phone: SymbolId,
    /// `contact:has-note` — free-text notes.
    pub has_note: SymbolId,
    /// `contact:merged-from` — provenance link for merged contacts.
    pub merged_from: SymbolId,
    /// `contact:linked-interlocutor` — link to InterlocutorProfile ID.
    pub linked_interlocutor: SymbolId,
    /// `contact:linked-sender` — link to email SenderStats address.
    pub linked_sender: SymbolId,
    /// `contact:created-at` — creation timestamp.
    pub created_at: SymbolId,
    // --- Phase 25c additions (per-person conversation memory) ---
    /// `contact:participated-in` — episode ↔ contact link.
    pub participated_in: SymbolId,
    /// `contact:discussed-topic` — contact ↔ topic with episode provenance.
    pub discussed_topic: SymbolId,
    /// `contact:shared-context` — two contacts share a topic.
    pub shared_context: SymbolId,
}

impl ContactPredicates {
    /// Resolve or create all contact predicates.
    pub fn init(engine: &Engine) -> ContactResult<Self> {
        Ok(Self {
            is_person: engine.resolve_or_create_relation("contact:is-person")?,
            has_alias: engine.resolve_or_create_relation("contact:has-alias")?,
            display_name: engine.resolve_or_create_relation("contact:display-name")?,
            has_organization: engine.resolve_or_create_relation("contact:has-organization")?,
            has_phone: engine.resolve_or_create_relation("contact:has-phone")?,
            has_note: engine.resolve_or_create_relation("contact:has-note")?,
            merged_from: engine.resolve_or_create_relation("contact:merged-from")?,
            linked_interlocutor: engine.resolve_or_create_relation("contact:linked-interlocutor")?,
            linked_sender: engine.resolve_or_create_relation("contact:linked-sender")?,
            created_at: engine.resolve_or_create_relation("contact:created-at")?,
            participated_in: engine.resolve_or_create_relation("contact:participated-in")?,
            discussed_topic: engine.resolve_or_create_relation("contact:discussed-topic")?,
            shared_context: engine.resolve_or_create_relation("contact:shared-context")?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Contact struct
// ═══════════════════════════════════════════════════════════════════════

/// A unified person entity linking interlocutors, email senders, and aliases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    /// Slug identifier, e.g. "alice-smith".
    pub contact_id: String,
    /// KG entity for this contact.
    pub symbol_id: SymbolId,
    /// Preferred display name.
    pub display_name: String,
    /// All known addresses / handles / usernames.
    pub aliases: Vec<String>,
    /// Linked InterlocutorProfile IDs.
    pub interlocutor_ids: Vec<String>,
    /// Linked email SenderStats addresses.
    pub sender_addresses: Vec<String>,
    /// Organization / employer.
    pub organization: Option<String>,
    /// Phone numbers.
    pub phones: Vec<String>,
    /// Free-text notes.
    pub notes: Vec<String>,
    /// Creation timestamp (UNIX seconds).
    pub created_at: u64,
    /// Most recent interaction across linked interlocutors.
    pub last_interaction: u64,
    /// Total interactions across linked interlocutors.
    pub total_interactions: u64,
}

// ═══════════════════════════════════════════════════════════════════════
// ContactManager
// ═══════════════════════════════════════════════════════════════════════

/// Manages contacts with O(1) alias-based lookups and KG persistence.
#[derive(Default)]
pub struct ContactManager {
    contacts: HashMap<String, Contact>,
    /// Reverse index: alias → contact_id.
    alias_index: HashMap<String, String>,
    predicates: Option<ContactPredicates>,
}

impl Serialize for ContactManager {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Only serialize contacts — predicates are re-initialized on restore.
        self.contacts.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ContactManager {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let contacts = HashMap::<String, Contact>::deserialize(deserializer)?;
        // Rebuild the alias index from contacts.
        let mut alias_index = HashMap::new();
        for (cid, contact) in &contacts {
            for alias in &contact.aliases {
                alias_index.insert(alias.clone(), cid.clone());
            }
        }
        Ok(Self {
            contacts,
            alias_index,
            predicates: None,
        })
    }
}

impl fmt::Debug for ContactManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContactManager")
            .field("contact_count", &self.contacts.len())
            .field("alias_count", &self.alias_index.len())
            .finish()
    }
}

/// Generate a contact slug from a display name.
fn slugify(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl ContactManager {
    /// Create a new manager, initializing predicates.
    pub fn new(engine: &Engine) -> ContactResult<Self> {
        let predicates = ContactPredicates::init(engine)?;
        Ok(Self {
            contacts: HashMap::new(),
            alias_index: HashMap::new(),
            predicates: Some(predicates),
        })
    }

    /// Ensure predicates are initialized (for post-deserialization).
    pub fn ensure_init(&mut self, engine: &Engine) -> ContactResult<()> {
        if self.predicates.is_none() {
            self.predicates = Some(ContactPredicates::init(engine)?);
        }
        Ok(())
    }

    /// Access predicates (panics if not initialized — always call `ensure_init` first).
    pub fn preds(&self) -> &ContactPredicates {
        self.predicates
            .as_ref()
            .expect("ContactManager predicates not initialized — call ensure_init()")
    }

    /// Number of contacts.
    pub fn contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Iterate over all contacts.
    pub fn contacts(&self) -> impl Iterator<Item = &Contact> {
        self.contacts.values()
    }

    /// Get a contact by ID.
    pub fn get(&self, contact_id: &str) -> Option<&Contact> {
        self.contacts.get(contact_id)
    }

    /// Get a mutable contact by ID.
    pub fn get_mut(&mut self, contact_id: &str) -> Option<&mut Contact> {
        self.contacts.get_mut(contact_id)
    }

    // ── CRUD ──────────────────────────────────────────────────────────

    /// Create a new contact with the given display name and aliases.
    ///
    /// Creates a KG entity and stores `contact:is-person` + alias triples.
    /// Returns the contact ID (slug).
    pub fn create_contact(
        &mut self,
        engine: &Engine,
        display_name: &str,
        aliases: &[String],
    ) -> ContactResult<String> {
        // Clone predicates to avoid borrow conflicts.
        let preds = self.preds_or_err()?.clone();

        // Check for alias conflicts.
        for alias in aliases {
            let normalized = alias.to_lowercase();
            if let Some(existing) = self.alias_index.get(&normalized) {
                return Err(ContactError::DuplicateAlias {
                    alias: alias.clone(),
                    existing_contact: existing.clone(),
                });
            }
        }

        let base_slug = slugify(display_name);
        let contact_id = if self.contacts.contains_key(&base_slug) {
            // Disambiguate with a numeric suffix.
            let mut n = 2u32;
            loop {
                let candidate = format!("{base_slug}-{n}");
                if !self.contacts.contains_key(&candidate) {
                    break candidate;
                }
                n += 1;
            }
        } else {
            base_slug
        };

        // Create KG entity.
        let symbol_id = engine.resolve_or_create_entity(&format!("contact:{contact_id}"))?;

        // Mark as person.
        let is_person_obj = engine.resolve_or_create_entity("contact:Person")?;
        engine.add_triple(&Triple::new(symbol_id, preds.is_person, is_person_obj))?;

        // Store display name.
        let name_sym = engine.resolve_or_create_entity(&format!("literal:{display_name}"))?;
        engine.add_triple(&Triple::new(symbol_id, preds.display_name, name_sym))?;

        let now = now_secs();

        // Store aliases.
        let mut normalized_aliases = Vec::new();
        for alias in aliases {
            let normalized = alias.to_lowercase();
            let alias_sym = engine.resolve_or_create_entity(&format!("alias:{normalized}"))?;
            engine.add_triple(&Triple::new(symbol_id, preds.has_alias, alias_sym))?;
            self.alias_index.insert(normalized.clone(), contact_id.clone());
            normalized_aliases.push(normalized);
        }

        let contact = Contact {
            contact_id: contact_id.clone(),
            symbol_id,
            display_name: display_name.to_string(),
            aliases: normalized_aliases,
            interlocutor_ids: Vec::new(),
            sender_addresses: Vec::new(),
            organization: None,
            phones: Vec::new(),
            notes: Vec::new(),
            created_at: now,
            last_interaction: 0,
            total_interactions: 0,
        };

        // Record provenance.
        let mut record = ProvenanceRecord::new(
            symbol_id,
            DerivationKind::ContactResolved {
                contact_id: contact_id.clone(),
                alias_count: contact.aliases.len() as u32,
            },
        );
        let _ = engine.store_provenance(&mut record);

        self.contacts.insert(contact_id.clone(), contact);

        Ok(contact_id)
    }

    /// Find a contact by alias (email, handle, etc.). O(1) lookup.
    pub fn find_by_alias(&self, alias: &str) -> Option<&Contact> {
        let normalized = alias.to_lowercase();
        self.alias_index
            .get(&normalized)
            .and_then(|cid| self.contacts.get(cid))
    }

    /// Resolve an alias to an existing contact, or create a new one.
    ///
    /// This is the primary identity resolution entry point. If the alias
    /// already maps to a contact, returns that contact's ID. Otherwise
    /// creates a new contact using the display name hint.
    pub fn resolve_or_create(
        &mut self,
        engine: &Engine,
        alias: &str,
        display_name_hint: Option<&str>,
    ) -> ContactResult<String> {
        let normalized = alias.to_lowercase();
        if let Some(cid) = self.alias_index.get(&normalized) {
            return Ok(cid.clone());
        }

        let display_name = display_name_hint.unwrap_or(alias);
        self.create_contact(engine, display_name, &[alias.to_string()])
    }

    /// Merge two contacts: keep one, discard the other.
    ///
    /// All aliases, interlocutor links, sender links, notes, and phones
    /// from the discarded contact are moved to the kept contact. The
    /// discarded contact is removed and a `contact:merged-from` provenance
    /// link is stored.
    pub fn merge(
        &mut self,
        engine: &Engine,
        keep_id: &str,
        discard_id: &str,
    ) -> ContactResult<()> {
        if keep_id == discard_id {
            return Err(ContactError::SelfMerge {
                contact_id: keep_id.to_string(),
            });
        }

        // Clone predicates up front to avoid borrow conflicts.
        let preds = self.preds_or_err()?.clone();

        let discard = self
            .contacts
            .remove(discard_id)
            .ok_or_else(|| ContactError::NotFound {
                contact_id: discard_id.to_string(),
            })?;

        let keep = self
            .contacts
            .get_mut(keep_id)
            .ok_or_else(|| ContactError::NotFound {
                contact_id: keep_id.to_string(),
            })?;

        // Move aliases.
        let mut moved_aliases = Vec::new();
        for alias in &discard.aliases {
            if !keep.aliases.contains(alias) {
                keep.aliases.push(alias.clone());
                let alias_sym = engine.resolve_or_create_entity(&format!("alias:{alias}"))?;
                engine.add_triple(&Triple::new(keep.symbol_id, preds.has_alias, alias_sym))?;
            }
            moved_aliases.push((alias.clone(), keep_id.to_string()));
        }

        // Move interlocutor links.
        for iid in discard.interlocutor_ids {
            if !keep.interlocutor_ids.contains(&iid) {
                keep.interlocutor_ids.push(iid.clone());
                let iid_sym = engine.resolve_or_create_entity(&format!("interlocutor:{iid}"))?;
                engine.add_triple(&Triple::new(keep.symbol_id, preds.linked_interlocutor, iid_sym))?;
            }
        }

        // Move sender links.
        for addr in discard.sender_addresses {
            if !keep.sender_addresses.contains(&addr) {
                keep.sender_addresses.push(addr.clone());
                let addr_sym = engine.resolve_or_create_entity(&format!("sender:{addr}"))?;
                engine.add_triple(&Triple::new(keep.symbol_id, preds.linked_sender, addr_sym))?;
            }
        }

        // Move phones.
        for phone in discard.phones {
            if !keep.phones.contains(&phone) {
                keep.phones.push(phone);
            }
        }

        // Move notes.
        for note in discard.notes {
            keep.notes.push(note);
        }

        // Aggregate interaction stats.
        keep.last_interaction = keep.last_interaction.max(discard.last_interaction);
        keep.total_interactions += discard.total_interactions;

        // Store merged-from provenance.
        let merged_sym = engine.resolve_or_create_entity(&format!("contact:{discard_id}"))?;
        engine.add_triple(&Triple::new(keep.symbol_id, preds.merged_from, merged_sym))?;

        let keep_symbol = keep.symbol_id;

        // Update alias index.
        for (alias, cid) in moved_aliases {
            self.alias_index.insert(alias, cid);
        }

        let mut record = ProvenanceRecord::new(
            keep_symbol,
            DerivationKind::ContactMerged {
                kept: keep_id.to_string(),
                discarded: discard_id.to_string(),
            },
        );
        let _ = engine.store_provenance(&mut record);

        Ok(())
    }

    /// Link an interlocutor to a contact.
    pub fn link_interlocutor(
        &mut self,
        engine: &Engine,
        contact_id: &str,
        interlocutor_id: &str,
    ) -> ContactResult<()> {
        let preds = self.preds_or_err()?.clone();
        let contact = self
            .contacts
            .get_mut(contact_id)
            .ok_or_else(|| ContactError::NotFound {
                contact_id: contact_id.to_string(),
            })?;

        if !contact.interlocutor_ids.iter().any(|id| id == interlocutor_id) {
            let iid_sym = engine.resolve_or_create_entity(&format!("interlocutor:{interlocutor_id}"))?;
            engine.add_triple(&Triple::new(contact.symbol_id, preds.linked_interlocutor, iid_sym))?;
            contact.interlocutor_ids.push(interlocutor_id.to_string());
        }

        Ok(())
    }

    /// Link an email sender address to a contact.
    pub fn link_sender(
        &mut self,
        engine: &Engine,
        contact_id: &str,
        sender_address: &str,
    ) -> ContactResult<()> {
        let preds = self.preds_or_err()?.clone();
        let contact = self
            .contacts
            .get_mut(contact_id)
            .ok_or_else(|| ContactError::NotFound {
                contact_id: contact_id.to_string(),
            })?;

        let normalized = sender_address.to_lowercase();
        if !contact.sender_addresses.iter().any(|a| a == &normalized) {
            let addr_sym = engine.resolve_or_create_entity(&format!("sender:{normalized}"))?;
            engine.add_triple(&Triple::new(contact.symbol_id, preds.linked_sender, addr_sym))?;
            contact.sender_addresses.push(normalized);
        }

        Ok(())
    }

    /// Search contacts by display name (case-insensitive substring match).
    pub fn search_by_name(&self, query: &str) -> Vec<&Contact> {
        let q = query.to_lowercase();
        self.contacts
            .values()
            .filter(|c| c.display_name.to_lowercase().contains(&q))
            .collect()
    }

    // ── Persistence ───────────────────────────────────────────────────

    /// Persist contact manager to the engine's durable store.
    pub fn persist(&self, engine: &Engine) -> ContactResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            ContactError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("failed to serialize contact manager: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:contact_manager", &bytes)
            .map_err(|e| ContactError::Engine(Box::new(crate::error::AkhError::Store(e))))?;
        Ok(())
    }

    /// Restore contact manager from the engine's durable store.
    pub fn restore(engine: &Engine) -> ContactResult<Self> {
        let bytes = engine
            .store()
            .get_meta(b"agent:contact_manager")
            .map_err(|e| ContactError::Engine(Box::new(crate::error::AkhError::Store(e))))?
            .ok_or(ContactError::NotFound {
                contact_id: "<store>".to_string(),
            })?;
        let mut manager: Self = bincode::deserialize(&bytes).map_err(|e| {
            ContactError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("failed to deserialize contact manager: {e}"),
                },
            )))
        })?;
        manager.ensure_init(engine)?;
        Ok(manager)
    }

    /// Private helper: safely access predicates, returning an error if not initialized.
    fn preds_or_err(&self) -> ContactResult<&ContactPredicates> {
        self.predicates.as_ref().ok_or_else(|| ContactError::Engine(
            Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: "ContactManager predicates not initialized".to_string(),
                },
            )),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
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

    #[test]
    fn create_contact_and_find_by_alias() {
        let engine = test_engine();
        let mut mgr = ContactManager::new(&engine).unwrap();

        let cid = mgr
            .create_contact(&engine, "Alice Smith", &["alice@work.com".to_string()])
            .unwrap();

        assert_eq!(cid, "alice-smith");
        assert_eq!(mgr.contact_count(), 1);

        let found = mgr.find_by_alias("alice@work.com").unwrap();
        assert_eq!(found.contact_id, "alice-smith");
        assert_eq!(found.display_name, "Alice Smith");
    }

    #[test]
    fn find_by_alias_case_insensitive() {
        let engine = test_engine();
        let mut mgr = ContactManager::new(&engine).unwrap();

        mgr.create_contact(&engine, "Bob", &["BOB@EXAMPLE.COM".to_string()])
            .unwrap();

        assert!(mgr.find_by_alias("bob@example.com").is_some());
        assert!(mgr.find_by_alias("Bob@Example.COM").is_some());
    }

    #[test]
    fn duplicate_alias_error() {
        let engine = test_engine();
        let mut mgr = ContactManager::new(&engine).unwrap();

        mgr.create_contact(&engine, "Alice", &["alice@work.com".to_string()])
            .unwrap();

        let result = mgr.create_contact(&engine, "Alicia", &["alice@work.com".to_string()]);
        assert!(matches!(result, Err(ContactError::DuplicateAlias { .. })));
    }

    #[test]
    fn merge_contacts() {
        let engine = test_engine();
        let mut mgr = ContactManager::new(&engine).unwrap();

        let cid1 = mgr
            .create_contact(&engine, "Alice Smith", &["alice@work.com".to_string()])
            .unwrap();
        let cid2 = mgr
            .create_contact(&engine, "Alice S", &["alice@personal.com".to_string()])
            .unwrap();

        mgr.merge(&engine, &cid1, &cid2).unwrap();

        assert_eq!(mgr.contact_count(), 1);

        let alice = mgr.get(&cid1).unwrap();
        assert!(alice.aliases.contains(&"alice@work.com".to_string()));
        assert!(alice.aliases.contains(&"alice@personal.com".to_string()));

        // Alias index points to kept contact.
        assert_eq!(
            mgr.find_by_alias("alice@personal.com").unwrap().contact_id,
            cid1
        );
    }

    #[test]
    fn self_merge_error() {
        let engine = test_engine();
        let mut mgr = ContactManager::new(&engine).unwrap();

        let cid = mgr
            .create_contact(&engine, "Alice", &["alice@work.com".to_string()])
            .unwrap();

        let result = mgr.merge(&engine, &cid, &cid);
        assert!(matches!(result, Err(ContactError::SelfMerge { .. })));
    }

    #[test]
    fn resolve_or_create_idempotency() {
        let engine = test_engine();
        let mut mgr = ContactManager::new(&engine).unwrap();

        let cid1 = mgr
            .resolve_or_create(&engine, "alice@work.com", Some("Alice"))
            .unwrap();
        let cid2 = mgr
            .resolve_or_create(&engine, "alice@work.com", Some("Alice Smith"))
            .unwrap();

        assert_eq!(cid1, cid2);
        assert_eq!(mgr.contact_count(), 1);
    }

    #[test]
    fn persist_and_restore() {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(crate::engine::EngineConfig {
            data_dir: Some(dir.path().to_path_buf()),
            dimension: crate::vsa::Dimension(1000),
            ..crate::engine::EngineConfig::default()
        })
        .unwrap();
        let mut mgr = ContactManager::new(&engine).unwrap();

        mgr.create_contact(
            &engine,
            "Alice",
            &["alice@work.com".to_string(), "alice@home.com".to_string()],
        )
        .unwrap();

        mgr.persist(&engine).unwrap();

        let restored = ContactManager::restore(&engine).unwrap();
        assert_eq!(restored.contact_count(), 1);
        assert!(restored.find_by_alias("alice@work.com").is_some());
        assert!(restored.find_by_alias("alice@home.com").is_some());
    }

    #[test]
    fn search_by_name() {
        let engine = test_engine();
        let mut mgr = ContactManager::new(&engine).unwrap();

        mgr.create_contact(&engine, "Alice Smith", &["alice@a.com".to_string()])
            .unwrap();
        mgr.create_contact(&engine, "Bob Jones", &["bob@b.com".to_string()])
            .unwrap();
        mgr.create_contact(&engine, "Alice Jones", &["alice2@c.com".to_string()])
            .unwrap();

        let results = mgr.search_by_name("alice");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn link_interlocutor_and_sender() {
        let engine = test_engine();
        let mut mgr = ContactManager::new(&engine).unwrap();

        let cid = mgr
            .create_contact(&engine, "Alice", &["alice@work.com".to_string()])
            .unwrap();

        mgr.link_interlocutor(&engine, &cid, "interlocutor-123")
            .unwrap();
        mgr.link_sender(&engine, &cid, "alice@personal.net")
            .unwrap();

        let contact = mgr.get(&cid).unwrap();
        assert!(contact.interlocutor_ids.contains(&"interlocutor-123".to_string()));
        assert!(contact.sender_addresses.contains(&"alice@personal.net".to_string()));

        // Idempotent — linking again should not duplicate.
        mgr.link_interlocutor(&engine, &cid, "interlocutor-123")
            .unwrap();
        let contact = mgr.get(&cid).unwrap();
        assert_eq!(contact.interlocutor_ids.len(), 1);
    }
}
