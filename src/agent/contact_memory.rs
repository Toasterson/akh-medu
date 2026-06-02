//! Per-Person Conversation Memory — Phase 25c.
//!
//! Indexes episodic memories by contact, enabling recall of what was discussed
//! with whom, topic intersection between contacts, and shared context discovery.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::symbol::SymbolId;

use super::contact::{ContactPredicates, ContactResult};

// ═══════════════════════════════════════════════════════════════════════
// PersonMemoryIndex
// ═══════════════════════════════════════════════════════════════════════

/// In-memory index mapping contacts to their episodic memories and discussed topics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersonMemoryIndex {
    /// contact_id → episode SymbolIds (most recent first).
    episodes_by_contact: HashMap<String, Vec<SymbolId>>,
    /// (contact_id, topic SymbolId) → episode SymbolIds where that topic was discussed.
    topic_index: HashMap<(String, SymbolId), Vec<SymbolId>>,
    /// Maximum episodes to retain per contact before pruning.
    max_per_contact: usize,
}

impl PersonMemoryIndex {
    /// Create a new index with the given per-contact episode limit.
    pub fn new(max_per_contact: usize) -> Self {
        Self {
            episodes_by_contact: HashMap::new(),
            topic_index: HashMap::new(),
            max_per_contact,
        }
    }

    /// Tag an episode as involving a contact, with optional topic links.
    ///
    /// Stores KG triples: `contact --participated-in--> episode`,
    /// `contact --discussed-topic--> topic` (per topic).
    pub fn tag_episode(
        &mut self,
        engine: &Engine,
        episode_id: SymbolId,
        contact_id: &str,
        contact_symbol: SymbolId,
        topics: &[SymbolId],
        predicates: &ContactPredicates,
    ) -> ContactResult<()> {
        // Store participation triple.
        engine.add_triple(&Triple::new(
            contact_symbol,
            predicates.participated_in,
            episode_id,
        ))?;

        // Index the episode.
        let episodes = self
            .episodes_by_contact
            .entry(contact_id.to_string())
            .or_default();
        episodes.insert(0, episode_id); // most recent first

        // Prune if over limit.
        if episodes.len() > self.max_per_contact {
            episodes.truncate(self.max_per_contact);
        }

        // Index topics.
        for &topic in topics {
            engine.add_triple(&Triple::new(
                contact_symbol,
                predicates.discussed_topic,
                topic,
            ))?;

            self.topic_index
                .entry((contact_id.to_string(), topic))
                .or_default()
                .insert(0, episode_id);
        }

        Ok(())
    }

    /// Recall episodes involving a contact, most recent first.
    pub fn recall_with_person(&self, contact_id: &str, top_k: usize) -> Vec<SymbolId> {
        self.episodes_by_contact
            .get(contact_id)
            .map(|eps| eps.iter().take(top_k).copied().collect())
            .unwrap_or_default()
    }

    /// Recall episodes involving a contact about a specific topic.
    pub fn recall_with_person_about(
        &self,
        contact_id: &str,
        topic: SymbolId,
        top_k: usize,
    ) -> Vec<SymbolId> {
        self.topic_index
            .get(&(contact_id.to_string(), topic))
            .map(|eps| eps.iter().take(top_k).copied().collect())
            .unwrap_or_default()
    }

    /// Topics discussed with a contact, ranked by frequency.
    pub fn topics_discussed_with(&self, contact_id: &str) -> Vec<(SymbolId, usize)> {
        let mut topic_counts: HashMap<SymbolId, usize> = HashMap::new();
        for ((cid, topic), episodes) in &self.topic_index {
            if cid == contact_id {
                *topic_counts.entry(*topic).or_default() += episodes.len();
            }
        }
        let mut ranked: Vec<_> = topic_counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked
    }

    /// Topics discussed with both contacts (shared context).
    pub fn shared_context_between(
        &self,
        contact_a: &str,
        contact_b: &str,
    ) -> Vec<SymbolId> {
        let topics_a: std::collections::HashSet<SymbolId> = self
            .topic_index
            .keys()
            .filter(|(cid, _)| cid == contact_a)
            .map(|(_, topic)| *topic)
            .collect();

        self.topic_index
            .keys()
            .filter(|(cid, topic)| cid == contact_b && topics_a.contains(topic))
            .map(|(_, topic)| *topic)
            .collect()
    }

    /// Total episodes indexed.
    pub fn total_episodes(&self) -> usize {
        self.episodes_by_contact.values().map(|v| v.len()).sum()
    }

    /// Number of contacts with indexed episodes.
    pub fn indexed_contact_count(&self) -> usize {
        self.episodes_by_contact.len()
    }

    /// Rebuild index from KG by scanning participation triples.
    ///
    /// Used on restore when the in-memory index was not persisted.
    pub fn rebuild_from_kg(
        &mut self,
        engine: &Engine,
        contact_manager: &super::contact::ContactManager,
        predicates: &ContactPredicates,
    ) {
        for contact in contact_manager.contacts() {
            // Query KG for participated-in triples.
            let triples = engine.triples_from(contact.symbol_id);
            let episodes: Vec<SymbolId> = triples
                .iter()
                .filter(|t| t.predicate == predicates.participated_in)
                .map(|t| t.object)
                .collect();
            if !episodes.is_empty() {
                self.episodes_by_contact
                    .insert(contact.contact_id.clone(), episodes);
            }

            // Query KG for discussed-topic triples.
            for triple in &triples {
                if triple.predicate == predicates.discussed_topic {
                    self.topic_index
                        .entry((contact.contact_id.clone(), triple.object))
                        .or_default()
                        .push(contact.symbol_id);
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
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

    fn setup() -> (Engine, ContactManager, PersonMemoryIndex) {
        let engine = test_engine();
        let mgr = ContactManager::new(&engine).unwrap();
        let idx = PersonMemoryIndex::new(100);
        (engine, mgr, idx)
    }

    #[test]
    fn tag_and_recall() {
        let (engine, mut mgr, mut idx) = setup();

        let cid = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let contact = mgr.get(&cid).unwrap();
        let preds = mgr.preds();

        let ep1 = engine.resolve_or_create_entity("episode:1").unwrap();
        let ep2 = engine.resolve_or_create_entity("episode:2").unwrap();

        idx.tag_episode(&engine, ep1, &cid, contact.symbol_id, &[], preds)
            .unwrap();
        idx.tag_episode(&engine, ep2, &cid, contact.symbol_id, &[], preds)
            .unwrap();

        let recalled = idx.recall_with_person(&cid, 10);
        assert_eq!(recalled.len(), 2);
        assert_eq!(recalled[0], ep2); // most recent first
    }

    #[test]
    fn recall_with_topic() {
        let (engine, mut mgr, mut idx) = setup();

        let cid = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let contact = mgr.get(&cid).unwrap();
        let preds = mgr.preds();

        let ep1 = engine.resolve_or_create_entity("episode:1").unwrap();
        let topic_rust = engine.resolve_or_create_entity("topic:rust").unwrap();
        let topic_python = engine.resolve_or_create_entity("topic:python").unwrap();

        idx.tag_episode(
            &engine,
            ep1,
            &cid,
            contact.symbol_id,
            &[topic_rust, topic_python],
            preds,
        )
        .unwrap();

        let rust_eps = idx.recall_with_person_about(&cid, topic_rust, 10);
        assert_eq!(rust_eps.len(), 1);

        let topics = idx.topics_discussed_with(&cid);
        assert_eq!(topics.len(), 2);
    }

    #[test]
    fn shared_context() {
        let (engine, mut mgr, mut idx) = setup();

        let alice = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let bob = mgr
            .create_contact(&engine, "Bob", &["bob@b.com".to_string()])
            .unwrap();

        let alice_contact = mgr.get(&alice).unwrap().clone();
        let bob_contact = mgr.get(&bob).unwrap().clone();
        let preds = mgr.preds().clone();

        let ep1 = engine.resolve_or_create_entity("episode:1").unwrap();
        let ep2 = engine.resolve_or_create_entity("episode:2").unwrap();
        let topic_rust = engine.resolve_or_create_entity("topic:rust").unwrap();

        idx.tag_episode(
            &engine,
            ep1,
            &alice,
            alice_contact.symbol_id,
            &[topic_rust],
            &preds,
        )
        .unwrap();
        idx.tag_episode(
            &engine,
            ep2,
            &bob,
            bob_contact.symbol_id,
            &[topic_rust],
            &preds,
        )
        .unwrap();

        let shared = idx.shared_context_between(&alice, &bob);
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0], topic_rust);
    }

    #[test]
    fn prune_at_limit() {
        let (engine, mut mgr, mut idx) = setup();
        idx.max_per_contact = 3;

        let cid = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let contact = mgr.get(&cid).unwrap();
        let preds = mgr.preds();

        for i in 0..5 {
            let ep = engine
                .resolve_or_create_entity(&format!("episode:{i}"))
                .unwrap();
            idx.tag_episode(&engine, ep, &cid, contact.symbol_id, &[], preds)
                .unwrap();
        }

        let recalled = idx.recall_with_person(&cid, 100);
        assert_eq!(recalled.len(), 3); // pruned to max
    }

    #[test]
    fn rebuild_from_kg() {
        let (engine, mut mgr, mut idx) = setup();

        let cid = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let contact = mgr.get(&cid).unwrap();
        let preds = mgr.preds();

        let ep1 = engine.resolve_or_create_entity("episode:1").unwrap();
        idx.tag_episode(&engine, ep1, &cid, contact.symbol_id, &[], preds)
            .unwrap();

        // Create a fresh index and rebuild from KG.
        let mut fresh = PersonMemoryIndex::new(100);
        fresh.rebuild_from_kg(&engine, &mgr, preds);

        assert_eq!(fresh.indexed_contact_count(), 1);
    }

    #[test]
    fn topics_discussed_ranked() {
        let (engine, mut mgr, mut idx) = setup();

        let cid = mgr
            .create_contact(&engine, "Alice", &["alice@a.com".to_string()])
            .unwrap();
        let contact = mgr.get(&cid).unwrap();
        let preds = mgr.preds();

        let topic_rust = engine.resolve_or_create_entity("topic:rust").unwrap();
        let topic_python = engine.resolve_or_create_entity("topic:python").unwrap();

        // Rust discussed in 2 episodes, Python in 1.
        let ep1 = engine.resolve_or_create_entity("episode:1").unwrap();
        let ep2 = engine.resolve_or_create_entity("episode:2").unwrap();

        idx.tag_episode(
            &engine,
            ep1,
            &cid,
            contact.symbol_id,
            &[topic_rust, topic_python],
            preds,
        )
        .unwrap();
        idx.tag_episode(
            &engine,
            ep2,
            &cid,
            contact.symbol_id,
            &[topic_rust],
            preds,
        )
        .unwrap();

        let topics = idx.topics_discussed_with(&cid);
        assert_eq!(topics[0].0, topic_rust);
        assert_eq!(topics[0].1, 2);
    }
}
