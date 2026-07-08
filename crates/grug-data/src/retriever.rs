//! Retrieval-scoped tool universe (Phase 51 P3a, ADR 089 mechanism 1).
//!
//! ADR 089's first mechanism: **there is no genuine open-world zero-shot in
//! normal operation.** The tool universe *is* the KG's capability/skillpack
//! registry, and at runtime a retrieval step selects the top-K request-relevant
//! tools and offers only their schemas. The model is trained on the whole
//! registry, so "offered tools" are almost always tools it has *seen* — the
//! whole-tool-holdout scenario (the neural floor at ~0.03) is one the organ
//! never faces. Zero-shot correctness is dissolved by *scoping*, not demanded
//! from the weights.
//!
//! # This is a lexical stand-in for the real (VSA) router
//!
//! In the live system the retriever is the Phase 28 semantic router / Phase 37
//! Seshat / VSA similarity over `goal ↔ tool` hypervectors. Here it is a
//! deterministic **lexical** retriever: it scores each registry tool by how
//! many of its `verb_noun` name tokens appear in the request text. This is a
//! *faithful* proxy, not a cheat — the procedural phrasings embed the tool's
//! verb+noun directly in every request (`verb_phrase_for`: `fetch_ledger` →
//! "…fetch the ledger…"), which is exactly the semantic signal an embedding
//! router keys on. It is also non-trivial: many tools share one token
//! (`fetch_ledger` / `archive_ledger` / `fetch_beacon`), so the top-K naturally
//! contains request-keyed *confusables* — the realistic hard offered set.

use std::collections::HashSet;

use crate::schema::ToolSchema;

/// A retriever over a fixed tool **registry**: given a request, returns the
/// indices of the top-K most relevant tools. Per-tool name-token sets are
/// precomputed once at construction (like `generator::ToolFingerprint`) so
/// scoring a request against a thousand-tool registry stays cheap enough to run
/// per generated example.
#[derive(Debug, Clone)]
pub struct Retriever {
    /// Per-tool lowercased name tokens (`verb_noun` split on `_`), parallel to
    /// the registry slice this was built from.
    tool_tokens: Vec<Vec<String>>,
}

impl Retriever {
    /// Build a retriever over `registry` (indices returned by [`Retriever::top_k`]
    /// index into this same slice).
    pub fn new(registry: &[ToolSchema]) -> Self {
        let tool_tokens = registry.iter().map(|t| name_tokens(&t.name)).collect();
        Self { tool_tokens }
    }

    /// Number of tools in the registry this retriever indexes.
    pub fn len(&self) -> usize {
        self.tool_tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tool_tokens.is_empty()
    }

    /// Score one registry tool against a pre-tokenized request: the count of the
    /// tool's name tokens present in the request. (Count, not Jaccard, so a
    /// two-token exact match always outranks a one-token partial — the property
    /// that makes the true tool the unique top hit when the request names both
    /// its verb and noun.)
    fn score(&self, tool_idx: usize, request_tokens: &HashSet<&str>) -> usize {
        self.tool_tokens[tool_idx]
            .iter()
            .filter(|tok| request_tokens.contains(tok.as_str()))
            .count()
    }

    /// Return the indices of the top-`k` registry tools for `request`, most
    /// relevant first. Ties (equal token-overlap) are broken by ascending tool
    /// index, so selection is a deterministic function of registry order — no
    /// RNG, matching how a real router returns a stable ranked list. Tools with
    /// zero overlap are still returned to fill `k` (a real top-K always returns
    /// `k` candidates); callers who care whether the target was *genuinely*
    /// retrieved use [`Retriever::retrieved`] / recall, not mere membership.
    pub fn top_k(&self, request: &str, k: usize) -> Vec<usize> {
        let request_tokens = tokenize(request);
        let k = k.min(self.tool_tokens.len());
        // (score, idx); sort by score desc then idx asc.
        let mut scored: Vec<(usize, usize)> = (0..self.tool_tokens.len())
            .map(|i| (self.score(i, &request_tokens), i))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    /// Whether `target_idx` is among the top-`k` retrieved for `request` **with
    /// nonzero overlap** — i.e. genuinely surfaced by relevance, not merely
    /// padding out `k`. This is the honest recall predicate: a target dragged
    /// into the top-K only because the registry is smaller than `k`, or only as
    /// a zero-score filler, does not count as retrieved.
    pub fn retrieved(&self, request: &str, target_idx: usize, k: usize) -> bool {
        let request_tokens = tokenize(request);
        if self.score(target_idx, &request_tokens) == 0 {
            return false;
        }
        let target_score = self.score(target_idx, &request_tokens);
        // Rank = number of tools strictly better, or equal-score-and-earlier.
        let rank = (0..self.tool_tokens.len())
            .filter(|&i| i != target_idx)
            .filter(|&i| {
                let s = self.score(i, &request_tokens);
                s > target_score || (s == target_score && i < target_idx)
            })
            .count();
        rank < k
    }
}

/// Lowercased `verb_noun` name tokens for one tool (`fetch_ledger` →
/// `["fetch","ledger"]`), the tool-side retrieval signal.
fn name_tokens(name: &str) -> Vec<String> {
    name.split('_')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

/// Tokenize a request into a lowercased word set for overlap scoring: split on
/// any non-alphanumeric char (so punctuation and the em-dash separators used by
/// the procedural shapes don't fuse to adjacent words), drop empties.
fn tokenize(request: &str) -> HashSet<&str> {
    request
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Retrieval **recall@k** over a batch of `(request, target_idx)` pairs: the
/// fraction whose target was genuinely retrieved into the top-`k` (see
/// [`Retriever::retrieved`]). This is the model-independent premise check for
/// ADR 089 mechanism 1 — if it is high, the offered set almost always contains
/// the target, so the model rarely faces a truly-unseen offered tool. `0.0` for
/// an empty batch.
pub fn retrieval_recall_at_k(retriever: &Retriever, pairs: &[(String, usize)], k: usize) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let hits = pairs
        .iter()
        .filter(|(req, target)| retriever.retrieved(req, *target, k))
        .count();
    hits as f64 / pairs.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedural::build_catalog;
    use crate::schema::{ParamSpec, ParamType, ToolSchema};

    fn tool(name: &str) -> ToolSchema {
        ToolSchema::new(name, "d", vec![ParamSpec::new("q", ParamType::String, true)])
    }

    #[test]
    fn exact_two_token_match_outranks_partial() {
        let registry = vec![
            tool("fetch_ledger"),
            tool("archive_ledger"), // shares "ledger"
            tool("fetch_beacon"),   // shares "fetch"
            tool("deploy_widget"),  // shares nothing
        ];
        let r = Retriever::new(&registry);
        // A request naming both the verb and the noun of tool 0.
        let req = "Please fetch the ledger for Zurich.";
        let top = r.top_k(req, 3);
        assert_eq!(top[0], 0, "the two-token exact match must rank first");
        // The two one-token confusables come next (order by index), the
        // zero-overlap tool is last / excluded from top-3.
        assert!(top.contains(&1) && top.contains(&2));
        assert!(!top.contains(&3));
        assert!(r.retrieved(req, 0, 3));
        assert!(r.retrieved(req, 1, 3)); // "ledger" overlaps → genuinely retrieved
    }

    #[test]
    fn zero_overlap_target_is_not_counted_as_retrieved() {
        let registry = vec![tool("fetch_ledger"), tool("deploy_widget")];
        let r = Retriever::new(&registry);
        // Request for tool 0; tool 1 shares no token, so even though top_k(_,2)
        // returns both (padding to k), tool 1 is NOT "retrieved".
        let req = "fetch the ledger now";
        assert!(!r.retrieved(req, 1, 2));
        assert!(r.retrieved(req, 0, 2));
    }

    #[test]
    fn recall_at_k_is_high_on_real_registry() {
        // On the procedural registry, each request embeds its tool's verb+noun,
        // and the two-token match is (almost always) unique → recall@8 ≈ 1.0.
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let catalog = build_catalog(42, 500);
        let registry = catalog.tools();
        let r = Retriever::new(registry);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);

        let mut pairs = Vec::new();
        for _ in 0..200 {
            let idx = (0..registry.len()).collect::<Vec<_>>().choose(&mut rng).copied().unwrap();
            let bank = catalog.phrasings().get(&registry[idx].name).unwrap();
            // Build a trivial call so the phrasing renders (values don't affect
            // name-token retrieval).
            let mut args = serde_json::Map::new();
            for p in &registry[idx].params {
                args.insert(p.name.clone(), crate::generator::fuzz_value(&mut rng, &p.ty));
            }
            let call = crate::schema::ToolCall::new(registry[idx].name.clone(), serde_json::Value::Object(args));
            let req = bank.render_train(&mut rng, &call);
            pairs.push((req, idx));
        }
        let recall = retrieval_recall_at_k(&r, &pairs, 8);
        assert!(recall > 0.9, "recall@8 should be high on the procedural registry, got {recall}");
    }
}
