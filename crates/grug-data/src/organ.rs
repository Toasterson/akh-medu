//! The organ's runtime response to a **fired gap** (Phase 51 P3a, ADR 089
//! mechanism 3's back half).
//!
//! Mechanism 3 (P3c) taught the model to emit the `__gap__` token when no
//! offered tool fits. That token is only half the mechanism — it says *"the
//! offered set didn't cover this"*, but the organ still has to *do* something.
//! ADR 089's answer: a gap is not a dead end, it is a **control signal** back to
//! the symbolic retriever (mechanism 1). Two things could have caused the gap:
//!
//! 1. **Retrieval miss** — a tool that genuinely covers the request *exists in
//!    the registry* but ranked outside the offered top-K, so the model never saw
//!    it. The fix is to **widen** retrieval (offer a larger K) and retry; the
//!    covering tool now appears and the call succeeds. The neural floor is never
//!    touched — this is still a tool the model was trained on.
//! 2. **Genuine novelty** — no registry tool covers the request at *any* K. The
//!    request is outside the registry's capability surface; widening only ever
//!    surfaces partial-overlap **confusables**, never a covering tool. The
//!    correct terminal is **synthesis** (ask the user / synthesize a new
//!    skillpack), *never* fabricate a call to a confusable.
//!
//! The load-bearing distinction is *coverage*, not mere overlap: [`Retriever::covers`]
//! requires **every** name token present (the strong two-token signal), so a
//! `fetch_zebra` novelty request that partially overlaps `fetch_ledger` on
//! `fetch` does **not** count as covered — it terminates in [`GapResponse::Synthesize`],
//! not a wrong widen. That is what keeps the widen loop from hallucinating
//! capability it doesn't have.
//!
//! This module is the deterministic, model-independent *policy + measurement*.
//! Wiring a live model's emitted gap into [`respond_to_gap`] (re-offer the
//! widened set, re-generate) is the integration point; the synthesis branch is
//! deliberately a seam (ADR 089 leaves skill synthesis to the KG/agent layer).

use std::collections::HashSet;

use crate::retriever::Retriever;

/// What the organ does when the model fires a gap on a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapResponse {
    /// Widening retrieval to `width` surfaced tool `tool_idx`, which **covers**
    /// the request (all its name tokens present) and was *not* in the original
    /// offered set — a recovered retrieval miss. The organ re-offers the widened
    /// set and retries the call.
    Widen { width: usize, tool_idx: usize },
    /// No widening (up to the whole registry) surfaced a covering tool: the
    /// request is genuinely out of the registry's capability. Hand to synthesis
    /// — never fabricate a call to a partial-overlap confusable.
    Synthesize,
}

/// Decide the organ's response to a gap the model fired on `request`, given the
/// offered set was the retriever's top-`from_k`. Widen along `widen_schedule`
/// (ascending widths, e.g. `[16, 64, registry_len]`); the first width that
/// surfaces a **newly-offered covering tool** yields [`GapResponse::Widen`].
/// If none do, [`GapResponse::Synthesize`].
///
/// A covering tool that was *already* in the `from_k` offered set is not a widen
/// candidate — if the model gapped despite being offered a covering tool that is
/// a model error (a false abstention), not a retrieval miss, and re-offering the
/// same tool won't help. The widen loop only rescues tools the offered set was
/// too narrow to include.
pub fn respond_to_gap(
    retriever: &Retriever,
    request: &str,
    from_k: usize,
    widen_schedule: &[usize],
) -> GapResponse {
    let base: HashSet<usize> = retriever.top_k(request, from_k).into_iter().collect();
    for &width in widen_schedule {
        if width <= from_k {
            continue;
        }
        for idx in retriever.top_k(request, width) {
            if !base.contains(&idx) && retriever.covers(request, idx) {
                return GapResponse::Widen {
                    width,
                    tool_idx: idx,
                };
            }
        }
    }
    GapResponse::Synthesize
}

/// Model-independent measurement of the gap→widen→synthesize loop (ADR 089
/// mechanism 3 back half). Splits gaps by their true cause and reports how the
/// organ handles each:
///
/// - Over **scoped** `(request, target)` pairs whose target ranked *outside*
///   `from_k` (a genuine retrieval miss the model would correctly gap on):
///   how many does widening recover?
/// - Over **novelty** requests (phrased for out-of-registry tools): how many
///   correctly terminate in [`GapResponse::Synthesize`] vs. how many are wrongly
///   "recovered" by a confusable (`novelty_false_widen` — should be ~0, the
///   coverage test's whole job).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GapRecoveryReport {
    pub from_k: usize,
    pub max_width: usize,
    /// Scoped pairs whose target ranked outside `from_k` — the retrieval-miss
    /// gaps the widen loop exists to rescue.
    pub retrieval_misses: usize,
    /// Of those misses, how many are recovered (target within `max_width`).
    pub widen_recovered: usize,
    /// Novelty requests measured.
    pub novelty: usize,
    /// Novelty requests correctly handed to synthesis (no covering tool at any
    /// width).
    pub novelty_synthesized: usize,
    /// Novelty requests a confusable wrongly "covered" (organ widened instead of
    /// synthesizing) — the coverage test's failure count.
    pub novelty_false_widen: usize,
}

impl GapRecoveryReport {
    /// Measure the loop. `scoped_pairs` are in-registry `(request, target_idx)`;
    /// `novelty_requests` are out-of-registry request texts; `widen_schedule`
    /// is the ascending widths to try (its max is `max_width`).
    pub fn measure(
        retriever: &Retriever,
        scoped_pairs: &[(String, usize)],
        novelty_requests: &[String],
        from_k: usize,
        widen_schedule: &[usize],
    ) -> Self {
        let max_width = widen_schedule
            .iter()
            .copied()
            .max()
            .unwrap_or(from_k)
            .max(from_k);

        let mut retrieval_misses = 0;
        let mut widen_recovered = 0;
        for (req, target) in scoped_pairs {
            // A miss is a covering target that ranked outside the offered top-K.
            if retriever.retrieved(req, *target, from_k) {
                continue; // covered by the offered set → not a gap the widen loop owns
            }
            retrieval_misses += 1;
            if retriever.retrieved(req, *target, max_width) {
                widen_recovered += 1;
            }
        }

        let mut novelty_synthesized = 0;
        let mut novelty_false_widen = 0;
        for req in novelty_requests {
            match respond_to_gap(retriever, req, from_k, widen_schedule) {
                GapResponse::Synthesize => novelty_synthesized += 1,
                GapResponse::Widen { .. } => novelty_false_widen += 1,
            }
        }

        Self {
            from_k,
            max_width,
            retrieval_misses,
            widen_recovered,
            novelty: novelty_requests.len(),
            novelty_synthesized,
            novelty_false_widen,
        }
    }

    /// Fraction of retrieval-miss gaps recovered by widening (`0.0` if none).
    pub fn recovery_rate(&self) -> f64 {
        if self.retrieval_misses == 0 {
            0.0
        } else {
            self.widen_recovered as f64 / self.retrieval_misses as f64
        }
    }

    /// Fraction of novelty gaps correctly terminated in synthesis (`0.0` if
    /// none).
    pub fn synthesis_rate(&self) -> f64 {
        if self.novelty == 0 {
            0.0
        } else {
            self.novelty_synthesized as f64 / self.novelty as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedural::build_catalog;
    use crate::schema::ToolCall;

    /// Build `(request, target_idx)` pairs from a catalog's own phrasings.
    fn scoped_pairs(
        seed: u64,
        registry: &[crate::schema::ToolSchema],
        catalog: &crate::catalog::ToolCatalog,
        n: usize,
    ) -> Vec<(String, usize)> {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut pairs = Vec::new();
        for _ in 0..n {
            let idx = rng.gen_range(0..registry.len());
            let bank = catalog.phrasings().get(&registry[idx].name).unwrap();
            let mut args = serde_json::Map::new();
            for p in &registry[idx].params {
                args.insert(
                    p.name.clone(),
                    crate::generator::fuzz_value(&mut rng, &p.ty),
                );
            }
            let call = ToolCall::new(registry[idx].name.clone(), serde_json::Value::Object(args));
            pairs.push((bank.render_train(&mut rng, &call), idx));
        }
        pairs
    }

    #[test]
    fn widening_recovers_low_k_misses_and_synthesizes_genuine_novelty() {
        // Registry A is what the retriever knows; catalog B (disjoint seed/size)
        // supplies genuine-novelty requests for tools A doesn't have.
        let catalog_a = build_catalog(42, 500);
        let registry: Vec<_> = catalog_a.tools().to_vec();
        let retriever = Retriever::new(&registry);
        let pairs = scoped_pairs(7, &registry, &catalog_a, 300);

        // Novelty requests: render from a differently-seeded catalog and keep
        // only those whose tool name is NOT in registry A (truly out-of-registry).
        let catalog_b = build_catalog(1234, 500);
        let known: HashSet<&str> = registry.iter().map(|t| t.name.as_str()).collect();
        let mut novelty = Vec::new();
        {
            use rand::Rng;
            use rand::SeedableRng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(99);
            let btools = catalog_b.tools();
            for _ in 0..300 {
                let idx = rng.gen_range(0..btools.len());
                if known.contains(btools[idx].name.as_str()) {
                    continue;
                }
                let bank = catalog_b.phrasings().get(&btools[idx].name).unwrap();
                let mut args = serde_json::Map::new();
                for p in &btools[idx].params {
                    args.insert(
                        p.name.clone(),
                        crate::generator::fuzz_value(&mut rng, &p.ty),
                    );
                }
                let call = ToolCall::new(btools[idx].name.clone(), serde_json::Value::Object(args));
                novelty.push(bank.render_train(&mut rng, &call));
            }
        }

        let schedule = [8usize, 16, 64, registry.len()];
        let report = GapRecoveryReport::measure(&retriever, &pairs, &novelty, 3, &schedule);

        // Honest finding: on this corpus the two-token exact match is unique, so
        // recall is already saturated even at k=3 → essentially *no* natural
        // retrieval misses exist (retrieval is not the bottleneck). Whatever few
        // misses do occur, widening must recover; if there are none, that is the
        // result, not a failure. (The widen *mechanism* is proven directly in
        // `widen_recovers_a_covering_tool_outside_from_k`.)
        if report.retrieval_misses > 0 {
            assert!(
                report.recovery_rate() > 0.8,
                "widening should recover most low-k misses: {}/{}",
                report.widen_recovered,
                report.retrieval_misses
            );
        }
        // The load-bearing half on this corpus: genuine novelty must
        // overwhelmingly go to synthesis, not a confusable widen — the coverage
        // (all-tokens) test's whole purpose.
        assert!(report.novelty > 0);
        assert!(
            report.synthesis_rate() > 0.9,
            "genuine novelty should synthesize, not false-widen: synth={} false_widen={} n={}",
            report.novelty_synthesized,
            report.novelty_false_widen,
            report.novelty
        );
    }

    /// Directly exercise the widen mechanism with a hand-built registry where a
    /// covering tool is *forced* outside the narrow offered set by same-score
    /// confusables (so the corpus's natural recall-saturation doesn't hide it).
    #[test]
    fn widen_recovers_a_covering_tool_outside_from_k() {
        use crate::schema::{ParamSpec, ParamType, ToolSchema};
        let tool = |name: &str| {
            ToolSchema::new(
                name,
                "d",
                vec![ParamSpec::new("q", ParamType::String, true)],
            )
        };
        // Request names five tokens; four tools each match two of them (score 2),
        // ties broken by index. The target `alpha_beta` sits at index 3, so the
        // narrow top-3 offered set excludes it → a genuine retrieval miss.
        let registry = vec![
            tool("beta_gamma"),    // 0: beta+gamma
            tool("gamma_delta"),   // 1: gamma+delta
            tool("delta_epsilon"), // 2: delta+epsilon
            tool("alpha_beta"),    // 3: alpha+beta  ← the target
            tool("no_overlap"),    // 4: score 0
        ];
        let r = Retriever::new(&registry);
        let request = "alpha beta gamma delta epsilon";

        // Precondition: target is genuinely outside the k=3 offer but covers.
        assert!(!r.retrieved(request, 3, 3), "target should miss at k=3");
        assert!(r.covers(request, 3), "target fully covers the request");

        match respond_to_gap(&r, request, 3, &[8]) {
            GapResponse::Widen { tool_idx, .. } => {
                assert_eq!(tool_idx, 3, "widen recovers the covering target")
            }
            GapResponse::Synthesize => {
                panic!("should have widened to recover the covering tool, not synthesized")
            }
        }
    }

    /// A request with no covering tool at any width must terminate in synthesis,
    /// even when a confusable partially overlaps (shares one token).
    #[test]
    fn genuine_novelty_synthesizes_despite_partial_confusable() {
        use crate::schema::{ParamSpec, ParamType, ToolSchema};
        let tool = |name: &str| {
            ToolSchema::new(
                name,
                "d",
                vec![ParamSpec::new("q", ParamType::String, true)],
            )
        };
        let registry = vec![tool("fetch_ledger"), tool("archive_widget")];
        let r = Retriever::new(&registry);
        // Novel request: shares only `fetch` with tool 0 (partial), covers nothing.
        let request = "fetch the zebra from the paddock";
        assert!(!r.covers(request, 0) && !r.covers(request, 1));
        assert_eq!(
            respond_to_gap(&r, request, 1, &[2]),
            GapResponse::Synthesize
        );
    }
}
