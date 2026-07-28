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

    /// Build a retriever keyed on an arbitrary per-tool text (e.g. the tool
    /// **description** rather than its name), parallel to the registry the
    /// caller will index against. Tokens are the text's lowercased alphanumeric
    /// words — the same tokenization requests get — so `covers`/`retrieved`/
    /// recall behave identically. This is what lets the P3d recall matrix
    /// compare name-keyed vs description-keyed retrieval on one code path.
    pub fn new_over<S: AsRef<str>>(texts: &[S]) -> Self {
        let tool_tokens = texts.iter().map(|t| word_tokens(t.as_ref())).collect();
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

    /// Does tool `tool_idx` **fully cover** `request` — is *every* one of its
    /// `verb_noun` name tokens present in the request? This is the strong
    /// coverage signal (both tokens, not just one) that makes the true tool the
    /// unique top hit; the organ's gap-response loop uses it to tell a genuine
    /// retrieval miss (a real tool that fully covers the request but ranked
    /// outside the offered top-K) apart from a mere partial-overlap confusable.
    pub fn covers(&self, request: &str, tool_idx: usize) -> bool {
        let request_tokens = tokenize(request);
        let toks = &self.tool_tokens[tool_idx];
        !toks.is_empty() && toks.iter().all(|t| request_tokens.contains(t.as_str()))
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

/// Lowercased whole-word tokens of arbitrary text (split on non-alphanumeric),
/// the tool-side signal for a description-keyed [`Retriever::new_over`].
fn word_tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
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

// ===========================================================================
// Semantic (HDC / VSA) retriever — Phase 51 P3a follow-up.
//
// The lexical `Retriever` above scores by *exact* name-token overlap, which is
// why its recall@8 is ~1.0: the procedural phrasings embed the tool's verb+noun
// tokens verbatim, so an exact-token retriever always finds them. That is the
// standing caveat — the retriever and the training *label* share one signal
// (the name tokens), so the 1.0 could be an artifact of surface coupling rather
// than evidence a realistic router would cover the request.
//
// This `SemanticRetriever` is a step toward the real (VSA) router: it encodes
// each tool and each request as a **binary hypervector** over character
// trigrams (hash-projected, bundled by majority) and ranks by Hamming
// similarity. It is *surface-semantic*, not a learned embedding — on this
// corpus the only tool-side content is the name (descriptions merely echo it),
// so an HDC retriever still keys on the name's surface form. What it adds over
// exact-token matching is **robustness**: char trigrams degrade gracefully
// under typos / morphological drift that break exact-token overlap outright.
//
// The honest experiment (see `semantic_vs_lexical_*` tests): on *clean*
// requests both retrievers hit ~1.0 (both key on the verbatim name); under
// request-side typos to the name tokens, the exact-token lexical retriever
// collapses while the HDC retriever largely survives. Neither is a learned
// semantic model, so neither can be claimed to retrieve under genuine
// *paraphrase* (synonyms with no shared surface) — that is the learned Phase 28
// VSA router's job, and this name-only corpus cannot stand in for that test.
// ===========================================================================

/// Hypervector dimensionality in bits (a multiple of 64). 2048 bits is ample
/// separation for a few-thousand-tool registry while staying cheap to build and
/// compare (32 × `u64` popcount per Hamming).
const HDC_DIM: usize = 2048;
const HDC_WORDS: usize = HDC_DIM / 64;

/// A packed binary hypervector.
type HyperVec = [u64; HDC_WORDS];

/// A retriever that ranks tools by **Hamming similarity of character-trigram
/// hypervectors** rather than exact name-token overlap. Tool hypervectors are
/// precomputed at construction; [`SemanticRetriever::top_k`] encodes the request
/// once and ranks the registry by bit agreement.
#[derive(Debug, Clone)]
pub struct SemanticRetriever {
    tool_hvs: Vec<HyperVec>,
}

impl SemanticRetriever {
    /// Build over `registry`, encoding each tool's `name` (underscores treated
    /// as spaces so `fetch_ledger` trigrams read like the phrase) plus its
    /// description into a hypervector.
    pub fn new(registry: &[ToolSchema]) -> Self {
        let tool_hvs = registry.iter().map(encode_tool).collect();
        Self { tool_hvs }
    }

    /// Build over an arbitrary per-tool text (e.g. the description alone),
    /// parallel to the registry — the HDC counterpart to [`Retriever::new_over`],
    /// so the P3d recall matrix can compare name-keyed vs description-keyed HDC
    /// retrieval.
    pub fn new_over<S: AsRef<str>>(texts: &[S]) -> Self {
        let tool_hvs = texts.iter().map(|t| encode_text(t.as_ref())).collect();
        Self { tool_hvs }
    }

    pub fn len(&self) -> usize {
        self.tool_hvs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tool_hvs.is_empty()
    }

    /// Bit-agreement (Hamming similarity) of a tool against a request HV: the
    /// count of matching bits, in `0..=HDC_DIM`. Higher = more similar.
    fn similarity(&self, tool_idx: usize, request_hv: &HyperVec) -> u32 {
        let tool = &self.tool_hvs[tool_idx];
        let mut agree = 0u32;
        for w in 0..HDC_WORDS {
            agree += (!(tool[w] ^ request_hv[w])).count_ones();
        }
        agree
    }

    /// Indices of the top-`k` registry tools for `request`, most similar first;
    /// ties broken by ascending index (deterministic, like [`Retriever::top_k`]).
    pub fn top_k(&self, request: &str, k: usize) -> Vec<usize> {
        let hv = encode_text(request);
        let k = k.min(self.tool_hvs.len());
        let mut scored: Vec<(u32, usize)> =
            (0..self.tool_hvs.len()).map(|i| (self.similarity(i, &hv), i)).collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        scored.into_iter().take(k).map(|(_, i)| i).collect()
    }

    /// Whether `target_idx` ranks within the top-`k` by Hamming similarity.
    /// Unlike the lexical retriever there is no "zero overlap" floor — every HV
    /// has *some* agreement — so a target dragged in only because the registry
    /// is smaller than `k` is still excluded by the `rank < k` test, but there
    /// is no separate nonzero-overlap gate (similarity is effectively always
    /// positive). This is the honest recall predicate for the HDC retriever.
    pub fn retrieved(&self, request: &str, target_idx: usize, k: usize) -> bool {
        let hv = encode_text(request);
        let target_sim = self.similarity(target_idx, &hv);
        let rank = (0..self.tool_hvs.len())
            .filter(|&i| i != target_idx)
            .filter(|&i| {
                let s = self.similarity(i, &hv);
                s > target_sim || (s == target_sim && i < target_idx)
            })
            .count();
        rank < k
    }
}

/// Encode a tool into a hypervector: its name (with `_`→space) and description.
fn encode_tool(tool: &ToolSchema) -> HyperVec {
    let text = format!("{} {}", tool.name.replace('_', " "), tool.description);
    encode_text(&text)
}

/// Encode text into a binary hypervector: bundle (majority-vote) the random
/// projection of every character trigram of the lowercased text. Deterministic
/// — no RNG state, each trigram maps to a fixed pseudo-random HV via its hash.
fn encode_text(text: &str) -> HyperVec {
    let lower = text.to_ascii_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    // Per-bit signed accumulator for the majority bundle.
    let mut counts = [0i32; HDC_DIM];
    let mut any = false;
    // Character trigrams (sliding window of 3); for very short text, fall back
    // to the whole thing as one gram so it still encodes to something.
    if chars.len() >= 3 {
        for w in chars.windows(3) {
            accumulate_gram(w, &mut counts);
            any = true;
        }
    } else if !chars.is_empty() {
        accumulate_gram(&chars, &mut counts);
        any = true;
    }
    let mut hv = [0u64; HDC_WORDS];
    if !any {
        return hv;
    }
    for (bit, &c) in counts.iter().enumerate() {
        if c > 0 {
            hv[bit / 64] |= 1u64 << (bit % 64);
        }
    }
    hv
}

/// Add one character gram's random-projection vector (±1 per bit, derived
/// deterministically from the gram's hash) into the signed accumulator.
fn accumulate_gram(gram: &[char], counts: &mut [i32; HDC_DIM]) {
    let mut seed = fnv1a(gram);
    for word in 0..HDC_WORDS {
        let bits = splitmix64(&mut seed);
        for b in 0..64 {
            let idx = word * 64 + b;
            if (bits >> b) & 1 == 1 {
                counts[idx] += 1;
            } else {
                counts[idx] -= 1;
            }
        }
    }
}

/// FNV-1a over a character gram (each char folded as its UTF-32 code point),
/// seeding the per-gram projection.
fn fnv1a(gram: &[char]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &c in gram {
        for byte in (c as u32).to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x00000100000001B3);
        }
    }
    // Avoid a zero state feeding splitmix (still fine, but keep it lively).
    h | 1
}

/// SplitMix64 step — a fast, well-distributed PRNG for expanding a gram seed
/// into `HDC_DIM` pseudo-random bits.
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Semantic (HDC) retrieval **recall@k** over `(request, target_idx)` pairs —
/// the [`retrieval_recall_at_k`] counterpart for [`SemanticRetriever`].
pub fn semantic_recall_at_k(
    retriever: &SemanticRetriever,
    pairs: &[(String, usize)],
    k: usize,
) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let hits = pairs
        .iter()
        .filter(|(req, target)| retriever.retrieved(req, *target, k))
        .count();
    hits as f64 / pairs.len() as f64
}

// ===========================================================================
// Learned-embedding retriever — Phase 51 P3d-2.
//
// The recall matrix (P3d-1) showed that under genuine paraphrase (D2) every
// *surface* retriever — exact-token lexical and char-trigram HDC — floors at
// ~0.01–0.02, because the request (alt gloss) and the description (primary
// gloss) share no surface tokens. Only a *learned* semantic embedding can
// bridge synonyms. This retriever ranks by cosine over precomputed embedding
// vectors; the embedding itself is done *outside* grug-data (which stays
// ML-free) — e.g. Seshat's ONNX all-MiniLM-L6-v2 (`crates/seshat/examples/
// semantic_recall_minilm.rs`) — so this type owns only the cosine ranking, and
// the recall predicate (`rank < k`) stays byte-for-byte identical to the
// lexical/HDC retrievers, keeping the matrix a fair comparison.
// ===========================================================================

/// A retriever over precomputed, **L2-normalized** per-tool embedding vectors.
/// Cosine similarity is a plain dot product on normalized vectors.
#[derive(Debug, Clone)]
pub struct EmbedRetriever {
    tool_vecs: Vec<Vec<f32>>,
}

impl EmbedRetriever {
    /// Build over per-tool embedding vectors, parallel to the registry the
    /// caller indexes against. Vectors should be L2-normalized (as Seshat's
    /// embedder returns) so `score` is cosine.
    pub fn new(tool_vecs: Vec<Vec<f32>>) -> Self {
        Self { tool_vecs }
    }

    pub fn len(&self) -> usize {
        self.tool_vecs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tool_vecs.is_empty()
    }

    /// Cosine similarity (dot product of L2-normalized vectors) of tool
    /// `tool_idx` against a query embedding. `0.0` on a dimension mismatch.
    fn score(&self, tool_idx: usize, query: &[f32]) -> f32 {
        let v = &self.tool_vecs[tool_idx];
        if v.len() != query.len() {
            return 0.0;
        }
        v.iter().zip(query).map(|(a, b)| a * b).sum()
    }

    /// Whether `target_idx` ranks within the top-`k` by cosine similarity to
    /// `query` (ties broken by ascending index) — the same `rank < k` predicate
    /// the surface retrievers use.
    pub fn retrieved(&self, query: &[f32], target_idx: usize, k: usize) -> bool {
        let target_sim = self.score(target_idx, query);
        let rank = (0..self.tool_vecs.len())
            .filter(|&i| i != target_idx)
            .filter(|&i| {
                let s = self.score(i, query);
                s > target_sim || (s == target_sim && i < target_idx)
            })
            .count();
        rank < k
    }
}

/// Learned-embedding retrieval **recall@k** over `(query_embedding, target_idx)`
/// pairs — the [`retrieval_recall_at_k`] counterpart for [`EmbedRetriever`].
pub fn embed_recall_at_k(retriever: &EmbedRetriever, pairs: &[(Vec<f32>, usize)], k: usize) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let hits = pairs
        .iter()
        .filter(|(q, target)| retriever.retrieved(q, *target, k))
        .count();
    hits as f64 / pairs.len() as f64
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

    /// Draw `n` `(request, target_idx)` pairs from the procedural registry —
    /// shared by the semantic-retriever tests below.
    fn sample_pairs(seed: u64, n_tools: usize, n: usize) -> (Vec<ToolSchema>, Vec<(String, usize)>) {
        use rand::Rng;
        use rand::SeedableRng;
        let catalog = build_catalog(42, n_tools);
        let registry = catalog.tools().to_vec();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut pairs = Vec::new();
        for _ in 0..n {
            let idx = rng.gen_range(0..registry.len());
            let bank = catalog.phrasings().get(&registry[idx].name).unwrap();
            let mut args = serde_json::Map::new();
            for p in &registry[idx].params {
                args.insert(p.name.clone(), crate::generator::fuzz_value(&mut rng, &p.ty));
            }
            let call = crate::schema::ToolCall::new(registry[idx].name.clone(), serde_json::Value::Object(args));
            let req = bank.render_train(&mut rng, &call);
            pairs.push((req, idx));
        }
        (registry, pairs)
    }

    /// Substitute one middle character in every request word that equals one of
    /// `name`'s `verb_noun` tokens — enough to break an *exact-token* match while
    /// leaving most character trigrams intact.
    fn corrupt_name_tokens(request: &str, name: &str) -> String {
        let toks: HashSet<String> = name.split('_').map(|s| s.to_ascii_lowercase()).collect();
        request
            .split_whitespace()
            .map(|w| {
                let key: String = w.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
                if toks.contains(&key) && w.chars().count() >= 2 {
                    let mut cs: Vec<char> = w.chars().collect();
                    let i = cs.len() / 2;
                    cs[i] = if cs[i] == 'x' { 'q' } else { 'x' };
                    cs.into_iter().collect()
                } else {
                    w.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn semantic_recall_matches_lexical_on_clean_requests() {
        let (registry, pairs) = sample_pairs(7, 500, 200);
        let lexical = Retriever::new(&registry);
        let semantic = SemanticRetriever::new(&registry);
        let lex = retrieval_recall_at_k(&lexical, &pairs, 8);
        let sem = semantic_recall_at_k(&semantic, &pairs, 8);
        // Both key on the verbatim name embedded in the request, so both are
        // high on clean requests — the caveat, made concrete.
        assert!(lex > 0.9, "clean lexical recall@8 = {lex}");
        assert!(sem > 0.9, "clean semantic recall@8 = {sem}");
    }

    #[test]
    fn embed_retriever_ranks_by_cosine() {
        // Three near-orthogonal unit vectors; a query closest to tool 1.
        let tools = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let r = EmbedRetriever::new(tools);
        let query = vec![0.1, 0.99, 0.0]; // closest to tool 1
        assert!(r.retrieved(&query, 1, 1), "nearest tool is retrieved at k=1");
        assert!(!r.retrieved(&query, 0, 1), "a far tool is not top-1");
        // recall helper over a batch.
        let pairs = vec![(vec![0.9, 0.1, 0.0], 0usize), (vec![0.0, 0.2, 0.98], 2usize)];
        assert_eq!(embed_recall_at_k(&r, &pairs, 1), 1.0);
    }

    #[test]
    fn semantic_survives_name_typos_that_break_lexical() {
        let (registry, pairs) = sample_pairs(9, 500, 200);
        let lexical = Retriever::new(&registry);
        let semantic = SemanticRetriever::new(&registry);

        let clean_lex = retrieval_recall_at_k(&lexical, &pairs, 8);

        let noisy: Vec<(String, usize)> = pairs
            .iter()
            .map(|(req, idx)| (corrupt_name_tokens(req, &registry[*idx].name), *idx))
            .collect();
        let noisy_lex = retrieval_recall_at_k(&lexical, &noisy, 8);
        let noisy_sem = semantic_recall_at_k(&semantic, &noisy, 8);

        // Exact-token overlap collapses under name typos...
        assert!(
            noisy_lex < clean_lex - 0.1,
            "lexical recall should degrade materially under name typos: clean={clean_lex} noisy={noisy_lex}"
        );
        // ...while the char-trigram HDC retriever is far more robust (rank order
        // survives even as absolute bit-agreement drops).
        assert!(
            noisy_sem > noisy_lex + 0.1,
            "semantic recall should beat lexical under typos: sem={noisy_sem} lex={noisy_lex}"
        );
    }
}
