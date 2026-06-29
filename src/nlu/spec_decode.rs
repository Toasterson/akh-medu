//! Speculative decoding for the Candle gemma boundary (Phase 47 / ADR 075).
//!
//! Decode is dispatch-bound (~8.8% of the bandwidth roofline → ~11× spare compute), so
//! verifying `γ+1` candidate tokens in **one** target forward costs ≈ one decode-time — spec
//! decode collapses up to `γ+1` sequential target forwards into a single one.
//!
//! gemma4-12b cannot use a separate draft model (its tokenizer differs from the gemma3 family
//! and no gemma4-mini exists), so the drafter here is **draft-model-free**: prompt-lookup
//! (n-gram) decoding, which proposes candidates from the target's own token history. It wins
//! where output echoes context (agentic / RAG / code / structured output) and is ~neutral
//! otherwise — a no-match simply yields an empty proposal and the caller takes one normal
//! decode step.
//!
//! This module is pure token logic (no model dependency): the [`NgramDrafter`] (47b) and the
//! greedy [`accept_run`] check (47c). The verify forward and KV rollback live in
//! [`crate::nlu::candle_backend`].

/// Configuration for the n-gram prompt-lookup drafter.
#[derive(Debug, Clone, Copy)]
pub struct DrafterConfig {
    /// Longest suffix n-gram to try matching first. A longer match is more specific → higher
    /// acceptance, so the drafter tries `max_ngram` down to `min_ngram`.
    pub max_ngram: usize,
    /// Shortest suffix n-gram to fall back to (`1` = match on the single last token).
    pub min_ngram: usize,
    /// γ — the maximum number of continuation tokens to propose per step. Small (≤ 8): a longer
    /// proposal that mismatches early just wastes the tail, and the verify forward grows with γ.
    pub max_draft: usize,
}

impl Default for DrafterConfig {
    fn default() -> Self {
        // A 3→1 suffix ladder with γ=8 is the common prompt-lookup-decoding default: selective
        // enough to keep acceptance high, cheap enough to verify in one forward.
        Self {
            max_ngram: 3,
            min_ngram: 1,
            max_draft: 8,
        }
    }
}

/// Draft-model-free prompt-lookup (n-gram) drafter (Phase 47b).
///
/// Proposes the next tokens by finding the **most recent earlier** occurrence of the current
/// suffix in the token history and returning the tokens that followed it — the basis of
/// prompt-lookup decoding. Uses only the target vocabulary, so it works for gemma4 where no
/// compatible separate draft model exists.
#[derive(Debug, Clone, Copy)]
pub struct NgramDrafter {
    cfg: DrafterConfig,
}

impl NgramDrafter {
    pub fn new(cfg: DrafterConfig) -> Self {
        Self { cfg }
    }

    /// The drafter's configuration.
    pub fn config(&self) -> DrafterConfig {
        self.cfg
    }

    /// Propose up to [`DrafterConfig::max_draft`] continuation tokens for `tokens` (the full
    /// prompt + generated history). Tries the longest suffix (`max_ngram`) first and returns the
    /// tokens that followed the most recent earlier occurrence of that suffix. Empty when no
    /// suffix down to `min_ngram` recurs earlier — the caller then takes one normal decode step.
    ///
    /// Cost is `O(len · max_ngram)` per call (a naive backward scan); fine for the contexts here.
    /// A suffix-automaton / rolling-hash index is a future optimization for very long contexts.
    pub fn propose(&self, tokens: &[u32]) -> Vec<u32> {
        let len = tokens.len();
        // A match needs both an n-token suffix and an earlier n-token window, so n ≤ len-1.
        let max_n = self.cfg.max_ngram.min(len.saturating_sub(1));
        for n in (self.cfg.min_ngram..=max_n).rev() {
            if n == 0 {
                continue;
            }
            let suffix = &tokens[len - n..];
            // Most recent earlier occurrence: scan candidate starts high → low. `p` ranges over
            // `[0, len - n)`, so the match never includes the suffix itself.
            for p in (0..len - n).rev() {
                if &tokens[p..p + n] == suffix {
                    let start = p + n; // < len, since p < len - n ⇒ p + n < len
                    let end = (start + self.cfg.max_draft).min(len);
                    return tokens[start..end].to_vec();
                }
            }
        }
        Vec::new()
    }
}

/// Greedy speculative-decode accept run (Phase 47c, pure half).
///
/// Given the `draft` tokens and the target's greedy `verify_argmax` at each corresponding
/// position (`verify_argmax[i]` is the target's argmax after consuming `draft[..=i]`'s prefix —
/// i.e. its prediction for `draft[i]`), returns how many leading draft tokens the target
/// confirms: the largest `k` such that `draft[i] == verify_argmax[i]` for all `i < k`. The
/// caller then emits those `k` draft tokens plus the bonus/correction token `verify_argmax[k]`
/// (which always exists because the verify forward scores one extra position), advancing up to
/// `k+1` tokens per target forward. Byte-exact-greedy by construction (every emitted token is
/// the target's own greedy choice under the verify forward).
///
/// `verify_argmax.len()` must be `draft.len() + 1` (the `+1` is the bonus position); the extra
/// entry is unused here but documents the caller's contract.
pub fn accept_run(draft: &[u32], verify_argmax: &[u32]) -> usize {
    draft
        .iter()
        .zip(verify_argmax.iter())
        .take_while(|(d, v)| d == v)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drafter(max_ngram: usize, min_ngram: usize, max_draft: usize) -> NgramDrafter {
        NgramDrafter::new(DrafterConfig {
            max_ngram,
            min_ngram,
            max_draft,
        })
    }

    #[test]
    fn proposes_continuation_of_recent_ngram() {
        // "the cat sat on the mat . | the cat sat" → suffix [1,2,3] recurs at p=0;
        // the proposal is what followed there: [4,5,6,7] (capped at max_draft=4).
        let toks = vec![1u32, 2, 3, 4, 5, 6, 7, 99, 1, 2, 3];
        assert_eq!(drafter(3, 1, 4).propose(&toks), vec![4, 5, 6, 7]);
    }

    #[test]
    fn picks_most_recent_occurrence() {
        // suffix [1] at the tail recurs at p=0 and p=1; the most recent (p=1) is used →
        // continuation starts at index 2: [9, 1].
        let toks = vec![1u32, 1, 9, 1];
        assert_eq!(drafter(1, 1, 8).propose(&toks), vec![9, 1]);
    }

    #[test]
    fn longer_ngram_preferred_over_shorter() {
        // n=3 suffix [4,2,3] has no earlier match; falls back to n=2 suffix [2,3] (matches at
        // p=1) → continuation [4,2,3].
        let toks = vec![1u32, 2, 3, 4, 2, 3];
        assert_eq!(drafter(3, 2, 4).propose(&toks), vec![4, 2, 3]);
    }

    #[test]
    fn empty_when_no_match() {
        assert!(drafter(3, 1, 8).propose(&[1u32, 2, 3, 4, 5]).is_empty());
    }

    #[test]
    fn empty_on_tiny_history() {
        let d = drafter(3, 1, 8);
        assert!(d.propose(&[]).is_empty());
        assert!(d.propose(&[7]).is_empty());
    }

    #[test]
    fn accept_run_counts_leading_matches() {
        // draft [10,11,12]; verify argmax [10,11,99, 7] (4th is the bonus). First mismatch at
        // index 2 → 2 accepted.
        assert_eq!(accept_run(&[10, 11, 12], &[10, 11, 99, 7]), 2);
        // all match → all accepted.
        assert_eq!(accept_run(&[10, 11, 12], &[10, 11, 12, 7]), 3);
        // first mismatch → 0 accepted.
        assert_eq!(accept_run(&[10, 11, 12], &[99, 11, 12, 7]), 0);
        // empty draft → 0.
        assert_eq!(accept_run(&[], &[7]), 0);
    }
}
