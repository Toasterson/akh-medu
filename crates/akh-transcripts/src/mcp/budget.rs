//! Byte-budget filling and truncation, shared by all five tools —
//! read-mcp-surface.md §1: "Every tool needs a hard cap and a continuation
//! token... Truncation is always visible."

/// Default and maximum `budget_bytes` for every tool.
pub const DEFAULT_BUDGET_BYTES: u64 = 32_000;
pub const MAX_BUDGET_BYTES: u64 = 256_000;

/// Clamp a caller-supplied budget into `[1, MAX_BUDGET_BYTES]`, defaulting to
/// `DEFAULT_BUDGET_BYTES` when absent or zero.
pub fn clamp_budget(requested: Option<u64>) -> u64 {
    match requested {
        None | Some(0) => DEFAULT_BUDGET_BYTES,
        Some(v) => v.min(MAX_BUDGET_BYTES),
    }
}

/// Fills a byte budget item by item, tracking what it had to leave out.
///
/// Every tool's "always-visible truncation" runs through this: callers push
/// candidate items in order and get back the ones that fit plus how many
/// bytes were omitted, so a truncated response can say so honestly instead
/// of silently stopping.
#[derive(Debug, Default)]
pub struct BudgetFiller {
    limit: u64,
    used: u64,
    omitted_bytes: u64,
    stopped_early: bool,
}

impl BudgetFiller {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            used: 0,
            omitted_bytes: 0,
            stopped_early: false,
        }
    }

    /// Try to admit an item of `bytes` size. Once the filler has stopped
    /// (budget exhausted), every subsequent call — admitted or not — is
    /// counted into `omitted_bytes`, so the final tally reflects everything
    /// that did not make it into the response, not just the first excluded
    /// item.
    pub fn try_take(&mut self, bytes: u64) -> bool {
        if !self.stopped_early && self.used + bytes <= self.limit {
            self.used += bytes;
            true
        } else {
            self.stopped_early = true;
            self.omitted_bytes += bytes;
            false
        }
    }

    /// Whether at least one item was rejected for budget reasons. Distinct
    /// from "there were more rows than fit in `limit`" — that is a
    /// separate, tool-level truncation reason this type does not track.
    pub fn truncated(&self) -> bool {
        self.stopped_early
    }

    pub fn omitted_bytes(&self) -> u64 {
        self.omitted_bytes
    }

    /// Bytes admitted so far. Exercised by this module's own tests; kept
    /// `pub` for a future caller that wants to report actual response size
    /// alongside `omitted_bytes` rather than reconstructing it.
    #[allow(dead_code)]
    pub fn used(&self) -> u64 {
        self.used
    }

    /// Admit a text item, guaranteeing forward progress: if nothing has been
    /// admitted yet and this item alone exceeds the remaining budget, it is
    /// admitted **truncated head-and-tail** with an explicit elision marker
    /// rather than refused — otherwise a caller whose first record is larger
    /// than its budget would receive zero items and a cursor pointing at the
    /// same record forever (observed live: a 34 KB Codex `session_meta`
    /// against a 4 KB budget). Once at least one item is in, an oversized item
    /// is refused and the cursor stops there, so the *next* page starts with
    /// it as the first item and truncates it.
    pub fn admit_text(&mut self, text: String) -> Admitted {
        let bytes = text.len() as u64;
        if self.try_take(bytes) {
            return Admitted::Whole(text);
        }
        if self.used == 0 && !text.is_empty() {
            // `try_take` already flagged stopped_early/omitted; undo the
            // omission accounting for the part we do emit.
            let keep = (self.limit.saturating_sub(64)).max(32) as usize;
            let (text, omitted) = head_and_tail(&text, keep);
            self.omitted_bytes = self.omitted_bytes.saturating_sub(bytes);
            self.omitted_bytes += omitted as u64;
            self.used = text.len() as u64;
            return Admitted::Truncated { text, omitted };
        }
        Admitted::Refused
    }
}

/// Outcome of [`BudgetFiller::admit_text`].
#[derive(Debug, PartialEq, Eq)]
pub enum Admitted {
    Whole(String),
    /// Emitted with `…[N bytes elided]…` in the middle; `omitted` is N.
    Truncated {
        text: String,
        omitted: usize,
    },
    Refused,
}

/// Keep roughly `keep` bytes of `text` as a head and a tail around a visible
/// elision marker. Splits on char boundaries only.
pub fn head_and_tail(text: &str, keep: usize) -> (String, usize) {
    if text.len() <= keep {
        return (text.to_owned(), 0);
    }
    let half = keep / 2;
    let mut head_end = half.min(text.len());
    while !text.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = text.len().saturating_sub(half);
    while !text.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    let omitted = tail_start.saturating_sub(head_end);
    (
        format!(
            "{}…[{} bytes elided]…{}",
            &text[..head_end],
            omitted,
            &text[tail_start..]
        ),
        omitted,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_budget_defaults_and_caps() {
        assert_eq!(clamp_budget(None), DEFAULT_BUDGET_BYTES);
        assert_eq!(clamp_budget(Some(0)), DEFAULT_BUDGET_BYTES);
        assert_eq!(clamp_budget(Some(100)), 100);
        assert_eq!(clamp_budget(Some(MAX_BUDGET_BYTES)), MAX_BUDGET_BYTES);
        assert_eq!(clamp_budget(Some(MAX_BUDGET_BYTES + 1)), MAX_BUDGET_BYTES);
        assert_eq!(clamp_budget(Some(u64::MAX)), MAX_BUDGET_BYTES);
    }

    #[test]
    fn admits_items_until_the_budget_is_exhausted() {
        let mut filler = BudgetFiller::new(10);
        assert!(filler.try_take(4));
        assert!(filler.try_take(4));
        assert!(!filler.try_take(4)); // 8 + 4 > 10
        assert!(filler.truncated());
        assert_eq!(filler.used(), 8);
        assert_eq!(filler.omitted_bytes(), 4);
    }

    #[test]
    fn once_stopped_every_further_item_counts_as_omitted() {
        let mut filler = BudgetFiller::new(5);
        assert!(filler.try_take(5));
        assert!(!filler.try_take(1));
        assert!(!filler.try_take(100));
        assert_eq!(filler.omitted_bytes(), 101);
    }

    #[test]
    fn an_item_exactly_at_the_limit_is_admitted() {
        let mut filler = BudgetFiller::new(10);
        assert!(filler.try_take(10));
        assert!(!filler.truncated());
    }

    #[test]
    fn empty_budget_admits_nothing() {
        let mut filler = BudgetFiller::new(0);
        assert!(!filler.try_take(1));
        assert!(filler.truncated());
    }

    #[test]
    fn never_taking_anything_is_not_truncated() {
        let filler = BudgetFiller::new(100);
        assert!(!filler.truncated());
        assert_eq!(filler.omitted_bytes(), 0);
    }

    #[test]
    fn oversized_first_item_is_truncated_not_refused() {
        let mut f = BudgetFiller::new(100);
        let big = "x".repeat(1_000);
        match f.admit_text(big) {
            Admitted::Truncated { text, omitted } => {
                assert!(text.contains("bytes elided"));
                assert!(text.len() <= 100 + 32, "got {}", text.len());
                assert!(omitted > 800);
            }
            other => panic!("expected Truncated, got {other:?}"),
        }
        assert!(f.truncated());
        // a second oversized item is refused: the cursor must stop here
        assert_eq!(f.admit_text("y".repeat(1_000)), Admitted::Refused);
    }

    #[test]
    fn small_items_are_admitted_whole_until_budget() {
        let mut f = BudgetFiller::new(10);
        assert_eq!(f.admit_text("abcd".into()), Admitted::Whole("abcd".into()));
        assert_eq!(f.admit_text("efgh".into()), Admitted::Whole("efgh".into()));
        assert_eq!(f.admit_text("ijkl".into()), Admitted::Refused);
        assert_eq!(f.omitted_bytes(), 4);
    }

    #[test]
    fn head_and_tail_respects_char_boundaries() {
        let (t, o) = head_and_tail("äöüäöüäöüäöü", 6);
        assert!(t.contains("elided"));
        assert!(o > 0);
    }
}
