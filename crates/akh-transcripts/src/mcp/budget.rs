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
}
