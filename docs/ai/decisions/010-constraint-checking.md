# ADR 010 — Pre-Communication Constraint Checking

> Date: 2026-02-21
> Status: Accepted
> Phase: 12c

## Context

After Phase 12b added grounded dialogue with provenance, outbound messages
still had no validation gate. Any response — even one containing contradictions,
low-confidence claims, or references to sensitive entities — would be emitted
without checks. Winter's key insight (checking Datalog rules before every post)
needs to be realized with akh-medu's full inference stack.

## Decision

Introduce a six-check constraint pipeline that runs before outbound message
emission, with configurable behavior per channel kind:

1. **ConstraintChecker** — stateful checker with per-channel `CommunicationBudget`
   tracking (sliding time window with cooldown), configurable confidence thresholds
   per channel kind, and sensitivity/provenance enforcement toggles.

2. **Six-check pipeline** (`check_grounded()`):
   - **Consistency**: Reconstructs `Triple` from `GroundedTriple` IDs and runs
     `check_contradictions()` against the KG (functional + disjointness checks).
   - **Confidence**: Compares each supporting triple's confidence against the
     channel kind's threshold (Operator: 0.0, Trusted: 0.3, Social: 0.5,
     Public: 0.7). Below-threshold → violation; below warning level → warning.
   - **Rate**: Checks `CommunicationBudget` with sliding window + cooldown.
   - **Relevance**: Placeholder for VSA similarity check (requires query vector
     context; deferred to Phase 12f).
   - **Sensitivity**: Looks for `onto:sensitivity-level` predicate on provenance
     entities. `SensitivityLevel` (Public/Low/Medium/High/Private) maps to
     minimum `ChannelKind` required.
   - **Provenance**: Checks that supporting triples have provenance records.
     Ungrounded claims generate warnings.

3. **CheckOutcome** — result struct with `passed` flag, violations list,
   warnings list.

4. **Emission decisions** per channel kind:
   - Operator: always emit (annotate with warnings)
   - Trusted: suppress on hard violations
   - Social/Public: suppress entirely on any violation

5. **ConstraintCheckStatus** evolution — the placeholder `Unchecked` variant
   is joined by `Passed { warning_count }` and `Failed { violation_count,
   warning_count }`. `from_outcome()` bridges `CheckOutcome` → status.
   `is_passed()` method for quick checks.

6. **GroundedTriple enrichment** — added `subject_id`, `predicate_id`,
   `object_id` (Option<SymbolId>) fields to enable consistency checking.
   Renamed label fields to `subject_label`/`predicate_label`/`object_label`
   for clarity. Confidence changed to `Option<f32>`, derivation to
   `Option<String>` (`derivation_tag`).

## Alternatives Considered

- **Constraint check inside `CommChannel::send()`**: Rejected because the
  trait method doesn't have access to `Engine`. The check must happen at the
  agent level before dispatch.

- **Blocking all emissions on any warning**: Too aggressive for the operator
  channel. The steward should see everything with annotations.

- **Per-triple suppression (partial emission)**: Considered for trusted
  channels but adds complexity. Deferred — full suppression is sufficient
  for Phase 12c.

## Consequences

- All grounded responses carry constraint check status with violation/warning
  counts.
- Rate limiting is tracked per channel via `CommunicationBudget` with sliding
  windows and cooldown.
- Sensitivity checks are ready but require `onto:sensitivity-level` triples
  to be asserted in the KG to activate.
- The relevance check is a documented placeholder — VSA query vector context
  will be added in Phase 12f.
- The `GroundedTriple` struct change is backward-compatible (new optional fields).
