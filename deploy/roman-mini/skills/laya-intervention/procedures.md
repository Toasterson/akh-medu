# Laya intervention (Phase 53g PR 6)

Source of truth for the `laya-intervention` skillpack. It replaces
`laya_controller.py`'s scoped-fixer brief (`SCOPE`, the dispatch
`instruction`) and its `record_fixer_result` MCP tool. The wake prompt the
reactor seeds carry is rendered from `prompts.json` by
`deploy/roman-mini/full_akh.py`; keep the two in step.

## When this runs

akheventd's `roman-anima` source emits `anima.work_item.changed` for every
projection change of a work item and `anima.plan.changed` for a plan. Routes
in `events.json` decide which Roman wakes, with data-driven `when` filters on
machine fields only (the mapping table is the comment above
`anima_route_filters` in `full_akh.py`):

| Work item (parked) | Wakes |
|---|---|
| `parked_reason` `operator-hold` / `human-parked` | nobody (hold preserved) |
| `parked_reason` starting `gate-needs-human` | nobody (human decision) |
| `ci-infra-failing`, `ci-toolchain-broken`, `lane-circuit-breaker*` | Vesta |
| any live session (`live_sessions > 0`) | nobody (live owner) |
| `too-big` | Minerva |
| anything else parked | Janus |
| a plan changed | Minerva |

A parked item wakes its role at most `max_emissions_per_episode` (2) times per
park episode; further changes in the same episode are suppressed and appear
in the watchdog notice on the Roman status thread.

## Procedure

1. **Read current evidence** (`read_current_evidence`). The event names the
   entity (`work_item:<id>` or `plan:<id>`); read its current detail,
   forensics, thread and referenced artifacts through the Anima tools. The
   event payload is a snapshot, not the current state.
2. **Check your prior outcome** (`check_prior_outcome`). Recall
   `intervention_outcome` and `intervention_next_event` for the entity. When
   you recorded `human_required` or `waiting` for this park episode and the
   awaited event has not happened (no new human reply, the dependency or lane
   unchanged), record a quiet check and stop. This replaces the controller's
   wait guard.
3. **Act within scope** (`apply_scoped_fix`), per role:
   - Minerva: requirements, acceptance criteria or decomposition within
     existing product authority; preserve accepted scope and explicit human
     decisions.
   - Janus: the execution or workflow blocker, with supported Anima tools;
     preserve exact-head review, CI, dependencies and live ownership.
   - Vesta: the evidenced infrastructure or provider fault within existing
     operator authority; preserve workload ownership and verify service and
     tool execution separately.

   Never release explicit holds, reset budgets, bypass gates, or infer
   success from occupancy. Do not repeat a prior unsuccessful action without
   new evidence.
4. **Record evidence in the WI thread**, then **record the typed outcome**
   with `memory_write`, subject = the event's `entity`:

   | Predicate | Values |
   |---|---|
   | `intervention_outcome` | `fix_applied`, `human_required`, `waiting`, `blocked` |
   | `intervention_next_event` | `human_reply`, `dependency_changed`, `lane_recovered`, `ci_completed`, `review_completed`, `work_item_changed`, `none` |

   Both predicates are single-valued and temporal in `roman-memory` 1.1.0:
   the newest value supersedes, older ones stay as history. Rules carried
   over from `record_fixer_result`: `human_required` needs `human_reply`;
   `waiting` needs a named event, never `none`; `fix_applied` is verified by
   a later event, not by this turn.

## Actions and effects

`actions.json` lists the steps as causal action schemas (preconditions and
effects as fluents): `read_current_evidence` initiates `evidence_current`;
`check_prior_outcome` initiates `prior_outcome_known`; `apply_scoped_fix`
initiates `fix_attempted`; each `record_*` action initiates
`outcome_recorded` and terminates `intervention_admitted`.
