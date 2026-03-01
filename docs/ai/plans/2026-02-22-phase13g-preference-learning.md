# Phase 13g — Preference Learning & Proactive Assistance

- **Status**: Complete
- **Date**: 2026-02-22
- **Phase**: 13g

## Summary

Added VSA-native preference profiles with temporal decay, Just-in-Time Information Retrieval (JITIR), and a serendipity engine for non-obvious connections. Single file `src/agent/preference.rs` (~630 lines), always-on (no feature gate), no new crate dependencies.

## Implementation

### New File
- `src/agent/preference.rs` — PreferenceError, ProactivityLevel, FeedbackSignal, PreferencePredicates, PreferenceRoleVectors, PreferenceProfile, Suggestion, JitirResult, PreferenceReview, PreferenceManager; 22 tests

### Modified Files
- `src/provenance.rs` — 3 new DerivationKind variants (tags 55–57: PreferenceLearned, JitirSuggestion, ProactiveAssistance)
- `src/agent/mod.rs` — `pub mod preference;` + re-exports
- `src/agent/error.rs` — `Preference` transparent variant
- `src/agent/agent.rs` — preference_manager field, init/resume/persist lifecycle, accessors
- `src/agent/nlp.rs` — `PrefCommand` intent variant + classification
- `src/agent/ooda.rs` — JITIR query in observe() (guarded by interaction_count > 0)
- `src/agent/trigger.rs` — ContextMatch + UrgencyThreshold conditions, SurfaceSuggestions + RefreshPreferences actions
- `src/agent/reflect.rs` — preference parameter, preference_review on ReflectionResult
- `src/agent/explain.rs` — 3 derivation_kind_prose arms
- `src/agent/goal_generation.rs` — preference_review field in test ReflectionResult literals
- `src/main.rs` — Commands::Pref with PrefAction, headless handler, format_derivation_kind arms
- `src/tui/mod.rs` — PrefCommand intent handler

## Key Decisions

1. `interest_prototype: Option<HyperVec>` instead of `HyperVec` because HyperVec has no Default impl
2. Negative feedback signals are recorded in history/provenance but don't modify the prototype (no VSA complement operation available)
3. `profile` field is `pub` (not `pub(crate)`) because main.rs (bin crate) needs access for CLI status display
4. Serendipity zone is Hamming similarity [0.3, 0.6] — related but not obvious connections
5. KG multi-hop BFS limited to depth 3 for serendipity discovery
