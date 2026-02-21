# ADR 015 — Email Channel Architecture

> Date: 2026-02-21
> Status: Accepted
> Phase: 13a

## Context

Phase 12 (Interaction) established the `CommChannel` abstraction, grounded dialogue,
constraint checking, social KG, federation, and multi-agent communication. However,
the agent had no way to interact with email — no connector, no MIME parser, no
threading. Phase 13a provides the email foundation: connectors, parsing, threading,
composition, and a `CommChannel` implementation.

## Decision

### 1. Top-level `src/email/` module (not nested under `src/agent/`)

Email is a substantial subsystem (~1,600 lines across 6 files in 13a alone) that grows
through Phases 13b–13d (spam classification, triage, extraction). Nesting it under
`agent/` would bloat that already-large directory (46 modules). The email module imports
from `agent::channel*` types but the agent module doesn't need to know email internals —
it registers `EmailChannel` via `ChannelRegistry` using the `CommChannel` trait.

### 2. Sync I/O with std::thread polling (no async)

**Options considered:**
- (a) Async JMAP + IMAP via tokio — matches oxifed pattern but adds tokio dependency
  to the email feature, which should be lightweight
- (b) Sync I/O with std::thread — JMAP is just HTTP POST (ureq already in deps),
  IMAP v2 crate is sync, lettre supports sync SMTP

**Chosen: (b)** — The email channel polls at intervals (30s–300s). There's no need
for the concurrency overhead of an async runtime. A single background std::thread
calling `thread::sleep()` between polls is simpler and avoids pulling in tokio for
email-only builds. The `email` feature is independent of the `daemon` feature.

### 3. JMAP via raw ureq (no jmap-client crate)

**Options considered:**
- (a) `jmap-client` crate — full JMAP client with typed methods
- (b) Raw ureq + serde_json — manual JSON construction

**Chosen: (b)** — The `jmap-client` crate is async/tokio-based, conflicting with our
sync-only decision. JMAP is a clean REST protocol: session discovery via
`GET /.well-known/jmap`, then `POST` with JSON method calls. Two methods are needed
initially (`Email/query` + `Email/get`), making raw ureq + serde_json straightforward.

### 4. IMAP via `imap` v2 + `native-tls`

The `imap` crate v2 provides sync IMAP with TLS support via the `native-tls` backend.
UID-based tracking enables delta sync (only fetch messages newer than the last seen UID).
The `imap::connect()` function establishes TLS connections directly.

### 5. Feature gating under `email`

The email feature is fully independent:
```toml
[features]
email = ["mail-parser", "imap", "native-tls", "lettre"]
```

No overlap with `daemon` or `oxifed`. The `AgentError::Email` variant and `pub mod email`
are both `#[cfg(feature = "email")]` gated. Builds without the feature see zero impact.

### 6. EmailPredicates pattern (following InterlocutorPredicates)

14 well-known relation SymbolIds resolved at channel initialization via
`engine.resolve_or_create_relation()`. This follows the same pattern as
`InterlocutorPredicates` (6 relations) and `AgentPredicates` (12 relations).
Email-specific KG data uses the `email:` prefix namespace.

### 7. Provenance: EmailIngested + EmailThreaded

Two new `DerivationKind` variants (tags 48–49) track the provenance of email data
entering the knowledge graph:
- `EmailIngested { message_id, channel_id }` — when a raw email is parsed and stored
- `EmailThreaded { thread_root_id, message_count }` — when JWZ threading groups messages

## Consequences

- Email becomes a first-class communication channel alongside operator and oxifed
- The `MockConnector` enables full pipeline testing without network I/O
- Phase 13b (spam classification) can add VSA encoding to `ParsedEmail` fields
- Phase 13c (triage) can use `EmailPredicates` for sender reputation in the KG
- Phase 13d (extraction) can parse `body_text` for dates, events, action items
- The JWZ `ThreadTree` provides conversation context for Phase 13c importance scoring
