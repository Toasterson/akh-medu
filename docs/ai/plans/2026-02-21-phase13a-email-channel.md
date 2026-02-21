# Phase 13a — Email Channel (JMAP/IMAP + MIME)

> Date: 2026-02-21
> Status: Complete
> Depends on: Phase 12a (channel abstraction), Phase 12d (interlocutor registry)

## Objective

Implement a feature-gated email subsystem (`--features email`) that provides the
foundation for personal assistant email capabilities. The email channel implements
the `CommChannel` trait from Phase 12a, enabling email as a bidirectional
communication channel alongside the existing operator and oxifed channels.

## Deliverables

### New files

1. **`src/email/error.rs`** (~130 lines)
   - `EmailError` miette diagnostic enum with 7 variants (Connection, Authentication, Parse, Send, Threading, Config, Engine)
   - `EmailResult<T>` type alias
   - Manual `From<AkhError>` impl for Engine variant (boxing to break type cycle)
   - 8 unit tests

2. **`src/email/connector.rs`** (~520 lines)
   - `EmailConnector` trait: `fetch_new()`, `fetch_by_id()`, `send_email()`, `sync_state()`
   - `RawEmail` struct (uid, mailbox, flags, data bytes)
   - `EmailConfig` with validation, `EmailCredentials` enum, `ConnectionType` enum
   - `JmapConnector` — JMAP over ureq with session discovery, Email/query + Email/get
   - `ImapConnector` — TLS via native_tls, imap::connect(), UID-based delta sync
   - `MockConnector` — in-memory queue for testing
   - 12 unit tests

3. **`src/email/parser.rs`** (~230 lines)
   - `ParsedEmail` struct with 15 fields
   - `parse_raw()` using mail_parser::MessageParser
   - `extract_domain()` utility
   - Helper functions for MIME header extraction
   - 12 unit tests

4. **`src/email/threading.rs`** (~270 lines)
   - `ThreadNode` struct (message_id, parent, children, parsed)
   - `ThreadTree` struct (nodes, roots, index HashMap)
   - `build_threads()` — 5-step JWZ algorithm (RFC 5256) with cycle protection
   - 8 unit tests

5. **`src/email/compose.rs`** (~170 lines)
   - `ComposedEmail` struct
   - `compose_reply()` — Re: prefix, In-Reply-To, References chain, quoted body
   - `compose_new()` — fresh email composition
   - `to_mime()` — RFC 5322 rendering via lettre Message builder
   - 11 unit tests

6. **`src/email/mod.rs`** (~290 lines)
   - Module root with re-exports
   - `EmailPredicates` — 14 well-known relation SymbolIds
   - `EmailInboundHandle` — Arc<Mutex<VecDeque<InboundMessage>>> with push_email()
   - `EmailChannel` implementing CommChannel with background std::thread polling
   - 11 unit tests

### Modified files

7. **`Cargo.toml`** — added `mail-parser`, `imap`, `native-tls`, `lettre` as optional deps; `email` feature gate
8. **`src/lib.rs`** — `#[cfg(feature = "email")] pub mod email;`
9. **`src/provenance.rs`** — `EmailIngested` (tag 48) and `EmailThreaded` (tag 49) DerivationKind variants
10. **`src/agent/error.rs`** — `AgentError::Email` variant (cfg-gated)
11. **`src/agent/explain.rs`** — derivation_kind_prose entries for new variants
12. **`src/main.rs`** — format_derivation_kind match arms for new variants

## Key Design Decisions

1. **Top-level `src/email/` module** — email is a large subsystem (6 files, ~1,600 lines)
   that grows through Phases 13b–13d.

2. **Sync I/O only** — JMAP via ureq (already in deps), IMAP via sync imap v2,
   SMTP via lettre. No async runtime required. Background std::thread for polling.

3. **Feature-gated** — `email` feature is independent of `daemon`/`oxifed`. Polling
   uses std::thread::spawn, not tokio.

4. **JMAP via raw ureq** — avoids async jmap-client crate. JMAP is HTTP POST + JSON.

5. **MockConnector for testing** — enables full pipeline testing without network I/O.

## Verification

- `cargo build` — passes (email feature not active)
- `cargo build --features email` — passes
- `cargo test --lib` — 1122 tests pass
- `cargo test --lib --features email` — 1184 tests pass (62 new)
- `cargo test --lib --features oxifed` — 1138 tests pass (no regressions)
