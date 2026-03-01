# ADR-027: Shared Workspace Agent

**Date:** 2026-03-01
**Status:** Accepted

## Context

`akhomed` previously created a throwaway `Agent` per HTTP request via `create_agent()`. Each call ran `Agent::resume()` (expensive: deserialization + KG reconstruction) then discarded the result. The daemon task owned a separate long-lived Agent.

Problems:
- **Wasteful**: full Agent reconstruction on every HTTP request.
- **Incoherent**: handler agents see stale state; mutations from one handler invisible to the next.
- **Racy**: multiple handlers calling `persist_session()` clobber each other and the daemon.
- **Conceptually wrong**: an Agent is the workspace's autonomous self -- there should be exactly one.

## Decision

One `Agent` per workspace, stored in `ServerState` as `Arc<std::sync::Mutex<Agent>>`. Both the daemon loop, HTTP handlers, and WebSocket sessions lock the same Agent.

### Why `std::sync::Mutex`

All Agent operations are synchronous. Handlers use `spawn_blocking`. `std::sync::Mutex` is simpler than `tokio::sync::Mutex` and avoids the "held across `.await`" footgun. The lock is never held across an await point.

### Contention

Daemon methods hold the lock for 1-100ms per tick, ticks spaced 15s-2h apart. HTTP handlers hold it for the duration of their synchronous operation (typically <50ms). Near-zero contention in practice.

### Agent lifecycle

- `ServerState::get_agent()` lazily creates one Agent per workspace and caches it.
- Uses `Agent::resume` if a persisted session exists, `Agent::new` otherwise.
- Wrapped in `spawn_blocking` since Agent construction is heavy sync work.
- `delete_workspace` evicts the cached agent.

### ChatProcessor remains session-local

`ChatProcessor` owns NLU pipeline state (parse ranker weights, tokenizer caches). Each WebSocket session gets its own `ChatProcessor`. Only the `Agent` is shared.

## Changes

| File | Change |
|------|--------|
| `src/bin/akhomed.rs` | Added `agents` to `ServerState`, added `get_agent()`, deleted `create_agent()`, updated 26+ handlers, updated `run_daemon_task`, updated `ws_handler`/`handle_ws_session` |
| `src/agent/daemon.rs` | Changed `agent` field to `Arc<Mutex<Agent>>`, updated `new()` signature, updated all daemon methods to lock-per-operation |

## Consequences

- One agent creation per workspace (at first access), not per request.
- Mutations from any source (HTTP, WS, daemon) are immediately visible to all others.
- No more persist races -- single Agent means single persist path.
- Daemon and handlers share the same goals, working memory, and session state.
- Slight complexity in daemon methods: each must lock/unlock the mutex, and methods that need partial field access use explicit reborrow (`let agent: &mut Agent = &mut *guard`).
