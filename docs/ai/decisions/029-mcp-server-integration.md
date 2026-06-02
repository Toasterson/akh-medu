# ADR-029: MCP Server Embedded in akhomed

**Date**: 2026-03-02
**Status**: Accepted

## Context

We want Claude Code (and other MCP-capable AI assistants) to interact with
akh-medu as a knowledge agent — querying the graph, asserting facts, managing
compartments, and running the OODA agent. The Model Context Protocol (MCP)
over HTTP is the standard interface for this.

## Decision

Embed the MCP server directly inside `akhomed` using rmcp's
`StreamableHttpService`, mounted at `/mcp` on the existing axum router.
This gives:

- **Zero overhead**: MCP tool calls are in-process Rust function calls to the
  engine and agent — no HTTP proxy or subprocess.
- **Single process**: No separate MCP bridge to deploy or monitor.
- **Shared state**: MCP tools use the same `Arc<RwLock<HashMap<_, Arc<Engine>>>>` and
  `Arc<RwLock<HashMap<_, Arc<Mutex<Agent>>>>>` maps as the daemon and REST
  handlers (see ADR-027). Workspaces are lazy-loaded on first access.
- **Multi-workspace**: All workspace-scoped tools accept an optional `workspace`
  parameter (default: `"default"`). No pre-warming or env var needed.

### Feature gate

The `mcp` Cargo feature (depends on `server`) gates all MCP code:
- `rmcp` dependency (server + streamable HTTP transport)
- `src/mcp/` module with `AkhMcpServer`
- `/mcp` route in akhomed

Building without `--features mcp` excludes all MCP code.

### Tool surface

| Category | Tools |
|----------|-------|
| Query | `ask`, `sparql_query`, `search` |
| Mutation | `assert_triple`, `ingest_text`, `ingest_url` |
| Compartments | `list_compartments`, `discover_compartments`, `load_compartment`, `unload_compartment`, `activate_compartment`, `deactivate_compartment` |
| Agent | `run_agent` |
| Status | `status` |
| Workspace | `list_workspaces`, `create_workspace`, `delete_workspace` |
| Seeds | `apply_seed`, `list_seeds` |
| Bootstrap | `awaken`, `awaken_parse` |
| Chat | `chat` |

### Configuration

- All workspace-scoped tools accept an optional `workspace` parameter that
  defaults to `"default"`. No env var needed.
- `.mcp.json` in the project root auto-registers with Claude Code.

## Alternatives Considered

1. **Separate MCP bridge process** — requires IPC, adds deployment complexity,
   duplicates state.
2. **stdio MCP server** — requires Claude Code to manage a child process;
   can't share the daemon's engine instance.
3. **REST-only (no MCP)** — Claude Code's native MCP integration is much
   richer than raw HTTP tool-calling.

## Consequences

- akhomed gains ~800 lines in `src/mcp/mod.rs` (feature-gated).
- `rmcp` + `schemars` v1 added as optional dependencies (~50 crates).
- Claude Code can now interact with akh-medu's knowledge graph natively.
- Full workspace management, bootstrap, and chat surface — 22 tools total.
- Compartment REST endpoints added for completeness (discover/load/unload/activate/deactivate).
