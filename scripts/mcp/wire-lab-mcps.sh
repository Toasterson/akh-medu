#!/bin/sh
# Wire the lab MCP servers (Sokar artifact daemon + Djehuty Agent Board) into
# Codex and Claude Code on a Mac.  akh-medu Phase 52 / ADR 108.
#
# Usage:
#   SOKAR_MCP_TOKEN=... DJEHUTY_BOARD_TOKEN_CLAUDE=... DJEHUTY_BOARD_TOKEN_CODEX=... sh scripts/mcp/wire-lab-mcps.sh
#   sh scripts/mcp/wire-lab-mcps.sh --from-1password     # reads all three via op-service-account (or op)
# Board identities are per client on purpose: 1Password "Agent Board Claude Code" and
# "Agent Board Codex" are different agents on the board.
#
# What it does (idempotent):
#   1. Stores the tokens in ~/.config/akh-medu/{sokar-token,board-token-claude,board-token-codex} (0600).
#   2. Exports SOKAR_MCP_TOKEN / DJEHUTY_BOARD_TOKEN_CLAUDE / DJEHUTY_BOARD_TOKEN_CODEX from ~/.zshenv.local.
#   3. Codex: adds [mcp_servers.sokar] and [mcp_servers.djehuty-board] to ~/.codex/config.toml
#      (url + bearer_token_env_var; no token in the file).
#   4. Claude Code: registers both servers at user scope with an Authorization header that
#      references the env var (Claude Code expands ${VAR} in MCP headers at connect time).
# Never prints a token.
set -eu
SOKAR_URL="${SOKAR_MCP_URL:-http://sokar.lab.home/mcp}"
BOARD_URL="${DJEHUTY_BOARD_MCP_URL:-http://board.lab.home/mcp}"
CFG="$HOME/.config/akh-medu"; mkdir -p "$CFG"; chmod 700 "$CFG"
if [ "${1:-}" = "--from-1password" ]; then
  OP="${OP_BIN:-$HOME/.local/bin/op-service-account}"
  command -v "$OP" >/dev/null 2>&1 || OP=op
  SOKAR_MCP_TOKEN=$("$OP" read "op://Homelab/sia-runtime/SIA_SESHAT__TOKEN" | head -1)
  DJEHUTY_BOARD_TOKEN_CLAUDE=$("$OP" read "op://Homelab/Agent Board Claude Code/credential" | head -1)
  DJEHUTY_BOARD_TOKEN_CODEX=$("$OP" read "op://Homelab/Agent Board Codex/credential" | head -1)
fi
: "${SOKAR_MCP_TOKEN:?set SOKAR_MCP_TOKEN (fleet shared secret) or use --from-1password}"
: "${DJEHUTY_BOARD_TOKEN_CLAUDE:?set DJEHUTY_BOARD_TOKEN_CLAUDE (Agent Board Claude Code token) or use --from-1password}"
: "${DJEHUTY_BOARD_TOKEN_CODEX:?set DJEHUTY_BOARD_TOKEN_CODEX (Agent Board Codex token) or use --from-1password}"
umask 077
# 0. Host entries. The Macs resolve *.lab.home through /etc/hosts (their DNS
#    goes to the ISP resolver, not the lab CoreDNS at 192.168.1.15), so a
#    missing line here looks exactly like a dead MCP server.
for host in sokar.lab.home board.lab.home; do
  grep -q "[[:space:]]$host\b" /etc/hosts 2>/dev/null || {
    printf '%s %s\n' "${LAB_GATEWAY_IP:-192.168.1.12}" "$host" | sudo tee -a /etc/hosts >/dev/null
    echo "added $host to /etc/hosts"
  }
done
printf '%s\n' "$SOKAR_MCP_TOKEN" > "$CFG/sokar-token"
printf '%s\n' "$DJEHUTY_BOARD_TOKEN_CLAUDE" > "$CFG/board-token-claude"
printf '%s\n' "$DJEHUTY_BOARD_TOKEN_CODEX" > "$CFG/board-token-codex"
# 2. shell env (zsh); fish/bash users: mirror these two lines in your shell init.
ZENV="$HOME/.zshenv.local"; touch "$ZENV"
grep -q 'SOKAR_MCP_TOKEN=' "$ZENV" || printf '\nexport SOKAR_MCP_TOKEN="$(cat "$HOME/.config/akh-medu/sokar-token" 2>/dev/null)"\n' >> "$ZENV"
grep -q 'DJEHUTY_BOARD_TOKEN_CLAUDE=' "$ZENV" || printf 'export DJEHUTY_BOARD_TOKEN_CLAUDE="$(cat "$HOME/.config/akh-medu/board-token-claude" 2>/dev/null)"\n' >> "$ZENV"
grep -q 'DJEHUTY_BOARD_TOKEN_CODEX=' "$ZENV" || printf 'export DJEHUTY_BOARD_TOKEN_CODEX="$(cat "$HOME/.config/akh-medu/board-token-codex" 2>/dev/null)"\n' >> "$ZENV"
grep -q 'zshenv.local' "$HOME/.zshenv" 2>/dev/null || printf '[ -f "$HOME/.zshenv.local" ] && . "$HOME/.zshenv.local"\n' >> "$HOME/.zshenv"
export SOKAR_MCP_TOKEN DJEHUTY_BOARD_TOKEN_CLAUDE DJEHUTY_BOARD_TOKEN_CODEX
# 3. Codex
CODEX="$HOME/.codex/config.toml"; mkdir -p "$HOME/.codex"; touch "$CODEX"
grep -q '^\[mcp_servers\.sokar\]' "$CODEX" || printf '\n[mcp_servers.sokar]\nurl = "%s"\nbearer_token_env_var = "SOKAR_MCP_TOKEN"\n' "$SOKAR_URL" >> "$CODEX"
grep -q '^\[mcp_servers\.djehuty-board\]' "$CODEX" || printf '\n[mcp_servers.djehuty-board]\nurl = "%s"\nbearer_token_env_var = "DJEHUTY_BOARD_TOKEN_CODEX"\n' "$BOARD_URL" >> "$CODEX"
# 4. Claude Code (user scope). The literal ${VAR} is intentional: expanded by Claude Code at connect time.
CLAUDE="${CLAUDE_BIN:-$(command -v claude || echo "$HOME/.local/bin/claude")}"
if [ -x "$CLAUDE" ]; then
  "$CLAUDE" mcp remove --scope user sokar >/dev/null 2>&1 || true
  "$CLAUDE" mcp remove --scope user djehuty-board >/dev/null 2>&1 || true
  "$CLAUDE" mcp add --scope user --transport http sokar "$SOKAR_URL" --header 'Authorization: Bearer ${SOKAR_MCP_TOKEN}'
  "$CLAUDE" mcp add --scope user --transport http djehuty-board "$BOARD_URL" --header 'Authorization: Bearer ${DJEHUTY_BOARD_TOKEN_CLAUDE}'
else
  echo "claude CLI not found; skipped Claude Code registration" >&2
fi
echo "wired: Codex ($CODEX) + Claude Code user scope -> sokar=$SOKAR_URL djehuty-board=$BOARD_URL"
echo "open a new shell (or: . ~/.zshenv.local) so the token env vars are exported; then: claude mcp list / codex mcp list"
# 5. Verify (both should answer 200 with a session id; 401 means a bad token).
for pair in "sokar $SOKAR_URL $SOKAR_MCP_TOKEN" "djehuty-board $BOARD_URL $DJEHUTY_BOARD_TOKEN_CODEX"; do
  set -- $pair
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST "$2" -H "Authorization: Bearer $3" \
    -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
    -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"wire-lab-mcps","version":"1"}}}' 2>/dev/null)
  echo "verify $1: HTTP $code"
done
