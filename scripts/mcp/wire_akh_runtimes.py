#!/usr/bin/env python3
"""Add Sokar + Agent Board MCP servers to akh-platform runtime profiles.

Usage: wire_runtimes.py <resources.json|runtime-mac.json> <credentials_dir> [--dry-run]
- codex runtimes: append [mcp_servers.sokar] + [mcp_servers.djehuty-board] to the managed
  codex/config.toml; environment_files SOKAR_MCP_TOKEN / DJEHUTY_BOARD_TOKEN -> credential files.
- claude runtimes: managed claude/.claude.json with mcpServers using ${VAR} headers
  (Claude Code expands env vars in MCP headers at connect time); same environment_files.
- internal runtimes: untouched (tracked by WI 01a0a439 52f-lite).
Idempotent: skips runtimes that already carry the sokar server.
"""
import json, sys, copy
SOKAR_URL = "http://sokar.lab.home/mcp"
BOARD_URL = "http://board.lab.home/mcp"
CODEX_BLOCK = ('\n[mcp_servers.sokar]\nurl = "%s"\nbearer_token_env_var = "SOKAR_MCP_TOKEN"\n'
               '\n[mcp_servers.djehuty-board]\nurl = "%s"\nbearer_token_env_var = "DJEHUTY_BOARD_TOKEN"\n') % (SOKAR_URL, BOARD_URL)
CLAUDE_JSON = json.dumps({"mcpServers": {
    "sokar": {"type": "http", "url": SOKAR_URL, "headers": {"Authorization": "Bearer ${SOKAR_MCP_TOKEN}"}},
    "djehuty-board": {"type": "http", "url": BOARD_URL, "headers": {"Authorization": "Bearer ${DJEHUTY_BOARD_TOKEN}"}}}}, indent=2) + "\n"

def wire(profile, cred_dir, name, per_runtime_board=False):
    iso = profile.setdefault("isolation", {})
    mf = iso.setdefault("managed_files", {})
    ef = iso.setdefault("environment_files", {}) or {}
    kind = profile.get("kind")
    changed = False
    if kind == "codex":
        cur = mf.get("codex/config.toml", "")
        if "[mcp_servers.sokar]" not in cur:
            mf["codex/config.toml"] = cur.rstrip("\n") + "\n" + CODEX_BLOCK; changed = True
    elif kind == "claude":
        cur = mf.get("claude/.claude.json", "")
        if '"sokar"' not in cur:
            mf["claude/.claude.json"] = CLAUDE_JSON; changed = True
    else:
        return False
    board_fn = f"board-token-{name}" if per_runtime_board else "board-token"
    for var, fn in (("SOKAR_MCP_TOKEN", "sokar-token"), ("DJEHUTY_BOARD_TOKEN", board_fn)):
        if var not in ef:
            ef[var] = f"{cred_dir}/{fn}"; changed = True
    iso["environment_files"] = ef
    return changed

def main():
    path, cred_dir = sys.argv[1], sys.argv[2]
    dry = "--dry-run" in sys.argv
    doc = json.load(open(path))
    touched = []
    if isinstance(doc, dict) and doc.get("kind") == "List":          # talos resources.json
        for it in doc["items"]:
            if it.get("kind") != "ConfigMap" or not it["metadata"]["name"].startswith("runtime-"): continue
            for fn, content in list((it.get("data") or {}).items()):
                try: c = json.loads(content)
                except Exception: continue
                ch = False
                for rn, rt in (c.get("runtimes") or {}).items():
                    if wire(rt, cred_dir, rn): ch = True; touched.append(f"{it['metadata']['name']}/{rn}")
                if ch: it["data"][fn] = json.dumps(c, indent=2) + "\n"
    else:                                                             # runtime-mac.json
        for rn, rt in (doc.get("runtimes") or {}).items():
            if wire(rt, cred_dir, rn, per_runtime_board=True): touched.append(rn)
    print("touched:", touched)
    if not dry and touched:
        json.dump(doc, open(path, "w"), indent=2); open(path, "a").write("\n")
main()
