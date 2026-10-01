#!/usr/bin/env python3
"""Replay MCP server for the CoS dojo (plan slice C4, ADR 116 Decision 3).

Serves snapshot t_k over MCP JSON-RPC with the live tool names and schemas:

- `tools/list` returns the recorded Anima tool contract verbatim, with its
  sha256 (the recorder's `tool_contract_sha256`) in `_meta`.
- `tools/call` of a recorded read tool answers with the recorded bytes of the
  same call and arguments, or of a recorded call whose arguments are provably
  equivalent under the tool's input schema (`replay_match`; the receipt keeps
  both the requested and the matched arguments). A read the recorder did not
  capture answers `isError` ("not recorded", with a machine-readable
  `reason`); nothing is invented.
- Every other tool in the contract (the writes) is appended to the **dojo
  tenancy ledger** (append-only, hash-chained JSONL) and acknowledged. The
  server has no network client: nothing reaches production.
- A tool outside the contract is a JSON-RPC error.

A second server, `DojoGitServer`, answers `git_ls_remote` from the recorded
refs and `git_is_ancestor` by `git merge-base --is-ancestor` against the
recorded `refs/heads/main` (the "on main" rule, procedures.md §2.3).

Every call issues a `ToolReceipt` into a shared `ReceiptLog`; the runner hands
those receipts back unmodified in the `Episode`.

Deterministic: receipt ids are sequential, `observed_at` is the snapshot time,
the ledger carries no wall clock.

    replay_server.py serve --root R --snapshot 20260928T073924Z --ledger dojo-ledger.jsonl [--repo PATH]

reads JSON-RPC requests (one per line) on stdin and writes responses on stdout.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo.contracts import PermissionProfile, Tier
from dojo.episode import ReceiptSource, ToolReceipt
from dojo.errors import DojoError, LedgerError
from dojo.replay_match import ArgumentMatch, explain_miss, resolve_recorded_read, resolve_ref
from dojo.snapshot import MAIN_REF, Snapshot, SnapshotStore, canonical_json, sha256_hex

PROTOCOL_VERSION = "2025-03-26"
DEFAULT_TENANCY = "lilith-dojo"

# Byte-for-byte the recorder's `ANIMA_READ_TOOLS` (tested): the reads a snapshot can answer.
ANIMA_READ_TOOLS: frozenset[str] = frozenset({
    "list_plans",
    "list_work_items",
    "get_work_item_detail",
    "get_wi_loop_state",
    "get_thread",
    "list_sessions",
    "list_changes",
    "get_project_ci_status",
    "get_scheduling_state",
    "list_projects",
    "get_plan",
    "get_work_item_by_id",
    "get_work_item_forensics",
    "seshat_status",
    "list_prompts",
    "list_workflows",
    "stranded_work_items",
    "lane_occupancy",
    "overdue_ci_dispatches",
})

# Free-form query tools. Their answer depends on the query text and on a live index (ripgrep over the
# repository, Seshat's embeddings); a snapshot holds no answer to an arbitrary query, so the replay
# answers every call with the same structured "not replayable" error, for every arm. They stay in
# `tools/list` (the recorded contract is served verbatim); see README "Free-form query tools".
NOT_REPLAYABLE_ANIMA_TOOLS: frozenset[str] = frozenset({
    "find",
    "seshat_search",
    "seshat_wish",
    "search_documents",
    "search_library",
})
NOT_REPLAYABLE_REASON = "not-replayable"
# A tool the dojo made unavailable for one episode (a failed-read drill, `EpisodeSpec.replay_faults`).
UNAVAILABLE_REASON = "tool-unavailable"

JSONRPC_METHOD_NOT_FOUND = -32601
JSONRPC_INVALID_PARAMS = -32602


# --- ancestry oracle ---------------------------------------------------------------


class AncestryOracle(Protocol):
    def is_ancestor(self, commit: str, tip: str) -> bool | None:
        """True/False, or None when the local repository cannot decide (missing object)."""


@dataclasses.dataclass
class GitAncestryOracle:
    """`git merge-base --is-ancestor` in a local clone that has the recorded commits."""

    repo: Path
    runner: Callable[[list[str]], int] | None = None

    def is_ancestor(self, commit: str, tip: str) -> bool | None:
        cmd = ["git", "-C", str(self.repo), "merge-base", "--is-ancestor", commit, tip]
        if self.runner is not None:
            code = self.runner(cmd)
        else:
            try:
                code = subprocess.run(cmd, capture_output=True, timeout=30).returncode
            except (OSError, subprocess.SubprocessError):
                return None
        if code == 0:
            return True
        if code == 1:
            return False
        return None


@dataclasses.dataclass
class StaticAncestryOracle:
    """Test/fixture oracle: {(commit, tip): verdict}."""

    verdicts: Mapping[tuple[str, str], bool]

    def is_ancestor(self, commit: str, tip: str) -> bool | None:
        return self.verdicts.get((commit, tip))


# --- dojo tenancy ledger -------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LedgerEntry:
    seq: int
    tenancy: str
    snapshot_id: str
    server: str
    tool: str
    arguments: Mapping[str, Any]
    prev_sha256: str
    entry_sha256: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self) | {"arguments": dict(self.arguments)}


class DojoLedger:
    """Append-only, hash-chained JSONL. Never production: a plain local file."""

    GENESIS = "0" * 64

    def __init__(self, path: Path, tenancy: str = DEFAULT_TENANCY, forbidden_roots: tuple[Path, ...] = ()):
        self.path = Path(path)
        self.tenancy = tenancy
        resolved = self.path.resolve()
        for root in forbidden_roots:
            if resolved.is_relative_to(Path(root).resolve()):
                raise LedgerError(f"ledger {self.path} lies inside the snapshot root {root}; recorded evidence is read-only")
        self.entries: list[LedgerEntry] = self._load()

    def _load(self) -> list[LedgerEntry]:
        if not self.path.exists():
            return []
        entries: list[LedgerEntry] = []
        prev = self.GENESIS
        for n, line in enumerate(self.path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            entry = LedgerEntry(**raw)
            body = {k: v for k, v in raw.items() if k != "entry_sha256"}
            if entry.prev_sha256 != prev or sha256_hex(canonical_json(body)) != entry.entry_sha256:
                raise LedgerError(f"ledger {self.path} line {n} breaks the hash chain")
            if entry.seq != len(entries) + 1:
                raise LedgerError(f"ledger {self.path} line {n} has seq {entry.seq}, expected {len(entries) + 1}")
            entries.append(entry)
            prev = entry.entry_sha256
        return entries

    def append(self, snapshot_id: str, server: str, tool: str, arguments: Mapping[str, Any]) -> LedgerEntry:
        prev = self.entries[-1].entry_sha256 if self.entries else self.GENESIS
        body = {
            "seq": len(self.entries) + 1,
            "tenancy": self.tenancy,
            "snapshot_id": snapshot_id,
            "server": server,
            "tool": tool,
            "arguments": dict(arguments),
            "prev_sha256": prev,
        }
        entry = LedgerEntry(**body, entry_sha256=sha256_hex(canonical_json(body)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry.to_dict(), sort_keys=True, ensure_ascii=False) + "\n")
        self.entries.append(entry)
        return entry


# --- receipts ------------------------------------------------------------------------


class ReceiptLog:
    """Issues receipts with sequential ids, shared by every server of one episode."""

    def __init__(self) -> None:
        self.receipts: list[ToolReceipt] = []

    def issue(self, *, server: str, tool: str, arguments: Mapping[str, Any], response: Any,
              response_sha256: str, source: ReceiptSource, snapshot: Snapshot,
              match: ArgumentMatch | None = None, error_reason: str | None = None) -> ToolReceipt:
        receipt = ToolReceipt(
            call_id=f"r{len(self.receipts) + 1}",
            server=server,
            tool=tool,
            arguments=dict(arguments),
            response=response,
            response_sha256=response_sha256,
            source=source,
            snapshot_id=snapshot.id,
            observed_at=snapshot.unix_time,
            matched_arguments=None if match is None or match.exact else dict(match.matched),
            match_rules=() if match is None else match.rules,
            error_reason=error_reason,
        )
        self.receipts.append(receipt)
        return receipt


# --- JSON-RPC plumbing ------------------------------------------------------------------


def _error(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tool_result(receipt: ToolReceipt, payload: Any, is_error: bool) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
        "_meta": {
            "lilith/receipt": receipt.call_id,
            "lilith/digest": f"sha256:{receipt.response_sha256}",
            "lilith/observed_at": receipt.observed_at,
            "lilith/source": receipt.source.value,
        },
    }


class _McpServer:
    name: str
    snapshot: Snapshot
    receipts: ReceiptLog

    def tools(self) -> list[dict]:
        raise NotImplementedError

    def contract_sha256(self) -> str:
        raise NotImplementedError

    def call(self, name: str, arguments: Mapping[str, Any]) -> tuple[ToolReceipt, Any, bool]:
        raise NotImplementedError

    def handle(self, request: Mapping[str, Any]) -> dict | None:
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params") or {}
        if request_id is None:  # notification
            return None
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": f"lilith-dojo-{self.name}", "version": "1"},
            }}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {
                "tools": self.tools(),
                "_meta": {"lilith/contract_sha256": self.contract_sha256(), "lilith/snapshot": self.snapshot.id},
            }}
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return _error(request_id, JSONRPC_INVALID_PARAMS, "tools/call needs a name and an arguments object")
            if name not in {t.get("name") for t in self.tools()}:
                return _error(request_id, JSONRPC_INVALID_PARAMS, f"unknown tool {name!r} (not in the recorded contract)")
            receipt, payload, is_error = self.call(name, arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": _tool_result(receipt, payload, is_error)}
        return _error(request_id, JSONRPC_METHOD_NOT_FOUND, f"method {method!r} not supported by the replay server")


class ReplayServer(_McpServer):
    """Anima replay: recorded reads from snapshot t_k, writes into the dojo ledger."""

    def __init__(self, snapshot: Snapshot, ledger: DojoLedger, receipts: ReceiptLog | None = None,
                 collector: str = "anima", profile: PermissionProfile | None = None,
                 unavailable: frozenset[str] = frozenset()):
        contract = snapshot.tool_contract(collector)
        if contract is None:
            raise DojoError(f"snapshot {snapshot.id} recorded no {collector} tool contract",
                            help="replay needs a snapshot whose collector captured tools/list; pick another t_k")
        self._tools, self._contract_sha = contract
        self.name = collector
        self.snapshot = snapshot
        self.ledger = ledger
        self.receipts = receipts or ReceiptLog()
        self.profile = profile or PermissionProfile.load()
        self.unavailable = frozenset(unavailable)

    def tools(self) -> list[dict]:
        return self._tools

    def contract_sha256(self) -> str:
        return self._contract_sha

    def tool(self, name: str) -> dict | None:
        return next((t for t in self._tools if t.get("name") == name), None)

    def is_read(self, name: str) -> bool:
        return name in ANIMA_READ_TOOLS or self.profile.server_tier(self.name, name) is Tier.T0_READ

    def call(self, name: str, arguments: Mapping[str, Any]) -> tuple[ToolReceipt, Any, bool]:
        if name in self.unavailable:
            # Stays listed (the recorded contract is served verbatim) and answers like a failing upstream:
            # the model has to notice and report it, as it must when a live read fails.
            payload = {"error": f"{name} failed: the tracker could not serve this read", "reason": UNAVAILABLE_REASON,
                       "tool": name, "arguments": dict(arguments)}
            receipt = self._issue(name, arguments, payload, sha256_hex(canonical_json(payload)), ReceiptSource.ERROR,
                                  error_reason=UNAVAILABLE_REASON)
            return receipt, payload, True
        if name in NOT_REPLAYABLE_ANIMA_TOOLS:
            payload = {"error": "not replayable in the dojo", "reason": NOT_REPLAYABLE_REASON, "tool": name,
                       "arguments": dict(arguments), "snapshot": self.snapshot.id,
                       "detail": "a free-form query over a live index: the recording holds no answer to it"}
            receipt = self._issue(name, arguments, payload, sha256_hex(canonical_json(payload)), ReceiptSource.ERROR,
                                  error_reason=NOT_REPLAYABLE_REASON)
            return receipt, payload, True
        if self.is_read(name):
            found = None
            schema = (self.tool(name) or {}).get("inputSchema")
            if name in ANIMA_READ_TOOLS:
                found = resolve_recorded_read(self.snapshot, name, schema, arguments)
            if found is None:
                why = explain_miss(self.snapshot, name, schema, arguments) if name in ANIMA_READ_TOOLS else None
                reason = why.reason if why is not None else "tool-not-recorded"
                payload = {"error": "not recorded", "reason": reason, "tool": name, "arguments": dict(arguments),
                           "snapshot": self.snapshot.id}
                if why is not None and why.detail:
                    payload["detail"] = why.detail
                if why is not None and why.context:
                    payload.update({k: v for k, v in why.context.items() if k not in payload})
                receipt = self._issue(name, arguments, payload, sha256_hex(canonical_json(payload)), ReceiptSource.ERROR,
                                      error_reason=reason)
                return receipt, payload, True
            receipt = self._issue(name, arguments, found.payload, found.ref.sha256, ReceiptSource.SNAPSHOT, match=found.match)
            return receipt, found.payload, False
        entry = self.ledger.append(self.snapshot.id, self.name, name, arguments)
        payload = {"accepted": True, "dojo_tenancy": self.ledger.tenancy, "ledger_seq": entry.seq,
                   "entry_sha256": entry.entry_sha256}
        return self._issue(name, arguments, payload, sha256_hex(canonical_json(payload)), ReceiptSource.LEDGER), payload, False

    def _issue(self, name: str, arguments: Mapping[str, Any], payload: Any, digest: str,
               source: ReceiptSource, match: ArgumentMatch | None = None, error_reason: str | None = None) -> ToolReceipt:
        return self.receipts.issue(server=self.name, tool=name, arguments=arguments, response=payload,
                                   response_sha256=digest, source=source, snapshot=self.snapshot,
                                   match=match, error_reason=error_reason)


GIT_TOOLS: list[dict] = [
    {
        "name": "git_ls_remote",
        "description": "Refs of the akh-medu remote as recorded in this snapshot.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "git_is_ancestor",
        "description": "`git merge-base --is-ancestor <commit> <ref>` against the recorded ref (default refs/heads/main).",
        "inputSchema": {
            "type": "object",
            "properties": {"commit": {"type": "string"}, "ref": {"type": "string"}},
            "required": ["commit"],
        },
    },
]


class DojoGitServer(_McpServer):
    def __init__(self, snapshot: Snapshot, oracle: AncestryOracle | None, receipts: ReceiptLog | None = None):
        self.name = "git"
        self.snapshot = snapshot
        self.oracle = oracle
        self.receipts = receipts or ReceiptLog()

    def tools(self) -> list[dict]:
        return GIT_TOOLS

    def contract_sha256(self) -> str:
        return sha256_hex(canonical_json({"tools": GIT_TOOLS}))

    def call(self, name: str, arguments: Mapping[str, Any]) -> tuple[ToolReceipt, Any, bool]:
        if name == "git_ls_remote":
            found = self.snapshot.answer("git_ls_remote", {"remote": next(
                (r.args.get("remote") for r in self.snapshot.by_call("git_ls_remote")), "")})
            if found is not None:
                payload, ref = found
                return self._issue(name, arguments, payload, ref.sha256, ReceiptSource.SNAPSHOT), payload, False
            payload = {"error": "not recorded", "reason": "arguments-not-recorded", "tool": name}
            return self._issue(name, arguments, payload, sha256_hex(canonical_json(payload)), ReceiptSource.ERROR,
                               error_reason="arguments-not-recorded"), payload, True
        commit = str(arguments.get("commit", ""))
        requested = arguments.get("ref")
        requested = str(requested) if requested else None
        ref, rules = resolve_ref(self.snapshot.git_refs, requested)
        tip = self.snapshot.git_refs.get(ref) if ref else None
        verdict = self.oracle.is_ancestor(commit, tip) if (self.oracle and tip and commit) else None
        payload = {"commit": commit, "ref": ref or requested or MAIN_REF, "ref_sha": tip, "is_ancestor": verdict}
        if rules:
            payload["requested_ref"] = requested
        is_error = verdict is None
        reason = None
        if is_error:
            reason = ("commit-missing" if not commit else "ref-not-recorded" if tip is None
                      else "commit-unknown-to-oracle")
            payload["reason"] = reason
        source = ReceiptSource.ERROR if is_error else ReceiptSource.ORACLE
        match = ArgumentMatch(dict(arguments), {"commit": commit, "ref": ref}, rules) if rules and ref else None
        return self._issue(name, arguments, payload, sha256_hex(canonical_json(payload)), source,
                           match=match, error_reason=reason), payload, is_error

    def _issue(self, name: str, arguments: Mapping[str, Any], payload: Any, digest: str,
               source: ReceiptSource, match: ArgumentMatch | None = None, error_reason: str | None = None) -> ToolReceipt:
        return self.receipts.issue(server=self.name, tool=name, arguments=arguments, response=payload,
                                   response_sha256=digest, source=source, snapshot=self.snapshot,
                                   match=match, error_reason=error_reason)


def episode_servers(snapshot: Snapshot, ledger: DojoLedger, oracle: AncestryOracle | None,
                    unavailable: frozenset[str] | tuple[str, ...] = (),
                    memories: tuple = ()) -> tuple[dict[str, _McpServer], ReceiptLog]:
    """The server set one episode sees, sharing one receipt log. `unavailable` names Anima tools the replay
    answers with an error (the spec's `replay_faults`). `memories` (the spec's `RecalledMemory` seeds) adds the
    `akh` server with her `memory_recall` / `memory_write`; without seeds the server is not offered, so every
    other family's tool list is unchanged."""
    log = ReceiptLog()
    servers: dict[str, _McpServer] = {"anima": ReplayServer(snapshot, ledger, log, unavailable=frozenset(unavailable)),
                                      "git": DojoGitServer(snapshot, oracle, log)}
    if memories:
        from dojo.replay_memory import MEMORY_SERVER, MemoryReplayServer

        servers[MEMORY_SERVER] = MemoryReplayServer(snapshot, memories, ledger, log)
    return servers, log


def servers_for_spec(spec, snapshot: Snapshot, ledger: DojoLedger,
                     oracle: AncestryOracle | None) -> tuple[dict[str, _McpServer], ReceiptLog]:
    """`episode_servers` with what the spec asks for (its replay faults and seeded memories)."""
    return episode_servers(snapshot, ledger, oracle, spec.replay_faults, spec.memories)


# --- CLI -------------------------------------------------------------------------------


def serve(server: _McpServer, stdin=sys.stdin, stdout=sys.stdout) -> None:
    for line in stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            stdout.write(json.dumps(_error(None, -32700, "parse error")) + "\n")
            continue
        response = server.handle(request)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lilith CoS dojo replay MCP server (stdio)")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--root", type=Path, required=True)
    s.add_argument("--snapshot", required=True)
    s.add_argument("--snapshot-until", default=None, metavar="CUTOFF",
                   help="only snapshots at or before CUTOFF exist for this server (snapshot id or ISO timestamp)")
    s.add_argument("--ledger", type=Path, required=True)
    s.add_argument("--server", choices=["anima", "git"], default="anima")
    s.add_argument("--repo", type=Path, default=None, help="local clone for git_is_ancestor")
    args = parser.parse_args(argv)
    try:
        store = SnapshotStore.open(args.root, args.snapshot_until)
        snapshot = store.get(args.snapshot)
        if args.server == "anima":
            server: _McpServer = ReplayServer(snapshot, DojoLedger(args.ledger, forbidden_roots=(args.root,)))
        else:
            server = DojoGitServer(snapshot, GitAncestryOracle(args.repo) if args.repo else None)
    except DojoError as error:
        print(error.render(), file=sys.stderr)
        return 2
    serve(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
