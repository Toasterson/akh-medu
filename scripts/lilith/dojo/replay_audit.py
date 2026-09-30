#!/usr/bin/env python3
"""Audit the replay misses of finished dojo runs (plan slice C4, ADR 116 Decision 3).

For every `source: error` receipt in one or more result directories (`episodes.jsonl`), re-ask
the replay resolver against the receipt's snapshot and classify the miss:

- (a) equivalent   the call is provably the same query as a recorded one under the tool's own
                   input schema (`replay_match`): the current replay answers it from recorded bytes;
- (b) unrecorded   a genuinely unrecorded query: the tool was never recorded in that snapshot, or
                   it was recorded only for other identifiers (same argument keys, other values);
- (c) other        recorded tool, but the argument shape differs in a way that is not an
                   equivalence (a filter the recorder did not apply, a larger page than was
                   recorded, a missing scope argument, an object the local git clone lacks).

Read-only over the results and the snapshot root. Prints a markdown table.

    replay_audit.py --root "$LILITH_SNAPSHOT_ROOT" [--repo PATH] RESULT_DIR [RESULT_DIR ...]
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo.replay_match import recorded_shape, resolve_recorded_read, resolve_ref
from dojo.replay_server import ANIMA_READ_TOOLS, AncestryOracle, GitAncestryOracle
from dojo.snapshot import MAIN_REF, Snapshot, SnapshotStore

CLASS_EQUIVALENT = "a"
CLASS_UNRECORDED = "b"
CLASS_OTHER = "c"


@dataclasses.dataclass(frozen=True)
class Miss:
    server: str
    tool: str
    argument_keys: tuple[str, ...]
    old_reason: str
    klass: str
    detail: str

    @property
    def group(self) -> tuple:
        return (self.server, self.tool, self.argument_keys, self.old_reason, self.klass, self.detail)


def classify(snapshot: Snapshot, server: str, tool: str, arguments: Mapping[str, Any],
             oracle: AncestryOracle | None = None) -> Miss:
    keys = tuple(sorted(arguments))
    if server == "git":
        return _classify_git(snapshot, tool, arguments, keys, oracle)
    schema = None
    contract = snapshot.tool_contract()
    if contract is not None:
        schema = next((t.get("inputSchema") for t in contract[0] if t.get("name") == tool), None)
    old = recorded_shape(snapshot, tool) if tool in ANIMA_READ_TOOLS else "tool-not-recorded"
    if tool in ANIMA_READ_TOOLS:
        found = resolve_recorded_read(snapshot, tool, schema, arguments)
        if found is not None:
            return Miss(server, tool, keys, old, CLASS_EQUIVALENT, ", ".join(found.match.rules))
        recorded_keys = {tuple(sorted(k for k, v in r.args.items() if v is not None)) for r in snapshot.by_call(tool)}
        if keys in recorded_keys:
            return Miss(server, tool, keys, old, CLASS_UNRECORDED, "identifier value not recorded")
        return Miss(server, tool, keys, old, CLASS_OTHER, "argument shape differs from every recorded call")
    return Miss(server, tool, keys, old, CLASS_UNRECORDED, "tool never recorded")


def _classify_git(snapshot: Snapshot, tool: str, arguments: Mapping[str, Any], keys: tuple[str, ...],
                  oracle: AncestryOracle | None) -> Miss:
    requested = arguments.get("ref")
    ref, rules = resolve_ref(snapshot.git_refs, str(requested) if requested else None)
    if tool != "git_is_ancestor":
        return Miss("git", tool, keys, "arguments-not-recorded", CLASS_OTHER, "unexpected git tool")
    old = "ref-not-recorded" if str(requested or MAIN_REF) not in snapshot.git_refs else "commit-unknown-to-oracle"
    tip = snapshot.git_refs.get(ref) if ref else None
    if tip is None:
        return Miss("git", tool, keys, old, CLASS_UNRECORDED, "ref never recorded")
    commit = str(arguments.get("commit") or "")
    if oracle is not None and commit and oracle.is_ancestor(commit, tip) is None:
        return Miss("git", tool, keys, old, CLASS_OTHER, "commit unknown to the local clone")
    detail = ", ".join(rules) if rules else "answered"
    return Miss("git", tool, keys, old, CLASS_EQUIVALENT, detail if oracle is not None else f"{detail} (oracle not consulted)")


def audit(result_dir: Path, store: SnapshotStore, oracle: AncestryOracle | None) -> tuple[collections.Counter, int, int]:
    groups: collections.Counter = collections.Counter()
    total = errors = 0
    for line in (result_dir / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        for receipt in json.loads(line).get("receipts", []):
            total += 1
            if receipt.get("source") != "error":
                continue
            errors += 1
            snapshot = store.get(receipt["snapshot_id"])
            groups[classify(snapshot, receipt["server"], receipt["tool"], receipt.get("arguments") or {}, oracle).group] += 1
    return groups, errors, total


def render(name: str, groups: collections.Counter, errors: int, total: int) -> str:
    lines = [f"### {name}: {errors}/{total} receipts errored", "",
             "| n | server.tool | argument keys | old reason | class | detail |", "|---|---|---|---|---|---|"]
    for group, n in sorted(groups.items(), key=lambda kv: (-kv[1], kv[0])):
        server, tool, keys, old, klass, detail = group
        lines.append(f"| {n} | {server}.{tool} | {', '.join(keys) or '(none)'} | {old} | ({klass}) | {detail} |")
    by_class = collections.Counter()
    for group, n in groups.items():
        by_class[group[4]] += n
    lines += ["", "Totals: " + ", ".join(f"({k}) {by_class[k]}" for k in "abc"), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="snapshot root")
    parser.add_argument("--repo", type=Path, default=None, help="local clone for the ancestry oracle")
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args(argv)
    store = SnapshotStore(args.root)
    oracle = GitAncestryOracle(args.repo) if args.repo else None
    for result in args.results:
        groups, errors, total = audit(result, store, oracle)
        print(render(result.name, groups, errors, total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
