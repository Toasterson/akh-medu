#!/usr/bin/env python3
"""Diff Codex record/payload types across observed cli_version values.

Used for docs/wayfinder/transcript-harvest/assets/source-format-stability.md (Q1).
Run: python3 codex_version_diff.py
"""
import json, glob, os, collections

ROOT = os.path.expanduser("~/.codex/sessions")
files = sorted(glob.glob(os.path.join(ROOT, "**", "*.jsonl"), recursive=True))

file_version = {}
for fp in files:
    with open(fp, errors="replace") as fh:
        first = fh.readline()
        try:
            obj = json.loads(first)
        except Exception:
            continue
        if obj.get("type") == "session_meta":
            file_version[fp] = obj.get("payload", {}).get("cli_version")

by_version_types = collections.defaultdict(collections.Counter)
by_version_payload_types = collections.defaultdict(collections.Counter)
session_meta_positions = collections.defaultdict(list)

for fp in files:
    v = file_version.get(fp, "<unknown>")
    with open(fp, errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get("type")
            by_version_types[v][t] += 1
            if t == "session_meta" and lineno > 1:
                session_meta_positions[fp].append(lineno)
            payload = obj.get("payload")
            if isinstance(payload, dict):
                pt = payload.get("type")
                by_version_payload_types[v][(t, pt)] += 1

print("== top-level types by cli_version ==")
for v in by_version_types:
    print(f"version {v}: {sorted(by_version_types[v].keys())}")

print()
print("== payload types present in 0.148.0 but NOT in 0.153.4 (or vice versa) ==")
v1_keys = set(by_version_payload_types.get("0.148.0", {}).keys())
v2_keys = set(by_version_payload_types.get("0.153.4", {}).keys())
print("only in 0.148.0:", sorted(v1_keys - v2_keys))
print("only in 0.153.4:", sorted(v2_keys - v1_keys))
print("shared:", len(v1_keys & v2_keys))

print()
print("== files with mid-stream session_meta re-emission (line>1) ==")
for fp, lns in session_meta_positions.items():
    print(f"  {os.path.basename(fp)}: lines {lns}")
