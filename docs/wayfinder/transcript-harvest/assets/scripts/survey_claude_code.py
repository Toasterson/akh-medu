#!/usr/bin/env python3
"""Survey Claude Code ~/.claude/projects/**/*.jsonl for version markers and record types.

Used for docs/wayfinder/transcript-harvest/assets/source-format-stability.md (Q1/Q2).
Run: python3 survey_claude_code.py
"""
import json, glob, os, collections

ROOT = os.path.expanduser("~/.claude/projects")
files = sorted(glob.glob(os.path.join(ROOT, "**", "*.jsonl"), recursive=True))
print(f"# files found: {len(files)}")

version_counts = collections.Counter()
type_counts = collections.Counter()
type_by_version = collections.defaultdict(lambda: collections.Counter())
keys_seen_by_type = collections.defaultdict(set)
total_lines = 0
bad_json_lines = 0
lines_missing_version = 0
file_versions = {}
missing_type_lines = 0

for fp in files:
    seen_versions_this_file = set()
    with open(fp, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_json_lines += 1
                continue
            if not isinstance(obj, dict):
                continue
            v = obj.get("version")
            if v is None:
                lines_missing_version += 1
            else:
                version_counts[v] += 1
                seen_versions_this_file.add(v)
            t = obj.get("type")
            if t is None:
                missing_type_lines += 1
                t = "<no-type-field>"
            type_counts[t] += 1
            if v is not None:
                type_by_version[v][t] += 1
            for k in obj.keys():
                keys_seen_by_type[t].add(k)
    file_versions[fp] = seen_versions_this_file

print(f"total lines: {total_lines}, bad json lines: {bad_json_lines}, "
      f"lines missing 'version': {lines_missing_version}, lines missing 'type': {missing_type_lines}")
print()
print("== version value counts ==")
for v, c in version_counts.most_common():
    print(f"  {v!r}: {c}")

print()
print("== distinct versions per file (files with >1 version are interesting) ==")
multi = {fp: vs for fp, vs in file_versions.items() if len(vs) > 1}
print(f"files with a single version: {sum(1 for vs in file_versions.values() if len(vs) == 1)}")
print(f"files with >1 version: {len(multi)}")
for fp, vs in multi.items():
    print(f"  {os.path.basename(fp)}: {sorted(vs)}")

print()
print("== top-level 'type' counts ==")
for t, c in type_counts.most_common():
    print(f"  {t!r}: {c}")

print()
print("== keys present per type (union across corpus) ==")
for t in sorted(keys_seen_by_type):
    print(f"  {t}: {sorted(keys_seen_by_type[t])}")

print()
print("== type x version matrix (did type-set change across versions?) ==")
for v in sorted(version_counts, key=str):
    types_here = sorted(type_by_version[v].keys())
    print(f"  version {v!r}: types={types_here}")
