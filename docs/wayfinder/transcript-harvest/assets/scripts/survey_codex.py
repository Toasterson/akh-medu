#!/usr/bin/env python3
"""Survey Codex ~/.codex/sessions/**/*.jsonl for version markers and record types.

Used for docs/wayfinder/transcript-harvest/assets/source-format-stability.md (Q1/Q2).
Run: python3 survey_codex.py
"""
import json, glob, os, collections

ROOT = os.path.expanduser("~/.codex/sessions")
files = sorted(glob.glob(os.path.join(ROOT, "**", "*.jsonl"), recursive=True))
print(f"# files found: {len(files)}")

type_counts = collections.Counter()
cli_versions = collections.Counter()
payload_type_counts = collections.Counter()
keys_by_type = collections.defaultdict(set)
file_cli_versions = collections.defaultdict(set)
total_lines = 0
bad = 0
first_line_types = collections.Counter()

for fp in files:
    with open(fp, "r", errors="replace") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            if i == 1:
                first_line_types[obj.get("type")] += 1
            t = obj.get("type", "<none>")
            type_counts[t] += 1
            for k in obj.keys():
                keys_by_type[t].add(k)
            payload = obj.get("payload")
            if isinstance(payload, dict):
                ptype = payload.get("type", "<none>")
                payload_type_counts[(t, ptype)] += 1
                if t == "session_meta":
                    cv = payload.get("cli_version")
                    if cv:
                        cli_versions[cv] += 1
                        file_cli_versions[fp].add(cv)

print(f"total lines: {total_lines}, bad json: {bad}")
print()
print("== first-line type (should be session header) ==")
for t, c in first_line_types.most_common():
    print(f"  {t!r}: {c}")
print()
print("== cli_version values (from session_meta payload) ==")
for v, c in cli_versions.most_common():
    print(f"  {v!r}: {c}")
print()
print("files with cli_version info:", len(file_cli_versions))
multi = {f: v for f, v in file_cli_versions.items() if len(v) > 1}
print("files with >1 distinct cli_version (shouldn't happen, one per file):", len(multi))
print()
print("== top-level 'type' counts ==")
for t, c in type_counts.most_common():
    print(f"  {t!r}: {c}")
print()
print("== keys per top-level type ==")
for t in sorted(keys_by_type):
    print(f"  {t}: {sorted(keys_by_type[t])}")
print()
print("== payload.type counts, grouped by top-level type ==")
for (t, pt), c in sorted(payload_type_counts.items(), key=lambda x: -x[1]):
    print(f"  {t} / payload.type={pt!r}: {c}")
