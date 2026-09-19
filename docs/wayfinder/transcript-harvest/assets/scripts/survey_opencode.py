#!/usr/bin/env python3
"""Survey the local OpenCode opencode.db (SQLite) for tables, migration history,
and event/message/part record shapes.

Used for docs/wayfinder/transcript-harvest/assets/source-format-stability.md (Q1/Q2/Q3).
Run: python3 survey_opencode.py
"""
import sqlite3, os

DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
cur = con.cursor()

print("== tables ==")
for row in cur.execute("select name from sqlite_master where type='table' order by name"):
    print(" ", row[0])

print()
print("== migration table (all rows) ==")
cols = [d[1] for d in cur.execute("pragma table_info(migration)")]
print("columns:", cols)
for row in cur.execute("select * from migration order by rowid"):
    print(" ", row)

for t in ["event", "message", "part", "session"]:
    print()
    print(f"== {t} table info ==")
    cols = [d[1] for d in cur.execute(f"pragma table_info({t})")]
    print("columns:", cols)
    cnt = cur.execute(f"select count(*) from {t}").fetchone()[0]
    print("row count:", cnt)

print()
print("== event.type distribution ==")
for row in cur.execute("select type, count(*) from event group by type order by count(*) desc"):
    print(" ", row)

print()
print("== sample event rows (first 3) ==")
for row in cur.execute("select aggregate_id, seq, type, data from event order by rowid limit 3"):
    agg, seq, typ, data = row
    print(f"  agg={agg} seq={seq} type={typ} data={str(data)[:300]}")
