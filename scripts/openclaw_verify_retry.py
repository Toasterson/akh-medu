#!/usr/bin/env python3
"""Verify-retry import: for each OpenClaw journal whose source tag is NOT yet in
Postgres, re-run an inline memory_write turn and confirm the tag:source:<path>
symbol appears. Retries per file. Trusts the KG, not the LLM's narration."""
import os, re, subprocess, sys, time
import psycopg2

AKH = os.environ.get("AKH_BIN", "/Users/toasty/ws/akh-medu/target/release/akh")
WS = "tecton"
MEM = os.path.expanduser("~/.openclaw/workspace/memory")
CERT = "/tmp/pgcerts"
MAX_ATTEMPTS = 3
TURN_TIMEOUT = 300

def pg():
    return psycopg2.connect(host="192.168.1.32", port=31004, dbname="akhomed", user="akhomed",
        sslmode="require", sslrootcert=f"{CERT}/ca.crt", sslcert=f"{CERT}/mac-client.crt",
        sslkey=f"{CERT}/mac-client.key", connect_timeout=10)

def tecton_id(cur):
    cur.execute("SELECT id FROM akhomed_akh WHERE name='tecton'"); return cur.fetchone()[0]

def covered_files(cur, tid):
    cur.execute("SELECT label FROM akhomed_symbols WHERE akh_id=%s AND label LIKE 'tag:source:%%memory/%%.md'", (tid,))
    out = set()
    for (l,) in cur.fetchall():
        m = re.search(r'memory/([A-Za-z0-9._-]+\.md)$', l)
        if m: out.add(m.group(1))
    return out

def run_turn(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = re.sub(r'\s+', ' ', fh.read())
    prompt = (f"Below is the full text of an OpenClaw memory journal at {path}. Do NOT read any file. "
              f"You MUST call the memory_write tool exactly once. Its `source` argument is REQUIRED and "
              f"must be the literal string {path}. Provide a concise factual `summary` and set origin to "
              f"openclaw. Do not answer in prose without calling the tool. TEXT: {text}")
    try:
        subprocess.run([AKH, "-w", WS, "chat", "--headless"], input=prompt + "\n",
                       text=True, timeout=TURN_TIMEOUT, capture_output=True)
    except subprocess.TimeoutExpired:
        print(f"    (turn timed out after {TURN_TIMEOUT}s)")

def main():
    all_md = sorted(f for f in os.listdir(MEM) if f.endswith(".md"))
    conn = pg(); cur = conn.cursor(); tid = tecton_id(cur)
    missing = sorted(set(all_md) - covered_files(cur, tid))
    conn.close()
    print(f"{len(all_md)} journals; {len(missing)} missing source tags:\n  " + ", ".join(missing))
    for name in missing:
        path = os.path.join(MEM, name)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f">>> {name} (attempt {attempt})", flush=True)
            run_turn(path)
            time.sleep(3)  # let WAL flush (2s interval)
            conn = pg(); cur = conn.cursor()
            cur.execute("SELECT 1 FROM akhomed_symbols WHERE akh_id=%s AND label=%s LIMIT 1",
                        (tid, f"tag:source:{path}"))
            ok = cur.fetchone() is not None
            conn.close()
            if ok:
                print(f"    OK source tag present"); break
            print(f"    still missing; {'retrying' if attempt<MAX_ATTEMPTS else 'GIVING UP'}")
    # final reconcile
    conn = pg(); cur = conn.cursor()
    still = sorted(set(all_md) - covered_files(cur, tid))
    conn.close()
    print(f"\nFINAL: {len(all_md)-len(still)}/{len(all_md)} covered in PG. "
          + ("ALL DONE" if not still else "STILL MISSING: " + ", ".join(still)))

if __name__ == "__main__":
    main()
