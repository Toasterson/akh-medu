#!/usr/bin/env python3
"""Tecton memory exercise — T1 (episodic growth) + T2 (retrieval+action).

Drives Tecton live over the Python ACP adapter. Each probe runs in a FRESH
session so a correct answer must come from the knowledge graph (recall), not
conversation carry-over. Grounded in facts from the 38 imported OpenClaw
journals. Scores retrieval (did memory_recall fire?) and action (does the answer
contain the grounded fact?). Measures episodic growth via the PG WAL + episode
symbols.

Run:  python3 scripts/exercise_memory.py [--strategy-label compacting]
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acp_adapter import AkhClient
import psycopg2

PGC = os.environ.get("PGCERTS", "/tmp/pgc2")
TID = "6ee4a554-004b-51ac-a6a9-25c16c510492"  # tecton akh_id

# Grounded probes: question + expected keywords (answer is "correct" if it
# contains ANY of the all_of groups fully, or ANY single any_of keyword).
PROBES = [
    {"id": "cluster-nodes", "q": "From your own memory: how many nodes are in the home Talos cluster, and what infrastructure failure happened on May 9 2026?",
     "any_of": ["11 node", "eleven node"], "also": ["switch 2", "switch2", "outage"]},
    {"id": "alert-routing", "q": "From your memory of the Anima observability design: where do CRITICAL alerts get routed versus WARNING alerts?",
     "any_of": ["telegram"], "also": ["moltbook"]},
    {"id": "similarity-bug", "q": "From your memory: what was the root cause of every VSA similarity score coming back as 1.000?",
     "any_of": ["ground_symbols", "grounding never", "vsa grounding", "never applied", "never called"]},
    {"id": "utf8-panic", "q": "From your memory: there was a UTF-8 / byte-index panic bug — which source file and line was it in?",
     "any_of": ["memory.rs:456", "agent/memory.rs", "456"]},
    {"id": "merge-loop", "q": "From your memory of the Anima pipeline findings: since roughly when had the merge-loop been broken?",
     "any_of": ["apr 30", "april 30", "30 apr", "04-30", "may 9", "switch"]},  # accept outage-linked
    {"id": "archibald-disk", "q": "From your memory: the Archibald maintenance design tackled disk pressure — what were the two big sources of wasted disk it cleaned up?",
     "any_of": ["worktree", "153g", "153 g"], "also": ["docker"]},
    {"id": "redb-path", "q": "From your memory: what is the redb knowledge-graph lock/db file path that was discovered?",
     "any_of": [".akhomed/data/knowledge-graph.db", "knowledge-graph.db"]},
    {"id": "hephaistos", "q": "From your memory of the cluster topology: is Hephaistos a Kubernetes node? What is it actually?",
     "any_of": ["not a k8s", "not a kubernetes", "workstation", "gaming", "7900 xtx"]},
]


def pg():
    return psycopg2.connect(host="192.168.1.32", port=31004, dbname="akhomed", user="akhomed",
        sslmode="require", sslrootcert=f"{PGC}/ca.crt", sslcert=f"{PGC}/mac-client.crt",
        sslkey=f"{PGC}/mac-client.key", connect_timeout=10)


def growth_snapshot():
    """(wal_seq, episode_count) from PG, or (None, None) if PG is unreachable —
    T1 growth is skipped gracefully so T2 (ACP retrieval) still runs."""
    try:
        c = pg(); cur = c.cursor()
        cur.execute("SELECT COALESCE(max(seq),0) FROM akhomed_mutation_log WHERE akh_id=%s", (TID,))
        wal = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM akhomed_symbols WHERE akh_id=%s AND label LIKE 'episode:%%'", (TID,))
        eps = cur.fetchone()[0]
        c.close()
        return wal, eps
    except Exception as e:
        print(f"[T1 skipped] PG unreachable: {str(e)[:80]}")
        return None, None


def score(answer: str, probe: dict) -> bool:
    a = answer.lower()
    hit_any = any(k.lower() in a for k in probe.get("any_of", []))
    also = probe.get("also")
    if also:
        hit_also = any(k.lower() in a for k in also)
        return hit_any and hit_also
    return hit_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy-label", default=os.environ.get("AKH_CONTEXT_STRATEGY", "compacting"))
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    wal0, eps0 = growth_snapshot()
    have_pg = wal0 is not None
    print(f"=== Tecton memory exercise · strategy={args.strategy_label} ===")
    if have_pg:
        print(f"[T1 pre]  WAL seq={wal0}  episode-symbols={eps0}\n")

    results = []
    for p in PROBES:
        with AkhClient("tecton", timeout=args.timeout) as c:  # fresh session each probe
            turn = c.prompt(p["q"])
        retrieved = "memory_recall" in turn.tools_used or "archival_search" in turn.tools_used
        correct = score(turn.answer, p)
        results.append({"id": p["id"], "retrieved": retrieved, "correct": correct,
                        "tools": sorted(set(turn.tools_used)), "elapsed": round(turn.elapsed_s, 1),
                        "answer": turn.answer.strip()[:280]})
        flag = "OK " if correct else ("~  " if retrieved else "XX ")
        print(f"[{flag}] {p['id']:16s} retrieved={retrieved!s:5s} correct={correct!s:5s} "
              f"{turn.elapsed_s:5.1f}s tools={','.join(sorted(set(turn.tools_used))) or '-'}")
        print(f"        A: {turn.answer.strip()[:200]}")

    time.sleep(4)  # let WAL flush
    wal1, eps1 = growth_snapshot() if have_pg else (None, None)
    n = len(results)
    acc = sum(r["correct"] for r in results)
    ret = sum(r["retrieved"] for r in results)
    print(f"\n[T2] retrieval-rate={ret}/{n}  answer-accuracy={acc}/{n}")
    wal_delta = eps_delta = None
    if have_pg and wal1 is not None:
        wal_delta, eps_delta = wal1 - wal0, eps1 - eps0
        print(f"[T1 post] WAL seq={wal1} (+{wal_delta})  episode-symbols={eps1} (+{eps_delta})")
    out = {"strategy": args.strategy_label, "n": n, "accuracy": acc, "retrieval": ret,
           "wal_delta": wal_delta, "episode_delta": eps_delta, "results": results}
    path = f"/tmp/exercise_{args.strategy_label}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
