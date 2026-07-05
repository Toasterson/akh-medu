#!/usr/bin/env python3
"""Episodic self-refinement pass (ADR 078, execution half).

Drives the akh to re-read the sources behind its THIN episodes and extract the
concrete details (exact commits/paths/numbers/dates) it dropped at first import,
recording them as facts via memory_write. This is the "act" half of the
self-cleaning loop; the goal-system prototype (proposals_from_episodic_gaps) is
the "notice" half. Runs over ACP (small per-episode prompts — no cloud timeout).

A "thin" episode here = has a `tag:source:` we can re-read AND fewer than
--threshold distilled learnings. Detail re-extraction is grounded in the source,
so it also surfaces (and the akh can prune) import-time hallucinations.

Usage:
  python3 scripts/enrich_memory.py --max 3            # enrich up to 3 thin episodes
  python3 scripts/enrich_memory.py --source <file>    # enrich one specific source
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from acp_adapter import AkhClient

AKH = os.environ.get("AKH_BIN", "/Users/toasty/akh-medu/target/release/akh")
WS = "tecton"


def kg_counts() -> tuple[int, int]:
    out = subprocess.run([AKH, "-w", WS, "acp", "status"], capture_output=True, text=True).stdout
    d = json.loads(out)
    return d.get("symbols", 0), d.get("triples", 0)


def thin_episodes(threshold: int, limit: int) -> list[dict]:
    """Recent episodes with a source tag and < threshold learnings."""
    out = subprocess.run([AKH, "-w", WS, "acp", "recall", "--limit", str(limit)],
                         capture_output=True, text=True).stdout
    eps = json.loads(out).get("episodes", [])
    thin = []
    for e in eps:
        source = next((t[len("tag:source:"):] for t in e.get("tags", [])
                       if t.startswith("tag:source:")), None)
        if source and len(e.get("learnings", [])) < threshold:
            thin.append({"summary": e.get("summary", "")[:60], "source": source,
                         "learnings": len(e.get("learnings", []))})
    return thin


def enrich(source: str, timeout: float) -> dict:
    prompt = (
        f"Re-read your memory source file {source}. It contains specific technical "
        "values you may have dropped or generalized when you first recorded it — "
        "exact git commit hashes, file paths, IP addresses, version numbers, dates, "
        "PIDs. Extract those CONCRETE values and record them with memory_write "
        f"(source={source}, origin openclaw) as precise (subject, predicate, object) "
        "facts. If you find a fact you previously recorded that the source does NOT "
        "support, note it (it may have been a hallucination). Be exhaustive about "
        "specific identifiers; then stop.")
    with AkhClient(WS, timeout=timeout) as c:
        t = c.prompt(prompt)
    nfacts = None
    m = re.search(r"(\d+)\s+(?:concrete\s+)?facts?", t.answer)
    if m:
        nfacts = int(m.group(1))
    return {"source": source, "elapsed": round(t.elapsed_s, 1), "stop": t.stop_reason,
            "tools": sorted(set(t.tools_used)), "reported_facts": nfacts,
            "answer_head": t.answer.strip()[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="enrich one specific source file")
    ap.add_argument("--max", type=int, default=3, help="max thin episodes to enrich")
    ap.add_argument("--threshold", type=int, default=6, help="learnings below = thin")
    ap.add_argument("--scan", type=int, default=60, help="episodes to scan for thin ones")
    ap.add_argument("--timeout", type=float, default=150.0)
    args = ap.parse_args()

    if args.source:
        targets = [args.source]
    else:
        thin = thin_episodes(args.threshold, args.scan)
        print(f"found {len(thin)} thin episode(s); enriching up to {args.max}:")
        for t in thin[:args.max]:
            print(f"  - {t['learnings']} learnings · {t['summary']} · {t['source']}")
        targets = [t["source"] for t in thin[:args.max]]
    if not targets:
        print("nothing to enrich.")
        return

    s0, tr0 = kg_counts()
    print(f"\n[KG before] symbols={s0} triples={tr0}\n")
    for src in targets:
        r = enrich(src, args.timeout)
        print(f"[enriched] {os.path.basename(src)} · {r['elapsed']}s · {r['stop']} · "
              f"~{r['reported_facts']} facts · tools={','.join(r['tools'])}")
        print(f"           {r['answer_head']}")
    s1, tr1 = kg_counts()
    print(f"\n[KG after]  symbols={s1} (+{s1-s0})  triples={tr1} (+{tr1-tr0})")


if __name__ == "__main__":
    main()
