#!/usr/bin/env python3
"""Render-format prototype: High-Grug-style transcript render vs raw JSON.

Measures BYTES (exact). Token ratios are NOT measured - see the ticket.
Usage: python3 render_prototype.py
"""
import json, glob, os, collections

SEP = " | "

def trunc(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else f"{s[:n//2]}…[{len(s)-n} elided]…{s[-n//2:]}"

def render(kind, fields, seq):
    parts = [f"{seq} {kind}"] + [f"{k} {v}" for k, v in fields if v]
    return SEP.join(parts)

# Tier caps from the read-surface decision: prose generous, process output hard.
CAP = {"prompt": 100000, "reply": 100000, "thought": 100000,
       "tool_call": 400, "tool_result": 600,
       "context": 200, "session_meta": 200, "lifecycle": 200,
       "error": 400, "telemetry": 80}

def claude(path, out):
    seq = 0
    for line in open(path, errors="replace"):
        seq += 1
        raw = len(line)
        try: d = json.loads(line)
        except Exception: continue
        t = d.get("type"); m = d.get("message")
        blocks = m.get("content") if isinstance(m, dict) else None
        if isinstance(blocks, str): blocks = [{"type": "text", "text": blocks}]
        if not isinstance(blocks, list):
            k = {"attachment": "context", "system": "error", "ai-title": "session_meta",
                 "mode": "session_meta", "started": "lifecycle", "result": "lifecycle"}.get(t)
            if not k: out(None, raw); continue
            out(render(k, [("t", trunc(str(d.get("content") or d.get("aiTitle")
                or d.get("mode") or d.get("agentId") or ""), CAP[k]))], seq), raw)
            continue
        for b in blocks:
            if not isinstance(b, dict): continue
            bt = b.get("type")
            if bt == "text":      out(render("reply" if t=="assistant" else "prompt",
                                             [("t", trunc(b.get("text"), CAP["reply"]))], seq), raw)
            elif bt == "thinking":out(render("thought", [("t", trunc(b.get("thinking"), CAP["thought"]))], seq), raw)
            elif bt == "tool_use":out(render("tool_call", [("n", b.get("name")),
                                             ("a", trunc(json.dumps(b.get("input")), CAP["tool_call"]))], seq), raw)
            elif bt == "tool_result": out(render("tool_result",
                                             [("o", trunc(json.dumps(b.get("content"))[:5000], CAP["tool_result"]))], seq), raw)
            else: out(None, raw)

def codex(path, out):
    seq = 0
    for line in open(path, errors="replace"):
        seq += 1
        raw = len(line)
        try: d = json.loads(line)
        except Exception: continue
        if d.get("type") != "response_item": out(None, raw); continue
        p = d.get("payload") or {}; pt = p.get("type")
        if pt == "message":
            txt = "".join(x.get("text","") for x in (p.get("content") or []) if isinstance(x, dict))
            k = "reply" if p.get("role") == "assistant" else ("prompt" if p.get("role")=="user" else "context")
            out(render(k, [("t", trunc(txt, CAP[k]))], seq), raw)
        elif pt == "agent_message": out(render("reply", [("t", trunc(p.get("message"), CAP["reply"]))], seq), raw)
        elif pt == "reasoning":
            # MEASURED: 100% of Codex reasoning is summary:[] + encrypted_content.
            # No readable text exists; emit a marker that thinking occurred.
            s = p.get("summary")
            if s: out(render("thought", [("t", trunc(json.dumps(s), CAP["thought"]))], seq), raw)
            else: out(render("thought", [("enc", "1")], seq), raw)
        elif pt in ("custom_tool_call","function_call"):
            out(render("tool_call", [("n", p.get("name")), ("a", trunc(json.dumps(p.get("input") or p.get("arguments")), CAP["tool_call"]))], seq), raw)
        elif pt in ("custom_tool_call_output","function_call_output"):
            out(render("tool_result", [("o", trunc(json.dumps(p.get("output")), CAP["tool_result"]))], seq), raw)
        else: out(None, raw)

stats = collections.Counter(); kindb = collections.Counter(); kindn = collections.Counter()
kindraw = collections.Counter()
def mk():
    def out(rendered, raw):
        stats["raw"] += raw
        if rendered is None: stats["dropped_raw"] += raw; stats["dropped_n"] += 1; return
        stats["rendered"] += len(rendered) + 1; stats["kept_n"] += 1
        k = rendered.split(" ",2)[1]; kindb[k] += len(rendered)+1; kindn[k] += 1
        kindraw[k] += raw
    return out

os.chdir(os.path.expanduser("~/.claude/projects"))
for f in glob.glob("**/*.jsonl", recursive=True): claude(f, mk())
os.chdir(os.path.expanduser("~/.codex/sessions"))
for f in glob.glob("**/*.jsonl", recursive=True): codex(f, mk())

raw, rend = stats["raw"], stats["rendered"]
print(f"raw corpus         {raw/1e6:9.1f} MB")
print(f"rendered           {rend/1e6:9.1f} MB   ({100*rend/raw:.1f}% of raw, {raw/rend:.1f}x smaller)")
print(f"  dropped as noise {stats['dropped_raw']/1e6:9.1f} MB in {stats['dropped_n']:,} records")
kept_raw = raw - stats["dropped_raw"]
print(f"  events rendered  {stats['kept_n']:,}")
print(f"\nraw of RENDERED records only: {kept_raw/1e6:.1f} MB -> {rend/1e6:.1f} MB = {kept_raw/rend:.1f}x")
print(f"\n{'kind':14}{'raw MB':>9}{'rend MB':>9}{'ratio':>8}{'events':>9}  lossy?")
LOSSY = {"tool_call","tool_result","context","session_meta","error","telemetry"}
for k, v in kindb.most_common():
    r = kindraw[k]
    print(f"{k:14}{r/1e6:>9.2f}{v/1e6:>9.2f}{r/max(v,1):>7.1f}x{kindn[k]:>9,}  {'TRUNCATED' if k in LOSSY else 'lossless'}")
pr = sum(kindraw[k] for k in ("prompt","reply")); pv = sum(kindb[k] for k in ("prompt","reply"))
print(f"\nPROSE ONLY (prompt+reply, never truncated): {pr/1e6:.2f} MB -> {pv/1e6:.2f} MB = {pr/pv:.1f}x")
print("  ^ this is the pure format-tax number, directly comparable to ADR 088.")
