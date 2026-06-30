#!/usr/bin/env python3
"""Drive Tecton through public long-term-memory benchmarks over ACP.

Supports two datasets:
  * LoCoMo (benchmarks/locomo/locomo10.json) — multi-session dialogues + QA.
  * LongMemEval_s (benchmarks/longmemeval/longmemeval_s.json) — 500 questions,
    each with a 53-session haystack. Full haystacks are far too expensive to
    replay over a cloud LLM turn-by-turn, so this runner uses an **oracle**
    setting by default: it injects only the answer-bearing sessions
    (`answer_session_ids`), testing recall-from-context rather than full
    long-context retrieval. Use --haystack to inject the first N sessions.

Each item: open ONE ACP session, replay the relevant dialogue sessions as
user turns (Tecton "hears" them), then ask the question and score the answer
against the gold (normalized containment + token-F1; raw answers are saved for
review since short-factual judging is noisy).

Costs are real (one GLM turn per injected session + one per question), so this
SAMPLES by default. Bump --sample for a fuller run.

Usage:
  python3 benchmarks/run_memory_bench.py --benchmark locomo --sample 3 --questions 5
  python3 benchmarks/run_memory_bench.py --benchmark longmemeval --sample 10
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from acp_adapter import AkhClient

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() and " ".join(
        re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()) or ""


def token_f1(pred: str, gold: str) -> float:
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def judge(pred: str, gold: str) -> tuple[bool, float]:
    """correct if gold is contained in pred (normalized) or token-F1 >= 0.5."""
    n_pred, n_gold = norm(pred), norm(gold)
    contained = bool(n_gold) and n_gold in n_pred
    f1 = token_f1(pred, gold)
    return (contained or f1 >= 0.5), f1


# ── dataset loaders → list of {sessions:[(label,text)], qas:[(q,a,meta)]} ──────
def load_locomo(sample: int, n_questions: int):
    data = json.load(open(os.path.join(HERE, "locomo", "locomo10.json")))
    items = []
    for conv in data[:sample]:
        c = conv["conversation"]
        a, b = c.get("speaker_a", "A"), c.get("speaker_b", "B")
        sessions = []
        i = 1
        while f"session_{i}" in c:
            turns = c[f"session_{i}"]
            date = c.get(f"session_{i}_date_time", "")
            lines = []
            for t in turns:
                spk = t.get("speaker", "")
                txt = t.get("text", "")
                if txt:
                    lines.append(f"{spk}: {txt}")
            sessions.append((f"session {i} ({date})", "\n".join(lines)))
            i += 1
        qas = [(q["question"], str(q["answer"]), {"category": q.get("category")})
               for q in conv["qa"] if q.get("answer") is not None][:n_questions]
        items.append({"id": conv.get("sample_id", "?"), "speakers": (a, b),
                      "sessions": sessions, "qas": qas})
    return items


def load_longmemeval(sample: int, haystack: int):
    data = json.load(open(os.path.join(HERE, "longmemeval", "longmemeval_s.json")))
    items = []
    for it in data[:sample]:
        sids = it["haystack_session_ids"]
        sessions_by_id = dict(zip(sids, it["haystack_sessions"]))
        if haystack > 0:
            chosen = sids[:haystack]
        else:  # oracle: only answer-bearing sessions
            chosen = it.get("answer_session_ids") or sids[:1]
        sessions = []
        for sid in chosen:
            turns = sessions_by_id.get(sid, [])
            lines = [f"{t.get('role','user')}: {t.get('content','')}" for t in turns
                     if t.get("content")]
            if lines:
                sessions.append((sid, "\n".join(lines)))
        items.append({"id": it["question_id"], "speakers": ("user", "assistant"),
                      "sessions": sessions,
                      "qas": [(it["question"], str(it["answer"]),
                               {"type": it.get("question_type")})]})
    return items


def run(items, label, timeout, max_inject_chars):
    results = []
    for it in items:
        with AkhClient("tecton", timeout=timeout) as c:
            # Inject each relevant session as a "remember this" turn.
            for sname, stext in it["sessions"]:
                stext = stext[:max_inject_chars]
                c.prompt(f"Here is a past conversation to remember ({sname}). "
                         f"Acknowledge briefly.\n\n{stext}")
            for q, gold, meta in it["qas"]:
                turn = c.prompt(f"Based only on the conversations you were just given, "
                                f"answer concisely: {q}")
                ok, f1 = judge(turn.answer, gold)
                results.append({"item": it["id"], "q": q, "gold": gold,
                                "pred": turn.answer.strip()[:240], "correct": ok,
                                "f1": round(f1, 2), "elapsed": round(turn.elapsed_s, 1)})
                flag = "OK" if ok else "XX"
                print(f"[{flag}] {it['id']!s:>10} f1={f1:.2f} {turn.elapsed_s:4.0f}s "
                      f"Q:{q[:50]} | gold:{gold[:30]} | pred:{turn.answer.strip()[:60]}")
    n = len(results)
    acc = sum(r["correct"] for r in results)
    mf1 = sum(r["f1"] for r in results) / n if n else 0.0
    print(f"\n[{label}] accuracy={acc}/{n}  mean_f1={mf1:.2f}")
    out = {"benchmark": label, "n": n, "accuracy": acc, "mean_f1": round(mf1, 2),
           "results": results}
    path = os.path.join(HERE, f"results_{label}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["locomo", "longmemeval"], required=True)
    ap.add_argument("--sample", type=int, default=3, help="how many conversations/items")
    ap.add_argument("--questions", type=int, default=5, help="LoCoMo: QAs per conversation")
    ap.add_argument("--haystack", type=int, default=0,
                    help="LongMemEval: inject first N sessions (0 = oracle = answer sessions only)")
    ap.add_argument("--max-sessions", type=int, default=0,
                    help="cap injected sessions per item (0 = all); bounds LoCoMo cost")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--max-inject-chars", type=int, default=6000)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    if args.benchmark == "locomo":
        items = load_locomo(args.sample, args.questions)
    else:
        items = load_longmemeval(args.sample, args.haystack)
    if args.max_sessions > 0:
        for it in items:
            it["sessions"] = it["sessions"][:args.max_sessions]
    label = args.label or f"{args.benchmark}-s{args.sample}"
    nq = sum(len(it["qas"]) for it in items)
    ns = sum(len(it["sessions"]) for it in items)
    print(f"=== {label}: {len(items)} items, {ns} sessions to inject, {nq} questions "
          f"(~{ns+nq} GLM turns) ===")
    run(items, label, args.timeout, args.max_inject_chars)


if __name__ == "__main__":
    main()
