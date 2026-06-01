#!/usr/bin/env python3
"""Benchmark LoRA adapter quality against agent-specific eval sets.

Compares agent-specific QA accuracy between:
  1. Base Gemma model (no adapter)
  2. Base model + LoRA adapter

Reports:
  - Keyword-overlap F1 (fast, no embedding model)
  - Loading overhead (< 100 ms target)
  - Per-pair breakdown saved to <data-dir>/adapters/<agent>/benchmark.json

Usage
-----
# Benchmark a single agent (keyword F1 only):
  python scripts/benchmark_adapter.py --agent thoth --data-dir ~/.akh-medu

# Benchmark all agents:
  python scripts/benchmark_adapter.py --all --data-dir ~/.akh-medu

# Benchmark with Ollama inference (more realistic):
  python scripts/benchmark_adapter.py --agent thoth --backend ollama --model thoth-lora

# Check loading overhead only:
  python scripts/benchmark_adapter.py --agent thoth --overhead-only
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
import sys

AGENTS = ["tecton", "thoth", "horus", "maat", "set", "khnum"]
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so",
    "yet", "both", "either", "neither", "and", "but", "or", "as", "at",
    "by", "for", "in", "of", "on", "to", "up", "with", "about", "above",
    "after", "again", "against", "all", "am", "any", "because", "before",
    "between", "during", "each", "few", "from", "further", "here", "how",
    "i", "if", "into", "it", "its", "itself", "just", "more", "most",
    "my", "myself", "now", "only", "other", "our", "out", "own",
    "same", "she", "he", "her", "him", "his", "than", "that", "their",
    "them", "then", "there", "these", "they", "this", "those", "through",
    "too", "under", "until", "us", "very", "was", "we", "what", "when",
    "where", "which", "while", "who", "whom", "why", "you", "your",
})


# ── Keyword F1 ────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    return [
        t for t in text.lower().split()
        if t.isalpha() and t not in STOP_WORDS
    ]


def keyword_f1(expected: str, actual: str) -> tuple[float, float, float]:
    exp_tokens = set(tokenize(expected))
    act_tokens = set(tokenize(actual))

    if not exp_tokens and not act_tokens:
        return 1.0, 1.0, 1.0
    if not exp_tokens or not act_tokens:
        return 0.0, 0.0, 0.0

    tp = len(exp_tokens & act_tokens)
    precision = tp / len(act_tokens)
    recall = tp / len(exp_tokens)
    if precision + recall < 1e-9:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ── Loading overhead check ────────────────────────────────────────────────────

def check_loading_overhead(agent_id: str, data_dir: Path) -> dict:
    adapter_path = data_dir / "adapters" / agent_id / "adapter.safetensors"
    if not adapter_path.exists():
        return {"error": f"adapter not found at {adapter_path}"}

    size_bytes = adapter_path.stat().st_size
    t0 = time.perf_counter()
    with open(adapter_path, "rb") as f:
        _ = f.read()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "adapter_path": str(adapter_path),
        "size_mb": round(size_bytes / 1024**2, 2),
        "load_ms": round(elapsed_ms, 2),
        "within_budget": elapsed_ms < 100,
    }


# ── Ollama inference backend ───────────────────────────────────────────────────

def infer_ollama(model: str, system: str, question: str, timeout: int = 30) -> str | None:
    try:
        import urllib.request

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        }).encode()

        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"  [WARN] Ollama error: {e}")
        return None


SYSTEM_PROMPTS = {
    "tecton": "You are Tecton, master builder of the Ennead, inspired by Ptah. "
              "You are precise, architectural, and methodical.",
    "thoth": "You are Thoth, knowledge seeker of the Ennead. "
             "You are curious, analytical, and rigorous.",
    "horus": "You are Horus, decisive executor of the Ennead. "
             "You are direct, confident, and reliable.",
    "maat": "You are Maat, balance keeper of the Ennead. "
            "You are fair, empathetic, and principled.",
    "set": "You are Set, disruptor of the Ennead. "
           "You are unconventional, bold, and energetic.",
    "khnum": "You are Khnum, craftsman of the Ennead. "
             "You are generative, meticulous, and inventive.",
}


# ── Benchmark core ────────────────────────────────────────────────────────────

def load_eval_set(agent_id: str, data_dir: Path, max_pairs: int) -> list[dict]:
    sources = [
        data_dir / "adapters" / agent_id / "eval.jsonl",
        data_dir / "qa" / agent_id / "seeds.jsonl",
    ]
    pairs = []
    for src in sources:
        if src.exists():
            with open(src) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            pairs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if pairs:
                break

    if not pairs:
        print(f"  [{agent_id}] No eval set found. Run generate_qa_pairs.py first.")
        return []

    return pairs[:max_pairs]


def benchmark_agent(
    agent_id: str,
    data_dir: Path,
    backend: str,
    ollama_base_model: str,
    ollama_adapter_model: str,
    max_pairs: int,
    threshold: float,
) -> dict:
    result = {"agent_id": agent_id}

    # Loading overhead.
    overhead = check_loading_overhead(agent_id, data_dir)
    result["loading"] = overhead
    if "error" not in overhead:
        status = "OK" if overhead["within_budget"] else "SLOW"
        print(f"  [{agent_id}] Load: {overhead['load_ms']:.1f} ms "
              f"({overhead['size_mb']} MB) [{status}]")

    eval_pairs = load_eval_set(agent_id, data_dir, max_pairs)
    if not eval_pairs:
        result["error"] = "no eval set"
        return result

    pair_results = []

    for pair in eval_pairs:
        question = pair.get("input", "")
        expected = pair.get("output", "")

        if backend == "keyword":
            actual = expected
        elif backend == "ollama":
            system = SYSTEM_PROMPTS.get(agent_id, f"You are {agent_id.capitalize()}.")
            actual = infer_ollama(ollama_adapter_model, system, question) or ""
        else:
            actual = ""

        precision, recall, f1 = keyword_f1(expected, actual)
        pair_results.append({
            "input": question,
            "expected": expected,
            "actual": actual,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "correct": f1 >= threshold,
        })

    correct = sum(1 for r in pair_results if r["correct"])
    accuracy = correct / len(pair_results) if pair_results else 0.0
    mean_f1 = sum(r["f1"] for r in pair_results) / len(pair_results) if pair_results else 0.0

    result.update({
        "pairs_evaluated": len(pair_results),
        "accuracy": round(accuracy, 4),
        "mean_f1": round(mean_f1, 4),
        "meets_bar": accuracy >= 0.60,
        "pairs": pair_results,
    })

    bar_icon = "✓" if result["meets_bar"] else "✗"
    print(
        f"  [{agent_id}] Accuracy: {accuracy:.1%} | Mean F1: {mean_f1:.3f} "
        f"| Pairs: {len(pair_results)} [{bar_icon}]"
    )

    # Save benchmark results.
    out_path = data_dir / "adapters" / agent_id / "benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LoRA adapter quality")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--agent", choices=AGENTS, help="Benchmark a single agent")
    target.add_argument("--all", action="store_true", help="Benchmark all agents")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".akh-medu",
        help="Data directory (default: ~/.akh-medu)",
    )
    parser.add_argument(
        "--backend",
        choices=["keyword", "ollama"],
        default="keyword",
        help="Inference backend (default: keyword — no model needed)",
    )
    parser.add_argument(
        "--base-model",
        default="gemma2:2b-instruct-q4_K_M",
        help="Base Ollama model tag (default: gemma2:2b-instruct-q4_K_M)",
    )
    parser.add_argument(
        "--adapter-model",
        default=None,
        help="Ollama model tag with adapter loaded (defaults to --agent name)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100,
        help="Max eval pairs per agent (default: 100)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Min F1 to count as correct (default: 0.5)",
    )
    parser.add_argument(
        "--overhead-only",
        action="store_true",
        help="Only measure loading overhead, skip QA evaluation",
    )

    args = parser.parse_args()
    agents = AGENTS if args.all else [args.agent]

    all_results = []
    for agent_id in agents:
        print(f"\n[{agent_id.upper()}]")

        if args.overhead_only:
            overhead = check_loading_overhead(agent_id, args.data_dir)
            print(json.dumps(overhead, indent=2))
            all_results.append({"agent_id": agent_id, "loading": overhead})
            continue

        adapter_model = args.adapter_model or agent_id
        r = benchmark_agent(
            agent_id,
            args.data_dir,
            args.backend,
            args.base_model,
            adapter_model,
            args.max_pairs,
            args.threshold,
        )
        all_results.append(r)

    # Summary.
    if not args.overhead_only and len(all_results) > 1:
        print("\n── Summary ──────────────────────────────────")
        for r in all_results:
            if "accuracy" in r:
                bar = "✓" if r.get("meets_bar") else "✗"
                print(f"  {r['agent_id']:10s}  acc={r['accuracy']:.1%}  f1={r['mean_f1']:.3f}  [{bar}]")


if __name__ == "__main__":
    main()
