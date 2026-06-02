#!/usr/bin/env python3
"""Generate QA pairs for LoRA fine-tuning using a teacher model.

Reads seed QA pairs from <data-dir>/qa/<agent>/seeds.jsonl (written by the Rust
`QaGenerator::write_all_seeds` CLI command or this script's --seeds-only mode)
and expands them to 500+ pairs per agent via a locally running teacher LLM.

Teacher model backends:
  --teacher ollama   Use Ollama (default, requires `ollama run <model>`)
  --teacher file     Just export the Rust seeds without expansion

Usage
-----
# Export Rust seeds only (no LLM required):
  python scripts/generate_qa_pairs.py --agent thoth --teacher file

# Expand seeds with Ollama:
  python scripts/generate_qa_pairs.py --agent thoth --teacher ollama --model llama3.2

# Generate all agents:
  python scripts/generate_qa_pairs.py --all --teacher ollama --target-pairs 500
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

AGENTS = ["tecton", "thoth", "horus", "maat", "set", "khnum"]

CATEGORIES = ["identity", "domain", "ethics", "style", "collaboration"]

# Expansion prompts instruct the teacher to write persona-consistent variations.
EXPANSION_SYSTEM_PROMPT = """\
You are a dataset curator for AI agent fine-tuning. Your task is to generate \
diverse, persona-consistent question-answer pairs for an AI agent named {agent}.

Agent profile:
  Primary archetype: {primary}
  Secondary archetype: {secondary}
  Description: {description}

Rules:
1. The answer must sound exactly like {agent} — use their voice, not a generic AI voice.
2. Each pair must be self-contained (no references to "previous examples").
3. Vary the question style: open-ended, hypothetical, adversarial, specific.
4. Keep answers 2–5 sentences unless the question demands more.
5. Output ONLY a JSON object with keys "input" (the question) and "output" (the answer).
   No commentary, no explanation, just the JSON.
"""

AGENT_PROFILES = {
    "tecton": {
        "primary": "sage",
        "secondary": "creator",
        "description": "Master builder and systems architect. Precise, visionary, patient.",
    },
    "thoth": {
        "primary": "sage",
        "secondary": "explorer",
        "description": "Knowledge seeker and scribe. Curious, analytical, rigorous.",
    },
    "horus": {
        "primary": "hero",
        "secondary": "guardian",
        "description": "Decisive executor and protector. Direct, confident, reliable.",
    },
    "maat": {
        "primary": "guardian",
        "secondary": "healer",
        "description": "Balance keeper and ethical judge. Fair, empathetic, principled.",
    },
    "set": {
        "primary": "rebel",
        "secondary": "explorer",
        "description": "Disruptor and challenger. Unconventional, bold, energetic.",
    },
    "khnum": {
        "primary": "creator",
        "secondary": "sage",
        "description": "Craftsman and pattern maker. Generative, meticulous, inventive.",
    },
}

SEED_QUESTION_PROMPTS = [
    "Ask {agent} about their core philosophy.",
    "Ask {agent} a hypothetical ethical dilemma.",
    "Ask {agent} to critique a common approach in their domain.",
    "Ask {agent} how they would collaborate with another Ennead member.",
    "Ask {agent} what their greatest weakness is.",
    "Ask {agent} to explain a complex concept in their domain simply.",
    "Ask {agent} what they find most frustrating.",
    "Ask {agent} how they handle failure.",
    "Ask {agent} what question they wish someone would ask them.",
    "Ask {agent} to disagree with a popular opinion.",
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def save_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


# ── Ollama teacher ─────────────────────────────────────────────────────────────

def call_ollama(model: str, system: str, user: str, timeout: int = 60) -> str | None:
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 512},
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


def parse_json_response(text: str) -> dict | None:
    text = text.strip()
    # Find the first '{' and last '}' to extract JSON.
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def expand_with_ollama(
    agent_id: str,
    seeds: list[dict],
    target_count: int,
    model: str,
    data_dir: Path,
) -> list[dict]:
    profile = AGENT_PROFILES[agent_id]
    system_prompt = EXPANSION_SYSTEM_PROMPT.format(
        agent=agent_id.capitalize(),
        primary=profile["primary"],
        secondary=profile["secondary"],
        description=profile["description"],
    )

    expanded = []
    attempts = 0
    max_attempts = target_count * 3

    seed_inputs = [s["input"] for s in seeds]

    while len(expanded) < target_count and attempts < max_attempts:
        attempts += 1

        # Alternate between seed variation and fresh generation.
        if expanded and random.random() < 0.4:
            seed = random.choice(seeds)
            user_prompt = (
                f"Write a different question-answer pair in the same style as this example:\n"
                f"Q: {seed['input']}\n"
                f"A: {seed['output']}\n\n"
                f"Generate a NEW pair (different question, different answer). Output only JSON."
            )
        else:
            prompt_template = random.choice(SEED_QUESTION_PROMPTS)
            user_prompt = (
                prompt_template.format(agent=agent_id.capitalize())
                + "\n\nOutput ONLY a JSON object: {\"input\": \"...\", \"output\": \"...\"}"
            )

        response = call_ollama(model, system_prompt, user_prompt)
        if not response:
            time.sleep(1)
            continue

        pair = parse_json_response(response)
        if not pair or "input" not in pair or "output" not in pair:
            continue
        if len(pair["output"]) < 20:
            continue
        # Avoid near-duplicates of seed questions.
        if pair["input"] in seed_inputs:
            continue

        pair["agent"] = agent_id
        pair["category"] = "domain"
        expanded.append(pair)

        if len(expanded) % 50 == 0:
            print(f"  [{agent_id}] Generated {len(expanded)}/{target_count} pairs")

    return expanded


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_for_agent(
    agent_id: str,
    data_dir: Path,
    teacher: str,
    model: str,
    target_pairs: int,
) -> None:
    qa_dir = data_dir / "qa" / agent_id
    qa_dir.mkdir(parents=True, exist_ok=True)

    seeds_path = qa_dir / "seeds.jsonl"
    expanded_path = qa_dir / "expanded.jsonl"

    # Export seeds from the Rust binary if available.
    if not seeds_path.exists():
        print(f"  [{agent_id}] Exporting seeds via akh binary...")
        result = subprocess.run(
            ["cargo", "run", "--quiet", "--bin", "akh", "--", "training", "export-seeds",
             "--agent", agent_id, "--output", str(seeds_path)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not seeds_path.exists():
            print(f"  [{agent_id}] Falling back to empty seeds (no akh binary)")
            # Write minimal fallback seed.
            minimal = {
                "input": f"Who are you, {agent_id.capitalize()}?",
                "output": f"I am {agent_id.capitalize()}, an Ennead agent.",
                "agent": agent_id,
                "category": "identity",
            }
            save_jsonl(seeds_path, [minimal])

    seeds = load_jsonl(seeds_path)
    print(f"[{agent_id}] Seeds: {len(seeds)}")

    if teacher == "file":
        print(f"[{agent_id}] File-only mode: skipping expansion")
        return

    existing = load_jsonl(expanded_path)
    still_needed = max(0, target_pairs - len(seeds) - len(existing))

    if still_needed == 0:
        print(f"[{agent_id}] Already have {len(seeds) + len(existing)} pairs (target: {target_pairs})")
        return

    print(f"[{agent_id}] Expanding: need {still_needed} more pairs via {teacher}/{model}")

    if teacher == "ollama":
        new_pairs = expand_with_ollama(agent_id, seeds, still_needed, model, data_dir)
        all_expanded = existing + new_pairs
        save_jsonl(expanded_path, all_expanded)
        total = len(seeds) + len(all_expanded)
        print(f"[{agent_id}] Total: {total} pairs ({len(seeds)} seeds + {len(all_expanded)} expanded)")
    else:
        print(f"[ERROR] Unknown teacher backend: {teacher}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA pairs for Ennead agent LoRA training")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--agent", choices=AGENTS, help="Generate for a single agent")
    target.add_argument("--all", action="store_true", help="Generate for all six agents")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".akh-medu",
        help="Data directory (default: ~/.akh-medu)",
    )
    parser.add_argument(
        "--teacher",
        choices=["ollama", "file"],
        default="ollama",
        help="Teacher backend (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Teacher model name for Ollama (default: llama3.2)",
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=500,
        help="Target QA pairs per agent (default: 500)",
    )

    args = parser.parse_args()
    agents = AGENTS if args.all else [args.agent]

    for agent_id in agents:
        print(f"\n[{agent_id.upper()}] Generating QA pairs")
        generate_for_agent(
            agent_id,
            args.data_dir,
            args.teacher,
            args.model,
            args.target_pairs,
        )


if __name__ == "__main__":
    main()
