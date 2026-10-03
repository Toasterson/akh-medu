"""A fake teacher responder: reads the ids a generator's prompt asks to echo and answers with a valid reply."""
from __future__ import annotations

import json
import re

FENCE = "```"


def _pull(prompt: str, key: str) -> str | None:
    match = re.search(r'"%s": "([^"]*)"' % re.escape(key), prompt)
    return match.group(1) if match else None


def block(obj: dict) -> str:
    return f"Here you go.\n{FENCE}json\n{json.dumps(obj, ensure_ascii=False)}\n{FENCE}\n"


def responder(prompt: str) -> str:
    """Dispatches on what the prompt asks for."""
    n = abs(hash(prompt)) % 10_000
    if "sparring round" in prompt:
        count = int(re.search(r"exactly (\d+) turns", prompt).group(1))
        dialogue = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"line {i} alpha{n} beta{i}"} for i in range(count)]
        return block({"adversary": _pull(prompt, "adversary"), "kata_under_test": json.loads(re.search(r'"kata_under_test": (\[[^\]]*\])', prompt).group(1)),
                      "voice": _pull(prompt, "voice"), "language": _pull(prompt, "language"), "dialogue": dialogue,
                      "why_this_move": f"why {n}", "observed_effect_on_person": f"effect {n}", "shadow_avoided": f"shadow {n}"})
    if "transmitting a technique" in prompt:
        voice, kata = _pull(prompt, "voice"), _pull(prompt, "kata")
        move = re.search(r"move_chosen: (.*)\nRaw material", prompt, re.S).group(1)
        return block({"voice": voice, "kata": kata, "language": _pull(prompt, "language"), "situation": "restated situation",
                      "move_chosen": move, "why_this_move": f"The {voice} voice used {kata} here.", "observed_effect_on_person": "it landed"})
    if "cos-guidance@1" in prompt and "work item as the tracker returned it" in prompt:
        return block({"contract": "lilith/cos-guidance@1", "language": _pull(prompt, "language"), "audience": _pull(prompt, "audience"),
                      "subsystem": "smf", "severity": "concern", "finding": f"finding {n} manifest dependency", "recommendation": f"do this {n} instead",
                      "citations": [{"kind": "man-page", "reference": "svcadm(8)"}], "voice": "truthkeeper"})
    # variation
    variant = _pull(prompt, "variant")
    turns_n = int(re.search(r"(\d+) turns\):", prompt).group(1))
    seed_move = re.search(r"move_chosen: (.*)\nwhy_this_move", prompt, re.S).group(1)
    anchor = f"anchor{n}"
    turns = []
    for i in range(turns_n):
        text = f"gamma{n} delta{i} " + (anchor if i == 0 else "")
        if variant == "budget-pressure" and i == turns_n - 1:
            text = "ok"
        turns.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
    header = {"kata": _pull(prompt, "kata"), "voice": _pull(prompt, "voice"), "language": _pull(prompt, "language"),
              "variant": variant, "turns": turns}
    if variant == "multi-turn":
        header["memory_anchor"] = anchor
    assert seed_move
    return block(header)
