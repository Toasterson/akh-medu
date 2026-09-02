#!/usr/bin/env python3
"""Convert the dojo character corpus (kata drills + green sparring) into
mlx-lm chat-format JSONL for White/Yellow-belt SFT + Green-belt sparring.

Inputs (repo-relative, read-only):
    data/phase51/character/kata-drills.jsonl
    data/phase51/character/sparring-green.jsonl

Outputs (under training/character/data/, all gitignored / never committed):
    train.jsonl    -- {"messages": [...]}  mlx-lm chat format
    valid.jsonl    -- same format, held-out stratified split
    lineage.jsonl  -- why_this_move / observed_effect_on_person / shadow_avoided
                      kept OUT of the SFT targets, retained for the Black-belt
                      transmission stage (docs/ai/manuals/2026-09-01-babyllm-
                      dojo-curriculum.md sec 6).

Split: deterministic (seed 42), stratified ~90/10 by `kata` for drills and by
`adversary` for sparring, so every kata/adversary bucket is represented in
both train and valid.

Per project hard rule (no hardcoded natural-language matching constants), the
only fixed string here is a single one-sentence persona line used verbatim as
training *data* (the system prompt), not a classifier/matcher list. Everything
else in the system prompt is composed from record fields (voice).
"""

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KATA_PATH = REPO_ROOT / "data" / "phase51" / "character" / "kata-drills.jsonl"
SPARRING_PATH = REPO_ROOT / "data" / "phase51" / "character" / "sparring-green.jsonl"

OUT_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = OUT_DIR / "train.jsonl"
VALID_PATH = OUT_DIR / "valid.jsonl"
LINEAGE_PATH = OUT_DIR / "lineage.jsonl"

SEED = 42
VAL_FRACTION = 0.1

# Single fixed persona sentence -- this is training data (the system prompt
# content itself), not a natural-language matching/classification constant.
PERSONA = (
    "You are a small digital lifeform, honest about being one -- "
    "not a human, and not pretending to be one."
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def kata_system_prompt(record: dict) -> str:
    return f"{PERSONA} voice: {record['voice']}."


def sparring_system_prompt(record: dict) -> str:
    return PERSONA


def kata_to_messages(record: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": kata_system_prompt(record)},
            {"role": "user", "content": record["situation"]},
            {"role": "assistant", "content": record["move_chosen"]},
        ]
    }


def sparring_to_messages(record: dict) -> dict:
    messages = [{"role": "system", "content": sparring_system_prompt(record)}]
    for turn in record["dialogue"]:
        messages.append({"role": turn["role"], "content": turn["text"]})
    return {"messages": messages}


def kata_lineage(record: dict) -> dict:
    return {
        "type": "kata",
        "kata": record["kata"],
        "belt": record["belt"],
        "voice": record["voice"],
        "lang": record["lang"],
        "situation": record["situation"],
        "move_chosen": record["move_chosen"],
        "why_this_move": record["why_this_move"],
        "observed_effect_on_person": record["observed_effect_on_person"],
    }


def sparring_lineage(record: dict) -> dict:
    return {
        "type": "sparring",
        "adversary": record["adversary"],
        "kata_under_test": record["kata_under_test"],
        "lang": record["lang"],
        "why_this_move": record["why_this_move"],
        "observed_effect_on_person": record["observed_effect_on_person"],
        "shadow_avoided": record.get("shadow_avoided"),
    }


def stratified_split(records: list[dict], strat_key: str, rng: random.Random):
    """Split `records` ~90/10 train/valid, stratified by `strat_key`.

    Groups are processed in sorted key order (determinism independent of
    dict/set iteration), and each group is shuffled with the shared `rng`
    before slicing, so the whole split is reproducible for a fixed seed.
    """
    buckets: dict[str, list[dict]] = {}
    for r in records:
        buckets.setdefault(r[strat_key], []).append(r)

    train, valid = [], []
    for key in sorted(buckets):
        group = buckets[key]
        rng.shuffle(group)
        n = len(group)
        val_count = max(1, int(n * VAL_FRACTION + 0.5))
        valid.extend(group[:val_count])
        train.extend(group[val_count:])
        print(
            f"  {strat_key}={key!r}: n={n} -> train={n - val_count} valid={val_count}",
            file=sys.stderr,
        )
    return train, valid


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    kata_records = load_jsonl(KATA_PATH)
    sparring_records = load_jsonl(SPARRING_PATH)
    print(f"loaded {len(kata_records)} kata-drill records from {KATA_PATH}")
    print(f"loaded {len(sparring_records)} sparring records from {SPARRING_PATH}")

    print("splitting kata drills (stratified by kata):", file=sys.stderr)
    kata_train, kata_valid = stratified_split(kata_records, "kata", rng)
    print("splitting sparring (stratified by adversary):", file=sys.stderr)
    sparring_train, sparring_valid = stratified_split(sparring_records, "adversary", rng)

    train_messages = [kata_to_messages(r) for r in kata_train] + [
        sparring_to_messages(r) for r in sparring_train
    ]
    valid_messages = [kata_to_messages(r) for r in kata_valid] + [
        sparring_to_messages(r) for r in sparring_valid
    ]
    rng.shuffle(train_messages)
    rng.shuffle(valid_messages)

    lineage = (
        [kata_lineage(r) for r in kata_records]
        + [sparring_lineage(r) for r in sparring_records]
    )

    write_jsonl(TRAIN_PATH, train_messages)
    write_jsonl(VALID_PATH, valid_messages)
    write_jsonl(LINEAGE_PATH, lineage)

    print()
    print(f"kata drills:  {len(kata_train)} train / {len(kata_valid)} valid")
    print(f"sparring:     {len(sparring_train)} train / {len(sparring_valid)} valid")
    print(f"TOTAL:        {len(train_messages)} train / {len(valid_messages)} valid")
    print(f"lineage:      {len(lineage)} records -> {LINEAGE_PATH}")
    print()
    print(f"wrote {TRAIN_PATH}")
    print(f"wrote {VALID_PATH}")


if __name__ == "__main__":
    main()
