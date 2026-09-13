#!/usr/bin/env python3
"""Deterministically mix engine-generated sibling sidecars into one supplemental JSONL.

Phase 51 P6.4 (ADR 106/107). Input: per-family `train.sft.jsonl` sidecars written by
`akh_episode_generator`; output: one `KnowledgeSftRecord` JSONL plus a manifest that
records the seed, the per-family draw, the share against a base record count, and the
blake3 of every input and of the output, so the mixture is reproducible and auditable
by `train_knowledge --remediate-from-bundle`.

    mix_siblings.py --base-records 45842 --max-share-ppm 50000 --seed 20260913 \
        --take memory_write=1000 --take reactor_manage=800 \
        --take structured_task_create=300 --take structured_task_update=300 \
        --corpus training/episodes/p6-3b-20260913-indist --out runs/p6-4/window-1/siblings

Records are drawn without replacement, in a seeded shuffle, and the whole draw is
rejected (exit 2) if it would exceed the share ceiling — the trainer enforces the same
ceiling, but failing here is cheaper than failing after the tokenizer pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


def blake3_file(path: Path) -> str:
    """blake3 via the `b3sum` CLI (cargo install b3sum); falls back to sha256 with a label."""
    try:
        out = subprocess.run(["b3sum", "--no-names", str(path)], check=True, capture_output=True, text=True)
        return "blake3:" + out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True, help="generator out-dir holding <family>/train.sft.jsonl")
    ap.add_argument("--take", action="append", default=[], metavar="FAMILY=N", help="records to draw per family")
    ap.add_argument("--base-records", type=int, required=True, help="record count of the base --train-jsonl")
    ap.add_argument("--max-share-ppm", type=int, default=50_000, help="supplemental / total ceiling in ppm (default 50000 = 5 percent)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory (train.sft.jsonl + manifest.json)")
    args = ap.parse_args()

    draws: dict[str, int] = {}
    for spec in args.take:
        family, _, count = spec.partition("=")
        if not family or not count.isdigit():
            ap.error(f"--take expects FAMILY=N, got {spec!r}")
        draws[family] = int(count)
    if not draws:
        ap.error("at least one --take FAMILY=N is required")

    total_siblings = sum(draws.values())
    share_ppm = total_siblings * 1_000_000 // (args.base_records + total_siblings)
    if share_ppm > args.max_share_ppm:
        print(
            f"refusing: {total_siblings} siblings over {args.base_records} base records is "
            f"{share_ppm} ppm > ceiling {args.max_share_ppm} ppm",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "train.sft.jsonl"
    inputs = []
    seen_ids: set[str] = set()
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for family in sorted(draws):
            src = args.corpus / family / "train.sft.jsonl"
            if not src.is_file():
                print(f"missing sidecar for {family}: {src}", file=sys.stderr)
                return 1
            lines = src.read_text(encoding="utf-8").splitlines()
            if draws[family] > len(lines):
                print(f"{family}: asked {draws[family]} but only {len(lines)} available", file=sys.stderr)
                return 1
            order = list(range(len(lines)))
            rng.shuffle(order)
            chosen = sorted(order[: draws[family]])
            for index in chosen:
                record = json.loads(lines[index])
                rid = record.get("id")
                if rid in seen_ids:
                    print(f"duplicate record id across families: {rid}", file=sys.stderr)
                    return 1
                seen_ids.add(rid)
                out.write(lines[index].rstrip("\n") + "\n")
                written += 1
            inputs.append(
                {
                    "family": family,
                    "file": str(src),
                    "file_hash": blake3_file(src),
                    "available": len(lines),
                    "drawn": draws[family],
                    "sidecar_manifest": str(src.parent / "manifest.json") if (src.parent / "manifest.json").is_file() else None,
                }
            )

    manifest = {
        "schema_version": 1,
        "purpose": "P6.4 supplemental sibling mixture (ADR 106/107)",
        "seed": args.seed,
        "base_records": args.base_records,
        "sibling_records": written,
        "share_ppm": share_ppm,
        "max_share_ppm": args.max_share_ppm,
        "inputs": inputs,
        "output": {"file": str(out_path), "file_hash": blake3_file(out_path), "records": written},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("sibling_records", "share_ppm", "seed")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
