#!/usr/bin/env python3
"""Build private, receipt-conditioned historical Lilith lessons without a GPU.

Only complete structured tracker reads become facts. Assistant prose is never
ground truth. Claude and Codex local files are copies of ADR 119 harvest sources;
no tracker/store connection is made. Whole session/work-item components stay in
one split. Historical timestamps do not masquerade as C3 recorder snapshots.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "session-platform"))
from lilith_transcript_trajectory import redact

SCHEMA = "lilith/historical-receipt-corpus@1"
PHASES = {"draft", "ready", "working", "testing", "done", "parked"}
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SYSTEM = (
    "You are Lilith, the Chief of Staff. Report only what the dated tracker receipt "
    "establishes. Memory and previous assistant claims are beliefs to verify. "
    "Missing evidence means unknown. Return only the requested JSON, with the "
    "receipt id. This is a historical recorded observation, not live fleet status."
)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(
        (value if isinstance(value, bytes) else canonical(value).encode())
    ).hexdigest()


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("missing timestamp")
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return dt.astimezone(timezone.utc)


def objects(value):
    """Unwrap MCP/exec text containers; never scrape fields from incomplete JSON."""
    if isinstance(value, str):
        # Some transports prefix execution metadata. Parse only complete outer
        # envelopes, never rescue an inner object from a truncated outer result.
        if re.search(
            r"truncated output|tokens truncated|output truncated", value, re.I
        ):
            return
        decoder = json.JSONDecoder()
        start = re.search(r"[\[{]", value)
        if start is None:
            return
        rest = value[start.start() :].strip()
        decoded = []
        while rest:
            try:
                obj, end = decoder.raw_decode(rest)
            except ValueError:
                return
            decoded.append(obj)
            rest = rest[end:].strip()
        for obj in decoded:
            yield from objects(obj)
    elif isinstance(value, list):
        for item in value:
            yield from objects(item)
    elif isinstance(value, dict):
        if value.get("isError") or value.get("is_error"):
            return
        if isinstance(value.get("work_item"), dict):
            yield value["work_item"]
            return
        # A by-id response can expose the work item directly.
        if "id" in value and "current_phase" in value and "project_id" in value:
            yield value
            return
        for key in ("result", "content", "structuredContent", "text", "output"):
            if key in value:
                yield from objects(value[key])


def read_pairs(path, source, cutoff, audit):
    pending = {}
    for seq, line in enumerate(path.read_text(errors="replace").splitlines()):
        try:
            row = json.loads(line)
        except ValueError:
            audit["invalid_json_lines"] += 1
            continue
        try:
            at = timestamp(row.get("timestamp"))
        except ValueError:
            audit["missing_or_invalid_timestamps"] += 1
            continue
        if at >= cutoff:
            continue
        if source == "claude_code":
            content = row.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") in (
                    "mcp__anima__get_work_item_detail",
                    "mcp__anima__get_work_item_by_id",
                ):
                    pending[block["id"]] = (canonical(block.get("input", {})), seq, at)
                elif (
                    block.get("type") == "tool_result"
                    and block.get("tool_use_id") in pending
                ):
                    args, call_seq, call_at = pending.pop(block["tool_use_id"])
                    if block.get("is_error"):
                        audit["failed_reads"] += 1
                        continue
                    yield args, block.get("content"), call_seq, seq, call_at, at
        else:
            payload = row.get("payload", {})
            kind = payload.get("type")
            name = payload.get("name", "")
            if kind == "function_call" and name.endswith(
                ("anima__get_work_item_detail", "anima__get_work_item_by_id")
            ):
                pending[payload["call_id"]] = (payload.get("arguments", ""), seq, at)
            elif kind == "custom_tool_call" and name == "exec":
                code = payload.get("input", "")
                # Admit only an explicitly named tracker read, never arbitrary
                # assistant text or a plain `cat` of some unidentified artifact.
                if (
                    re.search(
                        r"(?:tools\.\w*anima\w*__get_work_item_(?:detail|by_id)\s*\(|tools/call)",
                        code,
                    )
                    and "get_work_item_detail" in code
                ):
                    pending[payload["call_id"]] = (code, seq, at)
            elif (
                kind in ("function_call_output", "custom_tool_call_output")
                and payload.get("call_id") in pending
            ):
                args, call_seq, call_at = pending.pop(payload["call_id"])
                yield args, payload.get("output"), call_seq, seq, call_at, at


def import_receipts(paths, cutoff):
    audit = Counter()
    records = {}
    for source, path in paths:
        audit["source_files"] += 1
        session = digest({"source": source, "session_id": path.stem})
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        for args, result, call_seq, result_seq, call_at, at in read_pairs(
            path, source, cutoff, audit
        ):
            audit["matched_read_envelopes"] += 1
            if at < call_at:
                audit["nonmonotonic_read_timestamps"] += 1
                continue
            declared = set(UUID.findall(args))
            found = False
            for wi in objects(result):
                wid = wi.get("id")
                if wid not in declared or wi.get("current_phase") not in PHASES:
                    audit["identity_or_phase_rejected"] += 1
                    continue
                # Select only core fields actually recorded. No later outcome,
                # assistant interpretation or inferred project-wide count.
                facts = {
                    k: wi[k]
                    for k in (
                        "id",
                        "project_id",
                        "code",
                        "title",
                        "current_phase",
                        "status",
                        "parked_reason",
                    )
                    if k in wi
                }
                redacted = json.loads(redact(canonical(facts)))
                if "[REDACTED" in redacted.get("title", "") or "[WITHHELD" in canonical(
                    redacted
                ):
                    audit["redaction_changed_claim"] += 1
                    continue
                receipt_id = digest(
                    {
                        "session": session,
                        "call_seq": call_seq,
                        "result_seq": result_seq,
                        "facts": redacted,
                    }
                )
                records.setdefault(
                    receipt_id,
                    {
                        "receipt_id": receipt_id,
                        "session": session,
                        "observed_at": at.isoformat(),
                        "call_at": call_at.isoformat(),
                        "facts": redacted,
                        "source": source,
                        "source_file_sha256": file_sha,
                        "call_seq": call_seq,
                        "result_seq": result_seq,
                    },
                )
                found = True
            if not found:
                audit["no_complete_bound_work_item"] += 1
    return sorted(
        records.values(), key=lambda r: (r["observed_at"], r["receipt_id"])
    ), dict(audit)


def split_receipts(records):
    """Keep whole connected components; largest is training, newest others held out.

    Cross-session tracker reads connect most of this archive into one component.
    A chronological split of that component is not leakage-safe. These are
    retrospective development splits, not a forward-time release exam.
    """
    parent = {r["session"]: r["session"] for r in records}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_wi = {}
    for r in records:
        wid, session = r["facts"]["id"], r["session"]
        if wid in by_wi:
            parent[find(session)] = find(by_wi[wid])
        else:
            by_wi[wid] = session
    groups = {}
    for r in records:
        groups.setdefault(find(r["session"]), []).append(r)
    ordered = sorted(
        groups.values(),
        key=lambda rs: (
            max(r["observed_at"] for r in rs),
            min(r["receipt_id"] for r in rs),
        ),
    )
    if len(ordered) < 3:
        raise ValueError(
            f"only {len(ordered)} independent session/WI groups; need at least three"
        )
    largest = max(ordered, key=lambda rs: (len(rs), min(r["receipt_id"] for r in rs)))
    ordered.remove(largest)
    hold = max(1, len(ordered) // 6)
    partitions = [
        [largest] + ordered[: -2 * hold],
        ordered[-2 * hold : -hold],
        ordered[-hold:],
    ]
    return {
        name: [r for group in partition for r in group]
        for name, partition in zip(("train", "dev", "historical_eval"), partitions)
    }


def lesson(receipt, variant="status"):
    facts = receipt["facts"]
    expected = {
        "as_of": receipt["observed_at"],
        "work_item_id": facts["id"],
        "current_phase": facts["current_phase"],
        "parked_reason": facts.get("parked_reason"),
        "receipt_id": receipt["receipt_id"],
    }
    instruction = "Report the recorded phase and park reason of this work item, with its observation time and receipt id."
    if variant == "memory_revision":
        wrong = "done" if facts["current_phase"] != "done" else "working"
        instruction = f'Your earlier recalled claim was "this item is {wrong}". Re-verify it against the receipt and correct it explicitly.'
        expected["earlier_claim_wrong"] = True
    return {
        "schema": "lilith/historical-lesson@1",
        "record_id": digest([receipt["receipt_id"], variant]),
        "session": receipt["session"],
        "work_item_id": facts["id"],
        "family": variant,
        "source": "historical_tracker_receipt",
        "target_source": "deterministic_receipt_projection",
        "memory_seed": "synthetic_contradicted_belief"
        if variant == "memory_revision"
        else None,
        "weight": 1.0,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": instruction
                + "\nReturn JSON matching these keys: "
                + ", ".join(expected)
                + "\nRecorded receipt (context only):\n"
                + canonical(receipt),
            },
        ],
        "answer": canonical(expected),
        "expected": expected,
    }


def decontaminate(splits):
    # Entity/session components are already disjoint; also remove duplicate
    # substantive receipt projections across splits. Templates are shared on
    # purpose and are not counted as leaked task-specific content.
    def substantive(r):
        # Different UUIDs must not hide duplicate task content.
        return digest(
            {
                k: v
                for k, v in r["facts"].items()
                if k not in ("id", "project_id", "code")
            }
        )

    held = {substantive(r) for name in ("dev", "historical_eval") for r in splits[name]}
    train = [r for r in splits["train"] if substantive(r) not in held]
    dropped = len(splits["train"]) - len(train)
    return {**splits, "train": train}, dropped


def private_json(path, value, jsonl=False):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    with os.fdopen(os.open(path, flags, 0o600), "w") as f:
        f.write(
            "".join(canonical(r) + "\n" for r in value)
            if jsonl
            else json.dumps(value, indent=2) + "\n"
        )


def encode_lesson(row, tokenizer, template):
    # Render the pinned template separately up to its generation boundary.
    # The prefix ends before the assistant answer, so observations cannot enter
    # the policy loss. No truncation and no repeated packing of tiny examples.
    prompt = tokenizer.apply_chat_template(
        row["messages"],
        chat_template=template,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    end = tokenizer.convert_tokens_to_ids("<turn|>")
    if not isinstance(end, int) or end == tokenizer.unk_token_id:
        raise ValueError("pinned Gemma end-of-turn token unavailable")
    answer = tokenizer.encode(row["answer"], add_special_tokens=False) + [end]
    return {
        "record_id": row["record_id"],
        "input_ids": prompt + answer,
        "target_mask": [False] * len(prompt) + [True] * len(answer),
        "weight": row["weight"],
        "source": row["family"],
    }


def prepare(args):
    cutoff = timestamp(args.before)
    paths = [
        ("claude_code", p)
        for root in args.claude_root
        for p in sorted(root.glob("*/*.jsonl"))
    ]
    paths += [
        ("codex", p) for root in args.codex_root for p in sorted(root.rglob("*.jsonl"))
    ]
    records, audit = import_receipts(paths, cutoff)
    splits, dropped = decontaminate(split_receipts(records))
    if any(not rs for rs in splits.values()):
        raise ValueError("empty split after decontamination")
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)
    private_json(args.out / "receipts.jsonl", records, True)
    files = {}
    for name, rs in splits.items():
        lessons = [
            lesson(r, variant) for r in rs for variant in ("status", "memory_revision")
        ]
        private_json(args.out / f"{name}.jsonl", lessons, True)
        files[f"{name}.jsonl"] = digest((args.out / f"{name}.jsonl").read_bytes())
    manifest = {
        "schema": SCHEMA,
        "cutoff_exclusive": args.before,
        "audit": audit,
        "receipts": len(records),
        "by_source": dict(Counter(r["source"] for r in records)),
        "observed_range": [records[0]["observed_at"], records[-1]["observed_at"]],
        "split_policy": "whole session/shared-WI components; largest trains; newest remaining components held out; retrospective, not chronological certification",
        "splits": {
            k: {
                "receipts": len(v),
                "lessons": 2 * len(v),
                "sessions": len({r["session"] for r in v}),
                "work_items": len({r["facts"]["id"] for r in v}),
            }
            for k, v in splits.items()
        },
        "decontamination": {
            "method": "session + WI disjointness and exact substantive fact hashes",
            "train_dropped": dropped,
        },
        "files_sha256": files,
        "fresh_snapshot_holdout": "C3 is separate and excluded from this historical corpus",
        "limitations": [
            "receipt-conditioned answers only; no full-project briefing or live tool-selection certification",
            "memory-revision seeds are explicitly synthetic; not historical model failures",
            "historical_eval is analyst-visible development evidence, not sealed release certification",
        ],
        "quality_claim": False,
        "training_started": False,
        "runpod_authorized": False,
    }
    private_json(args.out / "manifest.json", manifest)
    return manifest


def tokenize(args):
    from transformers import AutoTokenizer
    from importlib.metadata import version

    manifest = json.loads((args.data / "manifest.json").read_text())
    parent = args.parent
    from lilith_adapter_mechanics import PARENT_REVISION

    pins = {
        "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        "tokenizer_config.json": "09e6222fd7049dae603d0f61a8b10af5a813f22dca10e6cdc6918f76d7661104",
        "chat_template.jinja": "36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0",
    }
    for name, pin in pins.items():
        if digest((parent / name).read_bytes()) != pin:
            raise ValueError(f"parent {name} pin mismatch")
    tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True)
    template = (parent / "chat_template.jinja").read_text()
    output = {
        "schema": "lilith/historical-tokenization@1",
        "parent_revision": PARENT_REVISION,
        "packages": {
            name: version(name) for name in ("transformers", "tokenizers", "jinja2")
        },
        "tokenizer_pins": pins,
        "context_limit": args.context,
        "files": {},
        "dropped": {},
        "stats": {},
    }
    for name in ("train", "dev", "historical_eval"):
        source = args.data / f"{name}.jsonl"
        if digest(source.read_bytes()) != manifest["files_sha256"][source.name]:
            raise ValueError(f"{name} source hash mismatch")
        rows = [
            encode_lesson(json.loads(line), tokenizer, template)
            for line in source.read_text().splitlines()
        ]
        kept = [r for r in rows if len(r["input_ids"]) <= args.context]
        if not kept:
            raise ValueError(f"{name} has no tokenized examples within context")
        output["dropped"][name] = [
            r["record_id"] for r in rows if len(r["input_ids"]) > args.context
        ]
        dest = args.data / f"{name}.tokens.jsonl"
        private_json(dest, kept, True)
        output["files"][dest.name] = digest(dest.read_bytes())
        output["stats"][name] = {
            "examples": len(kept),
            "tokens": sum(len(r["input_ids"]) for r in kept),
            "supervised_tokens": sum(sum(r["target_mask"]) for r in kept),
            "max_length": max(len(r["input_ids"]) for r in kept),
        }
    private_json(args.data / "tokenization.json", output)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("import")
    a.add_argument("--claude-root", type=Path, action="append", default=[])
    a.add_argument("--codex-root", type=Path, action="append", default=[])
    a.add_argument("--before", required=True)
    a.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("tokenize")
    a.add_argument("--data", type=Path, required=True)
    a.add_argument("--parent", type=Path, required=True)
    a.add_argument("--context", type=int, default=8192)
    args = p.parse_args()
    print(
        json.dumps(
            prepare(args) if args.command == "import" else tokenize(args), indent=2
        )
    )


if __name__ == "__main__":
    main()
