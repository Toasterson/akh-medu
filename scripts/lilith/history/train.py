#!/usr/bin/env python3
"""Bounded historical-receipt SFT pilot; no resource creation or activation.

Requires an explicit future operator approval bound to this exact data manifest,
plus a verified main-only payload. Imports torch only after both guards pass.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "runpod"))
sys.path.insert(0, str(HERE.parents[2] / "session-platform"))
import lilith_lora_trainer as trainer
import lilith_numerics
import verify_payload


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_id(value):
    return hashlib.sha256(value.encode()).hexdigest()


def check_inputs(data, approval):
    manifest = json.loads((data / "manifest.json").read_text())
    tokens = json.loads((data / "tokenization.json").read_text())
    consent = json.loads(approval.read_text())
    if (
        consent.get("decision") != "go"
        or consent.get("run_id") != "lilith-history-r1-20261001"
    ):
        raise ValueError("explicit operator go for this run is absent")
    if consent.get("data_manifest_sha256") != sha(
        data / "manifest.json"
    ) or consent.get("tokenization_sha256") != sha(data / "tokenization.json"):
        raise ValueError("operator go does not bind the exact prepared data")
    if not consent.get("operator") or not consent.get("approved_at"):
        raise ValueError("operator approval attribution/time missing")
    datetime.fromisoformat(consent["approved_at"].replace("Z", "+00:00"))
    if (
        manifest.get("schema") != "lilith/historical-receipt-corpus@1"
        or tokens.get("schema") != "lilith/historical-tokenization@1"
    ):
        raise ValueError("unrecognized data schema")
    from lilith_adapter_mechanics import PARENT_REVISION

    if tokens.get("parent_revision") != PARENT_REVISION:
        raise ValueError("tokenization parent revision mismatch")
    for name, pin in {**manifest["files_sha256"], **tokens["files"]}.items():
        if Path(name).name != name or sha(data / name) != pin:
            raise ValueError(f"data file {name} hash mismatch")
    if tokens["context_limit"] != 8192:
        raise ValueError("pilot context must be the frozen 8K recipe")
    rows = {}
    for split in ("train", "dev", "historical_eval"):
        rows[split] = [
            json.loads(line)
            for line in (data / f"{split}.tokens.jsonl").read_text().splitlines()
        ]
        if not rows[split]:
            raise ValueError(f"{split} is empty")
        for row in rows[split]:
            ids, mask = row["input_ids"], row["target_mask"]
            if (
                not ids
                or len(ids) > 8192
                or len(ids) != len(mask)
                or any(type(i) is not int or not 0 <= i < 262144 for i in ids)
                or any(type(m) is not bool for m in mask)
                or mask[0]
                or not any(mask)
                or mask != sorted(mask)
                or row["weight"] != 1.0
            ):
                raise ValueError(
                    "invalid tokens, context, observation mask or example weight"
                )
    # Recheck entity/session independence from source data on the pod too.
    seen_sessions, seen_items = set(), set()
    for split in ("train", "dev", "historical_eval"):
        lessons = [
            json.loads(line)
            for line in (data / f"{split}.jsonl").read_text().splitlines()
        ]
        if len({r["record_id"] for r in lessons}) != len(lessons) or len(
            {r["record_id"] for r in rows[split]}
        ) != len(rows[split]):
            raise ValueError("duplicate lesson or token record identity")
        if not {r["record_id"] for r in rows[split]} <= {
            r["record_id"] for r in lessons
        }:
            raise ValueError("tokenized record has no admitted lesson")
        sessions = {r["session"] for r in lessons}
        items = {r["work_item_id"] for r in lessons}
        if seen_sessions & sessions or seen_items & items:
            raise ValueError("session/work-item leakage between splits")
        seen_sessions |= sessions
        seen_items |= items
    return rows, manifest, tokens


def grade(answer, expected):
    """JSON validity and exact independently recorded fields; extras cannot hide claims."""

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(answer.strip(), object_pairs_hook=unique)
    except ValueError:
        return {
            "valid_json": False,
            "correct": False,
            "wrong_fields": [],
            "extra_fields": [],
        }
    if not isinstance(value, dict):
        return {
            "valid_json": False,
            "correct": False,
            "wrong_fields": [],
            "extra_fields": [],
        }
    wrong = [
        key
        for key, val in expected.items()
        if key not in value or type(value[key]) is not type(val) or value[key] != val
    ]
    extra = sorted(set(value) - set(expected))
    return {
        "valid_json": True,
        "correct": not wrong and not extra,
        "wrong_fields": wrong,
        "extra_fields": extra,
    }


def evaluate(model, tokenizer, data, tokens, split, deadline):
    import torch

    expected = {
        r["record_id"]: r["expected"]
        for r in (
            json.loads(line)
            for line in (data / f"{split}.jsonl").read_text().splitlines()
        )
    }
    results = []
    model.eval()
    for row in tokens:
        if time.monotonic() >= deadline:
            return {"status": "budget_exhausted", "results": results}
        boundary = row["target_mask"].index(True)
        prompt = torch.tensor([row["input_ids"][:boundary]], device="cuda:0")
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt,
                max_new_tokens=256,
                do_sample=False,
                max_time=max(1.0, deadline - time.monotonic()),
                use_cache=True,
                eos_token_id=tokenizer.convert_tokens_to_ids("<turn|>"),
            )
        answer = tokenizer.decode(
            generated[0, prompt.shape[1] :], skip_special_tokens=True
        )
        results.append(
            {
                "record_id": row["record_id"],
                "answer": answer,
                **grade(answer, expected[row["record_id"]]),
            }
        )
    return {
        "status": "completed",
        "examples": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": sum(r["correct"] for r in results) / len(results),
        "results": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--approval", type=Path, required=True)
    p.add_argument("--payload-root", type=Path, required=True)
    p.add_argument("--main-sha", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-wall-seconds", type=int, default=5400)
    args = p.parse_args()
    if (
        not 1 <= args.epochs <= 3
        or not 1 <= args.max_steps <= 200
        or not 60 <= args.max_wall_seconds <= 5400
    ):
        p.error(
            "pilot limits: 1–3 epochs, 1–200 steps, 60–5400 seconds; pod lifetime <=2h"
        )
    rows, manifest, tokenization = check_inputs(args.data, args.approval)
    provenance = verify_payload.verify(
        args.payload_root,
        args.payload_root / "source-manifest.json",
        args.main_sha,
        False,
    )
    if provenance["status"] != "verified":
        raise ValueError(
            "main-only source payload verification failed: "
            + "; ".join(provenance["problems"])
        )
    if json.loads(args.approval.read_text()).get("main_sha") != args.main_sha:
        raise ValueError("operator go does not bind the exact main source commit")
    if set(tokenization["tokenizer_pins"]) != {
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }:
        raise ValueError("incomplete tokenizer/template pin set")
    for name, pin in tokenization["tokenizer_pins"].items():
        if sha(args.parent / name) != pin:
            raise ValueError(f"on-pod tokenizer {name} differs from prepared data")
    from importlib.metadata import version

    for name, prepared_version in tokenization["packages"].items():
        if version(name) != prepared_version:
            raise ValueError(
                f"on-pod {name} differs from the prepared tokenizer environment"
            )
    args.out.mkdir(parents=True, exist_ok=False)
    args.out.chmod(0o700)
    start = time.monotonic()
    deadline = start + args.max_wall_seconds
    report = {
        "schema": "lilith/historical-training-result@1",
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "quality_claim": False,
        "release_selected": False,
        "adapter_activated": False,
        "data_manifest_sha256": sha(args.data / "manifest.json"),
        "tokenization_sha256": sha(args.data / "tokenization.json"),
        "operator_approval_sha256": sha(args.approval),
        "source": provenance,
        "config": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "seed": 73,
            "max_wall_seconds": args.max_wall_seconds,
            "context": 8192,
            "eval_max_per_split": 12,
        },
        "steps": [],
    }

    def persist():
        report["elapsed_seconds"] = time.monotonic() - start
        tmp = args.out / "result.tmp"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.chmod(0o600)
        tmp.replace(args.out / "result.json")

    persist()
    try:
        import torch

        if not torch.cuda.is_available() or torch.version.hip:
            raise ValueError("requires qualified NVIDIA CUDA path")
        if "H100" not in torch.cuda.get_device_name(0):
            raise ValueError("frozen first-run recipe requires an H100")
        torch.manual_seed(73)
        random.seed(73)
        config = trainer.TrainerConfig(
            rank=16, base_precision="bf16", learning_rate=1e-4, seed=73
        )
        numerics = lilith_numerics.cuda_default()
        report["numerics"] = numerics.requested()
        report["trainer_config"] = config.to_record()
        model, report["load"] = trainer.load_training_model(
            args.parent, config, device="cuda:0"
        )
        tokenizer = trainer.load_tokenizer(args.parent)
        evaluation_rows = {
            split: sorted(rows[split], key=lambda r: sha_id(r["record_id"]))[:12]
            for split in ("dev", "historical_eval")
        }
        report["evaluation_ids"] = {
            k: [r["record_id"] for r in v] for k, v in evaluation_rows.items()
        }
        report["baseline"] = {}
        for split in ("dev", "historical_eval"):
            with model.disable_adapter(), numerics.applied():
                report["baseline"][split] = evaluate(
                    model, tokenizer, args.data, evaluation_rows[split], split, deadline
                )
            persist()
            if report["baseline"][split]["status"] != "completed":
                raise TimeoutError(
                    "budget expired before baseline finished; no training performed"
                )
        optimizer = trainer.make_optimizer(model, config)
        step = 0
        for epoch in range(args.epochs):
            batch = rows["train"][:]
            random.Random(73 + epoch).shuffle(batch)
            for row in batch:
                if step >= args.max_steps or time.monotonic() >= deadline - 900:
                    break
                example = trainer.SftExample(
                    tuple(row["input_ids"]),
                    tuple(row["target_mask"]),
                    row["weight"],
                    row["record_id"],
                )
                receipt = trainer.sft_step(
                    model, optimizer, [example], config, step, numerics=numerics
                )
                if (
                    not math.isfinite(receipt.loss)
                    or not math.isfinite(receipt.grad_norm)
                    or (step == 0 and receipt.grad_norm <= 0)
                ):
                    raise ValueError(
                        "training control failed: nonfinite loss/gradient or zero first gradient"
                    )
                report["steps"].append(asdict(receipt))
                step += 1
                persist()
            if step >= args.max_steps or time.monotonic() >= deadline - 900:
                break
        if not step:
            raise TimeoutError("no optimizer steps fit the budget")
        report["adapter"] = trainer.save_adapter(
            model, args.out / "candidate-adapter", config
        ).to_record()
        probe_ids = rows["dev"][0]["input_ids"][:16]
        from reference_microrun import probe_logits, logits_record

        with numerics.applied():
            probe = logits_record(probe_logits(model, probe_ids))
        (args.out / "cold-reload.probe.json").write_text(
            json.dumps(
                {
                    "adapter_dir": str(args.out / "candidate-adapter"),
                    "adapter": report["adapter"],
                    "probe": {"input_ids": probe_ids, **probe},
                    "trainer_config": config.to_record(),
                },
                indent=2,
            )
            + "\n"
        )
        report["candidate"] = {}
        for split in ("dev", "historical_eval"):
            with numerics.applied():
                report["candidate"][split] = evaluate(
                    model, tokenizer, args.data, evaluation_rows[split], split, deadline
                )
            persist()
        report["status"] = (
            "completed"
            if all(r["status"] == "completed" for r in report["candidate"].values())
            else "budget_exhausted"
        )
        report["cold_reload"] = "pending_fresh_process_check"
        persist()
        return 0 if report["status"] == "completed" else 2
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        persist()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
