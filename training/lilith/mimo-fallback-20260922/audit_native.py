#!/usr/bin/env python3
"""Audit saved MiMo native receipts without executing model or guest code."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts/session-platform"))
import lilith_rl_tasks_v2 as tasks


def sha(value):
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def main():
    native = HERE / "native-illumos"
    result = json.loads((native / "result.json").read_text())
    records_path = HERE / "full8-thinking8192/records.json"
    records = {row["task_id"]: row for row in json.loads(records_path.read_text())}
    assert result["source_records_sha256"] == sha(records_path.read_bytes())
    assert result["executor_contract_sha256"] == sha(
        json.dumps(result["executor_contract"], sort_keys=True, separators=(",", ":"))
    )
    assert result["executor_contract_sha256"] == (
        "ce73f2c4da203e537a4c3899453bccec0093b127ae47b989ee2d9b5881217b59"
    )

    checked = []
    for answer in result["answers"]:
        raw = records[answer["task_id"]]["content"]
        artifact, presentation = tasks.extract_artifact(raw)
        assert answer["raw_output_sha256"] == sha(raw)
        assert answer["artifact_sha256"] == sha(artifact)
        assert answer["presentation"] == presentation
        if not answer["native_executed"]:
            assert not answer["grammar_admitted"]
            continue
        receipt = json.loads((native / answer["receipt"]).read_text())
        assert receipt["task_id"] == answer["task_id"]
        assert receipt["copy_in"]["content_sha256"] == sha(artifact)
        assert receipt["artifact_sha256"] == sha(artifact)
        assert receipt["command"] == ["python3", "-m", "unittest", "-q"]
        assert receipt["target"] == "solution.py"
        assert receipt["timed_out"] is False
        assert receipt["remote_cleanup_succeeded"] is True
        assert receipt["passed"] == answer["native_passed"]
        assert receipt["exit_code"] == answer["exit_code"]
        checked.append(answer["task_id"])

    assert (result["grammar_admitted"], result["native_passed"], result["native_failed"], result["not_executed"]) == (7, 6, 1, 1)
    audit = {
        "schema": "lilith/mimo-native-receipt-audit@1",
        "passed": True,
        "result_sha256": sha((native / "result.json").read_bytes()),
        "receipts_checked": checked,
        "native_passed": 6,
        "native_failed": 1,
        "grammar_rejected_not_executed": 1,
        "score_unchanged": "6/8 frozen interpreter",
    }
    (native / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
