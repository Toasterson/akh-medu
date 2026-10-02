"""Historical evidence boundaries and launch guards; no fleet or GPU access."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import prepare
import run_pod
import supervise
import train

WID = "019d9844-ce74-7993-a9f0-c953b8f04a4e"
FACT = {
    "id": WID,
    "project_id": WID,
    "current_phase": "working",
    "title": "Recorded task",
}


def receipt(session, wid, n=0):
    return {
        "receipt_id": str(n),
        "session": session,
        "observed_at": "2026-09-01T00:00:00+00:00",
        "facts": {**FACT, "id": wid},
    }


class EvidenceTests(unittest.TestCase):
    def test_different_ids_do_not_hide_duplicate_substantive_facts(self):
        r = receipt("training", "different-id")
        held = receipt("held", "held-id")
        result, dropped = prepare.decontaminate(
            {"train": [r], "dev": [held], "historical_eval": []}
        )
        self.assertEqual(dropped, 1)
        self.assertEqual(result["train"], [])

    def test_complete_outer_envelope_only(self):
        self.assertEqual(
            list(prepare.objects(json.dumps({"result": {"work_item": FACT}}))), [FACT]
        )
        incomplete = '{"result": {"work_item": ' + json.dumps(FACT)
        self.assertEqual(list(prepare.objects(incomplete)), [])
        self.assertEqual(
            list(prepare.objects(json.dumps({"isError": True, "work_item": FACT}))), []
        )
        self.assertEqual(
            list(prepare.objects("Warning: truncated output\n" + json.dumps(FACT))), []
        )

    def test_session_and_item_connected_components_remain_disjoint(self):
        rows = [
            receipt("s1", "w1", 1),
            receipt("s1", "w2", 2),
            receipt("s2", "w2", 3),
            receipt("s3", "w3", 4),
            receipt("s4", "w4", 5),
            receipt("s5", "w5", 6),
        ]
        splits = prepare.split_receipts(rows)
        self.assertEqual(sum(map(len, splits.values())), len(rows))
        seen_s, seen_w = set(), set()
        for values in splits.values():
            sessions = {r["session"] for r in values}
            items = {r["facts"]["id"] for r in values}
            self.assertFalse(sessions & seen_s)
            self.assertFalse(items & seen_w)
            seen_s |= sessions
            seen_w |= items
        self.assertTrue({"s1", "s2"} <= {r["session"] for r in splits["train"]})

    def test_insufficient_independent_groups_refused(self):
        with self.assertRaises(ValueError):
            prepare.split_receipts([receipt("one", "one")])

    def test_codex_result_after_cutoff_is_not_admitted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "session.jsonl"
            rows = [
                {
                    "timestamp": "2026-09-27T23:59:00Z",
                    "payload": {
                        "type": "function_call",
                        "name": "mcp__anima__get_work_item_detail",
                        "call_id": "a",
                        "arguments": json.dumps({"id": WID}),
                    },
                },
                {
                    "timestamp": "2026-09-28T00:00:00Z",
                    "payload": {
                        "type": "function_call_output",
                        "call_id": "a",
                        "output": json.dumps({"work_item": FACT}),
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            result, _ = prepare.import_receipts(
                [("codex", path)], prepare.timestamp("2026-09-28T00:00:00Z")
            )
            self.assertEqual(result, [])

    def test_read_identity_binding_and_claude_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "session.jsonl"
            rows = [
                {
                    "timestamp": "2026-09-01T00:00:00Z",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "mcp__anima__get_work_item_detail",
                                "id": "a",
                                "input": {"id": WID},
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "a",
                                "content": json.dumps({"work_item": FACT}),
                            },
                            {
                                "type": "tool_use",
                                "name": "mcp__anima__get_work_item_detail",
                                "id": "b",
                                "input": {"id": "wrong"},
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "b",
                                "content": json.dumps({"work_item": FACT}),
                            },
                            {
                                "type": "tool_use",
                                "name": "mcp__anima__get_work_item_detail",
                                "id": "c",
                                "input": {"id": WID},
                            },
                            {
                                "type": "tool_result",
                                "tool_use_id": "c",
                                "is_error": True,
                                "content": json.dumps({"work_item": FACT}),
                            },
                        ]
                    },
                }
            ]
            path.write_text("\n".join(json.dumps(r) for r in rows))
            result, audit = prepare.import_receipts(
                [("claude_code", path)], prepare.timestamp("2026-09-28T00:00:00Z")
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(audit["failed_reads"], 1)
            self.assertEqual(audit["identity_or_phase_rejected"], 1)

    def test_memory_counterclaim_is_labeled_synthetic(self):
        row = prepare.lesson(receipt("a", WID), "memory_revision")
        self.assertEqual(row["memory_seed"], "synthetic_contradicted_belief")
        self.assertTrue(row["expected"]["earlier_claim_wrong"])
        self.assertEqual(row["expected"]["current_phase"], "working")

    def test_private_outputs_are_write_once(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "data.json"
            prepare.private_json(p, {"test": True})
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                prepare.private_json(p, {})

    def test_double_writer_cannot_overwrite_admission_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "data.json"

            def write(n):
                try:
                    prepare.private_json(p, {"writer": n})
                    return True
                except FileExistsError:
                    return False

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(write, (1, 2)))
            self.assertEqual(sum(results), 1)
            self.assertIn(json.loads(p.read_text())["writer"], (1, 2))


class GuardTests(unittest.TestCase):
    def test_no_go_refused_without_torch_import(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            for name, body in [
                ("manifest.json", {}),
                ("tokenization.json", {}),
                ("approval.json", {"decision": "hold"}),
            ]:
                (p / name).write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "go"):
                train.check_inputs(p, p / "approval.json")
            self.assertNotIn("torch", sys.modules)

    def test_go_must_bind_data_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            for name, body in [
                ("manifest.json", {}),
                ("tokenization.json", {}),
                ("approval.json", {"decision": "go", "run_id": run_pod.RUN_ID}),
            ]:
                (p / name).write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError, "exact prepared data"):
                train.check_inputs(p, p / "approval.json")

    def test_exact_grading_rejects_missing_extra_or_invalid_answers(self):
        expected = {"current_phase": "working"}
        self.assertTrue(train.grade(json.dumps(expected), expected)["correct"])
        for answer in [
            "invalid",
            "{}",
            "[]",
            '{"current_phase":"done"}',
            '{"current_phase":"done","current_phase":"working"}',
            '{"current_phase":"working","unsupported":"claim"}',
        ]:
            self.assertFalse(train.grade(answer, expected)["correct"])
        self.assertFalse(train.grade('{"corrected":1}', {"corrected": True})["correct"])

    def test_completion_marker_written_after_checksum_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "partial.json").write_text("partial evidence")
            run_pod.seal(out, {"status": "failed"})
            result = json.loads((out / "history-result.json").read_text())
            self.assertEqual(
                result["files_sha256"]["partial.json"], train.sha(out / "partial.json")
            )
            self.assertNotIn("history-result.json", result["files_sha256"])

    def test_stop_uses_own_openbao_wrapper_and_checks_stopped_state(self):
        calls = []

        def fake(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(
                args,
                0,
                json.dumps({"runtimeStatus": "stopped", "desiredStatus": "EXITED"}),
                "",
            )

        result = supervise.stop_pod("testpod", runner=fake)
        self.assertEqual(result["runtimeStatus"], "stopped")
        self.assertEqual(
            calls[0][1:], ["runpod", "--", "runpodctl", "pod", "stop", "testpod"]
        )
        self.assertEqual(
            calls[1][1:], ["runpod", "--", "runpodctl", "pod", "get", "testpod"]
        )

    def test_stop_failure_is_not_reported_as_stopped(self):
        def fake(args, **kwargs):
            return subprocess.CompletedProcess(args, 1, "", "denied")

        with self.assertRaisesRegex(RuntimeError, "stop failed"):
            supervise.stop_pod("testpod", runner=fake)


class TokenizerTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("LILITH_PARENT"),
        "requires local pinned tokenizer; no model weights needed",
    )
    def test_real_template_masks_every_observation_and_supervises_only_answer(self):
        from transformers import AutoTokenizer

        parent = Path(os.environ["LILITH_PARENT"])
        t = AutoTokenizer.from_pretrained(parent, local_files_only=True)
        template = (parent / "chat_template.jinja").read_text()
        row = prepare.lesson(receipt("a", WID), "memory_revision")
        encoded = prepare.encode_lesson(row, t, template)
        boundary = encoded["target_mask"].index(True)
        self.assertFalse(any(encoded["target_mask"][:boundary]))
        self.assertTrue(all(encoded["target_mask"][boundary:]))
        self.assertEqual(t.decode(encoded["input_ids"][boundary:-1]), row["answer"])
        self.assertEqual(encoded["input_ids"][-1], 106)
        prefix = t.apply_chat_template(
            row["messages"],
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # The pinned no-thinking generation prefix includes an empty thought
        # channel. It belongs to the masked prompt, not the assistant target.
        self.assertTrue(prefix.endswith("<|channel>thought\n<channel|>"))
        self.assertEqual(
            t.decode(encoded["input_ids"]), prefix + row["answer"] + "<turn|>"
        )


if __name__ == "__main__":
    unittest.main()
