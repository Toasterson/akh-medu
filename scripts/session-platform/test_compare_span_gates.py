import copy
from pathlib import Path
import runpy
import unittest

compare = runpy.run_path(str(Path(__file__).with_name("compare-span-gates.py")))["compare"]


def fixture():
    return {"schema": "akh-span-paraphrase-gate/1", "backend": "synthetic",
            "model_path": "model", "tokenizer_path": "tokenizer",
            "policy_sha256": "a" * 64, "projector_sha256": "b" * 64,
            "groups": [["a", "a2"], ["b", "b2"], ["c", "c2"]],
            "vectors": [[[value] * 1250 for _ in range(2)] for value in [0, 85, 255]]}


class ComparisonTests(unittest.TestCase):
    def test_fact_queries_never_use_questions_as_evidence_candidates(self):
        left = fixture()
        left["task"] = "fact_query"
        report = compare(left, copy.deepcopy(left))
        self.assertEqual([r["query"] for r in report["left_queries_right_index"]], [1, 3, 5])
        self.assertEqual([r["nearest"] for r in report["left_queries_right_index"]], [[0], [2], [4]])

    def test_identity_and_one_bit_change_have_known_distances(self):
        left = fixture()
        report = compare(left, copy.deepcopy(left))
        self.assertTrue(report["bit_identical"])
        self.assertTrue(report["all_expected_neighbors_preserved"])
        right = copy.deepcopy(left)
        right["vectors"][0][0][0] = 1
        report = compare(left, right)
        self.assertEqual(report["same_prompt_bit_flips"], [1, 0, 0, 0, 0, 0])
        self.assertFalse(report["bit_identical"])
        self.assertTrue(report["all_expected_neighbors_preserved"])

    def test_ties_are_not_counted_as_success(self):
        left = fixture()
        left["vectors"] = [[[0] * 1250 for _ in range(2)] for _ in range(3)]
        report = compare(left, copy.deepcopy(left))
        self.assertTrue(report["bit_identical"])
        self.assertFalse(report["all_expected_neighbors_preserved"])

    def test_space_mismatch_and_malformed_vectors_are_rejected(self):
        left = fixture()
        right = copy.deepcopy(left)
        right["projector_sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            compare(left, right)
        for bad in [[], [0] * 1249, [256] * 1250, [True] * 1250]:
            right = copy.deepcopy(left)
            right["vectors"][0][0] = bad
            with self.assertRaises(ValueError):
                compare(left, right)


if __name__ == "__main__":
    unittest.main()
