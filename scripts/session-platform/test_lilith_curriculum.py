import json
from pathlib import Path
import tempfile
import unittest
from lilith_curriculum import build, compile_reference, exercises, verify


class CurriculumTests(unittest.TestCase):
    def test_every_reference_passes_and_every_misconception_has_a_witness(self):
        for item in exercises():
            with self.subTest(item=item.id):
                self.assertTrue(verify(item, item.source)['passed'])
                negative = verify(item, item.mutant)
                self.assertFalse(negative['passed'])
                self.assertIsNotNone(negative['first_counterexample'])

    def test_public_holdout_families_never_enter_training_or_corrections(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'export'
            manifest = build(out)
            heldout = set(manifest['development_holdout_families'])
            self.assertTrue(heldout.isdisjoint(manifest['train_families']))
            for name in ('candidates.jsonl', 'corrections.jsonl'):
                for line in (out/name).read_text().splitlines():
                    row = json.loads(line)
                    self.assertNotIn(row['family'], heldout)
                    self.assertIsNone(row['provenance']['teacher_model'])
                    self.assertFalse(row['provenance']['native_illumos_validated'])
            for line in (out/'development-holdouts.jsonl').read_text().splitlines():
                self.assertNotIn('output', json.loads(line))

    def test_replay_is_deterministic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/'first', Path(tmp)/'second'
            self.assertEqual(build(first), build(second))
            for file in first.iterdir():
                self.assertEqual(file.read_bytes(), (second/file.name).read_bytes())
            with self.assertRaises(FileExistsError):
                build(first)

    def test_bool_contract_does_not_accept_integer_return(self):
        item = next(e for e in exercises() if e.id == 'is-power-two/1')
        self.assertFalse(verify(item, 'def f(n):\n    return 0\n')['passed'])

    def test_side_effect_syntax_is_rejected(self):
        for source in ('import os', 'def f(x):\n    return open(x)',
                       'def f(x):\n    return x.real', '@bad\ndef f(x):\n    return x'):
            with self.subTest(source=source), self.assertRaises(ValueError):
                compile_reference(source)


if __name__ == '__main__':
    unittest.main()
