import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location('review_smoke', pathlib.Path(__file__).with_name('lilith-review-smoke.py'))
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class ClampVerifierTest(unittest.TestCase):
    def test_correct_boundary_behavior(self):
        result = smoke.verify_clamp('def clamp_u8(value: int) -> int:\n'
            '    if value < 0:\n        return 0\n'
            '    if value > 255:\n        return 255\n    return value\n')
        self.assertTrue(result['passed'])
        self.assertEqual(result['cases'], 2051)

    def test_plausible_wrong_patch_fails(self):
        with self.assertRaises(AssertionError):
            smoke.verify_clamp('def clamp_u8(value):\n    return value\n')

    def test_imports_calls_loops_and_decorators_are_not_executed(self):
        for source in [
            'import os\ndef clamp_u8(value):\n    return value\n',
            'def clamp_u8(value):\n    return eval(value)\n',
            'def clamp_u8(value):\n    while True:\n        pass\n',
            '@int\ndef clamp_u8(value):\n    return value\n',
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                smoke.verify_clamp(source)


if __name__ == '__main__':
    unittest.main()
