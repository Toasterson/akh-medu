import importlib.util
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'native_core_smoke', pathlib.Path(__file__).with_name('native-core-smoke.py'))
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


class PublicationTests(unittest.TestCase):
    def test_binary_digest_works_on_host_python(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'artifact'
            path.write_bytes(b'abc')
            self.assertEqual(canary.file_sha256(path),
                             'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')

    def test_captured_turn_has_a_public_receipt(self):
        events = [{'kind': 'cognitive_turn_accepted', 'turn_id': 'turn-1',
                   'data': {'receipt': 'turn-turn-1', 'data': {'captured': True}}}]
        self.assertEqual(canary.accepted_capture(events, 'turn-1'), 'turn-turn-1')

    def test_other_turn_and_completion_alone_do_not_prove_publication(self):
        self.assertIsNone(canary.accepted_capture([
            {'kind': 'completed', 'turn_id': 'turn-1'},
            {'kind': 'cognitive_turn_accepted', 'turn_id': 'another'}], 'turn-1'))

    def test_disabled_capture_and_wrong_receipt_fail_closed(self):
        for data in [{'receipt': 'turn-turn-1', 'data': {'captured': False}},
                     {'receipt': 'wrong', 'data': {'captured': True}}]:
            with self.subTest(data=data), self.assertRaises(AssertionError):
                canary.accepted_capture([
                    {'kind': 'cognitive_turn_accepted', 'turn_id': 'turn-1', 'data': data}], 'turn-1')


if __name__ == '__main__':
    unittest.main()
