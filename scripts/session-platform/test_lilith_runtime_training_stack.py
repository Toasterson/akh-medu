import copy
import json
import stat
import tempfile
import unittest
from pathlib import Path

from lilith_runtime_training_stack import capture_session_events, configure_training_session_timeout


class RuntimeTrainingStackTests(unittest.TestCase):
    def test_dedicated_session_outlives_training_deadline_and_preserves_source(self):
        shared = {
            'session_stall_threshold_ms': 15 * 60_000,
            'retirement_cancel_grace_ms': 2 * 60_000,
        }
        dedicated = copy.deepcopy(shared)

        effective = configure_training_session_timeout(dedicated, 72 * 60 * 60)

        self.assertEqual(effective, (72 * 60 * 60 * 1000) + (2 * 60_000))
        self.assertEqual(dedicated['session_stall_threshold_ms'], effective)
        self.assertEqual(shared['session_stall_threshold_ms'], 15 * 60_000)

    def test_training_timeout_must_be_positive(self):
        with self.assertRaises(ValueError):
            configure_training_session_timeout({}, 0)

    def test_session_events_are_appended_from_the_last_sequence(self):
        calls = []
        pages = [[{'sequence': 1, 'kind': 'turn_started'}],
                 [{'sequence': 2, 'kind': 'turn_completed'}]]

        def api(service, path):
            calls.append((service, path))
            return pages[len(calls) - 1]

        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / 'events.jsonl'
            events_path.touch(mode=0o600)
            cursor = capture_session_events(api, '/v1/sessions/session-1', events_path, 0)
            cursor = capture_session_events(api, '/v1/sessions/session-1', events_path, cursor)
            events = [json.loads(line) for line in events_path.read_text().splitlines()]

            self.assertEqual(cursor, 2)
            self.assertEqual(events, pages[0] + pages[1])
            self.assertEqual(calls, [
                ('akhsessd', '/v1/sessions/session-1/events?after=0'),
                ('akhsessd', '/v1/sessions/session-1/events?after=1'),
            ])
            self.assertEqual(stat.S_IMODE(events_path.stat().st_mode), 0o600)


if __name__ == '__main__':
    unittest.main()
