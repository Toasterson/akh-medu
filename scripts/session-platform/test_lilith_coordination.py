import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lilith_coordination import Coordinator, Ledger, inspect_ledger, request_cancellation


def verify(source):
    return {'passed': source == 'verified source', 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'cases': 3, 'scope': 'test fixture'}


class Api:
    def __init__(self, workspace):
        self.parent = dict(id='parent', akh='lilith', workspace_profile='local-dev',
            workspace={'root': str(workspace)}, turns=[{'id':'author', 'status':'awaiting_input'}])
        self.pending = [dict(id='request', session_id='parent', turn_id='author', generation=1,
            kind='client_tool', prompt={'name':'request_review', 'arguments':
                {'path':'clamp_u8.py', 'objective':'Check integer clamp contract'}})]
        self.children, self.posts = {}, {}
        self.fail_path = None
        self.complete_author = True
        self.review_status = 'completed'
        self.report_status = 'completed'
        self.accept_decision = True
        self.cancel_immediately = True
        self.cancel_calls = []

    def __call__(self, path, body=None):
        if body is None:
            if path == '/v1/sessions/parent':
                return copy.deepcopy(self.parent)
            if path.endswith('/pending'):
                return copy.deepcopy(self.pending)
            return copy.deepcopy(self.children[path.rsplit('/',1)[1]])
        if path.endswith('/cancel'):
            _, _, _, session, _, turn_id, _ = path.split('/')
            state = self.parent if session == 'parent' else self.children[session]
            turn = next(t for t in state['turns'] if t['id'] == turn_id)
            self.cancel_calls.append((session, turn_id))
            turn['cancel_requested'] = True
            if self.cancel_immediately:
                turn['status'] = 'cancelled'
            if self.fail_path == path:
                self.fail_path = None
                raise ConnectionError('cancel response lost after remote commit')
            return copy.deepcopy(turn)
        key = (path, body['request_id'])
        if key in self.posts:
            previous, result = self.posts[key]
            assert previous == body, 'idempotency intent changed'
            return copy.deepcopy(result)
        if path.endswith('/input-decisions'):
            self.pending = []
            if self.complete_author:
                self.parent['turns'][0]['status'] = 'completed'
            result = {'accepted':self.accept_decision}
        elif path == '/v1/sessions':
            assert self.parent['turns'][0]['status'] == 'completed', 'nested model dispatch'
            result = {'id':'child'}
            self.children['child'] = {'id':'child', 'turns':[]}
        elif path == '/v1/sessions/child/inputs':
            result = {'id':'review', 'status':self.review_status, 'output':'No contract defect found.'}
            self.children['child']['turns'].append(result)
        elif path == '/v1/sessions/parent/inputs':
            result = {'id':'report', 'status':self.report_status, 'output':'Tests passed; Ptah reviewed this artifact.'}
            self.parent['turns'].append(result)
        else:
            raise AssertionError(path)
        self.posts[key] = (copy.deepcopy(body), copy.deepcopy(result))
        if self.fail_path == path:
            self.fail_path = None
            raise ConnectionError('response lost after remote commit')
        return copy.deepcopy(result)


class CoordinationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root/'workspace'; self.workspace.mkdir()
        (self.workspace/'clamp_u8.py').write_text('verified source')
        self.path = self.root/'ledger.sqlite'
        self.api = Api(self.workspace)
        self.ledger = Ledger(self.path)
        self.addCleanup(lambda: self.ledger.close())
        self.controller = self.make()

    def make(self):
        return Coordinator(self.api, self.ledger, 'parent', 'author', self.root, verify)

    def restart(self):
        self.ledger.close()
        self.ledger = Ledger(self.path)
        self.controller = self.make()

    def test_complete_snapshot_review_and_human_report(self):
        row = self.controller.tick()[0]
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(row['source'], 'verified source')
        self.assertTrue(row['evidence']['passed'])
        self.assertEqual(row['review_turn'], 'review')
        self.assertEqual(row['return_turn'], 'report')
        self.controller.tick()
        self.assertEqual(len(self.api.children), 1)
        self.assertEqual(len(self.api.posts), 4)

    def test_handoff_distinguishes_inline_snapshot_and_independent_receipt(self):
        row = self.controller.tick()[0]
        body, _ = next(value for (path, _), value in self.api.posts.items()
                       if path == '/v1/sessions/child/inputs')
        payload = json.loads(body['text'].split('\n', 1)[1])
        self.assertEqual(payload['artifact']['representation'], 'inline_utf8_snapshot')
        self.assertNotIn('path', payload['artifact'])
        self.assertEqual(payload['artifact']['content'], row['source'])
        self.assertEqual(payload['verification']['receipt'], row['id'] + '-verification')
        self.assertEqual(payload['verification']['kind'], 'host_fixture_verification')

    def test_status_is_readable_while_writer_holds_ledger(self):
        self.api.complete_author = False
        self.controller.tick()
        self.assertEqual(inspect_ledger(self.path, 'parent', 'author')[0]['status'], 'queued')

    def test_disappearing_artifact_is_stale_before_dispatch(self):
        self.api.complete_author = False
        self.controller.tick()
        (self.workspace/'clamp_u8.py').unlink()
        self.api.parent['turns'][0]['status'] = 'completed'
        self.assertEqual(self.controller.tick()[0]['status'], 'stale')
        self.assertFalse(self.api.children)

    def test_response_loss_at_each_remote_commit_is_replayed_without_duplicates(self):
        paths=['/v1/sessions/parent/input-decisions','/v1/sessions',
               '/v1/sessions/child/inputs','/v1/sessions/parent/inputs']
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                if index:
                    self.ledger.close()
                    self.path = self.root/f'ledger{index}.sqlite'
                    self.ledger = Ledger(self.path)
                    self.api = Api(self.workspace)
                    self.controller = self.make()
                self.api.fail_path = path
                with self.assertRaises(ConnectionError):
                    self.controller.tick()
                self.restart()
                row = self.controller.tick()[0]
                self.assertEqual(row['status'], 'completed')
                self.assertEqual(len(self.api.children), 1)
                self.assertEqual(len(self.api.posts), 4)

    def test_recovery_replays_the_persisted_request_without_rerendering(self):
        self.api.fail_path = '/v1/sessions/child/inputs'
        with self.assertRaises(ConnectionError):
            self.controller.tick()
        self.restart()
        with patch('lilith_coordination.review_input', side_effect=AssertionError('renderer changed')):
            self.assertEqual(self.controller.tick()[0]['status'], 'completed')

    def test_no_child_while_author_waits_and_cancelled_author_is_not_dispatched(self):
        self.api.complete_author = False
        self.assertEqual(self.controller.tick()[0]['status'], 'queued')
        self.assertFalse(self.api.children)
        self.api.parent['turns'][0]['status'] = 'cancelled'
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')
        self.assertFalse(self.api.children)

    def test_lost_rejected_acknowledgement_does_not_authorize_dispatch(self):
        self.api.accept_decision = False
        self.api.fail_path = '/v1/sessions/parent/input-decisions'
        with self.assertRaises(ConnectionError):
            self.controller.tick()
        self.restart()
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')
        self.assertFalse(self.api.children)

    def test_changed_artifact_cannot_receive_a_current_completion(self):
        self.api.review_status = 'running'
        self.assertEqual(self.controller.tick()[0]['status'], 'reviewing')
        (self.workspace/'clamp_u8.py').write_text('changed')
        self.api.children['child']['turns'][0]['status'] = 'completed'
        self.assertEqual(self.controller.tick()[0]['status'], 'stale')
        self.assertEqual(len(self.api.parent['turns']), 1)

    def test_invalid_scope_and_failed_verification_never_spawn_child(self):
        for path in ('../escape.py', '/tmp/escape.py', 'other.py'):
            self.api.pending[0]['prompt']['arguments']['path'] = path
            self.api.pending[0]['id'] = path
            self.controller.tick()
            self.assertFalse(self.api.children)
            self.api.pending = copy.deepcopy(Api(self.workspace).pending)
        (self.workspace/'clamp_u8.py').write_text('wrong source')
        self.controller.tick()
        self.assertTrue(all(r['status']=='rejected' for r in self.ledger.rows('parent')))
        self.assertFalse(self.api.children)

    def test_invalid_python_semantics_rejects_without_losing_the_assignment(self):
        from lilith_verification import verify_clamp
        self.controller.verifier = verify_clamp
        (self.workspace/'clamp_u8.py').write_text('def clamp_u8(value):\n    return missing_name\n')
        row = self.controller.tick()[0]
        self.assertEqual(row['status'], 'rejected')
        self.assertTrue(row['acknowledged'])
        self.assertFalse(self.api.children)

    def test_symlink_snapshot_is_rejected(self):
        (self.workspace/'clamp_u8.py').unlink()
        (self.workspace/'clamp_u8.py').symlink_to('/etc/passwd')
        self.assertEqual(self.controller.tick()[0]['status'], 'rejected')
        self.assertFalse(self.api.children)

    def test_other_turns_do_not_gain_delegation_authority(self):
        self.api.pending[0]['turn_id'] = 'unassigned-turn'
        self.assertEqual(self.controller.tick(), [])
        self.assertFalse(self.api.posts)

    def test_failed_or_unknown_review_never_becomes_completed(self):
        self.api.review_status = 'unknown'
        self.assertEqual(self.controller.tick()[0]['status'], 'unknown')
        self.assertEqual(len(self.api.parent['turns']), 1)

    def cancel(self, row):
        return request_cancellation(self.path, 'parent', 'author', row['id'], 'Stop assigned work')

    def test_cancellation_mailbox_works_with_active_writer_and_waits_for_confirmation(self):
        self.api.review_status = 'running'
        self.api.cancel_immediately = False
        row = self.controller.tick()[0]
        self.cancel(row)
        self.cancel(row)  # Same request is idempotent.
        row = self.controller.tick()[0]
        self.assertEqual(row['status'], 'cancelling')
        self.assertEqual(len(self.api.parent['turns']), 1)
        self.restart()
        self.api.children['child']['turns'][0]['status'] = 'cancelled'
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')

    def test_queued_assignment_cancellation_never_dispatches_child(self):
        self.api.complete_author = False
        row = self.controller.tick()[0]
        self.cancel(row)
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')
        self.assertFalse(self.api.children)
        self.assertEqual(self.api.cancel_calls, [('parent', 'author')])

    def test_cancellation_reconciles_lost_submission_responses(self):
        for path in ['/v1/sessions', '/v1/sessions/child/inputs', '/v1/sessions/parent/inputs']:
            with self.subTest(path=path):
                self.api = Api(self.workspace)
                self.path = self.root/('ledger-' + str(len(list(self.root.glob('ledger-*')))) + '.sqlite')
                self.ledger.close()
                self.ledger = Ledger(self.path)
                self.controller = self.make()
                self.api.fail_path = path
                with self.assertRaises(ConnectionError):
                    self.controller.tick()
                row = self.ledger.rows('parent')[0]
                self.cancel(row)
                self.restart()
                self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')
                self.assertEqual(len(self.api.children), 1)
                self.assertLessEqual(len(self.api.parent['turns']), 2)
                self.controller.tick()
                self.assertEqual(len(self.api.children), 1)

    def test_lost_cancellation_response_is_reconciled(self):
        self.api.review_status = 'running'
        row = self.controller.tick()[0]
        self.cancel(row)
        self.api.fail_path = '/v1/sessions/child/turns/review/cancel'
        with self.assertRaises(ConnectionError):
            self.controller.tick()
        self.restart()
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')
        self.assertEqual(len(self.api.parent['turns']), 1)

    def test_stale_artifact_stops_running_child_before_stale_is_terminal(self):
        self.api.review_status = 'running'
        self.api.cancel_immediately = False
        self.controller.tick()
        (self.workspace/'clamp_u8.py').write_text('changed')
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelling')
        self.api.children['child']['turns'][0]['status'] = 'cancelled'
        self.assertEqual(self.controller.tick()[0]['status'], 'stale')

    def test_unknown_child_never_confirms_cancellation(self):
        self.api.review_status = 'running'
        row = self.controller.tick()[0]
        self.cancel(row)
        self.api.cancel_immediately = False
        self.api.children['child']['turns'][0]['status'] = 'unknown'
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelling')

    def test_cancellation_scope_and_completed_outcomes_are_immutable(self):
        self.api.complete_author = False
        row = self.controller.tick()[0]
        with self.assertRaises(ValueError):
            request_cancellation(self.path, 'other', 'author', row['id'], 'wrong scope')
        self.cancel(row)
        with self.assertRaises(ValueError):
            request_cancellation(self.path, 'parent', 'author', row['id'], 'changed reason')
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')
        with self.assertRaises(ValueError):
            request_cancellation(self.path, 'parent', 'author', 'unknown', 'stop')

    def test_completion_cannot_overwrite_a_committed_cancellation(self):
        self.api.review_status = 'running'
        row = self.controller.tick()[0]
        self.cancel(row)
        row.update(status='completed', report='completion raced with operator')
        self.ledger.save(row)
        self.assertEqual(self.ledger.get(row['id'])['status'], 'cancelling')
        view = inspect_ledger(self.path, 'parent', 'author')[0]
        self.assertEqual(view['cancellation_requested']['reason'], 'Stop assigned work')

    def test_completed_assignment_is_not_retroactively_cancelled(self):
        row = self.controller.tick()[0]
        with self.assertRaisesRegex(ValueError, 'terminal'):
            self.cancel(row)
        self.assertEqual(self.ledger.get(row['id'])['status'], 'completed')

    def test_unknown_assignment_can_still_request_cancellation(self):
        self.api.review_status = 'unknown'
        self.api.cancel_immediately = False
        row = self.controller.tick()[0]
        self.assertEqual(row['status'], 'unknown')
        self.cancel(row)
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelling')

    def test_retired_parent_stops_child_and_preserves_unrelated_turns(self):
        self.api.review_status = 'running'
        self.controller.tick()
        self.api.parent['turns'].append(dict(id='unrelated',status='running'))
        self.api.parent['retirement'] = {'reason':'operator retired session'}
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')
        self.assertEqual(self.api.parent['turns'][-1]['status'], 'running')
        self.assertEqual(self.api.cancel_calls, [('child','review')])

    def test_running_report_is_cancelled_without_touching_other_parent_turns(self):
        self.api.report_status = 'running'
        row = self.controller.tick()[0]
        self.assertEqual(row['status'], 'reporting')
        self.api.parent['turns'].append(dict(id='unrelated', status='running'))
        self.cancel(row)
        row = self.controller.tick()[0]
        self.assertEqual(row['status'], 'cancelled')
        self.assertNotIn('report', row)
        self.assertEqual(self.api.parent['turns'][-1]['status'], 'running')
        self.assertEqual(self.api.cancel_calls, [('parent','report')])

    def test_cancellation_arriving_during_review_post_fences_return(self):
        original = self.api
        def api(path, body=None):
            result = original(path, body)
            if path == '/v1/sessions/child/inputs':
                self.cancel(self.ledger.rows('parent')[0])
            return result
        self.controller.api = api
        self.assertEqual(self.controller.tick()[0]['status'], 'cancelled')
        self.assertEqual(len(self.api.parent['turns']), 1)

    def test_second_writer_cannot_open_ledger(self):
        with self.assertRaises(BlockingIOError):
            Ledger(self.path)


if __name__ == '__main__':
    unittest.main()
