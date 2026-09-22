import copy
import json
from pathlib import Path
import tempfile
import unittest

from lilith_coordination import Ledger, request_cancellation
from lilith_repository_coordination import (BoundedPythonFixtureExecutor,
    RepositoryCoordinator, repository_task_tool, request_correction)


GOOD = 'def add_one(value):\n    return value + 1\n'


class Api:
    def __init__(self):
        self.parent = {'id':'parent','akh':'lilith','workspace_profile':'local-dev',
            'turns':[{'id':'author','status':'awaiting_input'}]}
        self.pending = [{'id':'task','session_id':'parent','turn_id':'author','generation':1,
            'kind':'client_tool','prompt':{'name':'request_repository_task',
                'arguments':{'objective':'Implement add_one correctly'}}}]
        self.sessions = {}; self.posts = {}; self.next_session = 0
        self.patch_output = json.dumps({'path':'solution.py','content':GOOD})
        self.patch_status = 'completed'; self.review_status = 'completed'; self.report_status = 'completed'
        self.cancel_immediately = True; self.cancelled = []

    def __call__(self, path, body=None):
        if body is None:
            if path == '/v1/sessions/parent': return copy.deepcopy(self.parent)
            if path.endswith('/pending'): return copy.deepcopy(self.pending)
            return copy.deepcopy(self.sessions[path.rsplit('/',1)[1]])
        if path.endswith('/cancel'):
            session, turn_id = path.split('/')[3], path.split('/')[5]
            state = self.parent if session == 'parent' else self.sessions[session]
            turn = next(t for t in state['turns'] if t['id'] == turn_id)
            turn['cancel_requested'] = True; self.cancelled.append((session,turn_id))
            if self.cancel_immediately: turn['status'] = 'cancelled'
            return copy.deepcopy(turn)
        key = (path, body.get('request_id'))
        if key in self.posts: return copy.deepcopy(self.posts[key])
        if path.endswith('/input-decisions'):
            self.pending = []; self.parent['turns'][0]['status'] = 'completed'; result={'accepted':True}
        elif path == '/v1/sessions':
            self.next_session += 1; identity = body['akh']; result={'id':f's{self.next_session}'}
            self.sessions[result['id']]={'id':result['id'],'akh':identity,'turns':[]}
        elif path.endswith('/inputs'):
            session = path.split('/')[3]
            if session == 'parent':
                result={'id':'report-'+str(len(self.parent['turns'])),'status':self.report_status,
                    'output':'Disposable-copy Python tests passed; reviewer found no defect. Original tree unchanged.'}
                self.parent['turns'].append(result)
            else:
                identity = self.sessions[session]['akh']
                if identity.startswith('khnum'):
                    result={'id':'patch-'+session,'status':self.patch_status,'output':self.patch_output}
                elif identity == 'lilith-report-only':
                    result={'id':'report-'+session,'status':self.report_status,
                        'output':'Disposable-copy Python tests passed; reviewer found no defect. Original tree unchanged.'}
                else:
                    result={'id':'review-'+session,'status':self.review_status,'output':'No defect found.'}
                self.sessions[session]['turns'].append(result)
        else: raise AssertionError(path)
        self.posts[key]=copy.deepcopy(result); return copy.deepcopy(result)


class RepositoryCoordinationTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.repo=self.root/'repo'; self.repo.mkdir()
        (self.repo/'solution.py').write_text('def add_one(value):\n    return value\n')
        (self.repo/'test_solution.py').write_text(
            'import unittest\nfrom solution import add_one\n\n'
            'class T(unittest.TestCase):\n    def test_values(self):\n'
            '        self.assertEqual(add_one(0), 1)\n        self.assertEqual(add_one(-2), -1)\n')
        self.ledger=Ledger(self.root/'ledger.sqlite'); self.addCleanup(self.ledger.close)
        self.api=Api(); self.coordinator=RepositoryCoordinator(self.api,self.ledger,'parent','author',
            self.repo,'solution.py',['python3','-m','unittest','-q'],
            executor=BoundedPythonFixtureExecutor(), report_token_counter=lambda text:len(text.split()))

    def test_two_specialists_real_patch_tests_and_truthful_return(self):
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'completed')
        self.assertTrue(row['verification']['passed'])
        self.assertEqual(row['verification']['kind'],'bounded_python_fixture_process')
        self.assertEqual((self.repo/'solution.py').read_text(),'def add_one(value):\n    return value\n')
        self.assertEqual([s['akh'] for s in self.api.sessions.values()],
                         ['khnum-inline-patcher','ptah-inline-reviewer','lilith-report-only'])
        returned=json.loads(row['return_request']['text'].split('\n',1)[1])
        self.assertTrue(returned['verification']['passed'])
        self.assertFalse(returned['original_repository_modified'])
        self.assertEqual(returned['reviewer_opinion'],'No defect found.')
        self.assertLessEqual(row['report_prompt_tokens'],768)

    def test_repository_tool_is_an_explicit_client_destination(self):
        self.assertEqual(repository_task_tool()['destination'],{'kind':'client'})

    def test_backend_boundaries_run_after_each_producer_is_terminal(self):
        seen=[]
        def transition(role,row):
            if role=='reviewer':
                status=next(t for t in self.api.sessions[row['patch_session']]['turns']
                            if t['id']==row['patch_turn'])['status']
            else:
                status=next(t for t in self.api.sessions[row['review_session']]['turns']
                            if t['id']==row['review_turn'])['status']
            seen.append((role,status));return {'role':role,'status':status}
        self.coordinator.backend_transition=transition
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'completed')
        self.assertEqual(seen,[('reviewer','completed'),('reporter','completed')])
        self.assertEqual(row['backend_boundaries'],
                         [{'role':'reviewer','status':'completed'},{'role':'reporter','status':'completed'}])

    def test_failing_patch_never_reaches_reviewer_or_completion(self):
        self.api.patch_output=json.dumps({'path':'solution.py','content':'def add_one(value):\n return value\n'})
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'failed'); self.assertFalse(row['verification']['passed'])
        self.assertEqual(len(self.api.sessions),1); self.assertEqual(len(self.api.parent['turns']),1)

    def test_failed_peer_is_reported_without_false_success(self):
        self.api.patch_status='failed'
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'failed'); self.assertNotIn('verification',row)
        self.assertEqual(len(self.api.sessions),1)

    def test_operator_cancellation_stops_only_owned_inflight_specialist(self):
        self.api.patch_status='running'; row=self.coordinator.tick()[0]
        self.api.parent['turns'].append({'id':'unrelated','status':'running'})
        request_cancellation(self.ledger.db.execute('PRAGMA database_list').fetchone()[2],
                             'parent','author',row['id'],'Stop repository task')
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'cancelled')
        self.assertEqual(self.api.cancelled,[(row['patch_session'],row['patch_turn'])])
        self.assertEqual(self.api.parent['turns'][-1]['status'],'running')

    def test_correction_cancels_inflight_patch_and_changes_redispatch(self):
        self.api.patch_status='running'; row=self.coordinator.tick()[0]
        correction=request_correction(self.ledger,row['id'],'Also preserve negative integer behavior.')
        self.api.patch_status='completed'; row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'completed'); self.assertEqual(row['correction'],correction)
        self.assertEqual(len([s for s in self.api.sessions.values() if s['akh'].startswith('khnum')]),2)
        self.assertEqual(len(self.api.cancelled),1)
        self.assertIn('preserve negative', row['patch_request']['text'])

    def test_bad_path_and_empty_review_are_terminal_failures(self):
        self.api.patch_output=json.dumps({'path':'test_solution.py','content':'pass\n'})
        row=self.coordinator.tick()[0]; self.assertEqual(row['status'],'failed')

    def test_single_json_fence_is_recorded_but_extra_text_or_fences_are_rejected(self):
        self.api.patch_output='```json\n'+json.dumps({'path':'solution.py','content':GOOD})+'\n```'
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'completed')
        self.assertEqual(row['verification']['presentation'],'single_json_fence')
        self.assertEqual(row['verification']['proposal_output_sha256'],
                         __import__('hashlib').sha256(self.api.patch_output.encode()).hexdigest())
        for output in ('prefix '+self.api.patch_output, self.api.patch_output+' trailing',
                       '```json\n'+self.api.patch_output+'\n```'):
            with self.subTest(output=output[:20]):
                api=Api(); api.patch_output=output
                ledger=Ledger(self.root/('reject-'+str(len(list(self.root.glob('reject-*'))))+'.sqlite'))
                self.addCleanup(ledger.close)
                coordinator=RepositoryCoordinator(api,ledger,'parent','author',self.repo,'solution.py',
                    ['python3','-m','unittest','-q'],executor=BoundedPythonFixtureExecutor(),
                    report_token_counter=lambda text:len(text.split()))
                rejected=coordinator.tick()[0]
                self.assertEqual(rejected['status'],'failed')
                self.assertEqual(rejected['verification']['kind'],'proposal_validation')

    def test_unconfigured_executor_refuses_model_code(self):
        self.coordinator.executor=None
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'failed')
        self.assertEqual(row['verification']['kind'],'execution_refused')

    def test_bounded_executor_rejects_calls_and_imports(self):
        self.api.patch_output=json.dumps({'path':'solution.py',
            'content':'import os\ndef add_one(value):\n os.system("id")\n return value + 1\n'})
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'failed')
        self.assertEqual(row['verification']['kind'],'executor_rejection')

    def test_repository_snapshot_change_blocks_verification(self):
        self.api.patch_status='running'; self.coordinator.tick()
        (self.repo/'test_solution.py').write_text('raise AssertionError("changed")\n')
        self.api.sessions['s1']['turns'][0]['status']='completed'
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'failed')
        self.assertEqual(row['verification']['kind'],'snapshot_guard')

    def test_executor_contract_drift_blocks_verification(self):
        self.api.patch_status='running'; self.coordinator.tick()
        self.coordinator.executor.timeout=9
        self.api.sessions['s1']['turns'][0]['status']='completed'
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'failed')
        self.assertEqual(row['verification']['kind'],'executor_contract_guard')

    def test_correction_during_review_cancels_review_and_revises_all_ids(self):
        self.api.review_status='running'; row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'reviewing')
        request_correction(self.ledger,row['id'],'Return an integer for every integer input.')
        self.api.review_status='completed'; row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'completed')
        self.assertIn(('s2', 'review-s2'), self.api.cancelled)
        self.assertTrue(row['patch_request']['request_id'].endswith('-r1'))
        self.assertTrue(row['review_request']['request_id'].endswith('-r1'))
        self.assertTrue(row['return_request']['request_id'].endswith('-r1'))

    def test_correction_during_return_cancels_report_and_clears_stale_approval(self):
        self.api.report_status='running'; row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'reporting')
        old_return=row['return_turn']; old_return_session=row['return_session']
        request_correction(self.ledger,row['id'],'Use the corrected requirement.')
        self.api.report_status='completed'; row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'completed')
        self.assertIn((old_return_session,old_return),self.api.cancelled)
        self.assertNotEqual(row['return_turn'],old_return)

    def test_symlinked_repository_entry_is_rejected_before_dispatch(self):
        (self.repo/'linked.py').symlink_to('/etc/passwd')
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'rejected')
        self.assertIn('symlink',row['error'])
        self.assertFalse(self.api.sessions)

    def test_executor_timeout_becomes_durable_failure(self):
        self.coordinator=RepositoryCoordinator(self.api,self.ledger,'parent','author',self.repo,
            'solution.py',['python3','-c','import time; time.sleep(2)'],
            executor=BoundedPythonFixtureExecutor(timeout=0.05),
            report_token_counter=lambda text:len(text.split()))
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'failed')
        self.assertTrue(row['verification']['timed_out'])
        self.assertIsNone(row['verification']['exit_code'])

    def test_executor_output_is_bounded_and_records_truncation(self):
        self.coordinator=RepositoryCoordinator(self.api,self.ledger,'parent','author',self.repo,
            'solution.py',['python3','-c','print("x" * 10000)'],
            executor=BoundedPythonFixtureExecutor(), report_token_counter=lambda text:len(text.split()))
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'completed')
        self.assertTrue(row['verification']['output_truncated'])
        self.assertLessEqual(len(row['verification']['stdout']),4000)

    def test_report_capsule_fails_closed_without_truncating_long_review(self):
        self.api.review_status='running'; row=self.coordinator.tick()[0]
        review_session=row['review_session']
        review=next(t for t in self.api.sessions[review_session]['turns'] if t['id']==row['review_turn'])
        review['status']='completed'; review['output']='word ' * 800
        row=self.coordinator.tick()[0]
        self.assertEqual(row['status'],'failed')
        self.assertIn('limit is 768',row['error'])
        self.assertNotIn('return_session',row)


if __name__ == '__main__': unittest.main()
