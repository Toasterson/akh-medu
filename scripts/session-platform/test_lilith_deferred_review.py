import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lilith_deferred_review import DeferredReviewQueue, reconcile_with_reviewer


class DeferredReviewTests(unittest.TestCase):
    def fixture(self, root):
        workspace = root / 'workspace'; workspace.mkdir()
        (workspace / 'clamp_u8.py').write_text('def clamp_u8(x): return x\n')
        session = {'id':'session', 'workspace':{'id':'workspace','root':str(workspace)},
            'turns':[{'id':'turn','runtime_job_id':'job'}]}
        pending = {'id':'call','session_id':'session','turn_id':'turn','generation':1,
            'kind':'client_tool','prompt':{'name':'request_review','arguments':{
                'path':'clamp_u8.py','objective':'Review the bounded artifact.'}}}
        return workspace, session, pending

    def test_snapshot_precedes_ack_and_terminal_review_binds_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); workspace,session,pending=self.fixture(root)
            queue=DeferredReviewQueue(root/'queue',['clamp_u8.py'],'catalog')
            row=queue.admit(session,pending)
            self.assertEqual(Path(row['artifact_path']).read_bytes(),(workspace/'clamp_u8.py').read_bytes())
            self.assertFalse(row['review_complete']);self.assertEqual(row['reward'],0.0)
            self.assertFalse(queue.response(row)['data']['review_complete'])
            queue.record_decision(row['queue_id'],{'accepted':True})
            with self.assertRaises(RuntimeError): queue.require_terminal()
            final=queue.reconcile(row['queue_id'],{'status':'fulfilled','verifier_sha256':'v'*64,
                'artifact_sha256':row['artifact_sha256'],'reviewer_model_sha256':'m'*64,
                'review_text':'No defect found.'})
            self.assertTrue(final['review_complete']);self.assertEqual(final['reward'],0.0)
            self.assertEqual(queue.require_terminal()[0]['status'],'fulfilled')

    def test_symlink_and_changed_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);workspace,session,pending=self.fixture(root)
            queue=DeferredReviewQueue(root/'queue',['clamp_u8.py'],'catalog')
            row=queue.admit(session,pending)
            (workspace/'clamp_u8.py').write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'reused'):
                queue.admit(session,pending)
            other=root/'other';other.write_text('x')
            (workspace/'clamp_u8.py').unlink();(workspace/'clamp_u8.py').symlink_to(other)
            pending['id']='call2'
            with self.assertRaisesRegex(ValueError,'non-symlink'):
                queue.admit(session,pending)

    def test_post_training_reviewer_fulfills_original_queue_without_trainer_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,session,pending=self.fixture(root)
            queue=DeferredReviewQueue(root/'queue',['clamp_u8.py'],'catalog')
            row=queue.admit(session,pending);queue.record_decision(row['queue_id'],{'accepted':True})
            calls=[]
            def api(path, body=None):
                calls.append((path,body))
                if path=='/v1/sessions': return {'id':'review-session'}
                if path.endswith('/inputs'): return {'id':'review-turn'}
                return {'turns':[{'id':'review-turn','status':'completed','output':'No defect found.'}]}
            rows=reconcile_with_reviewer(api,queue,'ptah','m'*64,
                lambda queued,artifact:{'passed':True,'verifier_sha256':'v'*64,'cases':7})
            self.assertEqual(rows[0]['status'],'fulfilled');self.assertEqual(rows[0]['reward'],0.0)
            self.assertEqual([path for path,_ in calls],['/v1/sessions','/v1/sessions/review-session/inputs',
                '/v1/sessions/review-session'])
            self.assertNotIn('/v1/sessions/session/inputs',[path for path,_ in calls])

    def test_verifier_failure_is_durable_terminal_failure_without_reviewer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);_,session,pending=self.fixture(root)
            queue=DeferredReviewQueue(root/'queue',['clamp_u8.py'],'catalog')
            row=queue.admit(session,pending,{'verifier_sha256':'v'*64})
            def broken_verifier(queued, artifact):
                raise RuntimeError('guest unavailable')
            rows=reconcile_with_reviewer(lambda path,body=None: self.fail('reviewer was called'),
                queue,'ptah','m'*64,broken_verifier)
            self.assertEqual(rows[0]['status'],'failed')
            self.assertFalse(rows[0]['review_complete'])
            self.assertIn('guest unavailable',rows[0]['review']['review_text'])
            self.assertEqual(queue.require_terminal()[0]['status'],'failed')


if __name__ == '__main__': unittest.main()
