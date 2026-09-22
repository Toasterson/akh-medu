import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from adam_kaine_code_rl_tasks import tasks
from lilith_runtime_training_worker import checked_read, claim_bank, episode


class MockBroker:
    """A test double only. Integration evidence must come from actual daemons."""
    def __init__(self):
        self.content = None
        self.receipts = []
        self.request = {'id':'job-fixture', 'session_id':'session-fixture', 'workspace':{'id':'fixture'},
            'instructions':'Use supplied tools.', 'tools':[
                {'name':'read_file', 'description':'Read', 'input_schema':{'type':'object',
                 'required':['path'], 'properties':{'path':{'type':'string'}}, 'additionalProperties':False}},
                {'name':'write_file', 'description':'Write', 'input_schema':{'type':'object',
                 'required':['path','content','create'], 'properties':{'path':{'type':'string'},
                 'content':{'type':'string'}, 'create':{'type':'boolean'},
                 'expected_revision':{'type':'string'}}, 'additionalProperties':False}}]}

    def tool(self, name, arguments):
        if name == 'write_file':
            if arguments['create'] and self.content is not None:
                data = {'error':{'code':'conflict', 'message':'create conflict'}}
                result = {'success':False, 'text':json.dumps(data), 'data':data,
                          'receipt':'mock-' + str(len(self.receipts))}
                self.receipts.append({'name':name, 'arguments':arguments, 'result':result})
                return result
            if not arguments['create'] and self.content is None:
                data = {'error':{'code':'not_found', 'message':'missing'}}
                result = {'success':False, 'text':json.dumps(data), 'data':data,
                          'receipt':'mock-' + str(len(self.receipts))}
                self.receipts.append({'name':name, 'arguments':arguments, 'result':result})
                return result
            revision = hashlib.sha256(self.content.encode()).hexdigest() if self.content is not None else None
            if not arguments['create'] and arguments.get('expected_revision') != revision:
                data = {'error':{'code':'invalid_request', 'message':'write requires matching expected_revision'}}
                result = {'success':False, 'text':json.dumps(data), 'data':data,
                          'receipt':'mock-' + str(len(self.receipts))}
                self.receipts.append({'name':name, 'arguments':arguments, 'result':result})
                return result
            self.content = arguments['content']
            data = {'revision':hashlib.sha256(self.content.encode()).hexdigest(), 'bytes':len(self.content.encode())}
        else:
            if self.content is None:
                data = {'error':{'code':'not_found', 'message':'missing'}}
                result = {'success':False, 'text':json.dumps(data), 'data':data,
                          'receipt':'mock-' + str(len(self.receipts))}
                self.receipts.append({'name':name, 'arguments':arguments, 'result':result})
                return result
            data = {'revision':hashlib.sha256(self.content.encode()).hexdigest(), 'content':self.content, 'encoding':'utf8'}
        result = {'success':True, 'text':json.dumps(data), 'data':data, 'receipt':'mock-' + str(len(self.receipts))}
        self.receipts.append({'name':name, 'arguments':arguments, 'result':result})
        return result

    def progress(self, _phase):
        pass


class FakeGeneration:
    def __init__(self, status, text, identifier):
        self.completion_status, self.response_text, self.id = status, text, identifier
        self.reward = None

    def to_dict(self):
        return {'completion_status':self.completion_status, 'response_text':self.response_text, 'id':self.id}


class FakeEngine:
    def __init__(self, responses):
        self.responses = iter(responses)

    def generate(self, **_kwargs):
        return next(self.responses)


class ProgressEngine(FakeEngine):
    def generate(self, **kwargs):
        kwargs['token_observer']({'step':127})
        return super().generate(**kwargs)


class WorkerTests(unittest.TestCase):
    def test_reward_requires_real_readback_matching_model_write(self):
        task = tasks()[0]
        initial = '# Assigned task: ' + task.id + '\n# Implement the function in the task specification.\n'
        revision = hashlib.sha256(initial.encode()).hexdigest()
        call = '<|tool_call>call:write_file{path:<|"|>clamp_u8.py<|"|>,content:<|"|>' + task.reference + \
               '<|"|>,create:false,expected_revision:<|"|>' + revision + '<|"|>}<tool_call|>'
        engine = FakeEngine([FakeGeneration('tool_handoff', call, 'write'),
                             FakeGeneration('completed', 'I wrote the function. No native test was run.', 'final')])
        broker = MockBroker()
        with tempfile.TemporaryDirectory() as directory:
            rows, receipt = episode(engine, broker, task, 'fixture',
                {'artifact_path':'clamp_u8.py', 'max_iterations':4, 'max_output_tokens':16384}, Path(directory))
        self.assertTrue(receipt['write_bound_to_readback'])
        self.assertEqual(receipt['reward'], 1.0)
        self.assertTrue(receipt['verification']['passed'])
        self.assertEqual([row.reward for row in rows], [1.0, 1.0])

    def test_second_episode_resets_existing_artifact_without_create_conflict(self):
        broker = MockBroker()
        task = tasks()[0]
        config = {'artifact_path':'clamp_u8.py', 'max_iterations':4, 'max_output_tokens':16384}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            episode(FakeEngine([FakeGeneration('completed', 'Done.', 'first')]),
                    broker, task, 'fixture', config, output)
            episode(FakeEngine([FakeGeneration('completed', 'Done.', 'second')]),
                    broker, task, 'fixture', config, output)
        setup_writes = [row for row in broker.receipts if row['name'] == 'write_file']
        self.assertEqual([row['arguments']['create'] for row in setup_writes], [True, False])
        self.assertNotIn('expected_revision', setup_writes[0]['arguments'])
        self.assertEqual(setup_writes[1]['arguments']['expected_revision'],
                         hashlib.sha256(setup_writes[0]['arguments']['content'].encode()).hexdigest())
        self.assertTrue(all(row['result']['success'] for row in setup_writes))

    def test_generation_progress_records_prefill_and_token_count_without_text(self):
        broker = MockBroker()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            episode(ProgressEngine([FakeGeneration('completed', 'Done.', 'progress')]),
                    broker, tasks()[0], 'fixture',
                    {'artifact_path':'clamp_u8.py', 'max_iterations':4, 'max_output_tokens':16384}, output)
            progress = json.loads((output / 'progress.json').read_text())
        self.assertEqual(progress, {'phase':'generation', 'task_id':tasks()[0].id, 'tokens':128})

    def test_entire_invalid_plan_rejected_before_model_effects(self):
        text = '<|tool_call>call:write_file{path:<|"|>clamp_u8.py<|"|>,content:<|"|>bad<|"|>,create:false}<tool_call|> prose'
        broker = MockBroker()
        with tempfile.TemporaryDirectory() as directory:
            _, receipt = episode(FakeEngine([FakeGeneration('tool_handoff', text, 'bad')]), broker, tasks()[0],
                'fixture', {'artifact_path':'clamp_u8.py', 'max_iterations':4, 'max_output_tokens':16384}, Path(directory))
        self.assertEqual(receipt['status'], 'invalid_plan')
        self.assertEqual(receipt['reward'], 0.0)
        self.assertTrue(receipt['eligible_for_update'])
        self.assertEqual(sum(r['name'] == 'write_file' for r in broker.receipts), 1)  # Setup only.

    def test_truncation_is_ineligible_and_not_semantic_failure(self):
        broker = MockBroker()
        with tempfile.TemporaryDirectory() as directory:
            _, receipt = episode(FakeEngine([FakeGeneration('max_new_tokens', '<|channel>thought\n', 'cut')]), broker,
                tasks()[0], 'fixture', {'artifact_path':'clamp_u8.py', 'max_iterations':4, 'max_output_tokens':16384}, Path(directory))
        self.assertEqual(receipt['status'], 'resource_limited')
        self.assertFalse(receipt['eligible_for_update'])

    def test_text_without_bound_content_revision_is_not_read_evidence(self):
        with self.assertRaises(ValueError):
            checked_read({'success':True, 'text':'all tests passed', 'receipt':'r',
                          'data':{'content':'fabricated', 'encoding':'utf8', 'revision':'wrong'}})

    def test_consumed_bank_cannot_repeat_optimizer_claim(self):
        row = SimpleNamespace(id='trajectory', policy_sha256='policy')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.sqlite'
            claim_bank(path, [row], 'bank-1', 'verifier')
            with self.assertRaises(sqlite3.IntegrityError):
                claim_bank(path, [row], 'bank-2', 'verifier')


if __name__ == '__main__':
    unittest.main()
