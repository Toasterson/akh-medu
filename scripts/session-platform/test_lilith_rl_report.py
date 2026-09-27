"""Offline audit controls; synthetic records are not model observations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import lilith_rl as core
import lilith_rl_tasks as tasks
import lilith_rl_report as report


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.policy = dict(model_sha256='0'*64, tokenizer_sha256='1'*64,
                           adapter_sha256='2'*64, version='initial')
        self.config = dict(task_module_sha256=report.file_hash(tasks.__file__),
            trajectory_module_sha256=report.file_hash(core.__file__), max_new_tokens=128,
            max_context_tokens=384, mc_samples=2, eos_ids=[106],
            task_manifest_sha256=report.file_hash(Path(__file__).resolve().parents[2] / 'training/lilith/rl-20260921/tasks.json'))
        self.write('config.json', self.config)
        result = dict(status='passed', initial_policy=self.policy)
        for arm in ('base', 'sft', 'klpo'):
            result[arm] = dict(adapter_sha256=self.policy['adapter_sha256'])
            self.write(arm+'-trajectories.json', [self.record(t,arm) for t in tasks.tasks()])
        self.write('result.json', result)

    def write(self, name, value):
        (self.path/name).write_text(json.dumps(value))

    def record(self, task, arm):
        policy = dict(self.policy)
        if arm != 'base': policy['version'] = arm+'-'+policy['adapter_sha256'][:16]
        evidence = tasks.verify(task.reference, task)
        evidence['generated_text_sha256'] = core.digest(task.reference)
        return core.seal_record(dict(schema='lilith/trajectory@1',id=arm+'/'+task.id,
            task=dict(id=task.id,family=task.family,split=task.split,prompt_sha256=core.digest(task.prompt)),
            policy=policy,sampler=dict(kind='categorical',temperature=1.0,top_k=0,top_p=1.0,
                full_support=True,with_replacement=True,aux_independent=True,action_rng_id='a',
                aux_rng_id='b',termination_token_ids=[106]),
            context=dict(prompt=task.prompt,features=[],lesson_ids=[],prompt_token_ids=[2]),
            token_ids=[2,7,106],token_roles=['prompt','assistant','assistant'],policy_mask=[False,True,True],
            sampler_logps=[None,-1.0,-1.0],mc_token_ids=[[],[7,8],[7,8]],
            mc_sampler_logps=[[],[-1.0,-2.0],[-1.0,-2.0]],
            completion=dict(status='completed',termination_token_id=106,generated_text=task.reference),
            reward=dict(value=1.0,verifier_sha256=self.config['task_module_sha256'],evidence=evidence)))

    def test_complete_replay_separates_training_and_holdouts(self):
        result = report.audit(self.path)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['outcomes']),24)
        self.assertEqual(result['arms']['klpo']['development_holdout']['passed'],4)
        self.assertFalse(result['quality_improvement_established'])

    def test_missing_arm_remains_missing(self):
        (self.path/'klpo-trajectories.json').unlink()
        result=report.audit(self.path)
        self.assertFalse(result['complete'])
        self.assertEqual(result['arms']['klpo']['train']['observed'],0)

    def test_semantic_receipt_cannot_be_forged_by_resealing(self):
        rows=json.loads((self.path/'base-trajectories.json').read_text())
        row=rows[0]
        row['completion']['generated_text']='def f(a,b):\n    return 0\n'
        row['reward']['evidence']['generated_text_sha256']=core.digest(row['completion']['generated_text'])
        rows[0]=core.seal_record(row)
        self.write('base-trajectories.json',rows)
        with self.assertRaisesRegex(ValueError,'verifier replay differs'):report.audit(self.path)

    def test_duplicate_observation_is_rejected(self):
        rows=json.loads((self.path/'base-trajectories.json').read_text())
        rows.append(copy.deepcopy(rows[0]))
        self.write('base-trajectories.json',rows)
        with self.assertRaisesRegex(ValueError,'Repeated task'):report.audit(self.path)

    def test_wrapper_diagnostic_is_unscored_and_does_not_rewrite_program(self):
        task=tasks.tasks()[0]
        source='```python\n'+task.reference.rstrip()+'\n```'
        diagnostic=report.wrapper_diagnostic(source,task)
        self.assertFalse(diagnostic['scored'])
        self.assertTrue(diagnostic['verification']['passed'])
        self.assertFalse(tasks.verify(source,task)['passed'])
        self.assertIsNone(report.wrapper_diagnostic('explanation\n'+source,task))
        wrong=report.wrapper_diagnostic('```python\n'+task.negative_control.rstrip()+'\n```',task)
        self.assertFalse(wrong['verification']['passed'])

    def test_manifest_change_cannot_silently_reinterpret_run(self):
        self.config['task_manifest_sha256']='0'*64
        self.write('config.json',self.config)
        with self.assertRaisesRegex(ValueError,'Task manifest'):report.audit(self.path)


if __name__=='__main__':unittest.main()
