import copy
import json
from pathlib import Path
import unittest
import lilith_eval as e

MANIFEST = Path(__file__).resolve().parents[2] / 'benchmarks/lilith/apprenticeship-v2/manifest.json'


def solution(task):
    maximum=(1<<task['bits'])-1
    if task['family']=='alignment':
        return f'''def align_up(value, alignment):
    if value < 0 or value > {maximum} or alignment <= 0 or alignment > {maximum}:
        return -1
    if alignment & (alignment-1) != 0:
        return -1
    result = ((value+alignment-1)//alignment)*alignment
    if result > {maximum}:
        return -1
    return result
'''
    if task['family']=='interval_intersection':
        return f'''def overlap_size(a_start, a_end, b_start, b_end):
    if a_start < 0 or a_start > {maximum} or a_end < 0 or a_end > {maximum} or b_start < 0 or b_start > {maximum} or b_end < 0 or b_end > {maximum}:
        return -1
    if a_end < a_start or b_end < b_start:
        return -1
    lower = a_start if a_start > b_start else b_start
    upper = a_end if a_end < b_end else b_end
    return upper-lower if upper > lower else 0
'''
    return f'''def ring_distance(start, end, modulus):
    if start < 0 or start > {maximum} or end < 0 or end > {maximum} or modulus <= 0 or modulus > {maximum} or start >= modulus or end >= modulus:
        return -1
    return (end-start)%modulus
'''


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.m=json.loads(MANIFEST.read_text())

    def record(self, task=None, arm='none'):
        task=task or self.m['tasks'][0]
        model=self.m['models']['reference' if arm=='independent_reference' else 'candidate']
        return dict(task=task['id'],arm=arm,seed=0,manifest_sha256=e.digest(e.canonical(self.m)),
            model_sha256=model['sha256'],tokenizer_sha256=model['tokenizer_sha256'],
            budget=copy.deepcopy(self.m['budget']),usage=dict(attempts=1,tools=0,output_tokens=100,input_tokens=300,wall_seconds=10),
            retrieval=dict(context='',lessons=[]),source=solution(task),status='completed',finish_reason='stop')

    def test_all_reference_solutions_and_domains(self):
        for task in self.m['tasks']:
            with self.subTest(task=task['id']):
                result=e.verify(solution(task),task)
                self.assertTrue(result['passed'],result)
                self.assertGreater(result['cases'],100)

    def test_schedule_is_reproducible_and_rotates_arms(self):
        a=e.schedule(self.m)
        self.assertEqual(a,e.schedule(copy.deepcopy(self.m)))
        self.assertEqual(len(a),24)
        self.assertEqual(a[0]['arm'],'none')
        self.assertEqual(a[4]['arm'],'text')
        self.assertTrue(all(row['budget']==self.m['budget'] for row in a))

    def test_manifest_rejects_family_and_verifier_drift(self):
        for key,value in [('training_families',['alignment']),('verifier_sha256','0'*64),('seeds',[0,0])]:
            with self.subTest(key=key):
                m=copy.deepcopy(self.m);m[key]=value
                with self.assertRaises(ValueError):e.validate_manifest(m)

    def test_manifest_rejects_fake_certification(self):
        self.m['claims']['qwen_certified']=True
        with self.assertRaises(ValueError):e.validate_manifest(self.m)

    def test_source_is_not_executed_by_host(self):
        task=self.m['tasks'][0]
        for source in ('import os\nos.system("false")','def align_up(value, alignment):\n return __import__("os")',
                       'def align_up(value, alignment):\n while True: pass',
                       'def align_up(value, alignment):\n return value ** 999999'):
            with self.subTest(source=source):
                self.assertEqual(e.verify(source,task)['category'],'unsupported_syntax')

    def test_numeric_resource_exhaustion_is_bounded(self):
        source='def align_up(value, alignment):\n x=18446744073709551616\n x=x*x\n return x*x'
        self.assertEqual(e.verify(source,self.m['tasks'][0])['category'],'execution')

    def test_failure_taxonomy_is_evidence_based(self):
        task=self.m['tasks'][0]
        self.assertEqual(e.verify('def nope(',task)['category'],'syntax')
        source='def align_up(value, alignment):\n return 0'
        result=e.verify(source,task)
        self.assertEqual(result['category'],'domain_rejection')
        self.assertIn('input',result)
        row=self.record();row['finish_reason']='length'
        self.assertEqual(e.evaluate(self.m,[row])['outcomes'][0]['category'],'output_budget')

    def test_missing_runs_are_not_passing_or_statistical_claims(self):
        report=e.evaluate(self.m,[self.record()])
        self.assertFalse(report['complete'])
        self.assertEqual(report['arms']['none']['observed'],1)
        self.assertEqual(report['arms']['none']['expected'],6)
        self.assertEqual(report['paired_against_none']['vsa']['missing'],6)
        self.assertFalse(any(report['claims'].values()))

    def test_pairs_include_runtime_failures_and_abstention(self):
        rows=[self.record(arm=a) for a in e.ARMS]
        rows[1]['status']='runtime_error';rows[1]['source']=''
        report=e.evaluate(self.m,rows)
        self.assertEqual(report['paired_against_none']['text']['baseline_only_pass'],1)
        self.assertEqual(report['paired_against_none']['vsa']['both_pass'],1)

    def test_unknown_usage_remains_explicitly_unmeasured(self):
        row=self.record();row['usage']['output_tokens']=None
        report=e.evaluate(self.m,[row])
        self.assertFalse(report['budget_usage_complete'])
        self.assertTrue(report['outcomes'][0]['passed'])

    def test_duplicate_attempt_rejected(self):
        row=self.record()
        with self.assertRaises(ValueError):e.evaluate(self.m,[row,row])

    def test_budget_and_model_parity_fail_closed(self):
        for key in ('budget','model_sha256','usage'):
            row=self.record()
            if key=='budget':row[key]['max_new_tokens']+=1
            elif key=='model_sha256':row[key]='0'*64
            else:row[key]['output_tokens']=513
            with self.subTest(key=key),self.assertRaises(ValueError):e.evaluate(self.m,[row])

    def test_retrieval_family_leakage_and_reference_isolation(self):
        row=self.record(arm='vsa')
        row['retrieval']=dict(context='leaked',lessons=[dict(id='lesson',family='alignment',source_sha256='a'*64)])
        with self.assertRaisesRegex(ValueError,'family leakage'):e.evaluate(self.m,[row])
        row=self.record(arm='independent_reference');row['retrieval']['context']='lesson'
        with self.assertRaises(ValueError):e.evaluate(self.m,[row])

    def test_training_source_duplicate_detected_despite_whitespace(self):
        row=self.record()
        self.m['training_source_fingerprints'].append(e.source_fingerprint(row['source']))
        row['source']='\n'+row['source']+'\n'
        row['manifest_sha256']=e.digest(e.canonical(self.m))
        with self.assertRaisesRegex(ValueError,'duplicates training'):e.evaluate(self.m,[row])

    def test_all_observations_complete_but_still_no_release_claim(self):
        rows=[self.record(task,arm) for task in self.m['tasks'] for arm in e.ARMS]
        report=e.evaluate(self.m,rows)
        self.assertTrue(report['complete'])
        self.assertTrue(all(arm['passed']==6 for arm in report['arms'].values()))
        self.assertFalse(any(report['claims'].values()))


if __name__=='__main__':unittest.main()
