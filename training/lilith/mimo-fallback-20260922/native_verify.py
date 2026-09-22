#!/usr/bin/env python3
"""Replay exact grammar-admitted MiMo answers in the pinned illumos executor."""
import hashlib, json, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/session-platform'))
import lilith_rl_tasks_v2 as tasks
from lilith_vm_executor import local_executor

RUN=Path(__file__).with_name('full8-thinking8192')
OUT=Path(__file__).with_name('native-illumos')
sha=lambda value:hashlib.sha256(value if isinstance(value,bytes) else value.encode()).hexdigest()

def test_source(task):
    cases=repr(tuple(task.cases))
    return ('import unittest\nfrom solution import f\n\nclass NativeReplay(unittest.TestCase):\n'
            '    def test_frozen_cases(self):\n'
            f'        for a, b, expected in {cases}:\n'
            '            with self.subTest(a=a, b=b):\n'
            '                actual = f(a, b)\n'
            '                self.assertIs(type(actual), type(expected))\n'
            '                self.assertEqual(actual, expected)\n\n'
            "if __name__ == '__main__': unittest.main()\n")

def main():
    OUT.mkdir(exist_ok=False)
    rows=json.loads((RUN/'records.json').read_text());pack={item.id:item for item in tasks.tasks()}
    executor=local_executor();contract=executor.contract()
    summary=[]
    for row in rows:
        task=pack[row['task_id']]
        frozen=tasks.verify(row['content'],task)
        artifact,presentation=tasks.extract_artifact(row['content'])
        if frozen['category']=='grammar':
            summary.append({'task_id':task.id,'raw_output_sha256':sha(row['content']),
                'artifact_sha256':sha(artifact),'presentation':presentation,
                'grammar_admitted':False,'native_executed':False,'reason':'frozen grammar rejection'})
            continue
        with tempfile.TemporaryDirectory(prefix='mimo-native-') as directory:
            repo=Path(directory);(repo/'solution.py').write_text('def f(a, b):\n    return None\n')
            tests=test_source(task);(repo/'test_solution.py').write_text(tests)
            receipt=executor.execute(repo,'solution.py',artifact,
                                     ['python3','-m','unittest','-q'],timeout_s=60)
        receipt.update({'task_id':task.id,'raw_output_sha256':sha(row['content']),
            'artifact_sha256':sha(artifact),'presentation':presentation,
            'test_source_sha256':sha(tests),'frozen_interpreter_passed':frozen['passed'],
            'frozen_interpreter_category':frozen['category'],'frozen_cases':len(task.cases)})
        name=task.id.replace('/','-')+'-receipt.json'
        (OUT/name).write_text(json.dumps(receipt,indent=2)+'\n')
        summary.append({'task_id':task.id,'raw_output_sha256':sha(row['content']),
            'artifact_sha256':sha(artifact),'presentation':presentation,
            'grammar_admitted':True,'native_executed':True,'native_passed':receipt['passed'],
            'exit_code':receipt['exit_code'],'timed_out':receipt['timed_out'],
            'cleanup_succeeded':receipt['remote_cleanup_succeeded'],'receipt':name,
            'frozen_interpreter_passed':frozen['passed'],'frozen_cases':len(task.cases)})
    result={'schema':'lilith/mimo-native-illumos@1','executor_contract':contract,
        'executor_contract_sha256':sha(json.dumps(contract,sort_keys=True,separators=(',',':'))),
        'source_records_sha256':sha((RUN/'records.json').read_bytes()),'answers':summary,
        'grammar_admitted':sum(x['grammar_admitted'] for x in summary),
        'native_passed':sum(x.get('native_passed',False) for x in summary),
        'native_failed':sum(x.get('native_executed',False) and not x.get('native_passed',False) for x in summary),
        'not_executed':sum(not x.get('native_executed',False) for x in summary),
        'score_unchanged':'6/8 frozen interpreter; native replay is additional evidence, not rescoring'}
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'grammar_admitted':result['grammar_admitted'],'native_passed':result['native_passed'],
                      'native_failed':result['native_failed'],'not_executed':result['not_executed']}))

if __name__=='__main__':main()
