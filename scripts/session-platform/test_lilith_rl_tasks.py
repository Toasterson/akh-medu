import ast
import json
from pathlib import Path
import tempfile
import unittest
from lilith_rl_tasks import Interpreter,Rejected,export,tasks,verify


class RLTaskTests(unittest.TestCase):
    def test_references_pass_and_all_negative_controls_fail(self):
        for task in tasks():
            with self.subTest(task=task.id):
                self.assertTrue(verify(task.reference,task)['passed'])
                bad=verify(task.negative_control,task)
                self.assertFalse(bad['passed'])
                self.assertEqual(bad['category'],'contract_mismatch')
                self.assertIsNotNone(bad['counterexample'])

    def test_family_split_and_export_never_include_holdout_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tasks.json';data=export(path)
            training=[t for t in tasks() if t.split=='train']
            holdouts=[t for t in tasks() if t.split=='development_holdout']
            self.assertEqual((len(training),len(holdouts)),(4,4))
            self.assertTrue({t.family for t in training}.isdisjoint({t.family for t in holdouts}))
            self.assertEqual({r['id'] for r in data['training_targets']},{t.id for t in training})
            for row in data['tasks']:
                self.assertNotIn('reference',row)
                self.assertNotIn('output',row)
            with self.assertRaises(FileExistsError):export(path)

    def test_domain_invalid_values_are_tested_only_when_contract_requires(self):
        indexed={t.id:t for t in tasks()}
        for task_id in ('mask-intersection-u8/1','compose-le16/1'):
            task=indexed[task_id]
            self.assertTrue(any(a<0 or b>255 for a,b,_ in task.cases))
        self.assertTrue(all(0<=a<=65535 and b in (0,1) for a,b,_ in indexed['select-le16-byte/1'].cases))

    def test_interpreter_matches_python_primitives_without_running_candidate(self):
        samples=[('a // b',lambda a,b:a//b),('a % b',lambda a,b:a%b),
                 ('~a & 255',lambda a,b:~a&255),('a >> b',lambda a,b:a>>b),
                 ('a << b',lambda a,b:a<<b),('a ^ b',lambda a,b:a^b),
                 ('a | b',lambda a,b:a|b),('a and b',lambda a,b:a and b),
                 ('a or b',lambda a,b:a or b),('not a',lambda a,b:not a)]
        for text,oracle in samples:
            interpreter=Interpreter('def f(a, b):\n    return '+text+'\n')
            for a in (-255,-1,0,1,255):
                for b in (1,2,7):self.assertEqual(interpreter(a,b),oracle(a,b))
        self.assertEqual(Interpreter('def f(a, b):\n    return a == 0 or b // a\n')(0,2),True)

    def test_side_effects_large_shifts_and_growth_are_rejected(self):
        for source in ('import os','def f(a,b):\n    return open(a)',
                       'def f(a,b):\n    while a:\n        a=a-1\n    return a',
                       'def f(a,b):\n    return a.real',
                       'def f(a,b):\n    def g():\n        return 1\n    return a',
                       'def f(a,b=1):\n    return a'):
            with self.subTest(source=source),self.assertRaises(Rejected):Interpreter(source)
        for expr in ('1 << 64','1 << -1','1 // 0','(1 << 63) * (1 << 63) * 4'):
            with self.subTest(expr=expr),self.assertRaises(Rejected):
                Interpreter('def f(a,b):\n    return '+expr+'\n')(0,0)

    def test_optional_annotations_do_not_change_execution(self):
        function=Interpreter('def f(a: int, b: int) -> bool:\n    return a < b\n')
        self.assertIs(function(1,2),True)
        with self.assertRaises(Rejected):Interpreter('def f(a: unknown, b):\n    return a\n')

    def test_strict_format_and_bool_contract(self):
        task=next(t for t in tasks() if t.id=='required-flags-u8/1')
        self.assertEqual(verify('```python\n'+task.reference+'```',task)['category'],'syntax')
        result=verify('def f(a,b):\n    return 1\n',task)
        self.assertFalse(result['passed'])
        self.assertEqual(result['category'],'contract_mismatch')


if __name__=='__main__':unittest.main()
