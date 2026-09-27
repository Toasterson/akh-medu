import tempfile, unittest
from pathlib import Path
from lilith_rl_tasks_v2 import export, extract_artifact, tasks, verify

class V2Tasks(unittest.TestCase):
    def test_references_negative_controls_and_fences(self):
        for task in tasks():
            with self.subTest(task=task.id):
                self.assertTrue(verify(task.reference,task)['passed'])
                self.assertFalse(verify(task.negative_control,task)['passed'])
                fenced='```python\n'+task.reference.rstrip()+'\n```'
                receipt=verify(fenced,task)
                self.assertTrue(receipt['passed'])
                self.assertFalse(receipt['formatting_compliant'])
                self.assertEqual(receipt['presentation'],'single_markdown_fence')
    def test_disjoint_families_and_frozen_export(self):
        train={t.family for t in tasks() if t.split=='train'}
        held={t.family for t in tasks() if t.split=='development_holdout'}
        self.assertTrue(train.isdisjoint(held))
        self.assertEqual((sum(t.split=='train' for t in tasks()),sum(t.split=='development_holdout' for t in tasks())),(4,4))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tasks.json'; m=export(p)
            self.assertEqual(m['schema'],'lilith/rl-tasks@2')
            with self.assertRaises(FileExistsError): export(p)
    def test_only_one_complete_fence_is_extracted(self):
        self.assertEqual(extract_artifact('```python\nx\n```'),('x','single_markdown_fence'))
        text='before```python\nx\n```'
        self.assertEqual(extract_artifact(text),(text,'raw_source'))

if __name__=='__main__': unittest.main()
