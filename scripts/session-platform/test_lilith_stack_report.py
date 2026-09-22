import json
from pathlib import Path
import shutil
import tempfile
import unittest
import lilith_stack_report as report

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training/lilith/comparison-20260922/run1'
MANIFEST=ROOT/'training/lilith/rl-20260922/tasks.json'

class DeploymentReplayTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.run=Path(self.tmp.name)
        for name in ('protocol.json','records.json','result.json'):
            shutil.copyfile(RUN/name,self.run/name)
    def mutate(self,change):
        p=self.run/'records.json';rows=json.loads(p.read_text());change(rows)
        p.write_text(json.dumps(rows))
    def test_complete_actual_run(self):
        r=report.report(self.run,MANIFEST)
        self.assertTrue(r['complete']);self.assertFalse(r['release_parity_established'])
        self.assertEqual(r['scores']['lilith']['passed'],4)
        self.assertEqual(r['scores']['ptah']['passed'],6)
    def test_forged_pass_rejected(self):
        self.mutate(lambda rows:rows[0]['evidence'].update(passed=True))
        with self.assertRaisesRegex(ValueError,'evidence drift'):
            report.report(self.run,MANIFEST)
    def test_duplicate_task_rejected(self):
        self.mutate(lambda rows:rows.append(rows[0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            report.report(self.run,MANIFEST)
    def test_missing_observation_prevents_complete(self):
        self.mutate(lambda rows:rows.pop())
        self.assertFalse(report.report(self.run,MANIFEST)['complete'])

if __name__=='__main__':unittest.main()
