import copy, json, shutil, tempfile, unittest
from pathlib import Path
from lilith_rl_report_v2 import audit

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'training/lilith/rl-20260922/live/run2'
MANIFEST=ROOT/'training/lilith/rl-20260922/tasks.json'

class IndependentReplayV2(unittest.TestCase):
    def fixture(self):
        tmp=tempfile.TemporaryDirectory(); target=Path(tmp.name)/'run'; target.mkdir()
        for name in ('config.json','result.json','base-trajectories.json','sft-trajectories.json','klpo-trajectories.json'):
            shutil.copyfile(RUN/name,target/name)
        return tmp,target
    def test_actual_run_is_complete(self):
        report=audit(RUN,MANIFEST)
        self.assertTrue(report['complete'])
        self.assertFalse(report['quality_improvement_established'])
    def test_forged_reward_receipt_is_rejected(self):
        tmp,path=self.fixture()
        try:
            rows=json.loads((path/'base-trajectories.json').read_text())
            rows[0]['reward']['evidence']['passed']=False
            (path/'base-trajectories.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError): audit(path,MANIFEST)
        finally: tmp.cleanup()
    def test_duplicate_task_is_rejected(self):
        tmp,path=self.fixture()
        try:
            rows=json.loads((path/'sft-trajectories.json').read_text()); rows.append(copy.deepcopy(rows[0]))
            (path/'sft-trajectories.json').write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'Duplicate'): audit(path,MANIFEST)
        finally: tmp.cleanup()
    def test_collection_provenance_drift_is_rejected(self):
        tmp,path=self.fixture()
        try:
            cfg=json.loads((path/'config.json').read_text()); cfg['task_module_sha256']='0'*64
            (path/'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'provenance'): audit(path,MANIFEST)
        finally: tmp.cleanup()

if __name__=='__main__': unittest.main()
