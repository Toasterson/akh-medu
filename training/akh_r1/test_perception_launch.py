import copy,json,tempfile,unittest
from pathlib import Path
from control_native_launch import preflight,launch
from episode_pack import file_sha
BASE=Path('/Users/toasty/ws/akh-medu/runs/akh-r1-perception-dispatch-binding-20260913-v1')
class Tests(unittest.TestCase):
 def spec(self,name):
  binding=json.loads((BASE/(name+'-binding.json')).read_text());launcher=Path(__file__).with_name('control_native_launch.py').resolve();bridge=Path('/Users/toasty/ws/akh-medu-babyllm-recovery/target/debug/examples/akh_symbolic_episode_bridge');ref=lambda p:dict(path=str(p),sha256=file_sha(p))
  return dict(schema='akh-control-launch-v1',pack=binding['source'],episode_id=name,adapter='spatial-perception-v1',binding=binding,bridge=ref(bridge),launcher=ref(launcher),image='sha256:'+'a'*64,allowance=None)
 def test_retained_conditions_default_no_execution(self):
  for name in ['table-complete','table-missing_structure','table-transcript']:
   with tempfile.TemporaryDirectory() as t:
    calls=[];launch(self.spec(name),Path(t)/'out',Path(t)/'lock',runner=lambda *args:calls.append(args));self.assertFalse(calls)
 def test_identity_rejected(self):
  s=self.spec('table-complete');s['episode_id']='table-transcript'
  with self.assertRaises(ValueError):preflight(s)
 def test_explicit_48_call_allowance_required(self):
  s=self.spec('table-complete');s['allowance']=dict(id='test',max_episodes=1,max_calls=32,max_seconds=60)
  with self.assertRaises(ValueError):preflight(s,True)
  s['allowance']['max_calls']=48;self.assertEqual(preflight(s,True)['host_config']['modality'],'retained_actual_ocr')
  s['allowance']['max_calls']=49
  with self.assertRaises(ValueError):preflight(s,True)
if __name__=='__main__':unittest.main()
