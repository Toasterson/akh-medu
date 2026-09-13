import copy,tempfile,unittest,types
from unittest.mock import patch
from pathlib import Path
from control_native_launch import launch,preflight
from episode_pack import file_sha
import test_control_binding as binding_tests
RUNS=binding_tests.RUNS
class Tests(unittest.TestCase):
 def setUp(self):
  self.disk=patch('control_native_launch.shutil.disk_usage',return_value=types.SimpleNamespace(free=10*2**30));self.disk.start();self.addCleanup(self.disk.stop)
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);bridge=self.root/'bridge';bridge.write_text('not executable; no native calls')
  pack=RUNS/'akh-r1-control-preparation-20260913-v1/ownership-pack';adapter='ownership-v1';episode='ownership-owner_correction'
  launcher=Path(__file__).with_name('control_native_launch.py').resolve()
  self.spec=dict(schema='akh-control-launch-v1',pack=str(pack),adapter=adapter,episode_id=episode,binding=binding_tests.Tests().binding(pack,episode,adapter),bridge=dict(path=str(bridge),sha256=file_sha(bridge)),image='sha256:'+'a'*64,launcher=dict(path=str(launcher),sha256=file_sha(launcher)),allowance=None);self.calls=[];self.lock=self.root/'lock'
 def runner(self,*args):self.calls.append(args)
 def allowance(self):self.spec['allowance']=dict(id='test-allowance',max_episodes=1,max_calls=32,max_seconds=30)
 def test_default_does_not_execute(self):
  launch(self.spec,self.root/'out',self.lock,runner=self.runner);self.assertEqual(self.calls,[])
 def test_execution_requires_allowance(self):
  with self.assertRaises(ValueError):launch(self.spec,self.root/'out',self.lock,True,self.runner)
  self.assertFalse((self.root/'out').exists())
 def test_single_spend(self):
  self.allowance();launch(self.spec,self.root/'first',self.lock,True,self.runner)
  with self.assertRaises(FileExistsError):launch(self.spec,self.root/'second',self.lock,True,self.runner)
  self.assertEqual(len(self.calls),1)
 def test_failed_runner_records_failure_and_consumes_allowance(self):
  self.allowance()
  def fail(*args):raise RuntimeError('injected failure')
  with self.assertRaises(RuntimeError):launch(self.spec,self.root/'out',self.lock,True,fail)
  self.assertIn('failed',(self.root/'out/status.json').read_text())
  with self.assertRaises(FileExistsError):launch(self.spec,self.root/'retry',self.lock,True,self.runner)
 def test_invalid_pins_and_limits(self):
  self.allowance()
  for key,value in [('image','latest'),('max_seconds',301),('max_episodes',2),('bridge','0'*64)]:
   bad=copy.deepcopy(self.spec)
   if key in ['max_seconds','max_episodes']:bad['allowance'][key]=value
   elif key=='bridge':bad['bridge']['sha256']=value
   else:bad[key]=value
   with self.assertRaises(ValueError):preflight(bad,True)
if __name__=='__main__':unittest.main()
