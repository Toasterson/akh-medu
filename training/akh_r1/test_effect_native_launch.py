import copy,json,tempfile,unittest
from pathlib import Path
from effect_native_launch import ROOT,RUNTIME,launch,preflight
from episode_pack import file_sha
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.prepared=Path('/Users/toasty/ws/akh-medu/runs/akh-r1-identity-effect-adapter-20260913-v1/inventory-complete/host.json');bridge=self.root/'bridge';bridge.write_text('fake bridge, never executed');paths={'prepared':self.prepared,'bridge':bridge};paths.update({'runtime:'+n:ROOT/n for n in RUNTIME});self.spec=dict(schema='akh-effect-native-launch-v1',prepared=str(self.prepared),bridge=str(bridge),image='sha256:'+'a'*64,budget_seconds=30,fingerprints={k:file_sha(p) for k,p in paths.items()});self.out=self.root/'out';self.calls=[]
 def runner(self,*args):self.calls.append(args)
 def test_default_only_checks(self):
  r=launch(self.spec,self.out,self.root/'lock',runner=self.runner);self.assertFalse(r['execution_requested']);self.assertEqual(self.calls,[]);self.assertEqual((self.out/'prepared.json').stat().st_mode&0o777,0o600)
 def test_explicit_execution_routes_to_injected_runner(self):
  launch(self.spec,self.out,self.root/'lock',True,self.runner);self.assertEqual(len(self.calls),1)
 def test_mismatch_no_output_or_process(self):
  self.spec['fingerprints']['prepared']='0'*64
  with self.assertRaises(ValueError):launch(self.spec,self.out,self.root/'lock',True,self.runner)
  self.assertFalse(self.out.exists());self.assertEqual(self.calls,[])
 def test_mutable_image_and_excess_budget_rejected(self):
  for field,value in [('image','latest'),('budget_seconds',301)]:
   s=copy.deepcopy(self.spec);s[field]=value
   with self.assertRaises(ValueError):preflight(s)
 def test_failed_execution_never_claims_preflight_completion(self):
  def fail(*args):raise RuntimeError('failed')
  with self.assertRaises(RuntimeError):launch(self.spec,self.out,self.root/'lock',True,fail)
  self.assertFalse((self.out/'preflight.json').exists())
 def test_unsupported_adapter_rejected_even_with_updated_digest(self):
  p=self.root/'bad.json';d=json.loads(self.prepared.read_text());d['adapter']='other';p.write_text(json.dumps(d));self.spec['prepared']=str(p);self.spec['fingerprints']['prepared']=file_sha(p)
  with self.assertRaises(ValueError):launch(self.spec,self.out,self.root/'lock',True,self.runner)
  self.assertEqual(self.calls,[]);self.assertFalse(self.out.exists())
if __name__=='__main__':unittest.main()
