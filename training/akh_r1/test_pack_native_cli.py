import copy,json,os,tempfile,unittest
from pathlib import Path
from pack_native_cli import ROOT,RUNTIME,preflight,launch
from pack_preparation import prepare
from episode_pack import file_sha,sha
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)/'out';pack=Path(os.environ['AKH_GRAPH_PACK']);bridge=Path(os.environ['AKH_GRAPH_BRIDGE'])
  paths={'pack':pack/'manifest.json','registry':pack/'registry.json','bridge':bridge};paths.update({'runtime:'+n:ROOT/n for n in RUNTIME})
  self.spec=dict(schema='akh-native-pack-launch-v1',pack=str(pack),episode_id='dev-acquisition',adapter='graph-query-v1',bridge=str(bridge),image='sha256:'+'a'*64,fingerprints={n:file_sha(p) for n,p in paths.items()},assistance_sha256=sha(prepare(pack,'dev-acquisition','graph-query-v1')['assistance']))
 def test_check_only_never_launches(self):
  calls=[];r=launch(self.spec,self.out,'lock',run=lambda *a,**k:calls.append(a));self.assertTrue(r['preflight_passed']);self.assertEqual(calls,[]);self.assertFalse(r['R1_ready'])
 def test_mismatch_prevents_artifacts_and_process(self):
  for key in ['bridge','pack','runtime:graph_query_client.py']:
   s=copy.deepcopy(self.spec);s['fingerprints'][key]='0'*64
   with self.assertRaises(ValueError):launch(s,self.out,'lock',True,run=lambda *_:self.fail('launched'))
   self.assertFalse(self.out.exists())
 def test_assistance_and_adapter_rejected(self):
  for key,value in [('assistance_sha256','bad'),('adapter','language-v1'),('image','mutable:latest')]:
   s=copy.deepcopy(self.spec);s[key]=value
   with self.assertRaises(ValueError):preflight(s)
 def test_explicit_execute_routes_to_bounded_runner(self):
  calls=[];launch(self.spec,self.out,'lock',True,run=lambda *a,**k:calls.append((a,k)));self.assertEqual(len(calls),1);cmd=calls[0][0][0];self.assertIn('--prepared',cmd);self.assertIn('--lock',cmd);self.assertTrue(calls[0][1]['check'])
 def test_launcher_failure_propagates_without_success_receipt(self):
  def fail(*a,**k):raise RuntimeError('failed launch')
  with self.assertRaises(RuntimeError):launch(self.spec,self.out,'lock',True,run=fail)
  self.assertFalse((self.out/'preflight.json').exists())
if __name__=='__main__':unittest.main()
