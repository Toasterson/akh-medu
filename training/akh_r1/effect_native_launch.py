"""Check-only by default; explicit bounded native integration via registered adapter."""
import argparse,fcntl,json,os,re,shutil,signal
from pathlib import Path
from episode_pack import file_sha
from identity_effect_adapter import IdentityEffectAdapter,ADAPTER
from prepared_effect_dispatch import validate_effect,dispatch_effect
from build_curriculum import now
from run_public_breadth import write
ROOT=Path(__file__).parent
RUNTIME=['effect_native_launch.py','identity_effect_adapter.py','prepared_effect_dispatch.py','episode_pack.py','r0_contract.py','environment.py','public_episode.py','safe_pipeline.py','run_public_breadth.py','build_curriculum.py','stratified_action_client.py','stratified_effect_cli.py']

def preflight(spec):
 if set(spec)!={'schema','prepared','bridge','image','fingerprints','budget_seconds'} or spec['schema']!='akh-effect-native-launch-v1':raise ValueError('Invalid launch manifest')
 if type(spec['budget_seconds']) is not int or not 1<=spec['budget_seconds']<=300:raise ValueError('Invalid integration budget')
 if not isinstance(spec['image'],str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',spec['image']):raise ValueError('Immutable fixture image required')
 paths={k:Path(spec[k]) for k in ['prepared','bridge']}
 if any(not p.is_absolute() for p in paths.values()):raise ValueError('Absolute input paths required')
 paths.update({'runtime:'+n:ROOT/n for n in RUNTIME})
 if {k:file_sha(p) for k,p in paths.items()}!=spec['fingerprints']:raise ValueError('Launch fingerprint mismatch')
 prepared=json.loads(paths['prepared'].read_text())
 p,_=validate_effect(prepared,{ADAPTER:IdentityEffectAdapter(lambda:None)})
 if p['public_start']['max_calls']>48:raise ValueError('Integration call ceiling')
 return p

def execute_native(spec,prepared,out,lock):
 from environment import Graph,Fixture
 from run_public_breadth import candidate
 src=out/'source';src.mkdir()
 for name in RUNTIME:shutil.copy2(ROOT/name,src/name)
 shutil.copy2(spec['bridge'],out/'bridge');binary=out/'bridge'
 write(out/'plan.json',dict(created_at=now(),scope='New shared-driver integration using retained input; no new-world credit',hashes={str(p):file_sha(p) for p in [*src.iterdir(),binary,out/'prepared.json']},image=spec['image'],max_phases=3,max_calls=48,budget_seconds=spec['budget_seconds']))
 def status(stage):write(out/'status.json',dict(stage=stage,updated_at=now(),pid=os.getpid()))
 with Path(lock).open('a+') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if shutil.disk_usage(out).free<5*2**30:raise RuntimeError('Disk floor')
  def stop(*_):raise TimeoutError('Integration budget exhausted')
  signal.signal(signal.SIGALRM,stop);signal.alarm(spec['budget_seconds']);status('running')
  try:
   def open_graph(phase):
    dest=out/phase;dest.mkdir();return Graph(binary,out/'graph',dest/'engine.log')
   def run_candidate(phase,port):return candidate(spec['image'],src/'stratified_action_client.py',port,f'akh-effect-integrate-{os.getpid()}-{phase}')
   adapter=IdentityEffectAdapter(lambda:Fixture(spec['image'],f'akh-effect-fixture-{os.getpid()}',commands=['sana'],cli=src/'stratified_effect_cli.py'))
   result=dispatch_effect(prepared,{ADAPTER:adapter},open_graph,run_candidate);write(out/'dispatch.json',result)
   passed=result['all_phases_completed'];write(out/'summary.json',dict(all_passed=passed,phases=len(result['rows']),verification=[r['effect_verification'] for r in result['rows']],model_used=False,new_world_credit=0));status('completed' if passed else 'failed')
   if not passed:raise RuntimeError('Shared-driver integration failed')
  except BaseException:status('failed');raise
  finally:signal.alarm(0)

def launch(spec,out,lock,execute=False,runner=execute_native):
 prepared=preflight(spec)
 out=Path(out);out.mkdir(exist_ok=False);p=out/'prepared.json'
 with p.open('x') as f:p.chmod(0o600);json.dump(prepared,f,indent=2)
 write(out/'launch.json',spec)
 if execute:runner(spec,prepared,out,lock)
 result=dict(preflight_passed=True,execution_requested=execute,R1_ready=False);write(out/'preflight.json',result);return result

def main():
 ap=argparse.ArgumentParser()
 for n in ['manifest','out','lock']:ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--execute',action='store_true');a=ap.parse_args();launch(json.loads(a.manifest.read_text()),a.out,a.lock,a.execute)
if __name__=='__main__':main()
