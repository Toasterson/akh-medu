"""Bound single control launch. Default is preflight; execution needs allowance."""
import argparse,fcntl,json,os,re,shutil,signal
from pathlib import Path
from control_binding import bind,COMMON,FAMILY,ROOT
from perception_assistance import verified_reference
from prepared_control_dispatch import dispatch_control,LanguageControl,OwnershipControl
from run_public_breadth import write
from build_curriculum import now
from episode_pack import file_sha

def preflight(spec,execute=False):
 if set(spec)!={'schema','pack','episode_id','adapter','binding','bridge','image','launcher','allowance'} or spec['schema']!='akh-control-launch-v1':raise ValueError('Invalid control launch')
 if not re.fullmatch(r'sha256:[0-9a-f]{64}',spec['image']):raise ValueError('Immutable fixture required')
 verified_reference(spec['bridge'])
 if verified_reference(spec['launcher']).resolve()!=Path(__file__).resolve():raise ValueError('Wrong launcher')
 if not Path(spec['pack']).is_absolute():raise ValueError('Absolute pack required')
 if spec['adapter']=='spatial-perception-v1':
  from perception_binding import bind as bind_perception
  prepared=bind_perception(spec['binding'])
  if Path(spec['pack']).resolve()!=Path(spec['binding']['source']).resolve() or spec['episode_id']!=prepared['episode_id']:raise ValueError('Perception identity mismatch')
 else:prepared=bind(spec['pack'],spec['episode_id'],spec['adapter'],spec['binding'])
 allowance=spec['allowance']
 if allowance is not None:
  if set(allowance)!={'id','max_episodes','max_seconds','max_calls'} or not isinstance(allowance['id'],str) or not allowance['id']:raise ValueError('Invalid allowance')
  for key,ceiling in [('max_episodes',1),('max_seconds',300),('max_calls',48 if spec['adapter']=='spatial-perception-v1' else 32)]:
   if type(allowance[key]) is not int or not 1<=allowance[key]<=ceiling:raise ValueError('Allowance ceiling')
  if prepared['public_start']['max_calls']>allowance['max_calls']:raise ValueError('Call allowance exceeded')
 if execute and allowance is None:raise ValueError('Separate frozen integration allowance required')
 return prepared

def execute_native(spec,prepared,out):
 from environment import Graph,Fixture
 from run_public_breadth import candidate
 src=out/'source';src.mkdir()
 if spec['adapter']=='spatial-perception-v1':
  from perception_binding import RUNTIME
  names=RUNTIME+['control_native_launch.py']
 else:names=COMMON+FAMILY[spec['adapter']]+['control_native_launch.py']
 for name in names:shutil.copy2(ROOT/name,src/name)
 shutil.copy2(spec['bridge']['path'],out/'bridge')
 write(out/'runtime.json',dict(hashes={str(p):file_sha(p) for p in [*src.iterdir(),out/'bridge']},image=spec['image']))
 adapter=spec['adapter'];client='language_composition_client.py' if adapter=='language-v1' else 'ownership_client.py'
 registry={'language-v1':LanguageControl(lambda:Fixture(spec['image'],f'akh-control-{os.getpid()}',commands=['sana'],cli=src/'graph_effect_cli.py')),'ownership-v1':OwnershipControl()}
 if adapter=='spatial-perception-v1':
  from perception_control import PerceptionControl
  client='perception_effect_client.py'
  registry[adapter]=PerceptionControl(lambda:Fixture(spec['image'],f'akh-control-{os.getpid()}',commands=['sana'],cli=src/'perception_effect_cli.py'))
 result=dispatch_control(prepared,registry,lambda:Graph(out/'bridge',out/'graph',out/'engine.log'),lambda port:candidate(spec['image'],src/client,port,f'akh-control-client-{os.getpid()}'))
 write(out/'result.json',result)
 if not result['verification']['passed']:raise RuntimeError('Control verification failed')

def launch(spec,out,lock,execute=False,runner=execute_native):
 prepared=preflight(spec,execute)
 # A single frozen allowance may not be spent twice at different output paths.
 with Path(lock).open('a+') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  out=Path(out)
  if execute:
   if shutil.disk_usage(out.parent).free<5*2**30:raise RuntimeError('Disk floor')
   import hashlib
   token=hashlib.sha256(spec['allowance']['id'].encode()).hexdigest()
   claim=Path(str(lock)+'.allowance-'+token)
   with claim.open('x') as f:json.dump(dict(out=str(out.resolve()),claimed_at=now(),allowance=spec['allowance']),f)
  out.mkdir(exist_ok=False)
  write(out/'launch.json',spec)
  with (out/'prepared.json').open('x') as f:os.chmod(out/'prepared.json',0o600);json.dump(prepared,f,indent=2)
  if not execute:
   write(out/'status.json',dict(stage='prepared',updated_at=now(),executed=False));return
  def status(stage):write(out/'status.json',dict(stage=stage,updated_at=now(),pid=os.getpid()))
  previous=signal.getsignal(signal.SIGALRM)
  def stop(*_):raise TimeoutError('Integration allowance exhausted')
  signal.signal(signal.SIGALRM,stop);signal.alarm(spec['allowance']['max_seconds']);status('running')
  try:runner(spec,prepared,out);status('completed')
  except BaseException:status('failed');raise
  finally:signal.alarm(0);signal.signal(signal.SIGALRM,previous)

def main():
 ap=argparse.ArgumentParser()
 for name in ['manifest','out','lock']:ap.add_argument('--'+name,type=Path,required=True)
 ap.add_argument('--execute',action='store_true');a=ap.parse_args();launch(json.loads(a.manifest.read_text()),a.out,a.lock,a.execute)
if __name__=='__main__':main()
