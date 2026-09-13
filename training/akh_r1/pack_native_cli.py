"""Validate trusted launch fingerprints before dispatching a retained graph pack.

Default is check-only. --execute explicitly starts the existing bounded runner.
"""
import argparse,json,re,subprocess,sys
from pathlib import Path
from episode_pack import file_sha,sha
from pack_preparation import prepare
ROOT=Path(__file__).parent
RUNTIME=['pack_native_cli.py','pack_preparation.py','language_pack.py','episode_pack.py','r0_contract.py','run_prepared_graph_native.py','prepared_graph_dispatch.py','public_episode.py','safe_pipeline.py','environment.py','run_public_breadth.py','build_curriculum.py','graph_query_client.py']
def preflight(spec):
 required={'schema','pack','episode_id','adapter','bridge','image','fingerprints','assistance_sha256'}
 if set(spec)!=required or spec['schema']!='akh-native-pack-launch-v1':raise ValueError('Invalid launch manifest')
 if spec['adapter']!='graph-query-v1':raise ValueError('Native graph adapter required')
 if not re.fullmatch(r'sha256:[0-9a-f]{64}',spec['image']):raise ValueError('Immutable image ID required')
 pack=Path(spec['pack']);bridge=Path(spec['bridge'])
 if not pack.is_absolute() or not bridge.is_absolute():raise ValueError('Absolute local inputs required')
 paths={'pack':pack/'manifest.json','registry':pack/'registry.json','bridge':bridge}
 paths.update({'runtime:'+n:ROOT/n for n in RUNTIME})
 actual={name:file_sha(path) for name,path in paths.items()}
 if actual!=spec['fingerprints']:raise ValueError('Launch artifact fingerprint mismatch')
 prepared=prepare(pack,spec['episode_id'],spec['adapter'])
 if sha(prepared['assistance'])!=spec['assistance_sha256']:raise ValueError('Assistance identity mismatch')
 return prepared

def launch(spec,out,lock,execute=False,run=subprocess.run):
 prepared=preflight(spec)  # No output or process before all checks pass.
 out=Path(out);out.mkdir(exist_ok=False)
 p=out/'prepared.json'
 with p.open('x') as f:
  p.chmod(0o600);json.dump(prepared,f,indent=2)
 (out/'launch.json').write_text(json.dumps(spec,indent=2))
 if execute:
  run([sys.executable,str(ROOT/'run_prepared_graph_native.py'),'--prepared',str(p),'--out',str(out/'execution'),'--bridge',spec['bridge'],'--lock',str(lock),'--image',spec['image']],check=True)
 result=dict(preflight_passed=True,execution_requested=execute,R1_ready=False)
 (out/'preflight.json').write_text(json.dumps(result,indent=2));return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args();launch(json.loads(a.manifest.read_text()),a.out,a.lock,a.execute)
if __name__=='__main__':main()
