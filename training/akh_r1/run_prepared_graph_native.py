"""One bounded native smoke for the prepared dispatcher; no model invocation."""
import argparse,fcntl,json,os,shutil,signal
from pathlib import Path
from build_curriculum import now,digest
from environment import Graph
from prepared_graph_dispatch import dispatch
from run_public_breadth import candidate,write

def main():
 ap=argparse.ArgumentParser()
 for key in ['out','bridge','lock','prepared']:ap.add_argument('--'+key,type=Path,required=True)
 ap.add_argument('--image',required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);src=a.out/'source';src.mkdir()
 for n in ['run_prepared_graph_native.py','prepared_graph_dispatch.py','graph_query_client.py','public_episode.py','safe_pipeline.py','environment.py','run_public_breadth.py','build_curriculum.py']:shutil.copy2(Path(__file__).with_name(n),src/n)
 shutil.copy2(a.bridge,a.out/'bridge');shutil.copy2(a.prepared,a.out/'prepared.json');os.chmod(a.out/'prepared.json',0o600);binary=a.out/'bridge';prepared=json.loads((a.out/'prepared.json').read_text());hashes={str(p):digest(p) for p in [*src.iterdir(),binary,a.out/'prepared.json']}
 write(a.out/'plan.json',dict(created_at=now(),hashes=hashes,image=a.image,budget_seconds=300,model_used=False,new_world_credit=0,scope='Single new synthetic native transport smoke'))
 def status(stage):write(a.out/'status.json',dict(stage=stage,updated_at=now(),pid=os.getpid()))
 with a.lock.open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if shutil.disk_usage(a.out).free<5*2**30:raise RuntimeError('Disk floor')
  def stop(*_):raise TimeoutError('Budget exhausted')
  signal.signal(signal.SIGALRM,stop);signal.alarm(300);status('running')
  try:
   def open_graph(phase):
    dest=a.out/phase;dest.mkdir();return Graph(binary,a.out/'graph',dest/'engine.log')
   def run_candidate(phase,port):return candidate(a.image,src/'graph_query_client.py',port,f'akh-prepared-native-{os.getpid()}-{phase}')
   result=dispatch(prepared,open_graph,run_candidate);write(a.out/'dispatch.json',result)
   checks={}
   for row in result['rows']:
    phase=row['phase'];expected=prepared['host_config']['expected']['revised' if phase=='correction' else 'initial'];final=row['result']['finished'];checks[phase]=final['status']=='complete' and final['answer']['value']==expected
   checks['all_phases']=result['all_phases_completed'];checks['frozen']=all(digest(Path(p))==h for p,h in hashes.items());write(a.out/'summary.json',dict(checks=checks,all_passed=all(checks.values()),model_used=False,new_world_credit=0));status('completed' if all(checks.values()) else 'failed')
   if not all(checks.values()):raise RuntimeError('Native dispatch smoke failed')
  except BaseException:status('failed');raise
  finally:signal.alarm(0)
if __name__=='__main__':main()
