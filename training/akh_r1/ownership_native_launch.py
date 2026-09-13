"""Check-only default launcher for a frozen retained ownership pair."""
import hashlib,json,re,signal,shutil
from pathlib import Path
from inventory_ownership_adapter import Session,ADAPTER
from neutral_lifecycle import reject_private

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def preflight(path,pin):
 if sha(path)!=pin:raise ValueError('Manifest changed')
 m=json.loads(Path(path).read_text())
 if m.get('schema')!='neutral-ownership-launch-v1':raise ValueError('Wrong launch schema')
 if m['budget']!=dict(max_sessions=4,max_calls=48,max_seconds=300,retries=0):raise ValueError('Unsupported allowance')
 if not re.fullmatch('sha256:[0-9a-f]{64}',m['image']):raise ValueError('Immutable image required')
 required={str(p.resolve()) for p in Path(__file__).parent.glob('*.py')}|{m['bridge'],m['candidate']}
 if not required<=set(m['pins']):raise ValueError('Incomplete runtime closure')
 for p,h in m['pins'].items():
  if sha(p)!=h:raise ValueError('Changed dependency')
 if len(m['prepared'])!=2 or len({e['path'] for e in m['prepared']})!=2:raise ValueError('Retained pair required')
 rows=[]
 for e in m['prepared']:
  if sha(e['path'])!=e['sha256']:raise ValueError('Prepared input changed')
  p=json.loads(Path(e['path']).read_text())
  if p['adapter']!=ADAPTER or p['assistance']!=e['assistance']:raise ValueError('Adapter or assistance mismatch')
  if any(m['pins'].get(k)!=v for k,v in p['fingerprints'].items()):raise ValueError('Unbound preparation dependency')
  reject_private(p['public_start']['goal'])
  if p['public_start']['max_calls']!=48:raise ValueError('Call cap')
  Session(p);rows.append(p) # Pure host simulator construction, no interactions.
 if {p['episode_id'] for p in rows}!={'dojo_pressure-development','dojo_pressure-validation'}:raise ValueError('Wrong retained pair')
 if Path(m['output']).exists():raise ValueError('No retries or output reuse')
 if shutil.disk_usage(Path(path).parent).free<5*2**30:raise ValueError('Disk floor')
 return m,rows

def execute(path,pin):
 m,rows=preflight(path,pin)
 from environment import Graph
 from run_public_breadth import candidate
 from ownership_interaction_dispatch import dispatch
 out=Path(m['output']);out.mkdir(parents=True,exist_ok=False);status=dict(stage='running',native_sessions=0,completed_episodes=0,model_calls=0,new_world_credit=0)
 def save():
  p=out/'status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(status,indent=2));q.replace(p)
 def expired(*_):raise TimeoutError('Ownership allowance exhausted')
 old=signal.signal(signal.SIGALRM,expired);signal.alarm(300);save()
 try:
  for i,p in enumerate(rows):
   dest=out/str(i);dest.mkdir();calls=[0]
   def graph(stage):
    if status['native_sessions']>=4:raise RuntimeError('Session cap')
    status['native_sessions']+=1;save();return Graph(m['bridge'],dest/'graph',dest/'native.log')
   def client(port):
    calls[0]+=1;return candidate(m['image'],Path(m['candidate']),port,'akh-own-'+out.parent.name+'-'+str(i)+'-'+str(calls[0]))
   r=dispatch(p,graph,client);(dest/'results.json').write_text(json.dumps(r,indent=2))
   if r['stage']!='completed':raise RuntimeError('Ownership verification failed')
   status['completed_episodes']+=1;save()
  status['stage']='completed'
 except BaseException as e:status.update(stage='failed',error=str(e));raise
 finally:signal.alarm(0);signal.signal(signal.SIGALRM,old);save()
 return status
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();print(json.dumps(execute(a.manifest,a.sha256) if a.execute else dict(preflight=bool(preflight(a.manifest,a.sha256)))))
