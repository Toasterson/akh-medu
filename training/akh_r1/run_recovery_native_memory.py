"""One frozen fixture, two native graph sessions; policy remains host-authored."""
import argparse,fcntl,json,os,shutil,signal
from pathlib import Path
from environment import Graph
from public_episode import PublicEpisode
from recovery_fixture import RecoveryFixture
from recovery_policy import act
from recovery_memory import documents,acquire,recall,save_fixture,restore_fixture
from recovery_effect_score import score
from episode_pack import file_sha
from datetime import datetime,timezone
def now():return datetime.now(timezone.utc).isoformat()
RUNTIME=['run_recovery_native_memory.py','environment.py','public_episode.py','recovery_fixture.py','recovery_policy.py','recovery_memory.py','recovery_effect_score.py','episode_pack.py','r0_contract.py','safe_pipeline.py']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--lock',type=Path,required=True);a=ap.parse_args();r=a.run;plan=json.loads((r/'plan.json').read_text())
 if any(file_sha(p)!=h for p,h in plan['hashes'].items()):raise ValueError('Frozen dependency mismatch')
 if plan['max_sessions']!=2 or not 0<plan['max_seconds']<=180:raise ValueError('Invalid allowance')
 def write(name,d):(r/name).write_text(json.dumps(d,indent=2)+'\n')
 with a.lock.open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if shutil.disk_usage(r).free<5*2**30:raise RuntimeError('Disk floor')
  with (r/'consumed.json').open('x') as f:json.dump(dict(at=now(),max_sessions=2),f)
  def stop(*_):raise TimeoutError('Native integration budget')
  signal.signal(signal.SIGALRM,stop);signal.alarm(plan['max_seconds']);write('status.json',dict(stage='running',updated_at=now()))
  try:
   episode=json.loads((r/'episode.json').read_text());fixture=RecoveryFixture(episode);pids=[];traces=[];calls=[]
   def call(action,node):
    try:response=fixture.call(action,node);calls.append(dict(action=action,node=node,response=response));return response
    except (ValueError,ConnectionError) as e:calls.append(dict(action=action,node=node,error=type(e).__name__));raise
   g=Graph(r/'bridge',r/'graph',r/'acquire.log');pids.append(g.process.pid)
   try:
    initial=fixture.observe();port=PublicEpisode(g,episode['public_start'],documents(initial),max_calls=32);memory=acquire(port)
    try:act(initial,memory,call)
    except ConnectionError:pass
    if fixture.state['stage']!='disconnected':raise RuntimeError('No interruption')
    snapshot_hash=save_fixture(r/'fixture.json',episode,fixture);traces.append(port.events);write('acquire-export.json',g.call('exam_export'))
   finally:g.close()
   del memory,initial,port,g
   fixture=restore_fixture(r/'fixture.json',episode,snapshot_hash)
   g=Graph(r/'bridge',r/'graph',r/'cold.log');pids.append(g.process.pid)
   try:
    port=PublicEpisode(g,episode['public_start'],{},max_calls=32);memory=recall(port);result=act(fixture.observe(),memory,call);traces.append(port.events);exported=g.call('exam_export');write('cold-export.json',exported)
   finally:g.close()
   oracle=score(episode,fixture.state)
   checks=dict(oracle=oracle['passed'],completed=result['status']=='complete',fresh_graph_pid=pids[0]!=pids[1],cold_no_document_reads=all(e['request']['tool']!='read_document' for e in traces[1]),native_receipts=any(t['predicate_label']=='r0_public_receipt' for t in exported['triples']),hashes_unchanged=all(file_sha(p)==h for p,h in plan['hashes'].items()))
   write('evidence.json',dict(pids=pids,public_traces=traces,fixture_calls=calls,state=fixture.state,oracle=oracle,result=result,snapshot_sha256=snapshot_hash,checks=checks))
   write('summary.json',dict(all_passed=all(checks.values()),checks=checks,native_graph_sessions=2,new_world_credit=0,policy_in_host=True,fixture_in_process=True))
   if not all(checks.values()):raise RuntimeError('Integration checks failed')
   write('status.json',dict(stage='completed',updated_at=now()))
  except BaseException:write('status.json',dict(stage='failed',updated_at=now()));raise
  finally:signal.alarm(0)
if __name__=='__main__':main()
