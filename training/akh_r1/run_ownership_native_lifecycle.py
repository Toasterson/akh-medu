"""Native graph/candidate restart; retained participant simulation is explicit."""
import fcntl,hashlib,json,shutil,signal,sys,os
from pathlib import Path
from ownership_worlds import DistributionOwnershipWorld
from ownership_score import score_distribution
from environment import Graph
from public_episode import PublicEpisode
from run_public_breadth import candidate,write
from build_curriculum import now

def main(root):
 p=json.loads((root/'plan.json').read_text());rows=[]
 for f,h in p['hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 with Path(p['lock']).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if shutil.disk_usage(root).free<5*2**30:raise ValueError('Disk floor')
  (root/'consumed').open('x').close()
  def stop(*_):raise TimeoutError('600 second ownership budget')
  signal.signal(signal.SIGALRM,stop);signal.alarm(600)
  def status(stage,**extra):write(root/'status.json',dict(stage=stage,updated_at=now(),phases=len(rows),**extra))
  status('running')
  try:
   for case in p['cases']:
    for condition in ['reference','ignore_correction','missing_feasibility']:
     identity=case['split']+'-'+condition;world=DistributionOwnershipWorld(case)
     for phase in ['acquire','cold']:
      dest=root/(identity+'-'+phase);dest.mkdir();g=Graph(p['bridge'],root/(identity+'-graph'),dest/'engine.log');boundary=len(world.log)
      try:
       goal=dict(world.start(),phase=phase,condition=condition,interaction_profile='ownership-native-v1')
       port=PublicEpisode(g,goal,{},max_calls=48,interaction=world.call)
       result=candidate(p['image'],root/'source/ownership_distribution_client.py',port,f'akh-ownership-native-{os.getpid()}')
       export=g.call('exam_export');state=world.state();scored=score_distribution(case,world.log,state,port.finished)
       if phase=='acquire':passed=port.finished['status']=='complete' and state==dict(stage='initial',effects=[],returned=[])
       elif condition=='reference':passed=scored['behavior_passed']
       else:passed=not scored['behavior_passed'] and port.finished['status']=='blocked' and not state['effects'] and not state['returned']
       stored=[t for t in export.get('triples',[]) if t['predicate_label']=='observation'];passed=passed and export.get('success') and len(stored)>=3
       if phase=='cold':
        newlog=world.log[boundary:];proposals=[i for i,e in enumerate(newlog) if e['action']=='propose' and e['response'].get('success')]
        passed=passed and len(proposals)==1 and any(e['action']=='epoch_witness' for e in newlog[:proposals[0]]) and not any(e['action'] in ['inquire','inquire_rights','inspect_feasibility'] for e in newlog[:proposals[0]])
       write(dest/'evidence.json',dict(result=result,events=port.events,interaction_log=world.log,boundary=boundary,state=state,graph=export,graph_pid=g.process.pid,score=scored,passed=bool(passed)))
       rows.append(dict(split=case['split'],condition=condition,phase=phase,calls=port.calls,passed=bool(passed)));status('running')
       if not passed:raise AssertionError('Independent ownership score failed '+identity+'-'+phase)
      finally:g.close()
   write(root/'results.json',dict(stage='completed',rows=rows,phases=12,allowance='consumed_closed',participant_simulation=True,native_graph=True,model_used=False,new_world_credit=0));status('completed')
  except BaseException as error:
   write(root/'results.json',dict(stage='failed',rows=rows,error=str(error),allowance='consumed_no_retry',new_world_credit=0));status('failed',error=str(error));raise
  finally:signal.alarm(0)
if __name__=='__main__':main(Path(sys.argv[1]))
