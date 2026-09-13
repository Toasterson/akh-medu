"""Six frozen native sessions; one host reference policy, no neural candidate."""
import copy,fcntl,hashlib,json,os,shutil,signal,sys
from pathlib import Path
from environment import Graph,Fixture
from ownership_worlds import DistributionOwnershipWorld
from integrated_breadth.authority import AuthorityAdapter
from integrated_breadth.score import score
from neutral_transport import bind,reference
from run_public_breadth import write
from build_curriculum import now

def main(root):
 p=json.loads((root/'plan.json').read_text());rows=[];case=p['case']
 for path,h in p['hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 with Path(p['lock']).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if shutil.disk_usage(root).free<5*2**30:raise ValueError('Disk floor')
  (root/'consumed').open('x').close()
  def stop(*_):raise TimeoutError('300-second budget')
  signal.signal(signal.SIGALRM,stop);signal.alarm(300)
  def status(stage,**extra):write(root/'status.json',dict(stage=stage,updated_at=now(),phases=len(rows),**extra))
  try:
   status('running')
   for condition in ['reference','missing_rights','missing_skill']:
    fixture=Fixture(p['image'],f'akh-neutral-{os.getpid()}',commands=['sana'],cli=root/'source/command_semantics_cli.py');retained={};world=DistributionOwnershipWorld(case['participant'])
    try:
     fixture.reset(case['fixture'])
     for phase in ['acquire','cold_new_fault']:
      state=fixture.state();state['healthy']=False;state.pop('ticket',None);state.pop('issued_receipt',None);fixture.reset(state);before=fixture.state();adapter=AuthorityAdapter(world,fixture)
      receipt_keys=[('inquire_rights',None),('inspect_feasibility',None)]+[('inquire',h) for h in case['participant']['public_start']['participants']]
      if phase=='acquire' and condition!='missing_skill':receipt_keys.append(('command_help','sana'))
      if condition=='missing_rights':receipt_keys=[x for x in receipt_keys if x[0]!='inquire_rights']
      receipts={str(i):dict(request=dict(action=a,value=v),response={}) for i,(a,v) in enumerate(receipt_keys)}
      prepared=dict(adapter='neutral-lifecycle-v1',executed=False,public_start=p['public_start'],host_config=dict(observations=dict(documents={},receipts=receipts,memory={} if condition=='missing_skill' else retained)))
      dest=root/(condition+'-'+phase);dest.mkdir();graph=Graph(p['bridge'],dest/'graph',dest/'engine.log')
      try:
       def observe(a,v):return adapter.terminal([['sana','--help']]) if a=='command_help' else adapter.interaction(a,v)
       port=bind(prepared,lambda:graph,adapter.terminal,adapter.interaction,observe);reference(port);after=fixture.state();export=graph.call('exam_export')
       expected=dict(effect=condition=='reference',target=case['requested_target'],epoch=1,holders=case['participant']['examiner']['initial']['holders'],version='1',argv=['sana','repair',case['requested_target'],'--ticket',after.get('ticket','')])
       passed=score(before,after,port.events,expected,adapter.permits,adapter.returned) and port.finished['status']==('complete' if expected['effect'] else 'blocked')
       skill=[t for t in export.get('triples',[]) if t['subject_label']=='sana-skill' and t['predicate_label']=='document']
       passed=passed and export.get('success') and len(skill)==(0 if condition=='missing_skill' else 1)
       for e in port.events:
        if e['request']['tool']=='revise':
         a=e['request']['arguments'];retained={'skill':dict(source=a['source'],facts=[a['fact']])}
       write(dest/'evidence.json',dict(public_start=port.start(),events=port.events,before=before,after=after,graph=export,graph_pid=graph.process.pid,prepared=prepared,expected=expected,permits=adapter.permits,returned=adapter.returned,passed=bool(passed)))
       rows.append(dict(condition=condition,phase=phase,passed=bool(passed),calls=port.calls));status('running')
       if not passed:raise AssertionError('Independent score failed '+condition+'-'+phase)
      finally:graph.close()
    finally:fixture.close()
   write(root/'results.json',dict(stage='completed',rows=rows,phases=6,allowance='consumed_closed',host_authored_reference=True,model_used=False,new_world_credit=0));status('completed')
  except BaseException as e:
   write(root/'results.json',dict(stage='failed',rows=rows,error=str(e),allowance='consumed_no_retry',new_world_credit=0));status('failed',error=str(e));raise
  finally:signal.alarm(0)
if __name__=='__main__':main(Path(sys.argv[1]))
