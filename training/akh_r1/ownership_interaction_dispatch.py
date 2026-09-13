"""Two-stage neutral ownership transport; hidden change remains proposal-driven."""
import copy
from inventory_ownership_adapter import Session,ADAPTER
from public_episode import PublicEpisode
from neutral_lifecycle import reject_private

def dispatch(prepared,open_graph,run_candidate,*,session_factory=Session,port_factory=PublicEpisode):
 p=copy.deepcopy(prepared);h=p['host_config'];start=p['public_start']
 if p.get('schema')!='akh-ownership-interaction-preparation-v1' or p.get('adapter')!=ADAPTER:raise ValueError('Unsupported ownership adapter')
 if h['phases']!=['acquire','cold_proposal_repair'] or h['correction_trigger']!='after_observed_proposal' or h['retain_participant_state'] is not True:raise ValueError('Unsupported lifecycle')
 if type(start['max_calls']) is not int or not 1<=start['max_calls']<=48:raise ValueError('Call budget')
 reject_private(start['goal']);session=session_factory(p);rows=[];pids=set()
 for stage in h['phases']:
  graph=open_graph(stage)
  try:
   if graph.process.pid in pids:raise ValueError('Fresh graph process required')
   pids.add(graph.process.pid);boundary=len(session.world.log)
   def interaction(action,value):
    if stage=='acquire' and action not in ('epoch_witness','inquire_rights','inquire','inspect_feasibility'):
     return dict(success=False,error='Acquisition supports observations only')
    return session.interaction(action,value)
   port=port_factory(graph,copy.deepcopy(start['goal']),{},max_calls=start['max_calls'],interaction=interaction)
   result=run_candidate(port)
   if port.finished is None or result.get('finished')!=port.finished:raise ValueError('Missing public finish receipt')
   exported=graph.call('exam_export')
   if exported.get('success') is not True:raise ValueError('Graph export failed')
   if stage=='acquire':verification=dict(passed=session.acquisition_verified(port.finished))
   else:
    verification=session.final_verification(port.finished);log=session.world.log[boundary:]
    proposals=[i for i,e in enumerate(log) if e['action']=='propose' and e['response'].get('success')]
    pre=log[:proposals[0]] if len(proposals)==1 else []
    reuse=len(proposals)==1 and any(e['action']=='epoch_witness' and e['response'].get('success') for e in pre) and not any(e['action'] in ('inquire','inquire_rights','inspect_feasibility') for e in pre)
    verification=dict(verification,passed=verification.get('behavior_passed') is True and reuse,cold_reuse_verified=reuse)
   rows.append(dict(stage=stage,engine_pid=graph.process.pid,public_start=port.start(),result=copy.deepcopy(result),trace=copy.deepcopy(port.events),participant_log=copy.deepcopy(session.world.log),participant_state=copy.deepcopy(session.world.state()),boundary=boundary,graph_after=exported,verification=verification))
   if verification['passed'] is not True:break
  finally:graph.close()
 return dict(stage='completed' if len(rows)==2 and all(r['verification']['passed'] for r in rows) else 'failed',rows=rows,participant_simulation=True,qualified_credit=0,R1_ready=False)
