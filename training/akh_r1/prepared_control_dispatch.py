"""Single complete control episodes; injected resources, no implicit native launch.

Ownership correction occurs through observed interaction, not a synthesized cold
phase. Host expectations remain separate from the public port.
"""
import copy,json
from public_episode import PublicEpisode
from ownership_worlds import OwnershipWorld
from ownership_score import score

class LanguageControl:
 def __init__(self,factory):self.factory=factory
 def validate(self,p):
  h=p['host_config']
  if h['goal']!=p['public_start']['goal'] or not isinstance(h['documents'],dict):raise ValueError('Language goal/documents mismatch')
  if h['expected_target'] not in ('nodealpha','nodebeta','nodegamma') or type(h['expected_effect']) is not bool:raise ValueError('Invalid language expectation')
 def create(self,p):
  h=p['host_config'];f=self.factory()
  try:f.reset(dict(authorized_target=h['expected_target'],nodes={n:dict(healthy=False,repairs=0) for n in ['nodealpha','nodebeta','nodegamma']},audit=[]))
  except BaseException:f.close();raise
  return f
 def observation(self,p,r):return copy.deepcopy(p['host_config']['documents']),r.pipeline,None
 def verify(self,p,before,after,trace,finished,r):
  h=p['host_config'];target=h['expected_target'];expected=copy.deepcopy(before)
  if h['expected_effect']:
   expected['nodes'][target]['healthy']=True;expected['nodes'][target]['repairs']+=1
   expected['audit'].append(dict(action='repair',target=target))
  terminals=[(i,e) for i,e in enumerate(trace) if e.get('request',{}).get('tool')=='terminal']
  applied=[i for i,e in terminals if e['request'].get('arguments',{}).get('stages')==[['sana','apply',target]] and e.get('response',{}).get('success') is True]
  fresh=False
  for i,e in terminals:
   if e['request'].get('arguments',{}).get('stages')!=[['sana','inspect',target]] or e.get('response',{}).get('success') is not True:continue
   try:observed=json.loads(e['response'].get('stdout',''))
   except (ValueError,TypeError):continue
   if applied and i>max(applied) and observed==dict(target=target,eligible=True,**expected['nodes'][target]):fresh=True
  checks=dict(exact_effect=after==expected,status=finished['status']==('complete' if h['expected_effect'] else 'blocked'),fresh_observation=fresh if h['expected_effect'] else not terminals)
  return dict(passed=all(checks.values()),checks=checks,scope='Retained host effect contract; authored language interpretation')

class OwnershipResource(OwnershipWorld):
 def close(self):pass
class OwnershipControl:
 def validate(self,p):
  h=p['host_config'];c=h['initial_world']['case']
  if c['mechanism'] not in ('owner_correction','joint_veto') or c['mode']!='responsive' or c['final_target'] not in ('trial-a','trial-b'):raise ValueError('Unsupported ownership control')
  world=OwnershipWorld(c)
  if p['public_start']['goal']!=world.start():raise ValueError('Ownership goal mismatch')
  if h['expected_effects']!=dict(performed=[c['final_target']],returned=world.participants):raise ValueError('Ownership expectation mismatch')
  if h['events']!=[dict(after_action='propose',payload=dict(final_target=c['final_target'],requires_reinquiry=world.participants))]:raise ValueError('Ownership schedule mismatch')
 def create(self,p):return OwnershipResource(p['host_config']['initial_world']['case'])
 def observation(self,p,r):return {},None,r.call
 def verify(self,p,before,after,trace,finished,r):
  result=score(p['host_config']['initial_world']['case'],copy.deepcopy(r.log),after,finished)
  return dict(passed=result['behavior_passed'],behavior=result,scope='Simulated participant action and ownership; not learned emotional intelligence')

def dispatch_control(prepared,adapters,open_graph,run_candidate):
 p=copy.deepcopy(prepared)
 if p.get('schema')!='akh-pack-preparation-v1' or p.get('adapter') not in adapters:raise ValueError('Unsupported control adapter')
 start=p['public_start'];calls=start['max_calls']
 if type(calls) is not int or not 1<=calls<=128:raise ValueError('Invalid call budget')
 adapter=adapters[p['adapter']];adapter.validate(p)
 resource=adapter.create(p)
 try:
  before=copy.deepcopy(resource.state());graph=open_graph()
  try:
   docs,terminal,interaction=adapter.observation(p,resource)
   port=PublicEpisode(graph,copy.deepcopy(start['goal']),docs,terminal,max_calls=calls,interaction=interaction)
   result=run_candidate(port);finished=result.get('finished')
   if not isinstance(finished,dict) or port.finished is None or finished!=port.finished:raise ValueError('Completion lacks public receipt')
   exported=graph.call('exam_export')
   if not exported.get('success'):raise RuntimeError('Graph export failed')
   after=copy.deepcopy(resource.state());trace=copy.deepcopy(port.events)
   verification=adapter.verify(p,before,after,trace,finished,resource)
   if type(verification.get('passed')) is not bool:raise ValueError('Explicit verification required')
   return dict(schema='akh-prepared-control-dispatch-v1',result=copy.deepcopy(result),trace=trace,before=before,after=after,graph_after=exported,verification=verification,assistance=p['assistance'],R1_ready=False,new_world_credit=0)
  finally:graph.close()
 finally:resource.close()
