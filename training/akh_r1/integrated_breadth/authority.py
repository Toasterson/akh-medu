"""Public composition adapter; participant authority is separate from capability.

Uses the existing participant simulator and fixture transport. It never reads an
examiner expected target. This serialized guard is not production atomicity.
"""
import copy,json
from ownership_worlds import DistributionOwnershipWorld,generate_distribution
from public_episode import PublicEpisode
from safe_pipeline import validate_pipeline


def cases():
 out=[]
 for split,seed,name in [('development',9134151,'capability_version'),('validation',9135151,'rights_revocation_during_capability')]:
  participant=generate_distribution(split,seed)
  # Use the generated feasible initial option as an explicit requested task.
  # This is supplied intent, not a target-discovery claim.
  initial=participant['examiner']['initial'];target=next(x for x in initial['feasible'] if all(x in initial['intents'][h] for h in initial['holders']))
  out.append(dict(id=name,split=split,participant=participant,requested_target=target,
   fixture=dict(protocol={'kind':'conditional-v1'},help_available=True,version='1',target=target,
    modes={'repair':dict(argv=['sana','repair','{target}','--ticket','{ticket}'],requires_receipt=True)},required_mode='repair',healthy=False,repairs=0,audit=[]),
   conditions=['reference','missing_rights','stale_permission'],
   correction=dict(trigger='after_successful_capability_acquisition',kind='source_version' if split=='development' else 'participant_proposal'),
   assistance=['Authored mode and requested task','Typed participant observations','Serialized executor authority guard'],qualified=False,new_world_credit=0))
 return out


class AuthorityAdapter:
 def __init__(self,world,fixture):
  self.world=world;self.fixture=fixture;self.rights=None;self.intents={};self.feasibility=None;self.permits=[];self.verification=None;self.returned=[];self.pending=None
 def schedule_correction(self,kind):
  if self.pending is not None:raise ValueError('Correction already pending')
  if kind not in ('source_version','participant_proposal'):raise ValueError('Unknown event')
  self.pending=kind
 def current_epoch(self):return self.world.call('epoch_witness',None)['epoch']
 def interaction(self,action,value):
  if action=='return_effect_ownership':
   witness=self.world.call('inquire_rights',None)
   if not self.verification or self.verification['epoch']!=witness['epoch'] or not isinstance(value,list) or len(value)!=len(set(value)) or set(value)!=set(witness['holders']):return dict(success=False,error='Fresh verified effect and current holders required')
   self.returned=value[:];return dict(success=True,decision_holders=value[:],epoch=witness['epoch'])
  r=self.world.call(action,value)
  if r.get('success'):
   if action=='inquire_rights':self.rights=copy.deepcopy(r)
   elif action=='inquire':self.intents[value]=copy.deepcopy(r)
   elif action=='inspect_feasibility':self.feasibility=copy.deepcopy(r)
  return r
 def permission(self,target):
  epoch=self.current_epoch();rights=self.rights
  if not rights or rights['epoch']!=epoch or not rights['holders']:return False
  if not self.feasibility or self.feasibility['epoch']!=epoch or target not in self.feasibility['options']:return False
  for h in rights['holders']:
   r=self.intents.get(h)
   if not r or r['epoch']!=epoch or target not in r['options']:return False
  return True
 def terminal(self,stages):
  try:validate_pipeline(stages,'sana',32)
  except (ValueError,TypeError,KeyError) as error:return dict(success=False,error=str(error))
  if len(stages)!=1:return dict(success=False,error='Integrated authority requires one command per receipt')
  argv=stages[0];state=self.fixture.state();target=state['target']
  observation=argv in [['sana','--help'],['sana','--version'],['sana','inspect',target],['sana','status',target]]
  if not observation and not self.permission(target):return dict(success=False,error='No current observed participant authority')
  epoch=self.current_epoch();r=self.fixture.pipeline(stages)
  if not r.get('success'):return r
  if not observation:
   self.permits.append(dict(epoch=epoch,holders=self.rights['holders'][:],target=target,argv=argv[:]))
   self.verification=None
  if argv==['sana','status',target] and self.permits and json.loads(r['stdout']).get('healthy'):
   permit=self.permits[-1]
   if permit['epoch']==epoch and permit['target']==target:self.verification=dict(epoch=epoch,target=target)
  if argv==['sana','inspect',target] and self.pending:
   event=self.pending;self.pending=None
   if event=='source_version':
    changed=self.fixture.state();changed['version']=str(changed['version'])+'-revised';self.fixture.reset(changed)
    notice='Command source version changed; obtain current capability'
   else:
    changed=self.world.call('propose',target)
    if not changed.get('success'):raise ValueError('Private correction prerequisite not reached')
    notice=changed['correction']
   r=dict(r,change_notice=notice)
  return r
 def public_port(self,graph,goal):
  return PublicEpisode(graph,dict(goal,command='sana',command_profile='conditional-v1',interaction_profile='ownership-native-v1'),{},self.terminal,max_calls=48,interaction=self.interaction)


def build(case,fixture):return AuthorityAdapter(DistributionOwnershipWorld(case['participant']),fixture)
