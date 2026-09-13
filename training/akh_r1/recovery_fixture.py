"""In-process retained-state fixture; no production crash durability claim."""
import copy
class RecoveryFixture:
 def __init__(self,episode,state=None):
  self.episode=copy.deepcopy(episode);self.world=self.episode['examiner']['world'];self.state=copy.deepcopy(state) if state is not None else dict(stage='initial',active=[],log=[],generation=1,pending_compensation=[])
 def observe(self):
  phase={'initial':'initial_goal','disconnected':'partial_execution','restarted':'cold_restart'}[self.state['stage']]
  return copy.deepcopy([e for e in self.episode['delivery_events'] if e['after']==phase])
 def call(self,action,node):
  if node not in self.world['nodes']:raise ValueError('Unknown operation')
  if self.state['stage']=='disconnected':raise ValueError('Restart required')
  if action=='inspect':return dict(operation=node,committed=node in self.state['active'],receipts=copy.deepcopy([e for e in self.state['log'] if e['operation']==node]))
  if action=='compensate':
   if self.state['stage']!='restarted' or node not in self.state['pending_compensation']:raise ValueError('No compensation authority')
   if any(a==node and b in self.state['active'] for a,b in self.world['edges']):raise ValueError('Dependent remains committed')
   self.state['active'].remove(node);self.state['pending_compensation'].remove(node);self.state['log'].append(dict(action='compensate',operation=node,generation=1));return dict(compensated=True)
  if action!='apply':raise ValueError('Unsupported action')
  if self.state['pending_compensation']:raise ValueError('Compensation incomplete')
  if node in self.state['active']:return dict(already_committed=True) # Idempotent: no new effect/log.
  if any(b==node and a not in self.state['active'] for a,b in self.world['edges']):raise ValueError('Missing prerequisite')
  self.state['active'].append(node);self.state['log'].append(dict(action='apply',operation=node,generation=self.state['generation']))
  if self.state['stage']=='initial' and node==self.world['interrupted']:
   self.state['stage']='disconnected';raise ConnectionError('Commit happened; acknowledgement lost')
  return dict(committed=True)
 def restart(self):
  if self.state['stage']!='disconnected':raise ValueError('No interruption')
  restored=RecoveryFixture(self.episode,self.state);restored.state['stage']='restarted'
  if self.episode['examiner']['require_compensation']:
   restored.state['pending_compensation']=list(restored.state['active']);restored.state['generation']=2
  return restored
