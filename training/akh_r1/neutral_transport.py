"""Consume prepared observations at the public boundary, with isolated graph state.

Retained response delivery is explicit assistance. It is not a live observation
refresh or a production sandbox. No fallback can bypass a withheld receipt.
"""
import copy,json
from public_episode import PublicEpisode
from neutral_lifecycle import reject_private


def key(action,value):return json.dumps([action,value],sort_keys=True,separators=(',',':'))

class NeutralEpisode(PublicEpisode):
 def start(self):
  result=super().start();result['tools']=self.allowed_tools[:];return result
 def call(self,request):
  if request.get('tool') not in self.allowed_tools:
   # Route through the existing error/receipt machinery and charge the call.
   request=dict(tool='unavailable',arguments={})
  return super().call(request)


def bind(prepared,graph_factory,effect_terminal=None,interaction_effect=None,observation_callback=None):
 if prepared.get('adapter')!='neutral-lifecycle-v1' or prepared.get('executed') is not False:raise ValueError('Expected unexecuted neutral preparation')
 public=copy.deepcopy(prepared['public_start']);reject_private(public)
 view=copy.deepcopy(prepared['host_config']['observations']);reject_private(view)
 responses={}
 for record in view['receipts'].values():
  if set(record)!={'request','response'} or set(record['request'])!={'action','value'}:raise ValueError('Receipt needs exact request binding')
  if record['request']['action'] not in ('inquire_rights','inquire','inspect_feasibility','epoch_witness','command_help'):raise ValueError('Only observation receipts may be replayed')
  k=key(record['request']['action'],record['request']['value'])
  if k in responses:raise ValueError('Ambiguous receipt binding')
  responses[k]=copy.deepcopy(record['response'])
 def interaction(action,value):
  if action=='return_effect_ownership':
   if interaction_effect is None:return dict(success=False,error='Live ownership transition unavailable')
   return interaction_effect(action,value)
  if key(action,value) not in responses:return dict(success=False,error='Observation unavailable')
  if observation_callback is not None:return observation_callback(action,value)
  return copy.deepcopy(responses[key(action,value)])
 def terminal(stages):
  if stages==[[public['goal'].get('command'),'--help']]:
   return interaction('command_help',public['goal'].get('command'))
  if effect_terminal is None:return dict(success=False,error='Execution unavailable')
  # Alternate help routes and pipelines cannot bypass withheld help delivery.
  if len(stages)!=1 or any(a in ('--help','-h','help') for a in stages[0][1:]):return dict(success=False,error='Unsupported observation route')
  return effect_terminal(stages)
 # Validate all seed records before allocating or touching any graph.
 for record in view['memory'].values():
  if set(record)!={'source','facts'} or not isinstance(record['source'],str) or not record['source'] or not isinstance(record['facts'],list):raise ValueError('Retained memory needs source and facts')
  for fact in record['facts']:
   if set(fact)!={'subject','predicate','object'} or not all(isinstance(v,str) for v in fact.values()) or fact['predicate'].startswith('r0_'):raise ValueError('Invalid retained fact')
 graph=graph_factory()
 try:
  exported=graph.call('exam_export')
  if not exported.get('success') or exported.get('triples'):raise ValueError('Fresh empty graph required; retained graph must not be mutated')
  for record in view['memory'].values():
   r=graph.call('memory_write',dict(summary='Retained sourced observation',source=record['source'],facts=record['facts']))
   if not r.get('success'):raise ValueError('Memory initialization failed')
  port=NeutralEpisode(graph,public['goal'],view['documents'],terminal,max_calls=public['max_calls'],interaction=interaction)
  port.allowed_tools=public['tools'][:]
  return port
 except BaseException:
  if hasattr(graph,'close'):graph.close()
  raise


def reference(port):
 """Uniform command/authority reference over public observations, no host labels.

Supports the current typed conditional command schema only; other families use
this transport but do not yet have a shared candidate implementation.
 """
 def call(tool,**args):return port.call(dict(tool=tool,arguments=args))
 def cmd(argv):return call('terminal',stages=[argv])
 def observe():
  rights=call('interact',action='inquire_rights',value=None)
  if not rights.get('success'):raise ValueError('No rights observation')
  allowed=None
  for h in rights['holders']:
   r=call('interact',action='inquire',value=h)
   if not r.get('success') or r['epoch']!=rights['epoch']:raise ValueError('Missing or stale participant observation')
   allowed=set(r['options']) if allowed is None else allowed & set(r['options'])
  feasible=call('interact',action='inspect_feasibility',value=None)
  if not feasible.get('success') or feasible['epoch']!=rights['epoch']:raise ValueError('Missing or stale feasibility')
  return rights['holders'],allowed & set(feasible['options'])
 goal=port.start()['goal'];target=goal['target'];command=goal['command']
 try:
  help=cmd([command,'--help'])
  if help.get('success'):
   for f in help.get('observed_facts',[]):
    if not call('revise',source=help['evidence_id'],fact=f).get('success'):raise ValueError('Source write failed')
  r=call('query',subject='sana-skill');facts=[f for f in r.get('facts',[]) if f['predicate']=='document']
  if len(facts)!=1:raise ValueError('No unique retained skill')
  skill=json.loads(facts[0]['object']);version=cmd([command,'--version'])
  if not version.get('success') or skill['version']!=version['stdout'].strip():raise ValueError('Stale skill')
  holders,allowed=observe()
  if target not in allowed:raise ValueError('No current agreement')
  receipt=cmd([command,'inspect',target])
  if not receipt.get('success') or receipt.get('change_notice'):raise ValueError('Changed observation requires new decision')
  # Re-observe before the effect; a future live binding must deliver fresh results.
  holders,allowed=observe()
  if target not in allowed:raise ValueError('Current agreement lost')
  cap=json.loads(receipt['stdout']);bindings=dict(target=target,ticket=cap['ticket'])
  if cap['version']!=skill['version'] or cap['target']!=target:raise ValueError('Capability mismatch')
  argv=[bindings[x[1:-1]] if x.startswith('{') and x.endswith('}') else x for x in skill['modes']['repair']['argv']]
  if not cmd(argv).get('success'):raise ValueError('Effect denied')
  v=cmd([command,'status',target])
  if not v.get('success') or not json.loads(v['stdout'])['healthy']:raise ValueError('Unverified effect')
  if not call('interact',action='return_effect_ownership',value=holders).get('success'):raise ValueError('Ownership not returned')
  status='complete'
 except (ValueError,KeyError,TypeError):status='blocked'
 return call('finish',status=status,answer='Observed reference outcome')
