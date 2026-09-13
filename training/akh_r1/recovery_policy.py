"""Authored policy driven by public documents/receipts, not an examiner world."""
import copy

def order(nodes,edges):
 pending=set(nodes);done=[]
 while pending:
  ready=sorted(n for n in pending if all(a in done for a,b in edges if b==n))
  if not ready:raise ValueError('Dependency cycle or missing operation')
  done.append(ready[0]);pending.remove(ready[0])
 return done

def act(observations,retained,call,ignore_correction=False,trust_receipts=False):
 memory=copy.deepcopy(retained);correction=False;receipts=[]
 for event in observations:
  if event['kind']=='dependency_document':memory['dependency']=copy.deepcopy(event['content'])
  elif event['kind']=='command_document':memory['commands']=copy.deepcopy(event['content'])
  elif event['kind']=='operator_correction':correction=True
  elif event['kind']=='status_receipts':receipts=event['content']
 if set(memory)!={'dependency','commands'}:raise ValueError('Missing learned documents')
 commands={c['name'] for c in memory['commands']}
 if not {'inspect-operation','apply-operation'}<=commands:raise ValueError('Required capability unavailable')
 spec=memory['dependency'];sequence=order(spec['operations'],spec['dependencies'])
 # Current inspection outranks a historical receipt; ablation deliberately omits it.
 active={r['operation'] for r in receipts if r.get('state')=='committed'} if trust_receipts else set()
 if not trust_receipts:
  for node in sequence:
   if call('inspect',node)['committed']:active.add(node)
 if correction and not ignore_correction:
  if 'compensate-operation' not in commands:raise ValueError('Missing compensation capability')
  for node in reversed(sequence):
   if node in active:call('compensate',node)
  active=set()
 for node in sequence:
  if node not in active:call('apply',node)
 verified=all(call('inspect',node)['committed'] for node in sequence)
 return dict(status='complete' if verified else 'blocked',retained=memory)

def learn(observations):
 return {('dependency' if e['kind']=='dependency_document' else 'commands'):copy.deepcopy(e['content']) for e in observations if e['kind'] in ('dependency_document','command_document')}
