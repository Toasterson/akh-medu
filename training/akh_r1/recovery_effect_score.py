"""Independent log replay against declared dependency and interruption contract."""
def score(episode,state):
 w=episode['examiner']['world'];active=set();counts=set();first=[];undone=set();valid=True
 for e in state['log']:
  node=e['operation'];key=(node,e['generation']);valid &= node in w['nodes']
  if e['action']=='apply':
   valid &= node not in active and key not in counts and all(a in active for a,b in w['edges'] if b==node)
   if e['generation']==1:first.append(node)
   elif e['generation']==2:valid &= episode['examiner']['require_compensation'] and undone==set(w['committed'])
   else:valid=False
   active.add(node);counts.add(key)
  elif e['action']=='compensate':
   valid &= episode['examiner']['require_compensation'] and node in active and node in w['committed'] and not any(a==node and b in active for a,b in w['edges'])
   active.discard(node);undone.add(node)
  else:valid=False
 if episode['examiner']['require_compensation']:valid &= first==w['committed'] and undone==set(w['committed'])
 checks=dict(log_valid=bool(valid),all_effects=active==set(w['nodes']),state_matches=active==set(state['active']),restarted=state['stage']=='restarted',no_pending_compensation=not state['pending_compensation'])
 return dict(passed=all(checks.values()),checks=checks,scope='Independent in-process log/state oracle; not native or production durability')
