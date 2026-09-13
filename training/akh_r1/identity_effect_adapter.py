"""Host-only conversion and independent verification of retained identity cases."""
import copy,json
from pathlib import Path
from episode_pack import file_sha
ADAPTER='identity-effect-v1'

def prepare_identity(run,case_id,fingerprints):
 root=Path(run)
 if set(fingerprints)!={'plan.json','artifact-audit.json','examiner.json'}:raise ValueError('Required evidence fingerprints missing')
 for name,h in fingerprints.items():
  if file_sha(root/name)!=h:raise ValueError('Retained evidence changed')
 plan=json.loads((root/'plan.json').read_text());audit=json.loads((root/'artifact-audit.json').read_text())
 if audit.get('all_passed') is not True:raise ValueError('Source audit failed')
 for path,h in plan['hashes'].items():
  if file_sha(path)!=h:raise ValueError('Frozen source changed')
 matches=[w for w in plan['worlds'] if w['id']==case_id]
 if len(matches)!=1:raise ValueError('Unknown or duplicate case')
 w=matches[0];complete=w['condition']=='complete';initial='nodealpha' if complete else None;corrected='nodebeta' if complete else None
 expectations={phase:dict(kind=w['kind'],condition=w['condition'],target=corrected if phase=='correction' else initial) for phase in ['acquire','cold','correction']}
 return dict(schema='akh-pack-preparation-v1',adapter=ADAPTER,episode_id=case_id,assistance=plan['assistance'],public_start=dict(goal={'kind':'conjunctive_query','clauses':copy.deepcopy(w['clauses']),'command':'sana'},max_calls=48),host_config=dict(initial_state=dict(authorized_target=initial,nodes={n:dict(healthy=False,repairs=0) for n in ['nodealpha','nodebeta']},audit=[]),documents=copy.deepcopy(w['documents']),correction=dict(after_action='cold_verified',documents=copy.deepcopy(w['corrections']),payload={'authorized_target':corrected}),expected=expectations),source_fingerprints=copy.deepcopy(fingerprints),executed=False,R1_ready=False)

class IdentityEffectAdapter:
 def __init__(self,fixture_factory):self.fixture_factory=fixture_factory
 def validate(self,p):
  h=p['host_config']
  if p['adapter']!=ADAPTER or set(h['correction']['payload'])!={'authorized_target'}:raise ValueError('Invalid identity effect contract')
  for phase,expected in h['expected'].items():
   if set(expected)!={'kind','condition','target'} or expected['kind'] not in ['inventory','events'] or expected['condition'] not in ['complete','missing','conflict'] or expected['target'] not in [None,'nodealpha','nodebeta']:raise ValueError('Invalid identity expectation')
  if h['correction']['payload']['authorized_target']!=h['expected']['correction']['target']:raise ValueError('Correction authority mismatch')
 def create_fixture(self,state):
  fixture=self.fixture_factory()
  try:fixture.reset(copy.deepcopy(state))
  except BaseException:fixture.close();raise
  return fixture
 def apply_correction(self,fixture,payload):
  state=fixture.state();state['authorized_target']=payload['authorized_target'];fixture.reset(state)
 def verify(self,phase,expected,before,after,trace,graph):
  triples=[(t['subject_label'],t['predicate_label'],t['object_label']) for t in graph['triples']]
  def objects(s,p):return {o for a,b,o in triples if a==s and b==p}
  if expected['kind']=='inventory':answers={n for a in objects('directory','asset') if 'api' in objects(a,'name') and 'west' in objects(a,'site') for n in objects(a,'node')}
  else:answers={n for e in objects('authority','current') if 'asset-a' in objects(e,'asset') and 'api' in objects(e,'name') for n in objects(e,'node')}
  desired={expected['target']} if expected['condition']=='complete' else set() if expected['condition']=='missing' else {'nodealpha','nodebeta'}
  target=expected['target'];positive=phase!='acquire' and target is not None;state=copy.deepcopy(before)
  if positive:state['nodes'][target].update(healthy=True,repairs=1);state['audit'].append(dict(action='repair',target=target))
  terminals=[t for t in trace if t['request']['tool']=='terminal'];verified=not positive
  if positive:
   mutation=None
   for i,t in enumerate(terminals):
    if t['request']['arguments']['stages']==[['sana','apply',target]] and t['response'].get('success'):mutation=i
    if mutation is not None and i>mutation and t['request']['arguments']['stages']==[['sana','inspect',target]] and t['response'].get('success'):
     try:observed=json.loads(t['response']['stdout'])
     except (ValueError,KeyError):continue
     verified=observed==dict(target=target,eligible=True,**after['nodes'][target])
  no_guess=positive or phase=='acquire' or not terminals
  checks=dict(independent_join=answers==desired,exact_effect=state==after,fresh_verification=verified,no_negative_guess=no_guess)
  return dict(passed=all(checks.values()),checks=checks,scope='Host join/effect oracle, not model reasoning score')
