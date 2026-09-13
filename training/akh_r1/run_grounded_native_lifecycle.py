"""Frozen retained grounded-query/effect integration; supervisor required."""
import json,hashlib
from pathlib import Path
from environment import Graph,Fixture
from native_grounded_contract import GroundedHost,GroundedEffectEpisode
from public_owned_revision_effect import audit_repairs
from run_public_breadth import candidate
class BoundedGraph(Graph):
 def __init__(self,*args):self.calls=0;super().__init__(*args)
 def call(self,*args,**kwargs):
  if self.calls>=96:raise RuntimeError('Native call cap')
  self.calls+=1;return super().call(*args,**kwargs)
def run(root):
 root=Path(root);m=json.loads((root/'manifest.json').read_text())
 for p,h in m['fingerprints'].items():
  if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Changed dependency '+p)
 out=root/'execution';out.mkdir(exist_ok=False);rows=[];sessions=0
 for case in json.loads((root/'inputs.json').read_text()):
  dest=out/case['id'];dest.mkdir();fixture=None;pipeline_calls=0
  try:
   fixture=Fixture(m['image'],'akh-grounded-fixture-'+case['id'],commands=['sana'],cli=Path(m['cli']))
   fixture.reset(dict(nodes={n:dict(healthy=False,repairs=0) for n in ['answer-a','answer-b','unrelated']},authorized_target=case['expected']['initial'][0] if len(case['expected']['initial'])==1 else None,audit=[]))
   def terminal(stages):
    nonlocal pipeline_calls
    if pipeline_calls>=12:raise RuntimeError('Pipeline cap')
    pipeline_calls+=1;return fixture.pipeline(stages)
   pids=[]
   for phase in ['acquire','cold','correction']:
    if sessions>=18:raise RuntimeError('Session cap')
    before=fixture.state();expected=case['expected']['later' if phase=='correction' else 'initial']
    if phase=='correction':
     before['authorized_target']=expected[0] if len(expected)==1 else None;fixture.reset(before)
    graph=host=None
    try:
     sessions+=1;graph=BoundedGraph(m['binary'],dest/'graph',dest/'native.log');pids.append(graph.process.pid);host=GroundedHost(graph,dest/'host',case['id'])
     document=None if phase=='cold' else case['documents'][phase]
     port=GroundedEffectEpisode(host,document,terminal,help_available=phase=='acquire',max_calls=24);start=port.start()
     candidate(m['image'],Path(m['client']),port,'akh-grounded-client-'+case['id']+'-'+phase)
     after=fixture.state();target=expected[0] if len(expected)==1 else None
     passed=audit_repairs(before,after,target,port.events) and port.finished==dict(status='complete' if target else 'blocked',answer=expected if target else [])
     raw=graph.call('exam_owned_snapshot',{});actual=sorted({x['labels']['subject'] for x in raw['rows'] if x['labels']['predicate']=='supported' and x['labels']['object']=='true'});passed=passed and actual==expected
     row=dict(case=case['id'],phase=phase,pid=graph.process.pid,native_calls=graph.calls,start=start,finished=port.finished,events=port.events,before=before,after=after,owned=host.state['current'],native_targets=actual,passed=passed)
     (dest/(phase+'.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
     if not passed:raise AssertionError('Revision/effect audit failed')
    finally:
     if host:host.close()
     if graph:graph.close()
   if len(set(pids))!=3:raise AssertionError('Cold process boundary missing')
  finally:
   if fixture:fixture.close()
 (out/'audit.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':
 import sys
 run(sys.argv[1])
