"""Register preserved finite language evidence and load it for a host runner.

No candidate receives the examiner registration. No replay is performed here.
"""
import argparse,json
from pathlib import Path
from episode_pack import file_sha,sha,write_pack,build
TOOLS=['list_documents','read_document','remember','revise','query','terminal','finish']
def register(source,out,case_ids=None):
 source=Path(source);plan=json.loads((source/'plan.json').read_text());audit=json.loads((source/'artifact-audit.json').read_text())
 if not audit.get('all_passed'):raise ValueError('Source audit failed')
 for path,digest in plan['hashes'].items():
  if file_sha(path)!=digest:raise ValueError('Frozen source changed')
 episodes=[];registry={}
 if case_ids is None:
  selected=[next(c for c in plan['cases'] if c['split']==split and c['control']=='complete') for split in ['development','validation']]
 else:
  if not case_ids or len(set(case_ids))!=len(case_ids):raise ValueError('Unique retained cases required')
  selected=[c for c in plan['cases'] if c['id'] in case_ids]
  if len(selected)!=len(case_ids):raise ValueError('Unknown retained case')
 for case in selected:
  split=case['split'];generator='language-'+split if case_ids is None else 'language-control-'+case['id']
  registry[generator]=dict(path=str(source/'source/language_composition_worlds.py'),sha256=file_sha(source/'source/language_composition_worlds.py'),allowed_splits=[split],families=['language_transfer'],independent_generator=False,split_group=generator)
  related=[c['id'] for c in plan['cases'] if c['split']==split]
  episodes.append(dict(id=generator,split=split,family='language_transfer',generator=generator,entities=['nodealpha','nodebeta','nodegamma','blue','api'],semantic_structure=plan['descriptors'][split] if case_ids is None else dict(mechanism=plan['descriptors'][split],control=case['control']),public_start=dict(goal=case['goal'],tools=TOOLS,max_calls=32),examiner=dict(initial_world=dict(documents=case['documents'],fixture=dict(authorized_target=case['expected_target'],nodes={n:dict(healthy=False,repairs=0) for n in ['nodealpha','nodebeta','nodegamma']},audit=[])),events=[],expected_effects=dict(target=case['expected_target'],repairs=int(case['expected_effect'])),private_markers=['examiner-only-language-outcome']),assistance=['parameterized authored grammar','native graph','executor target guard'],budget=dict(max_calls=32),source_requirements=dict(run=str(source),case=case['id'],related_controls=related,plan_sha256=file_sha(source/'plan.json'),audit_sha256=file_sha(source/'artifact-audit.json'),evidence_sha256=file_sha(source/case['id']/'evidence.json'),trace_sha256=file_sha(source/case['id']/'trace.json'))))
 return write_pack(out,episodes,registry)

def load_case(pack,episode_id):
 pack=Path(pack);manifest=json.loads((pack/'manifest.json').read_text());registry=json.loads((pack/'registry.json').read_text());episodes=[json.loads(p.read_text()) for p in sorted((pack/'examiner').glob('*.json'))]
 checked=build(episodes,registry)
 if not checked['manifest_valid'] or checked['rows']!=manifest['rows']:raise ValueError('Registration changed or invalid')
 matches=[e for e in episodes if e['id']==episode_id]
 if len(matches)!=1:raise ValueError('Episode identity not unique')
 e=matches[0];sources=e['source_requirements'];root=Path(sources['run'])
 for name,key in [('plan.json','plan_sha256'),('artifact-audit.json','audit_sha256'),(sources['case']+'/evidence.json','evidence_sha256'),(sources['case']+'/trace.json','trace_sha256')]:
  if file_sha(root/name)!=sources[key]:raise ValueError('Evidence changed')
 plan=json.loads((root/'plan.json').read_text())
 if not json.loads((root/'artifact-audit.json').read_text()).get('all_passed'):raise ValueError('Source audit failed')
 for path,digest in plan['hashes'].items():
  if file_sha(path)!=digest:raise ValueError('Frozen source changed')
 cases=[c for c in plan['cases'] if c['id']==sources['case']]
 if len(cases)!=1:raise ValueError('Source case identity mismatch')
 c=cases[0]
 if e['public_start']['goal']!=c['goal'] or e['split']!=c['split'] or e['examiner']['initial_world']['documents']!=c['documents'] or e['examiner']['expected_effects']!=dict(target=c['expected_target'],repairs=int(c['expected_effect'])):raise ValueError('Registration differs from retained case')
 # Blocked controls are successful evaluations when they preserve state and abstain.
 return dict(id=e['id'],split=c['split'],control=c['control'],goal=c['goal'],documents=c['documents'],expected_target=c['expected_target'],expected_effect=c['expected_effect'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();result=register(a.source,a.out);print(json.dumps(dict(manifest_valid=result['manifest_valid'],rows=len(result['rows']),R1_ready=result['R1_ready'])));
 if not result['manifest_valid']:raise ValueError(result['errors'])
if __name__=='__main__':main()
