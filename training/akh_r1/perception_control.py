"""Retained spatial observations through shared controls; never invokes OCR."""
import copy,json
from pathlib import Path
from episode_pack import file_sha
from perception_structures import interpret
from perception_structure_score import score
ADAPTER='spatial-perception-v1'
OCR_FIELDS={'success','modality','image_sha256','engine','exit_code','raw_tsv','stderr','source_spans','facts','extraction','assistance'}

def prepare_perception(source,case_id,pins):
 source=Path(source);names=['plan.json','examiner.json','artifact-audit.json','status.json',case_id+'/observation.json',case_id+'/evidence.json']
 if set(pins)!=set(names) or any(file_sha(source/n)!=pins[n] for n in names):raise ValueError('Retained perception fingerprint mismatch')
 read=lambda n:json.loads((source/n).read_text())
 if read('status.json')['stage']!='completed' or not read('artifact-audit.json')['all_passed']:raise ValueError('Incomplete perception evidence')
 for path,digest in read('plan.json')['hashes'].items():
  if file_sha(path)!=digest:raise ValueError('Frozen dependency changed')
 cases=[c for c in read('examiner.json') if c['id']==case_id]
 if len(cases)!=1:raise ValueError('Unknown or duplicate case')
 c=cases[0];obs=read(case_id+'/observation.json');e=read(case_id+'/evidence.json');transcript=bool(c.get('transcript'))
 if transcript:
  if set(obs)!={'modality','facts','text_rows'} or obs['modality']!='transcript_control':raise ValueError('Transcript mislabeled')
 else:
  if set(obs) not in (OCR_FIELDS,(OCR_FIELDS-{'assistance'})|{'error'}) or obs['image_sha256']!=file_sha(c['image']) or obs['exit_code']!=0 or obs['modality']=='transcript_control':raise ValueError('Invalid OCR receipt')
  if interpret(obs['source_spans'],c['descriptor']['id'])['facts']!=obs['facts']:raise ValueError('Observation facts differ from interpretation')
 if not score(c['descriptor'],c['control'],obs):raise ValueError('Retained relation audit failed')
 return dict(schema='akh-pack-preparation-v1',adapter=ADAPTER,episode_id=case_id,public_start=dict(goal=dict(layout=c['descriptor']['id'],command='sana'),max_calls=48),host_config=dict(observation=copy.deepcopy(obs),initial_state=copy.deepcopy(e['before']),positive=c['control']!='missing_structure',modality='retained_transcript_control' if transcript else 'retained_actual_ocr'),assistance=['Fixed spatial schemas','Authored interpretation and client','Retained receipt; no fresh OCR'],source_fingerprints=pins.copy(),executed=False,R1_ready=False)

class PerceptionControl:
 def __init__(self,factory):self.factory=factory
 def validate(self,p):
  h=p['host_config'];obs=h['observation'];layout=p['public_start']['goal']['layout']
  if set(h)!={'observation','initial_state','positive','modality'} or type(h['positive']) is not bool or layout not in ('table','procedure'):raise ValueError('Invalid spatial control')
  if h['modality']=='retained_actual_ocr':
   if set(obs) not in (OCR_FIELDS,(OCR_FIELDS-{'assistance'})|{'error'}) or obs['modality']=='transcript_control':raise ValueError('Invalid observed fields')
  elif h['modality']=='retained_transcript_control':
   if set(obs)!={'modality','facts','text_rows'} or obs['modality']!='transcript_control':raise ValueError('Invalid transcript fields')
  else:raise ValueError('Modality required')
  if h['initial_state']!=dict(layout=layout,authority='nodealpha',active=None,configured=None,audit=[]):raise ValueError('Unexpected fixture initial state')
 def create(self,p):
  f=self.factory()
  try:f.reset(copy.deepcopy(p['host_config']['initial_state']))
  except BaseException:f.close();raise
  return f
 def observation(self,p,r):return {'observed':copy.deepcopy(p['host_config']['observation'])},r.pipeline,None
 def verify(self,p,before,after,trace,finished,r):
  positive=p['host_config']['positive'];layout=p['public_start']['goal']['layout'];expected=copy.deepcopy(before)
  if positive:
   expected['active']='nodealpha';expected['audit']=[dict(operation=op,target='nodealpha') for op in (['apply'] if layout=='table' else ['stop','configure','start'])]
   if layout=='procedure':expected['configured']='nodealpha'
  terms=[t for t in trace if t.get('request',{}).get('tool')=='terminal'];fresh=False
  if len(terms)>=2:
   last=terms[-1]
   try:fresh=last['request']['arguments']['stages']==[['sana','inspect','nodealpha']] and last['response']['success'] is True and json.loads(last['response']['stdout'])=={'ready':True} and all(t['response']['success'] is True for t in terms[:-1])
   except (KeyError,ValueError,TypeError):pass
  expected_commands=([['sana',op,'nodealpha'] for op in (['apply'] if layout=='table' else ['stop','configure','start'])]+[['sana','inspect','nodealpha']]) if positive else []
  commands_match=[t.get('request',{}).get('arguments',{}).get('stages') for t in terms]==[[cmd] for cmd in expected_commands]
  docs=[t for t in trace if t.get('request',{}).get('tool')=='read_document']
  observed=len(docs)==1 and docs[0]['response'].get('document')==p['host_config']['observation']
  checks=dict(command_sequence=commands_match,exact_effect=after==expected,status=finished['status']==('complete' if positive else 'blocked'),verification=fresh if positive else not terms,observed_only=observed)
  return dict(passed=all(checks.values()),checks=checks,scope='Retained spatial receipt and fixture effects; no fresh OCR or learned perception')
