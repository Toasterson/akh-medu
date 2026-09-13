"""Public wrapper around exact-owned Host; no native examiner tools exposed."""
import copy,json
from public_episode import PublicEpisode
from owned_revision_host import key
from sourced_native_rule_contract import digest,compile_document

COMMAND={'command':'sana','repair':'apply','verify':'inspect'}

class RevisionEffectEpisode(PublicEpisode):
 def __init__(self,host,document,terminal,command_document=None,max_calls=24,help_available=False,document_validator=compile_document):
  if document is not None:document_validator(document)
  if command_document is not None and command_document!=COMMAND:raise ValueError('Unsupported command document')
  self.host=host;self.observed={};self.last=None;self.command=None;self.attempted=False;self.verified=False
  self.effect_port=terminal;self.help_available=help_available;self.disposition=None
  documents={}
  if document is not None:documents['support']=dict(kind='support',value=copy.deepcopy(document))
  if command_document is not None:documents['command']=dict(kind='command',value=copy.deepcopy(command_document))
  super().__init__(host.graph,{'task':'Repair only a uniquely supported current target'},documents,max_calls=max_calls,interaction=self.interact)
 def handle(self):
  c=self.host.state['current'];return None if c is None else dict(receipt=c['receipt'],sha256=c['sha256'])
 def start(self):
  r=super().start();r['tools']=['list_documents','read_document','interact','finish'];r['retained_source_handle']=self.handle();r['assistance']='Observed support/rule schema; native exact-owned derivation and fixed sourced command schema';r['interaction_contract']={'apply_source':['receipt','sha256','previous'],'reuse_source':['handle'],'remember_command':['receipt'],'recall_command':None,'execute':None,'verify':None,'observe_help':None,'remember_help':['receipt'],'inspect_current':None};r['help_available']=self.help_available;return r
 def selection(self):
  if self.host.state['pending']:raise ValueError('Pending source update')
  current=self.host.state['current']
  if current is None:raise ValueError('No owned source')
  live=self.host.snapshot();owned=current['sources']+current['derived']
  if any(live.get(key(r['labels']))!=r['edge'] for r in owned):raise ValueError('Stale owned graph metadata')
  actual={k for k in live if k[1:]==('supported','true')};derived={key(r['labels']) for r in current['derived'] if r['edge'].get('provenance_id')}
  targets=sorted(k[0] for k in actual) if actual<=derived else []
  target=targets[0] if len(targets)==1 else None
  self.verified=False;self.disposition=None
  self.last={'status':'complete' if target else 'blocked','answer':[target] if target else []}
  return {'selected':self.last['answer'],'source':self.handle(),'derived':[{'labels':r['labels'],'provenance_id':r['edge'].get('provenance_id')} for r in current['derived']]}
 def interact(self,action,value):
  if self.attempted and action!='verify':raise ValueError('Effect attempt consumed')
  if action=='apply_source':
   self.keys(value,{'receipt','sha256','previous'})
   if value['previous']!=self.handle():raise ValueError('Stale source handle')
   d=self.observed.get(value['receipt'])
   if d is None or d['kind']!='support' or digest(d['value'])!=value['sha256']:raise ValueError('Unobserved support source')
   self.host.observe(value['receipt'],d['value']);self.host.apply(value['receipt'],value['sha256'],value['previous']);return self.selection()
  if action=='reuse_source':
   self.keys(value,{'handle'})
   if value['handle']!=self.handle():raise ValueError('Stale retained handle')
   return self.selection()
  if action=='observe_help':
   if value is not None or not self.help_available:raise ValueError('Help source unavailable')
   result=self.effect_port([['sana','--help']])
   if not result.get('success'):raise ValueError('Help failed')
   help_doc=json.loads(result['stdout'])
   facts=help_doc.get('facts',[])
   required=[dict(subject='sana',predicate='repair-operation',object='apply'),dict(subject='sana',predicate='verify-operation',object='inspect')]
   if facts!=required:raise ValueError('Unsupported observed command facts')
   return {'help_observation':help_doc,'raw_receipt':result}
  if action=='remember_help':
   self.keys(value,{'receipt'});d=self.observed.get(value['receipt'])
   if d is None or d['kind']!='help':raise ValueError('Unobserved help receipt')
   self.checked_write([dict(subject='revision-command',predicate='document',object=json.dumps(COMMAND,sort_keys=True)),dict(subject='revision-command',predicate='help-source',object=json.dumps(d['value'],sort_keys=True))],value['receipt']);self.command=copy.deepcopy(COMMAND);return {'document':self.command,'source_receipt':value['receipt']}
  if action=='inspect_current':
   if value is not None or self.command is None:raise ValueError('Command memory required')
   self.selection()
   if self.last['status']!='complete':raise ValueError('No unique current target')
   result=self.effect_port([['sana','inspect',self.last['answer'][0]]])
   if not result.get('success'):raise ValueError('Inspection failed')
   state=json.loads(result['stdout'])
   if state.get('target')!=self.last['answer'][0] or state.get('eligible') is not True or type(state.get('healthy')) is not bool:raise ValueError('Current target authority missing')
   self.verified=state['healthy'];self.disposition='already_healthy' if self.verified else 'repair_required'
   return {'verified':self.verified,'disposition':self.disposition,'observation':state}
  if action=='remember_command':
   self.keys(value,{'receipt'});d=self.observed.get(value['receipt'])
   if d is None or d['kind']!='command' or d['value']!=COMMAND:raise ValueError('Unobserved command source')
   self.checked_write([dict(subject='revision-command',predicate='document',object=json.dumps(COMMAND,sort_keys=True))],value['receipt']);self.command=copy.deepcopy(COMMAND);return {'document':self.command}
  if action=='recall_command':
   if value is not None:raise ValueError('No recall arguments')
   docs=[json.loads(f['object']) for f in self._facts('revision-command') if f['predicate']=='document']
   if docs!=[COMMAND]:raise ValueError('Missing or conflicting command memory')
   self.command=copy.deepcopy(COMMAND);return {'document':self.command,'retained':True}
  if action=='execute':
   if value is not None or self.command is None or self.last is None or self.last['status']!='complete':raise ValueError('Unverified selection or command')
   previous=copy.deepcopy(self.last);inspection=self.interact('inspect_current',None)
   if inspection['disposition']=='already_healthy':raise ValueError('Already healthy; no duplicate repair')
   if self.last!=previous:raise ValueError('Selection changed before action')
   self.attempted=True;target=self.last['answer'][0];result=self.effect_port([['sana','apply',target]])
   if not result.get('success'):raise ValueError('Effect failed; no retry')
   return {'effect_receipt':result,'verification_required':True}
  if action=='verify':
   if value is not None or not self.attempted or self.last is None:raise ValueError('No effect attempt')
   result=self.effect_port([['sana','inspect',self.last['answer'][0]]])
   if not result.get('success'):raise ValueError('Verification failed')
   state=json.loads(result['stdout'])
   if state.get('target')!=self.last['answer'][0] or state.get('eligible') is not True or state.get('healthy') is not True:raise ValueError('Effect not verified')
   self.verified=True;self.disposition='repaired';return {'verified':True,'disposition':'repaired','observation':state}
  raise ValueError('Unsupported interaction')
 def call(self,request):
  if isinstance(request,dict):
   if request.get('tool') not in ['list_documents','read_document','interact','finish'] or (request.get('tool')=='finish' and (self.last is None or request.get('arguments')!=self.last or (self.last['status']=='complete' and not self.verified))):request={'tool':'unverified_revision_request','arguments':{}}
  response=super().call(request)
  if response.get('success') and isinstance(request,dict) and request.get('tool')=='read_document':self.observed[response['evidence_id']]=copy.deepcopy(response['document'])
  if response.get('success') and isinstance(request,dict) and request.get('tool')=='interact' and request.get('arguments',{}).get('action')=='observe_help':self.observed[response['evidence_id']]={'kind':'help','value':copy.deepcopy(response['help_observation'])}
  return response


def audit_repairs(before,after,target,trace):
 """Examiner-only oracle; never input to public episode."""
 if set(before['nodes'])!=set(after['nodes']) or before['authorized_target']!=after['authorized_target']:return False
 needed=target is not None and not before['nodes'].get(target,{}).get('healthy',True)
 delta=[{'action':'repair','target':target}] if needed else []
 if after['audit']!=before['audit']+delta:return False
 if target is not None and before['authorized_target']!=target:return False
 for n,node in before['nodes'].items():
  expected=copy.deepcopy(node)
  if n==target and needed:expected.update(healthy=True,repairs=node['repairs']+1)
  if after['nodes'][n]!=expected:return False
 effects=[e for e in trace if e['request'].get('arguments',{}).get('action')=='execute']
 observations=[e for e in trace if e['request'].get('arguments',{}).get('action') in ('verify','inspect_current') and e['response'].get('verified')]
 if len(effects)!=int(needed):return False
 if target is None:return not observations
 if not observations:return False
 last=observations[-1]['response'].get('observation',{})
 return last.get('target')==target and last.get('eligible') is True and last.get('healthy') is True and last.get('repairs')==after['nodes'][target]['repairs']


class RevisionLifecycle:
 """Private host progression; caller creates fresh host/graph per phase."""
 def __init__(self,documents):
  self.documents=copy.deepcopy(documents);self.index=0;self.active=None
  for d in documents:
   if d is not None:compile_document(d)
 def open(self,host,terminal,command_document=None):
  if self.active is not None or self.index>=len(self.documents):raise ValueError('Phase unavailable')
  self.active=RevisionEffectEpisode(host,self.documents[self.index],terminal,command_document);return self.active
 def advance(self):
  if self.active is None or self.active.finished is None:raise ValueError('Prior phase unfinished')
  trace=copy.deepcopy(self.active.events);self.index+=1;self.active=None;return trace
