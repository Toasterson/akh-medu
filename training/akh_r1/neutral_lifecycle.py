"""Candidate-neutral observation preparation; no runtime or model execution.

The host owns intervention assignment. Candidates receive the resulting observed
view only; frozen legacy clients that branch on conditions are incompatible.
"""
import copy
import hashlib
from pathlib import Path

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

GOALS={
 'command':{'command','command_profile','target','mode'},
 'ownership':{'interaction_profile','kind','participants','options','goal'},
 'perception':{'breadth','layout','command','source_version'},
 'language':{'language_profile','instruction','command','source_version'},
 'authority':{'command','command_profile','interaction_profile','target'},
}
PRIVATE={'condition','control','stratum','expected','expected_target','examiner','private_markers','later_event','phase','split','seed'}

def reject_private(value):
 if isinstance(value,dict):
  if PRIVATE & set(value):raise ValueError('Private assignment in candidate-visible input')
  for v in value.values():reject_private(v)
 elif isinstance(value,list):
  for v in value:reject_private(v)

def observation_view(observed,interventions):
 """Actually remove named records from delivered views, never instruct failure.

Only host-verified observations may be passed. This function cannot establish
truth/provenance; source hashing and the live observation adapter must do that.
"""
 if set(observed)!={'documents','receipts','memory'}:raise ValueError('Expected three observation stores')
 view=copy.deepcopy(observed)
 if any(not isinstance(v,dict) for v in view.values()):raise ValueError('Stores require explicit stable record identities')
 used=set()
 for item in interventions:
  if set(item)!={'operation','store','ids'} or item['operation']!='withhold':raise ValueError('Only explicit withholding is supported')
  store=item['store'];ids=item['ids']
  if store not in view or not isinstance(ids,list) or not ids or len(set(ids))!=len(ids):raise ValueError('Invalid withholding selection')
  for identity in ids:
   if not isinstance(identity,str) or (store,identity) in used or identity not in view[store]:raise ValueError('Absent or duplicate intervention target')
   used.add((store,identity));del view[store][identity]
 reject_private(view)
 return view

def prepare(entry):
 if set(entry)!={'id','family','legacy_goal','tools','max_calls','assistance','source_hashes','observations','interventions'}:raise ValueError('Unexpected lifecycle entry field')
 family=entry['family']
 if family not in GOALS:raise ValueError('Unsupported lifecycle family')
 goal=copy.deepcopy(entry['legacy_goal'])
 # Migration is explicit and limited to known top-level legacy dispatch labels.
 removed={k:goal.pop(k) for k in ('condition','phase') if k in goal}
 if set(goal)-GOALS[family]:raise ValueError('Unsupported candidate goal field')
 reject_private(goal)
 if not entry['source_hashes']:raise ValueError('Pinned observation/runtime sources required')
 for path,digest in entry['source_hashes'].items():
  if file_sha(path)!=digest:raise ValueError('Source fingerprint changed')
 view=observation_view(entry['observations'],entry['interventions'])
 if type(entry['max_calls']) is not int or not 1<=entry['max_calls']<=48:raise ValueError('Invalid call budget')
 if not isinstance(entry['tools'],list) or any(t not in {'list_documents','read_document','remember','revise','query','terminal','interact','finish'} for t in entry['tools']):raise ValueError('Unknown public tool')
 return dict(schema='akh-pack-preparation-v1',adapter='neutral-lifecycle-v1',episode_id=entry['id'],assistance=copy.deepcopy(entry['assistance']),
  public_start=dict(goal=goal,tools=entry['tools'][:],max_calls=entry['max_calls']),
  host_config=dict(family=family,observations=view,interventions=copy.deepcopy(entry['interventions']),legacy_labels=removed,source_hashes=copy.deepcopy(entry['source_hashes']),legacy_client_compatible=False,executor_binding_ready=False),
  executed=False,R1_ready=False)
