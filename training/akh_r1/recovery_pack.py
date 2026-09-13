"""Preparation-only recovery pack binding; no qualification or execution."""
import copy,json
from pathlib import Path
from episode_pack import file_sha,write_pack
from recovery_admission import validate_candidate
ROLES={'generator','admission','transport','candidate','bridge','fixture','scorer','image'}

def check_bindings(bindings):
 if set(bindings)!=ROLES:raise ValueError('Incomplete runtime fingerprints')
 for record in bindings.values():
  if set(record)!={'path','sha256'} or file_sha(record['path'])!=record['sha256']:raise ValueError('Runtime fingerprint mismatch')

def convert(source,digest,bindings):
 source=Path(source)
 if file_sha(source)!=digest:raise ValueError('Source episode changed')
 check_bindings(bindings)
 image=json.loads(Path(bindings['image']['path']).read_text()).get('image','')
 if not isinstance(image,str) or not image.startswith('sha256:') or len(image)!=71 or any(x not in '0123456789abcdef' for x in image[7:]):raise ValueError('Immutable image identity required')
 c=json.loads(source.read_text());validate_candidate(c)
 if c['schema']!='akh-recovery-distribution-candidate-v1':raise ValueError('Unknown recovery schema')
 w=c['examiner']['world'];initial=[x for x in c['delivery_events'] if x['after']=='initial_goal'];later=[x for x in c['delivery_events'] if x['after']!='initial_goal']
 if any(x['after'] not in ('partial_execution','cold_restart') for x in later):raise ValueError('Unknown recovery event phase')
 doc=next(x['content'] for x in initial if x['kind']=='dependency_document')
 if doc['operations']!=w['nodes'] or [list(e) for e in doc['dependencies']]!=[list(e) for e in w['edges']]:raise ValueError('Observed/world dependency mismatch')
 if not any(x['kind']=='command_document' and x.get('source') for x in initial):raise ValueError('Missing observed command contract')
 generator='recovery-'+c['split']
 episode=dict(id=c['id'],split=c['split'],family='recovery_durability',generator=generator,entities=list(w['nodes']),semantic_structure=dict(mechanism=w['mechanism'],edges=w['edges'],committed=w['committed'],interrupted=w['interrupted']),public_start=dict(goal=copy.deepcopy(c['public_start']),tools=['list_documents','read_document','remember','revise','query','interact','finish'],max_calls=96),examiner=dict(initial_world=dict(source_episode=copy.deepcopy(c),documents=initial),events=[dict(after_action=e['after'],payload=copy.deepcopy(e)) for e in later],expected_effects=copy.deepcopy(c['examiner']),private_markers=['private-recovery-pack-outcome']),assistance=copy.deepcopy(c['assistance']),budget=dict(max_calls=96),source_requirements=dict(kind='recovery-v1',path=str(source.resolve()),sha256=digest,bindings=copy.deepcopy(bindings)))
 registry={generator:dict(path=bindings['generator']['path'],sha256=bindings['generator']['sha256'],allowed_splits=[c['split']],families=['recovery_durability'],independent_generator=False,split_group=generator)}
 return episode,registry

def register(source,out,bindings):
 e,registry=convert(source,file_sha(source),bindings)
 return write_pack(out,[e],registry)

def load_entry(entry):
 sources=entry['source_requirements']
 if sources['kind']!='recovery-v1':raise ValueError('Wrong source kind')
 canonical,_=convert(sources['path'],sources['sha256'],sources['bindings'])
 if entry!=canonical:raise ValueError('Registration differs from source episode')
 return dict(episode=copy.deepcopy(canonical['examiner']['initial_world']['source_episode']),bindings=copy.deepcopy(sources['bindings']),phase_order=['acquire','cold'],execution_coverage='unexecuted_registration',qualified=False)
