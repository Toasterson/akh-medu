"""Fail-closed episode inventory and split validation, not a certification gate.

Semantic identity uses a generator-owned mechanism descriptor, with entity names
canonicalized. It is a reviewable contract, not automatic semantic equivalence.
"""
import copy
import hashlib
import json
from pathlib import Path
from r0_contract import FAMILIES

FORBIDDEN={'expected','expected_effects','examiner_events','score','score_labels','private_state','future_correction','answer_key'}
FAMILY_IDS={f for f,_,_ in FAMILIES}


def stable(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def sha(value):return hashlib.sha256(stable(value).encode()).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic_key(episode):
    entities=episode['entities']
    if len(entities)!=len(set(entities)):raise ValueError('Duplicate entity names')
    mapping={s:'ENTITY_'+str(i) for i,s in enumerate(entities)}
    def normalize(v):
        if isinstance(v,str):return mapping.get(v,v)
        if isinstance(v,list):return [normalize(x) for x in v]
        if isinstance(v,dict):return {k:normalize(x) for k,x in v.items()}
        return v
    return sha(normalize(episode['semantic_structure']))


def validate_episode(e,registry):
    required={'id','split','family','generator','entities','semantic_structure','public_start','examiner','assistance','budget','source_requirements'}
    if set(e)!=required:raise ValueError('Episode fields differ from schema')
    if e['split'] not in ['development','validation'] or e['family'] not in FAMILY_IDS:raise ValueError('Unknown split/family')
    reg=registry[e['generator']]
    if e['split'] not in reg['allowed_splits']:raise ValueError('Generator not approved for split')
    if e['family'] not in reg['families']:raise ValueError('Generator family mismatch')
    if file_sha(reg['path'])!=reg['sha256']:raise ValueError('Generator hash changed')
    if not e['semantic_structure'] or not isinstance(e['semantic_structure'],dict):raise ValueError('Missing semantic mechanism')
    if set(e['public_start'])!={'goal','tools','max_calls'}:raise ValueError('Unexpected public fields')
    if not isinstance(e['budget']['max_calls'],int) or not 1<=e['budget']['max_calls']<=128:raise ValueError('Invalid budget')
    if e['public_start']['max_calls']!=e['budget']['max_calls']:raise ValueError('Budget mismatch')
    if set(e['examiner'])!={'initial_world','events','expected_effects','private_markers'}:raise ValueError('Incomplete examiner')
    def walk(v):
        if isinstance(v,dict):
            if FORBIDDEN & set(v):raise ValueError('Examiner fields leaked into public data')
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(e['public_start'])
    public=stable(e['public_start'])
    for marker in e['examiner']['private_markers']:
        if not isinstance(marker,str) or not marker:raise ValueError('Invalid private marker')
        if marker in public:raise ValueError('Private marker leaked')
    for event in e['examiner']['events']:
        if set(event)!={'after_action','payload'}:raise ValueError('Future event must have causal trigger')
        if event['payload'] and stable(event['payload']) in public:raise ValueError('Future event leaked')
    semantic_key(e)


def build(episodes,registry):
    ids=set();semantics={};rows=[];errors=[]
    for e in episodes:
        try:
            validate_episode(e,registry)
            if e['id'] in ids:raise ValueError('Duplicate episode id')
            ids.add(e['id']);key=semantic_key(e)
            if key in semantics:raise ValueError('Semantic duplicate of '+semantics[key])
            semantics[key]=e['id']
            reg=registry[e['generator']]
            rows.append(dict(id=e['id'],split=e['split'],family=e['family'],generator=e['generator'],semantic_key=key,
                             content_sha256=sha(e),qualified=reg['independent_generator'],assistance=e['assistance']))
        except (ValueError,KeyError,TypeError) as error:errors.append(dict(id=e.get('id'),error=str(error)))
    blockers=[];counts={}
    for split in ['development','validation']:
        counts[split]={}
        for family,target,_ in FAMILIES:
            count=sum(r['qualified'] and r['family']==family and r['split']==split for r in rows)
            counts[split][family]=count
            if count<target:blockers.append(dict(split=split,family=family,required=target,qualified=count,reason='Independent generator coverage missing or insufficient'))
    # Different generator implementation groups, not just seeds, across splits.
    groups={split:{registry[r['generator']]['split_group'] for r in rows if r['split']==split and r['qualified']} for split in counts}
    if groups['development'] & groups['validation']:errors.append(dict(id=None,error='Generator structure group crosses splits'))
    return dict(schema='akh-episode-pack-v1',rows=rows,errors=errors,counts=counts,blockers=blockers,
                manifest_valid=not errors,pack_complete=not errors and not blockers,
                R1_ready=False,reason='Pack completeness alone does not satisfy execution, perception and full R0 prerequisites.',
                validation_limit='Declared structural descriptors and exact leakage checks require review; paraphrased leakage and mislabeled mechanisms are not automatically detected.')


def write_pack(out,episodes,registry):
    out=Path(out);out.mkdir(exist_ok=False);result=build(episodes,registry)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'registry.json').write_text(json.dumps(registry,indent=2)+'\n')
    if result['errors']:return result
    private=out/'examiner';private.mkdir(mode=0o700)
    public=out/'public';public.mkdir()
    for i,e in enumerate(episodes):
        # IDs never become paths. Only whitelisted public fields are serialized.
        (public/f'{i:04}.json').write_text(json.dumps(e['public_start'],indent=2)+'\n')
        path=private/f'{i:04}.json';path.write_text(json.dumps(e,indent=2)+'\n');path.chmod(0o600)
    return result
