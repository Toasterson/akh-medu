"""Prepare retained ownership/OCR controls; no execution or qualification credit."""
import copy,json
from pathlib import Path
from episode_pack import file_sha,write_pack

def entries(source,kind):
    source=Path(source).resolve()
    if kind not in ('ownership','ocr'):raise ValueError('Unknown control kind')
    read=lambda name:json.loads((source/name).read_text())
    plan=read('plan.json')
    if not read('artifact-audit.json').get('all_passed') or read('status.json')['stage']!='completed':raise ValueError('Incomplete or failed source')
    for path,digest in plan['source_hashes'].items():
        if file_sha(path)!=digest:raise ValueError('Frozen source changed')
    # Bind all retained case artifacts, including negative controls and private descriptors.
    artifacts=[p for p in source.rglob('*.json') if 'graph' not in p.parts]
    fingerprints={str(p.relative_to(source)):file_sha(p) for p in artifacts}
    rows=[]; registry={}
    if kind=='ownership':
        cases=[c for c in read('examiner.json') if c['mode']=='responsive' and c['final_target']=='trial-a']
    else:cases=[c for c in read('examiner.json') if c[0] in ('inline','stacked')]
    for case in cases:
        if kind=='ownership':
            name=case['id'];mechanism=case['mechanism'];family='dojo_pressure';generator_file='ownership_worlds.py'
            participants=['owner'] if mechanism=='owner_correction' else ['owner','operator']
            goal=dict(kind='ownership_process',participants=participants,mode='responsive')
            world=dict(case=case)
            events=[dict(after_action='propose',payload=dict(final_target=case['final_target'],requires_reinquiry=participants))]
            expected=dict(performed=[case['final_target']],returned=participants)
            tools=['interact','finish'];assistance=['authored typed client','simulated participants and effects','native graph interaction receipts']
        else:
            name,filename,target,_=case;mechanism=name;family='perception_knowledge';generator_file='run_ocr_layouts.py'
            goal=dict(kind='graph_effect',clauses=[['panel','target','?answer']],command='sana')
            world=dict(image_path=str(source/filename),observation=read(name+'/observation.json'),fixture_target=target)
            events=[];expected=dict(target=target,repairs=1)
            tools=['list_documents','read_document','remember','revise','query','terminal','finish'];assistance=['local Tesseract English','strict two-identifier label parser','authored client and executor target guard','retained OCR receipt; preparation does not rerun OCR']
        generator=kind+'-'+mechanism
        registry[generator]=dict(path=str(source/'source'/generator_file),sha256=file_sha(source/'source'/generator_file),allowed_splits=['development'],families=[family],independent_generator=False,split_group=generator)
        rows.append(dict(id=generator,split='development',family=family,generator=generator,entities=[],semantic_structure=dict(kind=kind,mechanism=mechanism,credit='finite retained control'),public_start=dict(goal=goal,tools=tools,max_calls=32),examiner=dict(initial_world=world,events=events,expected_effects=expected,private_markers=['examiner-only-control-outcome']),assistance=assistance,budget=dict(max_calls=32),source_requirements=dict(run=str(source),kind=kind,case=name,fingerprints=fingerprints,related_controls=read('examiner.json'))))
    return rows,registry

def register(source,kind,out):
    rows,registry=entries(source,kind)
    result=write_pack(out,rows,registry)
    if not result['manifest_valid']:raise ValueError(result['errors'])
    return result

def load_entry(entry):
    sources=entry['source_requirements'];root=Path(sources['run'])
    for name,digest in sources['fingerprints'].items():
        if file_sha(root/name)!=digest:raise ValueError('Retained artifact changed')
    # Reconstruct from retained evidence: a rewritten manifest alone cannot change goals/outcomes.
    rows,_=entries(root,sources['kind'])
    matches=[e for e in rows if e['id']==entry['id']]
    if matches!=[entry]:raise ValueError('Registration differs from retained source')
    return copy.deepcopy(entry['examiner'])
