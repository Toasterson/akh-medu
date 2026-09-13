#!/usr/bin/env python3
"""Paired next-action probes of previously observed evidence; no tool execution."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys

from bounded_process import ExperimentStop,run_bounded
from build_curriculum import digest,now
from command_registry import registry_tools
from compact_evidence import CompactEvidence,compact_messages
from protocol import parse_action
from run_binding_kata import load,write
from run_exposure import unchanged
from typed_registry_choices import TypedChoices,TokenChoiceMask


def prepare(args):
    parent=args.parent.resolve();base=load(parent/'plan.json')
    if load(parent/'status.json')['stage']!='completed' or not unchanged(base['files']):raise ValueError('Completed unchanged parent required')
    root=args.out.resolve();root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'):shutil.copy2(path,source/path.name)
    trace=parent/'seeded_cold/trace.jsonl';turns=[json.loads(s) for s in trace.read_text().splitlines()]
    state=CompactEvidence();receipts=[];cases=[];checks=[]
    choices=TypedChoices(['fano','zuri'],turns[0]['input'][1]['content'])
    for t in turns:
        if t['index'] in [5,8,11,15]:
            compact=compact_messages(t['input'][:-1],state,t['input'][-1])
            checks.append(dict(turn=t['index'],passed=choices.choices()==t['menu'] and choices.evidence()==t['evidence'] and state.reconstruct(state.view())==receipts))
            cases.append(dict(turn=t['index'],history=t['input'],compact=compact,menu=t['menu'],state=state.view(),prior_receipts=copy.deepcopy(receipts)))
        request=t['wire_request'];response=t['response']
        choices.observe(request,response,t.get('request',{}).get('arguments',{}).get('stages'))
        state.observe(request,response);receipts.append(dict(request=request,response=response))
    if len(cases)!=4 or not all(c['passed'] for c in checks):raise ValueError('Causal paired inputs failed')
    write(root/'cases.json',cases);write(root/'preflight.json',dict(passed=True,checks=checks))
    files=dict(base['files']);files[str(trace)]=digest(trace)
    files.update({str(p):digest(p) for p in root.rglob('*') if p.is_file()})
    files.update({str(parent/p):digest(parent/p) for p in ['plan.json','results.json','status.json']})
    plan=dict(created_at=now(),model=base['model'],adapter=base['adapter'],parent=str(parent),files=files,
              cases=[5,8,11,15],conditions=['history','compact'],temperatures=[0,.7],sampling_seed=83,
              max_new_tokens=192,max_context=8192,timeout_seconds=600,min_free_bytes=5*2**30,
              lock_file=str(args.lock_file.resolve()),decisions=16,training_updates=0,actual_tools_executed=False,
              certification=False,automatic_retries=False,
              limitation='Selected previous development failures, independently sampled next decisions. No autonomous episode or capacity verdict. Both presentations preserve the same prior receipts and current menu.')
    write(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
    write(root/'status.json',dict(stage='prepared',updated_at=now()));print('Prepared 16 paired next-action decisions',flush=True)


def infer(root):
    import mlx.core as mx
    from mlx_lm import generate,load as load_model
    from mlx_lm.sample_utils import make_sampler
    p=load(root/'plan.json');model,tokenizer=load_model(p['model'],adapter_path=p['adapter']);records=[]
    with (root/'trace.jsonl').open('x') as log:
        for case in load(root/'cases.json'):
            for temperature in p['temperatures']:
                for condition in p['conditions']:
                    messages=case[condition];requests=[c['request'] for c in case['menu']]
                    prompt=tokenizer.apply_chat_template(messages,tools=registry_tools(['fano','zuri']),tokenize=True,add_generation_prompt=True,enable_thinking=False)
                    if len(prompt)>p['max_context']:raise ValueError('Context budget exceeded')
                    mx.random.seed(p['sampling_seed']);mask=TokenChoiceMask(tokenizer,requests)
                    raw=generate(model,tokenizer,prompt=prompt,max_tokens=p['max_new_tokens'],sampler=make_sampler(temp=temperature,top_p=.8,top_k=20),logits_processors=[mask])
                    request=parse_action(raw,'native')
                    if request not in requests:raise ValueError('Request outside observed menu')
                    row=dict(turn=case['turn'],temperature=temperature,condition=condition,input=messages,raw=raw,request=request,prompt_tokens=len(prompt),menu_valid=True)
                    records.append(row);log.write(json.dumps(row)+'\n');log.flush()
                    write(root/'progress.json',dict(updated_at=now(),decisions=len(records),total=p['decisions'],pid=os.getpid()))
                    print(json.dumps({k:v for k,v in row.items() if k not in ['input','raw']}),flush=True)
    write(root/'results.json',dict(completed_at=now(),decisions=len(records),all_menu_valid=all(r['menu_valid'] for r in records),records=records,peak_mlx_bytes=mx.get_peak_memory(),training_updates=0,actual_tools_executed=False,certification=False))


def execute(root):
    p=load(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip() or not unchanged(p['files']):raise ValueError('Frozen inputs changed')
    if load(root/'status.json')['stage']!='prepared':raise ValueError('Refusing replay')
    with Path(p['lock_file']).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(stage='running',pid=os.getpid(),started_at=now(),updated_at=now())
        def stop(sig,frame):raise ExperimentStop(f'Signal {sig}')
        signal.signal(signal.SIGTERM,stop)
        try:
            def heartbeat(pid):
                if shutil.disk_usage(root).free<p['min_free_bytes']:raise ExperimentStop('Disk below five GiB')
                status.update(child_pid=pid,updated_at=now());write(root/'status.json',status)
            rc=run_bounded([sys.executable,str(root/'source/run_compact_probe.py'),'infer','--out',str(root)],root/'inference.log',p['timeout_seconds'],heartbeat)
            if rc:raise RuntimeError(f'Inference exited {rc}')
            if not unchanged(p['files']):raise ValueError('Inputs changed')
            result=load(root/'results.json');cases={c['turn']:c for c in load(root/'cases.json')}
            assert result['decisions']==p['decisions']
            assert {(r['turn'],r['condition'],r['temperature']) for r in result['records']}=={(t,c,v) for t in p['cases'] for c in p['conditions'] for v in p['temperatures']}
            for row in result['records']:
                case=cases[row['turn']]
                assert row['input']==case[row['condition']] and parse_action(row['raw'],'native')==row['request']
                assert row['request'] in [c['request'] for c in case['menu']]
            write(root/'artifact-audit.json',dict(passed=True,checked_at=now(),frozen_files=len(p['files']),checked_decisions=len(result['records']),causal_preflight=load(root/'preflight.json')['passed']))
            status.update(stage='completed',updated_at=now());status.pop('child_pid',None);write(root/'status.json',status)
        except BaseException as error:
            status.update(stage='failed',error=str(error),updated_at=now());write(root/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','execute','infer'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--lock-file',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='execute':execute(a.out.resolve())
    else:infer(a.out.resolve())
