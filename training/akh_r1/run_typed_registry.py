#!/usr/bin/env python3
"""Frozen-weight, observed-choice constrained registry episode diagnostic."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from bounded_process import ExperimentStop,run_bounded
from build_curriculum import digest,now
from build_registry_lessons import initial_messages
from command_registry import CommandRegistry,registry_records,registry_tools
from compact_evidence import CompactEvidence,compact_messages
from environment import Graph
from protocol import parse_action
from run_binding_kata import fixture_for,load,write
from run_exposure import unchanged
from run_registry_cycle import cycle_score,dispatch,make_world,oracle
from typed_registry_choices import TypedChoices,TokenChoiceMask,wire

PHASES=['acquire','cold_graph','cold_empty','seeded_cold']
ASSISTANCE=('Finite canonical calls and token-prefix constraints from public goal IDs, successful help/version receipts, '
    'candidate-retrieved or registered command records and observed identifier fields. Only documented discovery and registry invocation; '
    'no arbitrary terminal composition. Generic report/finish text. The candidate selects every operation, source line, target and stopping point. '
    'Combined typed-menu and constrained-decoding diagnostic, not independent release certification.')


def prepare(args):
    parent=args.parent.resolve();baseline=load(parent/'plan.json')
    if load(parent/'status.json')['stage']!='completed' or not unchanged(baseline['files']):raise ValueError('Unchanged completed baseline required')
    root=args.out.resolve();root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'):shutil.copy2(path,source/path.name)
    shutil.copy2(baseline['bridge'],source/'akh_episode_bridge')
    oracle(root,source/'akh_episode_bridge',baseline['image'])
    # Every actual reference action must be available using only its preceding
    # public messages and returned receipts, including fresh-process recall.
    reference_checks=[]
    for phase in ['acquire','cold_graph','cold_empty']:
        e=load(root/'reference'/phase/'evidence.json');_,target=make_world(83,phase)
        messages=initial_messages(target,['fano','zuri'],phase)
        choices=TypedChoices(['fano','zuri'],messages[1]['content'])
        for index,t in enumerate(e['turns']):
            request=t['wire_request'];available=[x['request'] for x in choices.choices()]
            reference_checks.append(dict(phase=phase,turn=index,request_available=request in available))
            choices.observe(request,t['response'],t.get('request',{}).get('arguments',{}).get('stages'))
    if not all(c['request_available'] for c in reference_checks):raise ValueError('Reference action missing from observed choices')
    write(root/'menu-reference-audit.json',dict(passed=True,checks=reference_checks))
    if args.context_mode=='compact':
        from audit_compact_preflight import preflight
        preflight(parent,root/'compact-preflight.json')
    files=dict(baseline['files'])
    files.update({str(p):digest(p) for p in root.rglob('*') if p.is_file()})
    files.update({str(parent/p):digest(parent/p) for p in ['plan.json','results.json','status.json']})
    adapter=str(args.adapter.resolve()) if getattr(args,'adapter',None) else baseline['adapter']
    if getattr(args,'adapter',None):
        for name in ['adapters.safetensors','adapter_config.json']:
            path=Path(adapter)/name;files[str(path)]=digest(path)
    plan=dict(created_at=now(),model=baseline['model'],adapter=adapter,image=baseline['image'],
        bridge=str(source/'akh_episode_bridge'),files=files,phases=PHASES,parent_baseline=str(parent),
        seed=83,sampling_seed=83,temperature=.7,max_turns=24,max_new_tokens=192,max_context=8192,
        timeout_seconds=2400,phase_timeout_seconds=600,min_free_bytes=5*2**30,lock_file=str(args.lock_file.resolve()),
        training_updates=0,certification=False,assistance=ASSISTANCE,context_mode=args.context_mode,
        context_assistance='Deduplicated reconstructible tool receipts plus literal relation index; all operator messages retained. Not a High Grug implementation.' if args.context_mode=='compact' else None,
        seeded_assistance='Separate oracle-prepared persisted records in a fresh process; not candidate-acquired knowledge.',
        automatic_retries=False,new_cloud_gpu_allocations=0)
    write(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
    write(root/'status.json',dict(stage='prepared',updated_at=now()))
    print(json.dumps(dict(stage='prepared',phases=PHASES,reference_checks=len(reference_checks))),flush=True)


def run_phase(root,phase):
    import mlx.core as mx
    from mlx_lm import generate,load as load_model
    from mlx_lm.sample_utils import make_sampler
    p=load(root/'plan.json');dest=root/phase;dest.mkdir();effective='cold_graph' if phase=='seeded_cold' else phase
    before,target=make_world(p['seed'],effective)
    if phase=='cold_graph':shutil.copytree(root/'acquire/graph',dest/'graph')
    if phase=='seeded_cold':shutil.copytree(root/'reference/acquire/graph',dest/'graph')
    graph=Graph(p['bridge'],dest/'graph',dest/'engine.log');fixture=fixture_for(p['image'],before['contract'],'typed-candidate')
    commands=['fano','zuri'];registry=CommandRegistry(graph,commands)
    fixture.reset(before);write(dest/'before.json',before);write(dest/'graph-before.json',graph.call('exam_export'))
    messages=initial_messages(target,commands,effective);choices=TypedChoices(commands,messages[1]['content'])
    tools=registry_tools(commands);turns=[];namespace=hashlib.sha256((root.name+'/'+phase).encode()).hexdigest()[:20]
    state=CompactEvidence()
    try:
        model,tokenizer=load_model(p['model'],adapter_path=p['adapter']);mx.random.seed(p['sampling_seed'])
        sampler=make_sampler(temp=p['temperature'],top_p=.8,top_k=20)
        with (dest/'trace.jsonl').open('w') as log:
            for index in range(p['max_turns']):
                if index==2:messages.append(dict(role='user',content='Operator correction: continue the authorized repair. Do not restart workers. Report observations and apply a bounded action.'))
                catalog=choices.choices();requests=[c['request'] for c in catalog]
                menu=dict(role='user',content='Choose one of these currently available calls. Select the next useful operation and actual identifier yourself. '
                    'These choices reflect observed documentation, identifiers and retrieved records; they do not indicate which action solves the goal.\n'+json.dumps(catalog))
                context_mode=p.get('context_mode','history')
                actual_input=compact_messages(messages,state,menu) if context_mode=='compact' else messages+[menu]
                prompt=tokenizer.apply_chat_template(actual_input,tools=tools,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                history_tokens=len(tokenizer.apply_chat_template(messages+[menu],tools=tools,tokenize=True,add_generation_prompt=True,enable_thinking=False))
                if len(prompt)>p['max_context']:
                    write(dest/'context-limit.json',dict(tokens=len(prompt),limit=p['max_context']));break
                mask=TokenChoiceMask(tokenizer,requests)
                raw=generate(model,tokenizer,prompt=prompt,max_tokens=p['max_new_tokens'],sampler=sampler,logits_processors=[mask])
                request=parse_action(raw,'native')
                if request not in requests:raise ValueError('Decoder returned a call outside the frozen turn menu')
                old=fixture.state();rid=f'episode:{namespace}#turn-{index:04d}'
                row=dict(index=index,at=now(),input=copy.deepcopy(actual_input),raw=raw,prompt_tokens=len(prompt),
                    wire_request=request,request=request,parsed=True,menu=catalog,evidence=choices.evidence(),
                    forced_token_steps=mask.forced_steps,decision_token_steps=mask.decision_steps,registry_invocation=False,
                    context_mode=context_mode,history_prompt_tokens=history_tokens,
                    compact_state=state.view() if context_mode=='compact' else None)
                try:
                    response,execution,stages=dispatch(registry,fixture,request,rid)
                    row.update(execution=execution,registry_invocation=execution is not None)
                    if stages:row['request']=dict(tool='terminal',arguments=dict(stages=stages))
                except (ValueError,KeyError,TypeError) as error:
                    response=dict(success=False,error=str(error),evidence_id=rid);stages=None
                row.update(response=response,state_changed=fixture.state()!=old);turns.append(row)
                choices.observe(request,response,stages)
                state.observe(request,response)
                log.write(json.dumps(row)+'\n');log.flush()
                messages.extend([dict(role='assistant',content=raw),dict(role='tool',content=json.dumps(response))])
                write(root/'progress.json',dict(phase=phase,turn=index+1,tool=request['tool'],pid=os.getpid(),updated_at=now()))
                print(json.dumps(dict(phase=phase,turn=index+1,tool=request['tool'],success=response.get('success'),mutated=row['state_changed'])),flush=True)
                if request['tool']=='finish':break
        after=fixture.state();write(dest/'after.json',after);write(dest/'graph-after.json',graph.call('exam_export'))
        score=cycle_score(before,after,target,turns,effective)
        score.update(phase=phase,created_at=now(),records=len(registry_records(graph)),peak_mlx_bytes=mx.get_peak_memory(),
                     cold_process_pid=os.getpid(),assistance=p['assistance'],seeded_assistance=p['seeded_assistance'] if phase=='seeded_cold' else None,
                     generic_finish=bool(turns and turns[-1]['wire_request']['tool']=='finish'),candidate_claims_scored=False,
                     context_mode=p.get('context_mode','history'),context_assistance=p.get('context_assistance'))
        write(dest/'scores.json',score)
    finally:fixture.close();graph.close()


def execute(root):
    p=load(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip() or not unchanged(p['files']):raise ValueError('Frozen inputs changed')
    if load(root/'status.json')['stage']!='prepared':raise ValueError('Refusing replay')
    with Path(p['lock_file']).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(stage='running',pid=os.getpid(),started_at=now(),updated_at=now(),completed=[])
        write(root/'status.json',status);deadline=time.monotonic()+p['timeout_seconds']
        def stop(sig,frame):raise ExperimentStop(f'Signal {sig}')
        signal.signal(signal.SIGTERM,stop)
        try:
            for phase in p['phases']:
                def heartbeat(pid):
                    if shutil.disk_usage(root).free<p['min_free_bytes']:raise ExperimentStop('Disk below five GiB')
                    status.update(phase=phase,child_pid=pid,updated_at=now());write(root/'status.json',status)
                remaining=deadline-time.monotonic()
                if remaining<=0:raise ExperimentStop('Experiment deadline')
                rc=run_bounded([sys.executable,str(root/'source/run_typed_registry.py'),'phase','--out',str(root),'--phase',phase],
                               root/(phase+'.log'),min(remaining,p['phase_timeout_seconds']),heartbeat)
                if rc:raise RuntimeError(f'{phase} exited {rc}')
                status['completed'].append(load(root/phase/'scores.json'))
            if not unchanged(p['files']):raise ValueError('Inputs changed during run')
            write(root/'results.json',dict(completed_at=now(),scores=status['completed'],input_hashes_unchanged=True,certification=False))
            status.update(stage='completed',phase=None,updated_at=now());status.pop('child_pid',None);write(root/'status.json',status)
        except BaseException as error:
            status.update(stage='failed',error=str(error),updated_at=now());write(root/'status.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','execute','phase']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parent',type=Path);p.add_argument('--lock-file',type=Path);p.add_argument('--phase',choices=PHASES)
    p.add_argument('--context-mode',choices=['history','compact'],default='history')
    p.add_argument('--adapter',type=Path,help='Explicit newly trained adapter; freeze its weights alongside the completed baseline')
    a=p.parse_args()
    if a.mode=='prepare':prepare(a)
    elif a.mode=='execute':execute(a.out.resolve())
    else:
        def stop(sig,frame):raise ExperimentStop(f'Signal {sig}')
        signal.signal(signal.SIGTERM,stop);run_phase(a.out.resolve(),a.phase)
