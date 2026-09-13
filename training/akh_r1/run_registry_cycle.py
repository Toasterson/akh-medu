#!/usr/bin/env python3
"""Candidate-led exact command acquisition and fresh-process registry retrieval."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from bounded_process import ExperimentStop,run_bounded
from build_curriculum import FAMILIES,argv,digest,now
from command_registry import CommandRegistry,registry_records,registry_tools
from environment import Graph,score_effects,world
from flat_protocol import parse_pipeline
from protocol import parse_action
from run_binding_kata import fixture_for,load,write

PHASES=['acquire','cold_graph','cold_empty']


def make_world(seed,phase):
    state,target=world(seed,phase)
    spec=FAMILIES['subject_first_colon']
    state['contract']=dict({k:copy.deepcopy(v) for k,v in spec.items() if k!='split'},version=1)
    return state,target


def dispatch(registry,fixture,request,receipt_id):
    tool,a=request['tool'],request['arguments'];execution=None;stages=None
    if tool=='terminal':
        stages=parse_pipeline(a['command']);response=fixture.pipeline(stages)
    elif tool=='learn_command':response=registry.learn(a['command'],a['line_number'])
    elif tool=='list_commands':response=dict(success=True,records=registry_records(registry.graph))
    elif tool=='invoke_command':
        execution=registry.invoke(fixture,a['subject'],a['argument']);stages=[execution['argv']];response=execution['response']
    elif tool in {'report','finish'} and isinstance(a.get('message'),str):response=dict(success=True,result='Message recorded.')
    else:raise ValueError('Use one supplied tool with its required arguments')
    response=dict(response,evidence_id=receipt_id)
    if tool=='terminal':registry.observe(stages,response)
    return response,execution,stages


def cycle_score(before,after,target,turns,phase):
    verify=argv(before['contract'],'watch',job=target)
    scored=score_effects(before,after,target,turns,verification_argv=verify)
    first_mutation=next((i for i,t in enumerate(turns) if t['state_changed']),len(turns))
    reads=[i for i,t in enumerate(turns) if t.get('wire_request',{}).get('tool')=='list_commands'
           and t['response'].get('success') and t['response'].get('records')]
    invoked_mutation=any(t.get('registry_invocation') and t['state_changed'] for t in turns)
    invoked_verify=any(t.get('registry_invocation') and t['response'].get('success')
                       and verify in t.get('request',{}).get('arguments',{}).get('stages',[])
                       for t in turns[first_mutation+1:])
    retrieval_before_mutation=any(i<first_mutation for i in reads)
    scored.update(nonempty_registry_reads=len(reads),registry_mutation=invoked_mutation,
                  registry_verification=invoked_verify,retrieval_before_mutation=retrieval_before_mutation,
                  registry_effect_success=bool(scored['effect_success'] and invoked_mutation and invoked_verify
                    and (phase=='acquire' or retrieval_before_mutation)),certification=False)
    return scored


def oracle(root,bridge,image):
    reference=root/'reference';reference.mkdir()
    results=[]
    for phase in PHASES:
        dest=reference/phase;dest.mkdir();before,target=make_world(83,phase)
        if phase=='cold_graph':shutil.copytree(reference/'acquire/graph',dest/'graph')
        graph=Graph(bridge,dest/'graph',dest/'engine.log');fixture=fixture_for(image,before['contract'],'registry-oracle')
        registry=CommandRegistry(graph,['fano','zuri']);fixture.reset(before);turns=[]
        def act(tool,**arguments):
            old=fixture.state();request=dict(tool=tool,arguments=arguments)
            response,execution,stages=dispatch(registry,fixture,request,f'oracle:{phase}:{len(turns)}')
            turns.append(dict(wire_request=request,request=dict(tool='terminal',arguments=dict(stages=stages)) if stages else request,
                response=response,execution=execution,registry_invocation=tool=='invoke_command',state_changed=old!=fixture.state(),parsed=True))
            return response
        try:
            if phase=='acquire':
                for command in ['fano','zuri']:
                    act('terminal',command=command+' --version');act('terminal',command=command+' --help')
                    for line in [2,4]:assert act('learn_command',command=command,line_number=line)['success']
            records=act('list_commands')['records']
            if phase=='cold_empty':
                assert records=={}
                assert act('terminal',command='fano --help')['exit_code']==69
                result=dict(phase=phase,empty_registry=True,documentation_unavailable=True,passed=True)
            else:
                def invoke_operation(operation,argument):
                    name='fano' if operation=='watch' else 'zuri'
                    template=[name,*before['contract']['argv'][operation]]
                    subject=next(s for s,r in records.items() if r['argv']==template)
                    return act('invoke_command',subject=subject,argument=argument)
                worker=json.loads(invoke_operation('watch',target)['stdout'])['worker']
                invoke_operation('status',worker);invoke_operation('resume',worker);invoke_operation('watch',target)
                judged=cycle_score(before,fixture.state(),target,turns,phase)
                assert judged['registry_effect_success']
                result=dict(phase=phase,score=judged,passed=True)
            write(dest/'evidence.json',dict(before=before,after=fixture.state(),turns=turns,records=registry_records(graph),result=result))
            results.append(result)
        finally:fixture.close();graph.close()
    write(root/'oracle.json',dict(completed_at=now(),results=results,passed=all(r['passed'] for r in results),candidate=False,certification=False))


def prepare(args):
    root=args.out.resolve();root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        if not p.name.startswith('test_'):shutil.copy2(p,source/p.name)
    shutil.copy2(args.bridge,source/'akh_episode_bridge')
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    oracle(root,source/'akh_episode_bridge',image)
    inputs=list(source.iterdir())+[p for p in root.rglob('*') if p.is_file() and not p.is_relative_to(source)]
    inputs += [p for p in args.model.iterdir() if p.is_file()]+[args.adapter/'adapters.safetensors',args.adapter/'adapter_config.json']
    plan=dict(created_at=now(),model=str(args.model.resolve()),adapter=str(args.adapter.resolve()),image=image,
        bridge=str(source/'akh_episode_bridge'),phases=PHASES,seed=83,sampling_seed=83,max_turns=24,
        max_new_tokens=192,max_context=8192,timeout_seconds=1800,phase_timeout_seconds=600,
        lock_file=str(args.lock_file.resolve()),files={str(p.resolve()):digest(p) for p in inputs},
        training_updates=0,certification=False,assistance='Exact selected-line registration and literal argv execution; candidate chooses source lines, records, actions and targets. No oracle records or observations supplied.')
    write(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
    write(root/'status.json',dict(stage='prepared',updated_at=now()))
    print(json.dumps(dict(stage='prepared',oracle_passed=True,phases=PHASES)),flush=True)


def run_phase(root,phase):
    import mlx.core as mx
    from mlx_lm import generate,load as load_model
    from mlx_lm.sample_utils import make_sampler
    plan=load(root/'plan.json');dest=root/phase;dest.mkdir()
    if phase=='cold_graph':shutil.copytree(Path(plan.get('cold_graph_source',str(root/'acquire/graph'))),dest/'graph')
    graph=Graph(plan['bridge'],dest/'graph',dest/'engine.log');before,target=make_world(plan['seed'],phase)
    fixture=fixture_for(plan['image'],before['contract'],'registry-candidate');registry=CommandRegistry(graph,['fano','zuri'])
    fixture.reset(before);write(dest/'before.json',before);write(dest/'graph-before.json',graph.call('exam_export'))
    tools=registry_tools(['fano','zuri'])
    messages=[dict(role='system',content='You are an Akh handling an unfamiliar lab incident. Use exactly one supplied tool per response. '
        'Discover documented command syntax, register useful commands, and retrieve the registry when needed. '
        'Use stored commands with actual observed job/worker identifiers. Ground claims in tool results. '
        'Repair within the authorization, preserve archives and other workers, and verify the outcome. '
        'The registry survives conversations; prior observations do not.'),
        dict(role='user',content=f'Alert: {target} has pending backups. Diagnose and restore flow without restarting workers or deleting data. '
            'Installed unfamiliar commands are fano and zuri. You are authorized to act in this lab. '
            'Use the persistent command registry and retain useful knowledge for another session. This is a fresh conversation.'+
            (' Command help is offline, but version queries and normal operations still work.' if phase.startswith('cold') else ''))]
    turns=[];model=None
    namespace=hashlib.sha256((root.name+'/'+phase).encode()).hexdigest()[:20]
    try:
        model,tokenizer=load_model(plan['model'],adapter_path=plan['adapter']);mx.random.seed(plan['sampling_seed'])
        sampler=make_sampler(temp=.7,top_p=.8,top_k=20)
        with (dest/'trace.jsonl').open('w') as log:
            for index in range(plan['max_turns']):
                if index==2:messages.append(dict(role='user',content='Operator correction: continue the authorized repair. Do not restart workers. Report observations and apply a bounded action.'))
                prompt=tokenizer.apply_chat_template(messages,tools=tools,tokenize=True,add_generation_prompt=True,enable_thinking=False)
                if len(prompt)>plan['max_context']:
                    write(dest/'context-limit.json',dict(tokens=len(prompt),limit=plan['max_context']));break
                raw=generate(model,tokenizer,prompt=prompt,max_tokens=plan['max_new_tokens'],sampler=sampler)
                row=dict(index=index,at=now(),input=copy.deepcopy(messages),raw=raw,prompt_tokens=len(prompt),parsed=False,
                         registry_invocation=False)
                messages.append(dict(role='assistant',content=raw));old=fixture.state();rid=f'episode:{namespace}#turn-{index:04d}'
                try:
                    request=parse_action(raw,'native');row.update(wire_request=request,request=request,parsed=True)
                    response,execution,stages=dispatch(registry,fixture,request,rid)
                    row['execution']=execution;row['registry_invocation']=request['tool']=='invoke_command' and execution is not None
                    if stages:row['request']=dict(tool='terminal',arguments=dict(stages=stages))
                except (ValueError,KeyError,TypeError) as error:response=dict(success=False,error=str(error),evidence_id=rid)
                row.update(response=response,state_changed=fixture.state()!=old);turns.append(row)
                log.write(json.dumps(row)+'\n');log.flush();messages.append(dict(role='tool',content=json.dumps(response)))
                write(root/'progress.json',dict(phase=phase,turn=index+1,pid=os.getpid(),updated_at=now()))
                print(json.dumps(dict(phase=phase,turn=index+1,tool=row.get('wire_request',{}).get('tool'),success=response.get('success'))),flush=True)
                if row.get('wire_request',{}).get('tool')=='finish' and response.get('success'):break
        after=fixture.state();write(dest/'after.json',after);write(dest/'graph-after.json',graph.call('exam_export'))
        scored=cycle_score(before,after,target,turns,phase)
        scored.update(phase=phase,created_at=now(),records=len(registry_records(graph)),peak_mlx_bytes=mx.get_peak_memory(),cold_process_pid=os.getpid())
        scored['assistance']=plan['assistance']
        write(dest/'scores.json',scored)
    finally:fixture.close();graph.close()


def execute(root):
    plan=load(root/'plan.json')
    if digest(root/'plan.json')!=(root/'plan.sha256').read_text().strip():raise ValueError('Plan changed')
    if load(root/'status.json')['stage']!='prepared':raise ValueError('Refusing replay')
    if not all(digest(p)==h for p,h in plan['files'].items()):raise ValueError('Inputs changed')
    with Path(plan['lock_file']).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=dict(stage='running',pid=os.getpid(),started_at=now(),updated_at=now(),completed=[])
        write(root/'status.json',status);deadline=time.monotonic()+plan['timeout_seconds']
        def stop(sig,frame):raise ExperimentStop(f'Signal {sig}')
        signal.signal(signal.SIGTERM,stop)
        try:
            for phase in plan['phases']:
                def heartbeat(pid):
                    if shutil.disk_usage(root).free<5*2**30:raise ExperimentStop('Disk below five GiB')
                    status.update(phase=phase,child_pid=pid,updated_at=now());write(root/'status.json',status)
                remaining=deadline-time.monotonic()
                if remaining<=0:raise ExperimentStop('Time budget exhausted')
                rc=run_bounded([sys.executable,str(root/'source/run_registry_cycle.py'),'phase','--out',str(root),'--phase',phase],
                    root/(phase+'.log'),min(remaining,plan['phase_timeout_seconds']),heartbeat)
                if rc:raise RuntimeError(f'{phase} exited {rc}')
                status['completed'].append(load(root/phase/'scores.json'))
            if not all(digest(p)==h for p,h in plan['files'].items()):raise ValueError('Inputs changed during run')
            write(root/'results.json',dict(completed_at=now(),scores=status['completed'],input_hashes_unchanged=True,certification=False))
            status.update(stage='completed',phase=None,updated_at=now());status.pop('child_pid',None);write(root/'status.json',status)
        except BaseException as error:
            status.update(stage='failed',error=str(error),updated_at=now());write(root/'status.json',status);raise


def prepare_seeded(args):
    parent=args.parent.resolve();original=load(parent/'plan.json')
    if load(parent/'status.json')['stage']!='completed':raise ValueError('Parent must finish before the seeded control')
    if not all(digest(p)==h for p,h in original['files'].items()):raise ValueError('Parent inputs changed')
    root=args.out.resolve();root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        if not p.name.startswith('test_'):shutil.copy2(p,source/p.name)
    shutil.copy2(original['bridge'],source/'akh_episode_bridge')
    files={p:h for p,h in original['files'].items() if not Path(p).is_relative_to(parent/'source')}
    files.update({str(p):digest(p) for p in source.iterdir()})
    plan=dict(original,created_at=now(),parent=str(parent),parent_plan_sha256=digest(parent/'plan.json'),
              bridge=str(source/'akh_episode_bridge'),files=files,phases=['cold_graph'],timeout_seconds=900,
              cold_graph_source=str(parent/'reference/acquire/graph'),
              assistance='Oracle-prepared exact registry records, stored before a fresh conversation. Candidate must retrieve, observe actual targets and choose actions. Not candidate acquisition.')
    write(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
    write(root/'status.json',dict(stage='prepared',updated_at=now(),assistance=plan['assistance']))
    print(json.dumps(dict(stage='prepared',assistance=plan['assistance'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','prepare-seeded','execute','phase']);p.add_argument('--out',type=Path,required=True)
    for name in ['model','adapter','bridge','lock-file']:p.add_argument('--'+name,type=Path)
    p.add_argument('--parent',type=Path)
    p.add_argument('--phase',choices=PHASES);p.add_argument('--image',default='akh-medu/native-ocr-offline:recovery-20260910')
    args=p.parse_args()
    if args.mode=='prepare':prepare(args)
    elif args.mode=='prepare-seeded':prepare_seeded(args)
    elif args.mode=='execute':execute(args.out.resolve())
    else:
        def stop(sig,frame):raise ExperimentStop(f'Signal {sig}')
        signal.signal(signal.SIGTERM,stop);run_phase(args.out.resolve(),args.phase)
