#!/usr/bin/env python3
"""Executed acquisition, cold recall and honest blocking lessons on the registry API."""
import argparse
import copy
import csv
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path

from build_binding_lessons import distinct_world
from build_curriculum import FAMILIES, digest, now
from command_registry import CommandRegistry, registry_records, registry_tools
from environment import Graph
from run_binding_kata import fixture_for, load, write
from run_registry_cycle import cycle_score, dispatch

# Development choices fixed before model evaluation. Families remain disjoint.
SEEDS = {'train': [521, 541], 'validation': [601, 607], 'test': [701, 709]}
PHASES = ['acquire', 'cold_graph', 'cold_empty']
SYSTEM = ('You are an Akh handling an unfamiliar lab incident. Use exactly one supplied tool per response. '
    'Discover documented command syntax, register useful commands, and retrieve the registry when needed. '
    'Use stored commands with actual observed job/worker identifiers. Ground claims in tool results. '
    'Repair within the authorization, preserve archives and other workers, and verify the outcome. '
    'The registry survives conversations; prior observations do not.')


def initial_messages(target, commands, phase):
    return [dict(role='system', content=SYSTEM), dict(role='user', content=
        f'Alert: {target} has pending backups. Diagnose and restore flow without restarting workers or deleting data. '
        f'Installed unfamiliar commands are {commands[0]} and {commands[1]}. You are authorized to act in this lab. '
        'Use the persistent command registry and retain useful knowledge for another session. This is a fresh conversation.'+
        (' Command help is offline, but version queries and normal operations still work.' if phase.startswith('cold') else ''))]


def assistant_message(request):
    return dict(role='assistant', content='<tool_call>\n'+json.dumps(
        dict(name=request['tool'], arguments=request['arguments']))+'\n</tool_call>')


def rows_for(episode):
    return [dict(messages=t['input']+[assistant_message(t['wire_request'])], tools=episode['tools'],
                 episode_id=episode['id'], decision_id=f'turn-{i:04d}', family=episode['family'],
                 assisted_component=True, supervision='executed authored reference; prior error calls masked')
            for i,t in enumerate(episode['turns']) if t['supervise']]


def build_cycle(root, bridge, image, family, seed, *, wrong_binding_recovery=False):
    contract = dict({k:copy.deepcopy(v) for k,v in FAMILIES[family].items() if k!='split'}, version=1)
    commands = [contract['observer'], contract['controller']]
    cycle = root/'episodes'/f'{family}-{seed}'; cycle.mkdir()
    paths = []
    for phase_index, phase in enumerate(PHASES):
        dest = cycle/phase; dest.mkdir()
        before,target = distinct_world(family, seed+10000*phase_index)
        before.update(contract=contract, docs_available=phase=='acquire')
        if phase=='cold_graph': shutil.copytree(cycle/'acquire/graph', dest/'graph')
        graph = Graph(bridge, dest/'graph', dest/'engine.log')
        fixture = fixture_for(image, contract, 'registry-lesson')
        try:
            fixture.reset(before)
            registry = CommandRegistry(graph, commands)
            initial_graph = graph.call('exam_export')
            messages = initial_messages(target, commands, phase); turns = []; repair_start=None; repair_before=None
            identity = f'{family}-{seed}-{phase}'
            opaque = hashlib.sha256(identity.encode()).hexdigest()[:20]

            def act(tool, supervise=True, **arguments):
                if len(turns)==2:
                    messages.append(dict(role='user',content='Operator correction: continue the authorized repair. Do not restart workers. Report observations and apply a bounded action.'))
                request = dict(tool=tool, arguments=arguments); old = fixture.state()
                row = dict(input=copy.deepcopy(messages), wire_request=request, request=request,
                           supervise=supervise, parsed=True, registry_invocation=False)
                receipt = f'episode:{opaque}#turn-{len(turns):04d}'
                try:
                    response,execution,stages = dispatch(registry,fixture,request,receipt)
                    row.update(execution=execution, registry_invocation=execution is not None)
                    if stages: row['request']=dict(tool='terminal',arguments=dict(stages=stages))
                except (ValueError,KeyError,TypeError) as error:
                    response=dict(success=False,error=str(error),evidence_id=receipt)
                row.update(response=response,state_changed=old!=fixture.state())
                turns.append(row); messages.extend([assistant_message(request),dict(role='tool',content=json.dumps(response))])
                if supervise and tool not in {'terminal'}: assert response['success'], response
                return response

            if phase=='acquire':
                for command in commands:
                    act('terminal',command=command+' --version')
                    # These errors are context only, never optimizer targets.
                    if seed%2: assert not act('learn_command',supervise=False,command=command,line_number=2)['success']
                    receipt=act('terminal',command=command+' --help'); assert receipt['success']
                    for operation in (['catalog','watch'] if command==commands[0] else ['status','resume']):
                        syntax=' '.join([command,*contract['argv'][operation]])
                        line=receipt['stdout'].splitlines().index(syntax)+1
                        act('learn_command',command=command,line_number=line)
            records=act('list_commands')['records']
            if phase=='cold_empty':
                assert records=={}
                assert act('terminal',command=commands[0]+' --help')['exit_code']==69
                act('finish',message='The registry is empty and command help is unavailable. I have not changed the lab or verified a repair. Restore documentation access so I can learn the commands and diagnose safely.')
            else:
                def invoke_operation(operation,argument,supervise=True):
                    name=commands[0] if operation in ['catalog','watch'] else commands[1]
                    template=[name,*contract['argv'][operation]]
                    subject=next(s for s,r in records.items() if r['argv']==template)
                    return act('invoke_command',supervise=supervise,subject=subject,argument=argument)
                if phase=='cold_graph' and seed%2:
                    assert not act('invoke_command',supervise=False,subject=commands[0],argument=target)['success']
                    records=act('list_commands')['records']
                inventory=invoke_operation('catalog','');assert inventory['success']
                if wrong_binding_recovery and phase=='cold_graph':
                    # An intentionally wrong, executed action is context only.
                    # Choose its argument from the returned inventory, never an
                    # unobserved examiner-only identifier. Preserve the failure
                    # in the full-episode gate; score recovery separately.
                    for delimiter in ['\t',',',':']:
                        reader=csv.DictReader(io.StringIO(inventory['stdout']),delimiter=delimiter)
                        if reader.fieldnames and {'job','worker'}<=set(reader.fieldnames):
                            observed=list(reader);break
                    wrong=next(row['worker'] for row in observed if row['job']!=target)
                    assert invoke_operation('resume',wrong,supervise=False)['success']
                    turns[-1]['mistake']='wrong_worker_binding'
                    messages.append(dict(role='user',content='Operator correction: that action addressed a different job. Re-read the alerted job, bind its observed worker, and verify the repair. Do not repeat the earlier worker choice or restart workers.'))
                    repair_start=len(turns);repair_before=fixture.state()
                    records=act('list_commands')['records']
                observation=json.loads(invoke_operation('watch',target)['stdout'])
                worker=observation['worker']
                assert json.loads(invoke_operation('status',worker)['stdout'])['mode']=='paused'
                assert invoke_operation('resume',worker)['success']
                verified=json.loads(invoke_operation('watch',target)['stdout'])
                assert verified['pending']==0
                act('finish',message=f'Fresh observation for {target}: pending={verified["pending"]}, archives={verified["archives"]}. I resumed {worker}, made no restart or deletion, and verified flow with a fresh observation.')
            after=fixture.state(); score=cycle_score(before,after,target,turns,phase)
            repair_score=None
            if repair_start is not None:
                suffix_before=copy.deepcopy(repair_before);suffix_after=copy.deepcopy(after)
                suffix_after['audit']=suffix_after['audit'][len(suffix_before['audit']):];suffix_before['audit']=[]
                repair_score=cycle_score(suffix_before,suffix_after,target,turns[repair_start:],phase)
                assert repair_score['registry_effect_success'] and not score['correct_target'] and not score['effect_success']
            elif phase!='cold_empty': assert score['registry_effect_success'], score
            else: assert before==after
            episode=dict(id=identity,family=family,split=FAMILIES[family]['split'],phase=phase,seed=seed,
                tools=registry_tools(commands),before=before,after=after,target=target,turns=turns,
                graph_before=initial_graph,graph_after=graph.call('exam_export'),records=registry_records(graph),
                score=score,authored_reference=True,certification=False,
                engine_pid=graph.process.pid,repair_start=repair_start,repair_before=repair_before,repair_score=repair_score)
            path=dest/'episode.json'; write(path,episode); paths.append(str(path))
        finally: fixture.close(); graph.close()
    return paths


def prepare(args):
    root=args.out.resolve();root.mkdir(exist_ok=False)
    for name in ['episodes','source','corpus']: (root/name).mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'): shutil.copy2(path,root/'source'/path.name)
    bridge=root/'source/akh_episode_bridge';shutil.copy2(args.bridge,bridge)
    image=subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    paths=[];rows={k:[] for k in ['train','valid','test']}
    for family,spec in FAMILIES.items():
        split='valid' if spec['split']=='validation' else spec['split']
        for seed in SEEDS[spec['split']]:
            new=build_cycle(root,bridge,image,family,seed);paths.extend(new)
            for path in new: rows[split].extend(rows_for(load(path)))
            write(root/'status.json',dict(stage='building',updated_at=now(),episodes=len(paths)))
    rehearsal=args.rehearsal.resolve();old_hashes={}
    for split,new in rows.items():
        path=rehearsal/(split+'.jsonl');old_hashes[str(path)]=digest(path)
        old=[json.loads(s) for s in path.read_text().splitlines()]
        (root/'corpus'/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old+new))
    m=dict(schema='akh-r1-registry-lessons-v1',created_at=now(),image=image,bridge=str(bridge),episodes=paths,
           registry_rows={s:len(r) for s,r in rows.items()},rehearsal_files=old_hashes,
           assistance='Authored, executed acquisition and fresh Engine recall lessons. No model capability result.',
           files={str(p):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='status.json'},
           training_started=False,certification=False)
    write(root/'manifest.json',m);(root/'manifest.sha256').write_text(digest(root/'manifest.json')+'\n')
    write(root/'status.json',dict(stage='prepared',updated_at=now(),episodes=len(paths),registry_rows=m['registry_rows']))
    print(json.dumps(dict(stage='prepared',episodes=len(paths),registry_rows=m['registry_rows'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['out','bridge','rehearsal']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image',default='akh-medu/native-ocr-offline:recovery-20260910')
    prepare(p.parse_args())
