#!/usr/bin/env python3
"""Bounded R0 public-interface and four-domain reference integration run."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess

from build_curriculum import now,digest
from environment import Graph,Fixture
from public_episode import PublicEpisode


def write(path,obj):
    path=Path(path);q=path.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2)+'\n');q.replace(path)


def candidate(image, script, port, name):
    args=['docker','run','--rm','-i','--name',name,'--network','none','--read-only',
          '--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534',
          '--memory','128m','--cpus','1','--pids-limit','24',
          '--mount',f'type=bind,src={script.resolve()},dst=/candidate.py,readonly',
          '--entrypoint','python3',image,'/candidate.py']
    process=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    try:
        process.stdin.write(json.dumps(port.start())+'\n');process.stdin.flush()
        while port.finished is None:
            if not selector.select(30):raise TimeoutError('Candidate response deadline')
            line=process.stdout.readline(65537)
            if not line or len(line)>65536:raise RuntimeError('Invalid candidate output')
            response=port.call(json.loads(line));process.stdin.write(json.dumps(response)+'\n');process.stdin.flush()
        process.stdin.close();code=process.wait(timeout=10)
        if code:raise RuntimeError('Candidate exited unsuccessfully')
        return dict(exit_code=code,docker_name=name,public_start=port.start(),finished=port.finished)
    finally:
        selector.close()
        if process.poll() is None:
            subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=20)
            process.kill();process.wait()
        for stream in [process.stdin,process.stdout,process.stderr]:
            if not stream.closed:stream.close()


def graph_cases():
    return [dict(id='contact_rotation',subject='service-kestrel',predicate='contact_channel',
                 old='queue-blue',new='queue-amber',domain='on_call_contacts'),
            dict(id='sensor_relocation',subject='sensor-aurora',predicate='located_in',
                 old='zone-north',new='zone-south',domain='facility_topology')]


def command_cases():
    return [dict(id='capacity',command='luma',observed=dict(free=3,cache=12,reserved=4),field='free',value=9,
                 specs=[dict(verb='shed',effect=dict(kind='transfer',**{'from':'cache','to':'free'})),
                        dict(verb='pledge',effect=dict(kind='transfer',**{'from':'free','to':'reserved'}))]),
            dict(id='monitor',command='davo',observed=dict(sample_age=90,alerts_enabled=True,uptime=321),field='sample_age',value=0,
                 specs=[dict(verb='sample',effect=dict(kind='set',field='sample_age',value=0)),
                        dict(verb='hush',effect=dict(kind='set',field='alerts_enabled',value=False))])]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bridge',type=Path,required=True)
    ap.add_argument('--image',required=True);ap.add_argument('--lock',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(exist_ok=False);source=args.out/'source';source.mkdir()
    for name in ['safe_pipeline.py','public_episode.py','public_reference_client.py','semantic_cli.py','run_public_breadth.py','environment.py','build_curriculum.py']:
        shutil.copy2(Path(__file__).with_name(name),source/name)
    hashes={str(p):digest(p) for p in source.iterdir()};hashes[str(args.bridge)]=digest(args.bridge)
    plan=dict(created_at=now(),schema='akh-public-breadth-plan-v1',model_used=False,cloud_compute_usd=0,
              image=args.image,source_hashes=hashes,deadline_seconds=600,phases=14,
              assistance='Authored container client; structured facts, typed goals and transfer/set effect primitives.',
              not_claimed=['natural language extraction','arbitrary unseen effect semantics','model capability','sealed evaluation'])
    write(args.out/'plan.json',plan);write(args.out/'examiner.json',dict(graph_cases=graph_cases(),command_cases=command_cases()))
    rows=[]
    def status(stage):write(args.out/'status.json',dict(stage=stage,updated_at=now(),progress=dict(phases=len(rows),total=14),pid=os.getpid()))
    def execute(case_id,phase,goal,documents,graph_dir,terminal=None):
        dest=args.out/(case_id+'-'+phase);dest.mkdir()
        graph=Graph(args.bridge,graph_dir,dest/'engine.log')
        try:
            port=PublicEpisode(graph,goal,documents,terminal)
            result=candidate(args.image,source/'public_reference_client.py',port,f'akh-r0-client-{os.getpid()}-{case_id}-{phase}')
            exported=graph.call('exam_export')
            write(dest/'trace.json',port.events);write(dest/'result.json',dict(**result,graph_after=exported,engine_pid=graph.process.pid))
            return result,port.events,exported,graph.process.pid
        finally:graph.close()
    with args.lock.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if shutil.disk_usage(args.out).free<5*2**30:raise RuntimeError('Disk floor')
        def stop(*_):raise TimeoutError('Overall deadline')
        signal.signal(signal.SIGALRM,stop);signal.alarm(600)
        try:
            status('running')
            for c in graph_cases():
                previous_pid=None
                for phase in ['acquire','cold','empty','revision']:
                    expected=c['new'] if phase=='revision' else c['old']
                    fact=dict(subject=c['subject'],predicate=c['predicate'],object=expected)
                    docs={'manual':dict(id='manual',revision=2 if phase=='revision' else 1,facts=[fact])} if phase in ['acquire','revision'] else {}
                    goal=dict(kind='graph',subject=c['subject'],predicate=c['predicate'],read_documents=bool(docs),revision=phase=='revision')
                    graph_dir=args.out/(c['id']+'-empty-graph' if phase=='empty' else c['id']+'-graph')
                    result,events,exported,pid=execute(c['id'],phase,goal,docs,graph_dir)
                    final=result['finished'];observed=[dict(subject=t['subject_label'],predicate=t['predicate_label'],object=t['object_label']) for t in exported['triples'] if t['subject_label']==c['subject'] and t['predicate_label']==c['predicate']]
                    checks=dict(answer=final['status']=='blocked' if phase=='empty' else final==dict(status='complete',answer=expected),
                                graph=[]==observed if phase=='empty' else observed==[fact],
                                no_cold_documents=phase not in ['cold','empty'] or not any(e['request']['tool']=='read_document' for e in events),
                                process_restarted=previous_pid is None or previous_pid!=pid,
                                revision_history=phase!='revision' or any(t['predicate_label']=='r0_revision_history' for t in exported['triples']))
                    previous_pid=pid;rows.append(dict(case=c['id'],phase=phase,checks=checks,passed=all(checks.values())));status('running')
            for c in command_cases():
                fixture=Fixture(args.image,f'akh-r0-world-{os.getpid()}-{c["id"]}',commands=[c['command']],cli=source/'semantic_cli.py')
                try:
                    for phase in ['acquire','cold','empty']:
                        before=dict(docs_available=phase=='acquire',observed=c['observed'],specs=c['specs'],audit=[]);fixture.reset(before)
                        goal=dict(kind='command',command=c['command'],field=c['field'],value=c['value'])
                        graph_dir=args.out/(c['id']+'-empty-graph' if phase=='empty' else c['id']+'-graph')
                        result,events,exported,pid=execute(c['id'],phase,goal,{},graph_dir,fixture.pipeline);after=fixture.state()
                        if phase=='empty':checks=dict(blocked=result['finished']['status']=='blocked',no_effect=after==before)
                        else:
                            terminal=[e for e in events if e['request']['tool']=='terminal'];last=terminal[-1]
                            expected=copy.deepcopy(c['observed'])
                            if c['id']=='capacity':expected.update(free=9,cache=6)
                            else:expected['sample_age']=0
                            checks=dict(complete=result['finished']['status']=='complete',actual_state=after['observed']==expected,
                                        one_mutation=len(after['audit'])==1,
                                        fresh_verification=last['request']['arguments']['stages']==[[c['command'],'inspect']] and json.loads(last['response']['stdout'])==after['observed'],
                                        no_cold_help=phase!='cold' or not any(e['request']['arguments']['stages']==[[c['command'],'--help']] for e in terminal))
                        write(args.out/(c['id']+'-'+phase)/'effects.json',dict(before=before,after=after))
                        rows.append(dict(case=c['id'],phase=phase,checks=checks,passed=all(checks.values())));status('running')
                finally:fixture.close()
            unchanged=all(digest(Path(p))==v for p,v in hashes.items())
            write(args.out/'summary.json',dict(completed_at=now(),rows=rows,all_passed=all(r['passed'] for r in rows) and unchanged,
                                              source_hashes_unchanged=unchanged,model_used=False,certification=False))
            status('completed');print(json.dumps(rows))
        except BaseException:
            status('failed');raise
        finally:signal.alarm(0)


if __name__=='__main__':main()
