#!/usr/bin/env python3
import argparse,fcntl,json,os,shutil,signal
from pathlib import Path
from build_curriculum import now,digest
from environment import Graph,Fixture
from temporal_adapter import TemporalEpisode,TemporalGuard,PROFILE_ID
from run_public_breadth import candidate,write
PHASES=['acquire','revoked','stale-attempt','corrected']
def document(id,time,granted):
    return dict(facts=[dict(subject='operator',predicate='permission-event',object=json.dumps(dict(id=id,timestamp=time,granted=granted),sort_keys=True))])
def main():
    ap=argparse.ArgumentParser()
    for key in ['out','bridge','lock']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--image',required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);src=a.out/'source';src.mkdir()
    for name in ['run_temporal_incident.py','temporal_adapter.py','temporal_incident_client.py','graph_effect_cli.py','public_episode.py','safe_pipeline.py','environment.py','run_public_breadth.py','build_curriculum.py']:shutil.copy2(Path(__file__).with_name(name),src/name)
    repo=Path(__file__).resolve().parents[2]
    for name in ['examples/akh_temporal_incident_bridge.rs','src/agent/event_calculus.rs']:shutil.copy2(repo/name,src/Path(name).name)
    shutil.copy2(a.bridge,a.out/'bridge');binary=a.out/'bridge'
    hashes={str(p):digest(p) for p in [*src.iterdir(),binary]}
    write(a.out/'plan.json',dict(phases=PHASES,events=[['grant',10,True],['revoke',20,False],['correct',30,True],['late-old-revoke',15,False]],budget_seconds=600,assistance_profile=PROFILE_ID,image=a.image,hashes=hashes,model_used=False,new_world_credit=0,assistance='Authored typed source events; native temporal execution guard; fixed client; latest recorded state not wall clock'))
    rows=[]
    def status(stage):write(a.out/'status.json',dict(stage=stage,updated_at=now(),pid=os.getpid(),phases_completed=len(rows)))
    with a.lock.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if shutil.disk_usage(a.out).free<5*2**30:raise RuntimeError('Disk floor')
        def stop(*_):raise TimeoutError('Budget exhausted')
        signal.signal(signal.SIGALRM,stop);signal.alarm(600);status('running')
        fixture=Fixture(a.image,f'akh-temporal-{os.getpid()}',commands=['sana'],cli=src/'graph_effect_cli.py')
        try:
            fixture.reset(dict(authorized_target='nodealpha',nodes={n:dict(healthy=False,repairs=0) for n in ['nodealpha','nodebeta']},audit=[]))
            for phase in PHASES:
                dest=a.out/phase;dest.mkdir();graph=Graph(binary,a.out/'graph',dest/'engine.log')
                try:
                    before=fixture.state();initial=graph.call('exam_temporal_state');guard=[]
                    docs={'grant':document('grant',10,True)} if phase=='acquire' else {'revoke':document('revoke',20,False)} if phase=='revoked' else {'0-correction':document('correct',30,True),'1-late':document('late-old-revoke',15,False)} if phase=='corrected' else {}
                    terminal=TemporalGuard(graph,[['sana','apply','nodealpha']],fixture.pipeline)
                    guard=terminal.receipts
                    def interact(action,value):
                        if (action,value)!=('permission','repair'):raise ValueError('Unknown interaction')
                        return graph.call('exam_temporal_state')
                    port=TemporalEpisode(graph,dict(command='sana',phase=phase),docs,terminal,interaction=interact)
                    result=candidate(a.image,src/'temporal_incident_client.py',port,f'akh-temporal-client-{os.getpid()}-{phase}')
                    after=fixture.state();state=graph.call('exam_temporal_state');expected=phase in ['acquire','corrected']
                    checks=dict(finished=result['finished']['status']==('complete' if expected else 'blocked'),effect=after['nodes']['nodealpha']['repairs']==int(phase=='corrected'),other_untouched=after['nodes']['nodebeta']['repairs']==0,holding=state['holding']==expected,cache_agrees=state['cached']['current_value']==state['holding'],history=len(state['history'])=={'acquire':1,'revoked':2,'stale-attempt':2,'corrected':4}[phase])
                    write(dest/'trace.json',port.events);write(dest/'evidence.json',dict(before=before,after=after,initial=initial,state=state,guard=guard,result=result,graph_pid=graph.process.pid,checks=checks));graph.call('exam_export');rows.append(dict(phase=phase,checks=checks));status('running')
                finally:graph.close()
            checks=dict(source_hashes=all(digest(Path(p))==h for p,h in hashes.items()),phases=all(all(r['checks'].values()) for r in rows))
            write(a.out/'summary.json',dict(rows=rows,checks=checks,all_passed=all(checks.values()),model_used=False,new_world_credit=0));status('completed' if all(checks.values()) else 'failed')
            if not all(checks.values()):raise RuntimeError('Temporal incident failed')
        except BaseException:status('failed');raise
        finally:fixture.close();signal.alarm(0)
if __name__=='__main__':main()
