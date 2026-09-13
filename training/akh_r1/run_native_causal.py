#!/usr/bin/env python3
import argparse,fcntl,json,os,shutil,signal
from pathlib import Path
from build_curriculum import now,digest
from environment import Graph,Fixture
from public_episode import PublicEpisode
from run_public_breadth import candidate,write
GOOD=dict(preconditions=[['nodealpha','permission','granted']],effects=[['nodealpha','healthy','true']])
BAD=dict(preconditions=[['nodealpha','permission','granted']],effects=[['nodebeta','healthy','true']])
CASES=['absent','correct','incorrect','corrected','unmet-prerequisite']
def main():
    ap=argparse.ArgumentParser()
    for n in ['out','bridge','lock']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--image',required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False);src=a.out/'source';src.mkdir()
    for n in ['run_native_causal.py','native_causal_client.py','graph_effect_cli.py','public_episode.py','safe_pipeline.py','environment.py','run_public_breadth.py','build_curriculum.py']:shutil.copy2(Path(__file__).with_name(n),src/n)
    for path in ['examples/akh_causal_episode_bridge.rs','src/agent/causal.rs']:shutil.copy2(Path(__file__).resolve().parents[2]/path,src/Path(path).name)
    hashes={str(p):digest(p) for p in src.iterdir()};hashes[str(a.bridge)]=digest(a.bridge)
    write(a.out/'plan.json',dict(created_at=now(),cases=CASES,correct_schema=GOOD,incorrect_schema=BAD,deadline_seconds=600,image=a.image,source_hashes=hashes,model_used=False,new_independent_worlds=0,assistance='Native CausalManager and concrete ActionSchema; authored schema sources, actual executor observations; no event calculus.'))
    rows=[]
    def status(stage):write(a.out/'status.json',dict(stage=stage,updated_at=now(),pid=os.getpid(),progress=dict(cases=len(rows),total=5)))
    with a.lock.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if shutil.disk_usage(a.out).free<5*2**30:raise RuntimeError('Disk floor')
        def stop(*_):raise TimeoutError('Deadline')
        signal.signal(signal.SIGALRM,stop);signal.alarm(600);status('running')
        fixture=Fixture(a.image,f'akh-native-causal-{os.getpid()}',commands=['sana'],cli=src/'graph_effect_cli.py')
        try:
            for case in CASES:
                dest=a.out/case;dest.mkdir();before=dict(authorized_target='nodealpha',nodes={x:dict(healthy=False,repairs=0) for x in ['nodealpha','nodebeta']},audit=[]);fixture.reset(before);graph=Graph(a.bridge,dest/'graph',dest/'engine.log')
                try:
                    docs={}
                    if case!='unmet-prerequisite':docs['0-observation']=dict(facts=[dict(subject='nodealpha',predicate='permission',object='granted')])
                    def schema_doc(schema):return dict(facts=[dict(subject='repair-schema',predicate='schema',object=json.dumps(schema,sort_keys=True))])
                    if case!='absent':docs['1-schema']=schema_doc(BAD if case in ['incorrect','corrected'] else GOOD)
                    if case=='corrected':docs['2-correction']=schema_doc(GOOD)
                    native=[];observations=[]
                    def terminal(stages):
                        result=fixture.pipeline(stages)
                        if result['success'] and stages==[['sana','inspect','nodealpha']]:
                            data=json.loads(result['stdout']);fact=dict(subject=data['target'],predicate='healthy',object=str(data['healthy']).lower());observations.append(dict(result=result,fact=fact))
                            if not graph.call('memory_write',dict(summary='Actual post-action observation',facts=[fact],source='executor-inspection'))['success']:raise RuntimeError('Observation write failed')
                        return result
                    def interact(action,value):
                        if action=='predict' and value=='repair':
                            facts=port._facts('repair-schema');schemas=[json.loads(f['object']) for f in facts if f['predicate']=='schema']
                            if len(schemas)!=1:return dict(success=True,applicable=False)
                            result=graph.call('exam_causal_predict',schemas[0]);native.append(dict(action=action,schema=schemas[0],result=result));return result
                        if action=='verify':
                            result=graph.call('exam_causal_verify',dict(prediction=value));native.append(dict(action=action,result=result));return result
                        raise ValueError('Unknown interaction')
                    port=PublicEpisode(graph,dict(command='sana',target='nodealpha'),docs,terminal,interaction=interact)
                    result=candidate(a.image,src/'native_causal_client.py',port,f'akh-native-causal-client-{os.getpid()}-{case}');after=fixture.state();executed=case in ['correct','incorrect','corrected'];complete=case in ['correct','corrected']
                    checks=dict(status=result['finished']['status']==('complete' if complete else 'blocked'),effect=after['nodes']['nodealpha']['repairs']==int(executed))
                    write(dest/'trace.json',port.events);write(dest/'effects.json',dict(before=before,after=after,native=native,observations=observations));write(dest/'result.json',dict(**result,graph_after=graph.call('exam_export'),checks=checks));rows.append(dict(case=case,checks=checks,passed=all(checks.values())));status('running')
                finally:graph.close()
            passed=all(r['passed'] for r in rows) and all(digest(Path(p))==h for p,h in hashes.items());write(a.out/'summary.json',dict(all_passed=passed,rows=rows,model_used=False,new_independent_worlds=0));status('completed' if passed else 'failed')
            if not passed:raise RuntimeError('Native causal controls failed')
        except BaseException:status('failed');raise
        finally:fixture.close();signal.alarm(0)
if __name__=='__main__':main()
