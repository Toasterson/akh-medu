"""Dependency-injected graph dispatch. Private phase data stays with the host.

Factories create native graph sessions and public candidate transports; tests
supply recording implementations. No implicit process, model or cloud launch.
"""
import copy
from public_episode import PublicEpisode

def dispatch(prepared,open_graph,run_candidate):
    if prepared.get('schema')!='akh-pack-preparation-v1' or prepared.get('adapter')!='graph-query-v1':raise ValueError('Graph preparation required')
    start=prepared['public_start'];host=prepared['host_config'];events=host['events']
    if len(events)!=1 or events[0]['after_action']!='cold_completed' or set(events[0]['payload'])!={'replacement'}:raise ValueError('Unsupported event schedule')
    replacement=events[0]['payload']['replacement']
    triples=host['world']['facts']
    for triple in [*triples,replacement]:
        if not isinstance(triple,list) or len(triple)!=3 or not all(isinstance(x,str) for x in triple):raise ValueError('Invalid fact triple')
    rows=[];pids=set()
    for phase in ['acquire','cold','correction']:
        # Correction is materialized only after the preceding cold completion.
        observed=triples if phase=='acquire' else [] if phase=='cold' else [replacement]
        docs={f'source-{i}':dict(facts=[dict(zip(['subject','predicate','object'],t))],supersedes=phase=='correction') for i,t in enumerate(observed)}
        graph=open_graph(phase)
        try:
            pid=graph.process.pid
            if pid in pids:raise ValueError('Cold phase requires a fresh native session')
            pids.add(pid)
            port=PublicEpisode(graph,copy.deepcopy(start['goal']),docs,max_calls=start['max_calls'])
            result=run_candidate(phase,port)
            final=result.get('finished')
            if not isinstance(final,dict) or final!=port.finished:raise ValueError('Completion lacks public finish receipt')
            export=graph.call('exam_export')
            if not export.get('success'):raise RuntimeError('Graph persistence failed')
            rows.append(dict(phase=phase,result=result,trace=copy.deepcopy(port.events),graph_after=export,engine_pid=pid))
            if final['status']!='complete':break
        finally:graph.close()
    return dict(schema='akh-prepared-graph-dispatch-v1',rows=rows,all_phases_completed=len(rows)==3 and rows[-1]['result']['finished']['status']=='complete',
                assistance=copy.deepcopy(prepared['assistance']),correctness_scored=False,R1_ready=False)
