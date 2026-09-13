"""Host-owned effect lifecycle with injected, explicitly registered adapters.

No implicit process creation. Adapters own fixture creation, correction semantics
and independent verification. The driver owns closure once a resource is returned.
"""
import copy
from public_episode import PublicEpisode
PHASES=('acquire','cold','correction')

def validate_effect(prepared,adapters):
    p=copy.deepcopy(prepared)
    if p.get('schema')!='akh-pack-preparation-v1' or p.get('adapter') not in adapters:raise ValueError('Unsupported effect adapter')
    adapter=adapters[p['adapter']];start=p['public_start'];host=p['host_config']
    if not isinstance(start.get('max_calls'),int) or isinstance(start['max_calls'],bool) or not 1<=start['max_calls']<=128:raise ValueError('Invalid call budget')
    neutral=p.get('effect_contract')=='neutral-v2'
    if p.get('effect_contract') not in (None,'neutral-v2'):raise ValueError('Unknown effect contract')
    required={'initial_state','documents','correction','expected'} | ({'cold_task'} if neutral else set())
    if set(host)!=required:raise ValueError('Incomplete effect episode')
    if neutral:
        if getattr(adapter,'neutral_effect_version',None)!=2 or not callable(getattr(adapter,'prepare_cold',None)):raise ValueError('Adapter lacks neutral cold-task capability')
        from neutral_lifecycle import reject_private
        reject_private(start['goal'])
        if start['max_calls']>48:raise ValueError('Neutral call budget exceeded')
        if set(host['expected'])!=set(PHASES) or any(not isinstance(v,dict) or v.get('status') not in ('complete','blocked') for v in host['expected'].values()):raise ValueError('Neutral phases require explicit expected status')
    correction=host['correction']
    if set(correction)!={'after_action','documents','payload'} or correction['after_action']!='cold_verified':raise ValueError('Unsupported correction schedule')
    if set(host['expected'])!=set(PHASES):raise ValueError('Every phase requires independent expectations')
    if not isinstance(host['documents'],dict) or not isinstance(correction['documents'],dict):raise ValueError('Invalid documents')
    adapter.validate(p)  # All adapter/schema checks precede resource creation.
    return p,adapter

def dispatch_effect(prepared,adapters,open_graph,run_candidate):
    p,adapter=validate_effect(prepared,adapters);start=p['public_start'];host=p['host_config'];correction=host['correction']
    neutral=p.get('effect_contract')=='neutral-v2'
    rows=[];pids=set();fixture=adapter.create_fixture(copy.deepcopy(host['initial_state']))
    try:
        for phase in PHASES:
            if neutral and phase=='cold':adapter.prepare_cold(fixture,copy.deepcopy(host['cold_task']))
            if phase=='correction':adapter.apply_correction(fixture,copy.deepcopy(correction['payload']))
            docs=host['documents'] if phase=='acquire' else {} if phase=='cold' else correction['documents']
            before=copy.deepcopy(fixture.state());graph=open_graph(phase)
            try:
                pid=graph.process.pid
                if pid in pids:raise ValueError('Fresh native session required per phase')
                pids.add(pid)
                goal=copy.deepcopy(start['goal'])
                if not neutral:goal['phase']=phase
                port=PublicEpisode(graph,goal,copy.deepcopy(docs),fixture.pipeline,max_calls=start['max_calls'])
                result=run_candidate(port) if neutral else run_candidate(phase,port)
                final=result.get('finished')
                if not isinstance(final,dict) or port.finished is None or final!=port.finished:raise ValueError('Completion lacks public receipt')
                exported=graph.call('exam_export')
                if not exported.get('success'):raise RuntimeError('Graph export failed')
                after=copy.deepcopy(fixture.state());trace=copy.deepcopy(port.events)
                verification=adapter.verify(phase,copy.deepcopy(host['expected'][phase]),copy.deepcopy(before),copy.deepcopy(after),copy.deepcopy(trace),copy.deepcopy(exported))
                if not isinstance(verification,dict) or type(verification.get('passed')) is not bool:raise ValueError('Verifier must return explicit boolean passed')
                expected_status=host['expected'][phase]['status'] if neutral else 'complete'
                outcome_passed=final['status']==expected_status and verification['passed']
                rows.append(dict(expected_status=expected_status,outcome_passed=outcome_passed,phase=phase,result=copy.deepcopy(result),trace=trace,graph_after=copy.deepcopy(exported),engine_pid=pid,before=before,after=after,effect_verification=copy.deepcopy(verification)))
                if not outcome_passed:break
            finally:graph.close()
    finally:fixture.close()
    complete=len(rows)==len(PHASES) and all(r['outcome_passed'] for r in rows)
    return dict(schema='akh-prepared-effect-dispatch-v1',rows=rows,all_phases_completed=complete,assistance=copy.deepcopy(p['assistance']),verification_scope='Registered host adapter verification; not independently certified by driver',R1_ready=False)
