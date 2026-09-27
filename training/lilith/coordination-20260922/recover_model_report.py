"""Recover only the persisted Lilith report after a mixed-backend runtime hang."""
import argparse, copy, hashlib, json, os, subprocess, sys, time, urllib.request
from pathlib import Path

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts/session-platform'))
from lilith_coordination import Ledger, canonical, digest

def write(path,value): Path(path).write_text(json.dumps(value,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--private-root',required=True,type=Path)
    p.add_argument('--recovery-name',default='report-recovery')
    p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    recovery=a.run/a.recovery_name;recovery.mkdir(exist_ok=False)
    configs={n:json.loads((a.private_root/(n+'.json')).read_text())
             for n in ('akhexecd','akhsbxd','akhrtd','akhsessd')}
    # The recovery process cannot load the HIP reviewer; its completed session is
    # immutable evidence in akhsessd. Only Vulkan profiles remain dispatchable.
    configs['akhrtd']['runtimes']={k:v for k,v in configs['akhrtd']['runtimes'].items()
                                  if k in ('lilith','khnum')}
    report_identity='lilith-report-recovery'
    for config in configs.values():
        for principal in config['principals']:
            if report_identity not in principal['akhs']:principal['akhs'].append(report_identity)
    report_profile=copy.deepcopy(configs['akhsessd']['akhs']['lilith-coordinator'])
    report_profile['tools']=[]
    report_profile['instructions']=('Summarize only the supplied immutable patch, verifier receipt, and reviewer opinion. '
        'Distinguish observed evidence from opinion. Do not claim the original repository changed or infer broader '
        'native, runtime, deployment, or model-quality validation. No tools are available.')
    configs['akhsessd']['akhs'][report_identity]=report_profile
    for n,c in configs.items():write(recovery/(n+'.json'),c)
    token=(a.private_root/'credentials/operator.token').read_text().strip()
    def api(path,body=None):
        r=urllib.request.Request('http://'+configs['akhsessd']['listen']+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with urllib.request.urlopen(r,timeout=30) as f:return json.load(f)
    def health(name):
        r=urllib.request.Request('http://'+configs[name]['listen']+'/healthz',headers={'Authorization':'Bearer '+token})
        with urllib.request.urlopen(r,timeout=2) as f:return json.load(f)
    ledger=Ledger(a.run/'coordination.sqlite')
    # Ledger.rows filters by parent, so read the single assignment directly.
    body=ledger.db.execute('select body from assignments').fetchone()
    if body is None:raise ValueError('missing assignment')
    row=json.loads(body[0]); parent=row['parent']
    if row['status']!='reporting' or not row.get('return_request') or not row.get('review'):
        raise ValueError('assignment is not at the guarded report boundary')
    guards={'source_sha256':digest(row['source']),'proposal_sha256':digest(canonical(row['proposal'])),
            'verification_sha256':digest(canonical(row['verification'])),'review_sha256':digest(row['review']),
            'return_text_sha256':digest(row['return_request']['text'])}
    if (guards['source_sha256']!=row['source_sha256'] or not row['verification'].get('passed')):
        raise ValueError('persisted evidence guard failed')
    write(recovery/'guards.json',guards)
    processes=[];logs=[];result={'status':'running','services_stopped':False,'ptah_runtime_available':False}
    try:
        for name in configs:
            log=(recovery/(name+'.log')).open('w');logs.append(log)
            proc=subprocess.Popen([str(REPO/'target/release'/name),'--config',str(recovery/(name+'.json'))],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,
                    AKH_GPU_MATMUL='0',AKH_GPU_QUANT='0',AKH_VULKAN_PIPELINE_CACHE_ENTRIES='16384',
                    VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json'))
            processes.append(proc)
            for _ in range(150):
                if proc.poll() is not None:raise RuntimeError(name+' exited')
                try:
                    if health(name)['service_id']==name:break
                except Exception:time.sleep(.2)
            else:raise TimeoutError(name+' readiness')
        parent_state=api('/v1/sessions/'+parent);write(recovery/'initial-parent-session.json',parent_state)
        original=next((t for t in parent_state['turns'] if t['request_id']==row['id']+'-return'),None)
        if original and original['status']=='unknown' and not original.get('cancel_requested'):
            api('/v1/sessions/'+parent+'/turns/'+original['id']+'/cancel',{})
            result['original_unknown_cancel_requested']=True
        active_session=row.get('report_recovery',{}).get('recovery_session',parent)
        initial=parent_state if active_session==parent else api('/v1/sessions/'+active_session)
        old=next(t for t in initial['turns'] if t['id']==row['return_turn'])
        report_session=active_session
        result['original_return_status']=old['status']
        if old['status'] not in ('completed','failed','unknown','cancelled'):
            observe_deadline=time.monotonic()+15
            while time.monotonic()<observe_deadline:
                initial=api('/v1/sessions/'+active_session)
                old=next(t for t in initial['turns'] if t['id']==row['return_turn'])
                if old['status'] in ('completed','failed','unknown','cancelled'):break
                time.sleep(.5)
            result['original_return_reconciled_status']=old['status']
            if old['status'] not in ('completed','failed','unknown','cancelled'):
                cancelled=api('/v1/sessions/'+active_session+'/turns/'+old['id']+'/cancel',{})
                result['original_return_cancel_requested']=True
                cancel_deadline=time.monotonic()+60
                while time.monotonic()<cancel_deadline:
                    initial=api('/v1/sessions/'+active_session)
                    old=next(t for t in initial['turns'] if t['id']==row['return_turn'])
                    if old['status'] in ('completed','failed','unknown','cancelled'):break
                    time.sleep(.5)
                result['original_return_after_cancel']=old['status']
        if old['status']!='completed':
            if old['status'] not in ('failed','unknown','cancelled'):
                raise RuntimeError('original return is neither completed nor safely retryable: '+old['status'])
            prior_attempt=row.get('report_recovery',{}).get('attempt',0)
            marker='-return-process-isolation-recovery-'
            prior_id=row.get('return_request',{}).get('request_id','')
            if marker in prior_id:
                prior_attempt=max(prior_attempt,int(prior_id.rsplit(marker,1)[1]))
            attempt=prior_attempt+1
            request=copy.deepcopy(row['return_request'])
            request['request_id']=row['id']+f'-return-process-isolation-recovery-{attempt}'
            if digest(request['text'])!=guards['return_text_sha256']:
                raise ValueError('recovery changed immutable report evidence')
            turn=api('/v1/sessions/'+active_session+'/inputs',request)
            report_session=active_session
            row.update(return_request=request,return_turn=turn['id'],status='reporting',
                report_recovery={'kind':'fresh_vulkan_process','attempt':attempt,
                                 'prior_turn':old['id'],'guards':guards})
            ledger.save(row)
            queued_deadline=time.monotonic()+15
            while time.monotonic()<queued_deadline:
                queued_state=api('/v1/sessions/'+parent)
                turn=next(t for t in queued_state['turns'] if t['id']==row['return_turn'])
                if turn['status']!='queued':break
                time.sleep(.5)
            if turn['status']=='queued':
                api('/v1/sessions/'+parent+'/turns/'+turn['id']+'/cancel',{})
                recovery_session=api('/v1/sessions',{'request_id':row['id']+f'-report-recovery-session-{attempt}',
                    'akh':report_identity,'workspace_profile':'local-dev'})
                separate={'request_id':row['id']+f'-report-separate-session-{attempt}','text':request['text']}
                turn=api('/v1/sessions/'+recovery_session['id']+'/inputs',separate)
                report_session=recovery_session['id']
                row.update(return_request=separate,return_turn=turn['id'],status='reporting',
                    report_recovery={'kind':'fresh_vulkan_process_separate_session','attempt':attempt,
                        'prior_turn':old['id'],'stuck_same_session_turn':request['request_id'],
                        'recovery_session':recovery_session['id'],'guards':guards})
                ledger.save(row)
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            state=api('/v1/sessions/'+report_session);turn=next(t for t in state['turns'] if t['id']==row['return_turn'])
            if turn['status'] in ('completed','failed','unknown','cancelled'):break
            time.sleep(.5)
        else:raise TimeoutError('recovery report deadline')
        if turn['status']!='completed' or not (turn.get('output') or '').strip():
            raise RuntimeError('recovery report did not complete: '+turn['status'])
        row.update(status='completed',report=turn['output']);ledger.save(row)
        write(recovery/'final-parent-session.json',state);write(recovery/'assignment.json',row)
        result.update(status='completed',return_turn=row['return_turn'],report=row['report'],truth_review='pending_manual')
    except Exception as error:
        result.update(status='interrupted',error=str(error))
    finally:
        ledger.close()
        for proc in reversed(processes):
            if proc.poll() is None:proc.terminate()
        for proc in processes:
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for log in logs:log.close()
        result['services_stopped']=all(proc.poll() is not None for proc in processes)
        write(recovery/'result.json',result)
    return 0 if result['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
