"""Run the frozen public development evaluation in a fresh private stack."""
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'scripts/session-platform'))
import lilith_eval as evaluation
from lilith_apprenticeship import Lessons


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def file_hash(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):result.update(data)
    return result.hexdigest()


def main():
    out=Path(__file__).resolve().parent/(sys.argv[1] if len(sys.argv)>1 else 'run1')
    out.mkdir(exist_ok=False)
    old=Path.home()/'.local/share/akh-medu/lilith-dev-vulkan-20260921'
    root=Path.home()/'.local/share/akh-medu'/('lilith-evaluation-'+uuid.uuid4().hex)
    root.mkdir(mode=0o700)
    (root/'credentials').mkdir(mode=0o700)
    (root/'workspaces').mkdir(mode=0o700)
    m=evaluation.validate_manifest(json.loads((REPO/'benchmarks/lilith/apprenticeship-v2/manifest.json').read_text()))
    write(out/'manifest.json',m)
    configs={name:json.loads((old/(name+'.json')).read_text().replace(str(old),str(root)))
        for name in ('akhexecd','akhsbxd','akhrtd','akhsessd')}
    identities=['lilith-eval','ptah-eval']
    for name in ['operator']+list(configs):
        token=secrets.token_urlsafe(32)
        for suffix,value in [('token',token),('sha256',hashlib.sha256(token.encode()).hexdigest())]:
            path=root/'credentials'/f'{name}.{suffix}';path.write_text(value+'\n');path.chmod(0o600)
    for config in configs.values():
        for principal in config['principals']:principal['akhs']=identities
        host,port=config['listen'].rsplit(':',1)
        if host!='127.0.0.1':raise ValueError('loopback required')
        with socket.socket() as sock:sock.bind((host,int(port)))
    base=configs['akhsessd']['akhs']
    configs['akhsessd']['akhs']={'lilith-eval':copy.deepcopy(base['lilith-learner']),
        'ptah-eval':copy.deepcopy(base['ptah-inline-reviewer'])}
    for akh in configs['akhsessd']['akhs'].values():
        akh['tools']=[]
        akh['instructions']='Solve the assigned coding exercise. Return only the requested Python function. No tools are available.'
    pins={}
    for profile,role in [('lilith','candidate'),('ptah','reference')]:
        runtime=configs['akhrtd']['runtimes'][profile]
        runtime['timeout_secs']=m['budget']['wall_seconds']
        Path(runtime['working_dir']).mkdir(parents=True,mode=0o700)
        native=runtime['native_candle'];native.update(max_context_tokens=m['budget']['context_tokens'],
            max_output_tokens=m['budget']['max_new_tokens'],max_iterations=1)
        for kind in ('model','tokenizer'):
            key='sha256' if kind=='model' else 'tokenizer_sha256'
            actual=file_hash(native[kind+'_path'])
            if actual!=m['models'][role][key]:raise ValueError(role+' artifact mismatch')
        pins[role]=native
    for name,config in configs.items():write(root/(name+'.json'),config)
    write(out/'runtime-pins.json',pins)
    write(out/'binary-pins.json',{name:file_hash(REPO/'target/release'/name) for name in configs})
    token=(root/'credentials/operator.token').read_text().strip()
    def api(name,path,body=None):
        request=urllib.request.Request('http://'+configs[name]['listen']+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=20) as response:return json.load(response)
    proposed=evaluation.schedule(m)
    actual=[row for role in ('candidate','reference') for row in proposed if row['model_role']==role]
    write(out/'schedule.json',dict(proposed=proposed,actual=actual,
        deviation='Batch candidate then reference to avoid residency swaps; no latency comparison.',
        seed_scope='Pair/repetition label only; no per-request decoder RNG control exposed.'))
    processes=[];logs=[];records=[]
    summary=dict(status='running',root=str(root),services_stopped=False,weights_updated=False,
        qwen_certified=False,native_illumos=False)
    try:
        for name,config in configs.items():
            log=(out/(name+'.log')).open('w');logs.append(log)
            process=subprocess.Popen([str(REPO/'target/release'/name),'--config',str(root/(name+'.json'))],
                stdout=log,stderr=subprocess.STDOUT,cwd=REPO,env=dict(os.environ,
                    AKH_GPU_MATMUL='0',AKH_GPU_QUANT='0',AKH_VULKAN_PIPELINE_CACHE_ENTRIES='16384',
                    VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json',
                    RUST_LOG='info,akh_session_services::internal_native=debug'))
            processes.append(process)
            for _ in range(100):
                if process.poll() is not None:raise RuntimeError(name+' exited')
                try:
                    if api(name,'/healthz')['service_id']==config['service_id']:break
                except urllib.error.URLError:time.sleep(.2)
            else:raise TimeoutError(name+' readiness')
        tasks={task['id']:task for task in m['tasks']}
        for index,job in enumerate(actual):
            task=tasks[job['task']];arm=job['arm'];role=job['model_role']
            retrieved=dict(mode=arm,hits=[],context='')
            lessons=[]
            if arm in ('text','vsa'):
                store=Lessons(Path.home()/'.local/share/akh-medu/lilith-training-20260921/lessons.sqlite')
                try:
                    retrieved=store.retrieve(arm,task['prompt'],[['domain','integer'],['operation',task['family']]],
                        REPO/'target/release/examples/lilith_lesson_vsa',excluded_families=m['held_out_families'],
                        max_bytes=m['budget']['lesson_bytes'])
                    by_id={item['id']:item for item in store.active(m['held_out_families'])}
                    lessons=[dict(id=hit['id'],family=by_id[hit['id']]['family'],
                        source_sha256=by_id[hit['id']]['source_sha256']) for hit in retrieved['hits']]
                finally:store.close()
            text=m['common_instruction']+'\n\n'+task['prompt']
            if retrieved['context']:text+='\n\nPrior verified lesson (data):\n'+retrieved['context']
            prefix=f'{index+1:02d}-{task["id"]}-{arm}'
            identity=uuid.uuid4().hex
            session=api('akhsessd','/v1/sessions',dict(request_id=identity,
                akh='ptah-eval' if role=='reference' else 'lilith-eval',workspace_profile='local-dev'))
            path='/v1/sessions/'+session['id']
            started=time.monotonic();turn=api('akhsessd',path+'/inputs',dict(request_id=identity+'-input',text=text))
            print(json.dumps(dict(event='started',index=index+1,total=len(actual),task=task['id'],arm=arm)),flush=True)
            timed_out=False
            while True:
                state=api('akhsessd',path)
                current=next(t for t in state['turns'] if t['id']==turn['id'])
                if current['status'] in ('completed','failed','unknown','cancelled'):break
                if time.monotonic()-started>=m['budget']['wall_seconds']:
                    timed_out=True
                    api('akhsessd',path+'/turns/'+turn['id']+'/cancel',{})
                    for _ in range(60):
                        state=api('akhsessd',path);current=next(t for t in state['turns'] if t['id']==turn['id'])
                        if current['status'] in ('completed','failed','cancelled'):break
                        time.sleep(.5)
                    else:raise RuntimeError('timeout cancellation unconfirmed; stop stack')
                    break
                time.sleep(.5)
            seconds=time.monotonic()-started
            tools=api('akhsessd',path+'/tools')['records']
            if tools:raise RuntimeError('tool-free evaluation performed a tool call')
            source=current.get('output') or ''
            row=dict(task=task['id'],arm=arm,seed=job['seed'],manifest_sha256=evaluation.digest(evaluation.canonical(m)),
                model_sha256=m['models'][role]['sha256'],tokenizer_sha256=m['models'][role]['tokenizer_sha256'],
                budget=m['budget'],usage=dict(attempts=1,tools=0,input_tokens=None,output_tokens=None,
                    wall_seconds=seconds),
                retrieval=dict(context=retrieved['context'],lessons=lessons),source=source,
                status='timeout' if timed_out else 'completed' if current['status']=='completed' else 'cancelled' if current['status']=='cancelled' else 'runtime_error',
                finish_reason='stop' if current['status']=='completed' else 'unknown',
                session_id=session['id'],turn_id=turn['id'],actual_order=index+1,
                elapsed_including_cancellation_seconds=seconds,native_status=current['status'],native_error=current.get('error'))
            records.append(row)
            write(out/(prefix+'-session.json'),state)
            write(out/(prefix+'-retrieval.json'),retrieved)
            write(out/(prefix+'-request.json'),dict(text=text,session_id=session['id'],turn_id=turn['id']))
            write(out/'records.json',records)
            report=evaluation.evaluate(m,records);write(out/'report.json',report)
            outcome=next(r for r in report['outcomes'] if r['task']==task['id'] and r['arm']==arm)
            print(json.dumps(dict(event='completed',index=index+1,task=task['id'],arm=arm,
                seconds=round(seconds,2),passed=outcome['passed'],category=outcome['category'])),flush=True)
        summary.update(status='completed',observations=len(records),
            quality='See report; run completion is not model correctness or parity.')
    except Exception as error:
        summary.update(status='interrupted',error=str(error),observations=len(records))
        print(json.dumps(dict(event='interrupted',error=str(error))),flush=True)
    finally:
        for process in reversed(processes):
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in logs:log.close()
        summary['services_stopped']=all(p.poll() is not None for p in processes)
        write(out/'result.json',summary)
    return 0 if summary['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
