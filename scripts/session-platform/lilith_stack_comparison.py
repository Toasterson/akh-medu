#!/usr/bin/env python3
"""Paired deployment-backend Gemma/Ptah comparison on frozen v2 development tasks.

This is separate from the HF training-arm comparison: neither backend/precision
nor decoder budget is silently treated as identical to the training run.
"""
import argparse, copy, hashlib, json, os, secrets, socket, subprocess, time, urllib.request, uuid
from pathlib import Path
import lilith_rl_tasks_v2 as tasks

REPO = Path(__file__).resolve().parents[2]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(path, value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def run(output, manifest_path):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frozen=json.loads(Path(manifest_path).read_text())
    if frozen['verifier_sha256'] != digest(tasks.__file__) or frozen['interpreter_sha256'] != digest(Path(tasks.__file__).with_name('lilith_rl_tasks.py')):
        raise ValueError('Frozen task/interpreter source mismatch')
    pack=list(tasks.tasks()); pinned={x['id']:x for x in frozen['tasks']}
    for task in pack:
        if pinned[task.id]['prompt_sha256'] != tasks.digest(task.prompt):raise ValueError('Prompt drift')
    old=Path.home()/'.local/share/akh-medu/lilith-dev-vulkan-20260921'
    root=Path.home()/'.local/share/akh-medu'/('lilith-comparison-'+uuid.uuid4().hex)
    root.mkdir(mode=0o700)
    for directory in ('credentials','workspaces'):(root/directory).mkdir(mode=0o700)
    configs={name:json.loads((old/(name+'.json')).read_text().replace(str(old),str(root)))
             for name in ('akhexecd','akhsbxd','akhrtd','akhsessd')}
    identities=['lilith-compare','ptah-compare']
    for name in ['operator']+list(configs):
        value=secrets.token_urlsafe(32)
        for suffix,content in [('token',value),('sha256',hashlib.sha256(value.encode()).hexdigest())]:
            p=root/'credentials'/(name+'.'+suffix);p.write_text(content+'\n');p.chmod(0o600)
    for c in configs.values():
        for p in c['principals']:p['akhs']=identities
        host,port=c['listen'].rsplit(':',1)
        if host!='127.0.0.1':raise ValueError('Loopback required')
        with socket.socket() as sock:sock.bind((host,int(port)))
    profiles=configs['akhsessd']['akhs']
    configs['akhsessd']['akhs']={'lilith-compare':copy.deepcopy(profiles['lilith-learner']),
                               'ptah-compare':copy.deepcopy(profiles['ptah-inline-reviewer'])}
    for p in configs['akhsessd']['akhs'].values():
        p['tools']=[];p['instructions']='Solve the assigned coding exercise. No tools are available.'
    pins={}
    for name in ('lilith','ptah'):
        runtime=configs['akhrtd']['runtimes'][name];runtime['timeout_secs']=180
        Path(runtime['working_dir']).mkdir(parents=True,exist_ok=True)
        native=runtime['native_candle'];native.update(max_context_tokens=4096,max_output_tokens=512,max_iterations=1)
        for key in ('model','tokenizer'):
            actual=digest(native[key+'_path']);expected=native[key+'_sha256']
            if actual!=expected:raise ValueError(name+' '+key+' artifact mismatch')
        pins[name]=native
    plan=dict(schema='lilith/deployment-comparison@1',tasks_sha256=digest(manifest_path),
        source_sha256=digest(__file__),models=pins,order=['lilith','ptah'],
        reference_identity='JonathanColetti/Qwen3.8-27B-Uncensored',
        limitations=['Public development tasks, one observation each; not sealed certification.',
                     'Ptah is a derivative; no unmodified upstream Qwen parity claim.',
                     'Different inference backends/precision from the HF training arms.',
                     'Runtime does not expose per-request random seed; no deterministic paired-seed claim.'],
        max_new_tokens=512,context_tokens=4096,wall_seconds=180,
        binary_sha256={name:digest(REPO/'target/release'/name) for name in configs})
    write(output/'protocol.json',plan)
    for name,c in configs.items():write(root/(name+'.json'),c)
    token=(root/'credentials/operator.token').read_text().strip()
    def api(name,path,body=None):
        r=urllib.request.Request('http://'+configs[name]['listen']+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with urllib.request.urlopen(r,timeout=20) as f:return json.load(f)
    processes=[];logs=[];records=[]
    result=dict(status='running',private_root=str(root),quality_claim=False,native_illumos=False)
    try:
        for name,c in configs.items():
            log=(output/(name+'.log')).open('w');logs.append(log)
            proc=subprocess.Popen([str(REPO/'target/release'/name),'--config',str(root/(name+'.json'))],
                cwd=REPO,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,
                AKH_GPU_MATMUL='0',AKH_GPU_QUANT='0',AKH_VULKAN_PIPELINE_CACHE_ENTRIES='16384',
                VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json',RUST_LOG='info'))
            processes.append(proc)
            for _ in range(100):
                if proc.poll() is not None:raise RuntimeError(name+' exited')
                try:
                    if api(name,'/healthz')['service_id']==c['service_id']:break
                except (OSError,ValueError):time.sleep(.2)
            else:raise TimeoutError(name+' readiness')
        for role in ('lilith','ptah'):
            for task in pack:
                key=uuid.uuid4().hex
                session=api('akhsessd','/v1/sessions',dict(request_id=key,akh=role+'-compare',workspace_profile='local-dev'))
                path='/v1/sessions/'+session['id'];start=time.monotonic()
                turn=api('akhsessd',path+'/inputs',dict(request_id=key+'-input',text=task.prompt))
                timed_out=False
                write(output/(role+'-'+task.id.replace('/','-')+'-request.json'),dict(session_id=session['id'],turn_id=turn['id'],prompt=task.prompt))
                while True:
                    state=api('akhsessd',path);current=next(x for x in state['turns'] if x['id']==turn['id'])
                    if current['status'] in ('completed','failed','cancelled','unknown'):break
                    if time.monotonic()-start>180:
                        api('akhsessd',path+'/turns/'+turn['id']+'/cancel',{})
                        timed_out=True
                        break
                    time.sleep(.3)
                if api('akhsessd',path+'/tools')['records']:raise RuntimeError('Tool-free comparison used a tool')
                source=current.get('output') or ''
                outcome=tasks.verify(source,task)
                if timed_out or current['status']!='completed':outcome.update(passed=False,completion_gate='timeout' if timed_out else current['status'])
                record=dict(role=role,task_id=task.id,family=task.family,split=task.split,
                    status='timeout' if timed_out else current['status'],source=source,evidence=outcome,
                    elapsed_seconds=time.monotonic()-start,session_id=session['id'],turn_id=turn['id'])
                records.append(record);write(output/'records.json',records)
                write(output/(role+'-'+task.id.replace('/','-')+'-session.json'),state)
                print(json.dumps(dict(role=role,task=task.id,status=current['status'],passed=outcome['passed'])),flush=True)
                if timed_out or current['status']=='unknown':raise TimeoutError('Unconfirmed completion; preserved record and stopping owned stack')
        result.update(status='completed',observations=len(records),scores={role:{split:{
            'passed':sum(r['evidence']['passed'] for r in records if r['role']==role and r['split']==split),
            'total':sum(r['role']==role and r['split']==split for r in records)}
            for split in ('train','development_holdout')} for role in ('lilith','ptah')})
    except Exception as error:
        result.update(status='interrupted',error=str(error),observations=len(records))
        raise
    finally:
        for p in reversed(processes):
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()
        result['services_stopped']=all(p.poll() is not None for p in processes)
        write(output/'result.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
    a=p.parse_args();print(json.dumps(run(a.output,a.manifest),indent=2))
