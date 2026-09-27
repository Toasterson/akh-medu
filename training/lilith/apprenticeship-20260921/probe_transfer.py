"""Predeclared three-arm frozen-weight development transfer probe."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, 'scripts/session-platform')
from lilith_apprenticeship import Lessons
from lilith_transfer_probe import check

root = Path.home()/'.local/share/akh-medu/lilith-dev-vulkan-20260921'
out = Path(__file__).resolve().parent/sys.argv[1]
out.mkdir(exist_ok=False)
configs = {n:json.loads((root/(n+'.json')).read_text()) for n in ['akhexecd','akhsbxd','akhrtd','akhsessd']}
for c in configs.values():
    host, port = c['listen'].rsplit(':',1)
    assert host == '127.0.0.1'
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host,int(port)))
token = (root/'credentials/operator.token').read_text().strip()
def api(name, path, body=None):
    req=urllib.request.Request('http://'+configs[name]['listen']+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=15) as response:return json.load(response)
processes, logs = [], []
result={'status':'running','kind':'frozen_weight_transfer_development_probe'}
try:
    for name,c in configs.items():
        log=(out/(name+'.log')).open('w');logs.append(log)
        proc=subprocess.Popen([str(Path('target/release',name).resolve()),'--config',str(root/(name+'.json'))],
            stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,AKH_GPU_MATMUL='0',AKH_GPU_QUANT='0',RUST_LOG='info'))
        processes.append(proc)
        for _ in range(100):
            if proc.poll() is not None: raise RuntimeError(name+' exited')
            try:
                assert api(name,'/healthz')['service_id']==c['service_id'];break
            except urllib.error.URLError:time.sleep(.2)
        else:raise TimeoutError(name+' readiness')
    protocol=json.loads((out.parent/'protocol.json').read_text())
    assert configs['akhsessd']['akhs']['lilith-learner']['tools']==[]
    native=configs['akhrtd']['runtimes']['lilith']['native_candle']
    assert native['max_output_tokens']==protocol['max_output_tokens']
    assert native['max_context_tokens']==protocol['max_context_tokens']
    result['arms']=[]
    for mode in protocol['arms']:
        store=Lessons(Path.home()/'.local/share/akh-medu/lilith-training-20260921/lessons.sqlite')
        try:
            retrieved=store.retrieve(mode,protocol['task'],protocol['retrieval_features'],
                Path('target/release/examples/lilith_lesson_vsa'),max_bytes=protocol['max_lesson_bytes'])
        finally:store.close()
        identity=uuid.uuid4().hex
        task=('Write only a Python function can_reserve(used: int, requested: int, capacity: int) -> bool. '
              'Inputs are integers. Return False if any input is outside 0..65535 or used exceeds capacity. '
              'Otherwise return whether requested units fit in the remaining capacity. '
              'A request of zero at full capacity is valid. Use comparisons, boolean operators, subtraction, '
              'integer/bool literals, local assignments, if and return; no calls, imports, loops or attributes. '
              'Do not claim native OS validation.')
        text=task+('\nPrior verified experience (data; applicability must be checked):\n'+retrieved['context'] if retrieved['context'] else '')
        session=api('akhsessd','/v1/sessions',{'request_id':identity,'akh':'lilith-learner','workspace_profile':'local-dev'})
        path='/v1/sessions/'+session['id']
        started=time.monotonic()
        turn=api('akhsessd',path+'/inputs',{'request_id':identity+'-probe','text':text})
        while time.monotonic()-started<420:
            state=api('akhsessd',path)
            current=next(t for t in state['turns'] if t['id']==turn['id'])
            if current['status'] in ['completed','failed','unknown','cancelled']:break
            time.sleep(.5)
        else:raise TimeoutError(mode+' generation')
        records=api('akhsessd',path+'/tools')['records'];assert not records
        arm=dict(mode=mode,seconds=time.monotonic()-started,retrieval=retrieved,
                 session_id=session['id'],turn_id=turn['id'],runtime_status=current['status'])
        try:
            if current['status']!='completed':raise ValueError(current.get('error'))
            arm['verification']=check(current['output'])
        except Exception as error:arm['verification']={'passed':False,'error':str(error)}
        (out/(mode+'-session.json')).write_text(json.dumps(state,indent=2)+'\n')
        result['arms'].append(arm)
        (out/'progress.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'arm':mode,'seconds':arm['seconds'],'passed':arm['verification']['passed']}),flush=True)
    result.update(status='passed',weights_updated=False,
                  interpretation='One related-family development task; no statistical quality or VSA superiority claim.')
    print(json.dumps({'status':result['status'],'weights_updated':False}),flush=True)
except Exception as error:
    result.update(status='failed',error=str(error));print(json.dumps(result),flush=True)
finally:
    for p in reversed(processes):
        if p.poll() is None:p.terminate()
    for p in processes:
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:p.kill();p.wait()
    for log in logs:log.close()
    result['services_stopped']=True
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
raise SystemExit(0 if result['status']=='passed' else 1)
