"""Focused cancellation integration with real local services and a host-seeded artifact."""
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
from lilith_coordination import Coordinator, Ledger, request_cancellation, DONE
from lilith_verification import verify_clamp

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
result={'status':'running','kind':'durable_cancellation_api_probe'}
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
    previous=json.loads(Path('training/lilith/coordination-20260921/run1/assignments.json').read_text())[0]
    identity=uuid.uuid4().hex
    parent=api('akhsessd','/v1/sessions',{'request_id':identity+'-parent','akh':'lilith','workspace_profile':'local-dev'})
    parent_path='/v1/sessions/'+parent['id']
    author=api('akhsessd',parent_path+'/inputs',{'request_id':identity+'-author','text':'Reply with exactly Ready. Do not call tools.'})
    author_started=time.monotonic()
    while time.monotonic()-author_started<240:
        parent=api('akhsessd',parent_path)
        turn=next(t for t in parent['turns'] if t['id']==author['id'])
        if turn['status']=='completed':break
        if turn['status'] in DONE:raise RuntimeError(turn)
        time.sleep(.5)
    else:raise TimeoutError('fresh author readiness')
    (out/'author.json').write_text(json.dumps(parent,indent=2)+'\n')
    initial_turns=[t['id'] for t in parent['turns']]
    source=previous['source']
    (Path(parent['workspace']['root'])/'clamp_u8.py').write_text(source)
    evidence=verify_clamp(source)
    row=dict(schema_version=1,id=identity,parent=parent['id'],parent_turn=author['id'],
             intent=previous['intent'],source=source,source_sha256=evidence['source_sha256'],evidence=evidence,
             status='queued',acknowledged=True,reviewer='ptah-inline-reviewer')
    ledger=Ledger(out/'ledger.sqlite')
    try:
        ledger.save(row)
        controller=Coordinator(lambda path,body=None:api('akhsessd',path,body), ledger,
            row['parent'],row['parent_turn'],root/'workspaces',verify_clamp)
        row=controller.tick()[0]
        assert row['status']=='reviewing',row
        path='/v1/sessions/'+row['child']
        for _ in range(100):
            state=api('akhsessd',path)
            review=next(t for t in state['turns'] if t['id']==row['review_turn'])
            if review['status']=='running':break
            if review['status'] in DONE:raise RuntimeError('review ended before cancellation')
            time.sleep(.1)
        else:raise TimeoutError('review did not start')
        (out/'before-cancel.json').write_text(json.dumps(state,indent=2)+'\n')
        request_cancellation(out/'ledger.sqlite',row['parent'],row['parent_turn'],identity,'Stop focused integration review')
        started=time.monotonic()
        # Simulate controller restart after durable operator intent.
        ledger.close();ledger=Ledger(out/'ledger.sqlite')
        controller=Coordinator(lambda path,body=None:api('akhsessd',path,body), ledger,
            row['parent'],row['parent_turn'],root/'workspaces',verify_clamp)
        while time.monotonic()-started<180:
            row=controller.tick()[0]
            if row['status'] in DONE:break
            time.sleep(.2)
        assert row['status']=='cancelled',row
        assert 'return_turn' not in row
        assert [t['id'] for t in api('akhsessd','/v1/sessions/'+row['parent'])['turns']]==initial_turns
        state=api('akhsessd',path)
        review=next(t for t in state['turns'] if t['id']==row['review_turn'])
        assert review['status']=='cancelled',review
        (out/'after-cancel.json').write_text(json.dumps(state,indent=2)+'\n')
        (out/'assignment.json').write_text(json.dumps(row,indent=2)+'\n')
        result.update(status='passed',seconds=time.monotonic()-started,assignment_id=identity,
                      child_turn_status=review['status'],parent_turns_unchanged=True,controller_reopened=True)
        print(json.dumps(result),flush=True)
    finally:
        ledger.close()
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
