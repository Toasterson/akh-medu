"""Focused real Ptah check of the clarified inline-snapshot/receipt envelope."""
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
from lilith_coordination import review_input
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
result={'status':'running','kind':'clarified_handoff_review_probe'}
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
    source=(out.parent/'run1/clamp_u8.py').read_text()
    identity=uuid.uuid4().hex
    verification=verify_clamp(source)
    verification.update(receipt=identity+'-verification',kind='host_fixture_verification',verifier='integer-clamp/1')
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    row={'id':identity,'intent':{'path':'clamp_u8.py','objective':'Review the integer clamp contract.'},
         'source':source,'source_sha256':verification['source_sha256'],'evidence':verification}
    request={'request_id':identity+'-review','text':review_input(row)}
    (out/'request.json').write_text(json.dumps(request,indent=2)+'\n')
    session=api('akhsessd','/v1/sessions',{'request_id':identity,'akh':'ptah-inline-reviewer','workspace_profile':'local-dev'})
    path='/v1/sessions/'+session['id']
    turn=api('akhsessd',path+'/inputs',request)
    started=time.monotonic()
    while time.monotonic()-started<360:
        state=api('akhsessd',path)
        current=next(t for t in state['turns'] if t['id']==turn['id'])
        if current['status'] in ['failed','unknown','cancelled']:raise RuntimeError(current)
        if current['status']=='completed':break
        time.sleep(.5)
    else:raise TimeoutError('review')
    records=api('akhsessd',path+'/tools')['records']
    (out/'session.json').write_text(json.dumps(state,indent=2)+'\n')
    (out/'tools.json').write_text(json.dumps(records,indent=2)+'\n')
    assert current['output'].strip()
    assert not records, 'reviewer still tried to read an unmounted source instead of the inline snapshot'
    result.update(status='passed',seconds=time.monotonic()-started,review=current['output'])
    print(json.dumps(result),flush=True)
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
