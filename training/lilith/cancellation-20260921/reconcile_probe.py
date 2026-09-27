"""Audited recovery of the single tool-free preparation job from failed run2."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

root=Path.home()/'.local/share/akh-medu/lilith-dev-vulkan-20260921'
out=Path(__file__).resolve().parent/'run2'
job_id='01a0c4de-90d3-7101-a84a-4d3e341e9d22'
with socket.socket() as probe:probe.bind(('127.0.0.1',18421))
assert json.loads((out/'result.json').read_text())['services_stopped']
token=(root/'credentials/operator.token').read_text().strip()
def api(path,body=None):
    req=urllib.request.Request('http://127.0.0.1:18421'+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=15) as stream:return json.load(stream)
with (out/'audited-recovery.log').open('w') as log:
    process=subprocess.Popen([str(Path('target/release/akhrtd').resolve()),'--config',str(root/'akhrtd.json')],stdout=log,stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:api('/healthz');break
            except OSError:time.sleep(.1)
        path='/v1/jobs/'+job_id
        job=api(path)
        assert job['status']=='unknown'
        assert job['error']['message']=='cancelled: internal runtime cancelled'
        assert job['request']['tools']==[]
        encoded=json.dumps(job['request'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
        resolution=dict(request_id='lilith-cancel-run2-audited-recovery',
            expected_request_sha256=hashlib.sha256(encoded).hexdigest(),status='cancelled',quiescent=True,
            evidence=['Owned run2 runtime process has exited; result.json records service cleanup.',
                      'Runtime log records cancellation after 0.997s, before Ptah model load or inference.',
                      'Exact job has no tool grants and reports cancelled: internal runtime cancelled.',
                      'This is one audited recovery, not automatic reconciliation of unknown jobs.'],
            output=None,result=None)
        result=api(path+'/reconcile',resolution)
        assert result['status']=='cancelled'
        (out/'audited-recovery.json').write_text(json.dumps({'resolution':resolution,'status':result['status']},indent=2)+'\n')
        print('Audited tool-free run2 job reconciled to cancelled.')
    finally:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
