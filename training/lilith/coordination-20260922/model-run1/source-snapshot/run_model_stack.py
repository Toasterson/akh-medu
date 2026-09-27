"""Run the two-specialist episode on the private model stack and VM executor.

The launcher is inert unless both an executor factory and an explicit GPU handoff
environment marker are supplied.  It never uses the bounded host fixture executor.
"""
import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO/'scripts/session-platform'))
from lilith_coordination import Ledger
from lilith_repository_coordination import RepositoryCoordinator, repository_task_tool


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def file_hash(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_executor(spec):
    module_name, separator, factory_name = spec.partition(':')
    if not separator:
        raise ValueError('--executor must be module:factory')
    factory = getattr(importlib.import_module(module_name), factory_name)
    executor = factory()
    if not callable(getattr(executor, 'execute', None)):
        raise TypeError('executor factory did not return an execute-capable object')
    if getattr(executor, 'execution_boundary', None) != 'ssh_guest':
        raise ValueError('model runner requires executor.execution_boundary == "ssh_guest"')
    if not callable(getattr(executor, 'contract', None)):
        raise ValueError('model runner requires a versioned executor.contract() receipt')
    return executor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--executor', help='VM executor factory as module:factory')
    parser.add_argument('--pins', type=Path, help='Externally frozen run pins JSON')
    parser.add_argument('--output', default='model-run1')
    parser.add_argument('--check', action='store_true', help='Validate source stack only; start nothing')
    args = parser.parse_args()
    old = Path.home()/'.local/share/akh-medu/lilith-dev-vulkan-20260921'
    required = [old/(name+'.json') for name in ('akhexecd','akhsbxd','akhrtd','akhsessd')]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError('prepared private stack is incomplete')
    if args.check:
        print(json.dumps({'status':'prepared','services_started':False,'gpu_used':False,
                          'executor_configured':bool(args.executor)})); return 0
    if os.environ.get('LILITH_GPU_HANDOFF') != '1':
        raise RuntimeError('refusing to start before explicit LILITH_GPU_HANDOFF=1')
    if not args.executor:
        raise RuntimeError('refusing model code execution without --executor VM factory')
    if not args.pins:
        raise RuntimeError('refusing model run without --pins')
    executor = load_executor(args.executor)
    pins=json.loads(args.pins.read_text())
    if pins.get('schema') != 'lilith/coordination-model-pins@1':
        raise ValueError('unsupported pins schema')
    if pins.get('script_sha256') != file_hash(__file__):
        raise ValueError('model runner differs from frozen pin')
    if pins.get('coordinator_sha256') != file_hash(REPO/'scripts/session-platform/lilith_repository_coordination.py'):
        raise ValueError('repository coordinator differs from frozen pin')
    if pins.get('executor_contract_sha256') != canonical_hash(executor.contract()):
        raise ValueError('executor contract differs from frozen pin')

    out = Path(__file__).resolve().parent/args.output
    out.mkdir(exist_ok=False)
    root = Path.home()/'.local/share/akh-medu'/('lilith-coordination-'+uuid.uuid4().hex)
    root.mkdir(mode=0o700); (root/'credentials').mkdir(mode=0o700)
    (root/'workspaces').mkdir(mode=0o700)
    configs = {name:json.loads((old/(name+'.json')).read_text().replace(str(old),str(root)))
               for name in ('akhexecd','akhsbxd','akhrtd','akhsessd')}
    for name in configs:
        if pins.get('binaries',{}).get(name) != file_hash(REPO/'target/release'/name):
            raise ValueError(name+' binary differs from frozen pin')
    for profile in ('lilith','ptah'):
        native=configs['akhrtd']['runtimes'][profile]['native_candle']
        expected=pins.get('runtimes',{}).get(profile,{})
        for kind in ('model','tokenizer'):
            actual=file_hash(native[kind+'_path'])
            if actual != native[kind+'_sha256'] or actual != expected.get(kind+'_sha256'):
                raise ValueError(profile+' '+kind+' differs from frozen pin')
    identities = ['lilith-coordinator','khnum-inline-patcher','ptah-inline-reviewer']
    for name in ['operator']+list(configs):
        token=secrets.token_urlsafe(32)
        for suffix,value in [('token',token),('sha256',hashlib.sha256(token.encode()).hexdigest())]:
            path=root/'credentials'/f'{name}.{suffix}'; path.write_text(value+'\n'); path.chmod(0o600)
    for config in configs.values():
        for principal in config['principals']: principal['akhs']=identities
        host,port=config['listen'].rsplit(':',1)
        if host != '127.0.0.1': raise ValueError('loopback service required')
        with socket.socket() as sock: sock.bind((host,int(port)))

    runtimes=configs['akhrtd']['runtimes']
    for runtime in runtimes.values():
        Path(runtime['working_dir']).mkdir(parents=True,mode=0o700)
        Path(runtime['isolation']['directory']).mkdir(parents=True,mode=0o700)
    runtimes['khnum']=copy.deepcopy(runtimes['lilith'])
    runtimes['khnum']['working_dir']=str(root/'runtime/khnum')
    runtimes['khnum']['isolation']['directory']=str(root/'runtime/khnum/profile')
    Path(runtimes['khnum']['working_dir']).mkdir(parents=True,mode=0o700)
    Path(runtimes['khnum']['isolation']['directory']).mkdir(parents=True,mode=0o700)
    base=configs['akhsessd']['akhs']
    lilith=copy.deepcopy(base['lilith']); lilith['tools']=[repository_task_tool()]
    lilith['instructions']=('You are Lilith coordinating one bounded repository task. Call '
        'request_repository_task exactly once with the supplied objective, finish the turn after '
        'acknowledgement, then truthfully report returned test and reviewer evidence.')
    khnum=copy.deepcopy(base['lilith-learner']); khnum['runtime']['profile']='khnum'
    khnum['instructions']=('Return only JSON with exactly path and content. Path must be solution.py; '
        'content must be the complete replacement containing one Python function. Do not claim tests ran.')
    ptah=copy.deepcopy(base['ptah-inline-reviewer'])
    configs['akhsessd']['akhs']={'lilith-coordinator':lilith,
        'khnum-inline-patcher':khnum,'ptah-inline-reviewer':ptah}
    for name,config in configs.items(): write(root/(name+'.json'),config)

    fixture=out/'repository'; fixture.mkdir()
    (fixture/'solution.py').write_text('def range_within_u64(start, length, limit):\n    return False\n')
    (fixture/'test_solution.py').write_text('import unittest\nfrom solution import range_within_u64\n\n'
        'class TestRange(unittest.TestCase):\n'
        '    def test_boundaries(self):\n'
        '        m = 18446744073709551615\n'
        '        self.assertTrue(range_within_u64(0, 0, 0))\n'
        '        self.assertTrue(range_within_u64(5, 0, 5))\n'
        '        self.assertTrue(range_within_u64(m - 1, 1, m))\n'
        '        self.assertFalse(range_within_u64(6, 0, 5))\n'
        '        self.assertFalse(range_within_u64(m, 1, m))\n'
        '        self.assertFalse(range_within_u64(-1, 1, 2))\n'
        '        self.assertFalse(range_within_u64(0, m + 1, m))\n')
    protocol={'schema':'lilith/coordination-model-run@1','pins':pins,
        'pins_sha256':file_hash(args.pins),'executor_contract':executor.contract(),
        'executor_contract_sha256':canonical_hash(executor.contract()),
        'task':{'target':'solution.py','objective':
            'Inputs are arbitrary integers. Return false if start, length, or limit is outside 0..18446744073709551615. '
            'Otherwise return whether [start,start+length) fits within [0,limit). An empty range at start==limit is valid; '
            'reject start>limit and use length <= limit-start after checking start <= limit.',
            'solution_sha256':file_hash(fixture/'solution.py'),'tests_sha256':file_hash(fixture/'test_solution.py'),
            'test_command':['python3','-m','unittest','-q']},
        'scope':'orchestration smoke with one bounded Python reference task; not general repository or illumos competence'}
    write(out/'protocol.json',protocol)
    ledger=Ledger(out/'coordination.sqlite')
    token=(root/'credentials/operator.token').read_text().strip()
    def api(path,body=None):
        request=urllib.request.Request('http://'+configs['akhsessd']['listen']+path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
    def health(name):
        request=urllib.request.Request('http://'+configs[name]['listen']+'/healthz',
            headers={'Authorization':'Bearer '+token})
        with urllib.request.urlopen(request,timeout=2) as response:return json.load(response)
    processes=[]; logs=[]
    result={'status':'running','model_backed':True,'executor':args.executor,
            'execution_boundary':'ssh_guest','services_stopped':False}
    try:
        for name,config in configs.items():
            log=(out/(name+'.log')).open('w'); logs.append(log)
            process=subprocess.Popen([str(REPO/'target/release'/name),'--config',str(root/(name+'.json'))],
                stdout=log,stderr=subprocess.STDOUT,cwd=REPO,env=dict(os.environ,
                    AKH_GPU_MATMUL='0',AKH_GPU_QUANT='0',AKH_VULKAN_PIPELINE_CACHE_ENTRIES='16384',
                    VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json'))
            processes.append(process)
            for _ in range(150):
                if process.poll() is not None: raise RuntimeError(name+' exited')
                try:
                    if health(name)['service_id']==name: break
                except Exception: time.sleep(.2)
            else: raise TimeoutError(name+' readiness')
        identity=uuid.uuid4().hex
        session=api('/v1/sessions',{'request_id':identity,'akh':'lilith-coordinator','workspace_profile':'local-dev'})
        turn=api('/v1/sessions/'+session['id']+'/inputs',{'request_id':identity+'-input','text':
            'Implement range_within_u64(start, length, limit). Inputs are arbitrary integers. Return false if any input is '
            'outside 0..18446744073709551615. Otherwise return whether [start,start+length) fits within [0,limit). '
            'An empty range at start==limit is valid; reject start>limit and use length <= limit-start after checking start <= limit. '
            'Delegate the bounded repository task.'})
        coordinator=RepositoryCoordinator(api,ledger,session['id'],turn['id'],fixture,'solution.py',
            ['python3','-m','unittest','-q'],executor=executor)
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            rows=coordinator.tick()
            if rows and rows[0]['status'] in {'completed','failed','unknown','cancelled','rejected','stale'}: break
            state=api('/v1/sessions/'+session['id'])
            author=next(item for item in state['turns'] if item['id']==turn['id'])
            if author['status'] in {'failed','unknown','cancelled'} and not rows: break
            time.sleep(.5)
        else: raise TimeoutError('coordination deadline; ledger remains resumable')
        rows=coordinator.tick(); write(out/'assignments.json',rows)
        sessions={'lilith':api('/v1/sessions/'+session['id'])}
        if rows:
            row=rows[0]
            if row.get('patch_session'): sessions['khnum']=api('/v1/sessions/'+row['patch_session'])
            if row.get('review_session'): sessions['ptah']=api('/v1/sessions/'+row['review_session'])
            write(out/'sessions.json',sessions)
            write(out/'outputs.json',{'source_snapshot':row.get('source'),
                'patch_proposal':row.get('proposal'),'verification':row.get('verification'),
                'reviewer_output':row.get('review'),'lilith_report':row.get('report')})
        result.update(status='completed' if rows and rows[0]['status']=='completed' else 'failed',
                      truth_review='pending_manual' if rows and rows[0]['status']=='completed' else 'not_applicable',
                      assignment_status=rows[0]['status'] if rows else 'missing',session_id=session['id'],turn_id=turn['id'])
    except Exception as error:
        result.update(status='interrupted',error=str(error))
    finally:
        ledger.close()
        for process in reversed(processes):
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        for log in logs: log.close()
        result['services_stopped']=all(process.poll() is not None for process in processes)
        write(out/'result.json',result)
    return 0 if result['status']=='completed' else 1


if __name__=='__main__': raise SystemExit(main())
