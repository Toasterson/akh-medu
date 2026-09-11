#!/usr/bin/env python3
"""Four-process local conformance smoke. Uses a deterministic peer, never a native model.

Run after cargo build -p akh-session-services. All processes/files are private to
this invocation. No deployed services or existing workspace are changed.
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]


def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait(condition, label, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = condition()
            if value:
                return value
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.1)
    raise AssertionError(f'timed out: {label}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir', type=pathlib.Path, default=ROOT / 'target/debug')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='akh-platform-smoke-') as directory:
        root = pathlib.Path(directory).resolve()
        services = ['akhsessd', 'akhrtd', 'akhsbxd', 'akhexecd', 'akheventd']
        ports = {name: port() for name in services}
        identities = ['user', 'akhsessd', 'akhrtd', 'akhsbxd', 'akheventd', 'webhook']
        tokens = {name: secrets.token_urlsafe(32) for name in identities}
        for name, token in tokens.items():
            p = root / f'{name}.token'
            p.write_text(token)
            p.chmod(0o600)
        (root / 'workspaces').mkdir()
        def endpoint(service, caller):
            return {'url': f'http://127.0.0.1:{ports[service]}', 'token_file': str(root / f'{caller}.token')}
        def principal(name, role):
            return {'id': name, 'token_file': str(root / f'{name}.token'), 'roles': [role], 'akhs': ['test'], 'sources': ['alerts']}
        configs = {}
        for name in services:
            configs[name] = {'service_id': name, 'listen': f'127.0.0.1:{ports[name]}', 'data_dir': str(root / name),
                'principals': [principal('user', 'admin')], 'endpoints': {}}
        configs['akhsessd']['principals'] += [principal('akhrtd', 'runtime'), principal('akheventd', 'event')]
        configs['akhsessd']['endpoints'] = {'akhrtd': endpoint('akhrtd','akhsessd'), 'sandboxes': endpoint('akhsbxd','akhsessd'), 'akhexecd': endpoint('akhexecd','akhsessd')}
        configs['akhsessd']['akhs'] = {'test': {'runtime': {'runtime': 'akhrtd', 'profile': 'fixture', 'revision': 'smoke-1', 'kind': 'codex', 'execution_mode': 'colocated', 'model': None},
            'instructions': 'Use supplied tools for the requested operation.', 'workspace_profiles': ['local'], 'approval_policy': 'ask_writes',
            'tools': [{'name': 'write_file', 'description': 'Write with revision checks', 'read_only': False,
                'input_schema': {'type': 'object', 'required': ['path','content','create'], 'properties': {'path': {'type':'string'},'content':{'type':'string'},'create':{'type':'boolean'}}, 'additionalProperties':False}, 'destination': {'kind':'workspace','operation':'write'}}]}}
        configs['akhsessd']['automations'] = {'alert': {'id':'alert','enabled':True,'principal':'user','akh':'test','workspace_profile':'local','instructions':'Summarize this alert','sources':['alerts'],'target':{'kind':'incident'},'max_active':2,'max_age_ms':60000}}
        configs['akhrtd']['principals'] += [principal('akhsessd','session')]
        configs['akhrtd']['endpoints'] = {'sessions':endpoint('akhsessd','akhrtd')}
        configs['akhrtd']['runtimes'] = {'fixture':{'revision':'smoke-1','kind':'codex','working_dir':str(root/'workspaces'),'binary':str(ROOT/'scripts/session-platform/provider-fixture.py'),'execution_mode':'colocated','timeout_secs':60}}
        configs['akhsbxd']['principals'] += [principal('akhsessd','session')]
        configs['akhsbxd']['endpoints'] = {'akhexecd':endpoint('akhexecd','akhsbxd')}
        configs['akhsbxd']['workspaces'] = {'local':{'revision':'local-1','site':'smoke','provisioner':{'kind':'existing','executor':'akhexecd','root':str(root/'workspaces')}}}
        configs['akhexecd']['principals'] += [principal('akhsessd','session'),principal('akhsbxd','sandbox')]
        configs['akhexecd']['workspace_root'] = str(root/'workspaces')
        configs['akheventd']['principals'] += [principal('webhook','event')]
        configs['akheventd']['endpoints'] = {'sessions':endpoint('akhsessd','akheventd')}
        configs['akheventd']['routes'] = [{'id':'alerts','source':'alerts','event_type':'alert.firing','automation':'alert'}]
        processes = {}
        logs = {}
        def request(service, path, body=None, caller='user'):
            headers = {'Authorization':f'Bearer {tokens[caller]}','Content-Type':'application/json'}
            req = urllib.request.Request(f'http://127.0.0.1:{ports[service]}{path}', data=None if body is None else json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.load(response)
        def start(name):
            config_path = root/f'{name}.json'
            config_path.write_text(json.dumps(configs[name]))
            logs[name] = open(root/f'{name}.log','a')
            processes[name] = subprocess.Popen([str(args.bin_dir/name),'--config',str(config_path)],stdout=logs[name],stderr=logs[name],env={**os.environ,'RUST_LOG':'warn'})
            wait(lambda: request(name,'/healthz'),f'{name} listening')
        def stop(name):
            processes[name].terminate()
            processes[name].wait(timeout=10)
            logs[name].close()
        try:
            for name in reversed(services):
                start(name)
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{ports["akhsessd"]}/v1/sessions')
                raise AssertionError('anonymous session access was accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 401
            session = request('akhsessd','/v1/sessions',{'request_id':'create','akh':'test','workspace_profile':'local','conversation_key':'conversation'})
            sid = session['id']
            payload = {'request_id':'write','text':json.dumps({'tool':'write_file','arguments':{'path':'marker','content':'once','create':True}})}
            path = f'/v1/sessions/{sid}'
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                turns = list(pool.map(lambda _:request('akhsessd',path+'/inputs',payload),range(8)))
            assert len({t['id'] for t in turns}) == 1
            turn = turns[0]['id']
            pending = wait(lambda:request('akhsessd',path+'/pending'),'persisted approval')[0]
            state = request('akhsessd',path)
            workspace = pathlib.Path(state['workspace']['root'])
            assert not (workspace/'marker').exists(), 'write happened before approval'
            stop('akhsessd')
            start('akhsessd')
            assert request('akhsessd',path+'/pending')[0]['id'] == pending['id']
            request('akhsessd',f'{path}/turns/{turn}/responses/{pending["id"]}',{'approved':True})
            final = wait(lambda:(s if (s:=request('akhsessd',path))['turns'][0]['status']=='completed' else None),'completed brokered tool turn')
            assert (workspace/'marker').read_text() == 'once'
            continuation = final['provider_session']
            next_turn = request('akhsessd',path+'/inputs',{'request_id':'next','text':'second turn'})
            final = wait(lambda:(s if (s:=request('akhsessd',path))['turns'][-1]['status']=='completed' else None),'exact provider continuation')
            assert final['provider_session'] == continuation
            assert json.loads((workspace/'fixture-provider-session.json').read_text())['turns'] == 2
            hang = request('akhsessd',path+'/inputs',{'request_id':'hang','text':'{"hang":true}'})
            wait(lambda:request('akhsessd',path)['turns'][-1]['status']=='running','running cancellable turn')
            request('akhsessd',f'{path}/turns/{hang["id"]}/cancel',{})
            wait(lambda:request('akhsessd',path)['turns'][-1]['status']=='cancelled','runtime cancellation')
            event={'schema_version':1,'id':'alert-1','source':'alerts','type':'alert.firing','occurred_at':int(time.time()*1000),'subject':'host-1','correlation_id':'incident-1','data':{'summary':'disk pressure'}}
            receipt=request('akheventd','/v1/events',event,caller='webhook')
            assert request('akheventd','/v1/events',event,caller='webhook')==receipt
            delivery=wait(lambda:next((d for d in request('akheventd','/v1/deliveries') if d['admission']),None),'durable event delivery')
            assert delivery['admission']['session_id'] != sid
            events=request('akhsessd',path+'/events')
            sequences=[e['sequence'] for e in events]
            assert sequences==sorted(set(sequences))
            print(json.dumps({'result':'passed','processes':len(services),'checks':['authentication','duplicate admission','approval before write','session restart','workspace receipt','exact continuation','cancellation','webhook dedupe and delivery','ordered replay'],'provider':'deterministic fixture; native conformance not claimed'}))
        except BaseException:
            for name in logs:
                print(f'--- {name} ---\n{(root/f"{name}.log").read_text()}')
            raise
        finally:
            for name, process in processes.items():
                if process.poll() is None:
                    process.terminate()
            for process in processes.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs.values():
                log.close()


if __name__ == '__main__':
    main()
