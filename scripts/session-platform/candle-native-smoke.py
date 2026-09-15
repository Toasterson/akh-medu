#!/usr/bin/env python3
"""Opt-in four-daemon HTTP canary with a real resident Candle model.

Creates private loopback services, credentials and workspace. Never contacts the
homelab or modifies deployed services. Requires exclusive GPU access. Temporary
workspace/state is removed on exit; only test-owned child processes are stopped.
"""
import argparse
import copy
import json
import os
import pathlib
import re
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=pathlib.Path, default=ROOT / 'target/release')
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--tokenizer', type=pathlib.Path, required=True)
    parser.add_argument('--tokenizer-sha256', required=True)
    parser.add_argument('--device', choices=['hip', 'vulkan', 'cpu'], required=True)
    parser.add_argument('--switch-device', choices=['hip', 'vulkan'],
                        help='also round-trip another device in the same runtime process')
    parser.add_argument('--require-prefix-reuse', action='store_true',
                        help='require observed native token-prefix reuse across ordinary turns')
    args = parser.parse_args()
    if args.switch_device == args.device:
        parser.error('--switch-device must differ from --device')
    services = ['akhexecd', 'akhsbxd', 'akhrtd', 'akhsessd']
    for name in services:
        if not (args.bin_dir / name).is_file():
            parser.error(f'missing binary: {args.bin_dir / name}')
    with tempfile.TemporaryDirectory(prefix='akh-candle-http-canary-') as directory:
        root = pathlib.Path(directory)
        ports = {name: free_port() for name in services}
        tokens = {name: secrets.token_urlsafe(32) for name in ['operator'] + services}
        for name, token in tokens.items():
            with open(root / f'{name}.token', 'x', opener=lambda path, flags: os.open(path, flags, 0o600)) as stream:
                stream.write(token)

        def principal(name, role):
            akhs = ['ptah-canary'] + (['ptah-switch-canary'] if args.switch_device else [])
            return {'id': name, 'token_file': str(root / f'{name}.token'), 'roles': [role], 'akhs': akhs}

        def endpoint(name, caller):
            return {'url': f'http://127.0.0.1:{ports[name]}', 'token_file': str(root / f'{caller}.token')}

        configs = {name: {'service_id': name, 'listen': f'127.0.0.1:{ports[name]}',
                         'data_dir': str(root / name), 'principals': [principal('operator', 'admin')]}
                   for name in services}
        workspace = root / 'workspace'
        workspace.mkdir()
        configs['akhexecd'].update(workspace_root=str(workspace))
        configs['akhexecd']['principals'] += [principal('akhsessd', 'session'), principal('akhsbxd', 'sandbox')]
        configs['akhsbxd'].update(endpoints={'akhexecd': endpoint('akhexecd', 'akhsbxd')}, workspaces={
            'canary': {'revision': '1', 'site': 'local-test', 'provisioner': {
                'kind': 'existing', 'executor': 'akhexecd', 'root': str(workspace)}}})
        configs['akhsbxd']['principals'].append(principal('akhsessd', 'session'))
        configs['akhrtd']['principals'].append(principal('akhsessd', 'session'))
        configs['akhrtd'].update(endpoints={'sessions': endpoint('akhsessd', 'akhrtd')}, runtimes={
            'canary': {'revision': '1', 'kind': 'internal', 'execution_mode': 'brokered',
                'working_dir': str(workspace), 'binary': str(args.bin_dir / 'akhrtd'),
                'isolation': {'directory': str(root / 'runtime-profile')}, 'timeout_secs': 600,
                'native_candle': {'model': 'ptah-exact', 'model_path': str(args.model.resolve()),
                    'model_sha256': args.model_sha256, 'tokenizer_path': str(args.tokenizer.resolve()),
                    'tokenizer_sha256': args.tokenizer_sha256,
                    'device': {'kind': args.device, **({} if args.device == 'cpu' else {'ordinal': 0})},
                    'max_context_tokens': 2048, 'max_output_tokens': 256, 'max_iterations': 4}}})
        tools = [
            {'name': 'write_file', 'description': 'Create a UTF-8 workspace file.', 'read_only': False,
             'input_schema': {'type': 'object', 'required': ['path', 'content', 'create'],
                'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}, 'create': {'type': 'boolean'}},
                'additionalProperties': False}, 'destination': {'kind': 'workspace', 'operation': 'write'}},
            {'name': 'read_file', 'description': 'Read a UTF-8 workspace file.', 'read_only': True,
             'input_schema': {'type': 'object', 'required': ['path'], 'properties': {'path': {'type': 'string'}},
                'additionalProperties': False}, 'destination': {'kind': 'workspace', 'operation': 'read'}}]
        configs['akhsessd']['principals'].append(principal('akhrtd', 'runtime'))
        configs['akhsessd'].update(endpoints={
            'akhrtd': endpoint('akhrtd', 'akhsessd'), 'sandboxes': endpoint('akhsbxd', 'akhsessd'),
            'akhexecd': endpoint('akhexecd', 'akhsessd')}, akhs={'ptah-canary': {
                'runtime': {'runtime': 'akhrtd', 'profile': 'canary', 'revision': '1', 'kind': 'internal',
                            'execution_mode': 'brokered', 'model': 'ptah-exact'},
                'instructions': 'Answer briefly. Use the supplied tools for files. Do not invent tool results.',
                'workspace_profiles': ['canary'], 'approval_policy': 'ask_writes', 'tools': tools}})
        if args.switch_device:
            alternate = copy.deepcopy(configs['akhrtd']['runtimes']['canary'])
            alternate['isolation']['directory'] = str(root / 'runtime-profile-switch')
            alternate['native_candle']['device'] = {'kind': args.switch_device, 'ordinal': 0}
            configs['akhrtd']['runtimes']['switch'] = alternate
            alternate_akh = copy.deepcopy(configs['akhsessd']['akhs']['ptah-canary'])
            alternate_akh['runtime']['profile'] = 'switch'
            alternate_akh['tools'] = []
            configs['akhsessd']['akhs']['ptah-switch-canary'] = alternate_akh
        processes, logs = {}, {}

        def request(name, path, body=None, token=None):
            headers = {'Authorization': 'Bearer ' + (token or tokens['operator']), 'Content-Type': 'application/json'}
            req = urllib.request.Request(f'http://127.0.0.1:{ports[name]}{path}',
                data=None if body is None else json.dumps(body).encode(), headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.load(response)

        def wait(condition, label, seconds=600):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                for name, process in processes.items():
                    if process.poll() is not None:
                        raise AssertionError(f'{name} exited: {process.returncode}')
                try:
                    value = condition()
                    if value:
                        return value
                except urllib.error.HTTPError:
                    raise
                except (ConnectionError, urllib.error.URLError):
                    pass
                time.sleep(.2)
            raise AssertionError(f'timed out: {label}')

        def start(name):
            path = root / f'{name}.json'
            path.write_text(json.dumps(configs[name]))
            logs[name] = open(root / f'{name}.log', 'a')
            log_filter = ('warn,akh_session_services::internal::native=debug'
                          if name == 'akhrtd' and args.require_prefix_reuse else 'warn')
            processes[name] = subprocess.Popen([str(args.bin_dir / name), '--config', str(path)],
                stdout=logs[name], stderr=logs[name], env={**os.environ, 'RUST_LOG': log_filter})
            wait(lambda: request(name, '/healthz'), f'{name} ready', seconds=30)

        def stop(name):
            process = processes.pop(name)
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            logs[name].close()

        try:
            for name in services:
                start(name)
            print(json.dumps({'phase': 'isolated_stack_ready', 'device': args.device}), flush=True)
            try:
                request('akhsessd', '/v1/sessions', token='invalid-test-token')
                raise AssertionError('invalid credential accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 401
            session = request('akhsessd', '/v1/sessions', {
                'request_id': 'create', 'akh': 'ptah-canary', 'workspace_profile': 'canary'})
            path = '/v1/sessions/' + session['id']
            turn = request('akhsessd', path + '/inputs', {'request_id': 'write',
                'text': 'Remember the word cedar. Use write_file to create path proof.txt with exactly cedar as its content and create=true. Then confirm briefly.'})
            pending = wait(lambda: request('akhsessd', path + '/pending'), 'model-generated write approval')[0]
            print(json.dumps({'phase': 'real_write_awaits_approval'}), flush=True)
            state = request('akhsessd', path)
            fixture = pathlib.Path(state['workspace']['root']) / 'proof.txt'
            assert not fixture.exists(), 'model write happened without approval'
            stop('akhsessd')
            start('akhsessd')
            assert request('akhsessd', path + '/pending')[0]['id'] == pending['id']
            request('akhsessd', f'{path}/turns/{turn["id"]}/responses/{pending["id"]}', {'approved': True})

            def completed_at(session_path):
                state = request('akhsessd', session_path)
                status = state['turns'][-1]['status']
                if status in ('failed', 'unknown', 'cancelled'):
                    raise AssertionError(f'turn did not complete: {json.dumps(state["turns"][-1])}')
                return state if status == 'completed' else None

            def completed():
                return completed_at(path)

            first = wait(completed, 'approved real model write')
            assert fixture.read_text().strip() == 'cedar'
            print(json.dumps({'phase': 'approved_write_completed_after_session_restart'}), flush=True)
            binding = first['provider_session']
            request('akhsessd', path + '/inputs', {'request_id': 'read',
                'text': 'Use read_file to read proof.txt. Then reply with its contents.'})
            second = wait(completed, 'real model read and continuation')
            assert second['provider_session'] == binding
            records = request('akhsessd', path + '/tools')['records']
            for name in ('write_file', 'read_file'):
                completed_tools = [record for record in records if record['call']['name'] == name
                                   and not record['unknown'] and record.get('result')
                                   and record['result']['success'] and record['result'].get('receipt')]
                assert completed_tools, f'missing successful durable {name} receipt'
            assert 'cedar' in (second['turns'][-1]['output'] or '').lower()
            print(json.dumps({'phase': 'read_and_durable_receipts_verified'}), flush=True)
            if args.require_prefix_reuse:
                previous_log_chars = len((root / 'akhrtd.log').read_text())
                for index in range(2):
                    request('akhsessd', path + '/inputs', {
                        'request_id': f'prefix-reuse-{index}',
                        'text': 'What word did I ask you to remember? Reply with only that word, without tools.'})
                    continued = wait(completed, 'ordinary native prefix continuation')
                    assert continued['provider_session'] == binding
                    assert (continued['turns'][-1]['output'] or '').strip().lower().strip('.') == 'cedar'
                runtime_log = re.sub(r'\x1b\[[0-9;]*m', '',
                                     (root / 'akhrtd.log').read_text()[previous_log_chars:])
                reused = [int(value) for value in re.findall(r'reused_prompt_tokens=(\d+)', runtime_log)]
                assert reused and max(reused) > 0, 'no actual native prefix reuse observed'
                print(json.dumps({'phase': 'native_prefix_reuse_verified',
                                  'max_reused_prompt_tokens': max(reused)}), flush=True)
            if args.switch_device:
                runtime_pid = processes['akhrtd'].pid
                alternate = request('akhsessd', '/v1/sessions', {
                    'request_id': 'create-switch', 'akh': 'ptah-switch-canary', 'workspace_profile': 'canary'})
                alternate_path = '/v1/sessions/' + alternate['id']
                request('akhsessd', alternate_path + '/inputs', {
                    'request_id': 'switch-device', 'text': 'Reply with only the word cedar.'})
                switched = wait(lambda: completed_at(alternate_path), 'same-process device switch')
                assert (switched['turns'][-1]['output'] or '').strip().lower().strip('.') == 'cedar'
                print(json.dumps({'phase': 'alternate_device_completed', 'device': args.switch_device}), flush=True)
                request('akhsessd', path + '/inputs', {'request_id': 'return-device',
                    'text': 'What word did I originally ask you to remember? Reply with only that word, without tools.'})
                returned = wait(completed, 'original-device durable continuation')
                assert returned['provider_session'] == binding
                assert (returned['turns'][-1]['output'] or '').strip().lower().strip('.') == 'cedar'
                assert processes['akhrtd'].pid == runtime_pid, 'device switch must not restart the runtime'
                print(json.dumps({'phase': 'same_process_device_roundtrip_verified',
                    'initial_device': args.device, 'alternate_device': args.switch_device}), flush=True)
            # Restart the runtime as well: durable model/session pins must survive
            # losing every in-memory GPU tensor and reloading the exact artifacts.
            stop('akhrtd')
            start('akhrtd')
            request('akhsessd', path + '/inputs', {'request_id': 'resume',
                'text': 'What word did I originally ask you to remember? Reply with only that word, without tools.'})
            third = wait(completed, 'durable continuation after runtime restart')
            assert third['provider_session'] == binding
            assert (third['turns'][-1]['output'] or '').strip().lower().strip('.') == 'cedar'
            print(json.dumps({'result': 'passed', 'device': args.device, 'switch_device': args.switch_device, 'processes': 4,
                'checks': ['authentication', 'real model write', 'approval before effects',
                           'session restart with pending approval', 'real model read and durable receipts',
                           'exact provider continuation', 'runtime restart and artifact reload']
                          + (['same-process device roundtrip'] if args.switch_device else [])
                          + (['observed native prefix reuse'] if args.require_prefix_reuse else []),
                'scope': 'isolated loopback stack; not homelab cutover or performance acceptance'}), flush=True)
        except BaseException:
            for name in logs:
                print(f'{name} log:\n{(root / (name + ".log")).read_text()}', flush=True)
            raise
        finally:
            for name in list(reversed(processes)):
                stop(name)


if __name__ == '__main__':
    main()
