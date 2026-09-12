#!/usr/bin/env python3
"""Installed Codex app-server + synthetic Responses peer + real akhexecd.

No login, auth-cache copy, cloud model or conformance certificate. This measures
native protocol/tool registration and supplied-tool execution on one machine.
Build the native-provider-probe example with the runtime feature first.
"""
import argparse
import hashlib
import http.server
import json
import os
import pathlib
import queue
import secrets
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex', type=pathlib.Path, required=True)
    parser.add_argument('--bin-dir', type=pathlib.Path, default=ROOT / 'target/debug')
    args = parser.parse_args()
    assert args.codex.is_absolute()
    version = subprocess.run([str(args.codex), '--version'], check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix='akh-codex-native-wire-') as directory:
        root = pathlib.Path(directory).resolve()
        workspaces, runtime = root / 'workspaces', root / 'runtime'
        workspaces.mkdir(); runtime.mkdir()
        session, workspace, sandbox = (str(uuid.uuid4()) for _ in range(3))
        bound_root = workspaces / 'probe'
        token = secrets.token_urlsafe(32)
        token_file = root / 'executor.token'; token_file.write_text(token); token_file.chmod(0o600)
        endpoint = f'http://127.0.0.1:{free_port()}'
        config = {'service_id': 'executor', 'listen': endpoint.split('//')[1],
                  'data_dir': str(root / 'executor'), 'workspace_root': str(workspaces),
                  'principals': [{'id': 'probe', 'token_file': str(token_file), 'roles': ['admin']}]}
        config_file = root / 'executor.json'; config_file.write_text(json.dumps(config))

        def executor(path, body=None):
            request = urllib.request.Request(endpoint + path, data=None if body is None else json.dumps(body).encode(),
                headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)

        requests, tools_seen, peer_errors = [], [], []
        # Each returned function is a synthetic model action consumed by the
        # actual installed Codex. Native shell must never reach the host callback.
        actions = [
            ('exec_command', None, {'cmd': 'touch native-must-not-exist'}),
            ('exec_command', 'functions', {'cmd': 'touch native-must-not-exist'}),
            ('apply_patch', 'functions', '*** Begin Patch\n*** Add File: native-patch-must-not-exist\n+forbidden\n*** End Patch'),
            ('write_file', 'akh', {'path': 'probe.txt', 'content': 'native wire receipt', 'create': True}),
            ('read_file', 'akh', {'path': 'probe.txt'}),
            ('run_command', 'akh', {'command': "printf 'executor-process' > process.txt", 'timeout_ms': 5000}),
            None,
            ('read_file', 'akh', {'path': 'probe.txt'}),
            ('write_file', 'akh', {'path': 'denied-probe.txt', 'content': 'must not write', 'create': True}),
            ('exec_command', 'functions', {'cmd': 'touch denied-probe.txt'}),
            ('apply_patch', 'functions', '*** Begin Patch\n*** Add File: native-patch-must-not-exist\n+forbidden\n*** End Patch'),
            ('view_image', 'functions', {'path': str(runtime / 'native-image.png')}),
            None,
        ]

        class Peer(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                try:
                    assert self.path == '/v1/responses', self.path
                    assert self.headers.get('Authorization') is None, 'fixture unexpectedly received credentials'
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    index = len(requests); requests.append(body)
                    assert index < len(actions), 'unexpected extra model request'
                    tools_seen.append(body.get('tools', []))
                    action = actions[index]
                    if action:
                        name, namespace, arguments = action
                        item = {'type': 'function_call', 'id': f'fc_{index}', 'call_id': f'call_{index}',
                                'name': name, 'arguments': json.dumps(arguments), 'status': 'completed'}
                        if isinstance(arguments, str):
                            item['type'] = 'custom_tool_call'; item['input'] = arguments; del item['arguments']
                        if namespace: item['namespace'] = namespace
                    else:
                        item = {'type': 'message', 'id': f'msg_{index}', 'role': 'assistant', 'status': 'completed',
                                'phase': 'final_answer', 'content': [{'type': 'output_text', 'text': 'probe complete', 'annotations': []}]}
                    response = {'id': f'resp_{index}', 'object': 'response', 'status': 'completed', 'output': [item],
                                'usage': {'input_tokens': 10, 'output_tokens': 10, 'total_tokens': 20}}
                    events = [{'type': 'response.created', 'response': {**response, 'status': 'in_progress', 'output': []}},
                              {'type': 'response.output_item.added', 'output_index': 0, 'item': item},
                              {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
                              {'type': 'response.completed', 'response': response}]
                    data = ''.join('event: ' + event['type'] + '\ndata: ' + json.dumps(event) + '\n\n' for event in events).encode()
                    self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
                except Exception as error:
                    peer_errors.append(str(error)); self.send_error(500)

        peer = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Peer)
        thread = threading.Thread(target=peer.serve_forever, daemon=True); thread.start()
        native_config = f'''model_provider = "native_wire_fixture"
model = "gpt-5.4"
cli_auth_credentials_store = "file"
[model_providers.native_wire_fixture]
name = "Local protocol fixture"
base_url = "http://127.0.0.1:{peer.server_port}/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
'''
        profile = {'revision': 'native-wire-1', 'kind': 'codex', 'binary': str(args.codex), 'working_dir': str(runtime),
                   'isolation': {'directory': str(root / 'profile'), 'managed_files': {'codex/config.toml': native_config}},
                   'execution_mode': 'brokered', 'timeout_secs': 45}
        binding = {'id': workspace, 'sandbox_id': sandbox, 'executor': 'executor', 'generation': 1, 'site': 'local-probe',
                   'root': str(bound_root), 'platform': os.uname().sysname, 'confinement': 'local-test-only'}
        specs = []
        for name, operation, properties, required in [
            ('write_file', 'write', {'path': {'type': 'string'}, 'content': {'type': 'string'}, 'create': {'type': 'boolean'}}, ['path', 'content', 'create']),
            ('read_file', 'read', {'path': {'type': 'string'}}, ['path']),
            ('run_command', 'exec_start', {'command': {'type': 'string'}, 'timeout_ms': {'type': 'integer'}}, ['command', 'timeout_ms'])]:
            specs.append({'name': name, 'description': 'Native wire probe supplied workspace operation', 'read_only': operation == 'read',
                          'input_schema': {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False},
                          'destination': {'kind': 'workspace', 'operation': operation}})
        children, opened, calls, events = [], [], [], []
        try:
            log = (root / 'executor.log').open('w'); opened.append(log)
            child = subprocess.Popen([str(args.bin_dir / 'akhexecd'), '--config', str(config_file)], stdout=log, stderr=log); children.append(child)
            for _ in range(100):
                try:
                    executor('/healthz'); break
                except OSError: time.sleep(.1)
            else: raise AssertionError('executor did not become ready')
            binding = executor('/v1/workspaces', {'session_id': session, 'binding': binding})
            provider_session = None
            for iteration in range(2):
                turn = str(uuid.uuid4())
                job = {'id': turn, 'session_id': session, 'turn_id': turn, 'generation': iteration + 1, 'workspace': binding,
                       'binding': {'runtime': 'probe', 'profile': 'native-wire', 'revision': 'native-wire-1', 'kind': 'codex', 'model': 'gpt-5.4', 'execution_mode': 'brokered'},
                       'provider_session': provider_session, 'input': {'origin': {'kind': 'human', 'principal': 'probe'}, 'text': 'Synthetic local protocol test.'},
                       'instructions': 'Use only the supplied Akh workspace tools.', 'tools': specs, 'approval_policy': 'ask_writes'}
                path = root / 'probe.json'; path.write_text(json.dumps({'profile': profile, 'request': job}))
                log = (root / f'probe-{iteration}.log').open('w'); opened.append(log)
                child = subprocess.Popen([str(args.bin_dir / 'examples/native-provider-probe'), str(path)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1); children.append(child)
                deadline = time.monotonic() + 55
                lines = queue.Queue()
                def read_lines(stream, target):
                    for line in stream: target.put(line)
                    target.put(None)
                reader = threading.Thread(target=read_lines, args=(child.stdout, lines), daemon=True); reader.start()
                completed = False
                while time.monotonic() < deadline:
                    try: line = lines.get(timeout=.25)
                    except queue.Empty: continue
                    if not line: break
                    value = json.loads(line)
                    if value['type'] == 'event':
                        events.append(value['event'])
                        if value['event']['kind'] == 'bound':
                            bound = value['event']['provider_session']
                            assert provider_session in (None, bound), 'native resume changed session'
                            provider_session = bound
                    elif value['type'] == 'tool':
                        spec = next((spec for spec in specs if spec['name'] == value['name']), None)
                        assert spec is not None, 'native tool reached the supplied-tool callback'
                        calls.append(value['name'])
                        if value['arguments'].get('path') == 'denied-probe.txt':
                            child.stdin.write(json.dumps({'success': False, 'text': 'probe executor unavailable; nothing ran', 'receipt': None}) + '\n')
                            child.stdin.flush(); continue
                        op = {'id': hashlib.sha256(json.dumps([turn, value['id']]).encode()).hexdigest(), 'session_id': session,
                              'turn_id': turn, 'workspace_id': workspace, 'generation': 1,
                              'operation': spec['destination']['operation'], 'arguments': value['arguments']}
                        receipt = executor('/v1/operations', op)
                        for _ in range(100):
                            if receipt['status'] in ('completed', 'failed', 'cancelled'): break
                            time.sleep(.05); receipt = executor('/v1/operations/' + op['id'])
                        assert receipt['result']['success'] is True, receipt
                        assert executor('/v1/operations', op) == receipt, 'operation replay changed its receipt'
                        child.stdin.write(json.dumps(receipt['result']) + '\n'); child.stdin.flush()
                    elif value['type'] == 'completed':
                        assert value['output'] == 'probe complete'; completed = True; break
                    else: raise AssertionError(value)
                assert completed, 'native provider did not complete'
                assert child.wait(timeout=8) == 0
                reader.join(timeout=3)
            assert not peer_errors, peer_errors
            assert len(requests) == len(actions)
            assert calls == ['write_file', 'read_file', 'run_command', 'read_file', 'write_file'], calls
            assert (bound_root / 'probe.txt').read_text() == 'native wire receipt'
            assert (bound_root / 'process.txt').read_text() == 'executor-process'
            assert not (runtime / 'native-must-not-exist').exists()
            assert not (runtime / 'native-patch-must-not-exist').exists()
            assert not (runtime / 'denied-probe.txt').exists()
            assert not (bound_root / 'denied-probe.txt').exists()
            assert not (runtime / 'probe.txt').exists()
            def catalog_names(catalog):
                names = []
                for tool in catalog:
                    if tool.get('type') == 'namespace':
                        names.extend(tool['name'] + '.' + child['name'] for child in tool.get('tools', []))
                    else: names.append(tool.get('name', tool.get('type')))
                return names
            names = catalog_names(tools_seen[0])
            forbidden = {'exec_command', 'shell', 'shell_command', 'apply_patch', 'view_image', 'read_file', 'write_file'}
            for catalog in tools_seen:
                assert not any(name.split('.')[-1] in forbidden and not name.startswith('akh.') for name in catalog_names(catalog)), catalog
            outputs = [item for body in requests for item in body.get('input', []) if item.get('type') in ('function_call_output', 'custom_tool_call_output')]
            native_calls = ['call_0', 'call_1', 'call_2', 'call_9', 'call_10', 'call_11']
            refusals = {call: next((item.get('output') for item in outputs if item.get('call_id') == call), None) for call in native_calls}
            assert all(isinstance(value, str) and value.startswith('unsupported ') for value in refusals.values()), refusals
            assert any('probe executor unavailable; nothing ran' in json.dumps(item.get('output')) for item in outputs)
            print(json.dumps({'result': 'passed', 'codex_version': version, 'calls': calls, 'model_requests': len(requests),
                              'binary_sha256': hashlib.sha256(args.codex.read_bytes()).hexdigest(), 'platform': os.uname().sysname + '-' + os.uname().machine,
                              'advertised_tools': names, 'native_refusals': refusals,
                              'not_proven': ['live model behavior', 'kernel confinement', 'cross-host transport', 'production conformance']}))
        except Exception:
            print(json.dumps({'requests': requests, 'peer_errors': peer_errors, 'events': events}, default=str))
            for path in root.glob('probe-*.log'): print(path.read_text(errors='replace')[-3000:])
            raise
        finally:
            for child in children:
                if child.stdin is not None and not child.stdin.closed: child.stdin.close()
                if child.poll() is None: child.terminate()
            for child in children:
                try: child.wait(timeout=8)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            peer.shutdown(); peer.server_close(); thread.join(timeout=3)
            for file in opened: file.close()


if __name__ == '__main__':
    main()
