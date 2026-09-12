#!/usr/bin/env python3
"""Installed Claude Agent SDK + synthetic Messages peer + real akhexecd.

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
    parser.add_argument('--claude', type=pathlib.Path, required=True)
    parser.add_argument('--bin-dir', type=pathlib.Path, default=ROOT / 'target/debug')
    parser.add_argument("--node", type=pathlib.Path, default=pathlib.Path("/opt/homebrew/bin/node"))
    args = parser.parse_args()
    assert args.claude.is_absolute()
    version = subprocess.run([str(args.claude), '--version'], check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory(prefix='akh-claude-native-wire-') as directory:
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
        # actual installed Claude. Native shell must never reach the host callback.
        actions = [
            ('Bash', {'command': 'touch native-must-not-exist'}),
            ('Write', {'file_path': str(runtime / 'native-patch-must-not-exist'), 'content': 'forbidden'}),
            ('mcp__akh__write_file', {'path': 'probe.txt', 'content': 'native wire receipt', 'create': True}),
            ('mcp__akh__read_file', {'path': 'probe.txt'}),
            ('mcp__akh__run_command', {'command': "printf 'executor-process' > process.txt", 'timeout_ms': 5000}),
            None,
            ('mcp__akh__read_file', {'path': 'probe.txt'}),
            ('mcp__akh__write_file', {'path': 'denied-probe.txt', 'content': 'must not write', 'create': True}),
            ('Bash', {'command': 'touch denied-probe.txt'}),
            ('Read', {'file_path': str(runtime / 'probe.txt')}),
            ('Edit', {'file_path': str(runtime / 'probe.txt'), 'old_string':'old','new_string':'forbidden'}),
            None,
        ]
        class Peer(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_POST(self):
                try:
                    body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    assert self.headers.get('x-api-key') in (None, 'owned-local-protocol-fixture')
                    assert self.headers.get('Authorization') in (None, 'Bearer owned-local-protocol-fixture')
                    if self.path.startswith('/v1/messages/count_tokens'):
                        data=json.dumps({'input_tokens':100}).encode()
                        self.send_response(200); self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
                    assert self.path.split('?')[0] == '/v1/messages', self.path
                    index=len(requests);requests.append(body)
                    assert index<len(actions), 'unexpected extra model request'
                    tools_seen.append(body.get('tools', []));action=actions[index]
                    usage={'input_tokens':100,'output_tokens':10,'cache_creation_input_tokens':0,'cache_read_input_tokens':0}
                    message={'id':f'msg_{index}','type':'message','role':'assistant','model':body['model'],'content':[], 'stop_reason':None,'stop_sequence':None,'usage':usage}
                    events=[{'type':'message_start','message':message}]
                    if action:
                        name, arguments=action
                        events.extend([
                            {'type':'content_block_start','index':0,'content_block':{'type':'tool_use','id':f'call_{index}','name':name,'input':{}}},
                            {'type':'content_block_delta','index':0,'delta':{'type':'input_json_delta','partial_json':json.dumps(arguments)}}])
                    else:
                        events.extend([
                            {'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}},
                            {'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'probe complete'}}])
                    events.extend([{'type':'content_block_stop','index':0},
                        {'type':'message_delta','delta':{'stop_reason':'tool_use' if action else 'end_turn','stop_sequence':None},'usage':usage},
                        {'type':'message_stop'}])
                    data=''.join('event: '+event['type']+'\ndata: '+json.dumps(event)+'\n\n' for event in events).encode()
                    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
                except Exception as error:
                    peer_errors.append(str(error));self.send_error(500)

        peer = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Peer)
        thread = threading.Thread(target=peer.serve_forever, daemon=True); thread.start()
        profile = {'revision': 'native-wire-1', 'kind': 'claude', 'binary': str(args.claude), 'node_binary': str(args.node),
                   'bridge_script': str(ROOT / 'scripts/agent-backends/claude.mjs'), 'working_dir': str(runtime),
                   'isolation': {'directory': str(root / 'profile')},
                   'environment': {'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{peer.server_port}',
                       'ANTHROPIC_API_KEY': 'owned-local-protocol-fixture',
                       'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'DISABLE_PROMPT_CACHING':'1', 'ENABLE_TOOL_SEARCH':'false'},
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
                       'binding': {'runtime': 'probe', 'profile': 'native-wire', 'revision': 'native-wire-1', 'kind': 'claude', 'model': 'claude-sonnet-4-6', 'execution_mode': 'brokered'},
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
            names = [tool['name'] for tool in tools_seen[0]]
            forbidden = {'Bash','Read','Write','Edit','Glob','Grep','Agent','Task','Skill','NotebookEdit','WebFetch','WebSearch'}
            for catalog in tools_seen:
                assert not any(tool['name'] in forbidden for tool in catalog), catalog
                assert {'mcp__akh__write_file','mcp__akh__read_file','mcp__akh__run_command'} <= {tool['name'] for tool in catalog}, catalog
            outputs = [block for body in requests for message in body.get('messages',[]) if isinstance(message.get('content'),list) for block in message['content'] if block.get('type')=='tool_result']
            native_calls = ['call_0','call_1','call_8','call_9','call_10']
            refusals = {call: next((item for item in outputs if item.get('tool_use_id') == call), None) for call in native_calls}
            assert all(value and value.get('is_error') for value in refusals.values()), refusals
            assert any('probe executor unavailable; nothing ran' in json.dumps(item) for item in outputs)
            print(json.dumps({'result': 'passed', 'claude_version': version, 'calls': calls, 'model_requests': len(requests),
                              'binary_sha256': hashlib.sha256(args.claude.read_bytes()).hexdigest(), 'platform': os.uname().sysname + '-' + os.uname().machine,
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
