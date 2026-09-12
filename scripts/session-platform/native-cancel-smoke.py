#!/usr/bin/env python3
"""Measure Linux provider teardown against a stalled local model peer; no login."""
import argparse
import http.server
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import uuid


def descendants(pid):
    pairs = [tuple(map(int, line.split())) for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True).splitlines()]
    found = {pid}
    while True:
        added = {child for child, parent in pairs if parent in found} - found
        if not added:
            return found - {pid}
        found |= added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=['codex', 'claude'])
    args = parser.parse_args()
    started, release = threading.Event(), threading.Event()

    class Peer(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            started.set()
            release.wait(25)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    child = None
    try:
        with tempfile.TemporaryDirectory(prefix='akh-native-cancel-') as directory:
            root = Path(directory)
            work = root / 'work'; work.mkdir()
            profile = {'revision': 'cancel-1', 'kind': args.kind, 'binary': '/usr/local/bin/' + args.kind,
                'working_dir': str(work), 'execution_mode': 'brokered', 'timeout_secs': 40,
                'isolation': {'directory': str(root / 'profile')}}
            base = f'http://127.0.0.1:{server.server_port}'
            if args.kind == 'codex':
                profile['isolation']['managed_files'] = {'codex/config.toml':
                    'model_provider = "fixture"\nmodel = "gpt-5.4"\n'
                    '[model_providers.fixture]\nname = "Local cancellation measurement"\n'
                    f'base_url = "{base}/v1"\nwire_api = "responses"\nrequires_openai_auth = false\n'
                    'supports_websockets = false\nrequest_max_retries = 0\nstream_max_retries = 0\n'}
            else:
                profile.update(node_binary='/usr/local/bin/node', bridge_script='/opt/akh/scripts/agent-backends/claude.mjs',
                    environment={'ANTHROPIC_BASE_URL': base, 'ANTHROPIC_API_KEY': 'owned-local-cancel-fixture',
                        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', 'DISABLE_PROMPT_CACHING': '1', 'ENABLE_TOOL_SEARCH': 'false'})
            session, turn, workspace = (str(uuid.uuid4()) for _ in range(3))
            request = {'id': turn, 'session_id': session, 'turn_id': turn, 'generation': 1,
                'workspace': {'id': workspace, 'sandbox_id': workspace, 'executor': 'fixture', 'generation': 1,
                    'site': 'measurement', 'root': '/not-mounted', 'platform': 'linux', 'confinement': 'measurement'},
                'binding': {'runtime': 'fixture', 'profile': 'fixture', 'revision': 'cancel-1', 'kind': args.kind,
                    'model': 'gpt-5.4' if args.kind == 'codex' else 'claude-sonnet-4-6', 'execution_mode': 'brokered'},
                'input': {'origin': {'kind': 'human', 'principal': 'fixture'}, 'text': 'Wait for the local peer.'},
                'instructions': 'No tools.', 'tools': [], 'approval_policy': 'allow_granted'}
            path = root / 'job.json'; path.write_text(json.dumps({'profile': profile, 'request': request}))
            child = subprocess.Popen(['/usr/local/bin/examples/native-provider-probe', str(path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if not started.wait(20):
                child.terminate()
                out, err = child.communicate(timeout=10)
                raise RuntimeError('provider did not reach local peer: ' + err[-1000:] + out[-1000:])
            owned = descendants(child.pid)
            assert owned, 'no native provider child observed'
            begin = time.monotonic(); child.terminate()
            child.communicate(timeout=10)
            elapsed = time.monotonic() - begin
            for pid in owned:
                stat = Path(f'/proc/{pid}/stat')
                assert not stat.exists(), f'provider descendant survived cancellation: {pid}'
            print(json.dumps({'result': 'passed', 'provider': args.kind, 'platform': os.uname().sysname + '-' + os.uname().machine,
                'cancel_teardown': True, 'observed_descendants': len(owned), 'seconds': elapsed}))
    finally:
        release.set()
        if child and child.poll() is None:
            child.kill(); child.wait()
        server.shutdown(); server.server_close(); thread.join(timeout=3)


if __name__ == '__main__':
    main()
