#!/usr/bin/env python3
"""Exercise an installed, unmodified T3 0.0.40 through akh-service.

No package installation, provider account, model turn, or browser pairing. Private
state is removed and the helper owns upstream shutdown. Pairing tokens stay in
memory. Pass the absolute installed t3 executable with --t3.
"""
import argparse
import json
import os
import pathlib
import re
import socket
import subprocess
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
VERSION = '0.0.40'
DRIVERS = ['codex', 'claudeAgent', 'cursor', 'opencode', 'grok', 'antigravity']


def wait(condition, label):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        try:
            value = condition()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(.1)
    raise AssertionError(f'timed out: {label}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t3', type=pathlib.Path, required=True)
    parser.add_argument('--helper', type=pathlib.Path, default=ROOT / 'target/debug/akh-service')
    args = parser.parse_args()
    assert args.t3.is_absolute() and args.t3.is_file()
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    endpoint = f'http://127.0.0.1:{port}/'
    with tempfile.TemporaryDirectory(prefix='akh-t3-native-') as temp:
        root = pathlib.Path(temp).resolve()
        root.chmod(0o700)
        workspace = root / 'workspace'
        workspace.mkdir()
        (workspace / 'marker.txt').write_text('executor workspace\n')
        state = root / 'state'
        manifest = root / 'manifest.json'
        profile = {
            'id': 't3', 'revision': f't3-{VERSION}', 'upstream_version': VERSION,
            'version_command': [str(args.t3), '--version'],
            'version_output_contains': f't3 v{VERSION}',
            'command': [str(args.t3), 'serve', '--base-dir', '{config}/t3',
                        '--host', '127.0.0.1', '--port', str(port), '{workspace}'],
            'environment': {'PATH': os.environ['PATH']},
            # Editor-only service: avoid upstream probing local provider accounts.
            'managed_files': {'t3/userdata/settings.json': json.dumps({
                'providerInstances': {driver: {'driver': driver, 'enabled': False}
                                      for driver in DRIVERS}})},
            'health_url': endpoint, 'endpoint': endpoint, 'client_kind': 't3',
            'environment_id_file': '{config}/t3/userdata/environment-id',
            'pair_command': [str(args.t3), 'pair', '--base-dir', '{config}/t3'],
        }
        manifest.write_text(json.dumps({
            'schema_version': 1, 'id': 't3-native', 'session_id': 'native-session',
            'principal': 'native-user', 'state_dir': str(state),
            'workspace': {'id': 'workspace', 'sandbox_id': 'sandbox',
                          'executor': 'executor', 'generation': 1, 'site': 'test',
                          'root': str(workspace), 'platform': os.uname().sysname.lower(),
                          'confinement': 'temporary-process-test'}, 'profile': profile}))
        command = [str(args.helper), '--manifest', str(manifest)]

        def read_status():
            return json.loads((state / 'status.json').read_text())

        def run_once(previous=None):
            started = int(time.time() * 1000)
            process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                def ready():
                    assert process.poll() is None, 'helper exited before readiness'
                    status = read_status()
                    return status if status['state'] == 'ready' and status.get('access') and status['heartbeat'] >= started else None
                status = wait(ready, 'native T3 readiness and environment identity')
                access = status['access']
                assert access['environment_id']
                assert not access['central_agent_attachment']
                assert access['endpoint'] == endpoint
                if previous:
                    assert access['environment_id'] == previous, 'restart changed upstream identity'
                with urllib.request.urlopen(endpoint, timeout=5) as response:
                    assert response.status == 200
                result = subprocess.run(command + ['--action', 'pair'],
                                        capture_output=True, text=True, timeout=35)
                assert result.returncode == 0, 'native pairing action failed'
                output = json.loads(result.stdout)['output']
                links = re.findall(r'https?://[^\s\x1b]+', output)
                assert any(link.startswith(endpoint + 'pair#') for link in links), 'missing native pairing link'
                # Never persist or print the one-time pairing secret.
                assert 'pair#' not in (state / 'status.json').read_text()
                return access['environment_id']
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=12)
                assert read_status()['state'] == 'stopped', 'helper did not confirm child cleanup'

        environment = run_once()
        run_once(environment)
        assert (workspace / 'marker.txt').read_text() == 'executor workspace\n'
        print(f'PASS: upstream T3 {VERSION} version, private config, readiness, identity-preserving restart, native pairing link, helper shutdown')
        print('Not exercised: browser pairing, central agent attachment, native provider execution.')


if __name__ == '__main__':
    main()
