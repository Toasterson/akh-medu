#!/usr/bin/env python3
"""Measure the exact native telemetry image with offline Linux provider peers.

The container has no network except loopback, credentials, or host mounts.
Failed containers and captured output remain available for investigation.
"""
import argparse
from pathlib import Path
import subprocess
import uuid

RUNNER = r'''
import hashlib, json, pathlib, subprocess
root = pathlib.Path('/tmp/akh-native-observability-measurements')
root.mkdir()
scripts = '/opt/akh/scripts/session-platform/'
commands = {
    'codex-wire': ['python3', scripts+'codex-native-wire-smoke.py', '--codex', '/usr/local/bin/codex', '--bin-dir', '/usr/local/bin'],
    'claude-wire': ['python3', scripts+'claude-native-wire-smoke.py', '--claude', '/usr/local/bin/claude', '--node', '/usr/local/bin/node', '--bin-dir', '/usr/local/bin'],
    'codex-cancel': ['python3', scripts+'native-cancel-smoke.py', 'codex'],
    'claude-cancel': ['python3', scripts+'native-cancel-smoke.py', 'claude'],
}
artifacts = {}
for path in ['/usr/local/bin/akhrtd', '/usr/local/bin/akhexecd', '/usr/local/bin/examples/native-provider-probe']:
    with open(path, 'rb') as stream:
        artifacts[path] = hashlib.file_digest(stream, 'sha256').hexdigest()
receipts = {kind: {'runtime_artifact_sha256': artifacts} for kind in ['codex', 'claude']}
for name, command in commands.items():
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (root/(name+'.stdout')).write_text(result.stdout)
    (root/(name+'.stderr')).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(name+' failed: '+result.stderr[-2000:])
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt['result'] == 'passed' and receipt['platform'] == 'Linux-aarch64', receipt
    kind, check = name.split('-')
    receipts[kind][check] = receipt
    (root/'receipts.partial.json').write_text(json.dumps(receipts, indent=2)+'\n')
    print(name+': passed', flush=True)
(root/'receipts.json').write_text(json.dumps(receipts, indent=2)+'\n')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    name = 'akh-otel-measure-' + uuid.uuid4().hex[:12]
    (args.output/'container.txt').write_text(name+'\n')
    (args.output/'image.txt').write_text(args.image+'\n')
    result = subprocess.run(['docker', 'run', '--name', name, '--network', 'none',
                             args.image, 'python3', '-c', RUNNER])
    subprocess.run(['docker', 'cp', name+':/tmp/akh-native-observability-measurements/.',
                    str(args.output)], check=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    print('Measured receipts: ' + str(args.output/'receipts.json'))


if __name__ == '__main__':
    main()
