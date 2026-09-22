#!/usr/bin/env python3
"""Check prepared Lilith configs and briefly start/stop its isolated service stack.

No sessions, model jobs, VM allocations or benchmark evaluations are submitted.
"""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--bin-dir', type=pathlib.Path, default=pathlib.Path('target/release'))
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    services = ['akhexecd', 'akhsbxd', 'akhrtd', 'akhsessd']
    configs = {name: json.loads((args.root / f'{name}.json').read_text()) for name in services}
    for name, config in configs.items():
        host, port = config['listen'].rsplit(':', 1)
        if host != '127.0.0.1':
            parser.error('startup check only supports loopback configurations')
        with socket.socket() as probe:
            # Permit a recently stopped listener's TIME_WAIT sockets, but still
            # reject an active listener before starting project-owned processes.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, int(port)))
        subprocess.run([str((args.bin_dir / name).resolve()), '--config',
                        str((args.root / f'{name}.json').resolve()), '--check'], check=True, timeout=15)
    processes, logs = {}, {}
    result = {'kind': 'isolated_startup_not_model_or_native_validation', 'services': {}}
    env = os.environ.copy()
    env.update(AKH_GPU_MATMUL='0', AKH_GPU_QUANT='0')
    try:
        for name in services:
            logs[name] = tempfile.TemporaryFile(mode='w+')
            processes[name] = subprocess.Popen([str((args.bin_dir / name).resolve()), '--config',
                str((args.root / f'{name}.json').resolve())], env=env, stdout=logs[name], stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 15
            while True:
                if processes[name].poll() is not None:
                    raise RuntimeError(f'{name} exited {processes[name].returncode}')
                try:
                    with urllib.request.urlopen('http://' + configs[name]['listen'] + '/healthz', timeout=1) as response:
                        health = json.load(response)
                    if health.get('service_id') != configs[name]['service_id']:
                        raise RuntimeError('health identity does not match this stack')
                    result['services'][name] = health
                    break
                except (OSError, ValueError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'{name} health timeout')
                    time.sleep(.2)
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=str(error))
    finally:
        for process in reversed(list(processes.values())):
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
        result['services_stopped'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
