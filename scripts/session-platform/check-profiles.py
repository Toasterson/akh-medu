#!/usr/bin/env python3
"""Check independent build graphs without building a combined daemon artifact."""
import os
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
env = dict(os.environ, CARGO_BUILD_JOBS='2', CARGO_PROFILE_DEV_DEBUG='0',
           CARGO_PROFILE_TEST_DEBUG='0', CARGO_INCREMENTAL='0')
profiles = [('session', 'akhsessd'), ('runtime', 'akhrtd'), ('executor', 'akhexecd'),
            ('sandbox', 'akhsbxd'), ('events', 'akheventd'), ('client', 'akh-connect'),
            ('client', 'akh-acp'), ('helpers', 'akh-service'), ('runtime', 'akh-runtime-profile'), ('admin', 'akh-fingerprint')]
for feature, binary in profiles:
    common = ['-p', 'akh-session-services', '--no-default-features', '--features', feature]
    subprocess.run(['cargo', 'check', *common, '--bin', binary], cwd=ROOT, env=env, check=True)
    graph = subprocess.check_output(['cargo', 'tree', *common, '--edges', 'normal', '--prefix', 'none'],
                                    cwd=ROOT, env=env, text=True)
    packages = {line.split()[0] for line in graph.splitlines()}
    assert ('akh-runtime' in packages) == (feature == 'runtime'), (binary, 'provider dependency')
    assert ('lapin' in packages) == (feature == 'events'), (binary, 'RabbitMQ dependency')
    assert 'akh-medu' not in packages, (binary, 'cognitive engine dependency')
    print(f'{binary}: isolated profile checked', flush=True)
