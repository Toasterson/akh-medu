#!/usr/bin/env python3
"""Use authenticated persistent native builders; retain local image smoke tests."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

SCRIPTS = Path(__file__).resolve().parent
BUILDERS = {
    'amd64': ('95.216.66.53', 'root', 'AKH_BUILDKIT_AMD64_SSH_KEY'),
    'arm64': ('192.168.1.90', 'toasty', 'AKH_BUILDKIT_ARM64_SSH_KEY'),
}


def prepare_ssh(directory, environment):
    """Write private ephemeral delivery files and return a secret-free child env."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    clean = dict(environment)
    config = []
    for arch, (host, user, variable) in BUILDERS.items():
        key = clean.pop(variable, '')
        if not key.startswith('-----BEGIN OPENSSH PRIVATE KEY-----'):
            raise RuntimeError('Missing native builder credential: ' + variable)
        key_path = directory / arch
        key_path.write_text(key.rstrip() + '\n')
        key_path.chmod(0o600)
        config.extend([
            'Host akh-ci-' + arch,
            '  HostName ' + host,
            '  User ' + user,
            '  HostKeyAlias akh-ci-' + arch,
            '  IdentityFile ' + str(key_path),
            '  IdentityAgent none', '  IdentitiesOnly yes', '  BatchMode yes',
            '  StrictHostKeyChecking yes',
            '  UserKnownHostsFile ' + str(SCRIPTS / 'ci-builders-known-hosts'),
            '  GlobalKnownHostsFile /dev/null',
            '  ConnectTimeout 15', '  ServerAliveInterval 15', '  ServerAliveCountMax 3',
        ])
    config_path = directory / 'config'
    config_path.write_text('\n'.join(config) + '\n')
    config_path.chmod(0o600)
    executable = directory / 'bin' / 'ssh'
    executable.parent.mkdir(mode=0o700)
    executable.write_text('#!/bin/sh\nexec /usr/bin/ssh -F ' + shlex.quote(str(config_path)) + ' "$@"\n')
    executable.chmod(0o700)
    clean['PATH'] = str(executable.parent) + os.pathsep + clean.get('PATH', '/usr/bin:/bin')
    return clean


def build(group):
    source_environment = dict(os.environ)
    for _, _, variable in BUILDERS.values():
        os.environ.pop(variable, None)
    branch = subprocess.check_output(['sh', str(SCRIPTS / 'resolve-branch.sh')], text=True).strip()
    if branch != 'main':
        # PRs still run the pipeline's existing non-publishing validation path.
        subprocess.run([sys.executable, str(SCRIPTS / 'build-platform-images.py'), group], check=True)
        return
    with tempfile.TemporaryDirectory(prefix='akh-ci-builders-') as directory:
        environment = prepare_ssh(directory, source_environment)
        builder = 'akh-native-' + str(os.getpid())
        def run(*args, **kwargs):
            return subprocess.run(args, check=True, env=environment, **kwargs)
        try:
            run('sh', str(SCRIPTS / 'setup-platform-buildx.sh'))
            run('docker', 'buildx', 'create', '--name', builder, '--driver', 'remote',
                '--node', 'amd64', '--platform', 'linux/amd64', 'ssh://akh-ci-amd64')
            run('docker', 'buildx', 'create', '--name', builder, '--append',
                '--node', 'arm64', '--platform', 'linux/arm64', 'ssh://akh-ci-arm64')
            run('docker', 'buildx', 'use', builder)
            run('docker', 'buildx', 'inspect', '--bootstrap')
            run(sys.executable, str(SCRIPTS / 'build-platform-images.py'), group)
        finally:
            # Remote-driver removal removes only this client's registration.
            # Persistent daemons and their caches remain available to other jobs.
            subprocess.run(['docker', 'buildx', 'rm', builder], env=environment,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('group', choices=['home', 'core', 'daemons', 'fleet', 'collect'])
    try:
        build(parser.parse_args().group)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
