#!/usr/bin/env python3
"""Publish one verified main image set; no deployment uses partial sets.

Each bounded group shares a BuildKit builder; the collector rejects partial sets. Registry auto-auth comes from the CI
runner. The final akh-image-set image contains /image-set.json and is only
published after every architecture/binary/profile smoke check succeeds.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

REGISTRY = os.environ.get('AKH_IMAGE_REGISTRY', 'code.aopc.cloud/toasterson')
ARCHES = ['amd64', 'arm64']
HOMELAB_SHA = 'f17aa7a76b27bcba198cf2afbd34e65c82e95a11'
DAEMONS = {'sessions': ('akhsessd', 'session'), 'runtime': ('akhrtd', 'runtime'),
           'executor': ('akhexecd', 'executor'), 'sandbox': ('akhsbxd', 'sandbox'),
           'events': ('akheventd', 'events')}


GROUPS = {
    'home': {'home-runtime', 'home-services', 'home-control'},
    'core': {'core-service', 'akh'},
    'daemons': set(DAEMONS) | {'trigger', 'client', 'connect', 'kagent'},
    'fleet': {'workspace', 'relay', 'seshd', 'sokard', 'akh-transcripts'},
}
EXPECTED = set.union(*GROUPS.values())
AMD64_ONLY = {'workspace', 'seshd', 'sokard', 'akh-transcripts'}
REQUIRED_BINARIES = {
    'home-runtime': {'akhrtd', 'akhsessd', 'akheventd', 'akhexecd', 'akh-service', 'akh-runtime-profile'},
    'home-services': {'akhrtd', 'akhsessd', 'akheventd', 'akhexecd', 'akh-service'},
    'home-control': {'akhsessd', 'akheventd'}, 'core-service': {'akhomed'}, 'akh': {'akh'},
    'sessions': {'akhsessd'}, 'runtime': {'akhrtd', 'akh-service', 'akh-runtime-profile'},
    'executor': {'akhexecd', 'akh-service'}, 'sandbox': {'akhsbxd'}, 'events': {'akheventd'},
    'trigger': {'akh-trigger'}, 'client': {'akh-acp'}, 'connect': {'akh-connect'}, 'kagent': {'akh-kagent'},
    'workspace': {'akhexecd', 'akh-service'}, 'seshd': {'seshd'}, 'sokard': {'sokard'},
    'akh-transcripts': {'akh-harvest', 'akh-derive', 'akh-transcripts-mcp'},
}
HEAP_BINARIES = {name: binary for name, (binary, _) in DAEMONS.items()} | {
    'home-runtime': 'akhrtd', 'home-services': 'akhsessd', 'home-control': 'akheventd'}


def publish_manifest(evidence, name, sha):
    with tempfile.TemporaryDirectory(prefix='akh-image-set-') as directory:
        output = Path(directory)
        (output / 'image-set.json').write_text(json.dumps(evidence, indent=2) + '\n')
        (output / 'Dockerfile').write_text('FROM scratch\nCOPY image-set.json /image-set.json\n')
        run('docker', 'buildx', 'build', '--platform', 'linux/amd64', '-t',
            REGISTRY + '/' + name + ':main-' + sha,
            '--label', 'org.opencontainers.image.revision=' + sha, '--push', directory)


def read_manifest(image):
    run('docker', 'pull', '--platform', 'linux/amd64', image)
    container = run('docker', 'create', '--platform', 'linux/amd64', image, '/never-run', capture=True)
    try:
        with tempfile.TemporaryDirectory(prefix='akh-partial-manifest-') as directory:
            path = Path(directory) / 'image-set.json'
            run('docker', 'cp', container + ':/image-set.json', str(path))
            return json.loads(path.read_text())
    finally:
        run('docker', 'rm', container)


def read_partial(group, sha):
    return read_manifest(REGISTRY + '/akh-image-set-' + group + ':main-' + sha)


def released_set():
    """Only a missing marker means first release; registry errors fail closed."""
    marker = REGISTRY + '/akh-image-set:main-ready'
    result = subprocess.run(['docker', 'buildx', 'imagetools', 'inspect', '--raw', marker],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        error = result.stderr.lower()
        if 'manifest unknown' in error or 'not found' in error:
            return None
        raise RuntimeError('cannot inspect verified release marker: ' + result.stderr.strip())
    evidence = read_manifest(marker)
    if (evidence.get('complete') is not True
            or evidence.get('main_ancestry_verified') is not True
            or len(evidence.get('source_commit', '')) != 40):
        raise RuntimeError('invalid verified release marker')
    return evidence


def is_ancestor(older, newer):
    result = subprocess.run(['git', 'merge-base', '--is-ancestor', older, newer])
    if result.returncode not in (0, 1):
        raise RuntimeError('cannot establish release commit ancestry')
    return result.returncode == 0


def assemble_partials(partials, sha):
    """Require every group and proof before a complete artifact can be published."""
    if set(partials) != set(GROUPS):
        raise RuntimeError('complete set is missing image groups')
    result = {'schema_version': 1, 'source_commit': sha, 'main_ancestry_verified': True,
              'complete': True, 'images': {}, 'native_conformance': {},
              'groups': {}, 'homelab_source_commit': HOMELAB_SHA}
    for group, partial in partials.items():
        if partial.get('source_commit') != sha or partial.get('main_ancestry_verified') is not True:
            raise RuntimeError('partial source or ancestry mismatch: ' + group)
        if partial.get('group') != group or partial.get('complete') is not False:
            raise RuntimeError('invalid partial identity: ' + group)
        if set(partial.get('images', {})) != GROUPS[group]:
            raise RuntimeError('partial image membership mismatch: ' + group)
        result['groups'][group] = {'artifact': REGISTRY + '/akh-image-set-' + group + ':main-' + sha,
                                   'workflow_run': partial.get('workflow_run')}
        for name, image in partial['images'].items():
            arches = ['amd64'] if name in AMD64_ONLY else ARCHES
            if image.get('source_commit') != sha or set(image.get('platform_digests', {})) != set(arches):
                raise RuntimeError('image source/platform mismatch: ' + name)
            if image.get('tag', '').split(':')[-1] != 'main-' + sha:
                raise RuntimeError('image tag source mismatch: ' + name)
            if not image.get('index_digest', '').startswith('sha256:'):
                raise RuntimeError('image index is missing: ' + name)
            if set(image.get('smoke', {})) != set(arches):
                raise RuntimeError('architecture smoke proof missing: ' + name)
            for arch in arches:
                smoke = image['smoke'][arch]
                if name == 'relay':
                    if not smoke.get('python') or not smoke.get('ssh'):
                        raise RuntimeError('relay smoke proof missing')
                elif set(smoke.get('binaries', [])) != REQUIRED_BINARIES[name]:
                    raise RuntimeError('binary smoke proof missing: ' + name)
                if name in HEAP_BINARIES and not any(
                    check.get('binary') == HEAP_BINARIES[name] and check.get('profile_samples', 0) > 0 and check.get('application_symbols') is True
                    for check in smoke.get('heap', [])):
                    raise RuntimeError('heap smoke proof missing: ' + name)
            result['images'][name] = image
        if group == 'home':
            result['native_conformance'] = partial.get('native_conformance', {})
            result['provider_versions'] = partial.get('provider_versions', {})
        if group == 'fleet' and partial.get('homelab_source_commit') != HOMELAB_SHA:
            raise RuntimeError('fleet support source mismatch')
    for arch in ARCHES:
        for provider in ('codex', 'claude'):
            receipt = result['native_conformance'].get(arch, {}).get(provider, {})
            if receipt.get('wire', {}).get('result') != 'passed' or receipt.get('cancel', {}).get('result') != 'passed':
                raise RuntimeError('native conformance proof missing: ' + arch + '/' + provider)
            hashes = receipt.get('runtime_artifact_sha256', {})
            paths = {'/usr/local/bin/akhrtd', '/usr/local/bin/akhexecd', '/usr/local/bin/examples/native-provider-probe'}
            if set(hashes) != paths or any(len(value) != 64 for value in hashes.values()):
                raise RuntimeError('native artifact hashes missing: ' + arch + '/' + provider)
    return result


def run(*args, capture=False):
    result = subprocess.run(args, check=True, text=True,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else None


def bundle_native_receipts(image, record, receipts):
    """Ship measured receipts with the exact binaries that produced them."""
    with tempfile.TemporaryDirectory(prefix='akh-native-receipts-') as directory:
        root = Path(directory)
        for arch in ARCHES:
            (root / (arch + '.json')).write_text(json.dumps(receipts[arch]) + '\n')
        (root / 'Dockerfile').write_text(
            'ARG BASE\nFROM ${BASE}\nARG TARGETARCH\n'
            'COPY ${TARGETARCH}.json /usr/local/share/akh/native-conformance.json\n')
        run('docker', 'buildx', 'build', '--platform', 'linux/amd64,linux/arm64',
            '--build-arg', 'BASE=' + record['image'], '--tag', image, '--push', directory)
    bundled = image_manifest(image, ARCHES)
    for arch in ARCHES:
        pinned = image.split(':main-')[0] + '@' + bundled['platform_digests'][arch]
        measured = receipts[arch]['codex']['runtime_artifact_sha256']
        # A packaging-only layer must not change any measured runtime artifact.
        actual = json.loads(run('docker', 'run', '--rm', '--platform', 'linux/' + arch,
            '--entrypoint', 'python3', pinned, '-c',
            'import hashlib,json,pathlib; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in '
            + repr(list(measured)) + '}))', capture=True))
        if actual != measured:
            raise RuntimeError('receipt packaging changed measured native binaries: ' + arch)
        embedded = json.loads(run('docker', 'run', '--rm', '--platform', 'linux/' + arch,
            '--entrypoint', 'cat', pinned, '/usr/local/share/akh/native-conformance.json', capture=True))
        if embedded != receipts[arch]:
            raise RuntimeError('embedded native receipts differ from CI measurements: ' + arch)
    return {**record, **bundled, 'native_receipts_embedded': True}


def promote_verified_set(evidence, sha):
    """Keel watches only this channel, never intermediate build tags."""
    if evidence.get('complete') is not True or evidence.get('source_commit') != sha:
        raise RuntimeError('only a complete commit-matched image set can be promoted')
    if not evidence['images']['home-runtime'].get('native_receipts_embedded'):
        raise RuntimeError('automatic native updates require embedded CI receipts')
    # The Forgejo main concurrency group serializes release workflows. Compare
    # against the last COMPLETE release, not the moving source branch: frequent
    # merges must not starve delivery, and an old manual rerun must not roll back.
    previous = released_set()
    if previous is not None and not is_ancestor(previous['source_commit'], sha):
        print('A newer or divergent release exists; retaining immutable artifacts only.', flush=True)
        return False
    for name, record in sorted(evidence['images'].items()):
        alias = record['image'].split('@')[0] + ':main-ready'
        run('docker', 'buildx', 'imagetools', 'create', '--tag', alias, record['image'])
        live = image_manifest(alias, list(record['platform_digests']))
        if live['platform_digests'] != record['platform_digests']:
            raise RuntimeError('promoted channel differs from verified image set: ' + name)
    # Publish the release pointer last. A failed partial promotion can be retried;
    # a later successful release then prevents older reruns from moving it back.
    marker = REGISTRY + '/akh-image-set:main-ready'
    run('docker', 'buildx', 'imagetools', 'create', '--tag', marker,
        REGISTRY + '/akh-image-set:main-' + sha)
    published = released_set()
    if published is None or published['source_commit'] != sha or published['images'] != evidence['images']:
        raise RuntimeError('published release marker differs from verified image set')
    return True


def sample_count(raw):
    """Count top-level protobuf Profile.sample messages without dependencies."""
    def varint(offset):
        value = shift = 0
        while True:
            b = raw[offset]
            offset += 1
            value |= (b & 127) << shift
            if b < 128:
                return value, offset
            shift += 7
            if shift > 63:
                raise ValueError('invalid protobuf varint')
    offset = samples = 0
    while offset < len(raw):
        key, offset = varint(offset)
        wire = key & 7
        if wire == 0:
            _, offset = varint(offset)
        elif wire == 1:
            offset += 8
        elif wire == 2:
            size, offset = varint(offset)
            if key >> 3 == 2 and size:
                samples += 1
            offset += size
        elif wire == 5:
            offset += 4
        else:
            raise ValueError('unsupported protobuf wire type')
        if offset > len(raw):
            raise ValueError('truncated protobuf profile')
    return samples


def smoke_binary(image, arch, binary):
    # Never swallow a missing executable or architecture error as stub evidence.
    run('docker', 'run', '--rm', '--platform', 'linux/' + arch,
        '--entrypoint', '/bin/sh', image, '-ec',
        f'test -s /usr/local/bin/{binary} && test -x /usr/local/bin/{binary}')
    result = subprocess.run(['docker', 'run', '--rm', '--platform', 'linux/' + arch,
                             '--entrypoint', binary, image, '--help'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f'{image} {arch} {binary}: --help failed or silent')


def smoke_heap(image, arch, binary):
    with tempfile.TemporaryDirectory(prefix='akh-heap-smoke-') as directory:
        root = Path(directory)
        root.chmod(0o755)
        (root / 'token').write_text('ci-synthetic-token-for-isolated-heap-smoke')
        (root / 'token').chmod(0o644)
        config = {'service_id': 'ci-' + binary, 'listen': '0.0.0.0:8080',
                  'profiling_listen': '0.0.0.0:8081', 'data_dir': '/data',
                  'workspace_root': '/data/workspaces',
                  'principals': [{'id': 'ci', 'token_file': '/smoke/token',
                                  'roles': ['admin'], 'akhs': ['ci']}]}
        (root / 'config.json').write_text(json.dumps(config))
        (root / 'config.json').chmod(0o644)
        container = run('docker', 'run', '-d', '--platform', 'linux/' + arch,
                        '--tmpfs', '/data:uid=10001,gid=10001,mode=0755',
                        '-v', directory + ':/smoke:ro', '-p', '127.0.0.1::8081',
                        '-e', 'MALLOC_CONF=prof:true,prof_active:true,lg_prof_sample:10',
                        '--entrypoint', binary, image, '--config', '/smoke/config.json',
                        capture=True)
        try:
            port = run('docker', 'port', container, '8081/tcp', capture=True).split(':')[-1]
            error = None
            for _ in range(90):
                try:
                    with urllib.request.urlopen('http://127.0.0.1:' + port + '/debug/pprof/heap', timeout=5) as response:
                        profile = response.read()
                        if response.status != 200 or len(profile) < 128:
                            raise ValueError('heap endpoint returned an empty/error profile')
                        raw = gzip.decompress(profile)
                        count = sample_count(raw)
                        if count < 1:
                            raise ValueError('heap profile contains no samples')
                        if b'akh_session_services' not in raw:
                            raise ValueError('heap profile lacks symbolized application frames')
                        print(f'heap smoke {binary} linux/{arch}: {count} samples', flush=True)
                        return {'binary': binary, 'profile_samples': count, 'application_symbols': True}
                except Exception as exc:
                    error = exc
                    if run('docker', 'inspect', '-f', '{{.State.Running}}', container, capture=True) != 'true':
                        break
                    time.sleep(2)
            run('docker', 'logs', container)
            raise RuntimeError(f'{binary} {arch} heap smoke failed: {error}')
        finally:
            run('docker', 'rm', '-f', container)


def image_manifest(image, arches):
    raw = json.loads(run('docker', 'buildx', 'imagetools', 'inspect', '--raw', image, capture=True))
    platform_digests = {}
    for manifest in raw.get('manifests', []):
        platform = manifest.get('platform', {})
        if platform.get('os') == 'linux' and platform.get('architecture') in arches:
            platform_digests[platform['architecture']] = manifest['digest']
    if set(platform_digests) != set(arches):
        raise RuntimeError(f'{image}: missing requested platform manifests: {platform_digests}')
    # Human inspect returns the registry's canonical index digest (not a JSON rehash).
    info = run('docker', 'buildx', 'imagetools', 'inspect', image, capture=True)
    digest = next(line.split(':', 1)[1].strip() for line in info.splitlines() if line.startswith('Digest:'))
    return {'image': image.split(':main-')[0] + '@' + digest,
            'tag': image, 'index_digest': digest, 'platform_digests': platform_digests}


def native_conformance(image, arch):
    artifacts = ['/usr/local/bin/akhrtd', '/usr/local/bin/akhexecd', '/usr/local/bin/examples/native-provider-probe']
    hashes = json.loads(run('docker', 'run', '--rm', '--platform', 'linux/' + arch,
        '--entrypoint', 'python3', image, '-c',
        'import hashlib,json,pathlib; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in ' + repr(artifacts) + '}))', capture=True))
    receipts = {}
    for provider in ('codex', 'claude'):
        command = ['python3', '/opt/akh/scripts/session-platform/' + provider + '-native-wire-smoke.py',
                   '--' + provider, '/usr/local/bin/' + provider, '--bin-dir', '/usr/local/bin']
        if provider == 'claude':
            command += ['--node', '/usr/local/bin/node']
        wire = json.loads(run('docker', 'run', '--rm', '--init', '--platform', 'linux/' + arch,
                              '--entrypoint', command[0], image, *command[1:], capture=True))
        cancel = json.loads(run('docker', 'run', '--rm', '--init', '--platform', 'linux/' + arch,
            '--entrypoint', 'python3', image, '/opt/akh/scripts/session-platform/native-cancel-smoke.py', provider, capture=True))
        if wire.get('result') != 'passed' or cancel.get('result') != 'passed':
            raise RuntimeError('native provider conformance failed: ' + provider)
        receipts[provider] = {'runtime_artifact_sha256': hashes, 'wire': wire, 'cancel': cancel}
    return receipts


def workspace_and_relay(evidence, tag, sha):
    # Vendored nonsecret build inputs avoid a new private-repository CI credential.
    context = Path('docker/fleet-support')
    provenance = json.loads((context / 'provenance.json').read_text())
    source_commit = provenance['homelab_source_commit']
    if source_commit != HOMELAB_SHA:
        raise RuntimeError('fleet build inputs differ from the reviewed homelab source')
    import hashlib
    for relative, expected in provenance['files_sha256'].items():
        path = context / relative
        if not path.resolve().is_relative_to(context.resolve()):
            raise RuntimeError('fleet provenance path escapes build context')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('fleet input checksum mismatch: ' + relative)
    inputs = json.loads((context / 'workspace-build-inputs.json').read_text())
    inputs['EXECUTOR_IMAGE'] = evidence['images']['executor']['image']
    for name, arches in [('workspace', ['amd64']), ('relay', ARCHES)]:
        image = REGISTRY + '/akh-' + name + ':' + tag
        command = ['docker', 'buildx', 'build', '--platform', ','.join('linux/' + arch for arch in arches),
                   '-f', str(context / (name + '.Dockerfile')), '-t', image,
                   '--label', 'org.opencontainers.image.revision=' + sha,
                   '--label', 'it.wegmueller.homelab.revision=' + HOMELAB_SHA, '--push']
        if name == 'workspace':
            command += ['--target', 'workspace']
            for key, value in inputs.items():
                if '@sha256:' not in value:
                    raise RuntimeError('workspace input is not immutable: ' + key)
                command += ['--build-arg', key + '=' + value]
        run(*command, str(context))
        record = image_manifest(image, arches)
        record['homelab_source_commit'] = HOMELAB_SHA
        record['smoke'] = {}
        for arch in arches:
            if name == 'workspace':
                smoke_binary(image, arch, 'akhexecd')
                smoke_binary(image, arch, 'akh-service')
                run('docker', 'run', '--rm', '--platform', 'linux/' + arch, '--entrypoint', '/bin/sh', image,
                    '-ec', 'git --no-lazy-fetch --version; node --version; test -s /usr/local/share/akh/workspace-identity.json')
                record['smoke'][arch] = {'binaries': ['akhexecd', 'akh-service'], 'workspace_identity': True}
            else:
                run('docker', 'run', '--rm', '--platform', 'linux/' + arch, '--entrypoint', 'python3', image, '--version')
                run('docker', 'run', '--rm', '--platform', 'linux/' + arch, '--entrypoint', 'ssh', image, '-V')
                record['smoke'][arch] = {'python': True, 'ssh': True}
        if name == 'workspace':
            record['inputs'] = inputs
        evidence['images'][name] = record
    evidence['homelab_source_commit'] = HOMELAB_SHA


def build_set(group):
    sha = run('git', 'rev-parse', 'HEAD', capture=True)
    requested = os.environ.get('SOLSTICE_COMMIT_SHA', os.environ.get('GITHUB_SHA', sha))
    if requested != sha:
        raise RuntimeError('CI request SHA differs from checkout')
    if run('git', 'status', '--porcelain', '--untracked-files=no', capture=True):
        raise RuntimeError('deploy image checkout is dirty')
    shallow = run('git', 'rev-parse', '--is-shallow-repository', capture=True) == 'true'
    run('git', 'fetch', *(['--unshallow'] if shallow else []), 'origin',
        'main:refs/remotes/origin/main')
    run('git', 'merge-base', '--is-ancestor', sha, 'origin/main')
    tag = 'main-' + sha
    evidence = {'schema_version': 1, 'source_commit': sha, 'main_ancestry_verified': True,
                'workflow_run': os.environ.get('SOLSTICE_RUN_ID', os.environ.get('GITHUB_RUN_ID')),
                'provider_versions': {'codex': '0.153.4', 'claude': '2.1.269'}, 'images': {}, 'native_conformance': {},
                'group': group, 'complete': False}
    if group == 'collect':
        evidence = assemble_partials({name: read_partial(name, sha) for name in GROUPS}, sha)
        # Re-read immutable registry manifests; stale or overwritten tags cannot pass.
        for name, image in evidence['images'].items():
            arches = ['amd64'] if name in AMD64_ONLY else ARCHES
            live = image_manifest(image['tag'], arches)
            if live['index_digest'] != image['index_digest'] or live['platform_digests'] != image['platform_digests']:
                raise RuntimeError('registry digest changed since group smoke: ' + name)
        publish_manifest(evidence, 'akh-image-set', sha)
        evidence['channel_promoted'] = promote_verified_set(evidence, sha)
        print('AKH_NATIVE_CONFORMANCE_JSON=' + json.dumps(evidence['native_conformance']['arm64'], separators=(',', ':')), flush=True)
        print('AKH_IMAGE_SET_MANIFEST_JSON=' + json.dumps(evidence, separators=(',', ':')), flush=True)
        return
    # name, Dockerfile, target, build args, required binaries, profiled daemons
    recipes = [
        ('home-runtime', 'home-runtime', 'runtime', {}, ['akhrtd', 'akhsessd', 'akheventd', 'akhexecd', 'akh-service', 'akh-runtime-profile'], ['akhrtd']),
        ('home-services', 'home-runtime', 'services', {}, ['akhrtd', 'akhsessd', 'akheventd', 'akhexecd', 'akh-service'], ['akhsessd']),
        ('home-control', 'home-runtime', 'control', {}, ['akhsessd', 'akheventd'], ['akheventd']),
        ('core-service', 'core-service', None, {}, ['akhomed'], []),
        ('akh', 'akh', None, {}, ['akh'], []),
    ]
    for name, (binary, profile) in DAEMONS.items():
        binaries = sorted(REQUIRED_BINARIES[name])
        recipes.append((name, 'session-platform', None, {'SERVICE': binary, 'PROFILE': profile}, binaries, [binary]))
    for name, binary, profile in [('trigger', 'akh-trigger', 'trigger'), ('client', 'akh-acp', 'client'), ('connect', 'akh-connect', 'client'), ('kagent', 'akh-kagent', 'kagent')]:
        recipes.append((name, 'session-platform', None, {'SERVICE': binary, 'PROFILE': profile}, [binary], []))
    for name, dockerfile, target, args, binaries, heaps in recipes:
        if name not in GROUPS[group]:
            continue
        image = REGISTRY + '/akh-' + name + ':' + tag if name != 'akh' else REGISTRY + '/akh:' + tag
        command = ['docker', 'buildx', 'build', '--platform', 'linux/amd64,linux/arm64',
                   '--file', 'docker/' + dockerfile + '.Dockerfile', '--tag', image,
                   '--label', 'org.opencontainers.image.revision=' + sha,
                   '--label', 'org.opencontainers.image.source=https://code.aopc.cloud/toasterson/akh-medu', '--push']
        if target:
            command += ['--target', target]
        for key, value in args.items():
            command += ['--build-arg', key + '=' + value]
        run(*command, '.')
        record = image_manifest(image, ARCHES)
        record['smoke'] = {}
        for arch in ARCHES:
            pinned = image.split(':main-')[0] + '@' + record['platform_digests'][arch]
            for binary in binaries:
                smoke_binary(pinned, arch, binary)
            record['smoke'][arch] = {'binaries': binaries, 'heap': [smoke_heap(pinned, arch, binary) for binary in heaps]}
            if name == 'home-runtime':
                run('docker', 'run', '--rm', '--platform', 'linux/' + arch, '--entrypoint', '/bin/sh', pinned,
                    '-ec', 'codex --version | grep -F 0.153.4; claude --version | grep -F 2.1.269')
                evidence['native_conformance'][arch] = native_conformance(pinned, arch)
        if name == 'home-runtime':
            record = bundle_native_receipts(image, record, evidence['native_conformance'])
        evidence['images'][name] = record
    if group == 'fleet':
        daemon_partial = read_partial('daemons', sha)
        if daemon_partial.get('source_commit') != sha or daemon_partial.get('group') != 'daemons':
            raise RuntimeError('executor partial source mismatch')
        evidence['images']['executor'] = daemon_partial['images']['executor']
        workspace_and_relay(evidence, tag, sha)
        del evidence['images']['executor']
    # Reuse this run's successful archive jobs instead of compiling them twice.
    # The dependency gate guarantees their canonical short-SHA tags were pushed.
    # Archive images remain AMD64 by contract; the set records this explicitly.
    for name, binaries in [('seshd', ['seshd']), ('sokard', ['sokard']), ('akh-transcripts', ['akh-harvest', 'akh-derive', 'akh-transcripts-mcp'])]:
        if group != 'fleet':
            continue
        image = REGISTRY + '/' + name + ':' + tag
        source_image = REGISTRY + '/' + name + ':' + sha[:12]
        run('docker', 'pull', '--platform', 'linux/amd64', source_image)
        architecture = run('docker', 'image', 'inspect', '--format', '{{.Architecture}}', source_image, capture=True)
        if architecture != 'amd64':
            raise RuntimeError('archive job did not publish its required AMD64 image: ' + name)
        run('docker', 'buildx', 'imagetools', 'create', '--prefer-index=true', '--tag', image, source_image)
        record = image_manifest(image, ['amd64'])
        record['source_job'] = 'docker-' + name
        record['source_tag'] = source_image
        record['source_commit'] = sha
        record['smoke'] = {'amd64': {'binaries': binaries, 'missing_config_nonempty': True}}
        for binary in binaries:
            # Env-configured archive daemons need not accept --help; binary existence
            # and nonempty missing-config output are their existing CI contract.
            run('docker', 'run', '--rm', '--platform', 'linux/amd64', '--entrypoint', '/bin/sh', image, '-ec',
                f'test -s /usr/local/bin/{binary} && test -x /usr/local/bin/{binary}')
            output = subprocess.run(['docker', 'run', '--rm', '--platform', 'linux/amd64', '--entrypoint', binary, image, '/nonexistent'],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
            if not output:
                raise RuntimeError(f'{binary}: silent archive stub')
        evidence['images'][name] = record
    for image in evidence['images'].values():
        image['source_commit'] = sha
    if set(evidence['images']) != GROUPS[group]:
        raise RuntimeError('build group did not produce its exact image membership')
    publish_manifest(evidence, 'akh-image-set-' + group, sha)
    print('AKH_IMAGE_SET_PARTIAL_JSON=' + json.dumps(evidence, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', choices=[*GROUPS, 'collect'])
    args = parser.parse_args()
    branch = run('sh', '.forgejo/scripts/resolve-branch.sh', capture=True)
    if branch == 'main':
        build_set(args.group)
    else:
        print(f'Skipping deploy image publication for branch {branch!r}; main only.')
