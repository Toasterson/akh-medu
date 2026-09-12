#!/usr/bin/env python3
"""Exercise akhsbxd + akhexecd against a disposable kind Sandbox controller.

Requires agent-sandbox v1.0.0 already installed and the executor image loaded into
that cluster. Creates/deletes only a uniquely named test namespace. Always uses
the explicit kubeconfig/context; never changes the default kubeconfig. No model,
editor, production cluster, or external account is used.
"""
import argparse
import base64
import hashlib
import json
import pathlib
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
HOLD = 'akh.test/hold-cleanup'


def port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def wait(condition, label, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            value = condition()
            if value:
                return value
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.25)
    raise AssertionError(f'timed out: {label}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kubeconfig', type=pathlib.Path, required=True)
    parser.add_argument('--context', required=True)
    parser.add_argument('--image', default='akh-session-platform-local:executor')
    parser.add_argument('--manager', type=pathlib.Path, default=ROOT / 'target/debug/akhsbxd')
    args = parser.parse_args()
    assert args.kubeconfig.is_absolute() and args.kubeconfig.is_file()
    assert args.context.startswith('kind-akh-platform-'), 'only an explicitly disposable Akh kind cluster is accepted'
    command = ['kubectl', '--kubeconfig', str(args.kubeconfig), '--context', args.context]
    namespace = 'akh-contract-' + secrets.token_hex(5)

    def kube(*argv, body=None, check=True):
        result = subprocess.run(command + list(argv), input=None if body is None else json.dumps(body),
                                text=True, capture_output=True, timeout=40)
        if check and result.returncode:
            raise AssertionError(f'kubectl {argv[0]} failed: {result.stderr[:500]}')
        return result

    kube('create', 'namespace', namespace)
    children, logs = [], []
    try:
        with tempfile.TemporaryDirectory(prefix='akh-k8s-contract-') as directory:
            root = pathlib.Path(directory).resolve()
            manager_port, executor_port = port(), port()
            session_token, manager_token = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            for name, token in [('session', session_token), ('manager', manager_token)]:
                path = root / f'{name}.token'; path.write_text(token); path.chmod(0o600)
            cluster = json.loads(kube('config', 'view', '--raw', '--minify', '-o', 'json').stdout)['clusters'][0]['cluster']
            assert cluster['server'].startswith('https://127.0.0.1:'), 'kind API must be local'
            ca = root / 'ca.pem'; ca.write_bytes(base64.b64decode(cluster['certificate-authority-data']))
            executor_config = {
                'service_id': 'executor', 'listen': '0.0.0.0:8080', 'data_dir': '/data',
                'workspace_root': '/workspaces', 'confinement': 'kind-container-runtime-default',
                'principals': [
                    {'id': 'sessions', 'token_file': '/credentials/session', 'token_sha256': True, 'roles': ['session']},
                    {'id': 'manager', 'token_file': '/credentials/manager', 'token_sha256': True, 'roles': ['sandbox']},
                ],
            }
            resources = [
                {'apiVersion': 'v1', 'kind': 'ServiceAccount', 'metadata': {'name': 'manager'}},
                {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'Role', 'metadata': {'name': 'manager'},
                 'rules': [{'apiGroups': ['agents.x-k8s.io'], 'resources': ['sandboxes'], 'verbs': ['get', 'create', 'delete']}]},
                {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'RoleBinding', 'metadata': {'name': 'manager'},
                 'roleRef': {'apiGroup': 'rbac.authorization.k8s.io', 'kind': 'Role', 'name': 'manager'},
                 'subjects': [{'kind': 'ServiceAccount', 'name': 'manager', 'namespace': namespace}]},
                {'apiVersion': 'v1', 'kind': 'ConfigMap', 'metadata': {'name': 'executor'},
                 'data': {'service.json': json.dumps(executor_config)}},
                {'apiVersion': 'v1', 'kind': 'Secret', 'metadata': {'name': 'executor-verifiers'},
                 'stringData': {'session': hashlib.sha256(session_token.encode()).hexdigest(),
                                'manager': hashlib.sha256(manager_token.encode()).hexdigest()}},
            ]
            for resource in resources:
                kube('-n', namespace, 'apply', '-f', '-', body=resource)
            api_token = root / 'api.token'
            api_token.write_text(kube('-n', namespace, 'create', 'token', 'manager', '--duration=1h').stdout.strip())
            api_token.chmod(0o600)
            template = {
                'apiVersion': 'agents.x-k8s.io/v1beta1', 'kind': 'Sandbox',
                'metadata': {'finalizers': [HOLD]},
                'spec': {'service': True, 'podTemplate': {'spec': {
                    'automountServiceAccountToken': False,
                    'securityContext': {'runAsNonRoot': True, 'runAsUser': 10001, 'runAsGroup': 10001, 'fsGroup': 10001},
                    'containers': [{'name': 'executor', 'image': args.image, 'imagePullPolicy': 'Never',
                                    'command': ['/usr/local/bin/akhexecd'], 'args': ['--config', '/config/service.json'],
                                    'ports': [{'name': 'http', 'containerPort': 8080}],
                                    'readinessProbe': {'httpGet': {'path': '/healthz', 'port': 8080}},
                                    'securityContext': {'allowPrivilegeEscalation': False, 'readOnlyRootFilesystem': True,
                                                        'capabilities': {'drop': ['ALL']}, 'seccompProfile': {'type': 'RuntimeDefault'}},
                                    'volumeMounts': [{'name': 'workspace', 'mountPath': '/workspaces'},
                                                    {'name': 'state', 'mountPath': '/data'},
                                                    {'name': 'config', 'mountPath': '/config', 'readOnly': True},
                                                    {'name': 'credentials', 'mountPath': '/credentials', 'readOnly': True}]}],
                    'volumes': [{'name': 'workspace', 'emptyDir': {'sizeLimit': '128Mi'}},
                                {'name': 'state', 'emptyDir': {'sizeLimit': '128Mi'}},
                                {'name': 'config', 'configMap': {'name': 'executor'}},
                                {'name': 'credentials', 'secret': {'secretName': 'executor-verifiers'}}],
                }}},
            }
            pending_template = json.loads(json.dumps(template))
            pending_template['metadata'].pop('finalizers')
            pending_template['spec']['podTemplate']['spec']['nodeSelector'] = {'akh-test-unschedulable': 'true'}
            collection = f'/apis/agents.x-k8s.io/v1beta1/namespaces/{namespace}/sandboxes'
            def profile(template):
                return {'revision': 'native-test-1', 'site': 'disposable-kind', 'provisioner': {
                    'kind': 'kubernetes', 'api': 'k8s', 'namespace': namespace, 'resource': collection,
                    'executor': 'executor', 'template': template}}
            config = {
                'service_id': 'manager', 'listen': f'127.0.0.1:{manager_port}', 'data_dir': str(root / 'manager'),
                'principals': [{'id': 'sessions', 'token_file': str(root / 'session.token'), 'roles': ['session']}],
                'endpoints': {'k8s': {'url': cluster['server'], 'token_file': str(api_token), 'ca_file': str(ca)},
                              'executor': {'url': f'http://127.0.0.1:{executor_port}', 'token_file': str(root / 'manager.token')}},
                'workspaces': {'pod': profile(template), 'pending': profile(pending_template)},
            }
            path = root / 'manager.json'; path.write_text(json.dumps(config))

            def request(port, path, body=None):
                req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
                    data=None if body is None else json.dumps(body).encode(),
                    headers={'Authorization': f'Bearer {session_token}', 'Content-Type': 'application/json'})
                try:
                    with urllib.request.urlopen(req, timeout=35) as response: return json.load(response)
                except urllib.error.HTTPError as error:
                    value = json.load(error)
                    assert value.get('code') in ['busy', 'transport', 'conflict'], value
                    return None

            def start_manager():
                log = (root / 'manager.log').open('ab'); logs.append(log)
                process = subprocess.Popen([str(args.manager), '--config', str(path)], stdout=log, stderr=log)
                children.append(process)
                wait(lambda: request(manager_port, '/healthz'), 'manager readiness', 15)
                return process

            manager = start_manager()
            allocation_request = {'session_id': 'native', 'principal': 'test', 'profile': 'pod'}
            request(manager_port, '/v1/allocations', allocation_request)
            allocation = request(manager_port, '/v1/allocations/native')
            identity = allocation['id']
            def ready_pod():
                result = kube('-n', namespace, 'get', 'pod', identity, '-o', 'json', check=False)
                if result.returncode: return False
                pod = json.loads(result.stdout)
                return any(c['type'] == 'Ready' and c['status'] == 'True' for c in pod.get('status', {}).get('conditions', []))
            wait(ready_pod, 'executor pod readiness')
            log = (root / 'port-forward.log').open('ab'); logs.append(log)
            forward = subprocess.Popen(command + ['-n', namespace, 'port-forward', f'pod/{identity}', f'{executor_port}:8080'], stdout=log, stderr=log)
            children.append(forward)
            wait(lambda: request(executor_port, '/healthz'), 'executor port forward')
            allocation = wait(lambda: request(manager_port, '/v1/allocations', allocation_request), 'executor handshake')
            assert allocation['state'] == 'ready' and allocation['binding']['platform'] == 'linux'
            original_uid = allocation['resource_uid']
            assert request(manager_port, '/v1/allocations', allocation_request)['id'] == identity
            operation = {'id': 'write-once', 'session_id': 'native', 'turn_id': 'turn',
                         'workspace_id': identity, 'generation': 1, 'operation': 'write',
                         'arguments': {'path': 'marker', 'content': 'in the sandbox', 'create': True}}
            first = request(executor_port, '/v1/operations', operation)
            assert first['status'] == 'completed'
            assert request(executor_port, '/v1/operations', operation) == first
            assert kube('-n', namespace, 'exec', identity, '--', 'cat', allocation['binding']['root'] + '/marker').stdout == 'in the sandbox'

            assert request(manager_port, '/v1/allocations/native/release', {}) is None
            assert request(manager_port, '/v1/allocations/native')['state'] == 'releasing'
            manager.terminate(); manager.wait(timeout=12)
            manager = start_manager()
            assert request(manager_port, '/v1/allocations/native/release', {}) is None
            resource = json.loads(kube('-n', namespace, 'get', 'sandbox', identity, '-o', 'json').stdout)
            assert resource['metadata']['uid'] == original_uid
            index = resource['metadata']['finalizers'].index(HOLD)
            patch = [{'op': 'test', 'path': '/metadata/uid', 'value': original_uid},
                     {'op': 'test', 'path': f'/metadata/finalizers/{index}', 'value': HOLD},
                     {'op': 'remove', 'path': f'/metadata/finalizers/{index}'}]
            kube('-n', namespace, 'patch', 'sandbox', identity, '--type=json', '-p', json.dumps(patch))
            released = wait(lambda: request(manager_port, '/v1/allocations/native/release', {}), 'foreground deletion')
            assert released['state'] == 'released'
            assert not kube('-n', namespace, 'get', 'pod,service', identity, '--ignore-not-found', '-o', 'name').stdout.strip()

            early = {'session_id': 'early', 'principal': 'test', 'profile': 'pending'}
            assert request(manager_port, '/v1/allocations', early) is None
            pending = request(manager_port, '/v1/allocations/early')
            assert pending['binding'] is None
            released = wait(lambda: request(manager_port, '/v1/allocations/early/release', {}), 'unbound allocation cleanup')
            assert released['state'] == 'released'
            print(json.dumps({'result': 'passed', 'controller': 'agent-sandbox v1.0.0', 'context': args.context,
                              'checks': ['scoped Kubernetes API token and CA', 'native executor pod', 'verifier authentication',
                                         'stable allocation and receipt', 'pod filesystem write', 'manager restart during foreground deletion',
                                         'release before executor attachment'],
                              'not_proven': ['native provider conformance', 'gVisor/Kata isolation', 'production deployment']}))
    finally:
        for child in reversed(children):
            if child.poll() is None: child.terminate()
        for child in reversed(children):
            try: child.wait(timeout=12)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for log in logs: log.close()
        # Remove only this test's hold, if an assertion failed during deletion.
        resources = kube('-n', namespace, 'get', 'sandboxes', '-o', 'json', check=False)
        if resources.returncode == 0:
            for resource in json.loads(resources.stdout).get('items', []):
                finalizers = resource['metadata'].get('finalizers', [])
                if HOLD in finalizers:
                    index = finalizers.index(HOLD)
                    patch = [{'op': 'test', 'path': '/metadata/uid', 'value': resource['metadata']['uid']},
                             {'op': 'test', 'path': f'/metadata/finalizers/{index}', 'value': HOLD},
                             {'op': 'remove', 'path': f'/metadata/finalizers/{index}'}]
                    kube('-n', namespace, 'patch', 'sandbox', resource['metadata']['name'], '--type=json', '-p', json.dumps(patch), check=False)
        kube('delete', 'namespace', namespace, '--wait=false', check=False)


if __name__ == '__main__':
    main()
