#!/usr/bin/env python3
"""Released kagent BYO controller -> Akh facade -> TLS central sessions.

Requires kagent 0.10.1 in an explicitly disposable Akh kind cluster, with the
session/facade/TLS images already loaded. Installs the repository's unmodified
BYO Helm template, tests actual controller A2A routes, then deletes only its own
release and resources. No model or account is used. kindnet does not enforce
NetworkPolicy: this test cannot certify the trusted-hop network boundary.
"""
import argparse
import hashlib
import http.client
import json
import pathlib
import secrets
import socket
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]


def wait(condition, label, seconds=90):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            result = condition()
            if result:
                return result
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.3)
    raise AssertionError('timed out: ' + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kubeconfig', type=pathlib.Path, required=True)
    parser.add_argument('--context', required=True)
    parser.add_argument('--namespace', default='kagent-test')
    parser.add_argument('--session-image', default='akh-session-platform-local:session')
    parser.add_argument('--facade-image', required=True, help='loaded image pinned by its containerd digest')
    parser.add_argument('--tls-image', default='nginxinc/nginx-unprivileged@sha256:0c79d56aee561a1d81c63f00eee5fb5fe29279560cdc55e91425133104c7fbe6')
    args = parser.parse_args()
    assert args.kubeconfig.is_absolute() and args.kubeconfig.is_file()
    assert args.context.startswith('kind-akh-platform-'), 'requires an explicitly disposable Akh kind cluster'
    assert '@sha256:' in args.facade_image
    kube_cmd = ['kubectl', '--kubeconfig', str(args.kubeconfig), '--context', args.context, '-n', args.namespace]
    helm_cmd = ['helm', '--kubeconfig', str(args.kubeconfig), '--kube-context', args.context, '-n', args.namespace]
    name = 'akh-contract-' + secrets.token_hex(5)
    central = name + '-central'
    selector = 'akh-test=' + name
    children, logs = [], []
    installed = False

    def kube(*argv, body=None, check=True):
        result = subprocess.run(kube_cmd + list(argv), input=None if body is None else json.dumps(body),
                                text=True, capture_output=True, timeout=45)
        if check and result.returncode:
            raise AssertionError(f'kubectl {argv[0]} failed: {result.stderr[:1000]}')
        return result

    def apply(kind, identity, **fields):
        kube('apply', '-f', '-', body={'apiVersion': 'v1', 'kind': kind,
             'metadata': {'name': identity, 'namespace': args.namespace, 'labels': {'akh-test': name}}, **fields})

    # Refuse to run against a different controller or arbitrary kubeconfig URL.
    cluster = json.loads(kube('config', 'view', '--minify', '-o', 'json').stdout)['clusters'][0]['cluster']
    assert cluster['server'].startswith(('https://127.0.0.1:', 'https://localhost:'))
    controller = json.loads(kube('get', 'deployment', 'kagent-controller', '-o', 'json').stdout)
    assert controller['spec']['template']['spec']['containers'][0]['image'] == 'ghcr.io/kagent-dev/kagent/controller:0.10.1'
    try:
        with tempfile.TemporaryDirectory(prefix='akh-kagent-native-') as directory:
            root = pathlib.Path(directory)
            cert, key, ca = root / 'tls.crt', root / 'tls.key', root / 'ca.pem'
            dns = f'{central}.{args.namespace}.svc'
            subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                '-keyout', str(root / 'ca.key'), '-out', str(ca), '-subj', '/CN=Akh disposable test CA',
                '-addext', 'basicConstraints=critical,CA:TRUE'],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(['openssl', 'req', '-new', '-newkey', 'rsa:2048', '-nodes',
                '-keyout', str(key), '-out', str(root / 'tls.csr'), '-subj', '/CN=' + dns],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            extensions = root / 'extensions'
            extensions.write_text(f'subjectAltName=DNS:{dns}\nbasicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\n')
            subprocess.run(['openssl', 'x509', '-req', '-in', str(root / 'tls.csr'), '-CA', str(ca),
                '-CAkey', str(root / 'ca.key'), '-CAcreateserial', '-out', str(cert), '-days', '1', '-extfile', str(extensions)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            key.chmod(0o600)
            token = secrets.token_urlsafe(32)
            apply('Secret', central, stringData={'tls.crt': cert.read_text(), 'tls.key': key.read_text(),
                'verifier': hashlib.sha256(token.encode()).hexdigest()})
            apply('Secret', name + '-credentials', stringData={'token': token, 'ca.pem': ca.read_text()})
            config = {'service_id': 'native-kagent', 'listen': '127.0.0.1:8081',
                'public_url': f'https://{dns}:8443', 'data_dir': '/data',
                'principals': [{'id': 'kagent-integration', 'token_file': '/credentials/verifier',
                                'token_sha256': True, 'roles': ['client'], 'akhs': ['test']}],
                'akhs': {'test': {'runtime': {'runtime': 'deliberately-unavailable', 'profile': 'test',
                                'revision': '1', 'kind': 'internal', 'execution_mode': 'brokered'},
                                'instructions': 'wire conformance only', 'workspace_profiles': ['unallocated']}}}
            nginx = '''pid /tmp/nginx.pid;
error_log /dev/stderr warn;
events {}
http {
 # Only synthetic contract bodies, never headers or credentials.
 log_format contract escape=json '{"body":"$request_body"}';
 access_log /dev/stdout contract;
 client_body_temp_path /tmp/client_temp;
 proxy_temp_path /tmp/proxy_temp;
 fastcgi_temp_path /tmp/fastcgi_temp;
 uwsgi_temp_path /tmp/uwsgi_temp;
 scgi_temp_path /tmp/scgi_temp;
 server {
  listen 8443 ssl;
  ssl_certificate /credentials/tls.crt;
  ssl_certificate_key /credentials/tls.key;
  location / {
   proxy_pass http://127.0.0.1:8081;
   proxy_http_version 1.1;
   proxy_buffering off;
   proxy_read_timeout 60s;
  }
 }
}
'''
            apply('ConfigMap', central, data={'service.json': json.dumps(config), 'nginx.conf': nginx})
            security = {'runAsNonRoot': True, 'runAsUser': 10001, 'runAsGroup': 10001, 'fsGroup': 10001,
                        'seccompProfile': {'type': 'RuntimeDefault'}}
            container_security = {'readOnlyRootFilesystem': True, 'allowPrivilegeEscalation': False,
                                  'capabilities': {'drop': ['ALL']}}
            mounts = [{'name': 'config', 'mountPath': '/config', 'readOnly': True},
                      {'name': 'credentials', 'mountPath': '/credentials', 'readOnly': True}]
            apply('Pod', central, spec={'automountServiceAccountToken': False, 'securityContext': security,
                'containers': [
                    {'name': 'sessions', 'image': args.session_image, 'imagePullPolicy': 'IfNotPresent',
                     'securityContext': container_security,
                     'volumeMounts': mounts + [{'name': 'data', 'mountPath': '/data'}]},
                    {'name': 'tls', 'image': args.tls_image, 'imagePullPolicy': 'IfNotPresent',
                     'command': ['nginx', '-c', '/config/nginx.conf', '-g', 'daemon off;'],
                     'securityContext': container_security,
                     'volumeMounts': mounts + [{'name': 'tmp', 'mountPath': '/tmp'}],
                     'readinessProbe': {'httpGet': {'scheme': 'HTTPS', 'port': 8443, 'path': '/healthz'},
                                        'initialDelaySeconds': 1, 'periodSeconds': 2}}],
                'volumes': [{'name': 'data', 'emptyDir': {}}, {'name': 'tmp', 'emptyDir': {}},
                            {'name': 'config', 'configMap': {'name': central}},
                            {'name': 'credentials', 'secret': {'secretName': central, 'defaultMode': 288}}]})
            apply('Service', central, spec={'selector': {'akh-test': name}, 'ports': [{'port': 8443, 'targetPort': 8443}]})
            kube('wait', '--for=condition=Ready', 'pod/' + central, '--timeout=40s')
            values = {'kagent': {'enabled': True, 'name': name, 'akh': 'test', 'image': args.facade_image,
                'sessionsUrl': f'https://{dns}:8443', 'secretName': name + '-credentials',
                'controllerUrl': f'http://kagent-controller.{args.namespace}.svc:8083',
                'userId': 'akh-contract@example.invalid', 'workspaceProfile': 'unallocated',
                'allowedCallerLabels': controller['spec']['selector']['matchLabels']}}
            values_path = root / 'values.json'; values_path.write_text(json.dumps(values))
            # Set before invoking: cleanup also handles a partially installed release.
            installed = True
            result = subprocess.run(helm_cmd + ['upgrade', '--install', name, str(ROOT / 'charts/akh-session-platform'),
                                    '-f', str(values_path)], capture_output=True, text=True, timeout=45)
            assert result.returncode == 0, result.stderr[:1500]

            def ready_agent():
                agent = json.loads(kube('get', 'agent', name, '-o', 'json').stdout)
                conditions = agent.get('status', {}).get('conditions', [])
                return agent if any(c['type'] == 'Ready' and c['status'] == 'True' for c in conditions) else False
            agent = wait(ready_agent, 'released BYO controller readiness', 90)
            deployment = json.loads(kube('get', 'deployment', name, '-o', 'json').stdout)
            pod_labels = deployment['spec']['template']['metadata']['labels']
            assert pod_labels['app'] == 'kagent' and pod_labels['kagent'] == name, pod_labels
            assert deployment['spec']['template']['spec']['containers'][0]['image'] == args.facade_image
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); forwarded = sock.getsockname()[1]
            log = (root / 'forward.log').open('wb'); logs.append(log)
            children.append(subprocess.Popen(kube_cmd + ['port-forward', 'service/kagent-controller', f'{forwarded}:8083'],
                                            stdout=log, stderr=log))
            endpoint = f'http://127.0.0.1:{forwarded}/api/a2a/{args.namespace}/{name}/'

            def request(path='', body=None, accept='application/json'):
                return urllib.request.urlopen(urllib.request.Request(endpoint + path,
                    data=None if body is None else json.dumps(body).encode(), headers={
                        'Content-Type': 'application/json', 'Accept': accept, 'X-User-Id': 'akh-contract@example.invalid'}), timeout=20)

            def card_request():
                with request('.well-known/agent-card.json') as response:
                    return json.load(response)
            card = wait(card_request, 'controller agent discovery', 20)
            assert card.get('name'), card

            def controller_api(path, body=None, method=None):
                with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{forwarded}' + path,
                        data=None if body is None else json.dumps(body).encode(), method=method,
                        headers={'Content-Type': 'application/json', 'X-User-Id': 'akh-contract@example.invalid'}), timeout=20) as response:
                    return json.load(response) if response.status != 204 else None

            # This is the upstream UI's order: create a kagent context first.
            external = str(uuid.uuid4())
            created = controller_api('/api/sessions', {'id': external, 'agent_ref': f'{args.namespace}/{name}'})
            assert created['data']['id'] == external

            def rpc(method, params):
                with request(body={'jsonrpc': '2.0', 'id': secrets.token_hex(5), 'method': method, 'params': params}) as response:
                    value = json.load(response)
                assert 'error' not in value, value
                return value['result']

            def message(identity, context=None):
                value = {'message': {'kind': 'message', 'role': 'user', 'messageId': identity,
                        'parts': [{'kind': 'text', 'text': 'controller transport contract'}]}, 'configuration': {'blocking': False}}
                if context: value['message']['contextId'] = context
                return value
            first = rpc('message/send', message('immutable-occurrence', external))
            again = rpc('message/send', message('immutable-occurrence', external))
            assert first['id'] == again['id'] and first['contextId'] == again['contextId'], (first, again)
            assert first['contextId'] == external and first['metadata']['akhSessionId'] != external
            assert first['status']['state'] == 'submitted', first
            assert rpc('tasks/get', {'id': first['id']})['id'] == first['id']
            with urllib.request.urlopen(urllib.request.Request(endpoint,
                    data=json.dumps({'jsonrpc': '2.0', 'id': 'other-user', 'method': 'tasks/cancel',
                                     'params': {'id': first['id']}}).encode(),
                    headers={'Content-Type': 'application/json', 'X-User-Id': 'other@example.invalid'}), timeout=20) as response:
                assert 'error' in json.load(response), 'another user must not cancel the central task'
            assert rpc('tasks/get', {'id': first['id']})['status']['state'] == 'submitted'
            assert rpc('tasks/cancel', {'id': first['id']})['status']['state'] == 'canceled'
            followup = rpc('message/send', message('followup', external))
            assert followup['id'] != first['id'] and followup['contextId'] == external
            assert followup['metadata']['akhSessionId'] == first['metadata']['akhSessionId']
            # Upstream permanently burns a deleted task ID. It must neither be
            # silently recreated nor prevent other central updates from projecting.
            controller_api('/api/tasks/' + first['id'], method='DELETE')
            old_pod = json.loads(kube('get', 'pod', '-l', 'kagent=' + name, '-o', 'json').stdout)['items'][0]
            kube('patch', 'agent', name, '--type=merge', '-p', json.dumps({'spec': {'byo': {'deployment': {'replicas': 0}}}}))
            wait(lambda: not json.loads(kube('get', 'pod', '-l', 'kagent=' + name, '-o', 'json').stdout)['items'], 'facade stopped', 60)
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0)); central_port = sock.getsockname()[1]
            log = (root / 'central-forward.log').open('wb'); logs.append(log)
            children.append(subprocess.Popen(kube_cmd + ['port-forward', 'pod/' + central, f'{central_port}:8443'], stdout=log, stderr=log))
            tls_context = ssl.create_default_context(cafile=str(ca))
            def central_cancel():
                connection = http.client.HTTPSConnection(dns, 8443, context=tls_context, timeout=20)
                try:
                    # Retain CA AND DNS verification while connecting the explicit
                    # local port forward, without altering hosts or global trust.
                    connection.sock = tls_context.wrap_socket(socket.create_connection(('127.0.0.1', central_port), timeout=20), server_hostname=dns)
                    connection.request('POST', f'/v1/sessions/{first["metadata"]["akhSessionId"]}/turns/{followup["id"]}/cancel',
                        '{}', {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
                    response = connection.getresponse()
                    assert response.status == 200, response.status
                    return json.loads(response.read())
                finally:
                    connection.close()
            wait(central_cancel, 'central cancellation while facade stopped', 20)
            kube('patch', 'agent', name, '--type=merge', '-p', json.dumps({'spec': {'byo': {'deployment': {'replicas': 1}}}}))
            def replaced():
                pods = json.loads(kube('get', 'pod', '-l', 'kagent=' + name, '-o', 'json').stdout)['items']
                return any(p['metadata']['uid'] != old_pod['metadata']['uid'] and
                    any(c['type'] == 'Ready' and c['status'] == 'True' for c in p.get('status', {}).get('conditions', [])) for p in pods)
            wait(replaced, 'facade replacement readiness', 60)
            restored = wait(lambda: (task if (task := rpc('tasks/get', {'id': followup['id']}))['status']['state'] == 'canceled' else False), 'projection after restart', 30)
            assert restored['contextId'] == external
            with request(body={'jsonrpc': '2.0', 'id': 'stream', 'method': 'message/stream',
                               'params': message('stream-occurrence')}, accept='text/event-stream') as stream:
                events = []
                for line in stream:
                    if not line.startswith(b'data:'): continue
                    value = json.loads(line[5:])
                    assert 'error' not in value, value
                    event = value['result']; events.append(event)
                    if len(events) == 1:
                        assert event['status']['state'] == 'submitted', event
                        assert event['contextId'] != first['contextId']
                        rpc('tasks/cancel', {'id': event['id']})
                    if event.get('final'):
                        assert event['status']['state'] == 'canceled', event
                        break
                assert len(events) >= 2 and events[-1].get('final'), events
            print(json.dumps({'result': 'passed', 'controller': 'kagent 0.10.1',
                'checks': ['released BYO CRD and controller', 'repository Helm template', 'private-CA TLS to central sessions',
                           'actual controller A2A discovery and 0.3 forwarding', 'durable kagent context alias',
                           'immutable message dedupe', 'different user denied cancellation', 'polling and cancellation',
                           'central change projected after facade restart despite a deleted task view', 'SSE terminal event'],
                'not_proven': ['NetworkPolicy enforcement', 'native provider tools', 'UI/browser', 'production deployment']}))
    except Exception:
        # Only own workload status/logs; never dump Secrets/configuration credentials.
        for resource in ['pod/' + central, 'deployment/' + name]:
            result = kube('logs', resource, '--all-containers', '--tail=15', check=False)
            print(resource + ': ' + result.stdout[:3000])
        result = kube('get', 'agent', name, '-o', 'json', check=False)
        if result.returncode == 0:
            print('Agent status: ' + json.dumps(json.loads(result.stdout).get('status', {})))
        raise
    finally:
        for child in reversed(children):
            if child.poll() is None: child.terminate()
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for log in logs: log.close()
        if installed:
            subprocess.run(helm_cmd + ['uninstall', name, '--wait', '--timeout', '30s'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40)
        kube('delete', 'pod,service,configmap,secret', '-l', selector, '--wait=false', check=False)


if __name__ == '__main__':
    main()
