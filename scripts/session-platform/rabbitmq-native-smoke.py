#!/usr/bin/env python3
"""Actual RabbitMQ -> durable akheventd outbox -> actual central admission.

Creates one disposable Docker broker with loopback-only ports. No production
broker, provider, workspace allocation or account is used. Broker acknowledges
only after local persistence; central admission and model execution are separate.
"""
import argparse
import base64
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


def port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def wait(condition, label, seconds=60):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            result = condition()
            if result:
                return result
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(.25)
    raise AssertionError('timed out: ' + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=pathlib.Path, default=ROOT / 'target/debug')
    parser.add_argument('--image', default='rabbitmq@sha256:ac1201d1dc227779d93c607f976e7f9156fd5b551ee5ff7946cab6c242091bf3')
    args = parser.parse_args()
    container = 'akh-rabbit-contract-' + secrets.token_hex(5)
    processes, logs = {}, []
    started = False
    with tempfile.TemporaryDirectory(prefix='akh-rabbit-contract-') as directory:
        root = pathlib.Path(directory)
        secret = secrets.token_urlsafe(32)
        env = root / 'broker.env'
        env.write_text('RABBITMQ_DEFAULT_USER=akh-test\nRABBITMQ_DEFAULT_PASS=' + secret + '\n')
        env.chmod(0o600)
        try:
            # Docker may allocate different ephemeral host ports on restart.
            # Explicit ports keep the configured AMQP endpoint stable.
            amqp, management = port(), port()
            subprocess.run(['docker', 'run', '-d', '--name', container, '--env-file', str(env),
                '--cpus=1', '--memory=512m', '-p', f'127.0.0.1:{amqp}:5672', '-p', f'127.0.0.1:{management}:15672', args.image],
                check=True, capture_output=True, timeout=90)
            started = True
            bindings = json.loads(subprocess.run(['docker', 'inspect', container], check=True, capture_output=True, text=True).stdout)[0]['NetworkSettings']['Ports']
            amqp = int(bindings['5672/tcp'][0]['HostPort'])
            management = int(bindings['15672/tcp'][0]['HostPort'])
            auth = 'Basic ' + base64.b64encode(('akh-test:' + secret).encode()).decode()

            def broker(path, body=None, method=None):
                request = urllib.request.Request(f'http://127.0.0.1:{management}/api' + path,
                    data=None if body is None else json.dumps(body).encode(), method=method,
                    headers={'Authorization': auth, 'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=10) as response:
                    return json.load(response) if response.status not in (201, 204) else True

            overview = wait(lambda: broker('/overview'), 'RabbitMQ management readiness', 90)
            for queue, arguments in [('dead', {}), ('events', {'x-dead-letter-exchange': '', 'x-dead-letter-routing-key': 'dead'})]:
                broker('/queues/%2F/' + queue, {'durable': True, 'auto_delete': False, 'arguments': arguments}, 'PUT')
            uri = root / 'broker-uri'
            uri.write_text(f'amqp://akh-test:{secret}@127.0.0.1:{amqp}/%2f'); uri.chmod(0o600)
            tokens = {name: secrets.token_urlsafe(32) for name in ['observer', 'ingress']}
            for name, token in tokens.items():
                path = root / (name + '.token'); path.write_text(token); path.chmod(0o600)
            ports = {'akheventd': port(), 'akhsessd': port()}
            observer = {'id': 'observer', 'roles': ['admin'], 'akhs': ['test'], 'token_file': str(root / 'observer.token')}
            configs = {
                'akheventd': {'service_id': 'native-events', 'listen': f'127.0.0.1:{ports["akheventd"]}',
                    'data_dir': str(root / 'events'), 'principals': [observer],
                    'endpoints': {'sessions': {'url': f'http://127.0.0.1:{ports["akhsessd"]}', 'token_file': str(root / 'ingress.token')}},
                    'brokers': [{'uri_file': str(uri), 'queue': 'events', 'source': 'alerts'}],
                    'routes': [{'id': 'alerts', 'source': 'alerts', 'event_type': 'alert.firing', 'automation': 'alert'}]},
                'akhsessd': {'service_id': 'native-sessions', 'listen': f'127.0.0.1:{ports["akhsessd"]}',
                    'data_dir': str(root / 'sessions'), 'principals': [observer,
                        {'id': 'ingress', 'roles': ['event'], 'sources': ['alerts'], 'token_file': str(root / 'ingress.token')}],
                    'akhs': {'test': {'runtime': {'runtime': 'unavailable', 'profile': 'test', 'revision': '1', 'kind': 'internal', 'execution_mode': 'brokered'},
                                      'instructions': 'test', 'workspace_profiles': ['unallocated']}},
                    'automations': {'alert': {'id': 'alert', 'enabled': True, 'principal': 'observer', 'akh': 'test',
                        'workspace_profile': 'unallocated', 'instructions': 'Inspect this alert', 'sources': ['alerts'],
                        'target': {'kind': 'per_occurrence'}, 'notify_only': False, 'max_active': 8, 'max_age_ms': 300000}}}
            }
            for name, config in configs.items():
                (root / (name + '.json')).write_text(json.dumps(config))

            def request(name, path):
                with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{ports[name]}' + path,
                        headers={'Authorization': 'Bearer ' + tokens['observer']}), timeout=10) as response:
                    return json.load(response)

            def start(name):
                log = (root / (name + '.log')).open('ab'); logs.append(log)
                processes[name] = subprocess.Popen([str(args.bin_dir / name), '--config', str(root / (name + '.json'))], stdout=log, stderr=log)
                wait(lambda: request(name, '/healthz'), name + ' readiness', 20)

            def stop(name):
                child = processes.pop(name)
                child.terminate()
                try: child.wait(timeout=12)
                except subprocess.TimeoutExpired: child.kill(); child.wait()

            def publish(payload):
                value = broker('/exchanges/%2F/amq.default/publish', {'properties': {'delivery_mode': 2}, 'routing_key': 'events',
                    'payload': payload if isinstance(payload, str) else json.dumps(payload), 'payload_encoding': 'string'})
                assert value['routed'] is True

            start('akheventd') # Deliberately keep central sessions unavailable.
            event = {'schema_version': 1, 'id': 'alert-once', 'source': 'alerts', 'type': 'alert.firing',
                     'occurred_at': int(time.time() * 1000), 'subject': 'test', 'correlation_id': 'incident',
                     'data': {'summary': 'synthetic broker contract'}}
            publish(event); publish(event)
            publish('{')
            publish({**event, 'source': 'ungranted'})
            publish({**event, 'data': {'summary': 'conflicting reuse'}})
            pending = wait(lambda: (rows if len(rows := request('akheventd', '/v1/deliveries')) == 1 and rows[0]['attempts'] > 0 else False), 'durable outbox while central unavailable')
            assert pending[0]['admission'] is None and pending[0]['terminal'] is False
            def acknowledged(minimum):
                queue = broker('/queues/%2F/events')
                return queue.get('messages') == 0 and queue.get('message_stats', {}).get('ack', 0) >= minimum
            wait(lambda: acknowledged(2), 'persisted event acknowledgements')
            wait(lambda: broker('/queues/%2F/dead').get('messages_ready') == 3, 'malformed, forged and conflicting events dead-lettered')
            stop('akheventd'); start('akheventd')
            recovered = request('akheventd', '/v1/deliveries')
            assert recovered[0]['id'] == pending[0]['id'] and recovered[0]['request'] == pending[0]['request']
            start('akhsessd')
            delivered = wait(lambda: (rows[0] if (rows := request('akheventd', '/v1/deliveries'))[0]['admission'] else False), 'central admission from recovered outbox')
            assert delivered['terminal'] is True and delivered['error'] is None
            sessions = request('akhsessd', '/v1/sessions')
            assert len(sessions) == 1 and len(sessions[0]['turns']) == 1
            assert sessions[0]['id'] == delivered['admission']['session_id']
            assert sessions[0]['turns'][0]['id'] == delivered['admission']['turn_id']
            assert sessions[0]['turns'][0]['input']['origin']['kind'] == 'automation'
            # Restart only the owned broker and republish the original delivery.
            subprocess.run(['docker', 'restart', '--time=5', container], check=True, capture_output=True, timeout=40)
            wait(lambda: broker('/overview'), 'broker restart', 90)
            wait(lambda: broker('/queues/%2F/events').get('consumers') == 1, 'consumer reconnect')
            previous_acks = broker('/queues/%2F/events').get('message_stats', {}).get('ack', 0)
            publish(event)
            wait(lambda: acknowledged(previous_acks + 1), 'duplicate acknowledged after reconnect')
            assert len(request('akheventd', '/v1/deliveries')) == 1
            assert len(request('akhsessd', '/v1/sessions')[0]['turns']) == 1
            print(json.dumps({'result': 'passed', 'rabbitmq_version': overview['rabbitmq_version'],
                'checks': ['native AMQP consumption', 'acknowledgement after durable outbox', 'source authority and dead lettering',
                           'event ID conflict rejection', 'ingress restart while central unavailable', 'one central session and turn',
                           'broker restart and deduplicated redelivery'],
                'not_proven': ['model execution', 'workspace allocation', 'production broker deployment']}))
        except Exception:
            for name in configs if 'configs' in locals() else []:
                log = root / (name + '.log')
                if log.exists(): print(name + ': ' + log.read_text(errors='replace')[-3000:])
            raise
        finally:
            for child in processes.values():
                if child.poll() is None: child.terminate()
            for child in processes.values():
                try: child.wait(timeout=12)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            for log in logs: log.close()
            if started:
                subprocess.run(['docker', 'rm', '-f', '-v', container], check=True, capture_output=True, timeout=40)


if __name__ == '__main__':
    main()
