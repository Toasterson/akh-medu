#!/usr/bin/env python3
"""Actual ACP bridge over a private-CA TLS terminator and real akhsessd.

Uses only Python's standard library plus openssl and the built Akh binaries.
Checks trust rejection, authenticated initialization and durable session reload.
No providers, public endpoints, or existing credentials are used.
"""
import asyncio
import json
import pathlib
import secrets
import socket
import ssl
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


async def exercise(root):
    def openssl(*args):
        subprocess.run(['openssl', *args], cwd=root, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    openssl('req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-keyout', 'ca.key',
            '-out', 'ca.pem', '-days', '1', '-subj', '/CN=Akh temporary test CA')
    openssl('req', '-newkey', 'rsa:2048', '-nodes', '-keyout', 'server.key',
            '-out', 'server.csr', '-subj', '/CN=localhost')
    (root / 'extensions').write_text('subjectAltName=IP:127.0.0.1\nextendedKeyUsage=serverAuth\nbasicConstraints=CA:FALSE\n')
    openssl('x509', '-req', '-in', 'server.csr', '-CA', 'ca.pem', '-CAkey', 'ca.key',
            '-CAcreateserial', '-out', 'server.pem', '-days', '1', '-extfile', 'extensions')
    token = root / 'token'
    token.write_text(secrets.token_urlsafe(32)); token.chmod(0o600)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        central = listener.getsockname()[1]
    config = {'service_id': 'wss-test', 'listen': f'127.0.0.1:{central}',
              'data_dir': str(root / 'state'),
              'principals': [{'id': 'test', 'token_file': str(token), 'roles': ['client'], 'akhs': ['test']}],
              'akhs': {'test': {'instructions': 'test', 'workspace_profiles': ['unallocated'],
                               'runtime': {'runtime': 'absent', 'profile': 'test', 'revision': '1',
                                           'kind': 'internal', 'execution_mode': 'brokered'}}}}
    config_path = root / 'sessions.json'; config_path.write_text(json.dumps(config))
    daemon = await asyncio.create_subprocess_exec(str(ROOT / 'target/debug/akhsessd'),
                '--config', str(config_path), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    connections = set()

    async def forward(reader, writer):
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()

    async def proxy(reader, writer):
        task = asyncio.current_task(); connections.add(task)
        upstream = None
        pumps = []
        try:
            peer, upstream = await asyncio.open_connection('127.0.0.1', central)
            pumps = [asyncio.create_task(forward(reader, upstream)), asyncio.create_task(forward(peer, writer))]
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for pump in pumps: pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            writer.close()
            if upstream: upstream.close()
            connections.discard(task)

    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(root / 'server.pem', root / 'server.key')
    server = await asyncio.start_server(proxy, '127.0.0.1', 0, ssl=tls)
    port = server.sockets[0].getsockname()[1]
    command = [str(ROOT / 'target/debug/akh-acp'), '--endpoint', f'wss://127.0.0.1:{port}/acp',
               '--token-file', str(token), '--akh', 'test', '--workspace-profile', 'unallocated']
    children = []
    try:
        for _ in range(100):
            try:
                _, writer = await asyncio.open_connection('127.0.0.1', central)
                writer.close(); await writer.wait_closed(); break
            except OSError: await asyncio.sleep(.1)
        else: raise AssertionError('session daemon did not start')

        rejected = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.PIPE,
                         stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        children.append(rejected)
        _, error = await asyncio.wait_for(rejected.communicate(b''), 15)
        assert rejected.returncode != 0 and b'UnknownIssuer' in error, 'untrusted TLS was not rejected'

        async def connect(session=None):
            args = command + ['--ca-file', str(root / 'ca.pem')]
            if session: args += ['--session', session]
            bridge = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE,
                       stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            children.append(bridge)

            async def rpc(identity, method, params):
                bridge.stdin.write((json.dumps({'jsonrpc': '2.0', 'id': identity,
                                               'method': method, 'params': params}) + '\n').encode())
                await bridge.stdin.drain()
                while True:
                    line = await asyncio.wait_for(bridge.stdout.readline(), 10)
                    assert line, 'bridge closed before the response'
                    response = json.loads(line)
                    if response.get('id') == identity:
                        assert 'error' not in response, response
                        return response['result']
            assert (await rpc(1, 'initialize', {'protocolVersion': 1}))['protocolVersion'] == 1
            created = await rpc(2, 'session/new', {'cwd': '/client/path', 'mcpServers': []})
            bridge.stdin.close()
            await asyncio.wait_for(bridge.wait(), 5)
            assert bridge.returncode == 0
            return created['sessionId']

        identity = await connect()
        assert await connect(identity) == identity
        print('PASS: private-CA WSS trust rejection, authenticated ACP initialize, session create and reconnect/load')
    finally:
        for child in children + [daemon]:
            if child.returncode is None:
                child.terminate()
                try: await asyncio.wait_for(child.wait(), 10)
                except asyncio.TimeoutError: child.kill(); await child.wait()
        server.close(); await server.wait_closed()
        for task in list(connections): task.cancel()
        await asyncio.gather(*connections, return_exceptions=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='akh-wss-') as directory:
        asyncio.run(exercise(pathlib.Path(directory)))
