#!/usr/bin/env python3
"""Real A2A Python SDK against akhsessd + BYO facade, with no provider deployed.
Run: uv run --no-project --with a2a-sdk==0.3.22 python scripts/session-platform/a2a-sdk-smoke.py
Verifies discovery, typed requests, dedupe, task polling, cancellation and SSE.
This is SDK wire conformance, not a kagent controller or model execution test.
"""
import asyncio
import json
import pathlib
import secrets
import socket
import subprocess
import tempfile
import warnings

import httpx
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, SendStreamingMessageRequest, GetTaskRequest, CancelTaskRequest

ROOT = pathlib.Path(__file__).resolve().parents[2]

def port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def message(method, identity, text):
    return {'id': identity, 'method': method, 'params': {'message': {'kind': 'message', 'role': 'user', 'messageId': identity, 'parts': [{'kind': 'text', 'text': text}]}, 'configuration': {'blocking': False}}}

async def exercise(directory):
    root = pathlib.Path(directory)
    central, facade = port(), port()
    token = root / 'token'
    token.write_text(secrets.token_urlsafe(32)); token.chmod(0o600)
    config = {'service_id': 'sdk-test', 'listen': f'127.0.0.1:{central}', 'public_url': f'http://127.0.0.1:{central}', 'data_dir': str(root / 'state'), 'principals': [{'id': 'sdk-integration', 'token_file': str(token), 'roles': ['client'], 'akhs': ['test']}], 'akhs': {'test': {'runtime': {'runtime': 'deliberately-unavailable', 'profile': 'test', 'revision': '1', 'kind': 'internal', 'execution_mode': 'brokered'}, 'instructions': 'test', 'workspace_profiles': ['unallocated']}}}
    proxy = {'listen': f'127.0.0.1:{facade}', 'sessions': {'url': f'http://127.0.0.1:{central}', 'token_file': str(token)}, 'akh': 'test', 'public_url': f'http://127.0.0.1:{facade}/', 'trust_kagent_network': True}
    children, logs = [], []
    try:
        for binary, config in [('akhsessd', config), ('akh-kagent', proxy)]:
            path = root / f'{binary}.json'; path.write_text(json.dumps(config))
            log = (root / f'{binary}.log').open('wb'); logs.append(log)
            children.append(subprocess.Popen([str(ROOT / 'target/debug' / binary), '--config', str(path)], stdout=log, stderr=log))
        async with httpx.AsyncClient(timeout=10) as http:
            for _ in range(100):
                try:
                    response = await http.get(proxy['public_url'] + '.well-known/agent-card.json')
                    if response.status_code == 200: break
                except httpx.TransportError: pass
                await asyncio.sleep(.1)
            else: raise AssertionError('facade never became ready')
            card = await A2ACardResolver(http, proxy['public_url']).get_agent_card()
            assert card.name == 'test' and card.protocol_version == '0.3.0'
            assert card.url == proxy['public_url'] and not card.security
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                client = A2AClient(http, agent_card=card)
            request = SendMessageRequest.model_validate(message('message/send', 'immutable-occurrence', 'hello'))
            first = (await client.send_message(request)).root.result
            again = (await client.send_message(request)).root.result
            assert first.id == again.id and first.context_id == again.context_id
            assert first.status.state.value == 'submitted'
            fetched = (await client.get_task(GetTaskRequest(id='get', params={'id': first.id}))).root.result
            assert fetched.id == first.id
            cancelled = (await client.cancel_task(CancelTaskRequest(id='cancel', params={'id': first.id}))).root.result
            assert cancelled.status.state.value == 'canceled'
            request = SendStreamingMessageRequest.model_validate(message('message/stream', 'stream-occurrence', 'stream test'))
            stream = client.send_message_streaming(request)
            task = (await anext(stream)).root.result
            assert task.status.state.value == 'submitted' and task.context_id != first.context_id
            await client.cancel_task(CancelTaskRequest(id='cancel-stream', params={'id': task.id}))
            events = [event.root.result async for event in stream]
            assert events and events[-1].final and events[-1].status.state.value == 'canceled'
        print('a2a-sdk 0.3.22: discovery, identity, dedupe, polling, cancellation and SSE passed')
    finally:
        for child in reversed(children): child.terminate()
        for child in reversed(children):
            try: child.wait(timeout=12)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for log in logs: log.close()

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='akh-a2a-sdk-') as directory:
        asyncio.run(exercise(directory))
