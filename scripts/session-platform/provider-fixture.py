#!/usr/bin/env python3
"""Test-only Codex app-server peer; no model, credential or network access."""
import json
import pathlib
import sys
import time
import uuid


def read():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


def send(value):
    print(json.dumps(value), flush=True)


def reply(request, value):
    send({'id': request['id'], 'result': value})


init = read()
assert init['method'] == 'initialize'
reply(init, {})
assert read()['method'] == 'initialized'
start = read()
state_file = pathlib.Path('fixture-provider-session.json')
if start['method'] == 'thread/resume':
    state = json.loads(state_file.read_text())
    assert state['id'] == start['params']['threadId']
else:
    assert start['method'] == 'thread/start'
    state = {'id': str(uuid.uuid4()), 'turns': 0}
reply(start, {'thread': {'id': state['id']}})
turn = read()
assert turn['method'] == 'turn/start'
state['turns'] += 1
state_file.write_text(json.dumps(state))
reply(turn, {'turn': {'id': 'fixture-turn'}})
prompt = turn['params']['input'][0]['text']
try:
    action = json.loads(prompt)
except ValueError:
    action = {'text': prompt}
if action.get('hang'):
    time.sleep(90)
if action.get('crash'):
    sys.exit(2)
answer = action.get('text', 'fixture done')
if 'tool' in action:
    send({'id': 'fixture-tool', 'method': 'item/tool/call', 'params': {
        'threadId': state['id'], 'turnId': 'fixture-turn', 'namespace': 'akh',
        'tool': action['tool'], 'arguments': action['arguments']}})
    result = read()
    assert result['id'] == 'fixture-tool'
    answer = json.dumps(result['result'])
send({'method': 'item/agentMessage/delta', 'params': {'threadId': state['id'], 'itemId': 'answer', 'delta': answer}})
send({'method': 'turn/completed', 'params': {'threadId': state['id'], 'turn': {'id': 'fixture-turn', 'status': 'completed'}}})
