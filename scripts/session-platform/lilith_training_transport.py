#!/usr/bin/env python3
"""Bounded training-worker transport; workspace effects go through akhrtd."""
import hashlib
import json
from pathlib import Path

MAX_FRAME = 8 * 1024 * 1024


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


class RuntimeTransport:
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing = incoming, outgoing
        self.request = None
        self.sequence = 0
        self.receipts = []

    def receive(self):
        line = self.incoming.readline(MAX_FRAME + 1)
        if not line or len(line) > MAX_FRAME or not line.endswith(b'\n'):
            raise ValueError('incomplete or oversized runtime frame')
        value = json.loads(line, object_pairs_hook=unique_object)
        if not isinstance(value, dict):
            raise ValueError('runtime frame must be an object')
        return value

    def send(self, value):
        encoded = json.dumps(value, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        if len(encoded) > MAX_FRAME:
            raise ValueError('worker frame exceeds transport limit')
        self.outgoing.write(encoded)
        self.outgoing.flush()

    def start(self):
        frame = self.receive()
        if set(frame) != {'type', 'request', 'config_path', 'config_sha256'} or frame['type'] != 'start':
            raise ValueError('expected runtime start frame')
        config_path = Path(frame['config_path'])
        if not config_path.is_absolute():
            raise ValueError('worker configuration path must be absolute')
        with config_path.open('rb') as stream:
            config_bytes = stream.read(MAX_FRAME + 1)
        if len(config_bytes) > MAX_FRAME or hashlib.sha256(config_bytes).hexdigest() != frame['config_sha256']:
            raise ValueError('worker configuration identity mismatch')
        request = frame['request']
        if not isinstance(request, dict) or not isinstance(request.get('tools'), list):
            raise ValueError('missing authoritative runtime request/catalog')
        if not isinstance(request.get('id'), str) or not request['id']:
            raise ValueError('missing runtime job identity')
        if any(not isinstance(tool, dict) or not isinstance(tool.get('name'), str) or not tool['name'] for tool in request['tools']):
            raise ValueError('invalid tool catalog entry')
        names = [tool['name'] for tool in request['tools']]
        if len(set(names)) != len(names):
            raise ValueError('duplicate tool names')
        self.request = request
        return json.loads(config_bytes, object_pairs_hook=unique_object)

    def tool(self, name, arguments):
        if self.request is None:
            raise ValueError('worker has not received runtime authority')
        if name not in {tool['name'] for tool in self.request['tools']}:
            raise ValueError('tool absent from authoritative session catalog')
        if not isinstance(arguments, dict):
            raise ValueError('tool arguments must be an object')
        self.sequence += 1
        job_tag = hashlib.sha256(self.request['id'].encode()).hexdigest()[:16]
        call_id = 'training-' + job_tag + '-' + str(self.sequence)
        self.send({'type': 'tool_call', 'id': call_id, 'name': name, 'arguments': arguments})
        frame = self.receive()
        if set(frame) != {'type', 'id', 'result'} or frame['type'] != 'tool_result' or frame['id'] != call_id:
            raise ValueError('runtime tool result does not match outstanding call')
        result = frame['result']
        if not isinstance(result, dict) or type(result.get('success')) is not bool or not isinstance(result.get('text'), str):
            raise ValueError('invalid runtime tool result')
        self.receipts.append({'id': call_id, 'name': name, 'arguments': arguments, 'result': result})
        return result

    def progress(self, phase):
        self.send({'type': 'progress', 'phase': phase})
