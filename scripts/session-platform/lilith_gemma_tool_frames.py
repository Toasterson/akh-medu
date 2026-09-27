"""Parse the pinned Gemma template's tool syntax without executing model text."""
import json
import math
import re

QUOTE = '<|"|>'
OPEN = '<|tool_call>call:'
CLOSE = '<tool_call|>'


class Arguments:
    def __init__(self, text):
        self.text, self.pos = text, 0

    def space(self):
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def take(self, marker):
        self.space()
        if not self.text.startswith(marker, self.pos):
            raise ValueError('malformed Gemma tool arguments')
        self.pos += len(marker)

    def string(self):
        self.space()
        if self.text.startswith(QUOTE, self.pos):
            self.pos += len(QUOTE)
            end = self.text.find(QUOTE, self.pos)
            if end < 0:
                raise ValueError('unclosed Gemma string')
            value = self.text[self.pos:end]
            self.pos = end + len(QUOTE)
            return value
        value, length = json.JSONDecoder().raw_decode(self.text[self.pos:])
        if not isinstance(value, str):
            raise ValueError('expected string')
        self.pos += length
        return value

    def value(self, depth=0):
        if depth > 32:
            raise ValueError('tool argument nesting exceeds limit')
        self.space()
        if self.pos >= len(self.text):
            raise ValueError('incomplete argument')
        char = self.text[self.pos]
        if self.text.startswith(QUOTE, self.pos) or char == '"':
            return self.string()
        if char == '{':
            self.pos += 1
            result = {}
            self.space()
            if self.text.startswith('}', self.pos):
                self.pos += 1
                return result
            while True:
                self.space()
                if self.text.startswith(QUOTE, self.pos) or self.text.startswith('"', self.pos):
                    key = self.string()
                else:
                    match = re.match(r'[A-Za-z_][A-Za-z0-9_-]*', self.text[self.pos:])
                    if not match:
                        raise ValueError('invalid argument key')
                    key = match.group()
                    self.pos += len(key)
                if key in result:
                    raise ValueError('duplicate argument key')
                self.take(':')
                result[key] = self.value(depth + 1)
                self.space()
                if self.text.startswith('}', self.pos):
                    self.pos += 1
                    return result
                self.take(',')
        if char == '[':
            self.pos += 1
            result = []
            self.space()
            if self.text.startswith(']', self.pos):
                self.pos += 1
                return result
            while True:
                result.append(self.value(depth + 1))
                self.space()
                if self.text.startswith(']', self.pos):
                    self.pos += 1
                    return result
                self.take(',')
        match = re.match(r'(?:true|false|null|None|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)', self.text[self.pos:])
        if not match:
            raise ValueError('unsupported argument value')
        raw = match.group()
        self.pos += len(raw)
        value = None if raw == 'None' else json.loads(raw)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite argument value')
        return value


def parse_response(text):
    if len(text.encode()) > 1024 * 1024:
        raise ValueError('model response exceeds parser byte limit')
    text = text.strip()
    reasoning = None
    if text.startswith('<|channel>thought\n'):
        end = text.find('<channel|>')
        if end < 0:
            raise ValueError('unfinished Gemma reasoning')
        reasoning = text[len('<|channel>thought\n'):end]
        if '<|channel>' in reasoning:
            raise ValueError('ambiguous reasoning frame')
        text = text[end + len('<channel|>'):].strip()
    if '<|channel>' in text or '<channel|>' in text:
        raise ValueError('unexpected reasoning frame')
    if OPEN in text and not text.startswith(OPEN):
        raise ValueError('tool plans cannot mix prose or quoted examples with calls')
    calls, content = [], []
    while OPEN in text:
        if not text.startswith(OPEN):
            raise ValueError('tool plan contains content between calls')
        rest = text[len(OPEN):]
        name = re.match(r'[A-Za-z_][A-Za-z0-9_.-]*', rest)
        if not name:
            raise ValueError('invalid tool name')
        parser = Arguments(rest[len(name.group()):])
        arguments = parser.value()
        if not isinstance(arguments, dict):
            raise ValueError('tool arguments must be an object')
        parser.take(CLOSE)
        calls.append({'name': name.group(), 'arguments': arguments})
        text = parser.text[parser.pos:].strip()
        if text and not text.startswith(OPEN):
            raise ValueError('tool plan has trailing content')
    content.append(text)
    content = ''.join(content).strip()
    if any(marker in content for marker in ('<|tool', '<tool', '</tool')):
        raise ValueError('incomplete or unsupported tool framing')
    if not calls and not content:
        raise ValueError('empty completed model response')
    return {'content': content, 'reasoning': reasoning, 'calls': calls}
