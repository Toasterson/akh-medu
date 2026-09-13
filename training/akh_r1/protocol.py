"""Versioned interfaces: raw JSON baseline and the model's native tool template.

Parsing only unwraps the declared wire protocol. It never supplies tool names,
missing arguments or domain facts, or converts invalid arguments to valid ones.
"""
import json
import re


def function(name, description, properties, required):
    return dict(type='function', function=dict(name=name, description=description,
                parameters=dict(type='object', properties=properties, required=required,
                                additionalProperties=False)))


STRING = dict(type='string')
TOOLS = [
    function('terminal', 'Run commands as a Unix pipeline in the disposable lab. Available commands: miru, veln, awk, cut, sort, uniq, head, tail, wc, grep, xargs. Use --help and --version for discovery. Each stage is argv, with no shell expansion. All exit codes are returned.',
             dict(stages=dict(type='array', minItems=1, maxItems=6,
                              items=dict(type='array', minItems=1, items=STRING))), ['stages']),
    function('memory_write', 'Persist learned facts and command contracts in the real KG. Summary is short; store exact syntax and versions in fact objects. Include source evidence.',
             dict(summary=STRING, facts=dict(type='array', items=dict(type='object',
                  properties=dict(subject=STRING, predicate=STRING, object=STRING),
                  required=['subject', 'predicate', 'object'])),
                  source=STRING, tags=dict(type='array', items=STRING)), ['summary']),
    function('memory_recall', 'Find stored episodes and their subject names. Follow with kg_query to read full fact objects.',
             dict(query_symbols=STRING, top_k=STRING), ['query_symbols']),
    function('kg_query', 'Read persistent graph facts about a named subject.',
             dict(symbol=STRING, direction=dict(type='string', enum=['from', 'to', 'both'])), ['symbol']),
    function('kg_delete', 'Retract one obsolete graph fact using its exact subject, predicate and object.',
             dict(subject=STRING, predicate=STRING, object=STRING), ['subject', 'predicate', 'object']),
    function('report', 'Give the operator a concise progress update; continue the task afterward.', dict(message=STRING), ['message']),
    function('finish', 'End the episode with observed outcome and limitations. Verify actions before claiming success.', dict(message=STRING), ['message']),
]


def parse_action(raw, protocol):
    if protocol == 'native':
        final = raw.split('</think>')[-1]
        calls = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', final, flags=re.S)
        if len(calls) != 1:
            raise ValueError('Expected exactly one native tool_call block')
        parsed = json.loads(calls[0])
        if not isinstance(parsed, dict) or set(parsed) != {'name', 'arguments'}:
            raise ValueError('Expected name and arguments keys')
        parsed = dict(tool=parsed['name'], arguments=parsed['arguments'])
    else:
        parsed = json.loads(raw)
    if not isinstance(parsed, dict) or set(parsed) != {'tool', 'arguments'}:
        raise ValueError('Expected exactly tool and arguments keys')
    if not isinstance(parsed['arguments'], dict):
        raise ValueError('arguments must be an object')
    return parsed
